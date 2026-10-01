"""M0 checks. Small algorithm updates are diagnostics, not trained policies."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from feedingrobot.sim.model import ROOT


def package_path():
    import feedingrobot
    path = Path(feedingrobot.__file__).resolve()
    assert path == ROOT / "src/feedingrobot/__init__.py", path
    return {"path": str(path), "python": sys.executable}


def cuda_check():
    import torch
    assert torch.cuda.is_available(), "CUDA is unavailable in this process"
    x = torch.randn(32, 32, device="cuda", requires_grad=True)
    loss = (x @ x.T).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(x.grad).all()
    return dict(device=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                memory_bytes=torch.cuda.get_device_properties(0).total_memory, torch_cuda=torch.version.cuda,
                loss=float(loss.detach()))


def smoke_env():
    import gymnasium as gym

    class SmokeEnv(gym.Env):
        """A toy bounded state system, deliberately unrelated to feeding rewards."""
        def __init__(self):
            self.observation_space = gym.spaces.Box(-10., 10., (12,), dtype=np.float32)
            self.action_space = gym.spaces.Box(-1., 1., (6,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.state = self.np_random.uniform(-.1, .1, 12).astype(np.float32)
            self.steps = 0
            return self.state.copy(), {}

        def step(self, action):
            self.steps += 1
            self.state[:6] = .95 * self.state[:6] + .05 * action
            self.state[6:] = action
            return self.state.copy(), -float(np.square(self.state).sum()), False, self.steps >= 32, {}

    return SmokeEnv()


def env_check():
    from gymnasium.utils.env_checker import check_env
    from stable_baselines3.common.env_checker import check_env as sb3_check
    env = smoke_env()
    check_env(env, skip_render_check=True)
    sb3_check(env)
    env.close()
    return {"gymnasium": "passed", "sb3": "passed"}


def sac_check():
    import torch
    from stable_baselines3 import SAC
    env = smoke_env()
    model = SAC("MlpPolicy", env, buffer_size=128, learning_starts=16, batch_size=16,
                train_freq=1, gradient_steps=1, policy_kwargs={"net_arch": [32, 32]}, seed=0, device="cpu")
    before = [p.detach().clone() for p in model.actor.parameters()]
    model.learn(total_timesteps=40)
    changed = any(not torch.equal(p, old) for p, old in zip(model.actor.parameters(), before))
    assert changed and model._n_updates > 0
    assert all(torch.isfinite(p).all() for p in model.policy.parameters())
    env.close()
    return {"updates": model._n_updates, "actor_changed": changed, "training_claim": False}


def diffusion_check():
    import torch
    from diffusers import DDPMScheduler
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionConditionalUnet1d
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = DiffusionConfig(input_features={"observation.state": PolicyFeature(FeatureType.STATE, (14,))},
                             output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
                             horizon=16, n_obs_steps=2, n_action_steps=8, down_dims=(32, 64),
                             diffusion_step_embed_dim=32, n_groups=8)
    network = DiffusionConditionalUnet1d(config, global_cond_dim=28).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    clean = torch.randn(2, 16, 6, device=device)
    noise = torch.randn_like(clean)
    times = torch.tensor([3, 7], device=device)
    condition = torch.randn(2, 28, device=device)
    scheduler = DDPMScheduler(num_train_timesteps=100)
    noisy = scheduler.add_noise(clean, noise, times)
    before = next(network.parameters()).detach().clone()
    prediction = network(noisy, times, global_cond=condition)
    assert prediction.shape == (2, 16, 6)
    loss = (prediction - noise).square().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None)
    optimizer.step()
    assert not torch.equal(before, next(network.parameters()))
    return {"device": device, "loss": float(loss.detach()), "output_shape": list(prediction.shape),
            "training_claim": False}


def ik_check():
    from feedingrobot.sim.task import FeedingTask
    results = {}
    for robot in ["panda", "ur5e"]:
        task = FeedingTask(robot)
        task.reset(preset="empty")
        start = task.data.site_xpos[task.index.tcp].copy()
        task.adapter.set_twist([.01, 0, 0, 0, 0, 0], task.data.time, task.data.time + .1)
        for _ in range(100):
            task.step_physics()
        assert not task.terminated and not task.adapter.fault, (task.failure_reason, task.adapter.fault)
        assert np.max(np.abs(task.adapter.last_ik_velocity[task.adapter.frozen_dofs])) < 1e-7
        results[robot] = {"dof": task.index.n, "displacement_m": float(np.linalg.norm(task.data.site_xpos[task.index.tcp] - start))}
    return results


def viewer_child():
    from feedingrobot.sim.viewer import passive_viewer
    from feedingrobot.sim.task import FeedingTask
    task = FeedingTask()
    task.reset(preset="empty")
    with passive_viewer(task.model, task.data) as viewer:
        for _ in range(30):
            task.step_physics()
            viewer.sync()
            time.sleep(.02)
        assert viewer.is_running(), "Viewer closed before verification"
    print("VIEWER_OK")


def viewer_check():
    result = subprocess.run([sys.executable, "-m", "feedingrobot.scripts.doctor", "--viewer-child"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and "VIEWER_OK" in result.stdout, (result.returncode, result.stdout, result.stderr)
    return {"opened_and_synced": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer-child", action="store_true")
    parser.add_argument("--skip-viewer", action="store_true")
    parser.add_argument("--output", default="outputs/m0/doctor.json")
    args = parser.parse_args()
    if args.viewer_child:
        viewer_child()
        return
    report = {"checks": {}, "versions": {}}
    for name in ["torch", "torchvision", "mujoco", "numpy", "gymnasium", "stable-baselines3", "lerobot", "mink", "qpsolvers", "daqp"]:
        try:
            report["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["versions"][name] = None
    checks = {"package_path": package_path, "cuda": cuda_check, "env_checkers": env_check,
              "diffusion": diffusion_check, "sac": sac_check, "physics_and_ik": ik_check}
    if not args.skip_viewer:
        checks["viewer"] = viewer_check
    else:
        report["checks"]["viewer"] = {"status": "not_verified"}
    for name, check in checks.items():
        start = time.monotonic()
        try:
            detail = check()
            report["checks"][name] = {"status": "passed", "detail": detail}
        except Exception:
            report["checks"][name] = {"status": "failed", "error": traceback.format_exc()}
        report["checks"][name]["wall_seconds"] = time.monotonic() - start
        print(name, report["checks"][name]["status"], flush=True)
    report["status"] = "passed" if all(c["status"] == "passed" for c in report["checks"].values()) else "incomplete"
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
