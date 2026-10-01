import copy

import gymnasium as gym
import numpy as np
import pytest

from feedingrobot.envs import FeedingGymEnv


@pytest.fixture(params=["panda", "ur5e"])
def env(request):
    env = FeedingGymEnv(request.param)
    yield env
    env.close()


def assert_transition_equal(a, b):
    np.testing.assert_allclose(a[0], b[0], rtol=0, atol=1e-7)
    np.testing.assert_allclose(a[1], b[1], rtol=0, atol=1e-12)
    assert a[2:4] == b[2:4]
    for key in ("time", "elapsed_s", "phase", "success", "failure_reason", "events", "reward_terms"):
        assert a[4][key] == b[4][key]


def test_checkers_registration_and_schema(env):
    from gymnasium.utils.env_checker import check_env
    from stable_baselines3.common.env_checker import check_env as sb3_check
    check_env(env, skip_render_check=True)
    sb3_check(env)
    registered = gym.make("FeedingRobot-v0", robot_id=env.task.robot_id)
    obs, _ = registered.reset(seed=10)
    assert registered.observation_space.contains(obs)
    assert obs.shape == (82 + 2 * env.task.index.n,)
    registered.close()
    policy = env.task.provider.observe()["policy_obs"]
    assert not {"future_events", "seed", "contacts", "timers"}.intersection(policy)


def test_timing_action_and_seed_replay(env):
    def rollout():
        env.reset(seed=17)
        return [env.step(np.array([.1, 0, 0, 0, 0, .05])) for _ in range(4)]
    a, b = rollout(), rollout()
    for x, y in zip(a, b):
        assert_transition_equal(x, y)
        assert x[4]["elapsed_s"] == pytest.approx(.02)
    assert env.task.tick == 80
    np.testing.assert_allclose(env.task.adapter.command[0], [.005, 0, 0, 0, 0, .025])


def test_complete_snapshot_replay_and_independence(env):
    env.reset(seed=21, options={"preset": "food_on_spoon"})
    for _ in range(7):
        env.step(np.array([.05, 0, 0, 0, 0, 0]))
    state = env.get_state()
    a = [env.step(np.array([0, .05, 0, 0, 0, 0])) for _ in range(5)]
    sample = env.action_space.sample()
    env.set_state(state)
    b = [env.step(np.array([0, .05, 0, 0, 0, 0])) for _ in range(5)]
    for x, y in zip(a, b):
        assert_transition_equal(x, y)
    np.testing.assert_array_equal(sample, env.action_space.sample())
    bad = copy.deepcopy(state)
    bad["task"]["signature"]["robot_id"] = "wrong"
    with pytest.raises(ValueError, match="Incompatible"):
        env.set_state(bad)
    env.set_state(state)
    state["task"]["adapter"]["velocity"][:] = 99
    assert np.max(env.task.adapter.velocity) < 1


def test_snapshot_rejects_different_model_and_time_limit(env):
    env.reset(seed=0)
    state = env.get_state()
    env.task.model.opt.iterations += 1
    with pytest.raises(ValueError, match="Incompatible"):
        env.set_state(state)
    other = FeedingGymEnv(env.task.robot_id, max_episode_s=1)
    with pytest.raises(ValueError, match="Incompatible"):
        other.set_state(state)
    other.close()


def test_time_limit_vecenv_and_terminal_observation(env):
    from stable_baselines3.common.vec_env import DummyVecEnv
    short = FeedingGymEnv(env.task.robot_id, max_episode_s=.025)
    vec = DummyVecEnv([lambda: short])
    vec.seed(3)
    vec.reset()
    assert not vec.step(np.zeros((1, 6)))[2][0]
    obs, _, done, infos = vec.step(np.zeros((1, 6)))
    assert done[0] and infos[0]["TimeLimit.truncated"]
    assert infos[0]["time"] == pytest.approx(.025)
    assert infos[0]["elapsed_s"] == pytest.approx(.005)
    assert short.observation_space.contains(infos[0]["terminal_observation"])
    assert short.task.tick == 0  # VecEnv auto-reset has already happened.
    vec.close()


def test_terminated_vecenv_is_not_timeout(env):
    from stable_baselines3.common.vec_env import DummyVecEnv
    vec = DummyVecEnv([lambda: env])
    vec.reset()
    env.task.set_external_wrench([0, 0, 12], [0, 0, 0], env.task.data.site_xpos[env.task.index.tcp])
    _, _, done, infos = vec.step(np.zeros((1, 6)))
    assert done[0] and not infos[0]["TimeLimit.truncated"]
    assert infos[0]["failure_reason"] == "contact_limit"
    assert "terminal_observation" in infos[0]
    vec.close()


def test_substep_force_termination_reward_and_reset(env):
    env.reset(seed=0)
    env.task.set_external_wrench([0, 0, 12], [0, 0, 0], env.task.data.site_xpos[env.task.index.tcp])
    _, reward, terminated, truncated, info = env.step(np.zeros(6))
    assert terminated and not truncated
    assert info["elapsed_s"] == pytest.approx(.001)
    assert info["failure_reason"] == "contact_limit" and info["reward_terms"]["failure"] == -50
    assert reward == sum(info["reward_terms"].values())
    time = env.task.data.time
    with pytest.raises(RuntimeError):
        env.step(np.zeros(6))
    assert env.task.data.time == time
    env.reset(seed=0)
    assert not env.task.logic.events and not env.task.logic.awarded


def test_nonfinite_state_returns_last_valid_observation(env):
    obs, _ = env.reset(seed=0)
    env.task.data.qvel[env.task.index.dofs[0]] = np.nan
    got, _, terminated, _, info = env.step(np.zeros(6))
    assert terminated and not info["observation_valid"]
    assert info["failure_reason"] == "nonfinite_state"
    np.testing.assert_array_equal(obs, got)


@pytest.mark.parametrize("action", [[np.nan] * 6, [np.inf] * 6, [2.] * 6, [0.] * 5])
def test_invalid_action_stops_and_requires_reset(action):
    env = FeedingGymEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(action)
    assert env.done and env.task.adapter.command is None
    with pytest.raises(RuntimeError):
        env.step(np.zeros(6))
    env.close()


def test_pickup_reward_once_and_no_phase_progress_jump(env):
    env.reset(seed=0, options={"preset": "food_on_spoon"})
    awarded = []
    for _ in range(15):
        _, _, terminated, _, info = env.step(np.zeros(6))
        assert not terminated, info
        if info["reward_terms"]["pickup"]:
            awarded.append(info["reward_terms"]["pickup"])
            assert info["reward_terms"]["progress"] == 0
    assert awarded == [10.]


def test_wait_ready_has_no_approach_reward(env):
    env.reset(seed=0, options={"preset": "food_on_spoon"})
    env.task.logic.phase = "WAIT_READY"  # Synthetic reward isolation, not a physical success claim.
    _, _, _, _, info = env.step(np.array([.1, 0, 0, 0, 0, 0]))
    assert info["reward_terms"]["progress"] == 0


def test_substep_contact_peak_and_impulse_not_double_counted(env, monkeypatch):
    from feedingrobot.sim import task as task_module
    env.reset(seed=0)
    actual = task_module.read_contacts
    calls = 0
    def one_integrator_peak(model, data, index):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [dict(group1="spoon", group2="mouth", force_n=6., distance=0.)]
        return actual(model, data, index)
    monkeypatch.setattr(task_module, "read_contacts", one_integrator_peak)
    _, _, terminated, _, info = env.step(np.zeros(6))
    assert terminated and info["failure_reason"] == "contact_limit"
    assert info["step_metrics"]["contact_peak_n"] == 6
    assert info["step_metrics"]["contact_impulse_ns"] == pytest.approx(.006)
    assert info["step_metrics"]["contact_over_limit_s"] == pytest.approx(.001)
    assert info["reward_terms"]["contact"] == pytest.approx(-.0006)


def test_cross_instance_snapshot_and_rng(env):
    env.reset(seed=31)
    env.step(np.array([.05, 0, 0, 0, 0, 0]))
    state = env.get_state()
    other = FeedingGymEnv(env.task.robot_id)
    other.reset(seed=32)
    other.set_state(state)
    assert_transition_equal(env.step(np.zeros(6)), other.step(np.zeros(6)))
    np.testing.assert_array_equal(env.reset()[0], other.reset()[0])
    other.close()


def test_task_nonfinite_event_without_gym(env):
    env.reset(seed=0)
    env.task.data.qvel[env.task.index.dofs[0]] = np.nan
    env.task.step_physics()
    assert env.task.logic.failure_reason == "nonfinite_state"
    assert env.task.logic.events[-1]["reason"] == "nonfinite_state"


def test_invalid_timestep_and_time_limit():
    for kwargs in ({"timestep": .003}, {"max_episode_s": 0}, {"max_episode_s": np.nan}):
        with pytest.raises(ValueError):
            FeedingGymEnv(**kwargs)


def test_nonfinite_observation_is_terminal(env, monkeypatch):
    obs, _ = env.reset(seed=0)
    monkeypatch.setattr(env, "_observation", lambda: np.full_like(obs, np.nan))
    got, _, terminated, truncated, info = env.step(np.zeros(6))
    assert terminated and not truncated and not info["observation_valid"]
    assert info["failure_reason"] == "nonfinite_state" and info["reward_terms"]["failure"] == -50
    np.testing.assert_array_equal(got, obs)
