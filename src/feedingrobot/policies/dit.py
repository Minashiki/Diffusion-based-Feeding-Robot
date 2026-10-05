"""Action DiT with adaLN-Zero and cross-attention to causal state history.

Architecture reference: Peebles & Xie, ICCV 2023, arXiv:2212.09748.
This implementation operates on TCP twists, not image patches or image latents.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from diffusers import DDIMScheduler, DDPMScheduler


def positions(count, width, device):
    frequency = torch.exp(-math.log(10000) * torch.arange(width // 2, device=device) / (width // 2))
    angles = torch.arange(count, device=device)[:, None] * frequency
    return torch.cat([angles.sin(), angles.cos()], -1)


@dataclass
class Condition:
    global_features: torch.Tensor
    tokens: torch.Tensor
    valid: torch.Tensor


class StateConditionEncoder(nn.Module):
    def __init__(self, state_dim=122, history_dim=28, width=512):
        super().__init__()
        self.state = nn.Sequential(nn.Linear(state_dim, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
        self.conv1, self.conv2 = nn.Conv1d(history_dim, 64, 3), nn.Conv1d(64, 128, 3)
        self.stage, self.interaction = nn.Embedding(8, 32), nn.Linear(4, 32)
        self.fuse = nn.Sequential(nn.Linear(416, width), nn.SiLU(), nn.Linear(width, width))
        self.state_token, self.history_token, self.phase_token = nn.Linear(128, width), nn.Linear(128, width), nn.Linear(32, width)
        self.type_embedding = nn.Embedding(3, width)

    def forward(self, batch):
        state = self.state(batch['states'])
        history = batch['history'].transpose(1, 2)
        history = nn.functional.silu(self.conv1(nn.functional.pad(history, (2, 0))))
        history = nn.functional.silu(self.conv2(nn.functional.pad(history, (2, 0)))).transpose(1, 2)
        stage = self.stage(batch['phase']) + self.interaction(batch['interaction'])
        state = state * batch['state_mask'][..., None]
        history = history * batch['history_mask'][..., None]
        global_features = self.fuse(torch.cat([state.flatten(1), history[:, -1], stage], -1))
        state_tokens = self.state_token(state) + positions(2, global_features.shape[-1], state.device)
        history_tokens = self.history_token(history) + positions(10, global_features.shape[-1], state.device)
        state_tokens = state_tokens + self.type_embedding.weight[0]
        history_tokens = history_tokens + self.type_embedding.weight[1]
        phase_token = self.phase_token(stage)[:, None] + self.type_embedding.weight[2]
        tokens = torch.cat([state_tokens, history_tokens, phase_token], 1)
        valid = torch.cat([batch['state_mask'], batch['history_mask'],
                           torch.ones((len(state), 1), dtype=torch.bool, device=state.device)], 1)
        return Condition(global_features, tokens, valid)


class DiTBlock(nn.Module):
    def __init__(self, width, heads, mlp_hidden):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(width, elementwise_affine=False, eps=1e-6) for _ in range(3)])
        self.self_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.memory_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, mlp_hidden), nn.GELU(approximate='tanh'), nn.Linear(mlp_hidden, width))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 9 * width))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x, condition, time_condition, mask):
        modulation = self.modulation(time_condition).chunk(9, -1)
        for i in range(3):
            shift, scale, gate = modulation[i*3:i*3+3]
            query = self.norms[i](x) * (1 + scale[:, None]) + shift[:, None]
            if i == 0:
                value = self.self_attention(query, query, query, key_padding_mask=~mask, need_weights=False)[0]
            elif i == 1:
                memory = self.memory_norm(condition.tokens)
                value = self.cross_attention(query, memory, memory, key_padding_mask=~condition.valid, need_weights=False)[0]
            else:
                value = self.mlp(query)
            x = x + gate[:, None] * value
        return x


class ActionDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        m = config['model']
        width = m['hidden_size']
        if width % m['heads'] or width % 2 or m['horizon'] < 1:
            raise ValueError('Invalid DiT dimensions')
        self.horizon, self.width = m['horizon'], width
        from feedingrobot.envs.feeding import observation_schema
        from feedingrobot.sim.model import load_json
        n = len(load_json(f'configs/robots/{config["robot_id"]}.json')['joints'])
        state_dim = sum(size for _, size, _ in observation_schema(config['robot_id'], n)['fields']) + 14
        self.encoder = StateConditionEncoder(state_dim=state_dim, history_dim=14+2*n, width=width)
        self.action_projection = nn.Linear(6, width)
        self.time_mlp = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([DiTBlock(width, m['heads'], m['mlp_hidden']) for _ in range(m['depth'])])
        self.final_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, width*2))
        self.output = nn.Linear(width, 6)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def denoiser(self, noisy_actions, diffusion_step, condition, action_mask=None):
        if noisy_actions.shape[1:] != (self.horizon, 6):
            raise ValueError('Expected [B,H,6] actions')
        mask = (torch.ones(noisy_actions.shape[:2], dtype=torch.bool, device=noisy_actions.device)
                if action_mask is None else action_mask)
        if not mask.any(1).all():
            raise ValueError('Each action window needs at least one legal action')
        frequency = torch.exp(-math.log(10000) * torch.arange(128, device=noisy_actions.device) / 128)
        angles = diffusion_step.float()[:, None] * frequency
        time_condition = self.time_mlp(torch.cat([angles.cos(), angles.sin()], -1)) + condition.global_features
        x = self.action_projection(noisy_actions * mask[..., None]) + positions(self.horizon, self.width, noisy_actions.device)
        for block in self.blocks:
            x = block(x, condition, time_condition, mask)
        shift, scale = self.final_modulation(time_condition).chunk(2, -1)
        result = self.output(self.final_norm(x) * (1 + scale[:, None]) + shift[:, None])
        return result * mask[..., None]


def schedulers(config):
    settings = dict(num_train_timesteps=config['diffusion']['train_steps'],
                    beta_schedule=config['diffusion']['beta_schedule'], prediction_type='epsilon',
                    clip_sample=False, timestep_spacing='trailing')
    return DDPMScheduler(**settings), DDIMScheduler(**settings)


def noise_loss(model, scheduler, batch, *, noise=None, timesteps=None):
    actions, mask = batch['actions'], batch['action_mask']
    noise = torch.randn_like(actions) if noise is None else noise
    timesteps = (torch.randint(scheduler.config.num_train_timesteps, (len(actions),), device=actions.device)
                 if timesteps is None else timesteps)
    noisy = scheduler.add_noise(actions, noise, timesteps) * mask[..., None]
    predicted = model.denoiser(noisy, timesteps, model.encoder(batch), mask)
    return ((predicted.float() - noise.float()).square() * mask[..., None]).sum() / (mask.sum() * 6)


@torch.no_grad()
def sample_actions(model, config, batch, normalization, generator=None):
    _, scheduler = schedulers(config)
    scheduler.set_timesteps(config['diffusion']['inference_steps'], device=batch['states'].device)
    condition = model.encoder(batch)
    actions = torch.randn((len(batch['states']), model.horizon, 6), device=batch['states'].device, generator=generator)
    for step in scheduler.timesteps:
        steps = step.expand(len(actions))
        noise = model.denoiser(actions, steps, condition)
        actions = scheduler.step(noise, step, actions, eta=0).prev_sample
    mean = torch.as_tensor(normalization['action_mean'], device=actions.device)
    scale = torch.as_tensor(normalization['action_std'], device=actions.device)
    return actions * scale + mean
