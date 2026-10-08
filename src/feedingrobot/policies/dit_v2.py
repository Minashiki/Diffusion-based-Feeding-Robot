"""MinaDiTV2: six causal DiT blocks with channel-padded action inputs."""

import torch
from torch import nn

from feedingrobot.policies.dit import ActionDiT


class ZeroPadding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, actions):
        return nn.functional.pad(actions, (0, self.width - 6))


class CausalAttention(nn.MultiheadAttention):
    def forward(self, query, key, value, **kwargs):
        count = query.shape[1]
        mask = torch.ones((count, count), dtype=torch.bool, device=query.device).triu(1)
        return super().forward(query, key, value, attn_mask=mask, **kwargs)


class MinaDiTV2(ActionDiT):
    def __init__(self, config):
        m = config['model']
        if m['depth'] != 6 or m['hidden_size'] < 6:
            raise ValueError('MinaDiTV2 requires six blocks and at least six hidden channels')
        super().__init__(config)
        self.action_projection = ZeroPadding(self.width)
        for block in self.blocks:
            block.self_attention = CausalAttention(self.width, m['heads'], batch_first=True)
