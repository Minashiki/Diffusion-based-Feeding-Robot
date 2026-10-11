"""MinaDiTV5 keeps V4's architecture with fitted DP input normalization."""

from feedingrobot.policies.dit_v4 import MinaDiTV4


class MinaDiTV5(MinaDiTV4):
    """Normalization and trailing/100 sampling belong to the shared pipeline."""
