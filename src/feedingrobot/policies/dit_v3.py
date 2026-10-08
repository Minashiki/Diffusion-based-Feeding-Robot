"""MinaDiTV3: V2 architecture with raw physical inputs and FP32 inference."""

from feedingrobot.policies.dit_v2 import MinaDiTV2


class MinaDiTV3(MinaDiTV2):
    """Scaling and inference precision are enforced by the shared policy pipeline."""
