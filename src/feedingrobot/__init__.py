"""State-driven feeding simulation and M3 Gymnasium task."""

from gymnasium.envs.registration import register

__version__ = "0.3.0"

register(id="FeedingRobot-v0", entry_point="feedingrobot.envs:FeedingGymEnv")
