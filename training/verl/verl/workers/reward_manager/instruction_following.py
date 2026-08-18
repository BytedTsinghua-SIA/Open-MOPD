from __future__ import annotations

from verl.workers.reward_manager import register
from verl.workers.reward_manager.dapo import DAPORewardManager


@register("instruction_following")
class InstructionFollowingRewardManager(DAPORewardManager):
    """Rule-based verifier reward manager for instruction-following RL."""
