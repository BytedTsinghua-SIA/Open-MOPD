# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Batch-side guard for the inline mixed-domain reward manager."""

from typing import Any

import torch

from verl import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn
from verl.workers.reward_manager.registry import register


@register("mixed")
class MixedRewardManager(AbstractRewardManager):
    """Expose ``mixed`` to the trainer while keeping scoring trajectory-inline.

    Async rollout scores every completed trajectory through
    ``MixedRewardLoopManager``.  The trainer still constructs a batch reward
    manager during startup, even though the inline ``rm_scores`` path bypasses
    it.  Registering this guard keeps that startup contract explicit and avoids
    silently re-scoring a mixed batch after rollout.
    """

    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score: RawRewardFn | None,
        reward_fn_key: str = "data_source",
        **kwargs: Any,
    ) -> None:
        del tokenizer, num_examine, compute_score, reward_fn_key, kwargs

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        if "rm_scores" not in data.batch:
            raise RuntimeError(
                "mixed reward requires async rollout to materialize rm_scores; "
                "batch-side mixed scoring would lose rollout/scoring overlap"
            )
        reward_tensor = data.batch["rm_scores"]
        if return_dict:
            reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
            reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
