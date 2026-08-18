# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import logging
import os

import ray
from omegaconf import DictConfig

from verl.experimental.reward.reward_loop import get_reward_loop_manager_cls
from verl.protocol import DataProto
from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils import hf_tokenizer
from verl.utils.local_fs import copy_to_local

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@ray.remote
class RewardManagerWorker:
    def __init__(
        self,
        config: DictConfig,
        reward_router_address: str = None,
        reward_manager_name: str | None = None,
        load_reward_model_tokenizer: bool = True,
    ):
        self.config = config
        self.reward_router_address = reward_router_address
        self.reward_manager_name = reward_manager_name or self.config.reward_model.reward_manager
        self.load_reward_model_tokenizer = load_reward_model_tokenizer
        self._init_reward_fn()

    def _init_reward_fn(self):
        input_tokenizer_local_path = copy_to_local(self.config.actor_rollout_ref.model.path)
        self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path, trust_remote_code=True)
        self.reward_model_tokenizer = None
        if self.config.reward_model.enable and self.load_reward_model_tokenizer:
            reward_model_tokenizer_local_path = copy_to_local(self.config.reward_model.model.path)
            self.reward_model_tokenizer = hf_tokenizer(reward_model_tokenizer_local_path, trust_remote_code=True)
        # The 'code' reward loop (CodeRewardLoopManager) scores entirely via the shared
        # subprocess sandbox pool (build_row_jobs/aggregate_row_results -> final_score) and
        # documents compute_score as UNUSED. Loading the custom_reward_function here is thus
        # a no-op for scoring, and it is unsafe in this async Ray actor: rllm_code_reward's
        # signal-based SIGALRM timeout raises "signal only works in main thread of the main
        # interpreter" off the main thread. Skip it for 'code' (scoring is byte-identical);
        # other reward managers still load their custom fn.
        if self.reward_manager_name == "code":
            self.reward_fn = None
        else:
            self.reward_fn = get_custom_reward_fn(self.config)
        reward_loop_manager_cls = get_reward_loop_manager_cls(self.reward_manager_name)
        self.reward_loop = reward_loop_manager_cls(
            self.config, self.input_tokenizer, self.reward_fn, self.reward_router_address, self.reward_model_tokenizer
        )

    async def compute_score(self, data: DataProto) -> DataProto:
        return await self.reward_loop.run_single(data)
