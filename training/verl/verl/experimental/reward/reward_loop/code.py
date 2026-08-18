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

"""Per-sample (inline) code reward loop.

This is the async-rollout counterpart of the batched, driver-side code reward in
``verl.trainer.ppo.code_reward_pool``. When ``rollout.mode=async``, each
trajectory is scored the moment it finishes generating (see
``agent_loop.py``'s ``reward_manager_worker.compute_score``). For code RL we
must NOT run the model's untrusted code inside the reward actor's own thread
pool; instead we reuse the exact same shared subprocess sandbox pool the batched
path uses, so:

* untrusted code keeps its subprocess isolation + hard timeout, and
* scores are byte-identical to the batched path (both call ``build_row_jobs`` /
  ``aggregate_row_results``).

The pool's worker actors are named + detached, so the driver and every reward
actor converge on one shared set of workers (bounded cluster CPU budget).
"""

import asyncio

from verl import DataProto
from verl.experimental.reward.reward_loop import register
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase
from verl.trainer.ppo.code_reward_pool import (
    aggregate_row_results,
    build_row_jobs,
    parse_code_reward_config,
    submit_row_jobs,
)


async def _gather_futures(futures: list) -> list:
    """Await the pool's Ray futures on the event loop (non-blocking).

    Ray ``ObjectRef`` is awaitable inside an async actor, so this overlaps
    testcase execution with other in-flight trajectories instead of blocking the
    reward actor. Factored out as a module function so tests can stub it without
    a live Ray cluster.
    """
    return await asyncio.gather(*futures)


@register("code")
class CodeRewardLoopManager(RewardLoopManagerBase):
    """Inline code reward routed through the shared subprocess sandbox pool."""

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer)
        # compute_score / router / rm tokenizer are accepted for signature
        # parity with the other reward loops but are unused: the code reward is
        # computed entirely from testcases via the subprocess pool.
        self.cfg = parse_code_reward_config(config)

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        # Job construction (tokenizer decode + testcase parsing) is CPU work;
        # keep it off the event loop.
        info, jobs, row_job_specs = await self.loop.run_in_executor(
            None, lambda: build_row_jobs(data_item, 0, self.tokenizer, self.cfg)
        )

        if jobs:
            futures = await self.loop.run_in_executor(
                None, lambda: submit_row_jobs(jobs, info.route, self.cfg)
            )
            results = await _gather_futures(futures)
        else:
            results = []

        results_by_key = {(jr.owner_id, jr.job_idx): jr for jr in results}
        row = aggregate_row_results(info, row_job_specs, results_by_key)
        return {"reward_score": row["final_score"], "reward_extra_info": row}
