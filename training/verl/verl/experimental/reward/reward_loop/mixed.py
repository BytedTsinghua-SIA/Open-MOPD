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

"""Inline mixed-domain reward loop for Math, Code, and IF RL."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from verl import DataProto
from verl.experimental.reward.reward_loop import register
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase
from verl.experimental.reward.reward_loop.code import CodeRewardLoopManager
from verl.experimental.reward.reward_loop.naive import NaiveRewardLoopManager
from verl.utils.reward_score.mix_rl_dispatch import infer_domain
from verl.workers.reward_manager.naive import _ALC_DEFAULT_BREAKPOINTS, _answer_length_cap, _cfg_get


def _score_math_in_process(
    data_source: str,
    solution_str: str,
    ground_truth: object,
    extra_info: dict,
) -> dict:
    """Run the exact Math-RL scorer in a process whose main thread owns SIGALRM."""

    from verl.utils.reward_score import ttrl_math

    return ttrl_math.reward_func(data_source, solution_str, ground_truth, extra_info)


@register("mixed")
class MixedRewardLoopManager(RewardLoopManagerBase):
    """Route each completed trajectory to its single-domain reward path.

    Code delegates to ``CodeRewardLoopManager`` and therefore submits testcase
    jobs immediately after that trajectory's rollout finishes.  Math and IF use
    the normal inline scorer.  A compact, uniform extra-info schema is returned
    because the async agent-loop postprocessor stacks every key across domains.
    """

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer)
        self.code_loop = CodeRewardLoopManager(config, tokenizer)
        self.other_loop = NaiveRewardLoopManager(
            config,
            tokenizer,
            compute_score,
            reward_router_address,
            reward_model_tokenizer,
        )
        reward_kwargs = config.reward_model.get("reward_kwargs", {}) or {}
        math_workers = max(1, int(_cfg_get(reward_kwargs, "math_score_workers_per_reward_actor", 4)))
        self.math_score_pool = ProcessPoolExecutor(
            max_workers=math_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        self.max_resp_len = int(_cfg_get(reward_kwargs, "max_resp_len", config.data.max_response_length))
        self.length_cap_cfg = _cfg_get(reward_kwargs, "length_cap_cfg", None)

    async def _run_math(self, data: DataProto) -> dict:
        """Score Math outside the async actor process, preserving SIGALRM timeouts.

        ``NaiveRewardLoopManager`` deliberately moves synchronous scoring to a
        thread.  The selected Math-RL scorer uses SIGALRM around SymPy and Python
        only permits signal handlers in a process's main thread.  A persistent
        spawn process pool keeps the exact scorer and timeout behavior without
        blocking rollout or leaking one process per trajectory.
        """

        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}) or {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())

        result = await self.loop.run_in_executor(
            self.math_score_pool,
            _score_math_in_process,
            data_source,
            response_str,
            ground_truth,
            extra_info,
        )
        if isinstance(result, dict):
            return {
                "reward_score": float(result["score"]),
                "reward_extra_info": dict(result),
            }
        score = float(result)
        return {"reward_score": score, "reward_extra_info": {"acc": score}}

    def _apply_math_length_cap(self, data: DataProto, result: dict) -> dict:
        if not _cfg_get(self.length_cap_cfg, "enable", False):
            return result
        extra = result.get("reward_extra_info", {}) or {}
        if "answer_score" not in extra:
            return result

        response_ids = data[0].batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = int(data[0].batch["attention_mask"][-response_length:].sum().item())
        breakpoints = _cfg_get(self.length_cap_cfg, "breakpoints", _ALC_DEFAULT_BREAKPOINTS)
        length_cap = _answer_length_cap(valid_response_length, self.max_resp_len, breakpoints)
        think_score = float(extra.get("think_score", 0.0))
        answer_score = float(extra.get("answer_score", 0.0))
        reward = think_score + min(answer_score, length_cap)
        result["reward_score"] = reward
        extra["score"] = reward
        extra["final_score"] = reward
        extra["length_cap"] = length_cap
        extra["length_capped"] = float(answer_score > length_cap)
        return result

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        item = data[0]
        extra_info = item.non_tensor_batch.get("extra_info", {}) or {}
        domain = infer_domain(
            item.non_tensor_batch.get("data_source", ""),
            extra_info,
            item.non_tensor_batch.get("ability"),
        )
        if domain == "code":
            result = await self.code_loop.run_single(data)
        elif domain == "math":
            result = await self._run_math(data)
            result = self._apply_math_length_cap(data, result)
        else:
            result = await self.other_loop.run_single(data)

        reward = float(result["reward_score"])
        original_extra = result.get("reward_extra_info", {}) or {}
        acc = float(original_extra.get("acc", reward))
        return {
            "reward_score": reward,
            "reward_extra_info": {
                "score": reward,
                "acc": acc,
                "reward_domain": domain,
            },
        }
