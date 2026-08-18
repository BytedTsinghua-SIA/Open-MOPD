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

"""Validation-only inline scorer for single-domain and multi-domain OPD.

Code rows use the same shared subprocess pool as Code-RL. Math and instruction-
following rows use the configured OPD validation dispatcher. Every domain returns
the same ``score``/``acc`` schema so mixed MT-OPD batches can be postprocessed
safely.
"""

import asyncio
import inspect
import multiprocessing
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor

from verl import DataProto
from verl.experimental.reward.reward_loop import register
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase
from verl.trainer.ppo.code_reward_pool import (
    aggregate_row_results,
    build_row_jobs,
    parse_code_reward_config,
    submit_row_jobs,
)
from verl.utils.reward_score import default_compute_score


def _is_code_data_source(data_source: object) -> bool:
    value = str(data_source or "").lower()
    return value.startswith(("livecodebench", "deepcoder")) or value in {
        "apps",
        "codecontests",
        "codecontests_lcb",
        "codeforces",
        "primeintellect",
        "taco",
    }


def _is_if_data_source(data_source: object) -> bool:
    value = str(data_source or "").lower()
    return "ifeval" in value or value.startswith("ifbench") or value in {
        "nemotron_if",
        "instruction_following",
    }


def _score_math_in_process(
    data_source: str,
    solution_str: str,
    ground_truth: object,
    extra_info: dict,
) -> object:
    """Run the exact Math-RL scorer in a process whose main thread owns SIGALRM."""

    from verl.utils.reward_score import ttrl_math

    return ttrl_math.reward_func(data_source, solution_str, ground_truth, extra_info)


def _normalize_result(result: object) -> dict[str, float]:
    if isinstance(result, dict):
        score = result.get("score", result.get("acc", 0.0))
        acc = result.get("acc", score)
    else:
        score = result
        acc = result
    return {"score": float(score), "acc": float(acc)}


_FORMAL_METADATA_KEYS = (
    "dataset",
    "domain",
    "sample_id",
    "original_prompt",
    "answer",
    "metadata",
    "evaluator",
    "repeat_idx",
    "request_id",
)


def _validation_score_inputs(data_item) -> tuple[object | None, dict]:
    """Resolve scorer inputs for both RL-shaped and formal-eval rows.

    RL datasets keep answers under ``reward_model.ground_truth`` and usually
    place evaluator metadata in ``extra_info``.  The aligned formal Code/IF
    parquets instead expose ``answer``, ``metadata``, ``evaluator`` and the
    original prompt as top-level columns; IF rows intentionally have no
    ``reward_model`` column.  Preserve either representation without inventing
    placeholder RM metadata.
    """
    non_tensor = data_item.non_tensor_batch
    reward_model = non_tensor.get("reward_model")
    ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, Mapping) else None
    if ground_truth is None:
        ground_truth = non_tensor.get("answer")

    raw_extra_info = non_tensor.get("extra_info")
    extra_info = dict(raw_extra_info) if isinstance(raw_extra_info, Mapping) else {}
    for key in _FORMAL_METADATA_KEYS:
        value = non_tensor.get(key)
        if value is not None:
            extra_info.setdefault(key, value)

    # The official IF evaluator consumes the rewritten prompt, not the full
    # chat transcript that includes earlier assistant turns.
    original_prompt = non_tensor.get("original_prompt")
    if original_prompt is not None:
        extra_info.setdefault("eval_prompt_for_scorer", original_prompt)
    return ground_truth, extra_info


async def _gather_futures(futures: list) -> list:
    return await asyncio.gather(*futures)


@register("opd_validation")
class OPDValidationRewardLoopManager(RewardLoopManagerBase):
    """Score only validation trajectories without touching OPD train rewards."""

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.code_cfg = parse_code_reward_config(config)
        reward_kwargs = config.custom_reward_function.get("reward_kwargs", {}) or {}
        math_workers = max(1, int(reward_kwargs.get("math_score_workers_per_reward_actor", 4)))
        # Ray async actors execute coroutine methods on an event-loop thread, not
        # the Python process's main thread.  ttrl_math deliberately uses SIGALRM
        # around potentially hanging SymPy work, so neither direct execution nor
        # asyncio's thread executor is signal-safe.  A persistent spawn pool pays
        # process startup once and preserves the exact Math-RL verifier semantics.
        self.math_score_pool = ProcessPoolExecutor(
            max_workers=math_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )

    async def _decode_response(self, data_item) -> str:
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        return await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

    async def _score_code(self, data_item) -> dict[str, float]:
        info, jobs, row_job_specs = await self.loop.run_in_executor(
            None, lambda: build_row_jobs(data_item, 0, self.tokenizer, self.code_cfg)
        )
        if jobs:
            futures = await self.loop.run_in_executor(
                None, lambda: submit_row_jobs(jobs, info.route, self.code_cfg)
            )
            results = await _gather_futures(futures)
        else:
            results = []
        results_by_key = {(result.owner_id, result.job_idx): result for result in results}
        return _normalize_result(aggregate_row_results(info, row_job_specs, results_by_key))

    def _score_kwargs(self, data_item, response_str: str) -> dict:
        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth, extra_info = _validation_score_inputs(data_item)
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields")
        if isinstance(tool_extra_fields, Mapping):
            extra_info.update(tool_extra_fields.items())

        return {
            "data_source": data_source,
            "solution_str": response_str,
            "ground_truth": ground_truth,
            "extra_info": extra_info,
            "reward_router_address": None,
            "reward_model_tokenizer": None,
        }

    async def _score_math(self, data_item) -> dict[str, float]:
        response_str = await self._decode_response(data_item)
        kwargs = self._score_kwargs(data_item, response_str)
        result = await self.loop.run_in_executor(
            self.math_score_pool,
            _score_math_in_process,
            kwargs["data_source"],
            kwargs["solution_str"],
            kwargs["ground_truth"],
            kwargs["extra_info"],
        )
        return _normalize_result(result)

    async def _score_generic(self, data_item) -> dict[str, float]:
        response_str = await self._decode_response(data_item)
        kwargs = self._score_kwargs(data_item, response_str)
        if self.is_async_reward_score:
            result = await self.compute_score(**kwargs)
        else:
            # IF scoring is signal-free and can leave the actor event loop free to
            # accept other completed rollouts while its CPU work is in flight.
            result = await self.loop.run_in_executor(None, lambda: self.compute_score(**kwargs))
        return _normalize_result(result)

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        data_source = data_item.non_tensor_batch[self.code_cfg.reward_fn_key]
        if _is_code_data_source(data_source):
            reward_extra_info = await self._score_code(data_item)
        elif _is_if_data_source(data_source):
            reward_extra_info = await self._score_generic(data_item)
        else:
            # opd_val_dispatch intentionally treats math and unknown sources as
            # ttrl_math, so keep the same fallback here.
            reward_extra_info = await self._score_math(data_item)
        return {
            "reward_score": reward_extra_info["score"],
            "reward_extra_info": reward_extra_info,
        }
