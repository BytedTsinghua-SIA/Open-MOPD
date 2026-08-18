# Copyright 2025 Individual Contributor: Thibaut Barroyer
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

import importlib.util
import inspect
import math
import multiprocessing
import os
import sys
import warnings
from functools import partial
from typing import Any, Optional

import ray
import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.trainer.ppo.code_reward_pool import (
    build_code_reward_plan,
    gather_code_reward_plan,
)
from verl.utils.reward_score import default_compute_score
from verl.utils.transferqueue_utils import tqbridge
from verl.workers.reward_manager import get_reward_manager_cls
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn


_DISTRIBUTED_REWARD_KWARGS = {
    "distributed",
    "distributed_flatten_tests",
    "distributed_num_shards",
    "distributed_cpu_utilization",
    "distributed_cpus_per_shard",
    "distributed_max_workers_per_shard",
    "distributed_min_items_per_shard",
    "distributed_test_cpus",
    "testcase_workers_per_node",
    "testcase_worker_cpus",
    "testcase_memory_limit_mb",
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _reward_model_kwargs(config: DictConfig, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    kwargs = dict(config.reward_model.get("reward_kwargs", {}) or {})
    if overrides:
        kwargs.update(overrides)
    for key in _DISTRIBUTED_REWARD_KWARGS:
        kwargs.pop(key, None)
    return kwargs


def is_code_reward_config(config: DictConfig) -> bool:
    reward_fn_config = config.get("custom_reward_function") or {}
    path = str(reward_fn_config.get("path", "") or "")
    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}) or {})
    scoring_mode = str(reward_kwargs.get("scoring_mode", "") or "").strip().lower()
    return (
        path.endswith("rllm_code_reward.py")
        or path.endswith("official_lcb_reward.py")
        or scoring_mode in {"official_lcb", "livecodebench_official", "eval_official"}
    )


def should_use_distributed_reward(config: DictConfig) -> bool:
    if is_code_reward_config(config):
        return True
    reward_kwargs = config.reward_model.get("reward_kwargs", {}) or {}
    return _as_bool(reward_kwargs.get("distributed", False))


def should_flatten_reward_tests(config: DictConfig) -> bool:
    return is_code_reward_config(config)


def should_reuse_inline_reward(
    config: DictConfig,
    *,
    use_rm: bool,
    has_rm_scores: bool,
    is_validation: bool,
) -> bool:
    """Return whether an inline trajectory score can skip driver-side scoring.

    Rule-reward RL can reuse inline scores for both train and validation. OPD
    must reserve its training ``rm_scores`` for teacher distillation, but a
    recipe may explicitly enable a separate validation-only inline scorer.
    """
    if not has_rm_scores:
        return False
    if not use_rm:
        return True
    return is_validation and _as_bool(config.reward_model.get("enable_validation_reward_async", False))


def _cluster_cpu_count() -> int:
    try:
        return int(
            sum(float(node.get("Resources", {}).get("CPU", 0.0)) for node in ray.nodes() if node.get("Alive", False))
        )
    except Exception:
        return int(os.cpu_count() or 1)


def _distributed_reward_plan(config: DictConfig, batch_size: int) -> tuple[int, int, int]:
    reward_kwargs = config.reward_model.get("reward_kwargs", {}) or {}
    cpu_utilization = float(reward_kwargs.get("distributed_cpu_utilization", 0.9))
    cpus_per_shard = max(1, int(reward_kwargs.get("distributed_cpus_per_shard", 8)))
    min_items_per_shard = max(1, int(reward_kwargs.get("distributed_min_items_per_shard", 1)))
    requested_shards = int(reward_kwargs.get("distributed_num_shards", 0) or 0)

    usable_cpus = max(1, int(_cluster_cpu_count() * cpu_utilization))
    auto_shards = max(1, usable_cpus // cpus_per_shard)
    max_useful_shards = max(1, math.ceil(max(1, batch_size) / min_items_per_shard))
    num_shards = requested_shards if requested_shards > 0 else auto_shards
    num_shards = max(1, min(num_shards, max_useful_shards, max(1, batch_size)))

    max_workers = int(reward_kwargs.get("distributed_max_workers_per_shard", 0) or 0)
    if max_workers <= 0:
        max_workers = cpus_per_shard
    max_workers = max(1, max_workers)
    return num_shards, cpus_per_shard, max_workers


def _custom_reward_kwargs(config: DictConfig) -> dict[str, Any]:
    return dict((config.get("custom_reward_function") or {}).get("reward_kwargs", {}) or {})


def _call_with_kwargs(raw_fn, extra_kwargs, *args, **kwargs):
    """Calls `raw_fn` by merging `extra_kwargs` into call-time `kwargs`, with `extra_kwargs` taking precedence.

    This function is used to merge additional keyword arguments with the original function's arguments.
    """
    merged_kwargs = {**kwargs, **extra_kwargs}
    return raw_fn(*args, **merged_kwargs)


async def _call_with_kwargs_async(raw_fn, extra_kwargs, *args, **kwargs):
    """Calls `raw_fn` by merging `extra_kwargs` into call-time `kwargs`, with `extra_kwargs` taking precedence.

    This function is used to merge additional keyword arguments with the original function's arguments.
    """
    merged_kwargs = {**kwargs, **extra_kwargs}
    return await raw_fn(*args, **merged_kwargs)


def get_custom_reward_fn(config: DictConfig) -> Optional[RawRewardFn]:
    """Load and return a custom reward function from external file.

    Dynamically imports a reward function from a specified file path and wraps
    it with additional keyword arguments from the configuration.

    Args:
        config (dict): Configuration dictionary containing custom_reward_function
                      settings with 'path', 'name', and 'reward_kwargs' fields.

    Returns:
        callable or None: Wrapped reward function with merged kwargs, or None
                         if no custom reward function is configured.

    Raises:
        FileNotFoundError: If the specified reward function file doesn't exist.
        RuntimeError: If there's an error loading the module from file.
        AttributeError: If the specified function name isn't found in the module.
    """

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    function_name = reward_fn_config.get("name")
    assert function_name is not None

    module = sys.modules.get("custom_module", None)
    if module is None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

        spec = importlib.util.spec_from_file_location("custom_module", file_path)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_module"] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{module.__file__}'.")

    print(f"using customized reward function '{function_name}' from '{module.__file__}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    if not inspect.iscoroutinefunction(raw_fn):
        return partial(_call_with_kwargs, raw_fn, reward_kwargs)
    else:
        return partial(_call_with_kwargs_async, raw_fn, reward_kwargs)


def load_reward_manager(
    config: DictConfig, tokenizer: Any, num_examine: int, **reward_kwargs: Any
) -> AbstractRewardManager:
    """
    Load and initialize a reward manager based on the configuration.

    Args:
        config: PPO trainer configuration object containing reward_model fields.
        tokenizer: Tokenizer object used for processing text.
        num_examine: Number of samples to examine.
        **reward_kwargs: Additional keyword arguments for the reward manager.

    Returns:
        An instance of the specified reward manager class.
    """

    # Try to get a custom reward function based on the configuration
    # user defined reward manager can be registered in custom_reward_fn
    compute_score = get_custom_reward_fn(config)
    final_compute_score = compute_score

    # The list of pre-defined reward managers are defined in `verl/workers/reward_manager/`:
    # naive: NaiveRewardManager
    # prime: PrimeRewardManager
    # batch: BatchRewardManager
    # dapo: DAPORewardManager
    # Note(haibin.lin): For custom reward managers, please make sure they are imported and
    # registered via `verl.workers.reward_manager.register`
    # By default reward_manager is set to naive (NaiveRewardManager)
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    reward_manager_cls = get_reward_manager_cls(reward_manager_name)

    if compute_score is None:
        sandbox_config = config.reward_model.get("sandbox_fusion")
        sandbox_url = sandbox_config.get("url") if sandbox_config else None
        memory_limit_mb = sandbox_config.get("memory_limit_mb", 1024) if sandbox_config else 1024
        if sandbox_url:
            sandbox_manager = multiprocessing.Manager()
            # Create a semaphore to control concurrent access to the sandbox
            _concurrent_semaphore = sandbox_manager.Semaphore(sandbox_config.get("max_concurrent", 64))
            final_compute_score = partial(
                default_compute_score,
                sandbox_fusion_url=sandbox_url,
                concurrent_semaphore=_concurrent_semaphore,
                memory_limit_mb=memory_limit_mb,
            )
        else:
            final_compute_score = default_compute_score

    # Instantiate and return the reward manager with the specified parameters
    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=final_compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **_reward_model_kwargs(config, reward_kwargs),
    )


@tqbridge(put_data=False)
def compute_reward(data: DataProto, reward_fn: AbstractRewardManager) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute reward for a batch of data.
    Args:
        data: DataProto object containing the input data.
        reward_fn: Reward function to compute the reward.
    Returns:
        Tuple of reward tensor and extra info dictionary.
    """
    try:
        reward_result = reward_fn(data, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
    except Exception as e:
        print(f"Error in reward_fn: {e}")
        reward_tensor = reward_fn(data)
        reward_extra_infos_dict = {}

    return reward_tensor, reward_extra_infos_dict


@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config=None, tokenizer=None, reward_fn=None, reward_manager_kwargs_override=None):
    """
    Load the reward manager and compute the reward for a batch of data.
    This is meant to be run in a separate Ray worker.
    """
    if reward_fn is None:
        assert config is not None and tokenizer is not None, (
            "config and tokenizer must not be None when reward_fn is None"
        )

        warnings.warn("using config and tokenizer with compute_reward_async is deprecated", stacklevel=2)
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **(reward_manager_kwargs_override or {})
        )

    return compute_reward(data, reward_fn)


@ray.remote(num_cpus=1)
def compute_code_testcase_async(test_case: dict[str, Any], generation: str, timeout: int) -> dict[str, Any]:
    import multiprocessing
    import time
    from queue import Empty

    import verl.utils.reward_score.rllm_code_reward  # noqa: F401
    from rllm.rewards.code_reward import _temp_run_single, postprocess_lcb_sample

    output_queue = multiprocessing.Queue(maxsize=1)
    process = multiprocessing.Process(
        target=_temp_run_single,
        args=(postprocess_lcb_sample([test_case]), generation, False, timeout, output_queue),
    )
    process.start()
    deadline = time.monotonic() + timeout + 6
    process.join(timeout=max(0.0, deadline - time.monotonic()))

    if process.is_alive():
        process.kill()
        process.join()
        result = -3
        metadata = {"error_code": -3, "error_message": "Time Limit Exceeded"}
    else:
        try:
            raw_result, metadata = output_queue.get(timeout=1)
        except Empty:
            raw_result, metadata = [-4], {"error_code": -4, "error_message": "No result returned"}
        result = raw_result[0] if isinstance(raw_result, list) and raw_result else raw_result
        metadata = metadata or {}
    output_queue.close()
    output_queue.cancel_join_thread()
    return {
        "passed": bool(result is True),
        "result": result,
        "metadata": metadata,
    }


def _merge_reward_results(
    results: list[tuple[torch.Tensor, dict[str, Any]]],
) -> tuple[torch.Tensor, dict[str, list[Any]]]:
    reward_tensor = torch.cat([item[0] for item in results], dim=0)
    merged_extra: dict[str, list[Any]] = {}
    for _tensor, extra in results:
        for key, values in (extra or {}).items():
            bucket = merged_extra.setdefault(key, [])
            if isinstance(values, torch.Tensor):
                bucket.extend(values.detach().cpu().tolist())
            elif hasattr(values, "tolist"):
                converted = values.tolist()
                bucket.extend(converted if isinstance(converted, list) else [converted])
            elif isinstance(values, list):
                bucket.extend(values)
            else:
                bucket.append(values)
    return reward_tensor, merged_extra


def _prepare_code_test_jobs(data: DataProto, tokenizer: Any, config: DictConfig) -> dict[str, Any]:
    from verl.utils.reward_score.rllm_code_reward import DATA_SOURCE_ALIASES, _ensure_code_block, _load_ground_truth
    from rllm.rewards.code_reward import _select_longest_input_tests, extract_code_from_model

    reward_kwargs = _custom_reward_kwargs(config)
    max_tests = int(reward_kwargs.get("max_tests", 15) or 15)
    timeout = int(reward_kwargs.get("timeout", 6) or 6)
    reward_model_kwargs = config.reward_model.get("reward_kwargs", {}) or {}
    cpu_utilization = max(0.05, min(1.0, float(reward_model_kwargs.get("distributed_cpu_utilization", 0.9))))
    test_cpus = float(reward_model_kwargs.get("distributed_test_cpus", 0) or 0)
    if test_cpus <= 0:
        # One testcase normally consumes about one CPU. Reserving 1 / utilization
        # CPUs per task caps Ray scheduling near the requested cluster CPU fraction.
        test_cpus = 1.0 / cpu_utilization

    completion_infos: list[dict[str, Any]] = []
    test_futures: list[ray.ObjectRef] = []
    test_owner_indices: list[int] = []
    total_tests = 0

    for item_idx in range(len(data)):
        data_item = data[item_idx]
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_source = str(data_item.non_tensor_batch[config.data.reward_fn_key])
        reward_data_source = DATA_SOURCE_ALIASES.get(data_source, data_source)
        raw_tests = _load_ground_truth(ground_truth)

        tests: list[dict[str, Any]] = []
        if reward_data_source in {"livecodebench", "codeforces", "primeintellect"} and isinstance(raw_tests, list):
            tests = _select_longest_input_tests(raw_tests, max_tests)
        elif reward_data_source in {"livecodebench", "codeforces", "primeintellect"} and isinstance(raw_tests, dict):
            inputs = raw_tests.get("inputs") or []
            outputs = raw_tests.get("outputs") or []
            fn_name = raw_tests.get("fn_name")
            tests = [
                {
                    "input": inp,
                    "output": out,
                    **({"testtype": "functional", "metadata": {"func_name": fn_name}} if fn_name else {}),
                }
                for inp, out in zip(inputs, outputs, strict=False)
            ]
            tests = _select_longest_input_tests(tests, max_tests)

        model_code = extract_code_from_model(_ensure_code_block(response_str or ""))
        completion_info = {
            "index": item_idx,
            "valid_response_length": int(valid_response_length.item()),
            "num_tests": len(tests),
            "has_code": model_code is not None,
            "data_source": data_source,
        }
        completion_infos.append(completion_info)
        if not tests or model_code is None:
            continue

        for test_case in tests:
            test_owner_indices.append(item_idx)
            test_futures.append(
                compute_code_testcase_async.options(num_cpus=test_cpus).remote(test_case, model_code, timeout)
            )
        total_tests += len(tests)

    print(
        "[distributed reward:flatten_tests] "
        f"batch={len(data)} test_tasks={len(test_futures)} selected_tests={total_tests} "
        f"test_cpus={test_cpus} cluster_cpus={_cluster_cpu_count()}",
        flush=True,
    )
    return {
        "mode": "flatten_tests",
        "data": data,
        "completion_infos": completion_infos,
        "test_owner_indices": test_owner_indices,
        "test_futures": test_futures,
        # DAPO soft overlong punishment config (optional) threaded to the gather stage.
        "overlong_buffer_cfg": reward_model_kwargs.get("overlong_buffer_cfg", None),
        "max_resp_len": reward_model_kwargs.get("max_resp_len", config.data.get("max_response_length", None)),
    }


def _gather_code_test_jobs(plan: dict[str, Any]) -> tuple[torch.Tensor, dict[str, list[Any]]]:
    data = plan["data"]
    completion_infos = plan["completion_infos"]
    test_owner_indices = plan["test_owner_indices"]
    test_results = ray.get(plan["test_futures"]) if plan["test_futures"] else []

    item_results: dict[int, list[dict[str, Any]]] = {info["index"]: [] for info in completion_infos}
    for owner_idx, result in zip(test_owner_indices, test_results, strict=True):
        item_results[owner_idx].append(result)

    reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
    extras: dict[str, list[Any]] = {
        "acc": [],
        "score": [],
        "passed_tests": [],
        "total_tests": [],
        "reward_data_source": [],
    }
    # DAPO soft overlong punishment: graded length penalty added to the rule-based
    # code reward. acc/score stay PURE correctness (used as the dynamic-sampling
    # metric and the truth signal); only reward_tensor carries the penalty.
    _ol_cfg = plan.get("overlong_buffer_cfg")
    _ol_max = plan.get("max_resp_len")
    _ol_on = bool(_ol_cfg) and (_ol_cfg.get("enable", False) if hasattr(_ol_cfg, "get") else getattr(_ol_cfg, "enable", False)) and _ol_max
    if _ol_on:
        extras["overlong_reward"] = []
        _ol_len = _ol_cfg.get("len") if hasattr(_ol_cfg, "get") else _ol_cfg.len
        _ol_pf = _ol_cfg.get("penalty_factor") if hasattr(_ol_cfg, "get") else _ol_cfg.penalty_factor
    for info in completion_infos:
        results = item_results[info["index"]]
        passed_tests = sum(1 for result in results if result.get("passed"))
        total_tests = int(info["num_tests"])
        score = float(bool(total_tests > 0 and info["has_code"] and passed_tests == total_tests))
        reward = score
        if _ol_on:
            _exceed = int(info["valid_response_length"]) - (_ol_max - _ol_len)
            _ol_r = min(-_exceed / _ol_len * _ol_pf, 0.0)
            reward = score + _ol_r
            extras["overlong_reward"].append(_ol_r)
        reward_tensor[info["index"], info["valid_response_length"] - 1] = reward
        extras["acc"].append(score)
        extras["score"].append(score)
        extras["passed_tests"].append(passed_tests)
        extras["total_tests"].append(total_tests)
        extras["reward_data_source"].append(info["data_source"])
    return reward_tensor, extras


def submit_reward_distributed(data: DataProto, config: DictConfig, tokenizer: Any) -> list[ray.ObjectRef]:
    if should_flatten_reward_tests(config):
        return build_code_reward_plan(data, tokenizer, config)

    num_shards, cpus_per_shard, max_workers = _distributed_reward_plan(config, len(data))
    split_size = max(1, math.ceil(len(data) / num_shards))
    shards = data.split(split_size)
    print(
        "[distributed reward] "
        f"batch={len(data)} shards={len(shards)} split_size={split_size} "
        f"cpus_per_shard={cpus_per_shard} max_workers_per_shard={max_workers} "
        f"cluster_cpus={_cluster_cpu_count()}",
        flush=True,
    )
    return [
        compute_reward_async.options(num_cpus=cpus_per_shard).remote(
            data=shard,
            config=config,
            tokenizer=tokenizer,
            reward_manager_kwargs_override={"max_workers": max_workers},
        )
        for shard in shards
    ]


def gather_reward_distributed(futures: list[ray.ObjectRef] | dict[str, Any]) -> tuple[torch.Tensor, dict[str, list[Any]]]:
    if isinstance(futures, dict) and futures.get("mode") == "code_reward_pool":
        return gather_code_reward_plan(futures)
    if isinstance(futures, dict) and futures.get("mode") == "flatten_tests":
        return _gather_code_test_jobs(futures)
    return _merge_reward_results(ray.get(futures))


def compute_reward_distributed(data: DataProto, config: DictConfig, tokenizer: Any) -> tuple[torch.Tensor, dict[str, list[Any]]]:
    return gather_reward_distributed(submit_reward_distributed(data, config, tokenizer))


def collect_inline_reward_extra_infos(data: DataProto) -> dict[str, list[Any]]:
    """Rebuild ``reward_extra_infos_dict`` from an inline (async-rollout) reward.

    When ``rollout.mode=async``, each trajectory is scored the moment it finishes
    generating (per-sample reward loop in ``agent_loop.py``). That path writes
    every ``reward_extra_info`` field into ``non_tensor_batch`` and records the
    field names in ``meta_info['reward_extra_keys']``. The fit loop wants the
    ``dict[str, list]`` shape it would otherwise get from the batched reward, so
    recover it here. Rows are read from the same (possibly balance-reordered)
    DataProto, so they stay aligned with ``rm_scores``.
    """
    keys = data.meta_info.get("reward_extra_keys") or []
    out: dict[str, list[Any]] = {}
    for key in keys:
        if key in data.non_tensor_batch:
            out[key] = list(data.non_tensor_batch[key])
    return out
