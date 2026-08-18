"""Per-sample hybrid code reward, executed on the generic subprocess pool.

Routing is decided per row, by data shape, not by a global config switch:

* Rows carrying a full LiveCodeBench problem (top-level ``metadata`` or
  ``extra_info.metadata`` with
  question id + public/private testcases) -> **official** route: official code
  extraction + official testcase runner. This keeps RL validation aligned with
  the eval / SFT pipeline.
* Everything else (DeepCoder TACO / PrimeIntellect / LCB training rows, which
  ship only testcases and no problem metadata) -> **training** route: generic
  code extraction + rLLM testcase runner.

Both routes reduce a completion to "run code against N testcases, all pass -> 1"
and emit, per testcase, a pass/fail bool. The work is sharded to testcase
granularity (optionally chunked) and fanned out across the cluster.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import ray
import torch

from verl import DataProto
from verl.trainer.ppo.subprocess_pool import (
    DEFAULT_WORKERS_PER_NODE,
    Job,
    get_subprocess_worker_pool,
    _wait_with_progress,
)
from verl.utils.reward_score.code_testcase_runners import (
    decode_official_testcases,
    extract_official_code,
    extract_train_code,
    is_official_lcb_metadata,
    normalize_test_cases,
    parse_metadata,
    run_official_lcb_testcases,
    run_rllm_lcb_testcases,
    warmup_official_runner,
    warmup_rllm_runner,
)


THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>", re.DOTALL | re.IGNORECASE)
THINK_FORMAT_WEIGHT = 0.2
ANSWER_WEIGHT = 0.8
MIN_TESTCASE_MEMORY_LIMIT_MB = 2048

# answer_length_cap (alc) for the CODE reward path — mirrors the math/naive.py impl.
# Caps the *answer* reward (0..ANSWER_WEIGHT) by a piecewise-linear schedule of response
# length; think_score is untouched. breakpoints = [[offset_from_max, cap], ...] sorted by
# offset DESCENDING (offset = max_resp_len - resp_len). Wrong answers (answer_score 0) are
# unaffected; long/truncated correct answers are graded down (not masked).
_ALC_DEFAULT_BREAKPOINTS = [[4096, 0.8], [2048, 0.5], [0, 0.0]]


def _cfg_get(cfg, key, default):
    """Read a key from a dict / OmegaConf DictConfig / attr-style namespace."""
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except Exception:
            pass
    return getattr(cfg, key, default)


def _answer_length_cap(resp_len, max_resp_len, breakpoints):
    """Piecewise-linear cap on the answer reward as a function of response length."""
    offset = float(max_resp_len) - float(int(resp_len))
    pts = [(float(o), float(c)) for o, c in breakpoints]
    if offset >= pts[0][0]:
        return pts[0][1]
    for (o_hi, c_hi), (o_lo, c_lo) in zip(pts, pts[1:]):
        if o_lo <= offset <= o_hi:
            if o_hi == o_lo:
                return c_lo
            frac = (offset - o_lo) / (o_hi - o_lo)
            return c_lo + frac * (c_hi - c_lo)
    return pts[-1][1]


def _has_closed_think_block(text: str) -> bool:
    return THINK_BLOCK_RE.search(str(text or "")) is not None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_optional_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class CodeRewardInfo:
    index: int
    owner_id: str
    valid_response_length: int
    total_tests: int
    has_code: bool
    route: str
    data_source: str
    dataset: str
    sample_id: str
    request_id: str
    think_format_score: float
    answer_length_cap: float | None = None


@dataclass(frozen=True)
class CodeRewardConfig:
    """Resolved code-reward knobs, parsed once from the trainer config.

    Both the batched (driver-side) plan and the per-sample (reward-actor-side)
    inline path read scoring parameters through this struct so the two paths
    score identically.
    """

    timeout: int
    chunk_size: int
    lm_style: str
    enable_official: bool
    train_max_tests: int | None
    workers_per_node: int
    worker_num_cpus: float
    memory_limit_mb: int | None
    reward_fn_key: str
    # answer_length_cap (alc): cap the answer reward by response length when enabled.
    length_cap_enable: bool
    length_cap_breakpoints: tuple
    max_resp_len: int


def parse_code_reward_config(config: Any) -> CodeRewardConfig:
    reward_kwargs = dict((config.get("custom_reward_function") or {}).get("reward_kwargs", {}) or {})
    timeout = max(1, _as_int(reward_kwargs.get("timeout", 6), 6))
    chunk_size = max(1, _as_int(reward_kwargs.get("testcase_chunk_size", 1), 1))
    lm_style = str(reward_kwargs.get("lcb_lm_style", "CodeQwenInstruct") or "CodeQwenInstruct")
    enable_official = _as_bool(reward_kwargs.get("enable_official_lcb_val", True))
    train_max_tests_raw = _as_int(reward_kwargs.get("max_tests", 0), 0)
    train_max_tests = None if train_max_tests_raw <= 0 else train_max_tests_raw

    reward_model_kwargs = config.reward_model.get("reward_kwargs", {}) or {}
    workers_per_node = max(1, _as_int(reward_model_kwargs.get("testcase_workers_per_node", DEFAULT_WORKERS_PER_NODE), DEFAULT_WORKERS_PER_NODE))
    worker_num_cpus = max(0.001, _as_float(reward_model_kwargs.get("testcase_worker_cpus", 1.0), 1.0))
    sandbox_config = config.reward_model.get("sandbox_fusion", {}) or {}
    memory_limit_mb = _as_optional_positive_int(
        reward_model_kwargs.get("testcase_memory_limit_mb", sandbox_config.get("memory_limit_mb"))
    )
    if memory_limit_mb is not None:
        memory_limit_mb = max(MIN_TESTCASE_MEMORY_LIMIT_MB, memory_limit_mb)

    reward_fn_key = str(config.data.reward_fn_key)

    # answer_length_cap (alc) from reward_model.reward_kwargs.length_cap_cfg
    length_cap_cfg = reward_model_kwargs.get("length_cap_cfg")
    length_cap_enable = _as_bool(_cfg_get(length_cap_cfg, "enable", False))
    breakpoints_raw = _cfg_get(length_cap_cfg, "breakpoints", _ALC_DEFAULT_BREAKPOINTS)
    length_cap_breakpoints = tuple((float(o), float(c)) for o, c in breakpoints_raw)
    max_resp_len = _as_int(
        reward_model_kwargs.get("max_resp_len", _cfg_get(config.data, "max_response_length", 30000)),
        30000,
    )

    return CodeRewardConfig(
        timeout=timeout,
        chunk_size=chunk_size,
        lm_style=lm_style,
        enable_official=enable_official,
        train_max_tests=train_max_tests,
        workers_per_node=workers_per_node,
        worker_num_cpus=worker_num_cpus,
        memory_limit_mb=memory_limit_mb,
        reward_fn_key=reward_fn_key,
        length_cap_enable=length_cap_enable,
        length_cap_breakpoints=length_cap_breakpoints,
        max_resp_len=max_resp_len,
    )


def build_row_jobs(
    data_item: Any, item_idx: int, tokenizer: Any, cfg: CodeRewardConfig
) -> tuple[CodeRewardInfo, list[Job], list[tuple[int, int]]]:
    """Build the subprocess jobs scoring a single completion.

    Returns the row's :class:`CodeRewardInfo`, the list of :class:`Job` to run
    on the subprocess pool, and the row's ``job_specs`` entries (``(start,
    n_tests)`` per job) used by :func:`aggregate_row_results` to place results.
    """
    prompt_ids = data_item.batch["prompts"]
    prompt_length = prompt_ids.shape[-1]
    response_ids = data_item.batch["responses"]
    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
    valid_response_ids = response_ids[:valid_response_length]
    response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
    think_format_score = float(_has_closed_think_block(response_str))

    extra_info = data_item.non_tensor_batch.get("extra_info", {}) or {}
    metadata = parse_metadata(
        extra_info.get("metadata") or data_item.non_tensor_batch.get("metadata")
    )
    official = cfg.enable_official and is_official_lcb_metadata(metadata)
    raw_data_source = data_item.non_tensor_batch.get(cfg.reward_fn_key)
    if raw_data_source in (None, ""):
        raw_data_source = data_item.non_tensor_batch.get("dataset")
    if raw_data_source in (None, ""):
        if not official:
            raise KeyError(
                f"non-official code reward row is missing {cfg.reward_fn_key!r}"
            )
        # Formal LiveCodeBench rows are self-contained in extra_info.metadata.
        # The async agent-loop request may omit the RL routing key, which the
        # official runner neither consumes nor needs.  Keep a stable source
        # label for ownership/debugging without blocking validation scoring.
        raw_data_source = extra_info.get("data_source") or extra_info.get("dataset") or "livecodebench"
    data_source = str(raw_data_source)
    dataset = str(extra_info.get("dataset") or data_source)
    sample_id = str(extra_info.get("sample_id", item_idx))
    request_id = str(extra_info.get("request_id", ""))

    if official:
        # Standalone/formal LiveCodeBench validation rows carry the complete
        # problem and public/private tests in extra_info.metadata.  They do not
        # necessarily have the RL-only reward_model column, and the official
        # path does not need it.
        ground_truth = None
        test_cases = decode_official_testcases(metadata)
        model_code = extract_official_code(response_str, cfg.lm_style)
        job_code = model_code
        runner = run_official_lcb_testcases
    else:
        reward_model = data_item.non_tensor_batch.get("reward_model")
        if not isinstance(reward_model, dict) or "ground_truth" not in reward_model:
            raise KeyError(
                "non-official code reward row is missing reward_model.ground_truth "
                f"(data_source={data_source!r}, dataset={dataset!r}, sample_id={sample_id!r})"
            )
        ground_truth = reward_model["ground_truth"]
        test_cases = normalize_test_cases(ground_truth, max_tests=cfg.train_max_tests)
        model_code = extract_train_code(response_str)
        job_code = model_code
        runner = run_rllm_lcb_testcases

    has_code = bool(model_code)
    owner_id = f"{dataset}:{sample_id}:{item_idx}"
    vrl = int(valid_response_length.item())
    answer_length_cap = (
        _answer_length_cap(vrl, cfg.max_resp_len, cfg.length_cap_breakpoints)
        if cfg.length_cap_enable
        else None
    )
    info = CodeRewardInfo(
        index=item_idx,
        owner_id=owner_id,
        valid_response_length=vrl,
        total_tests=len(test_cases),
        has_code=has_code,
        route="official" if official else "train",
        data_source=data_source,
        dataset=dataset,
        sample_id=sample_id,
        request_id=request_id,
        think_format_score=think_format_score,
        answer_length_cap=answer_length_cap,
    )

    jobs: list[Job] = []
    job_specs: list[tuple[int, int]] = []
    if not test_cases or not has_code:
        return info, jobs, job_specs

    if official:
        for target_idx in range(len(test_cases)):
            prefix = test_cases[: target_idx + 1]
            job_specs.append((target_idx, 1))
            jobs.append(
                Job(
                    fn=runner,
                    kwargs={
                        "code": job_code,
                        "test_cases": prefix,
                        "timeout": cfg.timeout,
                        "target_test_idx": target_idx,
                    },
                    timeout=(cfg.timeout + 1) * len(prefix) + 5,
                    owner_id=owner_id,
                    job_idx=target_idx,
                    memory_limit_mb=cfg.memory_limit_mb,
                )
            )
    else:
        for start in range(0, len(test_cases), cfg.chunk_size):
            chunk = test_cases[start : start + cfg.chunk_size]
            job_timeout = cfg.timeout * len(chunk) + 6
            job_specs.append((start, len(chunk)))
            jobs.append(
                Job(
                    fn=runner,
                    kwargs={"code": job_code, "test_cases": chunk, "timeout": cfg.timeout},
                    timeout=job_timeout,
                    owner_id=owner_id,
                    job_idx=start,
                    memory_limit_mb=cfg.memory_limit_mb,
                )
            )
    return info, jobs, job_specs


def build_code_reward_plan(data: DataProto, tokenizer: Any, config: Any) -> dict[str, Any]:
    cfg = parse_code_reward_config(config)

    infos: list[CodeRewardInfo] = []
    jobs: list[Job] = []
    # owner_id -> list of (start_index, n_tests_in_job) so gather can place results
    # back in order and fail whole chunks whose job timed out / crashed.
    job_specs: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for item_idx in range(len(data)):
        info, row_jobs, row_specs = build_row_jobs(data[item_idx], item_idx, tokenizer, cfg)
        infos.append(info)
        jobs.extend(row_jobs)
        if row_specs:
            job_specs[info.owner_id].extend(row_specs)

    needs_official = any(i.route == "official" and job_specs.get(i.owner_id) for i in infos)
    needs_rllm = any(i.route == "train" and job_specs.get(i.owner_id) for i in infos)
    chunk_size = cfg.chunk_size
    timeout = cfg.timeout
    workers_per_node = cfg.workers_per_node
    worker_num_cpus = cfg.worker_num_cpus
    memory_limit_mb = cfg.memory_limit_mb

    print(
        "[code reward pool] "
        f"batch={len(data)} completions={len(infos)} jobs={len(jobs)} "
        f"train_chunk={chunk_size} official_chunk=prefix timeout={timeout} "
        f"workers_per_node={workers_per_node} worker_cpus={worker_num_cpus} "
        f"memory_limit_mb={memory_limit_mb or 'none'} "
        f"official={sum(1 for i in infos if i.route == 'official')} train={sum(1 for i in infos if i.route == 'train')} "
        f"has_code={sum(1 for i in infos if i.has_code)} total_tests={sum(i.total_tests for i in infos)}",
        flush=True,
    )

    pool = get_subprocess_worker_pool(workers_per_node=workers_per_node, worker_num_cpus=worker_num_cpus)
    warmups = []
    if needs_rllm:
        warmups.append(warmup_rllm_runner)
    if needs_official:
        warmups.append(warmup_official_runner)
    if warmups:
        print("[code reward pool] warmup begin", flush=True)
        pool.warmup(warmups)
        print("[code reward pool] warmup end", flush=True)
    print("[code reward pool] submit begin", flush=True)
    futures = pool.submit(jobs)
    print(f"[code reward pool] submit end futures={len(futures)}", flush=True)

    return {
        "mode": "code_reward_pool",
        "data": data,
        "infos": infos,
        "job_specs": dict(job_specs),
        "futures": futures,
    }


EXTRA_KEYS: tuple[str, ...] = (
    "acc",
    "score",
    "think_score",
    "answer_score",
    "final_score",
    "base_score",
    "length_cap",
    "length_capped",
    "format_score",
    "think_format_score",
    "passed_tests",
    "total_tests",
    "has_code",
    "route",
    "reward_data_source",
    "dataset",
    "sample_id",
    "request_id",
    "test_results",
    "error_codes",
)


def submit_row_jobs(jobs: list[Job], route: str, cfg: CodeRewardConfig) -> list[Any]:
    """Warm up (idempotently) and submit one completion's jobs to the shared pool.

    Returns the list of Ray futures; the caller is responsible for collecting
    them (the per-sample inline path awaits them on the asyncio loop, so it does
    not block while testcases run).
    """
    # shared=True: the inline per-sample path runs inside many RewardManagerWorker
    # actors, which must converge on one cluster-wide set of sandbox workers. The
    # batched (driver) path keeps shared=False (anonymous actors, unchanged).
    pool = get_subprocess_worker_pool(
        workers_per_node=cfg.workers_per_node, worker_num_cpus=cfg.worker_num_cpus, shared=True
    )
    warmup = warmup_official_runner if route == "official" else warmup_rllm_runner
    pool.warmup([warmup])
    return pool.submit(jobs)


def aggregate_row_results(
    info: CodeRewardInfo,
    row_job_specs: list[tuple[int, int]],
    results_by_key: dict[tuple[str, int], Any],
    on_non_timeout_failure: Any = None,
) -> dict[str, Any]:
    """Reduce one completion's testcase results into its reward + extras.

    Shared by the batched (driver) gather and the per-sample inline reward loop
    so both produce byte-identical scores and ``reward_extra_info`` keys. The
    returned dict carries every key in :data:`EXTRA_KEYS`; ``final_score`` is the
    scalar reward.
    """
    import json

    bools = [False] * info.total_tests
    errs = [0] * info.total_tests
    for start, n_tests in row_job_specs:
        jr = results_by_key.get((info.owner_id, start))
        if jr is not None and jr.ok and isinstance(jr.value, list):
            for offset, result in enumerate(jr.value[:n_tests]):
                bools[start + offset] = bool(result.get("passed"))
                error_code = int(result.get("error_code", 0) or 0)
                errs[start + offset] = error_code
                if on_non_timeout_failure is not None:
                    on_non_timeout_failure(info, start + offset, error_code, result)
        else:
            # Timed-out / crashed / missing job: whole chunk counts as failed.
            for offset in range(n_tests):
                errs[start + offset] = -3
            if jr is not None and not jr.ok and jr.error != "Time Limit Exceeded":
                if on_non_timeout_failure is not None:
                    for offset in range(n_tests):
                        on_non_timeout_failure(
                            info,
                            start + offset,
                            -4,
                            {"error": jr.error or "reward job failed without error text"},
                        )

    passed_tests = sum(bools)
    base_score = float(bool(info.total_tests > 0 and info.has_code and passed_tests == info.total_tests))
    think_score = THINK_FORMAT_WEIGHT * info.think_format_score
    answer_score = ANSWER_WEIGHT * base_score
    # answer_length_cap (alc): cap the answer reward by response length (think_score intact).
    length_cap = info.answer_length_cap
    capped_answer = answer_score if length_cap is None else min(answer_score, length_cap)
    final_score = think_score + capped_answer

    return {
        "acc": base_score,
        "score": final_score,
        "think_score": think_score,
        "answer_score": capped_answer,
        "final_score": final_score,
        "base_score": base_score,
        "length_cap": length_cap if length_cap is not None else float("nan"),
        "length_capped": float(length_cap is not None and capped_answer < answer_score),
        "format_score": info.think_format_score,
        "think_format_score": info.think_format_score,
        "passed_tests": passed_tests,
        "total_tests": info.total_tests,
        "has_code": float(info.has_code),
        "route": info.route,
        "reward_data_source": info.data_source,
        "dataset": info.dataset,
        "sample_id": info.sample_id,
        "request_id": info.request_id,
        "test_results": json.dumps(bools),
        "error_codes": json.dumps(errs),
    }


def gather_code_reward_plan(plan: dict[str, Any]) -> tuple[torch.Tensor, dict[str, list[Any]]]:
    import json

    data: DataProto = plan["data"]
    infos: list[CodeRewardInfo] = plan["infos"]
    job_specs: dict[str, list[tuple[int, int]]] = plan["job_specs"]
    futures = plan["futures"]
    print(f"[code reward pool] gather begin futures={len(futures)}", flush=True)
    results = _wait_with_progress(futures, label="code reward jobs") if futures else []
    print(f"[code reward pool] gather end results={len(results)}", flush=True)

    results_by_key: dict[tuple[str, int], Any] = {(jr.owner_id, jr.job_idx): jr for jr in results}
    non_timeout_failure_counts: Counter[int] = Counter()
    non_timeout_failure_examples: list[dict[str, Any]] = []

    def remember_non_timeout_failure(info: CodeRewardInfo, test_idx: int, error_code: int, result: Any) -> None:
        # -2 is ordinary wrong answer; -3 is testcase timeout. Everything else is
        # a non-timeout testcase failure, usually generated-code runtime error or
        # occasionally evaluator/import infrastructure trouble.
        if error_code in (0, -2, -3):
            return
        non_timeout_failure_counts[error_code] += 1
        if len(non_timeout_failure_examples) >= 8:
            return
        error = ""
        if isinstance(result, dict):
            error = str(result.get("error") or result.get("error_message") or "")
        non_timeout_failure_examples.append(
            {
                "dataset": info.dataset,
                "sample_id": info.sample_id,
                "request_id": info.request_id,
                "route": info.route,
                "test_idx": test_idx,
                "error_code": error_code,
                "error": error[:500],
            }
        )

    reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
    extras: dict[str, list[Any]] = {key: [] for key in EXTRA_KEYS}

    for info in infos:
        row = aggregate_row_results(
            info, job_specs.get(info.owner_id, []), results_by_key, remember_non_timeout_failure
        )
        reward_tensor[info.index, info.valid_response_length - 1] = row["final_score"]
        for key in EXTRA_KEYS:
            extras[key].append(row[key])

    if non_timeout_failure_counts:
        print(
            "[code reward pool] non-timeout testcase failures "
            f"counts={dict(non_timeout_failure_counts)} "
            f"examples={json.dumps(non_timeout_failure_examples, ensure_ascii=False)}",
            flush=True,
        )

    return reward_tensor, extras
