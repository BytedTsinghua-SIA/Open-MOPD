import json
import base64
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.subprocess_pool import Job, JobResult, run_job_in_subprocess

VENDOR_ROOT = Path(__file__).resolve().parents[1] / "verl" / "utils" / "reward_score" / "rllm_vendor"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

from verl.trainer.ppo import code_reward_pool
from verl.trainer.ppo.reward import (
    _reward_model_kwargs,
    should_flatten_reward_tests,
    should_reuse_inline_reward,
    should_use_distributed_reward,
)
from verl.utils.reward_score import code_testcase_runners
from verl.utils.reward_score.code_testcase_runners import normalize_test_cases, run_rllm_lcb_testcases
from verl.utils.reward_score.rllm_vendor.rllm.rewards.code_reward import (
    lcb_check_correctness_v2,
    postprocess_lcb_sample,
)


def test_postprocess_lcb_sample_allows_functional_tests_without_func_name():
    sample = [
        {
            "input": "1 2\n",
            "output": "3\n",
            "testtype": "functional",
            "metadata": {},
        }
    ]

    processed = postprocess_lcb_sample(sample)
    input_output = json.loads(processed["input_output"])

    assert input_output == {"inputs": ["1 2\n"], "outputs": ["3\n"]}


def test_postprocess_lcb_sample_preserves_functional_func_name():
    sample = [
        {
            "input": "[1, 2]",
            "output": "3",
            "testtype": "functional",
            "metadata": {"func_name": "twoSum"},
        }
    ]

    processed = postprocess_lcb_sample(sample)
    input_output = json.loads(processed["input_output"])

    assert input_output["fn_name"] == "twoSum"


def test_lcb_check_correctness_parallel_testcases():
    sample = [
        {"input": "1 2\n", "output": "3\n", "metadata": {}},
        {"input": "3 4\n", "output": "7\n", "metadata": {}},
    ]
    code = "a, b = map(int, input().split())\nprint(a + b)"

    is_correct, details = lcb_check_correctness_v2(
        sample,
        code,
        timeout=3,
        max_tests=2,
        testcase_max_workers=2,
    )

    assert is_correct
    assert details["passed_tests"] == 2
    assert details["timeout"] == 3
    assert details["testcase_max_workers"] == 2


def test_official_prefix_runner_marks_later_tests_failed_after_early_failure(monkeypatch):
    def fake_check_correctness(sample, code, timeout, debug):  # noqa: ARG001
        inputs_outputs = json.loads(sample["input_output"])
        # Official LCB stops at the first failed testcase and returns only the
        # prefix it executed.
        if len(inputs_outputs["inputs"]) >= 2:
            return [-2], {"error_code": -2, "error_message": "Wrong Answer"}
        return [True], {"execution time": 0.01}

    monkeypatch.setattr(code_testcase_runners, "_official_check_correctness", lambda: fake_check_correctness)

    test_cases = [
        {"input": "bad\n", "output": "ok\n"},
        {"input": "good\n", "output": "good\n"},
    ]

    target = code_testcase_runners.run_official_lcb_testcases(
        code="print(input())",
        test_cases=test_cases,
        timeout=6,
        target_test_idx=1,
    )

    assert target == [{"passed": False, "error_code": 0}]


def test_train_runners_handle_supported_datasource_shapes():
    normalized_shapes = [
        normalize_test_cases({"inputs": ["1\n"], "outputs": ["1\n"]}, max_tests=None),
        normalize_test_cases([{"input": "2\n", "output": "2\n"}], max_tests=None),
        normalize_test_cases([{"input": "3\n", "output": "3\n", "metadata": {}}], max_tests=None),
    ]

    for test_cases in normalized_shapes:
        result = run_job_in_subprocess(
            Job(fn=run_rllm_lcb_testcases, kwargs={"code": "print(input())", "test_cases": test_cases, "timeout": 3})
        )
        assert result.ok
        assert result.value == [{"passed": True, "error_code": 0}]


def test_gather_places_official_prefix_results_by_target_index(monkeypatch):
    data = DataProto(
        batch=TensorDict(
            {
                "prompts": torch.ones((1, 3), dtype=torch.long),
                "responses": torch.ones((1, 4), dtype=torch.long),
                "attention_mask": torch.ones((1, 7), dtype=torch.long),
            },
            batch_size=[1],
        ),
        non_tensor_batch={},
    )
    owner_id = "livecodebench_v5:problem-0:0"
    info = code_reward_pool.CodeRewardInfo(
        index=0,
        owner_id=owner_id,
        valid_response_length=4,
        total_tests=3,
        has_code=True,
        route="official",
        data_source="livecodebench_v5",
        dataset="livecodebench_v5",
        sample_id="problem-0",
        request_id="problem-0:0",
        think_format_score=0.0,
    )
    results = [
        JobResult(owner_id, 0, ok=True, value=[{"passed": True, "error_code": 0}]),
        JobResult(owner_id, 1, ok=True, value=[{"passed": False, "error_code": -2}]),
        JobResult(owner_id, 2, ok=True, value=[{"passed": False, "error_code": 0}]),
    ]
    monkeypatch.setattr(code_reward_pool, "_wait_with_progress", lambda futures, label: results)

    rewards, extras = code_reward_pool.gather_code_reward_plan(
        {
            "data": data,
            "infos": [info],
            "job_specs": {owner_id: [(0, 1), (1, 1), (2, 1)]},
            "futures": ["fake-ref"] * 3,
        }
    )

    assert rewards[0, 3].item() == 0.0
    assert extras["passed_tests"] == [1]
    assert extras["test_results"] == [json.dumps([True, False, False])]
    assert extras["error_codes"] == [json.dumps([0, -2, 0])]


def test_code_reward_defaults_to_distributed_without_bool():
    config = OmegaConf.create(
        {
            "custom_reward_function": {
                "path": "verl/verl/utils/reward_score/rllm_code_reward.py",
                "name": "compute_score",
                "reward_kwargs": {"scoring_mode": "official_lcb"},
            },
            "reward_model": {"reward_kwargs": {}},
        }
    )

    assert should_use_distributed_reward(config)
    assert should_flatten_reward_tests(config)


def test_reward_manager_kwargs_filters_testcase_pool_controls_after_overrides():
    config = OmegaConf.create(
        {
            "reward_model": {
                "reward_kwargs": {
                    "max_workers": 16,
                    "testcase_workers_per_node": 100,
                    "testcase_memory_limit_mb": 5120,
                }
            }
        }
    )

    kwargs = _reward_model_kwargs(
        config,
        {
            "timeout": 6,
            "testcase_workers_per_node": 100,
            "testcase_memory_limit_mb": 5120,
        },
    )

    assert kwargs == {"max_workers": 16, "timeout": 6}


@pytest.mark.parametrize(
    ("use_rm", "is_validation", "enabled", "expected"),
    [
        (False, False, False, True),
        (False, True, False, True),
        (True, False, True, False),
        (True, True, False, False),
        (True, True, True, True),
    ],
)
def test_inline_reward_reuse_never_uses_opd_train_scores_as_rule_reward(
    use_rm, is_validation, enabled, expected
):
    config = OmegaConf.create({"reward_model": {"enable_validation_reward_async": enabled}})

    assert (
        should_reuse_inline_reward(
            config,
            use_rm=use_rm,
            has_rm_scores=True,
            is_validation=is_validation,
        )
        is expected
    )
    assert not should_reuse_inline_reward(
        config,
        use_rm=use_rm,
        has_rm_scores=False,
        is_validation=is_validation,
    )


def test_naive_reward_manager_can_skip_opd_diagnostic_rule_scoring():
    from verl.workers.reward_manager.naive import NaiveRewardManager

    def fail_if_called(**kwargs):  # noqa: ARG001
        raise AssertionError("teacher rm_scores must short-circuit rule scoring")

    rm_scores = torch.tensor([[0.1, -0.2]], dtype=torch.float32)
    data = DataProto(
        batch=TensorDict(
            {
                "responses": torch.ones((1, 2), dtype=torch.long),
                "rm_scores": rm_scores,
            },
            batch_size=[1],
        ),
        non_tensor_batch={"inline_acc": np.array([1.0])},
        meta_info={"reward_extra_keys": ["inline_acc"]},
    )
    manager = NaiveRewardManager(
        tokenizer=None,
        num_examine=0,
        compute_score=fail_if_called,
        compute_true_reward=False,
    )

    result = manager(data, return_dict=True)

    assert result["reward_tensor"] is rm_scores
    assert result["reward_extra_info"] == {"inline_acc": data.non_tensor_batch["inline_acc"]}


def test_official_lcb_metadata_keeps_all_public_private_tests_by_default():
    public_tests = [{"input": f"{idx}\n", "output": f"{idx}\n"} for idx in range(8)]
    private_tests = [{"input": f"{idx}\n", "output": f"{idx}\n"} for idx in range(8, 16)]
    encoded_private = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(private_tests)))).decode("utf-8")
    metadata = {
        "public_test_cases": json.dumps(public_tests),
        "private_test_cases": encoded_private,
        "metadata": json.dumps({"func_name": "solve"}),
    }

    tests = normalize_test_cases(metadata, max_tests=None)

    assert len(tests) == 16
    assert tests[0]["input"] == "0\n"
    assert tests[-1]["input"] == "15\n"
    assert tests[0]["testtype"] == "functional"
    assert tests[0]["metadata"] == {"func_name": "solve"}


class _FakePool:
    workers_per_node = 100

    def __init__(self):
        self.jobs = None
        self.warmups = None

    def warmup(self, fns):
        self.warmups = fns

    def submit(self, jobs):
        self.jobs = jobs
        return ["fake-ref"] * len(jobs)


class _FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
        return "```python\nprint(input())\n```"


def test_code_reward_pool_routes_full_lcb_metadata_to_official(monkeypatch):
    prompts = 342
    val_n = 2
    testcase_count = 16
    completions = prompts * val_n

    public_tests = [{"input": f"{idx}\n", "output": f"{idx}\n"} for idx in range(8)]
    private_tests = [{"input": f"{idx}\n", "output": f"{idx}\n"} for idx in range(8, 16)]
    encoded_private = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(private_tests)))).decode("utf-8")
    metadata = json.dumps(
        {
            "question_id": "q1",
            "public_test_cases": json.dumps(public_tests),
            "private_test_cases": encoded_private,
            "metadata": json.dumps({"func_name": "solve"}),
        }
    )

    fake_pool = _FakePool()
    monkeypatch.setattr(code_reward_pool, "get_subprocess_worker_pool", lambda **_: fake_pool)
    monkeypatch.setattr(code_reward_pool, "extract_official_code", lambda completion, lm_style: "print(input())")

    reward_model = np.array(
        [{"style": "rule", "ground_truth": metadata} for _ in range(completions)],
        dtype=object,
    )
    extra_info = np.array(
        [
            {
                "dataset": "livecodebench_v5",
                "sample_id": f"problem-{idx // val_n}",
                "request_id": f"problem-{idx // val_n}:{idx % val_n}",
                "metadata": metadata,
            }
            for idx in range(completions)
        ],
        dtype=object,
    )
    data = DataProto(
        batch=TensorDict(
            {
                "prompts": torch.ones((completions, 3), dtype=torch.long),
                "responses": torch.ones((completions, 4), dtype=torch.long),
                "attention_mask": torch.ones((completions, 7), dtype=torch.long),
            },
            batch_size=[completions],
        ),
        non_tensor_batch={
            "reward_model": reward_model,
            "data_source": np.array(["livecodebench_v5"] * completions, dtype=object),
            "extra_info": extra_info,
        },
    )
    config = OmegaConf.create(
        {
            "data": {"reward_fn_key": "data_source"},
            "custom_reward_function": {
                "path": "verl/verl/utils/reward_score/rllm_code_reward.py",
                "name": "compute_score",
                "reward_kwargs": {"timeout": 6},
            },
            "reward_model": {"reward_kwargs": {}},
        }
    )

    plan = code_reward_pool.build_code_reward_plan(data, _FakeTokenizer(), config)

    assert plan["mode"] == "code_reward_pool"
    assert len(plan["infos"]) == completions
    assert all(info.route == "official" for info in plan["infos"])
    assert len(fake_pool.jobs) == prompts * val_n * testcase_count
    assert fake_pool.jobs[0].owner_id == "livecodebench_v5:problem-0:0"
    assert fake_pool.jobs[0].fn is code_reward_pool.run_official_lcb_testcases
    assert fake_pool.jobs[0].kwargs["target_test_idx"] == 0
    assert fake_pool.jobs[0].kwargs["test_cases"] == [
        {"input": "0\n", "output": "0\n", "testtype": "functional", "metadata": {"func_name": "solve"}}
    ]
    assert fake_pool.jobs[-1].kwargs["target_test_idx"] == testcase_count - 1
    assert len(fake_pool.jobs[-1].kwargs["test_cases"]) == testcase_count
    assert fake_pool.jobs[-1].kwargs["test_cases"][-1] == {
        "input": "15\n",
        "output": "15\n",
        "testtype": "functional",
        "metadata": {"func_name": "solve"},
    }
    assert code_reward_pool.warmup_official_runner in fake_pool.warmups


def test_official_code_reward_does_not_require_rl_routing_fields(monkeypatch):
    public_tests = [{"input": "1\n", "output": "1\n"}]
    metadata = json.dumps(
        {
            "question_id": "formal-q1",
            "public_test_cases": json.dumps(public_tests),
            "private_test_cases": "[]",
            "metadata": json.dumps({"func_name": "solve"}),
        }
    )
    data = DataProto(
        batch=TensorDict(
            {
                "prompts": torch.ones((1, 3), dtype=torch.long),
                "responses": torch.ones((1, 4), dtype=torch.long),
                "attention_mask": torch.ones((1, 7), dtype=torch.long),
            },
            batch_size=[1],
        ),
        non_tensor_batch={
            "dataset": np.array(["livecodebench_v5"], dtype=object),
            "metadata": np.array([metadata], dtype=object),
            "extra_info": np.array(
                [
                    {
                        "dataset": "livecodebench_v5",
                        "sample_id": "formal-q1",
                    }
                ],
                dtype=object,
            ),
        },
    )
    config = OmegaConf.create(
        {
            "data": {"reward_fn_key": "data_source"},
            "custom_reward_function": {"reward_kwargs": {"enable_official_lcb_val": True}},
            "reward_model": {"reward_kwargs": {}},
        }
    )
    monkeypatch.setattr(code_reward_pool, "extract_official_code", lambda completion, lm_style: "print(input())")

    cfg = code_reward_pool.parse_code_reward_config(config)
    info, jobs, specs = code_reward_pool.build_row_jobs(data[0], 0, _FakeTokenizer(), cfg)

    assert info.route == "official"
    assert info.data_source == "livecodebench_v5"
    assert len(jobs) == 1
    assert specs == [(0, 1)]


def test_code_reward_pool_routes_training_shapes_to_rllm(monkeypatch):
    fake_pool = _FakePool()
    monkeypatch.setattr(code_reward_pool, "get_subprocess_worker_pool", lambda **_: fake_pool)

    taco_tests = {"inputs": ["1\n", "2\n"], "outputs": ["1\n", "2\n"]}
    prime_tests = [{"input": "3\n", "output": "3\n"}, {"input": "4\n", "output": "4\n"}]
    lcb_tests = [{"input": "5\n", "output": "5\n"}, {"input": "6\n", "output": "6\n"}]
    data_sources = ["taco", "primeintellect", "livecodebench"]
    ground_truths = [taco_tests, prime_tests, lcb_tests]
    data = DataProto(
        batch=TensorDict(
            {
                "prompts": torch.ones((3, 3), dtype=torch.long),
                "responses": torch.ones((3, 4), dtype=torch.long),
                "attention_mask": torch.ones((3, 7), dtype=torch.long),
            },
            batch_size=[3],
        ),
        non_tensor_batch={
            "reward_model": np.array(
                [{"style": "rule", "ground_truth": json.dumps(value)} for value in ground_truths],
                dtype=object,
            ),
            "data_source": np.array(data_sources, dtype=object),
            "extra_info": np.array(
                [{"split": "train", "index": str(idx), "source_config": source} for idx, source in enumerate(data_sources)],
                dtype=object,
            ),
        },
    )
    config = OmegaConf.create(
        {
            "data": {"reward_fn_key": "data_source"},
            "custom_reward_function": {
                "path": "verl/verl/utils/reward_score/rllm_code_reward.py",
                "name": "compute_score",
                "reward_kwargs": {"timeout": 6},
            },
            "reward_model": {"reward_kwargs": {}},
        }
    )

    plan = code_reward_pool.build_code_reward_plan(data, _FakeTokenizer(), config)

    assert len(plan["infos"]) == 3
    assert all(info.route == "train" for info in plan["infos"])
    assert [info.data_source for info in plan["infos"]] == data_sources
    assert len(fake_pool.jobs) == 6
    assert fake_pool.jobs[0].fn is code_reward_pool.run_rllm_lcb_testcases
    assert fake_pool.jobs[0].kwargs["test_cases"] == [{"input": "1\n", "output": "1\n"}]
    assert fake_pool.jobs[-1].kwargs["test_cases"] == [{"input": "6\n", "output": "6\n"}]
    assert code_reward_pool.warmup_rllm_runner in fake_pool.warmups


# ---------------------------------------------------------------------------
# Per-sample (inline / async-rollout) code reward path.
#
# The inline path (CodeRewardLoopManager.run_single) reuses the SAME shared
# subprocess sandbox pool as the batched driver path, and must produce
# byte-identical scores. These tests pin that parity plus the supporting
# plumbing (shared named/detached pool actors, per-row submit routing).
# ---------------------------------------------------------------------------


def _fake_results_for_jobs(jobs):
    """Deterministic JobResult for each job: every testcase passes except '2\\n'.

    Keyed only by job content, so the batched build (one big job list) and the
    per-row build (rebuilt per completion) yield the same results map.
    """
    results = []
    for job in jobs:
        test_cases = job.kwargs["test_cases"]
        value = [{"passed": tc.get("input") != "2\n", "error_code": 0} for tc in test_cases]
        results.append(JobResult(job.owner_id, job.job_idx, ok=True, value=value))
    return results


def _train_code_data():
    taco_tests = {"inputs": ["1\n", "2\n"], "outputs": ["1\n", "2\n"]}  # has the failing "2\n"
    prime_tests = [{"input": "3\n", "output": "3\n"}, {"input": "4\n", "output": "4\n"}]  # all pass
    lcb_tests = [{"input": "5\n", "output": "5\n"}, {"input": "6\n", "output": "6\n"}]  # all pass
    data_sources = ["taco", "primeintellect", "livecodebench"]
    ground_truths = [taco_tests, prime_tests, lcb_tests]
    data = DataProto(
        batch=TensorDict(
            {
                "prompts": torch.ones((3, 3), dtype=torch.long),
                "responses": torch.ones((3, 4), dtype=torch.long),
                "attention_mask": torch.ones((3, 7), dtype=torch.long),
            },
            batch_size=[3],
        ),
        non_tensor_batch={
            "reward_model": np.array(
                [{"style": "rule", "ground_truth": json.dumps(value)} for value in ground_truths],
                dtype=object,
            ),
            "data_source": np.array(data_sources, dtype=object),
            "extra_info": np.array(
                [{"split": "train", "index": str(idx)} for idx in range(len(data_sources))],
                dtype=object,
            ),
        },
    )
    config = OmegaConf.create(
        {
            "data": {"reward_fn_key": "data_source"},
            "custom_reward_function": {
                "path": "verl/verl/utils/reward_score/rllm_code_reward.py",
                "name": "compute_score",
                "reward_kwargs": {"timeout": 6},
            },
            "reward_model": {"reward_kwargs": {}},
        }
    )
    return data, config


def test_per_row_helpers_match_batched_plan(monkeypatch):
    """aggregate_row_results (inline) == gather_code_reward_plan (batched), row for row."""
    data, config = _train_code_data()
    tokenizer = _FakeTokenizer()

    # --- batched path ---
    fake_pool = _FakePool()
    monkeypatch.setattr(code_reward_pool, "get_subprocess_worker_pool", lambda **_: fake_pool)
    plan = code_reward_pool.build_code_reward_plan(data, tokenizer, config)
    batched_results = _fake_results_for_jobs(fake_pool.jobs)
    monkeypatch.setattr(code_reward_pool, "_wait_with_progress", lambda futures, label: batched_results)
    reward_tensor, extras = code_reward_pool.gather_code_reward_plan(plan)

    # --- per-row path: rebuild + aggregate each completion independently ---
    cfg = code_reward_pool.parse_code_reward_config(config)
    all_jobs = []
    rows = []
    for idx in range(len(data)):
        info, jobs, specs = code_reward_pool.build_row_jobs(data[idx], idx, tokenizer, cfg)
        all_jobs.extend(jobs)
        rows.append((info, specs))
    results_by_key = {(jr.owner_id, jr.job_idx): jr for jr in _fake_results_for_jobs(all_jobs)}

    for idx, (info, specs) in enumerate(rows):
        row = code_reward_pool.aggregate_row_results(info, specs, results_by_key)
        # scalar reward lands at the same tensor slot (float32 storage -> approx)
        assert row["final_score"] == pytest.approx(
            reward_tensor[info.index, info.valid_response_length - 1].item()
        )
        # every reward_extra_info column matches the batched gather, key for key
        for key in code_reward_pool.EXTRA_KEYS:
            if isinstance(row[key], float) and np.isnan(row[key]):
                assert np.isnan(extras[key][idx]), f"mismatch on {key} row {idx}"
            else:
                assert row[key] == extras[key][idx], f"mismatch on {key} row {idx}"

    # sanity: the failing "2\n" makes taco partial (0), the other two all-pass (0.8)
    assert extras["score"] == [0.0, 0.8, 0.8]
    assert extras["base_score"] == [0.0, 1.0, 1.0]


def test_submit_row_jobs_routes_warmup_by_route(monkeypatch):
    fake_pool = _FakePool()
    monkeypatch.setattr(code_reward_pool, "get_subprocess_worker_pool", lambda **_: fake_pool)
    cfg = code_reward_pool.parse_code_reward_config(
        OmegaConf.create(
            {
                "data": {"reward_fn_key": "data_source"},
                "custom_reward_function": {"reward_kwargs": {}},
                "reward_model": {"reward_kwargs": {}},
            }
        )
    )
    jobs = [Job(fn=code_reward_pool.run_rllm_lcb_testcases, kwargs={"test_cases": []}, owner_id="o", job_idx=0)]

    futures = code_reward_pool.submit_row_jobs(jobs, "train", cfg)
    assert fake_pool.warmups == [code_reward_pool.warmup_rllm_runner]
    assert len(futures) == 1

    code_reward_pool.submit_row_jobs(jobs, "official", cfg)
    assert fake_pool.warmups == [code_reward_pool.warmup_official_runner]


# Ray validates that node ids are real 28-byte hex strings.
_FAKE_NODE_ID = "ab" * 28


def _patch_ray_one_node(monkeypatch, sp):
    monkeypatch.setattr(
        sp.ray, "nodes", lambda: [{"Alive": True, "NodeID": _FAKE_NODE_ID, "Resources": {"CPU": 4}}]
    )
    monkeypatch.setattr(sp.ray, "cluster_resources", lambda: {"CPU": 4})
    monkeypatch.setattr(sp.ray, "available_resources", lambda: {"CPU": 4})

    captured = []

    class _FakeHandle:
        def remote(self):
            return object()

    def fake_options(**kwargs):
        captured.append(kwargs)
        return _FakeHandle()

    monkeypatch.setattr(sp.SubprocessWorker, "options", staticmethod(fake_options))
    return captured


def test_shared_pool_creates_named_detached_actors(monkeypatch):
    import verl.trainer.ppo.subprocess_pool as sp

    captured = _patch_ray_one_node(monkeypatch, sp)

    pool = sp.SubprocessWorkerPool(workers_per_node=2, worker_num_cpus=0.5, shared=True)
    pool.start()

    assert len(captured) == 2
    assert sorted(c["name"] for c in captured) == [
        f"verl_subproc_worker_{_FAKE_NODE_ID}_0",
        f"verl_subproc_worker_{_FAKE_NODE_ID}_1",
    ]
    for c in captured:
        assert c["namespace"] == sp.POOL_NAMESPACE
        assert c["lifetime"] == "detached"
        assert c["get_if_exists"] is True
        assert c["num_cpus"] == 0.5


def test_shared_pool_start_is_thread_safe(monkeypatch):
    import time
    from concurrent.futures import ThreadPoolExecutor

    import verl.trainer.ppo.subprocess_pool as sp

    captured = _patch_ray_one_node(monkeypatch, sp)
    fake_options = sp.SubprocessWorker.options

    def slow_options(**kwargs):
        time.sleep(0.02)
        return fake_options(**kwargs)

    monkeypatch.setattr(sp.SubprocessWorker, "options", staticmethod(slow_options))
    pool = sp.SubprocessWorkerPool(workers_per_node=2, worker_num_cpus=0.5, shared=True)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: pool.start(), range(8)))

    assert len(captured) == 2
    assert len(pool._actors) == 2


def test_shared_pool_warmup_is_thread_safe(monkeypatch):
    import time
    from concurrent.futures import ThreadPoolExecutor

    import verl.trainer.ppo.subprocess_pool as sp

    calls = []

    class _FakeWarmup:
        def remote(self, pending):
            calls.append(pending)
            return object()

    class _FakeActor:
        warmup = _FakeWarmup()

    def slow_wait(refs, *, label):
        time.sleep(0.02)
        return [{"host": "test"} for _ in refs]

    monkeypatch.setattr(sp, "_wait_with_progress", slow_wait)
    pool = sp.SubprocessWorkerPool(workers_per_node=1, shared=True)
    pool._actors = [_FakeActor()]

    def warmup_fn():
        return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: pool.warmup([warmup_fn]), range(8)))

    assert calls == [[warmup_fn]]


def test_unshared_pool_does_not_name_actors(monkeypatch):
    import verl.trainer.ppo.subprocess_pool as sp

    captured = _patch_ray_one_node(monkeypatch, sp)

    pool = sp.SubprocessWorkerPool(workers_per_node=1, worker_num_cpus=1.0, shared=False)
    pool.start()

    assert len(captured) == 1
    assert "name" not in captured[0]
    assert "lifetime" not in captured[0]


def _import_code_reward_loop():
    """Import the inline code reward loop without triggering the heavy
    ``verl.experimental.reward`` package init (which pulls the full training
    stack via reward_model). Stub the parent package, keeping its __path__ so
    the real ``reward_loop`` subpackage still resolves."""
    import types

    import verl.experimental  # real, light parent package must load first

    name = "verl.experimental.reward"
    if not getattr(sys.modules.get(name), "_test_stub", False):
        pkg_path = Path(__file__).resolve().parents[1] / "verl" / "experimental" / "reward"
        stub = types.ModuleType(name)
        stub.__path__ = [str(pkg_path)]
        stub._test_stub = True
        sys.modules[name] = stub

    import verl.experimental.reward.reward_loop.code as code_loop_mod

    return code_loop_mod


def test_code_reward_loop_run_single_uses_pool_and_matches_aggregate(monkeypatch):
    import asyncio

    code_loop_mod = _import_code_reward_loop()
    data, config = _train_code_data()
    tokenizer = _FakeTokenizer()

    # single completion: the all-pass primeintellect row
    single = data[1:2]

    fake_pool = _FakePool()
    monkeypatch.setattr(code_reward_pool, "get_subprocess_worker_pool", lambda **_: fake_pool)

    async def fake_gather(futures):
        # results are produced from the jobs the loop just submitted
        return _fake_results_for_jobs(fake_pool.jobs)

    monkeypatch.setattr(code_loop_mod, "_gather_futures", fake_gather)

    async def run():
        manager = code_loop_mod.CodeRewardLoopManager(config, tokenizer)
        return await manager.run_single(single)

    out = asyncio.run(run())

    assert set(out["reward_extra_info"]) == set(code_reward_pool.EXTRA_KEYS)
    assert out["reward_score"] == out["reward_extra_info"]["final_score"]
    # primeintellect row, all tests pass, no <think> block -> 0.8 * 1.0
    assert out["reward_score"] == 0.8
    assert out["reward_extra_info"]["base_score"] == 1.0
    assert out["reward_extra_info"]["route"] == "train"
    assert code_reward_pool.warmup_rllm_runner in fake_pool.warmups


def test_opd_validation_code_loop_uses_pool_and_emits_uniform_schema(monkeypatch):
    import asyncio

    _import_code_reward_loop()
    import verl.experimental.reward.reward_loop.opd_validation as opd_validation_mod

    data, config = _train_code_data()
    tokenizer = _FakeTokenizer()
    fake_pool = _FakePool()
    monkeypatch.setattr(code_reward_pool, "get_subprocess_worker_pool", lambda **_: fake_pool)

    async def fake_gather(futures):  # noqa: ARG001
        return _fake_results_for_jobs(fake_pool.jobs)

    monkeypatch.setattr(opd_validation_mod, "_gather_futures", fake_gather)

    async def run():
        manager = opd_validation_mod.OPDValidationRewardLoopManager(config, tokenizer)
        return await manager.run_single(data[1:2])

    out = asyncio.run(run())

    assert out == {"reward_score": 0.8, "reward_extra_info": {"score": 0.8, "acc": 1.0}}


def test_opd_validation_if_loop_normalizes_domain_specific_extras():
    import asyncio

    _import_code_reward_loop()
    import verl.experimental.reward.reward_loop.opd_validation as opd_validation_mod

    data = DataProto(
        batch=TensorDict(
            {
                "prompts": torch.ones((1, 2), dtype=torch.long),
                "responses": torch.ones((1, 3), dtype=torch.long),
                "attention_mask": torch.ones((1, 5), dtype=torch.long),
            },
            batch_size=[1],
        ),
        non_tensor_batch={
            "data_source": np.array(["ifeval"], dtype=object),
            "extra_info": np.array([{}], dtype=object),
            "dataset": np.array(["ifeval"], dtype=object),
            "sample_id": np.array(["17"], dtype=object),
            "original_prompt": np.array(["Answer in exactly three words."], dtype=object),
            "answer": np.array([None], dtype=object),
            "metadata": np.array(['{"instruction_id_list": ["length:three_words"]}'], dtype=object),
            "evaluator": np.array(["ifeval"], dtype=object),
            "request_id": np.array(["ifeval:17:0"], dtype=object),
        },
    )
    config = OmegaConf.create(
        {
            "data": {"reward_fn_key": "data_source"},
            "custom_reward_function": {"reward_kwargs": {}},
            "reward_model": {"reward_kwargs": {}},
        }
    )

    captured = {}

    def fake_score(**kwargs):
        captured.update(kwargs)
        return {"score": 0.6, "acc": 0.5, "math_only_detail": 7}

    async def run():
        manager = opd_validation_mod.OPDValidationRewardLoopManager(config, _FakeTokenizer(), fake_score)
        return await manager.run_single(data)

    out = asyncio.run(run())

    assert out == {"reward_score": 0.6, "reward_extra_info": {"score": 0.6, "acc": 0.5}}
    assert captured["ground_truth"] is None
    assert captured["extra_info"]["dataset"] == "ifeval"
    assert captured["extra_info"]["evaluator"] == "ifeval"
    assert captured["extra_info"]["metadata"] == '{"instruction_id_list": ["length:three_words"]}'
    assert captured["extra_info"]["eval_prompt_for_scorer"] == "Answer in exactly three words."


def test_opd_validation_math_score_uses_spawn_process_from_non_main_thread():
    """Ray async actors run off-main-thread; Math scoring must still own SIGALRM."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    _import_code_reward_loop()
    import verl.experimental.reward.reward_loop.opd_validation as opd_validation_mod

    data = DataProto(
        batch=TensorDict(
            {
                "prompts": torch.ones((1, 2), dtype=torch.long),
                "responses": torch.ones((1, 3), dtype=torch.long),
                "attention_mask": torch.ones((1, 5), dtype=torch.long),
            },
            batch_size=[1],
        ),
        non_tensor_batch={
            "data_source": np.array(["aime24"], dtype=object),
            "reward_model": np.array([{"ground_truth": "42"}], dtype=object),
            "extra_info": np.array([{}], dtype=object),
        },
    )
    config = OmegaConf.create(
        {
            "data": {"reward_fn_key": "data_source"},
            "custom_reward_function": {"reward_kwargs": {}},
            "reward_model": {"reward_kwargs": {}},
        }
    )
    class MathTokenizer:
        def decode(self, *args, **kwargs):  # noqa: ARG002
            return r"\boxed{42}"

    async def run():
        manager = opd_validation_mod.OPDValidationRewardLoopManager(config, MathTokenizer())
        try:
            return await manager.run_single(data)
        finally:
            manager.math_score_pool.shutdown(wait=True)

    # This reproduces Ray's relevant constraint: the coroutine's event loop is
    # hosted by a non-main thread, while the spawned scorer process has its own
    # main thread and can safely install ttrl_math's SIGALRM handlers.
    with ThreadPoolExecutor(max_workers=1) as executor:
        out = executor.submit(asyncio.run, run()).result(timeout=30)

    # ttrl_math awards 0.8 answer credit plus no explicit think-tag credit for
    # this minimal completion; acc=1 confirms the boxed answer was accepted.
    assert out == {"reward_score": 0.8, "reward_extra_info": {"score": 0.8, "acc": 1.0}}


def test_collect_inline_reward_extra_infos_rebuilds_dict():
    from verl.trainer.ppo.reward import collect_inline_reward_extra_infos

    data = DataProto(
        batch=TensorDict({"rm_scores": torch.zeros((2, 1))}, batch_size=[2]),
        non_tensor_batch={
            "acc": np.array([0.0, 1.0]),
            "score": np.array([0.0, 0.8]),
            "route": np.array(["train", "train"], dtype=object),
            "data_source": np.array(["taco", "taco"], dtype=object),  # present but not a reward key
        },
        meta_info={"reward_extra_keys": ["acc", "score", "route", "missing_key"]},
    )

    out = collect_inline_reward_extra_infos(data)

    # only declared reward keys that exist are recovered; missing keys are skipped
    assert set(out) == {"acc", "score", "route"}
    assert out["acc"] == [0.0, 1.0]
    assert out["score"] == [0.0, 0.8]
    assert out["route"] == ["train", "train"]


def test_collect_inline_reward_extra_infos_empty_without_keys():
    from verl.trainer.ppo.reward import collect_inline_reward_extra_infos

    data = DataProto(
        batch=TensorDict({"rm_scores": torch.zeros((1, 1))}, batch_size=[1]),
        non_tensor_batch={"acc": np.array([1.0])},
        meta_info={},
    )

    assert collect_inline_reward_extra_infos(data) == {}
