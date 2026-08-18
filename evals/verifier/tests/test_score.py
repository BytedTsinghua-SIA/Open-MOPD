from __future__ import annotations

import json

import pandas as pd

from evals.verifier import score
from evals.verifier.score import compute_behavior_metrics, is_repetitive_completion, lcb_generation_passed


def test_is_repetitive_completion_detects_phrase_loop() -> None:
    text = "\n".join(["Let me think about this carefully."] * 20)
    assert is_repetitive_completion(text) is True


def test_compute_behavior_metrics_counts_trunc_and_repeat() -> None:
    df = pd.DataFrame(
        [
            {
                "completion": "short answer",
                "completion_tokens": 10,
                "max_tokens": 100,
            },
            {
                "completion": "\n".join(["wait, we compute the same thing again"] * 20),
                "completion_tokens": 100,
                "max_tokens": 100,
            },
        ]
    )

    metrics = compute_behavior_metrics(df)

    assert metrics["trunc_rows"] == 1
    assert metrics["repeat_rows"] == 1
    assert metrics["trunc_repeat_rows"] == 1
    assert metrics["trunc_rate"] == 0.5
    assert metrics["repeat_rate"] == 0.5
    assert metrics["trunc_repeat_rate"] == 0.5


def test_lcb_generation_passed_rejects_negative_error_codes() -> None:
    assert lcb_generation_passed([True, True]) is True
    assert lcb_generation_passed([1, 1]) is True
    assert lcb_generation_passed([1, -2]) is False
    assert lcb_generation_passed([-1]) is False
    assert lcb_generation_passed([False]) is False


def test_main_groups_same_dataset_across_rollout_files(tmp_path, monkeypatch, capsys) -> None:
    first = pd.DataFrame(
        [
            {"dataset": "aime24", "sample_id": 1, "answer": "1", "completion": r"\boxed{1}"},
            {"dataset": "aime24", "sample_id": 1, "answer": "1", "completion": r"\boxed{0}"},
        ]
    )
    second = pd.DataFrame(
        [
            {"dataset": "aime24", "sample_id": 2, "answer": "2", "completion": r"\boxed{2}"},
            {"dataset": "aime24", "sample_id": 2, "answer": "2", "completion": r"\boxed{0}"},
        ]
    )
    frames = {"rank0.parquet": first, "rank1.parquet": second}
    output = tmp_path / "scores.json"

    monkeypatch.setattr(score.pd, "read_parquet", lambda path: frames[path.name])
    monkeypatch.setattr(
        score.sys,
        "argv",
        ["score.py", "--rollout", "rank0.parquet", "rank1.parquet", "--output", str(output)],
    )

    score.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["results"]) == 1
    result = payload["results"][0]
    assert result["dataset"] == "aime24"
    assert result["correct"] == 2
    assert result["scored_rows"] == 4
    assert result["official_total"] == 4
    assert result["official"] == 0.5


def test_main_passes_default_lcb_parallelism_to_lcb_scorer(tmp_path, monkeypatch, capsys) -> None:
    df = pd.DataFrame(
        [
            {
                "dataset": "livecodebench_v6",
                "sample_id": 1,
                "completion": "print('ok')",
                "metadata": {},
            }
        ]
    )
    output = tmp_path / "scores.json"
    seen = []

    def fake_score_lcb(
        dataset_df,
        details_dir,
        workers,
        progress_interval=500,
        lm_style="CodeQwenInstruct",
        *,
        data_dir=None,
        subsets=None,
        process_workers=1,
        worker_concurrency=None,
        timeout=6,
    ):
        seen.append(
            {
                "workers": workers,
                "process_workers": process_workers,
                "worker_concurrency": worker_concurrency,
                "timeout": timeout,
            }
        )
        return {
            "correct": 0.0,
            "scored_rows": len(dataset_df),
            "metric": "livecodebench_pass_at_1",
            "official_extra": {
                "workers": workers,
                "process_workers": process_workers,
                "worker_concurrency": worker_concurrency,
                "lm_style": lm_style,
            },
        }

    monkeypatch.setattr(score.pd, "read_parquet", lambda path: df)
    monkeypatch.setattr(score, "score_lcb", fake_score_lcb)
    monkeypatch.setattr(
        score.sys,
        "argv",
        ["score.py", "--rollout", "lcb.parquet", "--output", str(output)],
    )

    score.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert seen == [{"workers": 160, "process_workers": 160, "worker_concurrency": 24, "timeout": 6}]
    assert payload["results"][0]["official_extra"]["workers"] == 160
    assert payload["results"][0]["official_extra"]["process_workers"] == 160
    assert payload["results"][0]["official_extra"]["worker_concurrency"] == 24
