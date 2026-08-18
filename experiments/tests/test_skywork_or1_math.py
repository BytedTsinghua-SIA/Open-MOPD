from __future__ import annotations

import json

import pandas as pd

from experiments.data.skywork_or1_math import BOXED_SUFFIX, BuildConfig, govern_rows


def _row(question: str, answer: str, difficulty: int, source_row: int) -> dict:
    return {
        "data_source": "skywork-source",
        "prompt": [{"role": "user", "content": question}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": json.dumps([answer])},
        "extra_info": {
            "index": source_row,
            "model_difficulty": {"DeepSeek-R1-Distill-Qwen-7B": difficulty},
        },
    }


def test_govern_rows_filters_formats_deduplicates_and_samples() -> None:
    raw = pd.DataFrame(
        [
            _row("keep one", "1", 1, 0),
            _row(" keep   one ", "1", 2, 1),
            _row("keep two", "2", 15, 2),
            _row("too easy", "3", 0, 3),
            _row("too hard", "4", 16, 4),
            _row("too long", "5", 5, 5),
        ]
    )

    def token_counter(messages: list[dict[str, str]]) -> int:
        return 2048 if "too long" in messages[-1]["content"] else 100

    governed, stats = govern_rows(
        raw,
        config=BuildConfig(
            source_revision="rev",
            source_sha256="sha",
            target_rows=2,
            max_prompt_tokens=1024,
        ),
        token_counter=token_counter,
    )

    assert len(governed) == 2
    assert stats["rejected_difficulty"] == 2
    assert stats["rejected_prompt_too_long"] == 1
    assert stats["deduplicated"] == 1
    assert set(governed.data_source) == {"math_dapo_boxed"}
    assert all(row[-1]["content"].endswith(BOXED_SUFFIX) for row in governed.prompt)
    assert all(row[0]["role"] == "system" for row in governed.prompt)
    assert list(governed.extra_info.map(lambda value: value["index"])) == [0, 1]
