from __future__ import annotations

import pandas as pd

from evals.score_rollouts import _math_rows


def test_math_rows_accepts_zero_padded_aime_reference() -> None:
    rows = pd.DataFrame(
        {
            "completion": [r"Reasoning... \boxed{33}", r"Reasoning... \boxed{25}."],
            "answer": ["033", "025"],
            "reward_model": [
                {"ground_truth": "033", "style": "rule-lighteval/MATH_v2"},
                {"ground_truth": "025", "style": "rule-lighteval/MATH_v2"},
            ],
        }
    )

    assert _math_rows(rows) == [1.0, 1.0]


def test_math_rows_rejects_wrong_aime_integer() -> None:
    rows = pd.DataFrame(
        {
            "completion": [r"Reasoning... \boxed{34}"],
            "answer": ["033"],
            "reward_model": [
                {"ground_truth": "033", "style": "rule-lighteval/MATH_v2"},
            ],
        }
    )

    assert _math_rows(rows) == [0.0]
