from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .common import score_ifeval_like


def score_ifbench(df: pd.DataFrame, details_dir: Path | None) -> dict[str, Any]:
    return score_ifeval_like(df, ifbench=True, details_dir=details_dir)

