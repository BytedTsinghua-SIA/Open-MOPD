from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .common import score_ifeval_like


def score_ifeval(df: pd.DataFrame, details_dir: Path | None) -> dict[str, Any]:
    return score_ifeval_like(df, ifbench=False, details_dir=details_dir)

