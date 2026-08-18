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

"""Domain dispatch for the Math/Code/IF mixed RL recipe.

Code is deliberately not evaluated here.  The async ``mixed`` reward loop
routes code trajectories through :class:`CodeRewardLoopManager`, which submits
each completed trajectory to the shared subprocess testcase pool immediately.
Math and IF retain the scorer functions used by their selected single-domain
RL recipes; IF calls the batch/GRM entry point with a one-row batch so its
per-trajectory score is semantically identical while remaining inline.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def infer_domain(
    data_source: object,
    extra_info: Mapping[str, Any] | None = None,
    ability: object | None = None,
) -> str:
    """Resolve one of ``math``, ``code``, or ``if`` from a mixed RL row."""

    extra_info = extra_info or {}
    explicit = str(extra_info.get("domain", "") or "").strip().lower()
    ability_text = str(ability or extra_info.get("ability", "") or "").strip().lower()
    source = str(data_source or "").strip().lower()

    if explicit in {"math", "code", "if"}:
        return explicit
    if ability_text in {"math", "mathematics"}:
        return "math"
    if ability_text in {"code", "coding", "programming"}:
        return "code"
    if ability_text in {"if", "instruction_following", "instruction-following"}:
        return "if"
    if source.startswith("aime") or "math" in source:
        return "math"
    if source in {"primeintellect", "taco", "codecontests_lcb", "livecodebench"} or source.startswith("livecodebench"):
        return "code"
    if source == "nemotron_if_rl" or "ifeval" in source or source.startswith("ifbench"):
        return "if"
    raise ValueError(f"cannot infer mixed RL domain from data_source={source!r}, ability={ability_text!r}")


def reward_func(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **reward_kwargs: Any,
) -> dict[str, Any]:
    """Score a non-code mixed RL trajectory with its source recipe."""

    domain = infer_domain(data_source, extra_info)
    if domain == "math":
        from verl.utils.reward_score import ttrl_math

        return ttrl_math.reward_func(data_source, solution_str, ground_truth, extra_info)
    if domain == "if":
        from verl.utils.reward_score import instruction_following

        return instruction_following.compute_score_batch(
            data_sources=[data_source],
            solution_strs=[solution_str],
            ground_truths=[ground_truth],
            extra_infos=[extra_info or {}],
            **reward_kwargs,
        )[0]
    raise RuntimeError("code trajectories must be routed through the inline shared testcase pool")
