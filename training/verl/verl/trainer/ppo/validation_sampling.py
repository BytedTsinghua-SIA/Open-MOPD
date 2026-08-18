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

"""Per-data-source validation sampling for heterogeneous eval suites.

A single MT-OPD validation dataloader contains Math, Code, and IF rows, while
their source RL recipes intentionally use different sampling protocols.  A
single global ``val_kwargs.n`` cannot represent AIME@64, LCB@10, and IF@1 at
the same time.  This module builds the repeat plan and the per-request sampling
payload without coupling it to Ray or a particular rollout backend.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


REQUEST_SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "repetition_penalty",
    "stop_token_ids",
)


def _to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "items"):
        return dict(value.items())
    return {
        key: getattr(value, key)
        for key in ("n", "do_sample", *REQUEST_SAMPLING_FIELDS)
        if hasattr(value, key)
    }


def resolve_validation_profile(
    data_source: object,
    default_profile: Any,
    profiles_by_data_source: Any,
) -> dict[str, Any]:
    """Merge the global validation profile with an exact source override."""

    source = str(data_source or "")
    default = _to_mapping(default_profile)
    overrides = _to_mapping(profiles_by_data_source)
    source_override = _to_mapping(overrides.get(source, overrides.get("*")))
    profile = {**default, **source_override}

    repeat_count = int(profile.get("n", 1))
    if repeat_count <= 0:
        raise ValueError(f"validation repeat count must be positive for {source!r}, got {repeat_count}")
    profile["n"] = repeat_count

    if not bool(profile.get("do_sample", True)):
        profile["temperature"] = 0.0
    return profile


def build_validation_repeat_plan(
    data_sources: Sequence[object],
    default_profile: Any,
    profiles_by_data_source: Any,
) -> tuple[list[int], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return source-row indices, per-request params, and resolved profiles.

    ``indices`` can be passed directly to :meth:`DataProto.select_idxs`.
    ``request_profiles`` deliberately excludes ``n`` because repetition has
    already been materialized in the index list.
    """

    indices: list[int] = []
    request_profiles: list[dict[str, Any]] = []
    resolved_by_source: dict[str, dict[str, Any]] = {}

    for row_index, raw_source in enumerate(data_sources):
        source = str(raw_source or "")
        profile = resolve_validation_profile(source, default_profile, profiles_by_data_source)
        resolved_by_source[source] = profile
        request_profile = {
            key: profile[key]
            for key in REQUEST_SAMPLING_FIELDS
            if profile.get(key) is not None
        }
        # Hydra represents list-valued overrides as ``ListConfig``.  vLLM's
        # SamplingParams intentionally requires an actual Python list for
        # stop_token_ids, so normalize it at the request boundary instead of
        # leaking an OmegaConf container into the rollout worker.
        if "stop_token_ids" in request_profile:
            request_profile["stop_token_ids"] = list(request_profile["stop_token_ids"])
        repeat_count = int(profile["n"])
        indices.extend([row_index] * repeat_count)
        request_profiles.extend([dict(request_profile) for _ in range(repeat_count)])

    return indices, request_profiles, resolved_by_source
