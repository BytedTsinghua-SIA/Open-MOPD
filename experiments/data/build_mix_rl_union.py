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

"""Restore exact single-RL rows selected by the old 24K MT-OPD template.

The MT-OPD selection parquet retained prompts and ``domain-N`` indices but
discarded Math/Code ground truths.  For each domain, its indexed prompt stream
is an exact subsequence of the selected single-RL source.  A greedy subsequence
join therefore recovers the exact original occurrence even when duplicate
prompts carry different testcases/answers.  The output is interleaved
Math/Code/IF so every 192-row training batch contains 64 rows per domain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DOMAINS = ("math", "code", "if")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _prompt_key(data_source: Any, prompt: Any) -> str:
    payload = [str(data_source), _jsonable(prompt)]
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _template_rows_by_domain(template: pd.DataFrame) -> dict[str, list[pd.Series]]:
    rows: dict[str, dict[int, pd.Series]] = {domain: {} for domain in DOMAINS}
    for _, row in template.iterrows():
        extra = dict(row["extra_info"] or {})
        domain = str(row.get("domain") or extra.get("domain") or "").lower()
        index_text = str(extra.get("index", ""))
        prefix, separator, suffix = index_text.rpartition("-")
        if (
            domain not in rows
            or separator != "-"
            or prefix != domain
            or not suffix.isdigit()
        ):
            raise ValueError(
                f"invalid selection index {index_text!r} for domain {domain!r}"
            )
        rows[domain][int(suffix)] = row
    ordered: dict[str, list[pd.Series]] = {}
    for domain, indexed in rows.items():
        expected = list(range(len(indexed)))
        if sorted(indexed) != expected:
            raise ValueError(f"selection indices for {domain} are not contiguous")
        ordered[domain] = [indexed[index] for index in expected]
    counts = {domain: len(items) for domain, items in ordered.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"selection is not 1:1:1: {counts}")
    return ordered


def _restore_domain(
    domain: str, selected: list[pd.Series], source: pd.DataFrame
) -> list[dict[str, Any]]:
    target_keys = [_prompt_key(row["data_source"], row["prompt"]) for row in selected]
    restored: list[dict[str, Any]] = []
    target_index = 0
    for source_position, (_, row) in enumerate(source.iterrows()):
        if target_index >= len(target_keys):
            break
        if _prompt_key(row["data_source"], row["prompt"]) != target_keys[target_index]:
            continue
        record = {key: _jsonable(value) for key, value in row.to_dict().items()}
        extra = dict(record.get("extra_info") or {})
        # Arrow structs require one stable type across all domains; source
        # indices are UUIDs for Math and integers for Code, so persist them as
        # strings without changing their identity.
        extra["source_index"] = str(_jsonable(extra.get("index", source_position)))
        extra["domain"] = domain
        extra["index"] = f"{domain}-{target_index}"
        extra["mix_index"] = target_index
        record["extra_info"] = extra
        record["domain"] = domain
        restored.append(record)
        target_index += 1
    if target_index != len(target_keys):
        raise ValueError(
            f"only restored {target_index}/{len(target_keys)} {domain} rows"
        )
    return restored


def build_mix_frame(
    template: pd.DataFrame,
    math_source: pd.DataFrame,
    code_source: pd.DataFrame,
    if_source: pd.DataFrame,
) -> pd.DataFrame:
    """Restore and strictly interleave the three selected source streams."""

    selected = _template_rows_by_domain(template)
    sources = {"math": math_source, "code": code_source, "if": if_source}
    restored = {
        domain: _restore_domain(domain, selected[domain], sources[domain])
        for domain in DOMAINS
    }
    records = [
        restored[domain][index]
        for index in range(len(restored["math"]))
        for domain in DOMAINS
    ]
    output = pd.DataFrame.from_records(records)
    for start in range(0, len(output), 192):
        counts = output.iloc[start : start + 192]["domain"].value_counts().to_dict()
        if len(output.iloc[start : start + 192]) == 192 and counts != {
            "math": 64,
            "code": 64,
            "if": 64,
        }:
            raise ValueError(f"batch {start // 192} violates 1:1:1: {counts}")
    for domain in ("math", "code"):
        for reward_model in output.loc[output["domain"] == domain, "reward_model"]:
            if not str(dict(reward_model or {}).get("ground_truth", "")):
                raise ValueError(f"restored {domain} row has empty ground truth")
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-template", type=Path, required=True)
    parser.add_argument("--math", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--if-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "selection_template": args.selection_template,
        "math": args.math,
        "code": args.code,
        "if": args.if_source,
    }
    frames = {name: pd.read_parquet(path) for name, path in paths.items()}
    output = build_mix_frame(
        frames["selection_template"], frames["math"], frames["code"], frames["if"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    manifest = {
        "contract": "exact selected single-RL rows; interleaved math/code/if; every full 192-row batch is 64/64/64",
        "rows": len(output),
        "domain_counts": output["domain"].value_counts().sort_index().to_dict(),
        "sources": {
            name: {
                "path": str(path),
                "rows": len(frames[name]),
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
        "output": {"path": str(args.output), "sha256": _sha256(args.output)},
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
