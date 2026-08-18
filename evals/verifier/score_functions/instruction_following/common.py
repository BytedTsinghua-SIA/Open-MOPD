from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from evals.verifier.score_functions.common.utils import parse_metadata, repo_path


def normalize_kwargs(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = ast.literal_eval(raw)
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    normalized = []
    for item in raw:
        current = {}
        for key, value in item.items():
            if value is None:
                continue
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            current[str(key)] = value
        normalized.append(current)
    return normalized


def filter_kwargs_for_instructions(
    kwargs: list[dict[str, Any]],
    instruction_ids: list[str],
    instruction_registry: Any,
) -> list[dict[str, Any]]:
    filtered = []
    registry = getattr(instruction_registry, "INSTRUCTION_DICT", {})
    for index, instruction_id in enumerate(instruction_ids):
        current = kwargs[index] if index < len(kwargs) else {}
        instruction_cls = registry.get(instruction_id)
        if instruction_cls is None:
            filtered.append(current)
            continue
        try:
            allowed = set(instruction_cls(instruction_id).get_instruction_args_keys() or [])
        except Exception:
            filtered.append(current)
            continue
        filtered.append({key: value for key, value in current.items() if key in allowed})
    return filtered


def patch_registry_missing(target_registry: Any, source_registry: Any) -> None:
    target = getattr(target_registry, "INSTRUCTION_DICT", None)
    source = getattr(source_registry, "INSTRUCTION_DICT", None)
    if not isinstance(target, dict) or not isinstance(source, dict):
        return
    for instruction_id, instruction_cls in source.items():
        target.setdefault(instruction_id, instruction_cls)


def summarize_instruction_outputs(outputs: list[Any]) -> dict[str, Any]:
    prompt_total = len(outputs)
    prompt_correct = sum(1 for output in outputs if output.follow_all_instructions)
    instruction_total = sum(len(output.follow_instruction_list) for output in outputs)
    instruction_correct = sum(sum(output.follow_instruction_list) for output in outputs)
    return {
        "prompt_level_accuracy": prompt_correct / prompt_total if prompt_total else 0.0,
        "instruction_level_accuracy": instruction_correct / instruction_total if instruction_total else 0.0,
        "prompt_total": prompt_total,
        "prompt_correct": prompt_correct,
        "instruction_total": instruction_total,
        "instruction_correct": instruction_correct,
    }


def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.DOTALL | re.IGNORECASE).strip()


def score_ifeval_like(df: pd.DataFrame, *, ifbench: bool, details_dir: Path | None) -> dict[str, Any]:
    from .google_ifeval import instructions_registry as google_instructions_registry

    if ifbench:
        ifbench_repo = repo_path("ifbench")
        nltk_data = ifbench_repo / ".nltk_data"
        if nltk_data.exists():
            import nltk  # type: ignore

            nltk.data.path.insert(0, str(nltk_data))
        if str(ifbench_repo) not in sys.path:
            sys.path.insert(0, str(ifbench_repo))
        import evaluation_lib  # type: ignore
        import instructions_registry  # type: ignore
        patch_registry_missing(instructions_registry, google_instructions_registry)
    else:
        from .google_ifeval import evaluation_lib
        from .google_ifeval import instructions_registry
        try:
            ifbench_repo = repo_path("ifbench")
        except Exception:
            ifbench_repo = None
        if ifbench_repo is not None:
            if str(ifbench_repo) not in sys.path:
                sys.path.insert(0, str(ifbench_repo))
            try:
                import instructions_registry as ifbench_instructions_registry  # type: ignore
            except Exception:
                pass
            else:
                patch_registry_missing(instructions_registry, ifbench_instructions_registry)

    inputs = []
    prompt_to_response = {}
    for index, row in df.iterrows():
        meta = parse_metadata(row)
        key = meta.get("key", row.get("sample_id", index))
        instruction_ids = meta.get("instruction_id_list", row.get("instruction_id_list", []))
        kwargs = normalize_kwargs(meta.get("kwargs", row.get("kwargs", [])))
        instruction_ids = [str(x) for x in instruction_ids]
        kwargs = filter_kwargs_for_instructions(kwargs, instruction_ids, instructions_registry)
        prompt = str(row["prompt"])
        inputs.append(
            evaluation_lib.InputExample(
                key=int(key) if str(key).isdigit() else key,
                instruction_id_list=instruction_ids,
                prompt=prompt,
                kwargs=kwargs,
            )
        )
        prompt_to_response[prompt] = strip_think_tags(str(row.get("completion", "")))
    strict_outputs = [evaluation_lib.test_instruction_following_strict(inp, prompt_to_response) for inp in inputs]
    loose_outputs = [evaluation_lib.test_instruction_following_loose(inp, prompt_to_response) for inp in inputs]
    if details_dir:
        details_dir.mkdir(parents=True, exist_ok=True)
        evaluation_lib.write_outputs(details_dir / "eval_results_strict.jsonl", strict_outputs)
        evaluation_lib.write_outputs(details_dir / "eval_results_loose.jsonl", loose_outputs)
    strict = summarize_instruction_outputs(strict_outputs)
    loose = summarize_instruction_outputs(loose_outputs)
    return {
        "correct": strict["prompt_correct"],
        "scored_rows": strict["prompt_total"],
        "metric": "strict_prompt_level_accuracy",
        "official_extra": {"strict": strict, "loose": loose},
        # per-row strict pass (1.0/0.0), aligned to df row order — lets callers write a
        # `score` column back into the rollout parquet for uniform total tallying.
        "per_row_strict": [1.0 if bool(o.follow_all_instructions) else 0.0 for o in strict_outputs],
    }
