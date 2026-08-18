"""Govern Skywork-OR1 math prompts into OpenOPD's SP + boxed-answer schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd


SYSTEM_PROMPT = (
    "You are an expert mathematical problem solver. Solve the problem carefully. "
    "You must first show all your step-by-step reasoning and derivations inside "
    "<think> and </think> tags. After your thought process, provide the final answer "
    "clearly and make it easy to verify."
)
BOXED_SUFFIX = "Please reason step by step, and put your final answer in \\boxed{}."
DIFFICULTY_MODEL = "DeepSeek-R1-Distill-Qwen-7B"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "as_py"):
        converted = value.as_py()
        return dict(converted) if isinstance(converted, dict) else {}
    try:
        return dict(value)
    except (TypeError, ValueError):
        return {}


def _as_messages(value: Any) -> list[dict[str, str]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [
        {"role": str(_as_dict(message).get("role", "")), "content": str(_as_dict(message).get("content", ""))}
        for message in list(value or [])
    ]


def _parse_single_answer(reward_model: Any) -> str | None:
    raw = _as_dict(reward_model).get("ground_truth")
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
    else:
        parsed = raw
    values = parsed if isinstance(parsed, list) else [parsed]
    answers = [str(value).strip() for value in values if str(value).strip()]
    return answers[0] if len(answers) == 1 else None


def _question(messages: list[dict[str, str]]) -> str | None:
    if len(messages) != 1 or messages[0].get("role") != "user":
        return None
    question = messages[0].get("content", "").strip()
    return question or None


def _normalized_question(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip()


def _formatted_prompt(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{question}\n\n{BOXED_SUFFIX}"},
    ]


@dataclass(frozen=True)
class BuildConfig:
    source_revision: str
    source_sha256: str
    seed: int = 42
    target_rows: int = 11635
    min_difficulty: int = 1
    max_difficulty: int = 15
    max_prompt_tokens: int = 1024


def govern_rows(
    raw: pd.DataFrame,
    *,
    config: BuildConfig,
    token_counter: Callable[[list[dict[str, str]]], int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    counters = {
        "raw_rows": len(raw),
        "rejected_prompt_schema": 0,
        "rejected_difficulty": 0,
        "rejected_answer": 0,
        "rejected_prompt_too_long": 0,
        "deduplicated": 0,
    }
    candidates: dict[str, dict[str, Any]] = {}
    for source_row, row in raw.iterrows():
        question = _question(_as_messages(row.get("prompt")))
        if question is None:
            counters["rejected_prompt_schema"] += 1
            continue
        extra_info = _as_dict(row.get("extra_info"))
        difficulty_raw = _as_dict(extra_info.get("model_difficulty")).get(DIFFICULTY_MODEL)
        try:
            difficulty = int(difficulty_raw)
        except (TypeError, ValueError):
            counters["rejected_difficulty"] += 1
            continue
        if not config.min_difficulty <= difficulty <= config.max_difficulty:
            counters["rejected_difficulty"] += 1
            continue
        answer = _parse_single_answer(row.get("reward_model"))
        if answer is None:
            counters["rejected_answer"] += 1
            continue
        prompt = _formatted_prompt(question)
        prompt_tokens = token_counter(prompt)
        if prompt_tokens > config.max_prompt_tokens:
            counters["rejected_prompt_too_long"] += 1
            continue
        normalized = _normalized_question(question)
        prompt_hash = _sha256_text(normalized)
        record = {
            "data_source": "math_dapo_boxed",
            "prompt": prompt,
            "ability": "MATH",
            "reward_model": {"ground_truth": answer, "style": "rule"},
            "extra_info": {
                "index": 0,
                "source_dataset": "Skywork/Skywork-OR1-RL-Data",
                "source_revision": config.source_revision,
                "source_split": "math",
                "source_row": int(source_row),
                "source_data_source": str(row.get("data_source", "")),
                "difficulty_model": DIFFICULTY_MODEL,
                "difficulty": difficulty,
                "prompt_tokens": prompt_tokens,
                "prompt_sha256": prompt_hash,
            },
            "_sample_key": _sha256_text(f"{config.seed}:{prompt_hash}"),
        }
        previous = candidates.get(prompt_hash)
        if previous is None or record["_sample_key"] < previous["_sample_key"]:
            if previous is not None:
                counters["deduplicated"] += 1
            candidates[prompt_hash] = record
        else:
            counters["deduplicated"] += 1

    eligible = sorted(candidates.values(), key=lambda record: record["_sample_key"])
    if len(eligible) < config.target_rows:
        raise ValueError(
            f"only {len(eligible)} eligible unique rows, fewer than target {config.target_rows}"
        )
    selected = eligible[: config.target_rows]
    for index, record in enumerate(selected):
        record["extra_info"]["index"] = index
        record.pop("_sample_key")
    result = pd.DataFrame(selected)
    stats = {
        **counters,
        "eligible_unique_rows": len(eligible),
        "selected_rows": len(result),
        "difficulty_distribution": {
            str(key): int(value)
            for key, value in result.extra_info.map(lambda value: value["difficulty"])
            .value_counts()
            .sort_index()
            .items()
        },
        "source_distribution": {
            str(key): int(value)
            for key, value in result.extra_info.map(lambda value: value["source_data_source"])
            .value_counts()
            .sort_index()
            .items()
        },
        "prompt_token_max": int(
            result.extra_info.map(lambda value: value["prompt_tokens"]).max()
        ),
    }
    return result, stats


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-sha256")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-rows", type=int, default=11635)
    parser.add_argument("--min-difficulty", type=int, default=1)
    parser.add_argument("--max-difficulty", type=int, default=15)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output directory exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    def token_counter(messages: list[dict[str, str]]) -> int:
        return len(
            tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        )

    source_sha256 = args.source_sha256 or _sha256_file(args.input)
    config = BuildConfig(
        source_revision=args.source_revision,
        source_sha256=source_sha256,
        seed=args.seed,
        target_rows=args.target_rows,
        min_difficulty=args.min_difficulty,
        max_difficulty=args.max_difficulty,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    raw = pd.read_parquet(args.input)
    governed, stats = govern_rows(raw, config=config, token_counter=token_counter)
    output_path = args.output_dir / "train.parquet"
    governed.to_parquet(output_path, index=False)
    output_sha256 = _sha256_file(output_path)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": "Skywork/Skywork-OR1-RL-Data",
            "split": "math",
            "revision": config.source_revision,
            "parquet_sha256": config.source_sha256,
            "url": (
                "https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data/"
                f"resolve/{config.source_revision}/data/math-00000-of-00001.parquet"
            ),
            "license_note": "The dataset card does not declare an explicit license field.",
        },
        "governance": {
            "difficulty_model": DIFFICULTY_MODEL,
            "difficulty_range_inclusive": [config.min_difficulty, config.max_difficulty],
            "deduplication": "normalized exact question SHA-256",
            "sampling": "ascending SHA-256(seed:prompt_sha256)",
            "seed": config.seed,
            "target_rows": config.target_rows,
            "max_prompt_tokens": config.max_prompt_tokens,
            "tokenizer_id": args.tokenizer_id,
            "system_prompt": SYSTEM_PROMPT,
            "boxed_suffix": BOXED_SUFFIX,
        },
        "stats": stats,
        "output": {
            "path": output_path.name,
            "rows": len(governed),
            "sha256": output_sha256,
            "columns": list(governed.columns),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "rows": len(governed),
                "sha256": output_sha256,
                **stats,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
