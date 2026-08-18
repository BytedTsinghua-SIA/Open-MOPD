#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from code_rl_prompts import CODE_RL_THINK_TAG_SYSTEM_PROMPT

LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the problem and enclose "
    "your code within delimiters."
)
LCB_FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly "
    "test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when "
    "the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
)

TRAIN_SOURCES: tuple[tuple[str, str], ...] = (
    ("taco", "taco"),
    ("primeintellect", "primeintellect"),
    ("lcbv5", "livecodebench"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert DeepCoder-Preview-Dataset train splits into Verl/OpenOPD RL parquet, "
            "filtering prompts longer than max_prompt_length under the given tokenizer."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def load_json(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def fetch_livecodebench_prompt(problem: str, starter_code: str | None) -> str:
    prompt = f"### Question:\n{problem}"
    starter_code = (starter_code or "").strip()
    if starter_code:
        prompt += f"\n\n### Format: {LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"\n\n### Format: {LCB_FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def build_prompt_messages(user_content: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": CODE_RL_THINK_TAG_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def attach_metadata(tests: Any, metadata: dict[str, Any] | None) -> Any:
    if not metadata:
        return tests
    if isinstance(tests, dict):
        merged = dict(tests)
        merged["metadata"] = metadata
        return merged
    if isinstance(tests, list):
        return [{**test, "metadata": metadata} for test in tests]
    return tests


def build_ground_truth(row: dict[str, Any], source_config: str) -> str:
    tests = load_json(row["tests"])
    metadata = row.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        metadata = dict(metadata)
    if source_config == "lcbv5":
        func_name = (metadata or {}).get("func_name")
        if func_name is not None:
            assert "func_name" in (metadata or {}), f"missing func_name metadata for lcbv5 row: {metadata}"
    tests = attach_metadata(tests, metadata if isinstance(metadata, dict) else None)
    return json.dumps(tests, ensure_ascii=False)


def build_user_content(row: dict[str, Any], source_config: str) -> str:
    problem = str(row["problem"])
    if source_config == "lcbv5":
        starter_code = row.get("starter_code")
        if starter_code is None:
            starter_code = ""
        return fetch_livecodebench_prompt(problem, str(starter_code))
    return problem


def build_schema() -> pa.Schema:
    message_struct = pa.struct([("role", pa.string()), ("content", pa.string())])
    return pa.schema(
        [
            ("data_source", pa.string()),
            ("prompt", pa.list_(message_struct)),
            ("ability", pa.string()),
            (
                "reward_model",
                pa.struct([("style", pa.string()), ("ground_truth", pa.string())]),
            ),
            (
                "extra_info",
                pa.struct(
                    [
                        ("split", pa.string()),
                        ("index", pa.string()),
                        ("source_config", pa.string()),
                    ]
                ),
            ),
        ]
    )


def prompt_token_length(tokenizer: AutoTokenizer, messages: list[dict[str, str]]) -> int:
    return len(
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
        )
    )


def iter_train_rows(input_root: Path) -> list[tuple[str, str, Path]]:
    files: list[tuple[str, str, Path]] = []
    for source_config, data_source in TRAIN_SOURCES:
        pattern = "train-*.parquet" if source_config == "lcbv5" else "train-*.parquet"
        for path in sorted((input_root / source_config).glob(pattern)):
            files.append((source_config, data_source, path))
    return files


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_path), trust_remote_code=True)
    schema = build_schema()

    kept_rows: list[dict[str, Any]] = []
    stats = {
        "input_root": str(args.input_root),
        "output_parquet": str(args.output_parquet),
        "tokenizer_path": str(args.tokenizer_path),
        "system_prompt": CODE_RL_THINK_TAG_SYSTEM_PROMPT,
        "max_prompt_length": args.max_prompt_length,
        "max_rows": args.max_rows,
        "sources": {},
        "filtered_overlong": 0,
        "filtered_errors": 0,
        "kept_rows": 0,
    }

    global_index = 0
    for source_config, data_source, parquet_path in iter_train_rows(args.input_root):
        source_stats = stats["sources"].setdefault(
            source_config,
            {
                "files": 0,
                "raw_rows": 0,
                "kept_rows": 0,
                "filtered_overlong": 0,
                "filtered_errors": 0,
            },
        )
        source_stats["files"] += 1

        table = pq.read_table(parquet_path)
        for row in table.to_pylist():
            if args.max_rows is not None and stats["kept_rows"] >= args.max_rows:
                break
            source_stats["raw_rows"] += 1
            try:
                user_content = build_user_content(row, source_config)
                messages = build_prompt_messages(user_content)
                if prompt_token_length(tokenizer, messages) > args.max_prompt_length:
                    source_stats["filtered_overlong"] += 1
                    stats["filtered_overlong"] += 1
                    continue
                kept_rows.append(
                    {
                        "data_source": data_source,
                        "prompt": messages,
                        "ability": "code",
                        "reward_model": {
                            "style": "rule",
                            "ground_truth": build_ground_truth(row, source_config),
                        },
                        "extra_info": {
                            "split": "train",
                            "index": str(global_index),
                            "source_config": source_config,
                        },
                    }
                )
                global_index += 1
                source_stats["kept_rows"] += 1
                stats["kept_rows"] += 1
            except Exception:
                source_stats["filtered_errors"] += 1
                stats["filtered_errors"] += 1
        if args.max_rows is not None and stats["kept_rows"] >= args.max_rows:
            break

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(args.output_parquet, schema=schema) as writer:
        for start in range(0, len(kept_rows), args.batch_size):
            batch = kept_rows[start : start + args.batch_size]
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))

    summary_path = args.summary_path or args.output_parquet.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
