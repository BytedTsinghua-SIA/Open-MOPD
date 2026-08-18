from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


MATH_SYSTEM = (
    "You are an expert mathematical problem solver. Solve the problem carefully, "
    "show the necessary reasoning, and make the final answer easy to verify."
)
MATH_SUFFIX = "Please reason step by step, and put your final answer in \\boxed{}."

MESSAGE_TYPE = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))
REWARD_MODEL_TYPE = pa.struct([("style", pa.string()), ("ground_truth", pa.string())])
EXTRA_INFO_TYPE = pa.struct([("index", pa.string())])


OLD_SUFFIX_RE = re.compile(
    r"\s*Please reason step by step, and put your final answer (?:within|in) \\+boxed\{\}\.?\s*$",
    flags=re.IGNORECASE,
)
ANSWER_LINE_SUFFIX_RE = re.compile(
    r'\s*Remember to put your answer on its own line after "Answer:"\.?\s*$',
    flags=re.IGNORECASE,
)
DAPO_PREFIX_RE = re.compile(
    r"^\s*Solve the following math problem step by step\. "
    r"The last line of your response should be of the form Answer: \$Answer "
    r"\(without quotes\) where \$Answer is the answer to the problem\.\s*",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize math eval/SFT/RL parquets into veRL chat-message format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="kind", required=True)

    for kind in ("eval", "sft", "rl"):
        sub = subparsers.add_parser(kind)
        sub.add_argument("--input", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def as_messages(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    if isinstance(value, str):
        value = json.loads(value)
    return [
        {"role": str(item.get("role", "")), "content": str(item.get("content", ""))}
        for item in list(value)
    ]


def strip_old_math_instruction(text: str) -> str:
    text = DAPO_PREFIX_RE.sub("", str(text or "")).strip()
    text = ANSWER_LINE_SUFFIX_RE.sub("", text).strip()
    text = OLD_SUFFIX_RE.sub("", text).strip()
    return text


def math_prompt_messages(problem: str) -> list[dict[str, str]]:
    problem = strip_old_math_instruction(problem)
    user = f"{problem}\n\n{MATH_SUFFIX}"
    return [
        {"role": "system", "content": MATH_SYSTEM},
        {"role": "user", "content": user},
    ]


def normalize_sft_messages(value: Any) -> list[dict[str, str]]:
    messages = as_messages(value)
    user_content = ""
    assistant_messages: list[dict[str, str]] = []
    for message in messages:
        role = message["role"]
        if role == "user" and not user_content:
            user_content = message["content"]
        elif role == "assistant":
            assistant_messages.append(message)
    if not user_content and messages:
        user_content = messages[0]["content"]
    return [*math_prompt_messages(user_content), *assistant_messages]


def normalize_rl_prompt(value: Any) -> list[dict[str, str]]:
    messages = as_messages(value)
    user_content = ""
    for message in messages:
        if message["role"] == "user":
            user_content = message["content"]
            break
    if not user_content and messages:
        user_content = messages[-1]["content"]
    return math_prompt_messages(user_content)


def iter_rows(path: Path, batch_size: int) -> Iterable[list[dict[str, Any]]]:
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        yield batch.to_pylist()


def write_batches(path: Path, schema: pa.Schema, batches: Iterable[list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(path, schema=schema) as writer:
        for rows in batches:
            if not rows:
                continue
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))


def normalize_eval(args: argparse.Namespace) -> None:
    schema = pa.schema(
        [
            ("dataset", pa.string()),
            ("domain", pa.string()),
            ("sample_id", pa.string()),
            ("prompt", MESSAGE_TYPE),
            ("original_prompt", pa.string()),
            ("answer", pa.string()),
            ("metadata", pa.string()),
            ("max_tokens", pa.int64()),
            ("temperature", pa.float64()),
            ("top_p", pa.float64()),
            ("top_k", pa.int64()),
            ("evaluator", pa.string()),
            ("repeat_idx", pa.int64()),
            ("request_id", pa.string()),
            ("data_source", pa.string()),
            ("reward_model", REWARD_MODEL_TYPE),
            ("extra_info", EXTRA_INFO_TYPE),
        ]
    )

    def batches() -> Iterable[list[dict[str, Any]]]:
        for rows in iter_rows(args.input, args.batch_size):
            out = []
            for row in rows:
                dataset = str(row.get("dataset") or "")
                sample_id = str(row.get("sample_id") or row.get("request_id") or "")
                answer = "" if row.get("answer") is None else str(row.get("answer"))
                original_prompt = str(row.get("prompt") or "")
                out.append(
                    {
                        "dataset": dataset,
                        "domain": "math",
                        "sample_id": sample_id,
                        "prompt": math_prompt_messages(original_prompt),
                        "original_prompt": original_prompt,
                        "answer": answer,
                        "metadata": "" if row.get("metadata") is None else str(row.get("metadata")),
                        "max_tokens": row.get("max_tokens"),
                        "temperature": row.get("temperature"),
                        "top_p": row.get("top_p"),
                        "top_k": row.get("top_k"),
                        "evaluator": "" if row.get("evaluator") is None else str(row.get("evaluator")),
                        "repeat_idx": row.get("repeat_idx"),
                        "request_id": "" if row.get("request_id") is None else str(row.get("request_id")),
                        "data_source": dataset,
                        "reward_model": {"style": "rule-lighteval/MATH_v2", "ground_truth": answer},
                        "extra_info": {"index": sample_id},
                    }
                )
            yield out

    write_batches(args.output, schema, batches())


def normalize_sft(args: argparse.Namespace) -> None:
    schema = pa.schema(
        [
            ("source_shard", pa.string()),
            ("source_row_index", pa.int64()),
            ("messages", MESSAGE_TYPE),
            ("ground_truth", pa.string()),
            ("boxed_answer", pa.string()),
            ("id", pa.int64()),
            ("source", pa.string()),
            ("subject", pa.string()),
            ("difficulty", pa.int64()),
            ("pass_rate", pa.float64()),
        ]
    )

    def batches() -> Iterable[list[dict[str, Any]]]:
        for rows in iter_rows(args.input, args.batch_size):
            out = []
            for row in rows:
                out.append(
                    {
                        "source_shard": row.get("source_shard"),
                        "source_row_index": row.get("source_row_index"),
                        "messages": normalize_sft_messages(row.get("messages")),
                        "ground_truth": row.get("ground_truth"),
                        "boxed_answer": row.get("boxed_answer"),
                        "id": row.get("id"),
                        "source": row.get("source"),
                        "subject": row.get("subject"),
                        "difficulty": row.get("difficulty"),
                        "pass_rate": row.get("pass_rate"),
                    }
                )
            yield out

    write_batches(args.output, schema, batches())


def normalize_rl(args: argparse.Namespace) -> None:
    schema = pa.schema(
        [
            ("data_source", pa.string()),
            ("prompt", MESSAGE_TYPE),
            ("ability", pa.string()),
            ("reward_model", REWARD_MODEL_TYPE),
            ("extra_info", EXTRA_INFO_TYPE),
        ]
    )

    def batches() -> Iterable[list[dict[str, Any]]]:
        for rows in iter_rows(args.input, args.batch_size):
            out = []
            for row in rows:
                reward_model = row.get("reward_model") or {}
                extra_info = row.get("extra_info") or {}
                out.append(
                    {
                        "data_source": str(row.get("data_source") or "math_dapo"),
                        "prompt": normalize_rl_prompt(row.get("prompt")),
                        "ability": str(row.get("ability") or "MATH"),
                        "reward_model": {
                            "style": str(reward_model.get("style") or "rule-lighteval/MATH_v2"),
                            "ground_truth": str(reward_model.get("ground_truth") or ""),
                        },
                        "extra_info": {"index": str(extra_info.get("index") or "")},
                    }
                )
            yield out

    write_batches(args.output, schema, batches())


def main() -> None:
    args = parse_args()
    if args.kind == "eval":
        normalize_eval(args)
    elif args.kind == "sft":
        normalize_sft(args)
    elif args.kind == "rl":
        normalize_rl(args)
    else:
        raise ValueError(args.kind)


if __name__ == "__main__":
    main()
