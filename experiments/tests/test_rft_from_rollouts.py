from __future__ import annotations

import io
import json
from argparse import Namespace
from pathlib import Path

import pandas as pd

from experiments.data.rft_from_rollouts import _messages, combine_command, extract_rows


def _stream(*rows: str) -> io.BytesIO:
    return io.BytesIO(("\n".join(rows) + "\n").encode())


def test_latest_success_and_provenance() -> None:
    rows, stats = extract_rows(
        _stream(
            '{"input":"system\\ns\\nuser\\nu","output":"old","score":1,"step":1}',
            '{"input":"system\\ns\\nuser\\nu","output":"bad","score":0,"step":3}',
            '{"input":"system\\ns\\nuser\\nu","output":"new","score":1,"step":2}',
        ),
        domain="math",
        source_run="run",
        source_dir="/tmp/trace",
    )
    assert stats == {"input_rows": 3, "invalid_json_rows": 0, "eligible_rows": 2, "selected_prompts": 1}
    assert rows[0]["source_step"] == 2
    assert rows[0]["source_file"] == "/tmp/trace/2.jsonl"
    assert rows[0]["messages"][-1] == {"role": "assistant", "content": "new"}


def test_if_requires_full_constraints_and_content() -> None:
    rows, stats = extract_rows(
        _stream(
            '{"input":"p1","output":"x","score":1,"step":1,"all_constraints_passed":0,"content_ok":1}',
            '{"input":"p2","output":"x","score":1,"step":1,"all_constraints_passed":1,"content_ok":0}',
            '{"input":"p3","output":"x","score":1,"step":1,"all_constraints_passed":1,"content_ok":1}',
        ),
        domain="if",
        source_run="run",
        source_dir="/tmp/trace",
    )
    assert stats["selected_prompts"] == 1
    assert rows[0]["messages"][0]["content"] == "p3"


def test_flat_prompt_parser() -> None:
    assert _messages("system\nsys\nuser\nquestion\nassistant\n", "answer") == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]


def test_combine_embeds_source_manifests(tmp_path: Path) -> None:
    inputs = []
    for domain in ("math", "code", "if"):
        source = tmp_path / domain
        source.mkdir()
        pd.DataFrame(
            [
                {
                    "messages": [{"role": "user", "content": domain}],
                    "domain": domain,
                    "prompt_sha256": f"{domain}-{row}",
                }
                for row in range(2)
            ]
        ).to_parquet(source / "train.parquet", index=False)
        (source / "manifest.json").write_text(
            json.dumps({"domain": domain, "source_dir": f"/tmp/trace/{domain}"}),
            encoding="utf-8",
        )
        inputs.append(f"{domain}={source / 'train.parquet'}")
    output = tmp_path / "combined"
    combine_command(
        Namespace(input=inputs, rows_per_domain=1, seed=42, output_dir=output)
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["rows"] == 3
    assert manifest["source_manifests"]["code"]["source_dir"] == "/tmp/trace/code"
