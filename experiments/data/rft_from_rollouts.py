"""Build traceable rejection-finetuning data from persisted RL rollout JSONL.

The extractor is intentionally stream-oriented: callers can pipe a local
rollout stream through stdin without staging the whole dataset in memory. For each prompt it
keeps the latest successful trajectory (highest score and a stable hash break
ties), then writes an SFT-compatible ``train.parquet`` plus a provenance
manifest.  ``combine`` deterministically balances several extracted domains.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterable

import pandas as pd

try:
    import orjson
except ImportError:  # pragma: no cover - stdlib fallback for minimal images
    orjson = None


SCHEMA_VERSION = 1


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _loads(line: bytes) -> dict[str, Any]:
    value = orjson.loads(line) if orjson is not None else json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("rollout row must be a JSON object")
    return value


def _messages(prompt: str, response: str) -> list[dict[str, str]]:
    """Convert VERL's flat ``system\n...\nuser\n...`` prompt to messages."""

    system = ""
    user = prompt
    if prompt.startswith("system\n") and "\nuser\n" in prompt:
        system, user = prompt[len("system\n") :].split("\nuser\n", 1)
    for suffix in ("\nassistant\n", "assistant\n"):
        if user.endswith(suffix):
            user = user[: -len(suffix)]
            break
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": response})
    return messages


def _successful(row: dict[str, Any], domain: str, min_score: float) -> bool:
    try:
        score = float(row.get("score", 0.0))
    except (TypeError, ValueError):
        return False
    if score < min_score:
        return False
    if domain == "if":
        if "all_constraints_passed" in row and float(row["all_constraints_passed"] or 0) < 1:
            return False
        if "content_ok" in row and float(row["content_ok"] or 0) < 1:
            return False
    return True


@dataclass(frozen=True)
class Candidate:
    step: int
    score: float
    output_hash: str
    record: dict[str, Any]

    @property
    def rank(self) -> tuple[int, float, str]:
        # Smaller output hash wins an otherwise exact tie.  Negating its byte
        # order would obscure the rule, so compare step/score first and handle
        # the stable hash explicitly in ``_is_better``.
        return self.step, self.score, self.output_hash


def _is_better(new: Candidate, old: Candidate) -> bool:
    if new.step != old.step:
        return new.step > old.step
    if new.score != old.score:
        return new.score > old.score
    return new.output_hash < old.output_hash


def extract_rows(
    stream: BinaryIO,
    *,
    domain: str,
    source_run: str,
    source_dir: str,
    min_score: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: dict[str, Candidate] = {}
    stats = {"input_rows": 0, "invalid_json_rows": 0, "eligible_rows": 0}
    for row_index, line in enumerate(stream, start=1):
        stats["input_rows"] += 1
        if not line.strip():
            continue
        try:
            row = _loads(line)
        except (ValueError, TypeError, json.JSONDecodeError):
            stats["invalid_json_rows"] += 1
            continue
        if not _successful(row, domain, min_score):
            continue
        prompt = str(row.get("input") or row.get("prompt") or "")
        response = str(row.get("output") or row.get("generation") or "")
        if not prompt or not response:
            continue
        stats["eligible_rows"] += 1
        step = int(row.get("step", -1))
        score = float(row.get("score", 0.0))
        prompt_hash = _sha256(prompt)
        output_hash = _sha256(response)
        record = {
            "messages": _messages(prompt, response),
            "domain": domain,
            "source_run": source_run,
            "source_step": step,
            "source_row": row_index,
            "source_score": score,
            "source_file": f"{source_dir.rstrip('/')}/{step}.jsonl",
            "prompt_sha256": prompt_hash,
            "output_sha256": output_hash,
            "trace_id": f"{source_run}:{step}:{row_index}:{output_hash[:12]}",
        }
        candidate = Candidate(step=step, score=score, output_hash=output_hash, record=record)
        previous = selected.get(prompt_hash)
        if previous is None or _is_better(candidate, previous):
            selected[prompt_hash] = candidate
    rows = [selected[key].record for key in sorted(selected)]
    stats["selected_prompts"] = len(rows)
    return rows, stats


def _write_dataset(output_dir: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["domain", "prompt_sha256"], kind="stable").reset_index(drop=True)
    frame.to_parquet(output_dir / "train.parquet", index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(frame),
        "columns": list(frame.columns),
        **manifest,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def extract_command(args: argparse.Namespace) -> None:
    rows, stats = extract_rows(
        sys.stdin.buffer,
        domain=args.domain,
        source_run=args.source_run,
        source_dir=args.source_dir,
        min_score=args.min_score,
    )
    _write_dataset(
        args.output_dir,
        rows,
        {
            "kind": "rft-latest-success-per-prompt",
            "domain": args.domain,
            "min_score": args.min_score,
            "source_run": args.source_run,
            "source_dir": args.source_dir,
            "stats": stats,
        },
    )
    print(json.dumps({"output_dir": str(args.output_dir), **stats}, sort_keys=True))


def _balanced_rows(
    domain_frames: dict[str, pd.DataFrame], *, rows_per_domain: int | None, seed: int
) -> tuple[list[dict[str, Any]], int]:
    available = {domain: len(frame) for domain, frame in domain_frames.items()}
    target = rows_per_domain if rows_per_domain is not None else min(available.values())
    if target <= 0:
        raise ValueError(f"rows_per_domain must be positive; available={available}")
    if any(count < target for count in available.values()):
        raise ValueError(f"requested {target} rows/domain but available={available}")
    rows: list[dict[str, Any]] = []
    for domain in sorted(domain_frames):
        frame = domain_frames[domain].copy()
        frame["_sample_key"] = frame["prompt_sha256"].map(
            lambda value: _sha256(f"{seed}:{domain}:{value}")
        )
        sampled = frame.sort_values("_sample_key", kind="stable").head(target).drop(columns="_sample_key")
        rows.extend(sampled.to_dict(orient="records"))
    return rows, target


def combine_command(args: argparse.Namespace) -> None:
    domain_frames: dict[str, pd.DataFrame] = {}
    sources: dict[str, str] = {}
    source_manifests: dict[str, dict[str, Any]] = {}
    for item in args.input:
        if "=" not in item:
            raise ValueError(f"--input must be DOMAIN=PARQUET, got: {item}")
        domain, raw_path = item.split("=", 1)
        path = Path(raw_path)
        if domain in domain_frames:
            raise ValueError(f"duplicate domain: {domain}")
        domain_frames[domain] = pd.read_parquet(path)
        sources[domain] = str(path)
        manifest_path = path.parent / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("domain") != domain:
                raise ValueError(
                    f"source manifest domain mismatch: {manifest_path} "
                    f"contains {manifest.get('domain')!r}, expected {domain!r}"
                )
            source_manifests[domain] = manifest
    if not domain_frames:
        raise ValueError("at least one --input is required")
    rows, target = _balanced_rows(domain_frames, rows_per_domain=args.rows_per_domain, seed=args.seed)
    _write_dataset(
        args.output_dir,
        rows,
        {
            "kind": "rft-balanced-multidomain",
            "seed": args.seed,
            "rows_per_domain": target,
            "available_rows": {domain: len(frame) for domain, frame in domain_frames.items()},
            "sources": sources,
            "source_manifests": source_manifests,
        },
    )
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(rows), "rows_per_domain": target}))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="read rollout JSONL from stdin")
    extract.add_argument("--domain", choices=("math", "code", "if"), required=True)
    extract.add_argument("--source-run", required=True)
    extract.add_argument("--source-dir", required=True)
    extract.add_argument("--min-score", type=float, default=1.0)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.set_defaults(func=extract_command)

    combine = subparsers.add_parser("combine", help="balance extracted domain parquets")
    combine.add_argument("--input", action="append", default=[], metavar="DOMAIN=PARQUET")
    combine.add_argument("--rows-per-domain", type=int)
    combine.add_argument("--seed", type=int, default=42)
    combine.add_argument("--output-dir", type=Path, required=True)
    combine.set_defaults(func=combine_command)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
