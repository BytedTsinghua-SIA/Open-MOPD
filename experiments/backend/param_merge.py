"""Build traceable parameter-merge baselines from sharded HF checkpoints.

The implementation follows each model's ``model.safetensors.index.json``
instead of assuming that checkpoints use the same shard boundaries.  Output
shards use the reference model's layout and are materialized one shard at a
time, which keeps peak host memory bounded for 7B-scale models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open
from safetensors.torch import save_file


INDEX_NAME = "model.safetensors.index.json"


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {value!r}")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {value!r}")
    return name, Path(raw_path).resolve()


SINGLE_SHARD_NAME = "model.safetensors"


def _load_weight_map(model_dir: Path) -> tuple[dict[str, str], dict]:
    index_path = model_dir / INDEX_NAME
    if not index_path.is_file():
        # Unsharded checkpoints ship a bare model.safetensors with no index;
        # synthesise the map so they can be merged against sharded ones.
        single = model_dir / SINGLE_SHARD_NAME
        if single.is_file():
            with safe_open(single, framework="pt", device="cpu") as handle:
                weight_map = {key: SINGLE_SHARD_NAME for key in handle.keys()}
            return weight_map, {"metadata": {}}
        raise FileNotFoundError(f"missing Hugging Face weight index: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid or empty weight_map in {index_path}")
    missing = sorted(
        {shard for shard in weight_map.values() if not (model_dir / shard).is_file()}
    )
    if missing:
        raise FileNotFoundError(f"missing shards under {model_dir}: {missing}")
    return weight_map, index


def _copy_reference_assets(reference_dir: Path, output_dir: Path) -> None:
    for source in sorted(reference_dir.iterdir()):
        if not source.is_file():
            continue
        if source.name == INDEX_NAME or source.suffix == ".safetensors":
            continue
        shutil.copy2(source, output_dir / source.name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _merge_float_tensors(
    teacher_tensors: list[torch.Tensor],
    *,
    mode: str,
    base_tensor: torch.Tensor | None,
    alphas: list[float],
) -> torch.Tensor:
    dtype = teacher_tensors[0].dtype
    accumulation_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    if mode == "average":
        merged = teacher_tensors[0].to(accumulation_dtype)
        for tensor in teacher_tensors[1:]:
            merged.add_(tensor.to(accumulation_dtype))
        merged.div_(len(teacher_tensors))
    else:
        if base_tensor is None:
            raise ValueError("task-arithmetic requires a base tensor")
        # W = W_base + sum_d alpha_d (W_d - W_base)
        #   = W_base (1 - sum_d alpha_d) + sum_d alpha_d W_d
        merged = base_tensor.to(accumulation_dtype).mul_(1.0 - sum(alphas))
        for tensor, alpha in zip(teacher_tensors, alphas):
            merged.add_(tensor.to(accumulation_dtype), alpha=alpha)
    return merged.to(dtype=dtype).contiguous()


def _merge_tensor(
    name: str,
    teacher_tensors: list[torch.Tensor],
    *,
    mode: str,
    base_tensor: torch.Tensor | None,
    alphas: list[float],
) -> torch.Tensor:
    first = teacher_tensors[0]
    for tensor in teacher_tensors[1:]:
        if tensor.shape != first.shape or tensor.dtype != first.dtype:
            raise ValueError(
                f"teacher tensor mismatch for {name}: "
                f"{first.shape}/{first.dtype} vs {tensor.shape}/{tensor.dtype}"
            )
    if base_tensor is not None and (
        base_tensor.shape != first.shape or base_tensor.dtype != first.dtype
    ):
        raise ValueError(
            f"base tensor mismatch for {name}: "
            f"{first.shape}/{first.dtype} vs {base_tensor.shape}/{base_tensor.dtype}"
        )
    if first.is_floating_point() or first.is_complex():
        return _merge_float_tensors(
            teacher_tensors, mode=mode, base_tensor=base_tensor, alphas=alphas
        )
    if any(not torch.equal(first, tensor) for tensor in teacher_tensors[1:]):
        raise ValueError(f"non-floating teacher tensors differ for {name}")
    if base_tensor is not None and not torch.equal(first, base_tensor):
        raise ValueError(f"non-floating base tensor differs for {name}")
    return first.contiguous()


def merge_models(
    *,
    inputs: list[tuple[str, Path]],
    reference_dir: Path,
    output_dir: Path,
    mode: str,
    base_dir: Path | None = None,
    alpha: float = 1.0,
    alpha_by_name: dict[str, float] | None = None,
    overwrite: bool = False,
) -> dict:
    if mode not in {"average", "task-arithmetic"}:
        raise ValueError(f"unsupported merge mode: {mode}")
    if not inputs:
        raise ValueError("at least one teacher input is required")
    if len({name for name, _ in inputs}) != len(inputs):
        raise ValueError("teacher input names must be unique")
    if alpha_by_name:
        if mode != "task-arithmetic":
            raise ValueError("per-teacher alpha is only valid for task-arithmetic")
        unknown = sorted(set(alpha_by_name) - {name for name, _ in inputs})
        if unknown:
            raise ValueError(f"alpha given for unknown teacher(s): {unknown}")
    alphas = [
        (alpha_by_name or {}).get(name, alpha) if mode == "task-arithmetic" else alpha
        for name, _ in inputs
    ]
    if mode == "task-arithmetic" and base_dir is None:
        raise ValueError("task-arithmetic requires base_dir")
    if mode == "average" and base_dir is not None:
        raise ValueError("base_dir is only valid for task-arithmetic")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    reference_map, reference_index = _load_weight_map(reference_dir)
    teacher_maps: dict[str, dict[str, str]] = {}
    for teacher_name, teacher_dir in inputs:
        teacher_maps[teacher_name], _ = _load_weight_map(teacher_dir)
        if set(teacher_maps[teacher_name]) != set(reference_map):
            missing = sorted(set(reference_map) - set(teacher_maps[teacher_name]))
            extra = sorted(set(teacher_maps[teacher_name]) - set(reference_map))
            raise ValueError(
                f"weight keys differ for teacher {teacher_name}; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
    base_map: dict[str, str] | None = None
    if base_dir is not None:
        base_map, _ = _load_weight_map(base_dir)
        if set(base_map) != set(reference_map):
            raise ValueError("base and reference weight keys differ")

    output_shards: dict[str, list[str]] = {}
    for tensor_name, shard_name in reference_map.items():
        output_shards.setdefault(shard_name, []).append(tensor_name)

    shard_records = []
    with ExitStack() as stack:
        teacher_handles: dict[str, dict[str, object]] = {}
        for teacher_name, teacher_dir in inputs:
            teacher_handles[teacher_name] = {
                shard: stack.enter_context(
                    safe_open(teacher_dir / shard, framework="pt", device="cpu")
                )
                for shard in sorted(set(teacher_maps[teacher_name].values()))
            }
        base_handles: dict[str, object] = {}
        if base_dir is not None and base_map is not None:
            base_handles = {
                shard: stack.enter_context(
                    safe_open(base_dir / shard, framework="pt", device="cpu")
                )
                for shard in sorted(set(base_map.values()))
            }

        for shard_index, (shard_name, tensor_names) in enumerate(
            sorted(output_shards.items()), start=1
        ):
            print(
                f"[{shard_index}/{len(output_shards)}] building {shard_name} "
                f"({len(tensor_names)} tensors)",
                flush=True,
            )
            output_tensors: dict[str, torch.Tensor] = {}
            for tensor_name in tensor_names:
                teachers = [
                    teacher_handles[teacher_name][teacher_maps[teacher_name][tensor_name]].get_tensor(
                        tensor_name
                    )
                    for teacher_name, _ in inputs
                ]
                base_tensor = (
                    base_handles[base_map[tensor_name]].get_tensor(tensor_name)
                    if base_map is not None
                    else None
                )
                output_tensors[tensor_name] = _merge_tensor(
                    tensor_name,
                    teachers,
                    mode=mode,
                    base_tensor=base_tensor,
                    alphas=alphas,
                )
            output_path = output_dir / shard_name
            save_file(output_tensors, output_path, metadata={"format": "pt"})
            shard_records.append(
                {
                    "name": shard_name,
                    "bytes": output_path.stat().st_size,
                    "sha256": _sha256(output_path),
                    "tensors": len(output_tensors),
                }
            )

    output_index = {
        "metadata": dict(reference_index.get("metadata") or {}),
        "weight_map": reference_map,
    }
    (output_dir / INDEX_NAME).write_text(
        json.dumps(output_index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _copy_reference_assets(reference_dir, output_dir)
    manifest = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "formula": (
            "mean(teachers)"
            if mode == "average"
            else "base + sum_d alpha_d * (teacher_d - base)"
        ),
        "alpha": alpha if mode == "task-arithmetic" else None,
        "alpha_by_teacher": (
            {name: a for (name, _), a in zip(inputs, alphas)}
            if mode == "task-arithmetic"
            else None
        ),
        "inputs": {name: str(path) for name, path in inputs},
        "base": str(base_dir) if base_dir is not None else None,
        "reference": str(reference_dir),
        "tensor_count": len(reference_map),
        "shards": shard_records,
    }
    (output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("average", "task-arithmetic"), required=True)
    parser.add_argument(
        "--input", action="append", type=_parse_named_path, required=True, metavar="NAME=PATH"
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--base", type=Path)
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="default alpha for every teacher"
    )
    parser.add_argument(
        "--alpha-for",
        action="append",
        default=[],
        metavar="NAME=ALPHA",
        help="per-teacher alpha override, e.g. if=0.3333333333333333",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    alpha_by_name: dict[str, float] = {}
    for item in args.alpha_for:
        if "=" not in item:
            raise SystemExit(f"--alpha-for expects NAME=ALPHA, got {item!r}")
        name, raw = item.split("=", 1)
        alpha_by_name[name] = float(raw)
    manifest = merge_models(
        inputs=args.input,
        reference_dir=args.reference.resolve(),
        output_dir=args.output_dir.resolve(),
        mode=args.mode,
        base_dir=args.base.resolve() if args.base is not None else None,
        alpha=args.alpha,
        alpha_by_name=alpha_by_name,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "mode": manifest["mode"],
                "tensor_count": manifest["tensor_count"],
                "shards": len(manifest["shards"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
