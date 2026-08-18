from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from experiments.backend.param_merge import merge_models


def _write_model(path: Path, values: dict[str, torch.Tensor], layout: list[list[str]]) -> None:
    path.mkdir(parents=True)
    weight_map = {}
    for index, names in enumerate(layout, start=1):
        shard = f"model-{index:05d}-of-{len(layout):05d}.safetensors"
        save_file({name: values[name] for name in names}, path / shard)
        weight_map.update({name: shard for name in names})
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 24}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    (path / "config.json").write_text('{"model_type":"tiny"}\n', encoding="utf-8")


def _read_model(path: Path) -> dict[str, torch.Tensor]:
    index = json.loads((path / "model.safetensors.index.json").read_text())
    result = {}
    for name, shard in index["weight_map"].items():
        with safe_open(path / shard, framework="pt", device="cpu") as handle:
            result[name] = handle.get_tensor(name)
    return result


def test_average_and_task_arithmetic_allow_different_shard_layouts(tmp_path: Path) -> None:
    base_values = {
        "a": torch.tensor([1.0, 2.0]),
        "b": torch.tensor([3.0, 4.0]),
        "c": torch.tensor([5.0, 6.0]),
    }
    teacher_1 = {name: value + 1 for name, value in base_values.items()}
    teacher_2 = {name: value + 3 for name, value in base_values.items()}
    base = tmp_path / "base"
    one = tmp_path / "one"
    two = tmp_path / "two"
    _write_model(base, base_values, [["a", "b"], ["c"]])
    _write_model(one, teacher_1, [["a"], ["b", "c"]])
    _write_model(two, teacher_2, [["a", "c"], ["b"]])

    average = tmp_path / "average"
    merge_models(
        inputs=[("one", one), ("two", two)],
        reference_dir=base,
        output_dir=average,
        mode="average",
    )
    average_values = _read_model(average)
    for name, value in base_values.items():
        torch.testing.assert_close(average_values[name], value + 2)
    assert json.loads((average / "config.json").read_text())["model_type"] == "tiny"

    task = tmp_path / "task"
    merge_models(
        inputs=[("one", one), ("two", two)],
        reference_dir=base,
        base_dir=base,
        output_dir=task,
        mode="task-arithmetic",
        alpha=0.5,
    )
    task_values = _read_model(task)
    for name, value in base_values.items():
        # base + 0.5 * ((base + 1 - base) + (base + 3 - base))
        torch.testing.assert_close(task_values[name], value + 2)
    manifest = json.loads((task / "merge_manifest.json").read_text())
    assert manifest["alpha"] == 0.5
    assert manifest["formula"] == "base + sum_d alpha_d * (teacher_d - base)"
    assert manifest["alpha_by_teacher"] == {"one": 0.5, "two": 0.5}

    # Per-teacher alpha: base + 1.0*(+1) + 0.25*(+3) = base + 1.75
    weighted = tmp_path / "weighted"
    merge_models(
        inputs=[("one", one), ("two", two)],
        reference_dir=base,
        base_dir=base,
        output_dir=weighted,
        mode="task-arithmetic",
        alpha=1.0,
        alpha_by_name={"two": 0.25},
    )
    weighted_values = _read_model(weighted)
    for name, value in base_values.items():
        torch.testing.assert_close(weighted_values[name], value + 1.75)
    weighted_manifest = json.loads((weighted / "merge_manifest.json").read_text())
    assert weighted_manifest["alpha_by_teacher"] == {"one": 1.0, "two": 0.25}

    # An alpha for a teacher that was not supplied is a hard error, not a no-op.
    with pytest.raises(ValueError, match="unknown teacher"):
        merge_models(
            inputs=[("one", one)],
            reference_dir=base,
            base_dir=base,
            output_dir=tmp_path / "bad",
            mode="task-arithmetic",
            alpha_by_name={"nope": 0.5},
        )
