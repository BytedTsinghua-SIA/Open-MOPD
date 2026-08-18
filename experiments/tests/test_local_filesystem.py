"""Tests for the local-only filesystem boundary used by the trainer."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
UTILS = ROOT / "training" / "verl" / "verl" / "utils"
PACKAGE_NAME = "_openopd_local_utils"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(UTILS)]
sys.modules[PACKAGE_NAME] = package


def _load(name: str):
    module_name = f"{PACKAGE_NAME}.{name}"
    module_spec = importlib.util.spec_from_file_location(module_name, UTILS / f"{name}.py")
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    return module


local_io = _load("local_io")
local_fs = _load("local_fs")
copy = local_io.copy
copy_to_local = local_fs.copy_to_local
is_remote_path = local_fs.is_remote_path
local_mkdir_safe = local_fs.local_mkdir_safe


def test_local_paths_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("local-only\n", encoding="utf-8")
    destination = tmp_path / "nested" / "destination.txt"
    destination.parent.mkdir()

    assert copy_to_local(str(source)) == str(source)
    assert copy(str(source), str(destination)) == str(destination)
    assert destination.read_text(encoding="utf-8") == "local-only\n"

    created = local_mkdir_safe(str(tmp_path / "checkpoints"))
    assert Path(created).is_dir()
    assert Path(created).is_absolute()


def test_uri_paths_are_rejected() -> None:
    with pytest.raises(ValueError, match="remote URI is not supported"):
        copy_to_local("https://example.invalid/model")
    with pytest.raises(ValueError, match="remote URI is not supported"):
        is_remote_path("custom://example.invalid/checkpoint")
