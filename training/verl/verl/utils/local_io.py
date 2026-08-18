"""Small local-filesystem helpers used by the vendored trainer.

The OpenOPD launchers accept local paths only.  Keeping these operations in a
separate module makes accidental remote staging impossible: URI-like paths are
rejected before any filesystem command is issued.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def _local(path: str | os.PathLike[str]) -> str:
    value = os.fspath(path)
    if "://" in value:
        raise ValueError(f"remote URI is not supported by the local trainer: {value}")
    return value


def exists(path: str, **_: object) -> bool:
    return os.path.exists(_local(path))


def makedirs(name: str, mode: int = 0o777, exist_ok: bool = False, **_: object) -> None:
    os.makedirs(_local(name), mode=mode, exist_ok=exist_ok)


def copy(src: str, dst: str, **kwargs: object) -> str:
    source = _local(src)
    target = _local(dst)
    if os.path.isdir(source):
        allowed = {key: value for key, value in kwargs.items() if key in {"symlinks", "ignore", "copy_function", "ignore_dangling_symlinks", "dirs_exist_ok"}}
        return shutil.copytree(source, target, **allowed)
    return shutil.copy2(source, target)


def is_local_path(path: str) -> bool:
    return "://" not in path


__all__ = ["copy", "exists", "is_local_path", "makedirs"]
