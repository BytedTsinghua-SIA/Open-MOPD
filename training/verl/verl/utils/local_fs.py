"""Local path helpers for the OpenOPD trainer."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

from .local_io import copy, exists, makedirs


def _local(path: str) -> str:
    if "://" in path:
        raise ValueError(f"remote URI is not supported by the local trainer: {path}")
    return path


def is_remote_path(path: str) -> bool:
    _local(path)
    return False


def md5_encode(path: str) -> str:
    return hashlib.md5(path.encode(), usedforsecurity=False).hexdigest()


def verify_copy(src: str, dest: str) -> bool:
    src_path, dest_path = Path(_local(src)), Path(_local(dest))
    if not src_path.exists() or not dest_path.exists():
        return False
    if src_path.is_file() != dest_path.is_file():
        return False
    if src_path.is_file():
        return src_path.stat().st_size == dest_path.stat().st_size
    src_files = {p.relative_to(src_path) for p in src_path.rglob("*")}
    dest_files = {p.relative_to(dest_path) for p in dest_path.rglob("*")}
    return src_files == dest_files


def copy_to_shm(src: str) -> str:
    source = Path(_local(src)).resolve()
    root = Path("/dev/shm/verl-cache")
    destination = root / md5_encode(str(source)) / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and verify_copy(str(source), str(destination)):
        return str(destination)
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=False, dirs_exist_ok=True)
    else:
        shutil.copy2(source, destination)
    return str(destination)


def copy_to_local(
    src: str, cache_dir: str | None = None, filelock: str = ".file.lock",
    verbose: bool = False, always_recopy: bool = False, use_shm: bool = False,
) -> str:
    """Validate and return a local path, optionally copying it to shared memory."""
    del cache_dir, filelock, verbose, always_recopy
    local_path = _local(src)
    if not os.path.exists(local_path):
        raise FileNotFoundError(local_path)
    return copy_to_shm(local_path) if use_shm else local_path


def copy_local_path(src: str, *args: object, **kwargs: object) -> str:
    """Backward-compatible local name for callers that previously staged data."""
    return copy_to_local(src, *args, **kwargs)


def local_mkdir_safe(path: str) -> str:
    local_path = str(Path(_local(path)).resolve())
    Path(local_path).mkdir(parents=True, exist_ok=True)
    return local_path


__all__ = [
    "copy", "copy_local_path", "copy_to_local", "copy_to_shm", "exists",
    "is_remote_path", "local_mkdir_safe", "makedirs", "md5_encode", "verify_copy",
]
