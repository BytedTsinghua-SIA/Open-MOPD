#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
VERIFIER_DIR = SCRIPT_DIR.parent
THIRD_PARTY_ROOT = VERIFIER_DIR / "third_party"
THIRD_PARTY_REPOS_ROOT = THIRD_PARTY_ROOT / "repos"
REPO_ROOT = VERIFIER_DIR.parent.parent


PATCH_ROOT = THIRD_PARTY_ROOT / "patches"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def repo_lock_path() -> Path:
    candidates = [
        THIRD_PARTY_ROOT / "repos.lock.json",
        REPO_ROOT / "third_party" / "repos.lock.json",
        REPO_ROOT.parent / "ModelMerging" / "third_party" / "repos.lock.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"repos.lock.json not found in: {candidates}")


def load_repo_lock() -> dict:
    return json.loads(repo_lock_path().read_text(encoding="utf-8"))


def repo_checkout_path(name: str) -> Path:
    entry = load_repo_lock()["repos"][name]
    relpath = entry.get("checkout")
    if relpath:
        rel = Path(relpath)
        if rel.parts[:2] == ("third_party", "repos"):
            rel = Path(*rel.parts[2:])
        return THIRD_PARTY_REPOS_ROOT / rel
    return THIRD_PARTY_REPOS_ROOT / name


def resolve_url(url: str) -> str:
    mirror = os.environ.get("OPENOPD_GIT_MIRROR", "").rstrip("/")
    if mirror and url.startswith("https://github.com/"):
        return mirror + "/" + url.removeprefix("https://github.com/")
    return url


def selected_repos(repo: str | None) -> dict[str, dict]:
    repos = load_repo_lock()["repos"]
    if repo:
        return {repo: repos[repo]}
    return repos


def sync_repo(name: str, entry: dict) -> None:
    checkout = repo_checkout_path(name)
    checkout.parent.mkdir(parents=True, exist_ok=True)
    url = resolve_url(entry["url"])
    if not (checkout / ".git").exists():
        checkout.mkdir(parents=True, exist_ok=True)
        run(["git", "init"], cwd=checkout)
        run(["git", "remote", "add", "origin", url], cwd=checkout)
    else:
        run(["git", "remote", "set-url", "origin", url], cwd=checkout)
    run(["git", "fetch", "--depth", "1", "--filter=blob:none", "origin", entry["commit"]], cwd=checkout)
    run(["git", "checkout", entry["commit"]], cwd=checkout)


def apply_patches(name: str) -> None:
    checkout = repo_checkout_path(name)
    patch_dir = PATCH_ROOT / name
    if not patch_dir.exists():
        patch_dir = REPO_ROOT.parent / "ModelMerging" / "third_party" / "patches" / name
    if not patch_dir.exists():
        return
    for patch in sorted(patch_dir.glob("*.patch")):
        run(["git", "apply", "--3way", str(patch)], cwd=checkout)


def status_repo(name: str) -> None:
    checkout = repo_checkout_path(name)
    print(f"[{name}] {checkout}")
    if not checkout.exists():
        print("  missing checkout")
        return
    run(["git", "status", "--short"], cwd=checkout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage OpenOPD third-party checkouts.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    choices = sorted(load_repo_lock()["repos"].keys())
    for command in ("sync", "apply-patches", "status"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--repo", choices=choices)
    args = parser.parse_args()

    repos = selected_repos(args.repo)
    if args.command == "sync":
        for name, entry in repos.items():
            sync_repo(name, entry)
        return
    if args.command == "apply-patches":
        for name in repos:
            apply_patches(name)
        return
    print(f"OpenOPD root: {REPO_ROOT}")
    for name in repos:
        status_repo(name)


if __name__ == "__main__":
    main()
