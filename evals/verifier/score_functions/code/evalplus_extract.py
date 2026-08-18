from __future__ import annotations

import re

try:
    from evalplus.sanitize import sanitize
except Exception:
    sanitize = None


_FENCED_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_solution_code(raw_text: str, entry_point: str) -> str:
    blocks = [match.group(1).strip() for match in _FENCED_BLOCK_RE.finditer(raw_text)]
    entry_pattern = re.compile(rf"(^|\n)\s*def\s+{re.escape(entry_point)}\s*\(")

    for block in reversed(blocks):
        if entry_pattern.search(block):
            return block

    if blocks:
        return blocks[-1]

    if sanitize is not None:
        return sanitize(raw_text, entrypoint=entry_point).strip()

    return raw_text.strip()
