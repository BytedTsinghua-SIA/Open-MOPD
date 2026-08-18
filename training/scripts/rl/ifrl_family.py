from __future__ import annotations

import re

# Keep in sync with verl/utils/reward_score/instruction_following.py
IF_RL_SYSTEM_PROMPT = (
    "You are an expert general-purpose assistant. Follow every user instruction exactly, "
    "especially formatting, wording, ordering, and length constraints. You must always begin "
    "your response with exactly <think>\n</think> (a single newline "
    "inside the tags), and then immediately provide your answer."
)
EXACT_THINK_PREFIX = "<think>\n</think>"


def classify_family(prompt: str) -> str:
    """Classify IF-RL prompts into coarse buckets used for sampling and guards."""
    text = " ".join(str(prompt or "").split())
    lowered = text.lower()

    if "entire response should be in" in lowered or "no other language is allowed" in lowered:
        return "language"
    if "keyword" in lowered or "keywords" in lowered:
        return "keywords"
    if (
        "stop word" in lowered
        or "no two consecutive words" in lowered
        or "consecutive words can share" in lowered
        or "consecutive words should" in lowered
        or "word repeats" in lowered
        or "words repeat" in lowered
        or "repeat words" in lowered
    ):
        return "lexical_constraints"
    if (
        "wrapped in double angular brackets" in lowered
        or "markdown divider" in lowered
        or "contain a title" in lowered
        or "numbered list" in lowered
        or "bullet list" in lowered
    ):
        return "detectable_format"
    if "less than" in lowered or "at least" in lowered:
        return "length_constraints"
    if (
        "there should be" in lowered
        or "should appear" in lowered
        or re.search(r"\binclude exactly \d+ numbers?\b", lowered)
        or re.search(r"\binclude at least \d+ numbers?\b", lowered)
        or re.search(r"\bexactly \d+ numbers?\b", lowered)
        or re.search(r"\b\d+ numbers? in the response\b", lowered)
        or re.search(r"\bletter [a-z] should appear\b", lowered)
    ):
        return "count_or_pattern"
    return "other"
