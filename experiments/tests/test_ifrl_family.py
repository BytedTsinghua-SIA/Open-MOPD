import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_RL = ROOT / "training" / "scripts" / "rl"
if str(SCRIPTS_RL) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_RL))

from ifrl_family import classify_family


def test_classify_family_keywords_from_prompt() -> None:
    prompt = "Please follow these instructions: do not include keywords ['foo'] in the response."
    assert classify_family(prompt) == "keywords"


def test_classify_family_language_from_prompt() -> None:
    prompt = "The entire response should be in English, no other language is allowed."
    assert classify_family(prompt) == "language"
