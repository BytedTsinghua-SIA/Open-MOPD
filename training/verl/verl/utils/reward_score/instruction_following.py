from __future__ import annotations

import json
import re
import sys
import threading
import urllib.request
import zipfile
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

# Keep in sync with training/scripts/rl/ifrl_family.py
IF_RL_SYSTEM_PROMPT = (
    "You are an expert general-purpose assistant. Follow every user instruction exactly, "
    "especially formatting, wording, ordering, and length constraints. You must always begin "
    "your response with exactly <think>\n</think> (a single newline "
    "inside the tags), and then immediately provide your answer."
)
EXACT_THINK_PREFIX = "<think>\n</think>"
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OFFICIAL_EVAL_INIT_LOCK = threading.Lock()


def _repo_root() -> Path:
    # .../training/verl/verl/utils/reward_score/instruction_following.py -> repo root
    return Path(__file__).resolve().parents[5]


def _add_local_verifiable_instructions_repo() -> None:
    current = Path(__file__).resolve()
    candidate_roots = [
        _repo_root() / "training" / "third_party" / "verifiable-instructions",
        _repo_root() / "third_party" / "verifiable-instructions",
        _repo_root().parent / "verifiable-instructions",
        current.parents[5] / "training" / "third_party" / "verifiable-instructions",
        current.parents[5] / "verifiable-instructions",
    ]
    for root in candidate_roots:
        if (root / "verifiable_instructions").exists():
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            return


def _ensure_nltk_data() -> None:
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
        return
    except LookupError:
        pass
    except zipfile.BadZipFile:
        for data_dir in map(Path, nltk.data.path):
            broken_zip = data_dir / "tokenizers" / "punkt_tab.zip"
            if broken_zip.exists():
                broken_zip.unlink()

    download_dir = Path(nltk.data.path[1])
    tokenizers_dir = download_dir / "tokenizers"
    tokenizers_dir.mkdir(parents=True, exist_ok=True)
    zip_path = tokenizers_dir / "punkt_tab.zip"

    with urllib.request.urlopen(
        "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip",
        timeout=30,
    ) as response:
        zip_path.write_bytes(response.read())

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(tokenizers_dir)

    nltk.data.find("tokenizers/punkt_tab")


def _load_registry():
    _add_local_verifiable_instructions_repo()
    try:
        from verifiable_instructions import instructions_registry
    except Exception as exc:  # pragma: no cover - exercised in runtime env rather than unit tests
        raise ImportError(
            "Failed to import official verifier package `verifiable_instructions`. "
            "Install the verifiable-instructions repo into the training environment, "
            f"for example under `{_repo_root() / 'training' / 'third_party' / 'verifiable-instructions'}`."
        ) from exc

    _ensure_nltk_data()
    return instructions_registry


def _parse_kwargs(raw_items: list[str]) -> list[Any]:
    return [json.loads(item) for item in raw_items]


def strip_think_tags(text: str) -> str:
    return THINK_BLOCK_RE.sub("", str(text or ""), count=1).strip()


def has_exact_think_format(text: str) -> bool:
    """True when the response starts with the exact empty think block required by IF-RL SP."""
    return str(text or "").lstrip().startswith(EXACT_THINK_PREFIX)


def answer_text_for_constraints(text: str) -> str:
    stripped = str(text or "").lstrip()
    if stripped.startswith(EXACT_THINK_PREFIX):
        return stripped[len(EXACT_THINK_PREFIX) :].lstrip("\r\n")
    return strip_think_tags(stripped)


def _tiered_think_format_score(*, format_ok: bool, content_ok: bool) -> float:
    if content_ok and format_ok:
        return 1.0
    if content_ok:
        return 0.8
    if format_ok:
        return 0.2
    return 0.0


def _scoring_mode(reward_kwargs: dict[str, Any]) -> str:
    return str(reward_kwargs.get("scoring_mode", "tiered_think_format") or "tiered_think_format").strip().lower()


def _should_use_official_eval_for_aligned_val(extra_info: dict[str, Any], reward_kwargs: dict[str, Any]) -> bool:
    if not _as_bool(reward_kwargs.get("auto_official_eval_for_aligned_val", True)):
        return False
    if extra_info.get("eval_prompt_for_scorer"):
        return True
    evaluator = str(extra_info.get("evaluator", "") or "").strip().lower()
    if evaluator in {"ifeval", "ifbench"}:
        return True
    dataset = str(extra_info.get("dataset", "") or "").strip().lower()
    return bool(extra_info.get("metadata")) and dataset in {
        "ifeval",
        "ifbench_test",
        "ifbench_mt_ifbench",
        "ifbench_mt_ifeval",
    }


def _parse_metadata(raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        return json.loads(raw)
    return dict(raw)


def _patch_registry_missing(target_registry: Any, source_registry: Any) -> None:
    target = getattr(target_registry, "INSTRUCTION_DICT", None)
    source = getattr(source_registry, "INSTRUCTION_DICT", None)
    if not isinstance(target, dict) or not isinstance(source, dict):
        return
    for instruction_id, instruction_cls in source.items():
        target.setdefault(instruction_id, instruction_cls)


@lru_cache(maxsize=2)
def _load_official_instruction_eval(ifbench: bool):
    # Async reward actors can enter this loader concurrently on their first
    # validation batch.  langdetect lazily initializes a process-global profile
    # table and is not safe under that first-call race (it raises "Need to load
    # profiles", which the official IFEval checker treats as a pass).  Serialize
    # imports/registry patching and eagerly prime the detector so validation is
    # deterministic instead of receiving false-positive language checks.
    with _OFFICIAL_EVAL_INIT_LOCK:
        repo_root = _repo_root()
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

        from evals.verifier.score_functions.common.utils import repo_path
        from evals.verifier.score_functions.instruction_following.common import (
            filter_kwargs_for_instructions,
            normalize_kwargs,
        )
        from evals.verifier.score_functions.instruction_following.google_ifeval import (
            instructions_registry as google_instructions_registry,
        )

        if ifbench:
            ifbench_repo = repo_path("ifbench")
            nltk_data = ifbench_repo / ".nltk_data"
            if nltk_data.exists():
                import nltk  # type: ignore

                nltk.data.path.insert(0, str(nltk_data))
            if str(ifbench_repo) not in sys.path:
                sys.path.insert(0, str(ifbench_repo))
            import evaluation_lib  # type: ignore
            import instructions_registry  # type: ignore
            _patch_registry_missing(instructions_registry, google_instructions_registry)
        else:
            from evals.verifier.score_functions.instruction_following.google_ifeval import evaluation_lib
            from evals.verifier.score_functions.instruction_following.google_ifeval import instructions_registry
            try:
                ifbench_repo = repo_path("ifbench")
            except Exception:
                ifbench_repo = None
            if ifbench_repo is not None:
                if str(ifbench_repo) not in sys.path:
                    sys.path.insert(0, str(ifbench_repo))
                try:
                    import instructions_registry as ifbench_instructions_registry  # type: ignore
                except Exception:
                    pass
                else:
                    _patch_registry_missing(instructions_registry, ifbench_instructions_registry)

        import langdetect

        langdetect.DetectorFactory.seed = 0
        langdetect.detect("This sentence initializes deterministic English language detection profiles.")

        return evaluation_lib, instructions_registry, normalize_kwargs, filter_kwargs_for_instructions


def _official_eval_score(
    solution_str: str,
    extra_info: dict[str, Any],
    data_source: str | None,
) -> dict[str, Any]:
    dataset = str(extra_info.get("dataset") or data_source or "")
    is_ifbench = dataset in {"ifbench_test", "ifbench_mt_ifbench"} or str(extra_info.get("evaluator")) == "ifbench"
    evaluation_lib, instructions_registry, normalize_kwargs, filter_kwargs = _load_official_instruction_eval(is_ifbench)

    metadata = _parse_metadata(extra_info.get("metadata"))
    instruction_ids = metadata.get("instruction_id_list", extra_info.get("instruction_id_list", [])) or []
    kwargs = metadata.get("kwargs", extra_info.get("kwargs", []))
    instruction_ids = [str(item) for item in instruction_ids]
    kwargs = filter_kwargs(normalize_kwargs(kwargs), instruction_ids, instructions_registry)
    key = metadata.get("key", extra_info.get("sample_id", extra_info.get("request_id", 0)))
    prompt = str(extra_info.get("eval_prompt_for_scorer") or extra_info.get("raw_prompt") or metadata.get("prompt") or "")

    input_example = evaluation_lib.InputExample(
        key=int(key) if str(key).isdigit() else key,
        instruction_id_list=instruction_ids,
        prompt=prompt,
        kwargs=kwargs,
    )
    response = {prompt: strip_think_tags(solution_str)}
    strict_output = evaluation_lib.test_instruction_following_strict(input_example, response)
    loose_output = evaluation_lib.test_instruction_following_loose(input_example, response)

    strict_checks = list(getattr(strict_output, "follow_instruction_list", []) or [])
    loose_checks = list(getattr(loose_output, "follow_instruction_list", []) or [])
    strict_pass = bool(getattr(strict_output, "follow_all_instructions", False))
    loose_pass = bool(getattr(loose_output, "follow_all_instructions", False))
    strict_instruction_accuracy = float(sum(strict_checks) / len(strict_checks)) if strict_checks else float(strict_pass)
    loose_instruction_accuracy = float(sum(loose_checks) / len(loose_checks)) if loose_checks else float(loose_pass)

    return {
        "score": float(strict_pass),
        "acc": float(strict_pass),
        "base_score": strict_instruction_accuracy,
        "constraint_ratio": strict_instruction_accuracy,
        "official_eval_strict_prompt": float(strict_pass),
        "official_eval_loose_prompt": float(loose_pass),
        "official_eval_strict_instruction": strict_instruction_accuracy,
        "official_eval_loose_instruction": loose_instruction_accuracy,
        "num_constraints": len(strict_checks),
        "num_failed_constraints": int(len(strict_checks) - sum(strict_checks)),
        "all_constraints_passed": float(strict_pass),
        "answer_text": strip_think_tags(solution_str),
        "raw_response_text": solution_str,
        "dataset": dataset,
        "scoring_mode": "official_eval",
    }


def _evaluate_constraints(
    solution_str: str,
    instruction_ids: list[str],
    kwargs_list: list[Any],
    registry: Any,
) -> tuple[float, list[float], list[str], list[str]]:
    checks: list[float] = []
    failed_constraints: list[str] = []
    checker_errors: list[str] = []
    for instruction_id, kwargs in zip(instruction_ids, kwargs_list, strict=True):
        instruction_cls = registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        kwargs = kwargs or {}
        filtered_kwargs = {key: value for key, value in kwargs.items() if value is not None}
        instruction.build_description(**filtered_kwargs)
        try:
            passed = bool(instruction.check_following(solution_str))
        except Exception as exc:
            # Some upstream checkers assume non-empty responses; fail closed for RL sampling.
            passed = False
            checker_errors.append(f"{instruction_id}:{type(exc).__name__}")
        checks.append(1.0 if passed else 0.0)
        if not passed:
            failed_constraints.append(instruction_id)
    base_score = float(sum(checks) / len(checks)) if checks else 0.0
    return base_score, checks, failed_constraints, checker_errors


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _is_effectively_single_token(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 50:
        return False
    return bool(stripped) and not any(ch.isspace() for ch in stripped) and "<<" not in stripped


def _is_digit_sequence(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and bool(re.fullmatch(r"[\d\s,.;:/+\-_]+", stripped)) and any(ch.isdigit() for ch in stripped)


def _looks_like_semantically_empty(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.isdigit():
        return True
    if _is_digit_sequence(stripped):
        return True
    if re.fullmatch(r'["\'`“”‘’\s]+', stripped):
        return True
    lowered = stripped.lower()
    if lowered in {"?", "!", "ok", "yes", "no", "maybe", "n/a", "null", "none"}:
        return True
    if "anything else i can help with" in lowered and len(stripped) <= 64:
        return True
    return False


def _csv_set(value: Any, default: set[str]) -> set[str]:
    if value is None:
        return default
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    if isinstance(value, Iterable):
        return {str(item).strip() for item in value if str(item).strip()}
    return default


def _matches_instruction_prefixes(instruction_ids: list[str], patterns: set[str]) -> bool:
    for instruction_id in instruction_ids:
        normalized_id = str(instruction_id).strip()
        if not normalized_id:
            continue
        for pattern in patterns:
            normalized_pattern = str(pattern).strip()
            if normalized_pattern and (
                normalized_id == normalized_pattern or normalized_id.startswith(normalized_pattern)
            ):
                return True
    return False


def _json_float_map(value: Any) -> dict[str, float]:
    if value is None or value == "":
        return {}
    parsed = json.loads(value) if isinstance(value, str) else value
    if isinstance(parsed, Mapping):
        items = parsed.items()
    elif hasattr(parsed, "items"):
        items = parsed.items()
    else:
        raise ValueError("Expected a JSON object or mapping for semantic guard reason-specific caps.")
    return {str(key).strip(): float(val) for key, val in items if str(key).strip()}


def _is_constraint_heavy(family: str, instruction_ids: list[str]) -> bool:
    if family in {"count_or_pattern", "lexical_constraints"}:
        return True
    return any(str(item).startswith(("count:", "words:", "ratio:")) for item in instruction_ids)


def _looks_like_short_answer_task(prompt: str, instruction_ids: list[str]) -> bool:
    lowered = prompt.lower()
    short_answer_markers = (
        "answer with one word",
        "respond with one word",
        "in one word",
        "one-word",
        "single word",
        "answer with a single",
        "respond with a single",
        "yes or no",
        "true or false",
        "multiple choice",
        "choose one",
        "select one",
        "pick one",
        "only the letter",
        "letter only",
        "answer only with",
        "respond only with",
    )
    if any(marker in lowered for marker in short_answer_markers):
        return True
    return any("option" in str(item).lower() or "multiple_choice" in str(item).lower() for item in instruction_ids)


def _apply_short_fullscore_penalty(
    score: float,
    answer_text: str,
    family: str,
    reward_kwargs: dict[str, Any],
) -> tuple[float, float]:
    if not _as_bool(reward_kwargs.get("enable_short_fullscore_penalty", False)):
        return score, 0.0
    exempt_families = _csv_set(
        reward_kwargs.get("short_fullscore_penalty_exempt_families"),
        {"length_constraints", "detectable_format"},
    )
    if score != 1.0 or family in exempt_families:
        return score, 0.0

    length = len(answer_text.strip())
    penalty = 0.0
    tiny_len = int(reward_kwargs.get("short_fullscore_penalty_tiny_len", 8))
    min_len = int(reward_kwargs.get("short_fullscore_penalty_min_len", 16))
    if length < tiny_len:
        penalty = float(reward_kwargs.get("short_fullscore_penalty_tiny", 0.5))
    elif length < min_len:
        penalty = float(reward_kwargs.get("short_fullscore_penalty_small", 0.25))

    if penalty <= 0:
        return score, 0.0
    return max(score - penalty, 0.0), penalty


def _short_fullscore_band_penalties(raw_value: Any) -> dict[int, float]:
    if raw_value is None or raw_value == "":
        return {120: 0.45, 200: 0.30, 400: 0.15}
    parsed = _json_float_map(raw_value)
    penalties: dict[int, float] = {}
    for raw_threshold, penalty in parsed.items():
        threshold = int(raw_threshold)
        if threshold <= 0:
            raise ValueError("short_fullscore_band_penalty_json thresholds must be positive integers.")
        penalties[threshold] = float(penalty)
    return dict(sorted(penalties.items()))


def _short_fullscore_band_penalty_config(
    family: str,
    instruction_ids: list[str],
    reward_kwargs: dict[str, Any],
) -> tuple[dict[int, float], str]:
    weak_families = _csv_set(reward_kwargs.get("short_fullscore_band_penalty_weak_families"), set())
    weak_instruction_prefixes = _csv_set(
        reward_kwargs.get("short_fullscore_band_penalty_weak_instruction_prefixes"), set()
    )
    if family in weak_families or _matches_instruction_prefixes(instruction_ids, weak_instruction_prefixes):
        weak_json = reward_kwargs.get("short_fullscore_band_penalty_weak_json")
        if weak_json is None or weak_json == "":
            weak_json = {"120": 0.20, "200": 0.10, "400": 0.0}
        return _short_fullscore_band_penalties(weak_json), "weak"
    return _short_fullscore_band_penalties(reward_kwargs.get("short_fullscore_band_penalty_json")), "standard"


def _apply_short_fullscore_band_penalty(
    score: float,
    base_score: float,
    answer_text: str,
    family: str,
    instruction_ids: list[str],
    raw_prompt: str,
    reward_kwargs: dict[str, Any],
) -> tuple[float, float, str, str]:
    if not _as_bool(reward_kwargs.get("enable_short_fullscore_band_penalty", False)):
        return score, 0.0, "", ""
    exempt_families = _csv_set(
        reward_kwargs.get("short_fullscore_band_penalty_exempt_families"),
        {"length_constraints", "detectable_format"},
    )
    exempt_instruction_prefixes = _csv_set(
        reward_kwargs.get("short_fullscore_band_penalty_exempt_instruction_prefixes"), set()
    )
    if base_score != 1.0 or family in exempt_families:
        return score, 0.0, "", ""
    if _matches_instruction_prefixes(instruction_ids, exempt_instruction_prefixes):
        return score, 0.0, "", ""
    if _looks_like_short_answer_task(raw_prompt, instruction_ids):
        return score, 0.0, "", ""

    length = len(answer_text.strip())
    penalties, mode = _short_fullscore_band_penalty_config(family, instruction_ids, reward_kwargs)
    for threshold, penalty in penalties.items():
        if length < threshold:
            if penalty <= 0:
                return score, 0.0, f"lt_{threshold}", mode
            new_score = max(score - penalty, 0.0)
            return new_score, score - new_score, f"lt_{threshold}", mode
    return score, 0.0, "", ""


def _apply_semantic_guard(
    score: float,
    answer_text: str,
    family: str,
    reward_kwargs: dict[str, Any],
) -> tuple[float, float]:
    if not _as_bool(reward_kwargs.get("enable_semantic_guard", False)):
        return score, 0.0
    guard_families = _csv_set(
        reward_kwargs.get("semantic_guard_families"),
        {"other", "keywords", "language", "count_or_pattern", "lexical_constraints"},
    )
    if score != 1.0 or family not in guard_families:
        return score, 0.0

    stripped = answer_text.strip()
    max_len = int(reward_kwargs.get("semantic_guard_max_len", 50))
    if len(stripped) > max_len or not (_is_effectively_single_token(stripped) or _looks_like_semantically_empty(stripped)):
        return score, 0.0

    penalty = float(reward_kwargs.get("semantic_guard_penalty", 0.5))
    if penalty <= 0:
        return score, 0.0
    return max(score - penalty, 0.0), penalty


def _semantic_guard_active_reason(
    base_score: float,
    answer_text: str,
    family: str,
    instruction_ids: list[str],
    reward_kwargs: dict[str, Any],
) -> str:
    if not _as_bool(reward_kwargs.get("enable_semantic_guard_active", False)):
        return ""
    if base_score != 1.0:
        return ""

    active_families = _csv_set(
        reward_kwargs.get("semantic_guard_active_families"),
        {"other", "keywords", "language", "count_or_pattern", "lexical_constraints"},
    )
    exempt_families = _csv_set(
        reward_kwargs.get("semantic_guard_active_exempt_families"),
        {"length_constraints", "detectable_format"},
    )
    if family not in active_families or family in exempt_families:
        return ""

    stripped = answer_text.strip()
    max_len = int(reward_kwargs.get("semantic_guard_active_max_len", 50))
    constraint_max_len = int(reward_kwargs.get("semantic_guard_active_constraint_max_len", max_len))
    tiny_len = int(reward_kwargs.get("semantic_guard_active_tiny_len", 20))
    if len(stripped) <= max_len:
        if _is_digit_sequence(stripped):
            return "digit_only"
        if _looks_like_semantically_empty(stripped):
            return "semantically_empty"
        if len(stripped) <= tiny_len and re.fullmatch(r"[A-Za-z0-9]", stripped):
            return "trivial_tiny_output"
        if _is_effectively_single_token(stripped):
            return "single_tokenish"
    if len(stripped) <= constraint_max_len and _is_constraint_heavy(family, instruction_ids):
        return "constraint_only_short_answer"
    return ""


def _apply_semantic_guard_active_cap(
    score: float,
    base_score: float,
    answer_text: str,
    family: str,
    instruction_ids: list[str],
    reward_kwargs: dict[str, Any],
) -> tuple[float, float, str, str]:
    reason = _semantic_guard_active_reason(base_score, answer_text, family, instruction_ids, reward_kwargs)
    if not reason:
        return score, 0.0, "", ""

    if (
        "semantic_guard_active_cap_reasons" not in reward_kwargs
        and "semantic_guard_active_soft_reasons" not in reward_kwargs
    ):
        cap_reasons = None
        soft_reasons: set[str] = set()
    else:
        cap_reasons = _csv_set(
            reward_kwargs.get("semantic_guard_active_cap_reasons"),
            {"digit_only", "single_tokenish", "semantically_empty", "trivial_tiny_output"},
        )
        soft_reasons = _csv_set(reward_kwargs.get("semantic_guard_active_soft_reasons"), set())

    if cap_reasons is None or reason in cap_reasons:
        cap_by_reason = _json_float_map(reward_kwargs.get("semantic_guard_active_cap_by_reason_json"))
        cap = cap_by_reason.get(reason, float(reward_kwargs.get("semantic_guard_active_cap", 0.5)))
        if cap < 0:
            cap = 0.0
        new_score = min(score, cap)
        return new_score, max(score - new_score, 0.0), reason, "cap"

    if reason in soft_reasons:
        penalty = float(reward_kwargs.get("semantic_guard_active_soft_penalty", 0.1))
        if penalty <= 0:
            return score, 0.0, reason, "observe"
        new_score = max(score - penalty, 0.0)
        return new_score, score - new_score, reason, "soft"

    return score, 0.0, reason, "observe"


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    data_source: str | None = None,
    **reward_kwargs: Any,
) -> dict[str, Any]:
    del ground_truth
    extra_info = extra_info or {}
    scoring_mode = _scoring_mode(reward_kwargs)
    if scoring_mode in {"official_eval", "eval_official", "standalone_eval"}:
        return _official_eval_score(solution_str, extra_info, data_source)
    if _should_use_official_eval_for_aligned_val(extra_info, reward_kwargs):
        result = _official_eval_score(solution_str, extra_info, data_source)
        result["scoring_mode"] = "official_eval_auto"
        return result

    instruction_ids = list(extra_info.get("instruction_id_list", []))
    kwargs_json = list(extra_info.get("instruction_kwargs_json", []))
    kwargs_list = _parse_kwargs(kwargs_json)
    family = str(extra_info.get("family", "other") or "other")
    raw_prompt = str(extra_info.get("raw_prompt", "") or "")

    registry = _load_registry()
    format_ok = has_exact_think_format(solution_str)
    answer_text = answer_text_for_constraints(solution_str)
    base_score, checks, failed_constraints, checker_errors = _evaluate_constraints(
        answer_text,
        instruction_ids,
        kwargs_list,
        registry,
    )
    content_ok = not failed_constraints
    if scoring_mode == "legacy":
        score = base_score
    else:
        score = _tiered_think_format_score(format_ok=format_ok, content_ok=content_ok)
    short_fullscore_penalty = 0.0
    short_fullscore_band_penalty = 0.0
    short_fullscore_band_bucket = ""
    short_fullscore_band_mode = ""
    semantic_guard_penalty = 0.0
    semantic_guard_active_penalty = 0.0
    semantic_guard_active_reason = ""
    semantic_guard_active_action = ""

    guard_text = answer_text
    score, short_fullscore_penalty = _apply_short_fullscore_penalty(score, guard_text, family, reward_kwargs)
    (
        score,
        short_fullscore_band_penalty,
        short_fullscore_band_bucket,
        short_fullscore_band_mode,
    ) = _apply_short_fullscore_band_penalty(
        score,
        base_score,
        guard_text,
        family,
        instruction_ids,
        raw_prompt,
        reward_kwargs,
    )
    score, semantic_guard_penalty = _apply_semantic_guard(score, guard_text, family, reward_kwargs)
    (
        score,
        semantic_guard_active_penalty,
        semantic_guard_active_reason,
        semantic_guard_active_action,
    ) = _apply_semantic_guard_active_cap(
        score, base_score, guard_text, family, instruction_ids, reward_kwargs
    )

    return {
        "score": score,
        "base_score": base_score,
        "constraint_ratio": base_score,
        "tiered_score": _tiered_think_format_score(format_ok=format_ok, content_ok=content_ok),
        "think_format_ok": float(format_ok),
        "content_ok": float(content_ok),
        "scoring_mode": scoring_mode,
        "num_constraints": len(checks),
        "num_failed_constraints": len(failed_constraints),
        "num_checker_errors": len(checker_errors),
        "all_constraints_passed": float(content_ok),
        "answer_text": answer_text,
        "raw_response_text": solution_str,
        "failed_constraints_text": ",".join(failed_constraints),
        "checker_errors_text": ",".join(checker_errors),
        "family": family,
        "response_char_len": len(answer_text.strip()),
        "is_single_tokenish": float(_is_effectively_single_token(guard_text)),
        "looks_semantically_empty": float(_looks_like_semantically_empty(guard_text)),
        "short_fullscore_penalty": short_fullscore_penalty,
        "short_fullscore_band_penalty": short_fullscore_band_penalty,
        "short_fullscore_band_bucket": short_fullscore_band_bucket,
        "short_fullscore_band_mode": short_fullscore_band_mode,
        "short_fullscore_total_penalty": short_fullscore_penalty + short_fullscore_band_penalty,
        "semantic_guard_penalty": semantic_guard_penalty,
        "semantic_guard_active_penalty": semantic_guard_active_penalty,
        "semantic_guard_active_cap_penalty": (
            semantic_guard_active_penalty if semantic_guard_active_action == "cap" else 0.0
        ),
        "semantic_guard_active_soft_penalty": (
            semantic_guard_active_penalty if semantic_guard_active_action == "soft" else 0.0
        ),
        "semantic_guard_active_triggered": float(bool(semantic_guard_active_reason)),
        "semantic_guard_active_reason": semantic_guard_active_reason,
        "semantic_guard_active_action": semantic_guard_active_action,
        "semantic_guard_family_enabled": float(
            family
            in _csv_set(
                reward_kwargs.get("semantic_guard_families"),
                {"other", "keywords", "language", "count_or_pattern", "lexical_constraints"},
            )
        ),
    }


def reward_func(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **reward_kwargs: Any,
) -> dict[str, Any]:
    """Adapter for verl NaiveRewardManager / custom_reward_function.name=reward_func."""
    return compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        data_source=data_source,
        **reward_kwargs,
    )


_GRM_EXTRA_KEYS_DEFAULTS: dict[str, float | str] = {
    "grm_called": 0.0,
    "grm_pass": 0.0,
    "grm_capped": 0.0,
    "grm_penalty": 0.0,
    "grm_error": 0.0,
}


def _ensure_grm_keys(result: dict[str, Any]) -> None:
    for key, default in _GRM_EXTRA_KEYS_DEFAULTS.items():
        result.setdefault(key, default)


def _semantic_fail_cap(result: dict[str, Any], config: "GRMConfig") -> float:
    """Score a semantic-fail rollout caps down to.

    By default the independent think-format bonus (0.2*F) is preserved and only the
    rule/semantic task reward is removed; otherwise it caps to the absolute grm_fail_cap.
    """
    if config.keep_think_on_fail:
        think_ok = float(result.get("think_format_ok", 0.0)) >= 1.0
        return _tiered_think_format_score(format_ok=think_ok, content_ok=False)
    return config.fail_cap


def _is_official_eval_row(
    scoring_mode: str,
    extra_info: dict[str, Any],
    reward_kwargs: dict[str, Any],
) -> bool:
    """GRM only applies to training rows; validation uses the official eval scorer."""
    if scoring_mode in {"official_eval", "eval_official", "standalone_eval"}:
        return True
    return _should_use_official_eval_for_aligned_val(extra_info, reward_kwargs)


def _grm_is_candidate(
    result: dict[str, Any],
    extra_info: dict[str, Any],
    config: "GRMConfig",
    scoring_mode: str,
    reward_kwargs: dict[str, Any],
) -> bool:
    if not config.is_usable:
        return False
    if _is_official_eval_row(scoring_mode, extra_info, reward_kwargs):
        return False
    # By default every IF family is judged; an explicit `grm_families` list restricts it.
    if config.families:
        family = str(result.get("family", extra_info.get("family", "other")) or "other")
        if family not in config.families:
            return False
    # Only judge rollouts that already satisfy every verifiable constraint, since the
    # hacking we care about is "rule passes but the task is not actually solved".
    return float(result.get("content_ok", 0.0)) >= 1.0


def compute_score_batch(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[str],
    extra_infos: list[dict[str, Any] | None],
    **reward_kwargs: Any,
) -> list[dict[str, Any]]:
    """Batch entry for verl's BatchRewardManager.

    Computes the existing rule-based instruction-following score per item, then (when
    enabled) gates rule-passing semantic-family rollouts with a generative reward model:
    if the GRM judge decides the underlying task was not genuinely solved, the reward is
    capped to ``grm_fail_cap``. The judge is only queried for rule-passing training rows.
    """
    from verl.utils.reward_score import if_grm

    config = if_grm.GRMConfig.from_reward_kwargs(reward_kwargs)

    results: list[dict[str, Any]] = []
    candidate_indices: list[int] = []
    candidate_conversations: list[Any] = []
    candidate_answers: list[str] = []

    for index, (data_source, solution_str, ground_truth, extra_info) in enumerate(
        zip(data_sources, solution_strs, ground_truths, extra_infos, strict=True)
    ):
        extra_info = extra_info or {}
        result = compute_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            data_source=data_source,
            **reward_kwargs,
        )
        _ensure_grm_keys(result)
        results.append(result)

        scoring_mode = str(result.get("scoring_mode", "") or "").strip().lower()
        if _grm_is_candidate(result, extra_info, config, scoring_mode, reward_kwargs):
            # raw_prompt may be a single instruction string or a multi-turn message list;
            # the judge client normalizes either form into a transcript.
            conversation = extra_info.get("raw_prompt", "")
            answer_text = str(result.get("answer_text", "") or "")
            candidate_indices.append(index)
            candidate_conversations.append(conversation)
            candidate_answers.append(answer_text)

    if not candidate_indices:
        return results

    selector = if_grm.make_selector(config)
    verdicts = if_grm.fan_out_judges(candidate_conversations, candidate_answers, config, selector)

    for index, verdict in zip(candidate_indices, verdicts, strict=True):
        result = results[index]
        result["grm_called"] = 1.0
        original_score = float(result.get("score", 0.0))

        if verdict is None:
            result["grm_error"] = 1.0
            should_cap = config.fail_mode == "closed"
        else:
            result["grm_pass"] = 1.0 if verdict else 0.0
            should_cap = verdict is False

        if should_cap:
            # Semantic failure removes the task reward but (by default) keeps the
            # independent think-format bonus; see _semantic_fail_cap.
            capped = min(original_score, _semantic_fail_cap(result, config))
            penalty = max(original_score - capped, 0.0)
            result["score"] = capped
            result["grm_capped"] = 1.0
            result["grm_penalty"] = penalty

    return results
