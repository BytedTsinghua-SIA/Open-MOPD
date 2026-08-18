from verl.utils.reward_score.instruction_following import (
    EXACT_THINK_PREFIX,
    _apply_semantic_guard,
    _apply_short_fullscore_penalty,
    answer_text_for_constraints,
    compute_score,
    has_exact_think_format,
    strip_think_tags,
)


def test_short_fullscore_penalty_hits_tiny_answers() -> None:
    score, penalty = _apply_short_fullscore_penalty(
        1.0,
        "ok",
        "other",
        {
            "enable_short_fullscore_penalty": True,
            "short_fullscore_penalty_tiny_len": 8,
            "short_fullscore_penalty_min_len": 16,
            "short_fullscore_penalty_tiny": 0.5,
        },
    )
    assert penalty == 0.5
    assert score == 0.5


def test_semantic_guard_penalizes_empty_single_token() -> None:
    score, penalty = _apply_semantic_guard(
        1.0,
        "yes",
        "keywords",
        {
            "enable_semantic_guard": True,
            "semantic_guard_penalty": 0.5,
            "semantic_guard_max_len": 50,
        },
    )
    assert penalty == 0.5
    assert score == 0.5


def test_strip_think_tags_removes_reasoning_block() -> None:
    raw = f"{EXACT_THINK_PREFIX}\nHello world"
    assert strip_think_tags(raw) == "Hello world"
    assert answer_text_for_constraints(raw) == "Hello world"


def test_has_exact_think_format_requires_empty_single_newline_block() -> None:
    assert has_exact_think_format(f"{EXACT_THINK_PREFIX}\nanswer")
    assert not has_exact_think_format("<think>plan</think>\nanswer")
    assert not has_exact_think_format("answer only")


def test_tiered_scoring_matrix() -> None:
    extra_info = {
        "instruction_id_list": ["keywords:forbidden_words"],
        "instruction_kwargs_json": ['{"forbidden_words": ["bad"]}'],
        "family": "keywords",
        "raw_prompt": "Do not use bad",
    }
    good_answer = "This is a good response with enough words."
    perfect = f"{EXACT_THINK_PREFIX}\n{good_answer}"
    wrong_format = f"<think>plan</think>\n{good_answer}"
    format_only = f"{EXACT_THINK_PREFIX}\nThis is bad"
    all_wrong = "plain bad answer"

    assert compute_score(perfect, "", extra_info=extra_info)["score"] == 1.0
    assert compute_score(wrong_format, "", extra_info=extra_info)["score"] == 0.8
    assert compute_score(format_only, "", extra_info=extra_info)["score"] == 0.2
    assert compute_score(all_wrong, "", extra_info=extra_info)["score"] == 0.0


def test_checker_exception_fails_closed_for_empty_answer() -> None:
    result = compute_score(
        EXACT_THINK_PREFIX,
        "",
        extra_info={
            "instruction_id_list": ["last_word:last_word_answer"],
            "instruction_kwargs_json": ['{"last_word": "done"}'],
            "family": "other",
            "raw_prompt": "End with done.",
        },
    )

    assert result["score"] == 0.2
    assert result["base_score"] == 0.0
    assert result["num_checker_errors"] == 1
    assert result["failed_constraints_text"] == "last_word:last_word_answer"
    assert result["checker_errors_text"] == "last_word:last_word_answer:IndexError"


def test_guard_only_variant_uses_passive_semantic_guard_only() -> None:
    score, penalty = _apply_semantic_guard(
        1.0,
        "42",
        "other",
        {
            "enable_semantic_guard": True,
            "enable_semantic_guard_active": False,
            "semantic_guard_penalty": 0.5,
        },
    )
    assert penalty == 0.5
    assert score == 0.5


def test_aligned_eval_metadata_uses_official_scorer(monkeypatch) -> None:
    def fake_official_eval_score(solution_str, extra_info, data_source):
        assert solution_str == "answer"
        assert extra_info["evaluator"] == "ifbench"
        assert data_source == "ifbench_test"
        return {"score": 1.0, "base_score": 1.0, "scoring_mode": "official_eval"}

    monkeypatch.setattr(
        "verl.utils.reward_score.instruction_following._official_eval_score",
        fake_official_eval_score,
    )
    result = compute_score(
        "answer",
        "",
        data_source="ifbench_test",
        extra_info={
            "dataset": "ifbench_test",
            "evaluator": "ifbench",
            "eval_prompt_for_scorer": "prompt",
            "metadata": '{"instruction_id_list": ["count:keywords_multiple"], "kwargs": []}',
        },
    )

    assert result["score"] == 1.0
    assert result["scoring_mode"] == "official_eval_auto"
