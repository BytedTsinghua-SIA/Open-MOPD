from verl.utils.reward_score import if_grm
from verl.utils.reward_score.instruction_following import EXACT_THINK_PREFIX, compute_score_batch

_KEYWORDS_EXTRA = {
    "instruction_id_list": ["keywords:forbidden_words"],
    "instruction_kwargs_json": ['{"forbidden_words": ["bad"]}'],
    "family": "keywords",
    "raw_prompt": "Explain how benzodiazepine scheduling works in NYS.",
}
_GOOD_ANSWER = "Benzodiazepines are scheduled separately and follow distinct prescribing rules here."
_PASSING = f"{EXACT_THINK_PREFIX}\n{_GOOD_ANSWER}"  # think OK + rule OK
_PASSING_NO_THINK = _GOOD_ANSWER  # rule OK but no think format
_FAILING = f"{EXACT_THINK_PREFIX}\nThis is bad"

_GRM_KWARGS = {
    "enable_grm": True,
    "grm_endpoints": ["http://fake-endpoint:8000"],
    "grm_fail_cap": 0.0,
}

_GRM_KEYS = {"grm_called", "grm_pass", "grm_capped", "grm_penalty", "grm_error"}


def test_parse_verdict_variants() -> None:
    assert if_grm.parse_verdict("<verdict>YES</verdict>") is True
    assert if_grm.parse_verdict("<verdict>no</verdict>") is False
    assert if_grm.parse_verdict("reason then <verdict>YES</verdict>") is True
    assert if_grm.parse_verdict("Reasoning...\nYES") is True
    assert if_grm.parse_verdict("Reasoning...\nNO") is False
    assert if_grm.parse_verdict("The final answer is \\boxed{yes}") is True
    assert if_grm.parse_verdict("\\boxed{No}") is False
    assert if_grm.parse_verdict("") is None
    assert if_grm.parse_verdict("yes and no together") is None


def test_build_judge_messages_uses_system_prompt_and_transcript() -> None:
    config = if_grm.GRMConfig.from_reward_kwargs({"enable_grm": True, "grm_endpoints": ["http://x"]})
    conversation = [
        {"role": "system", "content": "follow all formatting rules"},
        {"role": "user", "content": "First, summarize the article."},
        {"role": "assistant", "content": "Here is the summary..."},
        {"role": "user", "content": "Now slightly reduce the word count."},
    ]
    messages = if_grm.build_judge_messages(conversation, "1", config)

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == if_grm.GRM_JUDGE_SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    transcript = messages[1]["content"]
    # Constraint-only system turn is dropped; both user turns and the prior assistant turn are kept.
    assert "follow all formatting rules" not in transcript
    assert transcript.count("[User]") == 2
    assert "[Assistant]\nHere is the summary..." in transcript
    assert transcript.rstrip().endswith(f"{if_grm.TO_BE_JUDGED_LABEL}\n1")


def test_render_transcript_accepts_plain_string() -> None:
    transcript = if_grm.render_transcript("Explain quantum tunneling.", "theory theory theory")
    assert transcript == (
        "[User]\nExplain quantum tunneling.\n\n"
        f"{if_grm.TO_BE_JUDGED_LABEL}\ntheory theory theory"
    )


def test_grm_config_env_endpoint_fallback(monkeypatch) -> None:
    monkeypatch.setenv("OPENOPD_GRM_ENDPOINTS", "http://a:1/, http://b:2")
    config = if_grm.GRMConfig.from_reward_kwargs({"enable_grm": True})
    assert config.endpoints == ["http://a:1", "http://b:2"]
    assert config.is_usable


def test_request_completion_disables_env_proxy_by_default(monkeypatch) -> None:
    trust_env_values = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"choices": [{"message": {"content": "<verdict>YES</verdict>"}}]}

    class FakeSession:
        def __init__(self) -> None:
            self.trust_env = True

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, *args, **kwargs):
            trust_env_values.append(self.trust_env)
            return FakeResponse()

    monkeypatch.setattr(if_grm.requests, "Session", FakeSession)

    config = if_grm.GRMConfig.from_reward_kwargs({"enable_grm": True, "grm_endpoints": ["http://x"]})
    text = if_grm._request_completion("http://x", [{"role": "user", "content": "hi"}], config)

    assert text == "<verdict>YES</verdict>"
    assert trust_env_values == [False]


def test_request_completion_can_opt_into_env_proxy(monkeypatch) -> None:
    trust_env_values = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"choices": [{"message": {"content": "<verdict>YES</verdict>"}}]}

    class FakeSession:
        def __init__(self) -> None:
            self.trust_env = False

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, *args, **kwargs):
            trust_env_values.append(self.trust_env)
            return FakeResponse()

    monkeypatch.setattr(if_grm.requests, "Session", FakeSession)

    config = if_grm.GRMConfig.from_reward_kwargs(
        {"enable_grm": True, "grm_endpoints": ["http://x"], "grm_use_env_proxy": True}
    )
    text = if_grm._request_completion("http://x", [{"role": "user", "content": "hi"}], config)

    assert text == "<verdict>YES</verdict>"
    assert trust_env_values == [True]


def test_grm_config_judges_all_families_by_default() -> None:
    config = if_grm.GRMConfig.from_reward_kwargs({"enable_grm": True, "grm_endpoints": ["http://x"]})
    assert config.families == ()  # empty => no family filter, every IF task is judged

    restricted = if_grm.GRMConfig.from_reward_kwargs(
        {"enable_grm": True, "grm_endpoints": ["http://x"], "grm_families": ["keywords", "other"]}
    )
    assert restricted.families == ("keywords", "other")


def test_grm_keeps_think_bonus_on_semantic_fail(monkeypatch) -> None:
    # think OK + rule OK + semantic FAIL: by default the task reward (0.8) is removed
    # but the independent think-format bonus (0.2) is kept.
    monkeypatch.setattr(if_grm, "judge_semantic_pass", lambda *a, **k: False)

    results = compute_score_batch(
        data_sources=["nemotron_if"],
        solution_strs=[_PASSING],
        ground_truths=[""],
        extra_infos=[dict(_KEYWORDS_EXTRA)],
        **_GRM_KWARGS,
    )

    assert len(results) == 1
    result = results[0]
    assert result["grm_called"] == 1.0
    assert result["grm_pass"] == 0.0
    assert result["grm_capped"] == 1.0
    assert result["score"] == 0.2
    assert result["grm_penalty"] == 0.8


def test_grm_caps_to_zero_when_no_think_format(monkeypatch) -> None:
    # no think + rule OK + semantic FAIL: think bonus is 0, so the floor is 0.0.
    monkeypatch.setattr(if_grm, "judge_semantic_pass", lambda *a, **k: False)

    results = compute_score_batch(
        data_sources=["nemotron_if"],
        solution_strs=[_PASSING_NO_THINK],
        ground_truths=[""],
        extra_infos=[dict(_KEYWORDS_EXTRA)],
        **_GRM_KWARGS,
    )

    result = results[0]
    assert result["grm_called"] == 1.0
    assert result["grm_capped"] == 1.0
    assert result["score"] == 0.0


def test_grm_hard_zero_when_keep_think_disabled(monkeypatch) -> None:
    # grm_keep_think_on_fail=False restores the aggressive hard-0 behavior.
    monkeypatch.setattr(if_grm, "judge_semantic_pass", lambda *a, **k: False)

    results = compute_score_batch(
        data_sources=["nemotron_if"],
        solution_strs=[_PASSING],
        ground_truths=[""],
        extra_infos=[dict(_KEYWORDS_EXTRA)],
        grm_keep_think_on_fail=False,
        **_GRM_KWARGS,
    )

    result = results[0]
    assert result["grm_capped"] == 1.0
    assert result["score"] == 0.0
    assert result["grm_penalty"] == 1.0


def test_grm_keeps_score_when_judge_passes(monkeypatch) -> None:
    monkeypatch.setattr(if_grm, "judge_semantic_pass", lambda *args, **kwargs: True)

    results = compute_score_batch(
        data_sources=["nemotron_if"],
        solution_strs=[_PASSING],
        ground_truths=[""],
        extra_infos=[dict(_KEYWORDS_EXTRA)],
        **_GRM_KWARGS,
    )

    result = results[0]
    assert result["grm_called"] == 1.0
    assert result["grm_pass"] == 1.0
    assert result["grm_capped"] == 0.0
    assert result["score"] == 1.0


def test_grm_not_called_when_rule_fails(monkeypatch) -> None:
    judged_answers: list[str] = []

    def fake_judge(prompt, answer, config, selector):
        judged_answers.append(answer)
        return False

    monkeypatch.setattr(if_grm, "judge_semantic_pass", fake_judge)

    results = compute_score_batch(
        data_sources=["nemotron_if"],
        solution_strs=[_FAILING],
        ground_truths=[""],
        extra_infos=[dict(_KEYWORDS_EXTRA)],
        **_GRM_KWARGS,
    )

    result = results[0]
    assert result["grm_called"] == 0.0
    assert judged_answers == []
    # Rule already failed the constraint, so it scores the format-only tier and is untouched.
    assert result["score"] == 0.2


def test_grm_skips_official_eval_rows(monkeypatch) -> None:
    def fake_official_eval_score(solution_str, extra_info, data_source):
        return {"score": 1.0, "base_score": 1.0, "scoring_mode": "official_eval", "family": "keywords"}

    monkeypatch.setattr(
        "verl.utils.reward_score.instruction_following._official_eval_score",
        fake_official_eval_score,
    )

    judged = []
    monkeypatch.setattr(
        if_grm, "judge_semantic_pass", lambda *a, **k: judged.append(1) or False
    )

    results = compute_score_batch(
        data_sources=["ifbench_test"],
        solution_strs=["answer"],
        ground_truths=[""],
        extra_infos=[
            {
                "dataset": "ifbench_test",
                "evaluator": "ifbench",
                "eval_prompt_for_scorer": "prompt",
                "metadata": '{"instruction_id_list": ["count:keywords_multiple"], "kwargs": []}',
            }
        ],
        **_GRM_KWARGS,
    )

    result = results[0]
    assert result["grm_called"] == 0.0
    assert judged == []
    assert result["score"] == 1.0


def test_grm_keys_aligned_across_mixed_batch(monkeypatch) -> None:
    monkeypatch.setattr(if_grm, "judge_semantic_pass", lambda *a, **k: False)

    results = compute_score_batch(
        data_sources=["nemotron_if", "nemotron_if"],
        solution_strs=[_PASSING, _FAILING],
        ground_truths=["", ""],
        extra_infos=[dict(_KEYWORDS_EXTRA), dict(_KEYWORDS_EXTRA)],
        **_GRM_KWARGS,
    )

    assert len(results) == 2
    for result in results:
        assert _GRM_KEYS.issubset(result.keys())
    # First (rule-pass) was judged and capped, second (rule-fail) was skipped.
    assert results[0]["grm_called"] == 1.0
    assert results[1]["grm_called"] == 0.0


def test_grm_judges_previously_exempt_family_by_default(monkeypatch) -> None:
    monkeypatch.setattr(if_grm, "judge_semantic_pass", lambda *a, **k: False)

    extra = dict(_KEYWORDS_EXTRA)
    extra["family"] = "length_constraints"  # used to be exempt; now judged by default

    results = compute_score_batch(
        data_sources=["nemotron_if"],
        solution_strs=[_PASSING],
        ground_truths=[""],
        extra_infos=[extra],
        **_GRM_KWARGS,
    )

    assert results[0]["grm_called"] == 1.0
    assert results[0]["grm_capped"] == 1.0
    assert results[0]["score"] == 0.2  # think bonus retained


def test_grm_respects_explicit_family_restriction(monkeypatch) -> None:
    monkeypatch.setattr(if_grm, "judge_semantic_pass", lambda *a, **k: False)

    extra = dict(_KEYWORDS_EXTRA)
    extra["family"] = "length_constraints"

    results = compute_score_batch(
        data_sources=["nemotron_if"],
        solution_strs=[_PASSING],
        ground_truths=[""],
        extra_infos=[extra],
        grm_families=["keywords"],  # restrict to keywords -> length_constraints is skipped
        **_GRM_KWARGS,
    )

    assert results[0]["grm_called"] == 0.0
    assert results[0]["score"] == 1.0


def test_grm_fail_mode_open_vs_closed(monkeypatch) -> None:
    monkeypatch.setattr(if_grm, "judge_semantic_pass", lambda *a, **k: None)  # transport failure

    open_results = compute_score_batch(
        data_sources=["nemotron_if"],
        solution_strs=[_PASSING],
        ground_truths=[""],
        extra_infos=[dict(_KEYWORDS_EXTRA)],
        **_GRM_KWARGS,
        grm_fail_mode="open",
    )
    assert open_results[0]["grm_error"] == 1.0
    assert open_results[0]["grm_capped"] == 0.0
    assert open_results[0]["score"] == 1.0

    # keep_think_on_fail disabled so the cap is the absolute floor (0.0), isolating fail_mode.
    closed_results = compute_score_batch(
        data_sources=["nemotron_if"],
        solution_strs=[_PASSING],
        ground_truths=[""],
        extra_infos=[dict(_KEYWORDS_EXTRA)],
        **_GRM_KWARGS,
        grm_fail_mode="closed",
        grm_keep_think_on_fail=False,
    )
    assert closed_results[0]["grm_error"] == 1.0
    assert closed_results[0]["grm_capped"] == 1.0
    assert closed_results[0]["score"] == 0.0
