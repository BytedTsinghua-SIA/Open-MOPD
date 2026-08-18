import torch

from verl.experimental.agent_loop.agent_loop import (
    _compact_validation_prompt_contract,
    _resolve_response_length,
)


def test_training_keeps_configured_response_length():
    assert (
        _resolve_response_length(
            8192,
            [{"max_tokens": 30_000}, {"max_tokens": 10_000}],
            is_validation=False,
        )
        == 8192
    )


def test_validation_uses_largest_request_width_for_mixed_suite():
    assert (
        _resolve_response_length(
            8192,
            [{"max_tokens": 30_000}, {"max_tokens": 10_000}, {}],
            is_validation=True,
        )
        == 30_000
    )


def test_validation_worker_uses_cluster_wide_profile_width():
    """A Code-only worker chunk must still match an AIME worker's 31K width."""

    assert (
        _resolve_response_length(
            8192,
            [{"max_tokens": 30_000}],
            is_validation=True,
            validation_profile_sampling_params=[
                {"max_tokens": 31_000},
                {"max_tokens": 30_000},
                {"max_tokens": 10_000},
            ],
        )
        == 31_000
    )


def test_training_ignores_validation_profile_width():
    assert (
        _resolve_response_length(
            8192,
            [{"max_tokens": 30_000}],
            is_validation=False,
            validation_profile_sampling_params=[{"max_tokens": 31_000}],
        )
        == 8192
    )


def _prompt_contract(prompt_width: int, response_width: int):
    prompt_ids = torch.arange(prompt_width).unsqueeze(0)
    response_ids = torch.arange(10_000, 10_000 + response_width).unsqueeze(0)
    input_ids = torch.cat([prompt_ids, response_ids], dim=-1)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(prompt_width + response_width).unsqueeze(0)
    return prompt_ids, response_ids, input_ids, attention_mask, position_ids


def test_long_validation_prompt_is_compacted_for_returned_batch_contract():
    prompt_ids, response_ids, input_ids, attention_mask, position_ids = _prompt_contract(2155, 7)

    compact = _compact_validation_prompt_contract(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        configured_prompt_length=2048,
        is_validation=True,
    )

    compact_prompt_ids, compact_input_ids, compact_attention_mask, compact_position_ids = compact
    assert compact_prompt_ids.shape == (1, 2048)
    assert compact_input_ids.shape == compact_attention_mask.shape == compact_position_ids.shape == (1, 2055)
    assert torch.equal(compact_prompt_ids, prompt_ids[:, -2048:])
    assert torch.equal(compact_input_ids[:, -7:], response_ids)
    assert torch.equal(compact_position_ids[:, -7:], position_ids[:, -7:])


def test_long_validation_prompt_preserves_multidimensional_position_layout():
    prompt_ids, response_ids, input_ids, attention_mask, position_ids = _prompt_contract(2155, 7)
    position_ids = position_ids.unsqueeze(1).repeat(1, 3, 1)

    _, _, _, compact_position_ids = _compact_validation_prompt_contract(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        configured_prompt_length=2048,
        is_validation=True,
    )

    assert compact_position_ids.shape == (1, 3, 2055)
    assert torch.equal(compact_position_ids[..., -7:], position_ids[..., -7:])


def test_long_training_prompt_contract_is_unchanged():
    prompt_ids, response_ids, input_ids, attention_mask, position_ids = _prompt_contract(2155, 7)

    unchanged = _compact_validation_prompt_contract(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        configured_prompt_length=2048,
        is_validation=False,
    )

    expected = (prompt_ids, input_ids, attention_mask, position_ids)
    assert all(actual is original for actual, original in zip(unchanged, expected, strict=True))
