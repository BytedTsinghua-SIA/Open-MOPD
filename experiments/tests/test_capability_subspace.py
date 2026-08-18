"""Capability-subspace construction and DACSP update gating.

The properties worth pinning are the ones that make the aligned/opposing split
mean anything: the basis must actually span the task vector, its rows must be
oriented against their source directions (QR flips signs arbitrarily), and the
gating must reduce to identity when alpha_pos = alpha_neg = 1.
"""
import pytest
import torch

from experiments.analysis.capability_subspace import (
    build_subspace,
    gate_update_sharded,
    is_subspace_parameter,
    load_merged_hf,
    shard_basis,
    directions_for_tensor,
    gate_update,
    orthonormal_basis,
    random_coordinate_retained,
    random_directions,
    retained_fraction,
    subspace_report,
    top_rank_directions,
    top_rank_retained,
    task_vector,
)


def _sd(seed, shape=(6, 4)):
    g = torch.Generator().manual_seed(seed)
    return {"w": torch.randn(*shape, generator=g), "b": torch.randn(shape[0], generator=g)}


# ---------------- task vector ----------------

def test_task_vector_is_the_difference() -> None:
    before, after = _sd(1), _sd(2)
    tv = task_vector(after, before)
    torch.testing.assert_close(tv["w"], after["w"] - before["w"])


def test_task_vector_rejects_mismatched_checkpoints() -> None:
    before = _sd(1)
    after = _sd(2)
    del after["b"]
    with pytest.raises(ValueError, match="disagree on parameter names"):
        task_vector(after, before)


def test_task_vector_rejects_shape_change() -> None:
    before, after = _sd(1), _sd(2)
    after["w"] = torch.zeros(3, 3)
    with pytest.raises(ValueError, match="shape"):
        task_vector(after, before)


def test_identical_checkpoints_give_zero_task_vector() -> None:
    sd = _sd(3)
    tv = task_vector(sd, sd)
    assert torch.count_nonzero(tv["w"]) == 0


# ---------------- directions ----------------

def test_rank_one_delta_yields_exactly_one_direction() -> None:
    u = torch.randn(6, 1)
    v = torch.randn(1, 4)
    delta = u @ v
    dirs = directions_for_tensor(delta, energy=0.95)
    assert dirs.shape[0] == 1
    # and that direction reproduces the delta up to scale
    d = dirs[0].reshape(6, 4)
    cos = (d * delta).sum() / (d.norm() * delta.norm())
    assert abs(abs(cos.item()) - 1.0) < 1e-5


def test_higher_energy_keeps_more_directions() -> None:
    g = torch.Generator().manual_seed(4)
    delta = torch.randn(8, 8, generator=g)
    few = directions_for_tensor(delta, energy=0.5).shape[0]
    many = directions_for_tensor(delta, energy=0.999).shape[0]
    assert few <= many
    assert many > few or few == delta.shape[0]


def test_zero_delta_yields_no_directions() -> None:
    assert directions_for_tensor(torch.zeros(4, 3)).shape[0] == 0


def test_invalid_energy_rejected() -> None:
    with pytest.raises(ValueError, match="energy"):
        directions_for_tensor(torch.randn(3, 3), energy=1.5)


def test_one_dim_tensor_rejected() -> None:
    with pytest.raises(ValueError, match=">=2-D"):
        directions_for_tensor(torch.randn(5))


# ---------------- basis ----------------

def test_basis_rows_are_orthonormal() -> None:
    g = torch.Generator().manual_seed(5)
    dirs = torch.randn(4, 20, generator=g)
    basis = orthonormal_basis(dirs)
    gram = basis @ basis.t()
    torch.testing.assert_close(gram, torch.eye(basis.shape[0]), atol=1e-5, rtol=1e-5)


def test_basis_drops_duplicate_directions() -> None:
    d = torch.randn(1, 12)
    dirs = torch.cat([d, d * 2.0, d * -3.0], dim=0)   # all collinear
    basis = orthonormal_basis(dirs)
    assert basis.shape[0] == 1, "collinear directions must collapse to rank 1"


def test_basis_rows_are_oriented_with_their_source() -> None:
    """QR sign flips would make <delta, Q_i> arbitrary and destroy the
    aligned/opposing distinction DACSP depends on."""
    g = torch.Generator().manual_seed(6)
    dirs = torch.randn(3, 15, generator=g)
    basis = orthonormal_basis(dirs)
    sims = basis @ dirs.t()
    best = sims.abs().argmax(dim=1)
    chosen = sims[torch.arange(basis.shape[0]), best]
    assert (chosen >= 0).all(), "each basis row must align positively with its source"


def test_empty_directions_give_empty_basis() -> None:
    assert orthonormal_basis(torch.zeros(0, 10)).shape[0] == 0


# ---------------- build_subspace ----------------

def test_build_subspace_skips_one_dim_params() -> None:
    tv = task_vector(_sd(7), _sd(8))
    basis = build_subspace([tv])
    assert "w" in basis
    assert "b" not in basis, "1-D tensors have no left/right singular structure"


def test_build_subspace_accumulates_across_stages() -> None:
    tv1 = task_vector(_sd(9), _sd(10))
    tv2 = task_vector(_sd(11), _sd(12))
    one = build_subspace([tv1])["w"].shape[0]
    two = build_subspace([tv1, tv2])["w"].shape[0]
    assert two >= one, "adding a stage cannot shrink the capability subspace"


def test_build_subspace_needs_a_task_vector() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_subspace([])


def test_report_shape() -> None:
    tv = task_vector(_sd(13), _sd(14))
    rep = subspace_report(build_subspace([tv]))
    assert rep["tensors"] == 1
    assert rep["total_rank"] >= 1
    assert 0 < rep["mean_frac"] <= 1


# ---------------- gating ----------------

def test_unit_alphas_are_identity() -> None:
    """alpha_pos = alpha_neg = 1 must reproduce the raw update, so the feature is
    verifiably inert at its neutral setting."""
    g = torch.Generator().manual_seed(15)
    upd = torch.randn(6, 4, generator=g)
    basis = orthonormal_basis(torch.randn(3, 24, generator=g))
    out = gate_update(upd, basis, alpha_pos=1.0, alpha_neg=1.0)
    torch.testing.assert_close(out, upd, atol=1e-5, rtol=1e-5)


def test_hard_projection_removes_the_subspace_component() -> None:
    """alphas (0, 0) is the paper's over-protective extreme: it should leave only
    the orthogonal residual."""
    g = torch.Generator().manual_seed(16)
    upd = torch.randn(6, 4, generator=g)
    basis = orthonormal_basis(torch.randn(2, 24, generator=g))
    out = gate_update(upd, basis, alpha_pos=0.0, alpha_neg=0.0)
    residual_coeffs = basis @ out.reshape(-1)
    torch.testing.assert_close(residual_coeffs, torch.zeros_like(residual_coeffs),
                               atol=1e-5, rtol=1e-5)


def test_amplifies_aligned_and_damps_opposing() -> None:
    basis = torch.zeros(2, 4)
    basis[0, 0] = 1.0
    basis[1, 1] = 1.0
    upd = torch.tensor([[2.0, -3.0, 0.5, 0.0]])   # +2 along Q0, -3 along Q1
    out = gate_update(upd, basis, alpha_pos=1.2, alpha_neg=0.2).reshape(-1)
    assert out[0].item() == pytest.approx(2.0 * 1.2)    # aligned amplified
    assert out[1].item() == pytest.approx(-3.0 * 0.2)   # opposing damped
    assert out[2].item() == pytest.approx(0.5)          # orthogonal untouched


def test_orthogonal_residual_always_passes_through() -> None:
    basis = torch.zeros(1, 3)
    basis[0, 0] = 1.0
    upd = torch.tensor([[0.0, 7.0, -7.0]])
    for ap, an in ((1.2, 0.2), (0.0, 0.0), (3.0, 0.0)):
        out = gate_update(upd, basis, alpha_pos=ap, alpha_neg=an).reshape(-1)
        assert out[1].item() == pytest.approx(7.0)
        assert out[2].item() == pytest.approx(-7.0)


def test_empty_basis_is_a_noop() -> None:
    upd = torch.randn(3, 3)
    torch.testing.assert_close(gate_update(upd, torch.zeros(0, 9)), upd)


def test_negative_alpha_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        gate_update(torch.randn(2, 2), torch.eye(1, 4), alpha_pos=-1.0)


def test_shape_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="basis expects"):
        gate_update(torch.randn(2, 2), torch.eye(1, 99))


def test_gating_preserves_dtype_and_shape() -> None:
    upd = torch.randn(4, 5, dtype=torch.bfloat16)
    basis = orthonormal_basis(torch.randn(2, 20))
    out = gate_update(upd, basis)
    assert out.shape == upd.shape
    assert out.dtype == upd.dtype

# ---------------- sharded gating ----------------

def _simulate_sharded_gate(upd, basis, n_shards, alpha_pos=1.2, alpha_neg=0.2):
    """单进程模拟多 rank: 先各自算 partial, 求和(等价 all-reduce), 再各自 gate."""
    flat = upd.reshape(-1)
    numel = flat.numel()
    bounds = []
    step = (numel + n_shards - 1) // n_shards
    for r in range(n_shards):
        lo = min(r * step, numel)
        hi = min(lo + step, numel)
        bounds.append((lo, hi - lo))

    partials = []
    for lo, ln in bounds:
        lb = shard_basis(basis, lo, ln)
        _, p = gate_update_sharded(flat[lo:lo + ln], lb, None, alpha_pos, alpha_neg)
        partials.append(p)
    total = torch.stack(partials).sum(dim=0) if partials else torch.zeros(0)

    pieces = []
    for lo, ln in bounds:
        lb = shard_basis(basis, lo, ln)
        g, _ = gate_update_sharded(flat[lo:lo + ln], lb, total, alpha_pos, alpha_neg)
        pieces.append(g)
    return torch.cat(pieces).reshape(upd.shape)


@pytest.mark.parametrize("n_shards", [1, 2, 3, 4, 8])
def test_sharded_gating_equals_unsharded(n_shards: int) -> None:
    """The whole point of the shard-aware path: it must be numerically identical
    to gating the full tensor, or FSDP runs would silently diverge from tests."""
    g = torch.Generator().manual_seed(20 + n_shards)
    upd = torch.randn(6, 4, generator=g)
    basis = orthonormal_basis(torch.randn(3, 24, generator=g))
    want = gate_update(upd, basis)
    got = _simulate_sharded_gate(upd, basis, n_shards)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_sharded_unit_alphas_are_identity() -> None:
    g = torch.Generator().manual_seed(30)
    upd = torch.randn(5, 4, generator=g)
    basis = orthonormal_basis(torch.randn(2, 20, generator=g))
    got = _simulate_sharded_gate(upd, basis, 4, alpha_pos=1.0, alpha_neg=1.0)
    torch.testing.assert_close(got, upd, atol=1e-5, rtol=1e-5)


def test_partial_coefficients_sum_to_the_full_inner_product() -> None:
    """The all-reduce contract: per-shard partials must add up to <delta, Q_i>."""
    g = torch.Generator().manual_seed(31)
    upd = torch.randn(4, 5, generator=g)
    basis = orthonormal_basis(torch.randn(3, 20, generator=g))
    full = basis @ upd.reshape(-1)

    flat = upd.reshape(-1)
    partials = []
    for lo in range(0, 20, 5):
        _, p = gate_update_sharded(flat[lo:lo + 5], shard_basis(basis, lo, 5), None)
        partials.append(p)
    torch.testing.assert_close(torch.stack(partials).sum(dim=0), full, atol=1e-5, rtol=1e-5)


def test_shard_basis_rejects_out_of_range() -> None:
    basis = torch.zeros(2, 10)
    with pytest.raises(ValueError, match="outside basis width"):
        shard_basis(basis, 6, 8)


def test_shard_basis_slices_columns_only() -> None:
    basis = torch.arange(20, dtype=torch.float32).reshape(2, 10)
    sl = shard_basis(basis, 3, 4)
    assert sl.shape == (2, 4)
    torch.testing.assert_close(sl, basis[:, 3:7])


def test_sharded_empty_basis_is_a_noop() -> None:
    upd = torch.randn(2, 3)
    out, partial = gate_update_sharded(upd, torch.zeros(0, 6), None)
    torch.testing.assert_close(out, upd)
    assert partial.numel() == 0


def test_sharded_rejects_mismatched_shard_width() -> None:
    with pytest.raises(ValueError, match="basis slice expects"):
        gate_update_sharded(torch.randn(4), torch.zeros(2, 7), None)


def test_sharded_rejects_wrong_coeff_shape() -> None:
    basis = torch.zeros(3, 5)
    basis[0, 0] = 1.0
    with pytest.raises(ValueError, match="do not match"):
        gate_update_sharded(torch.randn(5), basis, torch.zeros(2))

# ---------------- merged_hf loading ----------------

def _write_single(tmp_path, sd):
    from safetensors.torch import save_file
    save_file(sd, str(tmp_path / "model.safetensors"))
    return str(tmp_path)


def _write_sharded(tmp_path, sd, n_shards=2):
    """Mimic the layout the merge backend emits for some runs: N shards plus an
    index. MixSFT-e4 is single-file while the IF teacher is sharded, so the
    loader has to cope with both."""
    import json as _json
    from safetensors.torch import save_file
    keys = sorted(sd)
    per = (len(keys) + n_shards - 1) // n_shards
    weight_map = {}
    for i in range(n_shards):
        chunk = keys[i * per:(i + 1) * per]
        if not chunk:
            continue
        fname = f"model-{i+1:05d}-of-{n_shards:05d}.safetensors"
        save_file({k: sd[k] for k in chunk}, str(tmp_path / fname))
        for k in chunk:
            weight_map[k] = fname
    with open(tmp_path / "model.safetensors.index.json", "w") as fh:
        _json.dump({"metadata": {}, "weight_map": weight_map}, fh)
    return str(tmp_path)


def test_loads_single_file_layout(tmp_path) -> None:
    sd = {"a": torch.randn(3, 4), "b": torch.randn(5)}
    got = load_merged_hf(_write_single(tmp_path, sd))
    assert sorted(got) == ["a", "b"]
    torch.testing.assert_close(got["a"], sd["a"])


def test_loads_sharded_layout(tmp_path) -> None:
    sd = {f"t{i}": torch.randn(2, 3) for i in range(6)}
    got = load_merged_hf(_write_sharded(tmp_path, sd, n_shards=3))
    assert sorted(got) == sorted(sd)
    for k in sd:
        torch.testing.assert_close(got[k], sd[k])


def test_single_file_takes_precedence(tmp_path) -> None:
    """If both layouts are present the single file is authoritative, so a stale
    index cannot silently shadow it."""
    from safetensors.torch import save_file
    import json as _json
    save_file({"a": torch.ones(2, 2)}, str(tmp_path / "model.safetensors"))
    save_file({"a": torch.zeros(2, 2)}, str(tmp_path / "model-00001-of-00001.safetensors"))
    with open(tmp_path / "model.safetensors.index.json", "w") as fh:
        _json.dump({"weight_map": {"a": "model-00001-of-00001.safetensors"}}, fh)
    got = load_merged_hf(str(tmp_path))
    torch.testing.assert_close(got["a"], torch.ones(2, 2))


def test_missing_both_layouts_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="neither"):
        load_merged_hf(str(tmp_path))


def test_index_listing_absent_shard_raises(tmp_path) -> None:
    import json as _json
    with open(tmp_path / "model.safetensors.index.json", "w") as fh:
        _json.dump({"weight_map": {"a": "model-00001-of-00002.safetensors"}}, fh)
    with pytest.raises(FileNotFoundError, match="shards not on disk"):
        load_merged_hf(str(tmp_path))


def test_round_trip_through_task_vector(tmp_path) -> None:
    """The loader's output must feed task_vector directly, which is the whole
    reason it exists."""
    before = {"w": torch.zeros(4, 3), "n": torch.zeros(4)}
    after = {"w": torch.ones(4, 3), "n": torch.ones(4)}
    p1 = tmp_path / "before"; p1.mkdir()
    p2 = tmp_path / "after"; p2.mkdir()
    tv = task_vector(load_merged_hf(_write_sharded(p2, after)),
                     load_merged_hf(_write_single(p1, before)))
    torch.testing.assert_close(tv["w"], torch.ones(4, 3))
    basis = build_subspace([tv])
    assert "w" in basis and "n" not in basis

# ---------------- parameter filtering ----------------

def test_vocab_projections_excluded() -> None:
    """embed_tokens on SmolLM3 is 128256x2048 against 512x2048 for k_proj, so
    including it would dominate SVD cost -- and it is a vocabulary projection,
    not part of the computation path the papers analyse."""
    w = torch.randn(4, 4)
    assert not is_subspace_parameter("model.embed_tokens.weight", w)
    assert not is_subspace_parameter("lm_head.weight", w)
    assert is_subspace_parameter("model.layers.0.self_attn.k_proj.weight", w)
    assert is_subspace_parameter("model.layers.0.mlp.gate_proj.weight", w)


def test_one_dim_params_excluded_by_the_same_predicate() -> None:
    assert not is_subspace_parameter("model.layers.0.input_layernorm.weight",
                                     torch.randn(8))


def test_build_subspace_skips_excluded_names() -> None:
    tv = {
        "model.embed_tokens.weight": torch.randn(6, 4),
        "model.layers.0.mlp.up_proj.weight": torch.randn(6, 4),
        "lm_head.weight": torch.randn(6, 4),
    }
    basis = build_subspace([tv])
    assert list(basis) == ["model.layers.0.mlp.up_proj.weight"]

# ---------------- checkpoint-surgery control ----------------

def test_top_rank_beats_random_on_a_low_rank_delta() -> None:
    """The decisive control. On a genuinely low-rank delta the top directions must
    capture nearly everything while an equal-rank random subspace captures little;
    without this contrast a high retained fraction means nothing."""
    g = torch.Generator().manual_seed(40)
    u = torch.randn(64, 3, generator=g)
    v = torch.randn(3, 48, generator=g)
    delta = u @ v                                    # rank 3 by construction

    top = retained_fraction(delta, top_rank_directions(delta, 3))
    rnd = retained_fraction(delta, random_directions(delta, 3, generator=g))
    assert top > 0.95, top
    assert rnd < 0.2, rnd
    assert top > 5 * rnd


def test_top_rank_and_random_converge_on_a_full_rank_delta() -> None:
    """The failure mode we measured on real checkpoints: when the delta is
    high-rank, a fixed small budget retains little either way, so the method has
    no signal to exploit."""
    g = torch.Generator().manual_seed(41)
    delta = torch.randn(64, 64, generator=g)         # full rank
    budget = 3
    top = retained_fraction(delta, top_rank_directions(delta, budget))
    rnd = retained_fraction(delta, random_directions(delta, budget, generator=g))
    assert top < 0.35, top
    assert top > rnd                                  # still ordered correctly
    assert top - rnd < 0.35                           # but the gap is small


def test_retained_fraction_is_one_for_a_complete_basis() -> None:
    g = torch.Generator().manual_seed(42)
    delta = torch.randn(8, 6, generator=g)
    full = top_rank_directions(delta, min(delta.shape))
    assert retained_fraction(delta, full) == pytest.approx(1.0, abs=1e-3)


def test_retained_fraction_is_zero_for_an_empty_basis() -> None:
    assert retained_fraction(torch.randn(4, 4), torch.zeros(0, 16)) == 0.0


def test_retained_fraction_of_zero_delta_is_zero() -> None:
    assert retained_fraction(torch.zeros(4, 4), torch.eye(2, 16)) == 0.0


def test_retained_fraction_is_monotone_in_rank() -> None:
    g = torch.Generator().manual_seed(43)
    delta = torch.randn(32, 24, generator=g)
    prev = -1.0
    for r in (1, 2, 4, 8, 16):
        f = retained_fraction(delta, top_rank_directions(delta, r))
        assert f >= prev - 1e-4, (r, f, prev)
        prev = f


def test_random_directions_are_orthonormal() -> None:
    g = torch.Generator().manual_seed(44)
    b = random_directions(torch.randn(16, 12, generator=g), 4, generator=g)
    gram = b @ b.t()
    torch.testing.assert_close(gram, torch.eye(b.shape[0]), atol=1e-4, rtol=1e-4)


def test_rank_budget_clamps_to_matrix_dimension() -> None:
    delta = torch.randn(5, 3)
    assert top_rank_directions(delta, 99).shape[0] == 3
    assert random_directions(delta, 99).shape[0] <= 3


def test_zero_delta_yields_no_top_directions() -> None:
    assert top_rank_directions(torch.zeros(4, 4), 2).shape[0] == 0


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_rank_rejected(bad: int) -> None:
    with pytest.raises(ValueError, match="rank must be positive"):
        top_rank_directions(torch.randn(3, 3), bad)
    with pytest.raises(ValueError, match="rank must be positive"):
        random_directions(torch.randn(3, 3), bad)

# ---------------- cheap retained-energy path ----------------

def test_top_rank_retained_matches_the_explicit_projection() -> None:
    """The cheap singular-value form must agree with materialising the basis,
    since the explicit path is what the unit tests reason about."""
    g = torch.Generator().manual_seed(50)
    delta = torch.randn(20, 14, generator=g)
    for r in (1, 3, 7, 14):
        cheap = top_rank_retained(delta, r)
        explicit = retained_fraction(delta, top_rank_directions(delta, r))
        # top_rank_directions uses randomised svd_lowrank while top_rank_retained
        # uses exact svdvals, so the cheap path is the more accurate of the two
        # and the explicit one sits slightly below it. Agreement to ~1e-2 is the
        # substantive claim; bit-equality is not available here.
        assert cheap == pytest.approx(explicit, abs=2e-2), (r, cheap, explicit)
        # and the gap always has this sign: measured 0.0127 at r=1 down to 0.0
        # at full rank, since svd_lowrank's approximation can only lose energy
        assert cheap >= explicit - 1e-6, "exact projection cannot capture less"


def test_top_rank_retained_is_one_at_full_rank() -> None:
    g = torch.Generator().manual_seed(51)
    delta = torch.randn(9, 6, generator=g)
    assert top_rank_retained(delta, 6) == pytest.approx(1.0, abs=1e-5)


def test_top_rank_retained_on_low_rank_delta_saturates_early() -> None:
    g = torch.Generator().manual_seed(52)
    delta = torch.randn(40, 3, generator=g) @ torch.randn(3, 30, generator=g)
    assert top_rank_retained(delta, 3) > 0.999


def test_top_rank_retained_zero_delta() -> None:
    assert top_rank_retained(torch.zeros(5, 5), 2) == 0.0


def test_random_coordinate_retained_tracks_the_budget_when_energy_is_spread() -> None:
    """For an isotropic delta a fraction f of coordinates holds about f of the
    energy, which is the baseline any structured selection must beat."""
    g = torch.Generator().manual_seed(53)
    delta = torch.randn(100, 100, generator=g)
    for f in (0.05, 0.25, 0.5):
        got = random_coordinate_retained(delta, f, generator=g)
        assert got == pytest.approx(f, abs=0.05), (f, got)


def test_random_coordinate_retained_full_budget_is_one() -> None:
    assert random_coordinate_retained(torch.randn(6, 6), 1.0) == pytest.approx(1.0, abs=1e-5)


def test_random_coordinate_retained_zero_delta() -> None:
    assert random_coordinate_retained(torch.zeros(4, 4), 0.5) == 0.0


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_invalid_fraction_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="fraction must be"):
        random_coordinate_retained(torch.randn(3, 3), bad)


def test_cheap_path_handles_a_tall_matrix_without_materialising() -> None:
    """The shape that broke the explicit path: a 20% budget here would be ~37 GB
    of outer products, while the singular-value form is bounded by the matrix."""
    g = torch.Generator().manual_seed(54)
    delta = torch.randn(2000, 256, generator=g)
    f = top_rank_retained(delta, int(256 * 0.2))
    assert 0.0 < f < 1.0
