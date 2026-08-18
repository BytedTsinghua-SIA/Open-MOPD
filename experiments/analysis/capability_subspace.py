"""Build per-layer capability subspaces from cascade stage task vectors.

Implements the basis construction that DACSP (CARE-RL) applies before gating
optimizer updates, and that Modular Gradient Surgery needs to identify conflict
modules. Both require stage boundaries, which is why the cascade line exists.

Pipeline, per parameter tensor:

  1. Task vector for stage d:  Delta_d = theta_d - theta_{d-1}
  2. Thin SVD of the reshaped 2-D matrix, keep the fewest leading components
     reaching ``energy`` (0.95 in the paper). Each retained pair (u_i, v_i)
     defines a rank-one direction B_i = u_i v_i^T.
  3. Vectorise every direction from all prior stages, concatenate, and run a
     rank-revealing QR. Drop columns whose R diagonal falls under tolerance.
  4. QR can flip signs, so re-orient each basis vector against its pivoted
     source direction; otherwise "aligned" and "opposing" lose their meaning,
     which is the whole point of the alpha+/alpha- split.

The paper's checkpoint-surgery control matters for interpretation: the top
subspace alone retained 91.0% of a single-domain gain while an equal-rank random
subspace retained 4.9%. Any use of this basis should carry that control.
"""
from __future__ import annotations

import torch


def load_merged_hf(path: str) -> dict[str, torch.Tensor]:
    """Load a ``merged_hf`` state dict, handling both layouts this project emits.

    The merge backend produces a single ``model.safetensors`` for some runs and a
    sharded ``model-0000N-of-0000M.safetensors`` set with a
    ``model.safetensors.index.json`` for others -- MixSFT-e4 is single-file while
    the IF teacher is sharded across two. Assuming either layout breaks on the
    other, so resolve it from what is actually on disk.
    """
    import json as _json
    import os as _os

    from safetensors.torch import load_file

    single = _os.path.join(path, "model.safetensors")
    index = _os.path.join(path, "model.safetensors.index.json")

    if _os.path.exists(single):
        return load_file(single)

    if not _os.path.exists(index):
        raise FileNotFoundError(
            f"{path} has neither model.safetensors nor model.safetensors.index.json")

    with open(index) as fh:
        weight_map = _json.load(fh)["weight_map"]
    shards = sorted(set(weight_map.values()))
    missing = [s for s in shards if not _os.path.exists(_os.path.join(path, s))]
    if missing:
        raise FileNotFoundError(f"index lists shards not on disk: {missing}")

    out: dict[str, torch.Tensor] = {}
    for shard in shards:
        out.update(load_file(_os.path.join(path, shard)))
    expected = set(weight_map)
    if set(out) != expected:
        raise ValueError(
            f"loaded {len(out)} tensors but index declares {len(expected)}")
    return out


def task_vector(theta_after: dict[str, torch.Tensor],
                theta_before: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Delta_d = theta_d - theta_{d-1}, restricted to keys present in both."""
    shared = sorted(set(theta_after) & set(theta_before))
    missing_after = sorted(set(theta_before) - set(theta_after))
    missing_before = sorted(set(theta_after) - set(theta_before))
    if missing_after or missing_before:
        raise ValueError(
            "checkpoints disagree on parameter names; refusing to guess. "
            f"only_before={missing_after[:3]} only_after={missing_before[:3]}"
        )
    out = {}
    for k in shared:
        a, b = theta_after[k], theta_before[k]
        if a.shape != b.shape:
            raise ValueError(f"{k}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
        out[k] = (a.float() - b.float())
    return out


def _as_matrix(t: torch.Tensor) -> torch.Tensor:
    """Reshape to 2-D for SVD. 1-D tensors (norms, biases) have no meaningful
    left/right factorisation, so callers should skip them rather than reshape."""
    if t.dim() < 2:
        raise ValueError(f"expected a >=2-D tensor, got {tuple(t.shape)}")
    return t.reshape(t.shape[0], -1)


def directions_for_tensor(delta: torch.Tensor, energy: float = 0.95,
                          max_rank: int | None = None) -> torch.Tensor:
    """Return the retained rank-one directions of one task-vector tensor.

    Output is ``[n_dirs, numel]``: each row is a flattened ``u_i v_i^T``, so the
    caller can concatenate directions across stages and QR them together.
    """
    if not 0 < energy <= 1:
        raise ValueError(f"energy must be in (0, 1], got {energy}")
    mat = _as_matrix(delta)
    if torch.count_nonzero(mat) == 0:
        return mat.new_zeros((0, mat.numel()))

    u, s, vh = torch.linalg.svd(mat, full_matrices=False)
    total = (s ** 2).sum()
    if total <= 0:
        return mat.new_zeros((0, mat.numel()))
    # fewest leading components reaching the energy threshold
    cum = torch.cumsum(s ** 2, dim=0) / total
    keep = int(torch.searchsorted(cum, torch.tensor(energy, dtype=cum.dtype)).item()) + 1
    keep = min(keep, s.numel())
    if max_rank is not None:
        keep = min(keep, max_rank)

    dirs = []
    for i in range(keep):
        dirs.append(torch.outer(u[:, i], vh[i, :]).reshape(-1))
    return torch.stack(dirs, dim=0)


def orthonormal_basis(directions: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
    """Rank-revealing QR over stacked directions, sign-corrected.

    ``directions`` is ``[n_dirs, numel]``. Returns ``[rank, numel]`` with rows
    orthonormal, each row oriented to have non-negative inner product with the
    source direction it best represents. Without that re-orientation the sign of
    ``<delta, Q_i>`` is arbitrary and the aligned/opposing split is meaningless.
    """
    if directions.numel() == 0 or directions.shape[0] == 0:
        return directions.new_zeros((0, directions.shape[-1] if directions.dim() > 1 else 0))

    # QR wants columns as vectors
    a = directions.t()                      # [numel, n_dirs]
    q, r = torch.linalg.qr(a, mode="reduced")
    diag = torch.abs(torch.diagonal(r))
    if diag.numel() == 0:
        return directions.new_zeros((0, directions.shape[-1]))
    keep = diag > (tol * diag.max().clamp(min=torch.finfo(diag.dtype).tiny))
    q = q[:, keep]                          # [numel, rank]

    basis = q.t().contiguous()              # [rank, numel]
    # orient each basis row against the source direction it aligns with most
    if basis.shape[0]:
        sims = basis @ directions.t()       # [rank, n_dirs]
        best = sims.abs().argmax(dim=1)
        signs = torch.sign(sims[torch.arange(basis.shape[0]), best])
        signs[signs == 0] = 1.0
        basis = basis * signs.unsqueeze(1)
    return basis


# Vocabulary projections are excluded from the capability subspace. They are not
# part of the computation path the papers analyse -- CARE-RL takes its per-layer
# SVD over transformer weights, and the perturbation-theory diagnostic covers MLP
# neurons and attention coordinates only. They are also by far the largest
# matrices (embed_tokens is 128256x2048 on SmolLM3, versus 512x2048 for k_proj),
# so including them would dominate SVD cost for parameters that carry no
# capability direction.
_EXCLUDED_SUBSTRINGS = ("embed_tokens", "lm_head", "shared_head")


def is_subspace_parameter(name: str, tensor: torch.Tensor, min_dim: int = 2) -> bool:
    """Whether a parameter should contribute a capability direction."""
    if tensor.dim() < min_dim:
        return False
    return not any(s in name for s in _EXCLUDED_SUBSTRINGS)


def build_subspace(task_vectors: list[dict[str, torch.Tensor]],
                   energy: float = 0.95, tol: float = 1e-6,
                   min_dim: int = 2) -> dict[str, torch.Tensor]:
    """Per-parameter capability basis from one or more stage task vectors.

    Tensors with fewer than ``min_dim`` dimensions are skipped: the paper's
    coordinate proxy is axis-aligned and explicitly does not cover normalisation
    parameters, and a 1-D vector has no left/right singular structure.
    """
    if not task_vectors:
        raise ValueError("need at least one task vector")
    keys = set(task_vectors[0])
    for tv in task_vectors[1:]:
        keys &= set(tv)

    out = {}
    for k in sorted(keys):
        if not is_subspace_parameter(k, task_vectors[0][k], min_dim=min_dim):
            continue
        stacked = []
        for tv in task_vectors:
            d = directions_for_tensor(tv[k], energy=energy)
            if d.shape[0]:
                stacked.append(d)
        if not stacked:
            continue
        basis = orthonormal_basis(torch.cat(stacked, dim=0), tol=tol)
        if basis.shape[0]:
            out[k] = basis
    return out


def gate_update(update: torch.Tensor, basis: torch.Tensor,
                alpha_pos: float = 1.2, alpha_neg: float = 0.2) -> torch.Tensor:
    """DACSP's update modulation.

        delta_tilde = delta_perp + alpha_pos * sum_{s_i>0} s_i Q_i
                                 + alpha_neg * sum_{s_i<0} s_i Q_i

    The orthogonal residual passes through untouched so new capability can still
    be acquired. The paper's sweep is the informative part: alpha_neg = 1 (no
    suppression) pushes regression above doing nothing, while hard projection
    (0, 0) gives the lowest regression but the worst total, because
    over-protecting history blocks learning. Hence partial damping.
    """
    if alpha_pos < 0 or alpha_neg < 0:
        raise ValueError(f"alphas must be non-negative, got {alpha_pos}, {alpha_neg}")
    if basis.numel() == 0 or basis.shape[0] == 0:
        return update
    flat = update.reshape(-1).float()
    if flat.numel() != basis.shape[1]:
        raise ValueError(
            f"update has {flat.numel()} elements but basis expects {basis.shape[1]}")

    coeffs = basis @ flat                       # s_i = <delta, Q_i>
    pos = torch.clamp(coeffs, min=0.0)
    neg = torch.clamp(coeffs, max=0.0)
    parallel = coeffs @ basis                   # projection onto the subspace
    perp = flat - parallel
    gated = perp + alpha_pos * (pos @ basis) + alpha_neg * (neg @ basis)
    return gated.reshape(update.shape).to(update.dtype)


def shard_basis(basis: torch.Tensor, offset: int, length: int) -> torch.Tensor:
    """Slice a full-matrix basis to the element range one rank owns.

    The basis rows are flattened over the *whole* parameter matrix, because it is
    built offline from unsharded ``merged_hf`` checkpoints. Under FSDP the
    gradient at optimizer time is a sharded DTensor, so each rank must gate only
    its own slice. Columns map one-to-one onto flattened parameter elements, so
    the slice is just a column range -- provided the caller passes the same
    flattening order the DTensor uses.
    """
    if basis.dim() != 2:
        raise ValueError(f"basis must be [rank, numel], got {tuple(basis.shape)}")
    if offset < 0 or length < 0 or offset + length > basis.shape[1]:
        raise ValueError(
            f"shard [{offset}, {offset + length}) outside basis width {basis.shape[1]}")
    return basis[:, offset:offset + length]


def gate_update_sharded(
    local_update: torch.Tensor,
    local_basis: torch.Tensor,
    partial_coeffs: torch.Tensor | None = None,
    alpha_pos: float = 1.2,
    alpha_neg: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gate one rank's shard of an update, given the matching basis slice.

    Returns ``(gated_local_update, local_partial_coeffs)``.

    The projection coefficient ``s_i = <delta, Q_i>`` is a sum over elements, so
    it decomposes across shards: each rank computes its partial sum, the caller
    all-reduces the ``rank``-length vector, then passes the total back in as
    ``partial_coeffs`` to apply the gating locally. Only ``rank`` scalars cross
    the wire per parameter, rather than the whole gradient.

    Call it twice: once with ``partial_coeffs=None`` to obtain the local
    partials, then again with the reduced totals to get the gated shard.
    """
    if alpha_pos < 0 or alpha_neg < 0:
        raise ValueError(f"alphas must be non-negative, got {alpha_pos}, {alpha_neg}")
    flat = local_update.reshape(-1).float()
    if local_basis.numel() == 0 or local_basis.shape[0] == 0:
        return local_update, flat.new_zeros(0)
    if flat.numel() != local_basis.shape[1]:
        raise ValueError(
            f"local shard has {flat.numel()} elements but basis slice expects "
            f"{local_basis.shape[1]}")

    local_partial = local_basis @ flat
    if partial_coeffs is None:
        return local_update, local_partial

    if partial_coeffs.shape != local_partial.shape:
        raise ValueError(
            f"reduced coeffs {tuple(partial_coeffs.shape)} do not match "
            f"{tuple(local_partial.shape)}")

    coeffs = partial_coeffs.to(local_partial.dtype)
    pos = torch.clamp(coeffs, min=0.0)
    neg = torch.clamp(coeffs, max=0.0)
    parallel = coeffs @ local_basis
    perp = flat - parallel
    gated = perp + alpha_pos * (pos @ local_basis) + alpha_neg * (neg @ local_basis)
    return gated.reshape(local_update.shape).to(local_update.dtype), local_partial


def top_rank_retained(delta: torch.Tensor, rank: int) -> float:
    """Energy fraction captured by the leading ``rank`` singular directions.

    Computed from singular values alone. Materialising the directions costs
    ``O(rank * numel)`` -- for an 11008x2048 MLP delta at 20% budget that is
    409 x 22.5M floats, about 37 GB, which is what silently killed an earlier
    probe. The projected norm is just the sum of the leading squared singular
    values, so no outer product is needed.
    """
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    mat = _as_matrix(delta).float()
    s = torch.linalg.svdvals(mat)
    total = (s ** 2).sum()
    if total <= 0:
        return 0.0
    k = min(rank, s.numel())
    return float(((s[:k] ** 2).sum() / total).clamp(0.0, 1.0))


def random_coordinate_retained(delta: torch.Tensor, fraction: float,
                               generator: torch.Generator | None = None) -> float:
    """Energy fraction captured by a random subset of coordinates.

    This is the control the perturbation-theory paper actually uses -- random
    neurons at a matched budget -- rather than a random subspace of the matrix's
    singular space. It is also the only form that stays cheap at these sizes,
    since it needs no basis at all.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    flat = _as_matrix(delta).reshape(-1).float()
    total = flat.dot(flat)
    if total <= 0:
        return 0.0
    k = max(1, int(flat.numel() * fraction))
    idx = torch.randperm(flat.numel(), generator=generator)[:k]
    picked = flat[idx]
    return float((picked.dot(picked) / total).clamp(0.0, 1.0))


def top_rank_directions(delta: torch.Tensor, rank: int) -> torch.Tensor:
    """Leading ``rank`` singular directions as explicit flattened rank-one rows.

    Only for small tensors and for tests. Cost is ``O(rank * numel)``, so prefer
    :func:`top_rank_retained` when the question is just how much energy a budget
    captures.
    """
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    mat = _as_matrix(delta)
    if torch.count_nonzero(mat) == 0:
        return mat.new_zeros((0, mat.numel()))
    q = min(rank, min(mat.shape))
    u, s, v = torch.svd_lowrank(mat, q=q, niter=4)
    dirs = [torch.outer(u[:, i], v[:, i]).reshape(-1) for i in range(q)]
    return torch.stack(dirs, dim=0)


def random_directions(delta: torch.Tensor, rank: int,
                      generator: torch.Generator | None = None) -> torch.Tensor:
    """Equal-rank random orthonormal directions, the control the paper pairs its
    top-subspace measurement against. Without it a high retained fraction says
    nothing, since any large enough subspace retains a lot by construction."""
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    mat = _as_matrix(delta)
    numel = mat.numel()
    q = min(rank, min(mat.shape))
    raw = torch.randn(q, numel, generator=generator, dtype=mat.dtype)
    return orthonormal_basis(raw)


def retained_fraction(delta: torch.Tensor, basis: torch.Tensor) -> float:
    """Fraction of the task vector's squared norm the basis captures.

    ``|P_S delta|^2 / |delta|^2``. This is the cheap proxy for the paper's
    surgery experiment: it measures how much of the *displacement* lives in the
    subspace, not how much of the *gain* survives, which needs an eval.
    """
    flat = _as_matrix(delta).reshape(-1).float()
    denom = flat.dot(flat)
    if denom <= 0:
        return 0.0
    if basis.numel() == 0 or basis.shape[0] == 0:
        return 0.0
    coeffs = basis.float() @ flat
    return float((coeffs.dot(coeffs) / denom).clamp(0.0, 1.0))


def subspace_report(basis: dict[str, torch.Tensor]) -> dict[str, float]:
    """Summary for logging: how much of the parameter space the basis occupies."""
    if not basis:
        return {"tensors": 0, "total_rank": 0, "mean_rank": 0.0, "mean_frac": 0.0}
    ranks = [b.shape[0] for b in basis.values()]
    fracs = [b.shape[0] / b.shape[1] for b in basis.values()]
    return {
        "tensors": len(basis),
        "total_rank": int(sum(ranks)),
        "mean_rank": float(sum(ranks) / len(ranks)),
        "mean_frac": float(sum(fracs) / len(fracs)),
    }
