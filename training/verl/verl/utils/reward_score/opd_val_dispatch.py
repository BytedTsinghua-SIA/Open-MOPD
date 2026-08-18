"""OPD validation reward dispatcher.

OPD jobs set ``custom_reward_function`` so the verl reward manager has a rule-based
scorer for *validation* (training reward is the teacher log-prob under
``reward_mode=opd_kl``/``mt_opd`` and never goes through this). The historical default
hard-wired the math-only ``ttrl_math.reward_func`` for ALL val rows, so IF and code
val data_sources were scored by the math (think/boxed) scorer -> ``acc`` was always 0.

This dispatcher routes each row to its domain's verifiable scorer by ``data_source``,
mirroring ``verl.utils.reward_score.default_compute_score`` but:
  * covering ALL four IF data_sources (``ifeval``, ``ifbench_test``,
    ``ifbench_mt_ifbench``, ``ifbench_mt_ifeval``) via a startswith check — the upstream
    ``default_compute_score`` misses ``ifbench_test`` / ``ifbench_mt_ifbench``;
  * leaving math / unknown data_sources on the exact previous ``ttrl_math`` behavior so
    single-domain math OPD is unchanged.

Works for single-domain (math/code/IF) and MT-OPD (mixed) val alike.
"""
from __future__ import annotations

# Absolute imports: verl loads this file via spec_from_file_location("custom_module", ...)
# as a plain top-level module (NOT a package), so relative `from . import` would raise
# "attempted relative import with no known parent package". verl is on sys.path at runtime.
from verl.utils.reward_score import ttrl_math


def _is_if(ds: str) -> bool:
    return (
        "ifeval" in ds
        or ds.startswith("ifbench")
        or ds in ("nemotron_if", "ifbench", "instruction_following")
    )


def _is_code(ds: str) -> bool:
    return ds.startswith("livecodebench") or ds in ("livecodebench", "codecontests_lcb")


def reward_func(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    **kwargs,
):
    ds = str(data_source or "")
    if _is_if(ds):
        from verl.utils.reward_score import instruction_following

        res = instruction_following.compute_score(
            solution_str, ground_truth, extra_info, data_source, scoring_mode="official_eval"
        )
    elif _is_code(ds):
        from verl.utils.reward_score import rllm_code_reward

        res = rllm_code_reward.compute_score(
            data_source, solution_str, ground_truth, extra_info, scoring_mode="official_lcb"
        )
    else:
        # math + everything else: ttrl_math
        res = ttrl_math.reward_func(data_source, solution_str, ground_truth, extra_info)
    return _normalize(res)


def _normalize(res):
    """Collapse every domain scorer's output to a UNIFORM ``{score, acc}`` dict.

    The val reward manager appends EVERY key of the returned dict to a per-key list
    and ray_trainer asserts each key's list length == #samples (one list per key,
    must cover all samples or none). Different domain scorers return different aux
    keys (math: think_score/boxed_format_score; code: passed_tests; IF: constraint_ratio),
    so in MT-OPD's mixed val a key like ``think_score`` only appears for math rows ->
    AssertionError. Returning only the two keys present for ALL domains keeps per-row
    schema uniform; per-data_source ``val-core/<ds>/acc`` still works (driven by ``acc``).
    """
    if isinstance(res, dict):
        score = res.get("score", res.get("acc", 0.0))
        acc = res.get("acc", res.get("score", 0.0))
        return {"score": float(score), "acc": float(acc)}
    if isinstance(res, (int, float, bool)):
        return {"score": float(res), "acc": float(res)}
    try:
        return {"score": float(res[0]), "acc": float(res[0])}
    except Exception:
        return {"score": 0.0, "acc": 0.0}
