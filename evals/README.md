# OpenOPD Evals

This directory is the evaluation layer for OpenOPD. It is split into two
independent stages:

- `rollout_engine`: offline vLLM generation from one or more parquet files.
- `verifier`: scoring and aggregation over rollout outputs.

## Dataset Coverage

The evaluation layer covers the following benchmark inputs:

| Domain | Dataset | Rows | Repeats | Rollout rows |
|---|---|---:|---:|---:|
| math | `aime24` | 30 | 32 | 960 |
| math | `aime25` | 30 | 32 | 960 |
| code | `humaneval_plus` | 164 | 10 | 1640 |
| code | `mbpp_plus` | 378 | 10 | 3780 |
| code | `livecodebench_v6` | 175 | 1 | 175 |
| instruction | `ifeval` | 541 | 1 | 541 |
| instruction | `ifbench_test` | 300 | 1 | 300 |
| instruction | `ifbench_mt_ifbench` | 1387 | 1 | 1387 |
| instruction | `ifbench_mt_ifeval` | 1774 | 1 | 1774 |

The `data/` directory is for generated parquet inputs. Each parquet is already
expanded by `repeat_idx`, so a rollout engine can treat every row as one
generation request.

## Build Parquet Inputs

```bash
python -m evals.rollout_engine.build_data --dataset aime24 --source-root /path/to/datasets
python -m evals.rollout_engine.build_data --dataset all --source-root /path/to/datasets
```

Expected source layout:

- `aime24/data/train-00000-of-00001.parquet`
- `aime25/test.jsonl`
- `ifeval/ifeval_input_data.jsonl`
- `allenai/IFBench_test/data/train-00000-of-00001.parquet`
- `allenai/IFBench_multi-turn/ifbench_constraints/test-00000-of-00001.parquet`
- `allenai/IFBench_multi-turn/ifeval_constraints/test-00000-of-00001.parquet`

EvalPlus and LiveCodeBench loaders are package-backed. Install their official
packages/checkouts before building those parquet files.

Verifier-side Python dependencies used in this repo are listed in:

```text
evals/verifier/requirements.txt
```

Install them with:

```bash
python -m pip install -r evals/verifier/requirements.txt
```

LiveCodeBench v6 follows the patched local-asset flow from `ModelMerging`.
Expected local asset directory:

```text
evals/data/benchmark_assets/livecodebench/code_generation_lite/
  test.jsonl
  test2.jsonl
  test3.jsonl
  test4.jsonl
  test5.jsonl
  test6.jsonl
```

You can fetch them from ModelScope with:

```bash
python -m evals.misc.fetch_livecodebench_assets
```

## Third-Party Verifiers

Pinned third-party repos and patches live under:

```text
evals/verifier/third_party/repos.lock.json
evals/verifier/third_party/patches/
evals/verifier/score_functions/instruction_following/google_ifeval/
```

Sync the official repos with:

```bash
python -m evals.verifier.scripts.manage_third_party sync --repo evalplus
python -m evals.verifier.scripts.manage_third_party sync --repo ifbench
python -m evals.verifier.scripts.manage_third_party sync --repo livecodebench
python -m evals.verifier.scripts.manage_third_party apply-patches --repo livecodebench
```

`evalplus` and `ifbench` are used at fixed commits without local patching.
`livecodebench` requires the local patch to load offline `code_generation_lite`
assets.

## Run Offline vLLM Rollout

```bash
python -m evals.rollout_engine.vllm_rollout \
  --model /path/to/checkpoint \
  --input evals/data/parquet/aime24.parquet \
  --output-dir results/evals/run_001 \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 2048 \
  --max-num-batched-tokens 8192
```

The default scheduler caps are `max_num_seqs=2048` and
`max_num_batched_tokens=8192`. Override them when changing model length, GPU
count, or memory targets.

## Aggregate Scores

```bash
python -m evals.verifier.score --rollout results/evals/run_001/aime24_rollouts.parquet
```

The verifier writes JSON when `--output` is provided. Each scored dataset reports
two rates:

- `official`: correct rollout rows divided by the complete dataset parquet size.
- `live`: correct rollout rows divided by the rows actually present in this run.

AIME scoring uses the answer-extraction baseline implemented in this repository.
Code and instruction benchmarks must use their official verifier packages:
EvalPlus, LiveCodeBench, Google IFEval, and AllenAI IFBench.

## Shell Entrypoints

All scripts take a read-only HF model path and a read-write output path:

```bash
MODEL_PATH=/path/to/hf_model OUTPUT_ROOT=/path/to/output \
  bash evals/rollout_engine/scripts/run_all.sh
```

Useful overrides:

- `DATA_ROOT`: downloaded source data, default `evals/data/source`.
- `PARQUET_DIR`: generated parquet inputs, default `evals/data/parquet`.
- `ROLLOUT_LIMIT_BASE`: optional base problems per dataset to run, e.g. `100`;
  this does not modify the complete parquet files.
- `TENSOR_PARALLEL_SIZE`: vLLM tensor parallel size.
- `GPU_MEMORY_UTILIZATION`: default `0.9`; lower it on busy GPUs.
- `OVERRIDE_MAX_TOKENS`: smoke-test cap that overrides per-row max tokens.
