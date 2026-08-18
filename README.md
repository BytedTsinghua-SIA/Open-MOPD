<h1 align="center">Open-MOPD: Diagnosing and Fixing Capability Imbalance in Multi-Teacher On-Policy Distillation</h1>

<p align="center">
  <b>An open recipe for multi-teacher on-policy distillation.</b>
</p>

<p align="center">
  Huan-ang Gao<sup>*,1,2,3</sup> · Haohan Chi<sup>*,1,2,3</sup> · Yong Yan<sup>1,2,3</sup> · Shiyuan Feng<sup>1,2</sup> · Hanlin Wu<sup>1,2</sup><br>
  Zheng Jiang<sup>3</sup> · Bingxiang He<sup>3</sup> · Wei-Ying Ma<sup>1,2</sup> · Ya-Qin Zhang<sup>1,2</sup> · Hao Zhou<sup>1,2,†</sup>
</p>

<p align="center">
  <sup>1</sup>SIA-Lab of Tsinghua AIR and ByteDance Seed<br>
  <sup>2</sup>Institute for AI Industry Research (AIR), Tsinghua University<br>
  <sup>3</sup>Department of Computer Science and Technology, Tsinghua University
</p>

<p align="center">
  * Equal contribution · ‡ Project Lead · † Corresponding author
</p>

<p align="center">
  <a href="https://bytedtsinghua-sia.github.io/Open-MOPD/">
    <img src="https://img.shields.io/badge/Project-Page-0A8AA0?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page">
  </a>
  <a href="https://huggingface.co/collections/BytedTsinghua-SIA/open-mopd-multi-teacher-on-policy-distillation">
    <img src="https://img.shields.io/badge/Hugging%20Face-Models%20%26%20Data-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face">
  </a>
</p>

**Open-MOPD is an open recipe for multi-teacher on-policy distillation
(M-OPD/MOPD).** It specifies the complete path from a public base model to one
unified student: mixed-domain supervised fine-tuning, three independently
trained domain RL teachers, and multi-teacher on-policy distillation. We release
the training and evaluation code, model checkpoints, datasets, and the settings
needed to reproduce the pipeline.

Open-MOPD uses known domain labels to study capability integration separately
from teacher routing. In this controlled setting, Naive M-OPD remains 3.50
points below the domain-routed RouteOPD reference. Open-MOPD traces this gap to
an imbalanced optimization budget across domains and reduces it from **3.50**
to **0.31** points in one shared student.

## Highlights

- **End-to-end open recipe.** The release covers MixSFT, three domain RL
  teachers, the final Open-MOPD student, training data, evaluation data, and
  runnable code for every stage.
- **Optimization-budget diagnosis.** The analysis connects capability imbalance
  to realized response-token share, changing teacher-student gaps, and stale
  rewards during repeated updates.
- **Balanced consolidation.** Open-MOPD controls the shared student's
  optimization budget across math, code, and instruction following without
  requiring a larger model at deployment.
- **Strong capability integration.** Open-MOPD reduces the integration gap from
  **3.50** to **0.31** points and raises RouteRL headroom recovery from **35.6%**
  to **83.4%**.

## Results

Scores are macro-averaged within each domain and then across math, code, and
instruction following. RouteRL and RouteOPD select separate domain models using
the domain label; they are references rather than single deployable models.

| Method | Math | Code | IF | Total | Recovery |
|---|---:|---:|---:|---:|---:|
| MixSFT (epoch 4) | 17.95 | 17.60 | 41.46 | 25.67 | — |
| Naive M-OPD | 21.26 | 19.26 | 43.64 | 28.05 | 35.6% |
| RouteOPD | 23.15 | 21.71 | 49.80 | 31.55 | 88.0% |
| **Open-MOPD** | **22.42** | **21.73** | **49.58** | **31.24** | **83.4%** |
| RouteRL | 24.24 | 21.73 | 51.08 | 32.35 | 100% |

The evaluation uses AIME24/AIME25 (mean@64), LiveCodeBench v5/v6 (mean@10),
and IFEval/IFBench_test (mean@1). See the paper and released data card for the
complete protocol.

## Released Artifacts

The [Hugging Face collection](https://huggingface.co/collections/BytedTsinghua-SIA/open-mopd-multi-teacher-on-policy-distillation)
contains the complete five-model family and the training/evaluation data:

- [MixSFT initialization](https://huggingface.co/BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-MixSFT)
- [Math RL teacher](https://huggingface.co/BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-RL-Math)
- [Code RL teacher](https://huggingface.co/BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-RL-Code)
- [Instruction-following RL teacher](https://huggingface.co/BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-RL-IF)
- [Open-MOPD final student](https://huggingface.co/BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-Final)
- [Training and evaluation data](https://huggingface.co/datasets/BytedTsinghua-SIA/Open-MOPD-Data)

## Quick Start

Install the project dependencies:

```bash
cd training
bash install_requirements.sh
cd ..
```

The local launchers print the generated command without executing it. The
following command previews an Open-MOPD training run:

```bash
bash scripts/local/mt_opd.sh \
  --model /path/to/mixsft \
  --teacher /path/to/math-teacher \
  --teacher /path/to/code-teacher \
  --teacher /path/to/if-teacher \
  --domains math,code,if \
  --train /path/to/rl_prompt_mix/train.parquet \
  --val /path/to/validation.parquet \
  --output /path/to/runs/open-mopd
```

Review the printed command and add `--run` to execute it. See
[`scripts/local/README.md`](scripts/local/README.md) for the full configuration,
environment setup, and launcher options.

The repository also provides local entry points for the other pipeline stages:

```bash
bash scripts/local/sft.sh --help
bash scripts/local/rl.sh --help
bash scripts/local/opd.sh --help
bash scripts/local/eval.sh --help
```

## Repository Layout

```text
scripts/local/   Local SFT, RL, OPD, M-OPD, and evaluation launchers
training/verl/   Patched verl training framework and Open-MOPD implementation
training/scripts/ Dataset preparation utilities for SFT and domain RL
evals/           Offline rollout generation, official verifiers, and aggregation
experiments/     Data, analysis, parameter-merging, and focused test utilities
```

All local launchers use ordinary filesystem paths. They do not submit jobs to a
scheduler, stage remote files, or publish artifacts automatically.

## Reference

```bibtex
@article{gao2026openmopd,
  title  = {Open-MOPD: Diagnosing and Fixing Capability Imbalance in Multi-Teacher On-Policy Distillation},
  author = {Gao, Huan-ang and Chi, Haohan and Yan, Yong and Feng, Shiyuan and Wu, Hanlin and Jiang, Zheng and He, Bingxiang and Ma, Wei-Ying and Zhang, Ya-Qin and Zhou, Hao},
  journal = {arXiv preprint},
  year   = {2026}
}
```
