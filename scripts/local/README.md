# Local shell launchers

These five launchers expose the common training and evaluation entry points as
plain shell commands. They use paths already present on the machine and do not
fetch or publish artifacts. Every launcher prints its command and exits without
running it by default; add `--run` after checking the paths.

Examples:

```bash
bash scripts/local/opd.sh \
  --model /data/models/student \
  --teacher /data/models/teacher \
  --train /data/train.parquet \
  --val /data/valid.parquet \
  --gpus 8

bash scripts/local/opd.sh --model /data/models/student \
  --teacher /data/models/teacher --train /data/train.parquet \
  --val /data/valid.parquet --gpus 8 --run

bash scripts/local/mt_opd.sh \
  --model /data/models/student \
  --teacher /data/models/math \
  --teacher /data/models/code \
  --teacher /data/models/if \
  --domains math,code,if --train /data/train.parquet \
  --val /data/valid.parquet --gpus 8

bash scripts/local/eval.sh --model /data/models/student \
  --input /data/test.parquet --output /data/eval-output
```

Environment variables mirror the option names (`MODEL_PATH`, `TRAIN_FILE`,
`VAL_FILE`, `OUTPUT_DIR`, `CHECKPOINT_DIR`, `GPUS`, `NODES`, `PYTHON_BIN`, and
`TORCHRUN_BIN`). For teachers, use `REWARD_MODEL_PATH` for OPD or a
comma-separated `TEACHER_MODEL_PATHS` plus `TEACHER_DOMAINS` for MT-OPD.
Additional trainer overrides may be passed after `--`.
