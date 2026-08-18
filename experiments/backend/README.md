# Local model tools

The backend directory contains utilities that operate on local checkpoint
directories. They do not submit jobs or access remote storage.

`param_merge.py` builds traceable parameter-merge baselines from local
Hugging Face checkpoints:

```bash
python -m experiments.backend.param_merge \
  --base /path/to/base \
  --model math=/path/to/math \
  --model code=/path/to/code \
  --output /path/to/merged
```

Checkpoint conversion should likewise use local input and output directories;
the training launchers write checkpoints under `CHECKPOINT_DIR`.
