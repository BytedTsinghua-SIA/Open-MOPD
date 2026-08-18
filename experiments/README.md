# Experiments

实验入口已经统一为本地 shell recipes：

```bash
bash scripts/local/sft.sh --help
bash scripts/local/rl.sh --help
bash scripts/local/opd.sh --help
bash scripts/local/mt_opd.sh --help
bash scripts/local/eval.sh --help
```

它们直接调用仓库中的训练和评测模块，使用当前机器上的模型、数据和输出
目录。默认只打印命令，加入 `--run` 后执行；当前只支持单节点，不进行
远程存储 staging、上传或队列提交。

## 目录

```text
experiments/
├── data/       # 数据转换和 RFT 工具
├── analysis/   # 可复现的本地分析产物
├── backend/    # 本地模型合并工具
└── tests/      # 纯 Python 和 shell smoke tests
```

新实验应复制最接近的 `scripts/local/*.sh`，通过 CLI 或环境变量传入路径和
训练参数。不要新增 JSON 任务描述、提交器、远程存储配置或主机绝对路径。

常用验证：

```bash
bash -n scripts/local/*.sh eval.sh
PYTHONPATH=. pytest -q experiments/tests/test_local_shell_entrypoints.py
```
