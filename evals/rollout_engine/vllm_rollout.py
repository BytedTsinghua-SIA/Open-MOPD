from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline vLLM rollout over eval parquet files with Data Parallelism.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", nargs="+", type=Path, required=True, help="One or more parquet inputs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    
    # ---------------------------------------------------------
    # Base 和 Offset 参数，用于大规模数据集断点续跑或任务分发
    # ---------------------------------------------------------
    parser.add_argument("--base", type=int, default=0, help="Starting index of the dataset to process.")
    parser.add_argument(
        "--offset", 
        type=int, 
        default=None, 
        help="Number of elements to process from the base index. If not set, processes till the end."
    )
    
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.93)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true", default=True)
    parser.add_argument("--max-num-seqs", type=int, default=1024, help="Maximum sequences per scheduler step.")
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Maximum batched tokens per scheduler step.",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--n", type=int, default=1, help="Number of completions to return per prompt.")
    parser.add_argument("--max-tokens", type=int, default=30000)
    parser.add_argument(
        "--stop-token-ids",
        help="Comma-separated token ids used to stop generation, for example 151643,151645 for Qwen3 Base.",
    )
    parser.add_argument(
        "--enable-thinking",
        choices=["true", "false"],
        help="Pass enable_thinking to tokenizer.apply_chat_template for chat-style prompt rows.",
    )
    
    # 分布式节点参数 (如果只在单机跑，默认即可)
    parser.add_argument("--node-size", type=int, default=1, help="Total number of nodes")
    parser.add_argument("--node-rank", type=int, default=0, help="Rank of the current node")

    return parser.parse_args()


def llm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "enable_prefix_caching": args.enable_prefix_caching,
    }
    if args.max_num_seqs is not None:
        kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    
    # 注意：这里坚决不能传入 data_parallel_size，因为我们要用 CUDA_VISIBLE_DEVICES 物理隔离
    return kwargs


def parse_stop_token_ids(raw: Any) -> list[int] | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        if raw.startswith("["):
            return [int(item) for item in json.loads(raw)]
        return [int(item.strip()) for item in raw.split(",") if item.strip()]
    if isinstance(raw, (list, tuple)):
        return [int(item) for item in raw]
    return [int(raw)]


def resolve_stop_token_ids(model_path: str, explicit_stop_token_ids: Any) -> list[int] | None:
    """Resolve explicit stop IDs or load the model's generation contract."""

    explicit = parse_stop_token_ids(explicit_stop_token_ids)
    if explicit is not None:
        return explicit

    from transformers import GenerationConfig

    generation_config = GenerationConfig.from_pretrained(model_path)
    return parse_stop_token_ids(generation_config.eos_token_id)


def parse_optional_bool(raw: str | None) -> bool | None:
    if raw is None:
        return None
    return raw.lower() == "true"


def sampling_from_config(
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    n: int,
    max_tokens: int,
    stop_token_ids: list[int] | None,
) -> Any:
    from vllm import SamplingParams

    kwargs: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "max_tokens": max_tokens,
    }
    if top_k >= 0:
        kwargs["top_k"] = top_k
    if stop_token_ids:
        kwargs["stop_token_ids"] = stop_token_ids
    return SamplingParams(**kwargs)


def prompt_to_text(prompt: Any, tokenizer: Any, enable_thinking: bool | None) -> str:
    if hasattr(prompt, "tolist") and not isinstance(prompt, str):
        prompt = prompt.tolist()
    if isinstance(prompt, str):
        stripped = prompt.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return prompt
            if isinstance(parsed, list):
                prompt = parsed
            else:
                return prompt
        else:
            return prompt
    if isinstance(prompt, list):
        kwargs = {"enable_thinking": enable_thinking} if enable_thinking is not None else {}
        return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, **kwargs)
    return str(prompt)


def get_tokenizer(llm: Any) -> Any:
    if hasattr(llm, "get_tokenizer"):
        return llm.get_tokenizer()
    engine = getattr(llm, "llm_engine", None)
    tokenizer = getattr(engine, "tokenizer", None)
    if hasattr(tokenizer, "tokenizer"):
        return tokenizer.tokenizer
    if tokenizer is not None:
        return tokenizer
    raise AttributeError("Unable to access tokenizer from vLLM LLM instance")


def generated_special_token_counts(token_ids: list[int], tokenizer: Any) -> str:
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    counts = {
        str(token_id): token_ids.count(token_id)
        for token_id in sorted(special_ids.intersection(token_ids))
    }
    return json.dumps(counts, sort_keys=True)


def cleanup_env_and_memory():
    from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    # 清理 vLLM 和 PyTorch 进程及显存
    try:
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass
        
    with contextlib.suppress(AssertionError, Exception):
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run_file_partition(
    llm: Any,
    path: Path,
    output_dir: Path,
    temperature: float,
    top_p: float,
    top_k: int,
    n: int,
    max_tokens: int,
    stop_token_ids: list[int] | None,
    enable_thinking: bool | None,
    dp_size: int,
    global_dp_rank: int,
    base: int,
    offset: int | None,
) -> list[Path]:
    df = pd.read_parquet(path)
    
    # 第一步：先按 base 和 offset 截取目标区间
    start_idx = base
    end_idx = base + offset if offset is not None else len(df)
    
    # 确保索引不越界
    end_idx = min(end_idx, len(df))
    df_target = df.iloc[start_idx:end_idx].copy()
    
    # 第二步：将截取后的目标区间均分给各个 DP Rank
    total_target_len = len(df_target)
    floor = total_target_len // dp_size
    remainder = total_target_len % dp_size

    def get_dp_start(rank):
        return rank * floor + min(rank, remainder)

    dp_start_idx = get_dp_start(global_dp_rank)
    dp_end_idx = get_dp_start(global_dp_rank + 1)
    
    df_slice = df_target.iloc[dp_start_idx:dp_end_idx].copy()

    # 如果当前 Rank 没有分到数据，直接跳过
    if len(df_slice) == 0:
        return []

    df_slice["temperature"] = temperature
    df_slice["top_p"] = top_p
    df_slice["top_k"] = top_k
    df_slice["max_tokens"] = max_tokens
    if stop_token_ids:
        df_slice["stop_token_ids"] = ",".join(str(token_id) for token_id in stop_token_ids)
    
    outputs = []
    tokenizer = get_tokenizer(llm)

    params = sampling_from_config(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        n=n,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids,
    )
    prompts = [prompt_to_text(prompt, tokenizer, enable_thinking) for prompt in df_slice["prompt"].tolist()]
    results = llm.generate(prompts, params)
    
    for (_, row), result in zip(df_slice.iterrows(), results):
        if not result.outputs:
            item = row.to_dict()
            item["completion_index"] = 0
            item["completion"] = ""
            item["completion_tokens"] = 0
            outputs.append(item)
            continue
        for completion_index, output in enumerate(result.outputs):
            item = row.to_dict()
            item["completion_index"] = completion_index
            item["completion"] = output.text
            output_token_ids = list(output.token_ids)
            item["completion_tokens"] = len(output_token_ids)
            item["finish_reason"] = getattr(output, "finish_reason", None)
            item["stop_reason"] = getattr(output, "stop_reason", None)
            item["generated_special_token_counts"] = generated_special_token_counts(
                output_token_ids, tokenizer
            )
            outputs.append(item)

    out_df = pd.DataFrame.from_records(outputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    
    # 文件名打上标记，防止并发覆写
    file_suffix = f"base{base}_offset{offset}_rank{global_dp_rank}" if offset else f"base{base}_rank{global_dp_rank}"

    if len(out_df) and "dataset" in out_df:
        dataset_col = out_df["dataset"]
        missing_dataset = dataset_col.isna() | (dataset_col.astype(str).str.len() == 0)
        if missing_dataset.any():
            fallback_dataset = path.stem
            if "_input_name" in out_df:
                input_names = out_df.loc[~out_df["_input_name"].isna(), "_input_name"].astype(str)
                if len(input_names):
                    fallback_dataset = input_names.iloc[0]
            out_df = out_df.copy()
            out_df.loc[missing_dataset, "dataset"] = fallback_dataset
        for dataset, dataset_df in out_df.groupby("dataset", sort=False):
            out_path = output_dir / f"{dataset}_rollouts_{file_suffix}.parquet"
            dataset_df.to_parquet(out_path, index=False)
            out_paths.append(out_path)
    if not out_paths:
        out_path = output_dir / f"{path.stem}_rollouts_{file_suffix}.parquet"
        out_df.to_parquet(out_path, index=False)
        out_paths.append(out_path)
    return out_paths


def worker_main(
    args: argparse.Namespace,
    dp_size: int,
    local_dp_rank: int,
    global_dp_rank: int,
    tp_size: int,
):
    # 核心隔离逻辑：为该进程分配硬隔离的显卡 ID
    start_gpu = local_dp_rank * tp_size
    gpu_ids = ",".join(str(start_gpu + i) for i in range(tp_size))
    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    print(f"[Worker Rank {global_dp_rank}] Started processing. Bound to GPU(s): {gpu_ids}")

    from vllm import LLM
    from time import sleep

    # 当前进程只能看到自己被分配的卡，vLLM 内部自然作为独立实例运行
    llm = LLM(**llm_kwargs(args))
    stop_token_ids = resolve_stop_token_ids(args.model, args.stop_token_ids)
    print(f"[Worker Rank {global_dp_rank}] stop_token_ids={stop_token_ids}", flush=True)
    enable_thinking = parse_optional_bool(args.enable_thinking)
    
    for path in args.input:
        out_paths = run_file_partition(
            llm,
            path,
            args.output_dir,
            args.temperature,
            args.top_p,
            args.top_k,
            args.n,
            args.max_tokens,
            stop_token_ids,
            enable_thinking,
            dp_size=dp_size,
            global_dp_rank=global_dp_rank,
            base=args.base,
            offset=args.offset,
        )
        for out in out_paths:
            print(f"[Worker Rank {global_dp_rank}] Output written to: {out}")

    sleep(5)
    del llm
    cleanup_env_and_memory()


def main() -> None:
    args = parse_args()
    
    dp_size = args.data_parallel_size or 1
    tp_size = args.tensor_parallel_size
    node_size = args.node_size
    node_rank = args.node_rank

    assert dp_size % node_size == 0, "data_parallel_size should be divisible by node_size"
    dp_per_node = dp_size // node_size

    from multiprocessing import Process
    procs = []
    
    # 启动多进程，分发给不同显卡
    for local_dp_rank, global_dp_rank in enumerate(range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node)):
        proc = Process(
            target=worker_main,
            args=(
                args,
                dp_size,
                local_dp_rank,
                global_dp_rank,
                tp_size,
            ),
        )
        proc.start()
        procs.append(proc)

    exit_code = 0
    # 等待所有进程完成
    for proc in procs:
        proc.join()
        if proc.exitcode:
            exit_code = proc.exitcode

    exit(exit_code)

if __name__ == "__main__":
    main()
