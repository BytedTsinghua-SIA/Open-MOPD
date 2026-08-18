from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace
from typing import Any

from evals.verifier.score_functions.code.livecodebench_official import load_lcb_official_modules
from evals.verifier.score_functions.common.utils import log


def score_lcb_batch(payload: dict[str, Any]) -> dict[str, Any]:
    modules = load_lcb_official_modules()
    combine_results = modules["combine_results"]
    extract_instance_results = modules["extract_instance_results"]
    get_metrics = modules["get_metrics"]
    LanguageModelStore = modules["LanguageModelStore"]
    Scenario = modules["Scenario"]
    model_alias = payload.get("model_alias", "Qwen/Qwen2.5-72B-Instruct")
    model = LanguageModelStore[model_alias]
    combined_results = combine_results(
        Scenario.codegeneration,
        payload["completions"],
        model,
    )
    eval_args = SimpleNamespace(
        scenario=Scenario.codegeneration,
        num_process_evaluate=max(1, int(payload["concurrency"])),
        timeout=payload["timeout"],
    )
    problems = payload["problems"]
    completions = [outputs for outputs, _extracted in combined_results]
    generations = [extracted for _outputs, extracted in combined_results]
    metrics = get_metrics(Scenario.codegeneration, eval_args, problems, combined_results)
    graded = extract_instance_results(metrics[1])
    metadatas = metrics[2]
    evaluated = [
        problem.insert_output_evaluation(
            outputs_list,
            extracted_list,
            graded_list,
            metadata=meta,
        )
        for problem, outputs_list, extracted_list, graded_list, meta in zip(
            problems,
            completions,
            generations,
            graded,
            metadatas,
        )
    ]
    return {
        "graded": graded,
        "metadata": metadatas,
        "evaluated": evaluated,
        "problem_count": len(problems),
        "completion_count": sum(len(items) for items in generations),
    }


def score_lcb_single_process(
    problems: list[Any],
    completions: list[list[str]],
    generations: list[list[str]],
    workers: int,
    progress_interval: int,
    timeout: int,
) -> tuple[list[list[Any]], list[Any], list[Any]]:
    all_graded = []
    all_metadata = []
    evaluated = []
    total_completions = sum(len(items) for items in generations)
    evaluated_completions = 0
    evaluated_problems = 0
    progress_interval = max(1, int(progress_interval))
    batch_problems = []
    batch_generations = []
    batch_completions = []
    batch_size = 0

    def flush_batch() -> None:
        nonlocal batch_problems, batch_generations, batch_completions, batch_size, evaluated_completions, evaluated_problems
        if not batch_problems:
            return
        log(
            f"livecodebench: evaluating batch of {batch_size} completions "
            f"({len(batch_problems)} problems) with {max(1, int(workers))} workers"
        )
        result = score_lcb_batch(
            {
                "problems": batch_problems,
                "generations": batch_generations,
                "completions": batch_completions,
                "concurrency": max(1, int(workers)),
                "timeout": timeout,
                "model_alias": "Qwen/Qwen2.5-72B-Instruct",
            }
        )
        all_graded.extend(result["graded"])
        all_metadata.extend(result["metadata"])
        evaluated.extend(result["evaluated"])
        evaluated_completions += batch_size
        evaluated_problems += len(batch_problems)
        log(
            f"livecodebench: evaluated {evaluated_completions}/{total_completions} completions "
            f"({evaluated_problems}/{len(problems)} problems)"
        )
        batch_problems = []
        batch_generations = []
        batch_completions = []
        batch_size = 0

    for problem, problem_completions, problem_generations in zip(problems, completions, generations):
        problem_size = len(problem_generations)
        if batch_problems and batch_size + problem_size > progress_interval:
            flush_batch()
        batch_problems.append(problem)
        batch_completions.append(problem_completions)
        batch_generations.append(problem_generations)
        batch_size += problem_size
    flush_batch()
    return all_graded, all_metadata, evaluated


def score_lcb_multi_process(
    problems: list[Any],
    completions: list[list[str]],
    generations: list[list[str]],
    worker_count: int,
    worker_concurrency: int,
    timeout: int,
) -> tuple[list[list[Any]], list[Any], list[Any]]:
    total = len(problems)
    shard_size = max(1, math.ceil(total / worker_count))
    payloads = []
    for start in range(0, total, shard_size):
        payloads.append(
            {
                "problems": problems[start : start + shard_size],
                "completions": completions[start : start + shard_size],
                "generations": generations[start : start + shard_size],
                "concurrency": worker_concurrency,
                "timeout": timeout,
                "model_alias": "Qwen/Qwen2.5-72B-Instruct",
                "start": start,
            }
        )
    log(
        f"livecodebench: evaluating {total} problems with {len(payloads)} process workers "
        f"and {worker_concurrency} concurrency per worker"
    )
    results_by_start = {}
    with ProcessPoolExecutor(max_workers=len(payloads)) as executor:
        futures = {executor.submit(score_lcb_batch, payload): payload for payload in payloads}
        for future in as_completed(futures):
            payload = futures[future]
            result = future.result()
            results_by_start[payload["start"]] = result
            log(
                f"livecodebench: worker shard start={payload['start']} done "
                f"problems={result['problem_count']} completions={result['completion_count']}"
            )
    all_graded = []
    all_metadata = []
    evaluated = []
    for start in sorted(results_by_start):
        result = results_by_start[start]
        all_graded.extend(result["graded"])
        all_metadata.extend(result["metadata"])
        evaluated.extend(result["evaluated"])
    return all_graded, all_metadata, evaluated

