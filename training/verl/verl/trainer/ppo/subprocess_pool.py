"""Generic, domain-agnostic Ray actor pool for running arbitrary callables.

The pool exists to spread small, untrusted, possibly-hanging units of work (e.g.
running a candidate program against a single testcase) across the CPUs of every
live Ray node.

Design goals:
- The pool knows *nothing* about code, testcases, or scoring. A unit of work is
  a :class:`Job` carrying a module-level ``fn`` plus its ``kwargs``. The worker
  simply executes ``fn(**kwargs)`` and returns whatever it produced.
- Each job runs inside a fresh, killable subprocess so a segfault / infinite
  loop / OOM in the executed code cannot take down the long-lived Ray actor.
- Heavy, repeated imports (e.g. the LiveCodeBench runner) can be paid once per
  worker via :meth:`SubprocessWorkerPool.warmup`: the warmup callables run in
  the *actor* process so every subsequently forked child inherits the modules.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import socket
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

DEFAULT_WORKERS_PER_NODE = 100
DEFAULT_WORKER_NUM_CPUS = 1.0

# When ``shared=True`` the worker actors are created as named, detached Ray
# actors in this namespace so that every process touching the pool (the driver
# running the batched code reward, and each async ``RewardManagerWorker`` actor
# running the per-sample inline reward) converges on the SAME set of workers
# instead of each spawning its own. This keeps the cluster-wide CPU budget
# bounded at ``workers_per_node * live_nodes`` regardless of how many callers
# there are. Names embed the node id + worker index, so all callers iterate the
# cluster-global node list and generate identical names; ``get_if_exists`` makes
# the first caller create and the rest reuse, atomically.
POOL_NAMESPACE = "verl_subprocess_pool"
POOL_ACTOR_PREFIX = "verl_subproc_worker"


def _summarize_resources(resources: dict[str, float] | None) -> dict[str, float]:
    if not resources:
        return {}
    keys = ("CPU", "GPU", "memory", "object_store_memory")
    return {key: float(resources[key]) for key in keys if key in resources}


def _wait_with_progress(refs: list[ray.ObjectRef], *, label: str, log_interval_s: float = 10.0) -> list[Any]:
    """Wait for Ray refs while periodically logging scheduling progress."""
    pending = list(refs)
    results: list[Any] = []
    start = time.monotonic()
    last_log = start
    total = len(pending)
    while pending:
        ready, pending = ray.wait(pending, num_returns=1, timeout=log_interval_s)
        now = time.monotonic()
        if ready:
            results.extend(ray.get(ready))
        if pending and (now - last_log >= log_interval_s):
            print(
                f"[subprocess pool] {label} pending={len(pending)}/{total} "
                f"elapsed_s={now - start:.1f} "
                f"available={_summarize_resources(ray.available_resources())}",
                flush=True,
            )
            last_log = now
    print(
        f"[subprocess pool] {label} done total={total} elapsed_s={time.monotonic() - start:.1f}",
        flush=True,
    )
    return results


@dataclass(frozen=True)
class Job:
    """A single unit of work: run ``fn(**kwargs)`` in a guarded subprocess.

    ``owner_id`` / ``job_idx`` are opaque to the pool; callers use them to map
    results back to whatever produced the job (e.g. a completion and the slice
    of testcases this job covers).
    """

    fn: Callable[..., Any]
    kwargs: dict[str, Any] = field(default_factory=dict)
    timeout: float = 30.0
    owner_id: str = ""
    job_idx: int = 0
    memory_limit_mb: int | None = None


@dataclass(frozen=True)
class JobResult:
    owner_id: str
    job_idx: int
    ok: bool
    value: Any = None
    error: str = ""


def _apply_memory_limit(memory_limit_mb: int | None) -> None:
    if not memory_limit_mb or memory_limit_mb <= 0:
        return
    try:
        import resource

        limit_bytes = int(memory_limit_mb) * 1024 * 1024
        # Avoid RLIMIT_AS here: Ray/torch workers inherit very large virtual
        # address maps, so a low AS cap can break even an empty forked child.
        # RLIMIT_DATA still catches ordinary Python heap blowups from generated
        # testcase code without destabilizing the worker runtime.
        for limit_name in ("RLIMIT_DATA",):
            limit = getattr(resource, limit_name, None)
            if limit is not None:
                resource.setrlimit(limit, (limit_bytes, limit_bytes))
    except BaseException as exc:  # noqa: BLE001 - limit setup must not crash the actor
        queue.put(("error", f"Failed to set memory limit: {exc!r}"))
        raise SystemExit(1) from exc


def _subprocess_entry(fn: Callable[..., Any], kwargs: dict[str, Any], queue: Any, memory_limit_mb: int | None) -> None:
    try:
        _apply_memory_limit(memory_limit_mb)
        queue.put(("ok", fn(**kwargs)))
    except MemoryError:
        queue.put(("error", "Memory Limit Exceeded"))
    except BaseException as exc:  # noqa: BLE001 - never let the child raise past here
        queue.put(("error", repr(exc)))


def run_job_in_subprocess(job: Job) -> JobResult:
    """Execute one :class:`Job` in a forked subprocess with a hard kill deadline."""
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_subprocess_entry, args=(job.fn, job.kwargs, queue, job.memory_limit_mb))
    process.start()
    deadline = time.monotonic() + float(job.timeout)
    process.join(timeout=max(0.0, deadline - time.monotonic()))

    if process.is_alive():
        process.kill()
        process.join()
        queue.close()
        queue.cancel_join_thread()
        return JobResult(job.owner_id, job.job_idx, ok=False, error="Time Limit Exceeded")

    try:
        status, payload = queue.get(timeout=1)
    except Exception:  # noqa: BLE001
        if job.memory_limit_mb and process.exitcode is not None:
            if process.exitcode < 0:
                try:
                    signame = signal.Signals(-process.exitcode).name
                except ValueError:
                    signame = f"signal {-process.exitcode}"
                return JobResult(job.owner_id, job.job_idx, ok=False, error=f"Memory Limit Exceeded ({signame})")
            if process.exitcode != 0:
                return JobResult(
                    job.owner_id,
                    job.job_idx,
                    ok=False,
                    error=f"Memory Limit Exceeded or subprocess exited with code {process.exitcode}",
                )
        return JobResult(job.owner_id, job.job_idx, ok=False, error="No result returned")
    finally:
        queue.close()
        queue.cancel_join_thread()

    if status == "ok":
        return JobResult(job.owner_id, job.job_idx, ok=True, value=payload)
    return JobResult(job.owner_id, job.job_idx, ok=False, error=str(payload))


@ray.remote(num_cpus=1)
class SubprocessWorker:
    """Long-lived actor that runs each job in a throwaway child process."""

    def warmup(self, fns: list[Callable[[], Any]]) -> dict[str, Any]:
        # Imports done here live in the actor process, so every later fork
        # inherits them for free instead of re-importing per job.
        for fn in fns:
            try:
                fn()
            except BaseException:  # noqa: BLE001 - warmup is best-effort
                pass
        return {"host": socket.gethostname(), "pid": os.getpid()}

    def run(self, job: Job) -> JobResult:
        return run_job_in_subprocess(job)


class SubprocessWorkerPool:
    """Round-robin scheduler over ``workers_per_node`` actors on every live node."""

    def __init__(
        self,
        workers_per_node: int = DEFAULT_WORKERS_PER_NODE,
        worker_num_cpus: float = DEFAULT_WORKER_NUM_CPUS,
        shared: bool = False,
    ):
        self.workers_per_node = max(1, int(workers_per_node))
        self.worker_num_cpus = max(0.001, float(worker_num_cpus))
        self.shared = bool(shared)
        self._actors: list[Any] = []
        self._next_actor = 0
        self._warmed: set[Any] = set()
        # Async reward actors can finish several trajectories at nearly the
        # same time and enter this synchronous client from different executor
        # threads.  Protect one-time setup and the round-robin cursor so those
        # callers reuse one handle list instead of appending it repeatedly.
        self._start_lock = threading.Lock()
        self._warmup_lock = threading.Lock()
        self._submit_lock = threading.Lock()

    def start(self) -> None:
        with self._start_lock:
            if self._actors:
                return
            live_nodes = [node for node in ray.nodes() if node.get("Alive", False)]
            print(
                f"[subprocess pool] start live_nodes={len(live_nodes)} "
                f"workers_per_node={self.workers_per_node} worker_num_cpus={self.worker_num_cpus} "
                f"shared={self.shared} "
                f"cluster={_summarize_resources(ray.cluster_resources())} "
                f"available={_summarize_resources(ray.available_resources())}",
                flush=True,
            )
            for node in live_nodes:
                node_id = node["NodeID"]
                strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
                node_resources = _summarize_resources(node.get("Resources") or {})
                print(
                    f"[subprocess pool] create actors node_id={node_id} "
                    f"node_resources={node_resources}",
                    flush=True,
                )
                for worker_idx in range(self.workers_per_node):
                    if self.shared:
                        # Named + detached + get_if_exists => first caller creates,
                        # every later caller (other reward actors, the driver)
                        # reuses the same actor. ``num_cpus``/options are honoured
                        # only on creation; reuse returns the existing handle.
                        actor = SubprocessWorker.options(
                            name=f"{POOL_ACTOR_PREFIX}_{node_id}_{worker_idx}",
                            namespace=POOL_NAMESPACE,
                            lifetime="detached",
                            get_if_exists=True,
                            scheduling_strategy=strategy,
                            num_cpus=self.worker_num_cpus,
                        ).remote()
                    else:
                        actor = SubprocessWorker.options(
                            scheduling_strategy=strategy,
                            num_cpus=self.worker_num_cpus,
                        ).remote()
                    self._actors.append(actor)
            if not self._actors:
                raise RuntimeError("No live Ray nodes available for SubprocessWorkerPool")
            print(f"[subprocess pool] actor handles created total={len(self._actors)}", flush=True)

    def warmup(self, fns: list[Callable[[], Any]]) -> None:
        # Idempotent: only broadcast callables not already warmed on this pool.
        with self._warmup_lock:
            pending = [fn for fn in fns if fn not in self._warmed]
            if not pending:
                return
            self.start()
            print(
                f"[subprocess pool] warmup start actors={len(self._actors)} "
                f"functions={[getattr(fn, '__name__', repr(fn)) for fn in pending]}",
                flush=True,
            )
            warmup_results = _wait_with_progress(
                [actor.warmup.remote(pending) for actor in self._actors], label="warmup"
            )
            host_counts = Counter(str(item.get("host", "unknown")) for item in warmup_results if isinstance(item, dict))
            print(f"[subprocess pool] warmup actor_hosts={dict(host_counts)}", flush=True)
            self._warmed.update(pending)

    def submit(self, jobs: list[Job]) -> list[ray.ObjectRef]:
        self.start()
        print(
            f"[subprocess pool] submit jobs={len(jobs)} actors={len(self._actors)} "
            f"available={_summarize_resources(ray.available_resources())}",
            flush=True,
        )
        with self._submit_lock:
            actors = []
            for _ in jobs:
                actors.append(self._actors[self._next_actor % len(self._actors)])
                self._next_actor += 1
        futures = []
        for actor, job in zip(actors, jobs, strict=True):
            futures.append(actor.run.remote(job))
        print(f"[subprocess pool] submit done futures={len(futures)}", flush=True)
        return futures

    def map(self, jobs: list[Job]) -> list[JobResult]:
        return ray.get(self.submit(jobs))


_GLOBAL_POOL: SubprocessWorkerPool | None = None
_GLOBAL_POOL_LOCK = threading.Lock()


def get_subprocess_worker_pool(
    workers_per_node: int = DEFAULT_WORKERS_PER_NODE,
    worker_num_cpus: float = DEFAULT_WORKER_NUM_CPUS,
    shared: bool = False,
) -> SubprocessWorkerPool:
    global _GLOBAL_POOL
    with _GLOBAL_POOL_LOCK:
        if (
            _GLOBAL_POOL is None
            or _GLOBAL_POOL.workers_per_node != max(1, int(workers_per_node))
            or _GLOBAL_POOL.worker_num_cpus != max(0.001, float(worker_num_cpus))
            or _GLOBAL_POOL.shared != bool(shared)
        ):
            _GLOBAL_POOL = SubprocessWorkerPool(
                workers_per_node=workers_per_node,
                worker_num_cpus=worker_num_cpus,
                shared=shared,
            )
        return _GLOBAL_POOL
