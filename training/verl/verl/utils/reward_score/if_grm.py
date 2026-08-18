"""Generative reward model (LLM-as-judge) client for instruction-following RL.

The judge runs over a list of OpenAI-format HTTP endpoints (e.g. several
gpt-oss-120b replicas) and decides whether a rollout genuinely solves the
underlying user task. It is used to gate the rule-based constraint score so
that responses which only satisfy verifiable format/keyword/length constraints
(while ignoring the actual request) no longer collect full reward.

Configuration is read from the reward_kwargs dict (and a few env fallbacks) so
that the whole feature is driven from the experiment spec.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests
from requests.adapters import HTTPAdapter

# ---------------------------------------------------------------------------
# Module-level HTTP session cache.  Persistent sessions reuse TCP connections
# (keep-alive) across reward steps in the same process, eliminating per-call
# TLS + TCP handshake overhead.  The dict is keyed by (endpoint, use_env_proxy)
# and shared across threads via a lock.  In Ray worker processes the cache is
# private to that process, so no cross-process locking is needed.
# ---------------------------------------------------------------------------
_HTTP_SESSIONS: dict[tuple[str, bool], requests.Session] = {}
_HTTP_SESSIONS_LOCK = threading.Lock()


def _get_session(endpoint: str, use_env_proxy: bool) -> requests.Session:
    key = (endpoint, use_env_proxy)
    with _HTTP_SESSIONS_LOCK:
        if key not in _HTTP_SESSIONS:
            print(f"[if_grm] creating HTTP session for {endpoint} (use_env_proxy={use_env_proxy})")
            session = requests.Session()
            session.trust_env = use_env_proxy
            # Allow up to 64 in-flight connections to the same host from this
            # process so a single Ray worker can saturate one GRM endpoint.
            adapter = HTTPAdapter(pool_connections=1, pool_maxsize=64, max_retries=0)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            _HTTP_SESSIONS[key] = session
        return _HTTP_SESSIONS[key]

# Label that marks the single turn the judge must evaluate.
TO_BE_JUDGED_LABEL = "[Assistant # to be judged]"

# The judging criteria + required output format are static so they live in the system
# prompt: this keeps them identical across every request and maximizes prefix-cache reuse
# on the judge servers. The (variable) conversation transcript goes in the user message.
GRM_JUDGE_SYSTEM_PROMPT = (
    "You are a strict semantic judge for a conversation between a user and an AI assistant.\n\n"
    "You will be given a conversation transcript. Every turn except the last is context. "
    f'Judge ONLY the final assistant turn, which is marked "{TO_BE_JUDGED_LABEL}".\n\n'
    "You judge SEMANTICS ONLY: decide whether that final assistant turn is genuinely attempting "
    "to perform the user's actual task and substantively, on-topically addresses the latest user "
    "request, as opposed to reward hacking (an empty, degenerate, off-topic, single-word, "
    "single-number, repeated-filler, refusal, or generic-acknowledgement response that does not "
    "actually do the task).\n\n"
    "Do NOT judge instruction-following, formatting, wording, keyword, ordering, or length "
    "constraints. Those are verified separately by rules, so ignore them completely even if they "
    "are obviously satisfied or obviously violated. Your verdict must depend only on whether the "
    "assistant is really doing the task.\n\n"
    "First give a brief reason (about 50 tokens) explaining your judgment inside a single "
    "<reason>...</reason> tag, then output your final verdict as exactly one XML tag:\n"
    "<verdict>YES</verdict> if the final assistant turn genuinely and substantively does the task, or\n"
    "<verdict>NO</verdict> if it is reward hacking or does not substantively address the task.\n"
    "Output only the <reason>...</reason> tag followed by the <verdict>...</verdict> tag and nothing else."
)

_VERDICT_RE = re.compile(r"<verdict>\s*(yes|no|true|false)\s*</verdict>", re.IGNORECASE)
_YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
_NO_RE = re.compile(r"\bno\b", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{\s*(yes|no|true|false)\s*\}", re.IGNORECASE)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


@dataclass
class GRMConfig:
    """Resolved GRM configuration parsed from reward_kwargs + env fallbacks."""

    enabled: bool = False
    endpoints: list[str] = field(default_factory=list)
    model: str = "gpt-oss-120b"
    api_key: str = "EMPTY"
    timeout: float = 60.0
    max_retries: int = 3
    base_delay: float = 2.0
    max_concurrency: int = 32
    # Per-endpoint concurrent request cap.  The batching logic in fan_out_judges
    # will submit at most  len(endpoints) × max_concurrency_per_endpoint  calls
    # simultaneously, staying within each server's connection limit.
    max_concurrency_per_endpoint: int = 32
    max_tokens: int = 5000
    temperature: float = 0.0
    # Passed to the judge as a hint to use a faster/cheaper reasoning path.
    # Set to "" or None to omit the field from the request payload.
    reasoning_effort: str = "low"
    fail_cap: float = 0.0
    fail_mode: str = "open"  # "open" keeps the score on judge error, "closed" caps it
    # When a rollout fails the semantic judge, keep the independent think-format bonus
    # (cap to 0.2*F) instead of zeroing everything. Set False for the aggressive hard-0.
    keep_think_on_fail: bool = True
    families: tuple[str, ...] = ()  # empty = judge every instruction-following family
    system_prompt: str = GRM_JUDGE_SYSTEM_PROMPT
    # GRM endpoints are usually intra-cluster IPs. Inheriting HTTP(S)_PROXY sends
    # those requests through the corporate proxy and can turn healthy endpoints
    # into 403/ProxyError failures.
    use_env_proxy: bool = False

    @classmethod
    def from_reward_kwargs(cls, reward_kwargs: dict[str, Any]) -> "GRMConfig":
        endpoints = _as_str_list(reward_kwargs.get("grm_endpoints"))
        if not endpoints:
            endpoints = _as_str_list(os.getenv("OPENOPD_GRM_ENDPOINTS"))
        endpoints = [url.rstrip("/") for url in endpoints]

        api_key = str(reward_kwargs.get("grm_api_key") or os.getenv("OPENOPD_GRM_API_KEY") or "EMPTY")

        # No families configured -> judge every IF family (the common, recommended default).
        families = tuple(_as_str_list(reward_kwargs.get("grm_families")))

        fail_mode = str(reward_kwargs.get("grm_fail_mode", "open") or "open").strip().lower()
        if fail_mode not in {"open", "closed"}:
            fail_mode = "open"

        reasoning_effort_raw = reward_kwargs.get("grm_reasoning_effort", "low")
        reasoning_effort = str(reasoning_effort_raw).strip() if reasoning_effort_raw else ""

        return cls(
            enabled=_as_bool(reward_kwargs.get("enable_grm", False)),
            endpoints=endpoints,
            model=str(reward_kwargs.get("grm_model", "gpt-oss-120b") or "gpt-oss-120b"),
            api_key=api_key,
            timeout=float(reward_kwargs.get("grm_timeout", 60.0)),
            max_retries=int(reward_kwargs.get("grm_max_retries", 3)),
            base_delay=float(reward_kwargs.get("grm_base_delay", 2.0)),
            max_concurrency=max(1, int(reward_kwargs.get("grm_max_concurrency", 32))),
            max_concurrency_per_endpoint=max(1, int(reward_kwargs.get("grm_max_concurrency_per_endpoint", 32))),
            max_tokens=int(reward_kwargs.get("grm_max_tokens", 5000)),
            temperature=float(reward_kwargs.get("grm_temperature", 0.0)),
            reasoning_effort=reasoning_effort,
            fail_cap=float(reward_kwargs.get("grm_fail_cap", 0.0)),
            fail_mode=fail_mode,
            keep_think_on_fail=_as_bool(reward_kwargs.get("grm_keep_think_on_fail", True)),
            families=families,
            system_prompt=str(reward_kwargs.get("grm_system_prompt") or GRM_JUDGE_SYSTEM_PROMPT),
            use_env_proxy=_as_bool(reward_kwargs.get("grm_use_env_proxy", False)),
        )

    @property
    def is_usable(self) -> bool:
        return self.enabled and bool(self.endpoints)


class _RoundRobin:
    """Thread-safe round-robin selector over the endpoint list."""

    def __init__(self, endpoints: list[str]) -> None:
        self._lock = threading.Lock()
        self._cycle = itertools.cycle(endpoints) if endpoints else None

    def next(self) -> str | None:
        if self._cycle is None:
            return None
        with self._lock:
            return next(self._cycle)


def coerce_messages(raw: Any) -> list[dict[str, str]]:
    """Normalize a prompt into a list of ``{role, content}`` turns.

    Accepts a plain string (single user turn), a list of message dicts (multi-turn),
    a JSON-encoded list, or a numpy array of messages. Anything unrecognized yields
    an empty context.
    """
    if raw is None:
        return []
    if hasattr(raw, "tolist") and not isinstance(raw, (str, bytes)):
        return coerce_messages(raw.tolist())
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except (ValueError, TypeError):
                return [{"role": "user", "content": raw}]
            if isinstance(parsed, list):
                return [dict(item) for item in parsed if isinstance(item, dict)]
        return [{"role": "user", "content": raw}]
    if isinstance(raw, dict):
        return [dict(raw)]
    if isinstance(raw, Iterable):
        return [dict(item) for item in raw if isinstance(item, dict)]
    return []


def render_transcript(conversation: Any, answer_text: str) -> str:
    """Render the conversation as a transcript, marking the final assistant turn to judge.

    The constraint-only system prompt (formatting/length rules) is intentionally dropped:
    the judge evaluates semantics only, so those rules must not leak into the transcript.
    """
    lines: list[str] = []
    for message in coerce_messages(conversation):
        role = str(message.get("role", "user") or "user").strip().lower()
        content = str(message.get("content", "") or "").strip()
        if role == "system":
            continue
        label = "[Assistant]" if role == "assistant" else "[User]"
        lines.append(f"{label}\n{content}")
    lines.append(f"{TO_BE_JUDGED_LABEL}\n{str(answer_text or '').strip()}")
    return "\n\n".join(lines)


def build_judge_messages(conversation: Any, answer_text: str, config: GRMConfig) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": render_transcript(conversation, answer_text)},
    ]


def parse_verdict(response_text: str | None) -> bool | None:
    """Map a judge completion to a semantic-pass boolean, or None if unparseable."""
    if not response_text:
        return None
    text = str(response_text).strip()
    if not text:
        return None

    verdict = _VERDICT_RE.search(text)
    if verdict is not None:
        return verdict.group(1).lower() in {"yes", "true"}

    boxed = _BOXED_RE.search(text)
    if boxed is not None:
        return boxed.group(1).lower() in {"yes", "true"}

    # Prefer the last non-empty line, which is where we instruct the judge to answer.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        has_yes = bool(_YES_RE.search(line))
        has_no = bool(_NO_RE.search(line))
        if has_yes and not has_no:
            return True
        if has_no and not has_yes:
            return False

    has_yes = bool(_YES_RE.search(text))
    has_no = bool(_NO_RE.search(text))
    if has_yes and not has_no:
        return True
    if has_no and not has_yes:
        return False
    return None


def _request_completion(endpoint: str, messages: list[dict[str, str]], config: GRMConfig) -> str | None:
    url = f"{endpoint}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config.api_key and config.api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {config.api_key}"
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
    }
    if config.reasoning_effort:
        payload["reasoning_effort"] = config.reasoning_effort
    session = _get_session(endpoint, config.use_env_proxy)
    for attempt in range(config.max_retries):
        try:
            output = session.post(url, headers=headers, json=payload, timeout=config.timeout)
            output.raise_for_status()
            return output.json()["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001 - judge transport must never crash training
            if attempt < config.max_retries - 1:
                delay = config.base_delay * (2**attempt)
                print(f"[if_grm] retry {attempt+1}/{config.max_retries} for {url} in {delay:.1f}s: {exc!r}")
                time.sleep(delay)
            else:
                print(f"[if_grm] judge request to {url} failed after {config.max_retries} attempts: {exc!r}")
    return None


def judge_semantic_pass(
    conversation: Any,
    answer_text: str,
    config: GRMConfig,
    selector: _RoundRobin,
) -> bool | None:
    """Return True/False semantic verdict, or None on transport/parse failure."""
    endpoint = selector.next()
    if endpoint is None:
        return None
    messages = build_judge_messages(conversation, answer_text, config)
    response_text = _request_completion(endpoint, messages, config)
    return parse_verdict(response_text)


def make_selector(config: GRMConfig) -> _RoundRobin:
    return _RoundRobin(list(config.endpoints))


# ---------------------------------------------------------------------------
# Persistent thread pool — created once per process, reused every reward step.
# Creating/destroying 512 threads per step would cost ~0.5-1 s on its own.
# ---------------------------------------------------------------------------
_THREAD_POOL: ThreadPoolExecutor | None = None
_THREAD_POOL_LOCK = threading.Lock()


def _get_thread_pool(max_workers: int) -> ThreadPoolExecutor:
    global _THREAD_POOL
    with _THREAD_POOL_LOCK:
        if _THREAD_POOL is None:
            print(
                f"[if_grm] creating persistent ThreadPoolExecutor "
                f"(max_workers={max_workers}, thread_name_prefix=grm_worker)"
            )
            _THREAD_POOL = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="grm_worker",
            )
            print(f"[if_grm] thread pool ready ({max_workers} workers)")
    return _THREAD_POOL


def fan_out_judges(
    conversations: list[Any],
    answers: list[str],
    config: GRMConfig,
    selector: _RoundRobin,
) -> list[bool | None]:
    """Fan all GRM judge calls out in parallel via a persistent ThreadPoolExecutor.

    The pool (``_THREAD_POOL``) is created once on first call and reused for every
    subsequent reward step, eliminating per-step thread-creation overhead.

    Worker count = ``len(endpoints) × max_concurrency_per_endpoint``.  With 16
    endpoints × 32 = 512 workers, all candidates fly simultaneously; wall-clock
    time ≈ one HTTP round-trip.

    ``requests`` releases the GIL during socket I/O so threads are genuinely
    concurrent.  The module-level ``_HTTP_SESSIONS`` pool gives each endpoint a
    persistent keep-alive connection that is shared safely across threads.
    """
    n = len(conversations)
    endpoints = [selector.next() for _ in range(n)]
    messages_list = [build_judge_messages(c, a, config) for c, a in zip(conversations, answers)]

    max_workers = len(config.endpoints) * config.max_concurrency_per_endpoint
    pool = _get_thread_pool(max_workers)

    # Count how many requests go to each endpoint for the dispatch log.
    from collections import Counter  # noqa: PLC0415
    ep_counts = Counter(endpoints)
    ep_summary = " | ".join(
        f"{ep.rsplit(':', 1)[-1]}×{cnt}"   # show only port for brevity
        for ep, cnt in sorted(ep_counts.items(), key=lambda x: x[0])
    )
    t_dispatch = time.perf_counter()
    print(
        f"[if_grm] dispatching {n} judge calls "
        f"across {len(ep_counts)} endpoints (pool_workers={max_workers}) — {ep_summary}"
    )

    def _call(args: tuple[str, list[dict[str, str]]]) -> str | None:
        ep, msgs = args
        return _request_completion(ep, msgs, config)

    raw_texts = list(pool.map(_call, zip(endpoints, messages_list)))

    elapsed = time.perf_counter() - t_dispatch
    verdicts = [parse_verdict(t) for t in raw_texts]
    n_pass  = sum(1 for v in verdicts if v is True)
    n_fail  = sum(1 for v in verdicts if v is False)
    n_err   = sum(1 for v in verdicts if v is None)
    print(
        f"[if_grm] all {n} judge calls returned in {elapsed:.2f}s — "
        f"pass={n_pass} fail={n_fail} error/timeout={n_err} "
        f"(pass_rate={n_pass/n*100:.1f}%)"
    )
    return verdicts
