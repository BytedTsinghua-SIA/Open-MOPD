from types import SimpleNamespace

from verl.experimental.agent_loop.agent_loop import AgentLoopManager


def _manager(*, hybrid: bool, free_cache_engine: bool) -> AgentLoopManager:
    manager = AgentLoopManager.__new__(AgentLoopManager)
    manager.worker_group = object() if hybrid else None
    manager.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(free_cache_engine=free_cache_engine),
        )
    )
    return manager


def test_hybrid_lifecycle_switch_is_required_without_free_cache():
    assert _manager(hybrid=True, free_cache_engine=False)._needs_rollout_lifecycle_switch()


def test_standalone_lifecycle_switch_follows_free_cache_setting():
    assert not _manager(hybrid=False, free_cache_engine=False)._needs_rollout_lifecycle_switch()
    assert _manager(hybrid=False, free_cache_engine=True)._needs_rollout_lifecycle_switch()
