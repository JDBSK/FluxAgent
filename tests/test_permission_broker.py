import tempfile
import threading
import time
from pathlib import Path

from fluxagent.core.permission_broker import ActionStatus, PermissionBroker


def broker(tmp: str, ttl: float = 60) -> PermissionBroker:
    return PermissionBroker(Path(tmp) / "state.sqlite3", ttl_seconds=ttl)


def test_action_lifecycle_and_single_claim():
    with tempfile.TemporaryDirectory() as directory:
        with broker(directory) as store:
            action = store.register(
                thread_id="parent",
                tool_call_id="call-1",
                tool_name="write_office_file",
                args={"filepath": "x.txt", "content": "x"},
                actor="subagent",
                child_run_id="child",
            )
            assert action.status == ActionStatus.PENDING
            assert store.approve(action.action_id)
            assert store.claim(action.action_id)
            assert not store.claim(action.action_id)
            assert store.succeed(action.action_id, "done")
            assert not store.succeed(action.action_id, "twice")
            saved = store.get(action.action_id)
            assert saved is not None
            assert saved.status == ActionStatus.SUCCEEDED
            assert saved.result == "done"


def test_concurrent_claim_has_exactly_one_winner():
    with tempfile.TemporaryDirectory() as directory:
        with broker(directory) as store:
            action = store.register(
                thread_id="parent",
                tool_call_id="call-1",
                tool_name="write_office_file",
                args={"filepath": "x.txt", "content": "x"},
                actor="subagent",
                child_run_id="child",
            )
            assert store.approve(action.action_id)
            barrier = threading.Barrier(8)
            results: list[bool] = []

            def claim() -> None:
                barrier.wait()
                results.append(store.claim(action.action_id))

            threads = [threading.Thread(target=claim) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            assert results.count(True) == 1


def test_expiry_rejection_and_restart_marks_executing_unknown():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "state.sqlite3"
        expiring = PermissionBroker(path, ttl_seconds=0.01)
        expired = expiring.register(
            thread_id="parent",
            tool_call_id="expire",
            tool_name="write_office_file",
            args={},
            actor="main_agent",
        )
        time.sleep(0.02)
        assert expiring.get(expired.action_id).status == ActionStatus.EXPIRED

        action = expiring.register(
            thread_id="parent",
            tool_call_id="unknown",
            tool_name="write_office_file",
            args={},
            actor="main_agent",
        )
        assert expiring.approve(action.action_id)
        assert expiring.claim(action.action_id)
        expiring.close()

        with PermissionBroker(path) as reopened:
            restored = reopened.get(action.action_id)
            assert restored is not None
            assert restored.status == ActionStatus.UNKNOWN
            assert restored.error_code == "process_interrupted"


def test_same_tool_and_args_are_distinct_actions():
    with tempfile.TemporaryDirectory() as directory:
        with broker(directory) as store:
            first = store.register(
                thread_id="parent",
                tool_call_id="call-1",
                tool_name="write_office_file",
                args={"filepath": "x.txt", "content": "x"},
                actor="main_agent",
            )
            second = store.register(
                thread_id="parent",
                tool_call_id="call-2",
                tool_name="write_office_file",
                args={"filepath": "x.txt", "content": "x"},
                actor="main_agent",
            )
            assert first.action_id != second.action_id
            assert store.approve(first.action_id)
            assert store.get(second.action_id).status == ActionStatus.PENDING
