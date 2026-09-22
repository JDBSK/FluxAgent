from types import SimpleNamespace

from entry.main import (
    _configure_console_streams,
    render_terminal_status,
    resumable_checkpoint,
)


def test_failed_budget_run_has_visible_terminal_message():
    message = render_terminal_status(
        {
            "task_status": "failed",
            "budget_exceeded_reason": "wall_timeout:524.46>300.0",
        }
    )

    assert "任务未完成" in message
    assert "524.46>300.0" in message


def test_cancelled_run_has_visible_terminal_message():
    assert "任务已取消" in render_terminal_status({"task_status": "cancelled"})


def test_successful_run_does_not_add_failure_message():
    assert render_terminal_status({"task_status": "done"}) == ""


def test_console_streams_replace_unencodable_ui_symbols(monkeypatch):
    class Stream:
        def __init__(self):
            self.errors = None

        def reconfigure(self, **kwargs):
            self.errors = kwargs.get("errors")

    stdout = Stream()
    stderr = Stream()
    monkeypatch.setattr("entry.main.sys.stdout", stdout)
    monkeypatch.setattr("entry.main.sys.stderr", stderr)

    _configure_console_streams()

    assert stdout.errors == "replace"
    assert stderr.errors == "replace"


def test_done_checkpoint_is_not_resumable():
    snapshot = SimpleNamespace(
        values={"task_status": "done"},
        next=(),
    )

    assert resumable_checkpoint(snapshot) is None


def test_running_checkpoint_with_next_node_is_resumable():
    snapshot = SimpleNamespace(
        values={"task_status": "running"},
        next=("agent",),
    )

    assert resumable_checkpoint(snapshot) == {
        "status": "running",
        "next": ("agent",),
        "pending_actions": 0,
        "active_subagents": 0,
    }


def test_waiting_approval_checkpoint_with_pending_action_is_resumable():
    snapshot = SimpleNamespace(
        values={
            "task_status": "waiting_approval",
            "pending_action_ids": ["action-1"],
        },
        next=(),
    )

    assert resumable_checkpoint(snapshot) == {
        "status": "waiting_approval",
        "next": (),
        "pending_actions": 1,
        "active_subagents": 0,
    }


def test_missing_checkpoint_is_not_resumable():
    assert resumable_checkpoint(None) is None
