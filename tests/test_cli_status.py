from entry.main import render_terminal_status


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
