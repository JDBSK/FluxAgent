import pytest

from fluxagent.core.recovery import (
    ACTION_ABORT,
    ACTION_REPLAN,
    CLASS_PERMANENT,
    CLASS_TRANSIENT,
    classify_failure,
    is_tool_failure_text,
)


@pytest.mark.parametrize(
    "text",
    [
        "FileNotFoundError: [Errno 2] No such file or directory",
        "参数错误：缺少必要参数 path",
        "路径不存在：G:/data/missing.csv",
    ],
)
def test_permanent_failure_replans_without_retry(text):
    decision = classify_failure(
        [text],
        retry_count=0,
        max_retries=3,
        replan_count=0,
        max_replans=2,
    )

    assert decision.action == ACTION_REPLAN
    assert decision.failure_class == CLASS_PERMANENT
    assert decision.retry_count == 0
    assert decision.replan_count == 1


def test_permanent_failure_aborts_without_replan_budget():
    decision = classify_failure(
        ["File not found: sales.csv"],
        retry_count=0,
        max_retries=3,
        replan_count=1,
        max_replans=1,
    )

    assert decision.action == ACTION_ABORT
    assert decision.failure_class == CLASS_PERMANENT
    assert decision.retry_count == 0


def test_transient_failure_still_retries():
    decision = classify_failure(
        ["❌ 执行失败：network timeout"],
        retry_count=0,
        max_retries=3,
        replan_count=0,
        max_replans=2,
    )

    assert decision.action != ACTION_REPLAN
    assert decision.failure_class == CLASS_TRANSIENT


def test_structured_tool_error_is_detected():
    text = '{"ok":false,"error":{"code":"command_timeout","message":"超时","retryable":true}}'
    assert is_tool_failure_text(text)
