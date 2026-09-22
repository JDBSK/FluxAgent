import pytest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from fluxagent.core.recovery import (
    ACTION_ABORT,
    ACTION_REPLAN,
    CLASS_PERMANENT,
    CLASS_TRANSIENT,
    any_tool_failure,
    classify_failure,
    extract_tool_call_batch,
    extract_tool_failure_texts,
    is_tool_failure_text,
    normalize_tool_result,
    tool_call_key,
    tool_result_failed,
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


def test_success_text_mentioning_an_error_is_not_a_failure():
    """旧实现用 ``"Error" in s[:80]``：结果里提到 Error 字样就被判成失败。"""
    assert not is_tool_failure_text("读取完成：共 3 行，其中两行包含 Error: none 的占位文本")


@pytest.mark.parametrize(
    "text",
    ["Error: connection reset by peer", "  Timeout: 30s exceeded", "失败：网络不可用"],
)
def test_error_like_prefixes_are_still_failures(text):
    assert is_tool_failure_text(text)


@pytest.mark.parametrize("text", ['{"ok": true}', '{"ok": "true"}', '{"ok": 1}'])
def test_ok_true_payloads_are_success(text):
    assert not is_tool_failure_text(text)


@pytest.mark.parametrize("text", ['{"ok": false}', '{"ok": "false"}', '{"ok": 0}'])
def test_ok_false_payloads_are_failures(text):
    """工具是手写代码：``ok`` 写成字符串时不能漏报失败。"""
    assert is_tool_failure_text(text)


def test_ok_null_is_not_guessed():
    assert not is_tool_failure_text('{"ok": null, "rows": 3}')


def test_tool_result_failed_prefers_execution_status_over_content():
    ok_content = ToolMessage(content="done:1", tool_call_id="c1", name="t")
    hard_failed = ToolMessage(
        content="done:1", tool_call_id="c2", name="t", status="error"
    )

    assert not tool_result_failed(ok_content)
    assert tool_result_failed(hard_failed)


def test_normalize_tool_result_keeps_content_and_adds_state():
    message = ToolMessage(content="done:1", tool_call_id="c1", name="t", status="error")

    normalized = normalize_tool_result(message)

    assert normalized.content == "done:1"
    assert normalized.status == "error"
    assert normalized.additional_kwargs["fluxagent_result"] == {"ok": False}
    # 文本本身没有失败特征，只有执行状态说了算
    assert not is_tool_failure_text(normalized.content)


def test_missing_tool_result_counts_as_failure():
    messages = [
        HumanMessage(content="跑两个工具"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "a", "args": {}, "id": "c1"},
                {"name": "b", "args": {}, "id": "c2"},
            ],
        ),
        ToolMessage(content="ok:1", tool_call_id="c1", name="a"),
    ]

    texts = extract_tool_failure_texts(messages)

    assert any("missing_tool_result:c2" in text for text in texts)
    assert any_tool_failure(texts)


def test_consumed_batch_is_not_rejudged():
    """模型已经用文本回答过的批次算已消化，不能拿旧结果反复触发重试。"""
    messages = [
        HumanMessage(content="跑一下"),
        AIMessage(
            content="",
            tool_calls=[{"name": "a", "args": {}, "id": "c1"}],
        ),
        ToolMessage(content="❌ 执行失败", tool_call_id="c1", name="a"),
        AIMessage(content="这次没成功，我换个方式"),
    ]

    assert extract_tool_call_batch(messages) == []
    assert extract_tool_failure_texts(messages) == []


def test_tool_call_key_ignores_argument_order_only():
    assert tool_call_key({"name": "t", "args": {"a": 1, "b": 2}}) == tool_call_key(
        {"name": "t", "args": {"b": 2, "a": 1}}
    )
    assert tool_call_key({"name": "t", "args": {"a": 1}}) != tool_call_key(
        {"name": "t", "args": {"a": 2}}
    )
    assert tool_call_key({"name": "t", "args": {}}) != tool_call_key(
        {"name": "other", "args": {}}
    )
