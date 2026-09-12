"""Regression tests for bugs found by the multi-agent project audit."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from fluxagent.core.budget import RunBudget, budget_violation
from fluxagent.core.installer import (
    InstallError,
    install_mcp_source,
    list_installed_mcp_servers,
)
from fluxagent.core.logger import JSONLEventLogger
from fluxagent.core.recovery import ACTION_REPLAN, ACTION_RETRY, classify_failure
from fluxagent.core.safe_calc import safe_calculate
from fluxagent.core.tools import sandbox_tools
from fluxagent.mcp.registry import build_registry


def test_budget_reservation_blocks_one_extra_llm_call():
    """A state at the exact limit may finish, but may not reserve another call."""
    budget = RunBudget(max_llm_calls=2)
    state = {"llm_calls": 2}

    assert budget_violation(state, budget) is None
    assert budget_violation(state, budget, pending_llm_calls=1) == "llm_calls_exceeded:3>2"


def test_retry_limit_means_total_attempts():
    """max_retries=3 means the first attempt plus at most two retries."""
    for previous in (0, 1):
        decision = classify_failure(
            ["network timeout"],
            retry_count=previous,
            max_retries=3,
            replan_count=0,
            max_replans=1,
        )
        assert decision.action == ACTION_RETRY
        assert decision.retry_count == previous + 1

    exhausted = classify_failure(
        ["network timeout"],
        retry_count=2,
        max_retries=3,
        replan_count=0,
        max_replans=1,
    )
    assert exhausted.action == ACTION_REPLAN


def test_dynamic_skill_is_blocked_by_default_mcp_policy():
    """A dynamic Skill can execute shell, so MCP must treat it as side effecting."""

    @tool
    def project_skill(mode: str = "help") -> str:
        """Demo dynamic skill."""
        return mode

    with patch("fluxagent.mcp.registry.BUILTIN_TOOLS", []), patch(
        "fluxagent.mcp.registry.load_dynamic_skills", return_value=[project_skill]
    ):
        registry = build_registry(include_skills=True)

    assert registry.is_blocked("project_skill")
    result, is_error = registry.call("project_skill", {"mode": "run"})
    assert is_error
    assert "拒绝执行" in result


@pytest.mark.parametrize(
    "command",
    ["git --git-dir=../../outside status", "git -C../outside status"],
)
def test_shell_rejects_paths_hidden_inside_options(command):
    """Dash-prefixed options must not bypass traversal checks."""
    with pytest.raises(PermissionError):
        sandbox_tools._parse_command(command)


def test_mcp_index_failure_rolls_back_installed_tree():
    """Source files and registry entry are one installation transaction."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source"
        source.mkdir()
        (source / "server.py").write_text("print('ok')\n", encoding="utf-8")
        target = root / "servers"

        # 本用例验证的是"索引写失败 → 已安装目录回滚"的事务性。
        # 夹具里的 print 桩服务器无法完成 MCP initialize 握手，
        # 启动健康检查由 tests/test_installer.py 单独覆盖，这里替换为桩探针。
        with patch(
            "fluxagent.core.installer._write_mcp_index",
            side_effect=OSError("disk full"),
        ), patch(
            "fluxagent.core.installer._probe_mcp",
            return_value=["stub_tool"],
        ), pytest.raises(OSError, match="disk full"):
            install_mcp_source(
                "local",
                str(source),
                name="demo",
                command="python",
                args=["server.py"],
                target_root=target,
                index_path=target / "index.json",
            )

        assert not (target / "demo").exists()


def test_corrupt_mcp_index_fails_closed():
    """A broken registry must not silently become an empty registry."""
    with tempfile.TemporaryDirectory() as tmp:
        index = Path(tmp) / "index.json"
        index.write_text("{broken", encoding="utf-8")
        with pytest.raises(InstallError, match="索引损坏"):
            list_installed_mcp_servers(index)


def test_safe_calculator_rejects_power_bomb():
    """Nested powers must be rejected before allocating a huge integer."""
    result = safe_calculate("9**9**9")
    assert "error" in result
    assert "too large" in result


def test_logger_shutdown_is_idempotent():
    """Explicit shutdown followed by atexit shutdown must not block."""
    with tempfile.TemporaryDirectory() as tmp:
        logger = JSONLEventLogger(tmp)
        logger.log_event("session", "test", value=1)
        logger.shutdown()
        logger.shutdown()
        assert (Path(tmp) / "session.jsonl").is_file()


def test_planner_model_error_ends_without_entering_agent():
    """Planning provider failures must not be reported as a valid plan."""
    from fluxagent.core.agent import create_agent_app

    class FailingModel:
        def bind_tools(self, _tools):
            return self

        def invoke(self, *_args, **_kwargs):
            raise RuntimeError("provider unavailable")

    with patch("fluxagent.core.agent.get_provider", return_value=FailingModel()):
        app = create_agent_app(
            tools=[],
            checkpointer=MemorySaver(),
            require_approval=False,
        )
        config = {"configurable": {"thread_id": "planner-error"}}
        result = app.invoke(
            {"messages": [HumanMessage(content="先创建文件然后执行最后检查")]},
            config,
        )

    assert result["task_status"] == "failed"
    assert result["last_failure_class"] == "planner"
    assert result["llm_calls"] == 1


def test_broken_mcp_schema_is_not_advertised():
    """A broken schema must be omitted instead of published as no-argument."""
    from fluxagent.mcp.registry import ToolRegistry

    class BrokenTool:
        name = "broken"
        description = "broken schema"

        @property
        def args_schema(self):
            raise RuntimeError("schema failed")

    registry = ToolRegistry(tools=[BrokenTool()])
    assert registry.describe() == []
