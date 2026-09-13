# -*- coding: utf-8 -*-
"""记忆持久化端到端真实验证（收尾轮补充）。

单元测试覆盖了 MemoryStore 的存取语义；本脚本用真实模型验证完整链路：
  1. 会话回合 1 路由到某能力 → task_state 记忆写入并落盘（机械断言）；
  2. "重启"：全新装配的 Agent 实例从磁盘恢复记忆；
  3. 同能力续问 → task_state 重新注入编译上下文（机械断言 audit 的
     context_compiled.memory_records）；
  4. 对照组：无持久化文件的新实例不注入（证明注入来自恢复而非巧合）。

纯机械 oracle，不消耗 judge——持久化是否生效与模型输出质量无关。
产物：artifacts/memory_persistence_eval.json
"""
from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch

from langchain_core.messages import HumanMessage


def _real_llm_recorder():
    from fluxagent.core.config import load_dotenv

    load_dotenv()
    from fluxagent.core.provider import get_provider

    return get_provider(
        provider_name=os.getenv("DEFAULT_PROVIDER", "aliyun"),
        model_name=os.getenv("DEFAULT_MODEL", ""),
    )


class _AuditRecorder:
    def __init__(self):
        self.events: list[dict] = []

    def log_event(self, thread_id="", event="", **kwargs):
        self.events.append({"event": event, **kwargs})


def _cooking_tool():
    """合成 cooking 能力：让回合 1 真正路由成功（task_state 仅在
    route==route 时写入）；描述与生产 MCP 卡一致保证可检索。"""
    from langchain_core.tools import StructuredTool

    def _run(value: str = "") -> str:
        return f"菜谱推荐结果：{value or '家常菜三菜一汤'}"

    return StructuredTool.from_function(
        func=_run,
        name="mcp_cooking_meals",
        description="按人数忌口过敏原推荐中餐菜谱和一周膳食计划",
    )


def _run_turn(memory_dir: str, user_text: str, thread_id: str) -> tuple[_AuditRecorder, dict]:
    """独立装配一个 Agent 实例（= 一次"进程启动"）并跑一个用户回合。"""
    from fluxagent.core.agent import create_agent_app

    recorder = _AuditRecorder()
    with patch("fluxagent.core.agent.get_provider", return_value=_real_llm_recorder()), \
         patch("fluxagent.core.agent.MEMORY_DIR", memory_dir), \
         patch("fluxagent.core.agent.audit_logger", recorder):
        app = create_agent_app(
            tools=[_cooking_tool()], enable_planner=False, require_approval=False
        )
        result = app.invoke(
            {"messages": [HumanMessage(content=user_text)]},
            config={"configurable": {"thread_id": thread_id}},
        )
    return recorder, result


def main() -> int:
    summary: dict = {"turns": [], "checks": []}
    with tempfile.TemporaryDirectory() as tmp:
        memory_dir = os.path.join(tmp, "memory")
        os.makedirs(memory_dir, exist_ok=True)
        persist_path = os.path.join(memory_dir, "structured_memory.json")

        # 回合 1：路由到 cooking 能力 → task_state 写入（compile 之后写，
        # 本回合不自我注入），变更即落盘
        print("[turn 1] 建立任务状态 ...", flush=True)
        recorder1, _ = _run_turn(
            memory_dir, "用 cooking 服务推荐四人的晚餐", "eval_mem"
        )
        compiled_1 = [e for e in recorder1.events if e.get("event") == "context_compiled"]
        task_in_turn1 = any(
            "task_state:" in rid
            for e in compiled_1
            for rid in e.get("memory_records", [])
        )
        persisted = os.path.exists(persist_path)
        on_disk_task = False
        if persisted:
            with open(persist_path, encoding="utf-8") as fh:
                on_disk_task = any(
                    r.get("scope") == "task_state" for r in json.load(fh)["records"]
                )
        summary["turns"].append(
            {"turn": 1, "compiled_events": len(compiled_1), "answer": ""}
        )
        summary["checks"].append(
            {"check": "turn1_persisted_task_state_on_disk", "passed": persisted and on_disk_task}
        )

        # 回合 2（"重启"后的新实例，同 thread_id）：同能力续问 →
        # 回合 1 的 task_state 从磁盘恢复并重新注入
        print("[turn 2] 重启后同能力续问 ...", flush=True)
        recorder2, result2 = _run_turn(
            memory_dir, "继续刚才的 cooking 任务，把人数改成两人", "eval_mem"
        )
        compiled_2 = [e for e in recorder2.events if e.get("event") == "context_compiled"]
        # 只看回合内【第一次】编译：此刻存储刚从磁盘恢复，task_state 若在，
        # 只可能来自回合 1 的持久化（回合自身的写入发生在 compile 之后）
        first_compile_2 = compiled_2[0] if compiled_2 else {}
        recovered_ids = list(first_compile_2.get("memory_records", []))
        task_reinjected = any("task_state:" in rid for rid in recovered_ids)
        summary["turns"].append(
            {
                "turn": 2,
                "recovered_memory_ids": recovered_ids,
                "answer": str(result2["messages"][-1].content)[:300],
            }
        )
        summary["checks"].append(
            {"check": "turn2_reinjected_task_state_after_restart", "passed": task_reinjected}
        )

        # 对照组：全新目录（无持久化文件）→ 不应注入任何 task_state
        print("[control] 无持久化对照 ...", flush=True)
        with tempfile.TemporaryDirectory() as empty_tmp:
            empty_dir = os.path.join(empty_tmp, "memory")
            os.makedirs(empty_dir, exist_ok=True)
            recorder3, _ = _run_turn(
                empty_dir, "继续刚才的 cooking 任务，把人数改成两人", "eval_mem"
            )
            compiled_3 = [
                e for e in recorder3.events if e.get("event") == "context_compiled"
            ]
            first_compile_3 = compiled_3[0] if compiled_3 else {}
            control_injected = any(
                "task_state:" in rid
                for rid in first_compile_3.get("memory_records", [])
            )
        summary["checks"].append(
            {"check": "control_no_task_state_without_persisted_file", "passed": not control_injected}
        )

    summary["all_passed"] = all(c["passed"] for c in summary["checks"])
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/memory_persistence_eval.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    for check in summary["checks"]:
        print(f"  {'PASS' if check['passed'] else 'FAIL'}  {check['check']}")
    print(f"all_passed={summary['all_passed']} -> artifacts/memory_persistence_eval.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
