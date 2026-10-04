# -*- coding: utf-8 -*-
"""验证已确认工具经验的事件持久化和重启恢复，不调用 LLM。"""
from __future__ import annotations

import json
import os
import tempfile


def main() -> int:
    from fluxagent.capability.memory_events import MemoryEventLog, MemoryProjector
    from fluxagent.capability.memory_store import MemoryScope, MemoryStore

    with tempfile.TemporaryDirectory() as tmp:
        event_path = os.path.join(tmp, "memory_events.sqlite3")
        snapshot_path = os.path.join(tmp, "structured_memory.json")
        log = MemoryEventLog(event_path)
        store = MemoryStore(persist_path=snapshot_path)
        projector = MemoryProjector(log, store)
        event = log.append(
            "tool_failed",
            {
                "verified": True,
                "key": "tool.weather.timeout",
                "value": "天气工具曾因网络超时失败",
                "ttl_seconds": 86400,
                "capability_tags": ["mcp.weather"],
            },
            session_id="persistence-eval",
            run_id="run-1",
            capability_id="mcp.weather",
        )
        projected = projector.process_one()
        restored = MemoryStore(persist_path=snapshot_path)
        loaded = restored.load(snapshot_path)
        recovered = restored.query(
            scopes=[MemoryScope.EPISODIC],
            capability_id="mcp.weather",
            session_id="persistence-eval",
        )
        duplicate = projector.apply(event)
        checks = [
            {"check": "event_projected", "passed": projected},
            {"check": "episodic_restored_after_reload", "passed": loaded == 1 and len(recovered) == 1},
            {"check": "duplicate_projection_ignored", "passed": duplicate is False},
        ]
        report = {"checks": checks, "all_passed": all(item["passed"] for item in checks)}

    os.makedirs("artifacts", exist_ok=True)
    path = os.path.join("artifacts", "memory_persistence_eval.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    for item in checks:
        print(f"  {'PASS' if item['passed'] else 'FAIL'}  {item['check']}")
    print(f"all_passed={report['all_passed']} -> {path}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
