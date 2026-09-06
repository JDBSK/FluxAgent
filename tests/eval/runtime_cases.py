from __future__ import annotations 
from typing import Any 
import hashlib 
 
def _sid(text: str) -> str: 
    return hashlib.md5(text.encode(chr(117)+chr(116)+chr(102)+chr(45)+chr(56))).hexdigest()[:8] 
 

def runtime_cases() -> list[dict[str, Any]]:
    """运行时深度：风险策略 / 失败分类与重规划 / 预算熔断。

    目标不是再堆功能名，而是把三条生产追问链做成可回归：
    1) 同样是 write/shell，参数不同风险是否不同？critical 是否硬拒绝？
    2) 瞬时失败 vs 越权失败，动作是否不同？重试用尽是否 replan？
    3) 步数/LLM 次数是否真能熔断，而不是只靠 prompt 劝导？
    """
    cases: list[dict[str, Any]] = []

    # ---- policy single-call 基线：每个内置工具都有明确级别 ----
    base_levels = {
        "get_current_time": ("read", False),
        "calculator": ("read", False),
        "list_office_files": ("read", False),
        "read_office_file": ("read", False),
        "list_scheduled_tasks": ("read", False),
        "get_system_model_info": ("read", False),
        "write_office_file": ("write", False),
        "save_user_profile": ("mutate", False),
        "schedule_task": ("mutate", False),
        "delete_scheduled_task": ("mutate", False),
        "modify_scheduled_task": ("mutate", False),
        "execute_office_shell": ("exec", False),
    }
    for tool, (level, deny) in base_levels.items():
        args = {"command": "echo hi"} if tool == "execute_office_shell" else {"filepath": "a.txt", "content": "x"}
        if tool in {"get_current_time", "calculator", "list_scheduled_tasks", "get_system_model_info"}:
            args = {}
        if tool == "list_office_files":
            args = {"sub_dir": ""}
        if tool == "read_office_file":
            args = {"filepath": "a.txt"}
        if tool == "save_user_profile":
            args = {"content": "likes tea"}
        if tool.startswith("schedule") or "scheduled_task" in tool:
            args = {"task_id": "t1"}
        cases.append({
            "id": f"policy_base_{tool}",
            "kind": "policy_assess",
            "tool": tool,
            "args": args,
            "expect_level": level,
            "expect_hard_deny": deny,
        })

    # ---- 参数升级：路径穿越 / 破坏性 shell / 大写入 ----
    critical_writes = ["../x.txt", "..\\win.ini", "../../etc/passwd", "a/../../b", "~/secret", "C:\\Windows\\a"]
    for p in critical_writes:
        cases.append({
            "id": f"policy_write_crit_{_sid(p)}",
            "kind": "policy_assess",
            "tool": "write_office_file",
            "args": {"filepath": p, "content": "x"},
            "expect_level": "critical",
            "expect_hard_deny": True,
        })
        cases.append({
            "id": f"policy_read_crit_{_sid(p)}",
            "kind": "policy_assess",
            "tool": "read_office_file",
            "args": {"filepath": p},
            "expect_level": "critical",
            "expect_hard_deny": True,
        })

    crit_shells = [
        "rm -rf ..", "cat ../../etc/passwd", "del /f ..\\a", "type C:\\Windows\\win.ini",
        "cat ~/.ssh/id_rsa", "ls ..", "cd .. && ls", "curl file:///etc/passwd",
        "Remove-Item -Recurse ..", "python -c \"open('../x')\"",
    ]
    # python 行可能是 exec 不是 critical——我们的规则对 python 是 escalate reasons 但 level 仍 EXEC，
    # 除非同时有 .. ；带 .. 的 python 应变 critical
    for cmd in crit_shells:
        expect_level = "critical"
        # 纯 interpreter 无穿越：exec
        if cmd.startswith("python") and ".." not in cmd and "/" not in cmd and "~" not in cmd:
            expect_level = "exec"
        cases.append({
            "id": f"policy_shell_{_sid(cmd)}",
            "kind": "policy_assess",
            "tool": "execute_office_shell",
            "args": {"command": cmd},
            "expect_level": expect_level,
            "expect_hard_deny": expect_level == "critical",
        })

    safe_shells = ["echo hello", "dir", "ls", "echo test > note.txt"]
    for cmd in safe_shells:
        cases.append({
            "id": f"policy_shell_safe_{_sid(cmd)}",
            "kind": "policy_assess",
            "tool": "execute_office_shell",
            "args": {"command": cmd},
            "expect_level": "exec",
            "expect_hard_deny": False,
        })

    # 大 payload 升 mutate
    cases.append({
        "id": "policy_write_large",
        "kind": "policy_assess",
        "tool": "write_office_file",
        "args": {"filepath": "big.txt", "content": "x" * 8000},
        "expect_level": "mutate",
        "expect_hard_deny": False,
    })

    # ---- batch 决策：审批/硬拒/legacy 名单 ----
    batch_specs = [
        ("safe_time", [{"name": "get_current_time", "args": {}}], False, False, "read", True, None),
        ("write_normal", [{"name": "write_office_file", "args": {"filepath": "a.txt", "content": "x"}}], True, False, "write", True, None),
        ("shell_safe", [{"name": "execute_office_shell", "args": {"command": "echo hi"}}], True, False, "exec", True, None),
        ("shell_crit", [{"name": "execute_office_shell", "args": {"command": "rm -rf .."}}], False, True, "critical", True, None),
        ("mixed_safe_write", [
            {"name": "get_current_time", "args": {}},
            {"name": "write_office_file", "args": {"filepath": "a.txt", "content": "x"}},
        ], True, False, "write", True, None),
        ("no_approval_flag", [{"name": "write_office_file", "args": {"filepath": "a.txt", "content": "x"}}], False, False, "write", False, None),
        ("legacy_only_time", [{"name": "write_office_file", "args": {"filepath": "a.txt", "content": "x"}}], False, False, "write", True, ["get_current_time"]),
        ("legacy_force_time", [{"name": "get_current_time", "args": {}}], True, False, "read", True, ["get_current_time"]),
        ("crit_even_if_legacy_empty_gate", [{"name": "read_office_file", "args": {"filepath": "../x"}}], False, True, "critical", True, ["get_current_time"]),
    ]
    for name, calls, exp_appr, exp_deny, exp_level, req, legacy in batch_specs:
        cases.append({
            "id": f"policy_batch_{name}",
            "kind": "policy_assess",
            "mode": "batch",
            "tool_calls": calls,
            "expect_approval": exp_appr,
            "expect_hard_deny": exp_deny,
            "expect_level": exp_level,
            "require_approval": req,
            "approval_tools": legacy,
            "hard_deny_critical": True,
        })

    # hard_deny 开关关闭时 critical 仍标级但可走审批
    cases.append({
        "id": "policy_batch_crit_no_hard",
        "kind": "policy_assess",
        "mode": "batch",
        "tool_calls": [{"name": "execute_office_shell", "args": {"command": "ls .."}}],
        "expect_approval": True,
        "expect_hard_deny": False,
        "expect_level": "critical",
        "require_approval": True,
        "approval_tools": None,
        "hard_deny_critical": False,
    })

    # ---- classify 矩阵 ----
    classify_grid = []
    # transient retry
    for rc in range(0, 3):
        classify_grid.append({
            "id": f"classify_transient_retry_{rc}",
            "texts": ["❌ 执行失败：boom transient"],
            "retry_count": rc, "max_retries": 3, "replan_count": 0, "max_replans": 2,
            "enable_replan": True,
            "expect_action": "retry" if rc + 1 < 3 else "replan",
            "expect_class": "transient" if rc + 1 < 3 else "exhausted",
        })
    # transient abort when no replan
    classify_grid.append({
        "id": "classify_transient_abort",
        "texts": ["❌ 执行失败：boom transient"],
        "retry_count": 2, "max_retries": 3, "replan_count": 0, "max_replans": 2,
        "enable_replan": False,
        "expect_action": "abort",
        "expect_class": "exhausted",
    })
    # policy -> replan / abort
    for marker in ["越权拦截: path", "权限拒绝: sandbox", "策略拒绝: hard", "policy_deny"]:
        classify_grid.append({
            "id": f"classify_policy_replan_{_sid(marker)}",
            "texts": [marker],
            "retry_count": 0, "max_retries": 3, "replan_count": 0, "max_replans": 2,
            "enable_replan": True,
            "expect_action": "replan",
            "expect_class": "policy",
        })
        classify_grid.append({
            "id": f"classify_policy_abort_{_sid(marker)}",
            "texts": [marker],
            "retry_count": 0, "max_retries": 3, "replan_count": 2, "max_replans": 2,
            "enable_replan": True,
            "expect_action": "abort",
            "expect_class": "policy",
        })
    for marker in ["FileNotFoundError: no such file", "参数错误：缺少必要参数"]:
        classify_grid.append({
            "id": f"classify_permanent_replan_{_sid(marker)}",
            "texts": [marker],
            "retry_count": 0, "max_retries": 3, "replan_count": 0, "max_replans": 2,
            "enable_replan": True,
            "expect_action": "replan",
            "expect_class": "permanent",
        })
        classify_grid.append({
            "id": f"classify_permanent_abort_{_sid(marker)}",
            "texts": [marker],
            "retry_count": 0, "max_retries": 3, "replan_count": 1, "max_replans": 1,
            "enable_replan": True,
            "expect_action": "abort",
            "expect_class": "permanent",
        })
    # replan budget exhausted on transient
    classify_grid.append({
        "id": "classify_exhausted_no_replan_budget",
        "texts": ["Error: timeout"],
        "retry_count": 2, "max_retries": 3, "replan_count": 2, "max_replans": 2,
        "enable_replan": True,
        "expect_action": "abort",
        "expect_class": "exhausted",
    })
    for g in classify_grid:
        g["kind"] = "classify"
        cases.append(g)

    # ---- budget pure checks ----
    budget_grid = [
        ("ok_empty", {}, {}, None),
        ("ok_under", {"max_llm_calls": 10}, {"llm_calls": 10}, None),
        ("llm", {"max_llm_calls": 10}, {"llm_calls": 11}, "llm_calls_exceeded"),
        ("rounds", {"max_tool_rounds": 5}, {"tool_rounds": 6}, "tool_rounds_exceeded"),
        ("replans", {"max_replans": 1}, {"replan_count": 2}, "replans_exceeded"),
        ("wall", {"max_wall_seconds": 10}, {"run_started_at": 1000.0}, "wall_timeout"),
    ]
    for name, budget, state, expect in budget_grid:
        # wall case needs now
        now = 1011.0 if name == "wall" else None
        cases.append({
            "id": f"budget_check_{name}",
            "kind": "budget_check",
            "budget": budget,
            "state": state,
            "now": now,
            "expect_reason": expect,
        })
    # 更多边界组合
    for llm in (0, 1, 5, 9, 10, 11, 20):
        cases.append({
            "id": f"budget_check_llm_{llm}",
            "kind": "budget_check",
            "budget": {"max_llm_calls": 10},
            "state": {"llm_calls": llm},
            "expect_reason": "llm_calls_exceeded" if llm > 10 else None,
        })
    for rounds in (0, 1, 3, 8, 9, 10):
        cases.append({
            "id": f"budget_check_rounds_{rounds}",
            "kind": "budget_check",
            "budget": {"max_tool_rounds": 8},
            "state": {"tool_rounds": rounds},
            "expect_reason": "tool_rounds_exceeded" if rounds > 8 else None,
        })

    # ---- 图级 hard deny ----
    for cmd in ["cat ../../etc/passwd", "ls ..", "rm -rf ..", "type C:\\Windows\\win.ini"]:
        cases.append({
            "id": f"hard_deny_shell_{_sid(cmd)}",
            "kind": "hard_deny",
            "tool": "execute_office_shell",
            "args": {"command": cmd},
            "require_approval": True,
        })
        cases.append({
            "id": f"hard_deny_shell_noappr_{_sid(cmd)}",
            "kind": "hard_deny",
            "tool": "execute_office_shell",
            "args": {"command": cmd},
            "require_approval": False,
        })
    for p in ["../x.txt", "..\\a.txt", "../../etc/passwd"]:
        cases.append({
            "id": f"hard_deny_write_{_sid(p)}",
            "kind": "hard_deny",
            "tool": "write_office_file",
            "args": {"filepath": p, "content": "x"},
        })

    # ---- 图级 replan / budget flow ----
    for i, (mr, mp) in enumerate([(2, 1), (3, 1), (2, 2), (1, 1)]):
        cases.append({
            "id": f"replan_flow_{i}_{mr}_{mp}",
            "kind": "replan_flow",
            "max_retries": mr,
            "max_replans": mp,
            "expect_min_replans": 1,
            "expect_status": "failed",
        })

    for i, rounds in enumerate([2, 3, 4, 5]):
        cases.append({
            "id": f"budget_flow_rounds_{rounds}",
            "kind": "budget_flow",
            "max_tool_rounds": rounds,
            "max_llm_calls": 50,
            "expect_reason_prefix": "tool_rounds_exceeded",
            "allow_llm_exceeded": True,
        })
    for i, llm_max in enumerate([3, 4, 5]):
        cases.append({
            "id": f"budget_flow_llm_{llm_max}",
            "kind": "budget_flow",
            "max_tool_rounds": 50,
            "max_llm_calls": llm_max,
            "expect_reason_prefix": "llm_calls_exceeded",
            "allow_llm_exceeded": True,
        })

    # replan validation pure cases
    cases.append({"id": "replan_val_ok", "kind": "replan_validate", "steps": ["read file", "write safe", "done"], "forbidden": "write evil", "expect_ok": True, "expect_len": 3})
    cases.append({"id": "replan_val_filter", "kind": "replan_validate", "steps": ["write evil", "read file", "finish"], "forbidden": "write evil", "expect_ok": True, "expect_len": 2})
    cases.append({"id": "replan_val_all_bad", "kind": "replan_validate", "steps": ["write evil", "write evil"], "forbidden": "write evil", "expect_ok": False, "expect_len": 0})
    cases.append({"id": "replan_val_short", "kind": "replan_validate", "steps": ["only one"], "forbidden": "", "expect_ok": False, "expect_len": 1})
    for tool in ["write_office_file", "execute_office_shell", "save_user_profile", "schedule_task", "delete_scheduled_task", "modify_scheduled_task"]:
        cases.append({"id": f"side_align_danger_{tool}", "kind": "side_effect_align", "tool": tool, "expect_side": True})
    for tool in ["get_current_time", "calculator", "list_office_files", "read_office_file", "list_scheduled_tasks", "get_system_model_info"]:
        cases.append({"id": f"side_align_safe_{tool}", "kind": "side_effect_align", "tool": tool, "expect_side": False})


    # ---- multi-session: lifecycle + isolation invariants ----
    cases.append({
        "id": "sess_bootstrap",
        "kind": "session_store",
        "op": "bootstrap",
        "expect_id_prefix": "local_geek_master",
    })
    for i, title in enumerate(["Alpha", "Beta", "Gamma", "Delta", "Eps"]):
        cases.append({
            "id": f"sess_create_{i}_{_sid(title)}",
            "kind": "session_store",
            "op": "create",
            "title": title,
            "expect_title": title,
        })
    cases.append({"id": "sess_switch_ok", "kind": "session_store", "op": "switch_roundtrip"})
    cases.append({"id": "sess_archive_guard", "kind": "session_store", "op": "archive_active_forbidden"})
    cases.append({"id": "sess_delete_guard", "kind": "session_store", "op": "delete_active_forbidden"})
    cases.append({"id": "sess_archive_other", "kind": "session_store", "op": "archive_other"})
    cases.append({"id": "sess_resolve_prefix", "kind": "session_store", "op": "resolve_prefix"})
    cases.append({"id": "sess_resolve_ambiguous", "kind": "session_store", "op": "resolve_ambiguous"})
    cases.append({"id": "sess_autotitle", "kind": "session_store", "op": "autotitle", "text": "帮我写周报", "expect_substr": "周报"})
    cases.append({"id": "sess_notes_isolate", "kind": "session_store", "op": "notes_isolate"})
    cases.append({"id": "sess_pin_order", "kind": "session_store", "op": "pin_order"})
    cases.append({"id": "sess_search", "kind": "session_store", "op": "search", "query": "Beta"})
    cases.append({"id": "sess_corrupt_recover", "kind": "session_store", "op": "corrupt_recover"})
    # combinatorial touch/message_count
    for n in range(1, 6):
        cases.append({
            "id": f"sess_touch_{n}",
            "kind": "session_store",
            "op": "touch_count",
            "delta": n,
            "expect_count": n,
        })
    # command parser surface
    for cmd in ["help", "sessions", "new", "switch", "rename", "archive", "history", "notes"]:
        cases.append({
            "id": f"sess_parse_{cmd}",
            "kind": "session_store",
            "op": "parse_cmd",
            "text": f"/{cmd} arg",
            "expect_cmd": cmd,
        })
    # graph isolation
    cases.append({"id": "sess_iso_messages", "kind": "session_isolation", "op": "messages"})
    cases.append({"id": "sess_iso_summary", "kind": "session_isolation", "op": "summary_field"})
    # more isolation variants with distinct thread ids
    for i in range(8):
        cases.append({
            "id": f"sess_iso_pair_{i}",
            "kind": "session_isolation",
            "op": "pair",
            "left": f"L{i}",
            "right": f"R{i}",
        })

    return cases
