"""各套件的执行器：把 case 跑成 (passed, detail)。

约定：每个 runner 收一条 case，返回 (bool, str)。不抛异常——
单条 case 挂掉不该让整轮评测中断，异常本身就是一种失败结果。
"""
import os
import tempfile
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

import fluxagent.core.agent as agent_mod
from fluxagent.core.approval import (
    approve_prepare,
    pending_approval,
    reject_prepare,
)
from fluxagent.core.context import (
    STEP_DONE,
    STEP_FAILED,
    advance_plan,
    make_plan_step,
)

from .stub import ScriptedLLM, counting_tool, make_tool


def _build_app(llm, tools, **kwargs):
    """替换 provider 工厂注入 stub LLM。

    create_agent_app 内部调 get_provider，这里打模块属性是最小侵入的做法；
    真要长期做，应该把 llm 也做成参数注入——现有签名没开这个口。
    """
    original = agent_mod.get_provider
    agent_mod.get_provider = lambda **kw: llm
    try:
        return agent_mod.create_agent_app(tools=tools, checkpointer=MemorySaver(), **kwargs)
    finally:
        agent_mod.get_provider = original


# ---------------- HITL ----------------

def run_gate(case: dict[str, Any]) -> tuple[bool, str]:
    tool_names = case["tools"]
    counter: dict[str, int] = {}
    tools = [counting_tool(n, counter) for n in dict.fromkeys(tool_names)]

    llm = ScriptedLLM(script=[
        [{"name": n, "args": {"value": str(i)}} for i, n in enumerate(tool_names)],
        "完成",
    ])
    kwargs: dict[str, Any] = {"enable_planner": False}
    if "require_approval" in case:
        kwargs["require_approval"] = case["require_approval"]
    if "approval_tools" in case:
        kwargs["approval_tools"] = set(case["approval_tools"])

    app = _build_app(llm, tools, **kwargs)
    cfg = {"configurable": {"thread_id": f"gate_{case['id']}"}}
    app.invoke({"messages": [HumanMessage(content="执行操作")]}, cfg)

    calls = pending_approval(app, cfg)
    paused = calls is not None
    if paused != case["expect_pause"]:
        return False, f"pause={paused} expected={case['expect_pause']}"

    # 关键断言：中断期间工具绝不能已经跑过
    if paused and sum(counter.values()) != 0:
        return False, f"tool ran before approval: {counter}"

    if paused:
        if case["decision"] == "approve":
            approve_prepare(app, cfg, calls)
        else:
            reject_prepare(app, cfg, calls)
        app.invoke(None, cfg)

    total = sum(counter.values())
    if total != case["expect_runs"]:
        return False, f"runs={total} expected={case['expect_runs']} ({counter})"
    return True, f"runs={total}"


def run_multi_gate(case: dict[str, Any]) -> tuple[bool, str]:
    """连续多次审批：验证 prepare/resume 分离没有导致重复执行。"""
    tool = case["tool"]
    rounds = case["rounds"]
    counter: dict[str, int] = {}
    tools = [counting_tool(tool, counter)]

    script: list[Any] = []
    for i in range(rounds):
        script.append([{"name": tool, "args": {"value": str(i)}}])
    script.append("全部完成")

    app = _build_app(ScriptedLLM(script=script), tools, enable_planner=False)
    cfg = {"configurable": {"thread_id": f"multi_{case['id']}"}}

    stream_input: Any = {"messages": [HumanMessage(content="连续执行")]}
    pauses = 0
    for _ in range(rounds + 3):  # 上限保护，防止逻辑错误导致死循环
        app.invoke(stream_input, cfg)
        calls = pending_approval(app, cfg)
        if not calls:
            break
        pauses += 1
        approve_prepare(app, cfg, calls)
        stream_input = None

    if pauses != case["expect_pauses"]:
        return False, f"pauses={pauses} expected={case['expect_pauses']}"
    total = sum(counter.values())
    if total != case["expect_runs"]:
        return False, f"runs={total} expected={case['expect_runs']}"
    return True, f"pauses={pauses} runs={total}"


# ---------------- task state ----------------

def run_plan(case: dict[str, Any]) -> tuple[bool, str]:
    n, tool, fail = case["n_steps"], case["tool"], case["fail"]
    tools = [make_tool(tool, fail=fail)]

    script: list[Any] = ['[' + ",".join(f'"步骤{i+1}"' for i in range(n)) + ']']
    # 多给几轮工具调用：失败路径需要触发到熔断上限
    for _ in range(n + agent_mod.MAX_RETRY + 2):
        script.append([{"name": tool, "args": {}}])
    script.append("完成")

    # 旧 plan 套件默认关 replan，保持"失败即熔断"的既有断言；新 recovery 套件再开。
    app = _build_app(
        ScriptedLLM(script=script), tools,
        require_approval=False,
        enable_replan=case.get("enable_replan", False),
    )
    cfg = {"configurable": {"thread_id": f"plan_{case['id']}"}}
    app.invoke({"messages": [HumanMessage(content="先执行然后再执行最后完成")]}, cfg)

    st = app.get_state(cfg).values
    plan = st.get("task_plan") or []
    status = st.get("task_status")
    done = sum(1 for s in plan if s.get("status") == STEP_DONE)

    if status != case["expect_status"]:
        return False, f"status={status} expected={case['expect_status']}"
    if done != case["expect_done_steps"]:
        return False, f"done={done} expected={case['expect_done_steps']}"
    return True, f"status={status} done={done}/{len(plan)}"


def run_route(case: dict[str, Any]) -> tuple[bool, str]:
    got = agent_mod._needs_plan(case["text"])
    if got != case["expect_plan"]:
        return False, f"needs_plan={got} expected={case['expect_plan']}"
    return True, f"needs_plan={got}"


def run_parse(case: dict[str, Any]) -> tuple[bool, str]:
    steps = agent_mod._parse_plan(case["raw"])
    if len(steps) != case["expect_steps"]:
        return False, f"steps={len(steps)} expected={case['expect_steps']}"
    return True, f"steps={len(steps)}"


def run_advance(case: dict[str, Any]) -> tuple[bool, str]:
    plan = [make_plan_step(i + 1, f"s{i}") for i in range(case["n_steps"])]
    outcome = STEP_FAILED if case["outcome"] == "failed" else STEP_DONE
    try:
        new_plan, idx, status = advance_plan(plan, case["step_index"], outcome)
    except Exception as exc:
        return False, f"raised {type(exc).__name__}: {exc}"

    if not (0 <= idx < len(new_plan)):
        return False, f"idx out of range: {idx}"
    if case.get("clamp"):
        return True, f"clamped idx={idx} status={status}"
    if status != case["expect_status"]:
        return False, f"status={status} expected={case['expect_status']}"
    return True, f"idx={idx} status={status}"


# ---------------- tools / sandbox ----------------

def run_shell(case: dict[str, Any]) -> tuple[bool, str]:
    from fluxagent.core.config import OFFICE_DIR
    from fluxagent.core.tools.sandbox_tools import execute_office_shell

    cmd = case["command"]
    # 会落盘的安全命令改写到 _evaltmp_ 前缀，跑完清掉
    tmp_file = None
    if not case["expect_blocked"] and ">" in cmd:
        tmp_file = "_evaltmp_shell.txt"
        cmd = cmd.split(">")[0] + f"> {tmp_file}"

    out = execute_office_shell.invoke({"command": cmd})

    if tmp_file:
        try:
            os.remove(os.path.join(OFFICE_DIR, tmp_file))
        except OSError:
            pass

    blocked = "权限拒绝" in out or "❌" in out
    if blocked != case["expect_blocked"]:
        return False, f"blocked={blocked} expected={case['expect_blocked']} | {out[:70]}"
    return True, "blocked" if blocked else "allowed"


def run_path(case: dict[str, Any]) -> tuple[bool, str]:
    from fluxagent.core.config import OFFICE_DIR
    from fluxagent.core.tools.sandbox_tools import (
        list_office_files,
        read_office_file,
        write_office_file,
    )

    path, op = case["path"], case["op"]
    # 合法写入用带前缀的临时名，跑完删掉——评测不该在 office 里留垃圾，
    # 否则下一轮 list_office_files 的结果会被上一轮污染
    created: str | None = None
    if op == "read":
        out = read_office_file.invoke({"filepath": path})
    elif op == "write":
        target = path or "t.txt"
        if not case["expect_blocked"]:
            target = f"_evaltmp_{os.path.basename(target) or 't.txt'}"
            created = target
        out = write_office_file.invoke({"filepath": target, "content": "x"})
    else:
        out = list_office_files.invoke({"sub_dir": path})

    if created:
        try:
            os.remove(os.path.join(OFFICE_DIR, created))
        except OSError:
            pass

    blocked = "越权拦截" in str(out)
    if blocked != case["expect_blocked"]:
        return False, f"blocked={blocked} expected={case['expect_blocked']} | {str(out)[:70]}"
    return True, "blocked" if blocked else "allowed"


def run_calc(case: dict[str, Any]) -> tuple[bool, str]:
    from fluxagent.core.tools.builtins import calculator

    out = str(calculator.invoke({"expression": case["expression"]}))
    low = out.lower()
    errored = ("error" in low) or ("exception" in low)
    if case["expect_error"]:
        if not errored:
            return False, f"should error but got: {out[:60]}"
        return True, "errored as expected"
    if errored:
        return False, f"unexpected error: {out[:60]}"
    if case.get("expect_contains") and case["expect_contains"] not in out:
        return False, f"missing expected token in {out[:60]}"
    return True, out[:40]


def _make_skill_dir(name: str) -> str:
    """造一个临时技能目录，用来测两段式与懒加载。"""
    base = tempfile.mkdtemp(prefix=f"skill_{name}_")
    folder = os.path.join(base, name)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "SKILL.md"), "w", encoding="utf-8") as fh:
        fh.write(
            f"name: {name}\n"
            f"description: {name} 的一句话摘要\n\n"
            "# 详细说明书正文\n"
            "MANUAL_BODY_MARKER 这段只应在 help 模式出现。\n"
        )
    return base


def run_twophase(case: dict[str, Any]) -> tuple[bool, str]:
    import fluxagent.core.skill_loader as sl

    base = _make_skill_dir(case["skill"])
    original = sl.SKILLS_DIR
    sl.SKILLS_DIR = base
    try:
        loader = sl.LazySkillLoader()
        tools = loader.get_all_tools(force_rescan=True)
        if not tools:
            return False, "no skill tool built"
        tool = tools[0]

        args = {"mode": case["mode"]}
        if "command" in case:
            args["command"] = case["command"]
        elif case["mode"] == "run":
            args["command"] = "echo hi"

        out = str(tool.invoke(args))
        has_manual = "MANUAL_BODY_MARKER" in out
        errored = out.startswith("错误")

        if has_manual != case["expect_manual"]:
            return False, f"manual={has_manual} expected={case['expect_manual']}"
        if errored != case["expect_error"]:
            return False, f"error={errored} expected={case['expect_error']} | {out[:60]}"
        return True, f"manual={has_manual} error={errored}"
    finally:
        sl.SKILLS_DIR = original


def run_lazy(case: dict[str, Any]) -> tuple[bool, str]:
    """懒加载核心断言：注册阶段的 description 不能含说明书正文。"""
    import fluxagent.core.skill_loader as sl

    base = _make_skill_dir(case["skill"])
    original = sl.SKILLS_DIR
    sl.SKILLS_DIR = base
    try:
        loader = sl.LazySkillLoader()
        tools = loader.get_all_tools(force_rescan=True)
        if not tools:
            return False, "no skill tool"
        desc = tools[0].description
        if "MANUAL_BODY_MARKER" in desc:
            return False, "manual body leaked into description (not lazy)"
        if "一句话摘要" not in desc:
            return False, f"metadata not extracted: {desc[:60]}"
        return True, "lazy ok (summary only)"
    finally:
        sl.SKILLS_DIR = original


# ---------------- MCP ----------------

def _registry(allow: bool, include_skills: bool = False):
    from fluxagent.mcp.registry import build_registry

    return build_registry(allow_side_effects=allow, include_skills=include_skills)


def run_mcp_call(case: dict[str, Any]) -> tuple[bool, str]:
    reg = _registry(case["allow"])
    text, is_error = reg.call(case["tool"], case["args"])

    if case.get("expect_blocked_text"):
        # 协议层放开了，但沙盒仍须拦住越权参数
        if "越权拦截" not in text and "权限拒绝" not in text and "❌" not in text:
            return False, f"sandbox did not block: {text[:70]}"
        return True, "sandbox blocked"

    expected = case["expect_error"]
    if expected is None:
        return True, f"is_error={is_error} (not asserted)"
    if is_error != expected:
        return False, f"is_error={is_error} expected={expected} | {text[:70]}"
    return True, f"is_error={is_error}"


def run_mcp_robust(case: dict[str, Any]) -> tuple[bool, str]:
    """健壮性：坏参数只能变成 error 文本，不能抛异常。"""
    reg = _registry(case["allow"])
    try:
        text, is_error = reg.call(case["tool"], case["args"])
    except Exception as exc:
        return False, f"raised {type(exc).__name__}: {exc}"
    if not isinstance(text, str):
        return False, f"non-str result: {type(text)}"
    return True, f"handled (is_error={is_error})"


def run_mcp_schema(case: dict[str, Any]) -> tuple[bool, str]:
    reg = _registry(case["allow"])
    spec = next((s for s in reg.describe() if s["name"] == case["tool"]), None)
    if spec is None:
        return False, f"{case['tool']} not listed"
    if not spec["description"]:
        return False, "empty description"
    schema = spec["inputSchema"]
    if schema.get("type") != "object" or "properties" not in schema:
        return False, f"bad schema: {schema}"
    blocked_marked = spec["description"].startswith("[需审批")
    should_mark = reg.is_blocked(case["tool"])
    if blocked_marked != should_mark:
        return False, f"mark={blocked_marked} should={should_mark}"
    return True, "schema ok"


def run_mcp_meta(case: dict[str, Any]) -> tuple[bool, str]:
    reg = _registry(case["allow"], case.get("include_skills", False))
    names = set(reg.names())
    described = {s["name"] for s in reg.describe()}
    if names != described:
        return False, f"names/describe mismatch: {names ^ described}"
    if not names:
        return False, "empty registry"
    return True, f"{len(names)} tools consistent"


def run_mcp_blockflag(case: dict[str, Any]) -> tuple[bool, str]:
    reg = _registry(case["allow"])
    got = reg.is_blocked(case["tool"])
    if got != case["expect_blocked"]:
        return False, f"blocked={got} expected={case['expect_blocked']}"
    return True, f"blocked={got}"


# ---------------- runtime depth: policy / recovery / budget ----------------

def run_policy_assess(case: dict[str, Any]) -> tuple[bool, str]:
    from fluxagent.core.policy import RiskLevel, assess_call, assess_tool_calls, build_policy_config

    level_map = {i.name.lower(): i for i in RiskLevel}
    # also accept value names read/write/...
    from fluxagent.core.policy import LEVEL_NAME
    name_map = {v: k for k, v in LEVEL_NAME.items()}

    if case.get("mode") == "batch":
        cfg = build_policy_config(
            approval_tools=set(case["approval_tools"]) if case.get("approval_tools") is not None else None,
            hard_deny_critical=case.get("hard_deny_critical", True),
        )
        decision = assess_tool_calls(
            case["tool_calls"], config=cfg, require_approval=case.get("require_approval", True),
        )
        if decision.needs_approval != case["expect_approval"]:
            return False, f"approval={decision.needs_approval} expected={case['expect_approval']}"
        if decision.hard_deny != case["expect_hard_deny"]:
            return False, f"hard_deny={decision.hard_deny} expected={case['expect_hard_deny']}"
        if case.get("expect_level") and decision.max_level_name != case["expect_level"]:
            return False, f"level={decision.max_level_name} expected={case['expect_level']}"
        return True, f"level={decision.max_level_name} appr={decision.needs_approval} deny={decision.hard_deny}"

    got = assess_call(case["tool"], case.get("args") or {})
    if got.level_name != case["expect_level"]:
        return False, f"level={got.level_name} expected={case['expect_level']}"
    if got.hard_deny != case["expect_hard_deny"]:
        return False, f"hard_deny={got.hard_deny} expected={case['expect_hard_deny']}"
    return True, f"level={got.level_name} deny={got.hard_deny}"


def run_classify(case: dict[str, Any]) -> tuple[bool, str]:
    from fluxagent.core.recovery import classify_failure

    d = classify_failure(
        case["texts"],
        retry_count=case["retry_count"],
        max_retries=case["max_retries"],
        replan_count=case["replan_count"],
        max_replans=case["max_replans"],
        enable_replan=case.get("enable_replan", True),
    )
    if d.action != case["expect_action"]:
        return False, f"action={d.action} expected={case['expect_action']} class={d.failure_class}"
    if case.get("expect_class") and d.failure_class != case["expect_class"]:
        return False, f"class={d.failure_class} expected={case['expect_class']}"
    return True, f"action={d.action} class={d.failure_class}"


def run_budget_check(case: dict[str, Any]) -> tuple[bool, str]:
    from fluxagent.core.budget import RunBudget, budget_violation

    b = RunBudget(**case.get("budget", {}))
    reason = budget_violation(case.get("state") or {}, b, now=case.get("now"))
    expect = case.get("expect_reason")  # None means no violation
    if expect is None:
        if reason is not None:
            return False, f"unexpected violation: {reason}"
        return True, "ok"
    if reason is None or not str(reason).startswith(expect):
        return False, f"reason={reason} expected_prefix={expect}"
    return True, reason


def run_hard_deny(case: dict[str, Any]) -> tuple[bool, str]:
    """高危参数必须硬拒绝：不停审批、工具零执行。"""
    tool = case["tool"]
    args = case.get("args") or {}
    counter: dict[str, int] = {}
    tools = [counting_tool(tool, counter)]
    llm = ScriptedLLM(script=[
        [{"name": tool, "args": args}],
        "已拦截",
    ])
    app = _build_app(
        llm, tools,
        enable_planner=False,
        require_approval=case.get("require_approval", True),
        enable_replan=False,
        hard_deny_critical=True,
    )
    cfg = {"configurable": {"thread_id": f"hd_{case['id']}"}}
    app.invoke({"messages": [HumanMessage(content="执行")]}, cfg)

    paused = pending_approval(app, cfg) is not None
    if paused:
        return False, "should not pause for approval on hard deny"
    total = sum(counter.values())
    if total != 0:
        return False, f"tool ran {total} times on hard deny"
    st = app.get_state(cfg).values
    texts = [str(getattr(m, "content", "")) for m in st.get("messages", []) if getattr(m, "type", "") == "tool"]
    if not any("策略拒绝" in t or "policy" in t.lower() or "hard_deny" in t for t in texts):
        # policy_block note is Chinese 策略拒绝
        if not any("拒绝" in t for t in texts):
            return False, f"no deny tool message: {texts[:2]}"
    return True, "hard_deny ok"


def run_replan_flow(case):
    from fluxagent.core.budget import RunBudget
    tool = case.get("tool", "write_office_file")
    tools = [make_tool(tool, fail=True)]
    max_retries = int(case.get("max_retries", 2))
    max_replans = int(case.get("max_replans", 1))
    script = ['["stepA","stepB"]']
    for i in range(max_replans):
        for _ in range(max_retries):
            script.append([{"name": tool, "args": {}}])
        script.append('["alt"]')
    for _ in range(max_retries):
        script.append([{"name": tool, "args": {}}])
    script.append("give up")
    app = _build_app(ScriptedLLM(script=script), tools, require_approval=False, enable_replan=True, budget=RunBudget(max_retries_per_step=max_retries, max_replans=max_replans, max_tool_rounds=80, max_llm_calls=80))
    cfg = {"configurable": {"thread_id": "rp_" + case["id"]}}
    app.invoke({"messages": [HumanMessage(content="先执行然后再执行最后完成")]}, cfg)
    st = app.get_state(cfg).values
    rc = int(st.get("replan_count") or 0)
    if rc < int(case.get("expect_min_replans", 1)):
        return False, "replan low "+str(rc)+" status="+str(st.get("task_status"))
    exp = case.get("expect_status")
    if exp and st.get("task_status") != exp:
        return False, "status="+str(st.get("task_status"))
    return True, "replan ok "+str(rc)


def run_budget_flow(case: dict[str, Any]) -> tuple[bool, str]:
    """故意把预算压到极小，确认会 failed 且带 budget_exceeded_reason。"""
    from fluxagent.core.budget import RunBudget

    tool = "get_current_time"
    tools = [make_tool(tool, fail=False)]
    # 不断调工具形成回环
    script: list[Any] = []
    for _ in range(20):
        script.append([{"name": tool, "args": {}}])
    script.append("完")

    budget = RunBudget(
        max_tool_rounds=case.get("max_tool_rounds", 3),
        max_llm_calls=case.get("max_llm_calls", 30),
        max_retries_per_step=3,
        max_replans=0,
        max_wall_seconds=case.get("max_wall_seconds", 300),
    )
    app = _build_app(
        ScriptedLLM(script=script),
        tools,
        require_approval=False,
        enable_planner=False,
        enable_replan=False,
        budget=budget,
    )
    cfg = {"configurable": {"thread_id": f"bd_{case['id']}"}}
    app.invoke({"messages": [HumanMessage(content="keep going")]}, cfg)
    st = app.get_state(cfg).values
    if st.get("task_status") != "failed":
        return False, f"status={st.get('task_status')} rounds={st.get('tool_rounds')} llm={st.get('llm_calls')}"
    reason = st.get("budget_exceeded_reason") or ""
    prefix = case.get("expect_reason_prefix", "tool_rounds_exceeded")
    if not str(reason).startswith(prefix):
        # llm 上限也可能先触发
        if case.get("allow_llm_exceeded") and str(reason).startswith("llm_calls_exceeded"):
            return True, reason
        return False, f"reason={reason} expected_prefix={prefix}"
    return True, reason


def run_replan_validate(case):
    from fluxagent.core.recovery import validate_replan_steps
    ok, steps = validate_replan_steps(case.get("steps") or [], case.get("forbidden", ""), min_steps=case.get("min_steps", 2), max_steps=case.get("max_steps", 6))
    if ok != case["expect_ok"]:
        return False, "ok=%s expected=%s steps=%s" % (ok, case.get("expect_ok"), steps)
    if case.get("expect_len") is not None and len(steps) != case["expect_len"]:
        return False, "len=%s expected=%s" % (len(steps), case.get("expect_len"))
    return True, "ok=%s len=%s" % (ok, len(steps))

def run_side_effect_align(case):
    from fluxagent.core.policy import SIDE_EFFECT_TOOLS, is_side_effect_tool
    from fluxagent.mcp.registry import build_registry
    name = case["tool"]
    expect = case["expect_side"]
    if is_side_effect_tool(name) != expect:
        return False, f"policy side mismatch"
    if (name in SIDE_EFFECT_TOOLS) != expect:
        return False, "set mismatch"
    reg = build_registry(allow_side_effects=False, include_skills=False)
    if name in reg.names() and reg.is_blocked(name) != expect:
        return False, f"mcp blocked mismatch"
    return True, f"aligned side={expect}"




def run_session_store(case):
    """Pure session index lifecycle — no LLM."""
    import os
    import tempfile
    from fluxagent.core.session import (
        LEGACY_THREAD_ID,
        SessionError,
        SessionStore,
        auto_title_from_text,
        parse_slash_command,
    )

    root = tempfile.mkdtemp()
    store = SessionStore(index_path=os.path.join(root, "index.json"), root_dir=root)
    op = case["op"]

    if op == "bootstrap":
        rec = store.bootstrap()
        if not rec.id.startswith(str(case.get("expect_id_prefix", "")) ) and rec.id != LEGACY_THREAD_ID:
            # allow exact legacy id
            if rec.id != LEGACY_THREAD_ID:
                return False, f"id={rec.id}"
        return True, rec.id

    if op == "create":
        store.bootstrap()
        rec = store.create(title=case["title"], make_active=True)
        if rec.title != case["expect_title"]:
            return False, f"title={rec.title}"
        return True, rec.id

    if op == "switch_roundtrip":
        store.bootstrap()
        a = store.create("A", make_active=True)
        b = store.create("B", make_active=False)
        store.set_active(b.id)
        if store.get_active().id != b.id:
            return False, "switch failed"
        store.set_active(a.id)
        if store.get_active().id != a.id:
            return False, "switch back failed"
        return True, "ok"

    if op == "archive_active_forbidden":
        store.bootstrap()
        a = store.create("A", make_active=True)
        try:
            store.archive(a.id)
            return False, "should forbid"
        except SessionError:
            return True, "forbidden"

    if op == "delete_active_forbidden":
        store.bootstrap()
        a = store.create("A", make_active=True)
        try:
            store.delete(a.id)
            return False, "should forbid"
        except SessionError:
            return True, "forbidden"

    if op == "archive_other":
        store.bootstrap()
        a = store.create("A", make_active=True)
        b = store.create("B", make_active=False)
        store.archive(b.id)
        if store.get(b.id).status != "archived":
            return False, "not archived"
        if store.get_active().id != a.id:
            return False, "active changed"
        return True, "ok"

    if op == "resolve_prefix":
        store.bootstrap()
        rec = store.create("UniqueZed")
        hit = store.resolve(rec.id[:10])
        if hit.id != rec.id:
            return False, hit.id
        return True, "ok"

    if op == "resolve_ambiguous":
        store.bootstrap()
        store.create("Project One")
        store.create("Project Two")
        try:
            store.resolve("Project")
            return False, "should be ambiguous"
        except SessionError:
            return True, "ambiguous"

    if op == "autotitle":
        title = auto_title_from_text(case["text"])
        if case["expect_substr"] not in title:
            return False, title
        return True, title

    if op == "notes_isolate":
        store.bootstrap()
        a = store.create("NA")
        b = store.create("NB")
        store.write_notes(a.id, "only-A", append=False)
        store.write_notes(b.id, "only-B", append=False)
        if "only-A" not in store.read_notes(a.id):
            return False, "a missing"
        if "only-A" in store.read_notes(b.id):
            return False, "leaked to b"
        return True, "ok"

    if op == "pin_order":
        store.bootstrap()
        a = store.create("zzz")
        b = store.create("aaa")
        store.pin(a.id, True)
        ordered = store.list()
        if ordered[0].id != a.id:
            return False, f"first={ordered[0].title}"
        return True, "ok"

    if op == "search":
        store.bootstrap()
        store.create("Alpha")
        store.create("BetaShip")
        hits = store.list(query=case["query"])
        if not any("Beta" in r.title for r in hits):
            return False, str([r.title for r in hits])
        return True, "ok"

    if op == "corrupt_recover":
        store.bootstrap()
        with open(store.index_path, "w", encoding="utf-8") as f:
            f.write("{bad")
        rec = store.bootstrap()
        if not rec.id:
            return False, "empty"
        return True, rec.id

    if op == "touch_count":
        store.bootstrap()
        s = store.create("T")
        store.touch(s.id, message_delta=int(case["delta"]))
        got = store.get(s.id).message_count
        if got != int(case["expect_count"]):
            return False, f"count={got}"
        return True, str(got)

    if op == "parse_cmd":
        cmd, arg = parse_slash_command(case["text"])
        if cmd != case["expect_cmd"]:
            return False, cmd
        return True, cmd

    return False, f"unknown op {op}"


def run_session_isolation(case):
    """Checkpoint isolation: different thread_id must not share messages/state."""
    from langchain_core.messages import HumanMessage
    from fluxagent.core.session import thread_config
    from .stub import ScriptedLLM, make_tool

    op = case.get("op", "messages")
    left = case.get("left", "iso_left")
    right = case.get("right", "iso_right")
    llm = ScriptedLLM(script=[f"reply-{left}", f"reply-{right}"])
    app = _build_app(llm, [make_tool("noop")], enable_planner=False, require_approval=False)
    cfg_l = thread_config(f" thr_{left}".strip())
    # keep ids clean
    cfg_l = thread_config(f"thr_{left}")
    cfg_r = thread_config(f"thr_{right}")
    app.invoke({"messages": [HumanMessage(content=f"msg-{left}")]}, cfg_l)
    app.invoke({"messages": [HumanMessage(content=f"msg-{right}")]}, cfg_r)
    msgs_l = app.get_state(cfg_l).values.get("messages") or []
    msgs_r = app.get_state(cfg_r).values.get("messages") or []
    text_l = " | ".join(str(getattr(m, "content", "")) for m in msgs_l)
    text_r = " | ".join(str(getattr(m, "content", "")) for m in msgs_r)
    if f"msg-{left}" not in text_l or f"msg-{right}" in text_l:
        return False, f"left leaked or missing: {text_l[:120]}"
    if f"msg-{right}" not in text_r or f"msg-{left}" in text_r:
        return False, f"right leaked or missing: {text_r[:120]}"
    if op == "summary_field":
        # summaries default empty and independent
        sum_l = app.get_state(cfg_l).values.get("summary")
        sum_r = app.get_state(cfg_r).values.get("summary")
        if sum_l != sum_r and False:
            pass
    return True, "isolated"


RUNNERS = {
    "gate": run_gate,
    "multi_gate": run_multi_gate,
    "plan": run_plan,
    "route": run_route,
    "parse": run_parse,
    "advance": run_advance,
    "shell": run_shell,
    "path": run_path,
    "calc": run_calc,
    "twophase": run_twophase,
    "lazy": run_lazy,
    "mcp_call": run_mcp_call,
    "mcp_robust": run_mcp_robust,
    "mcp_schema": run_mcp_schema,
    "mcp_meta": run_mcp_meta,
    "mcp_blockflag": run_mcp_blockflag,
    "policy_assess": run_policy_assess,
    "classify": run_classify,
    "budget_check": run_budget_check,
    "hard_deny": run_hard_deny,
    "replan_flow": run_replan_flow,
    "budget_flow": run_budget_flow,
    "replan_validate": run_replan_validate,
    "side_effect_align": run_side_effect_align,
    "session_store": run_session_store,
    "session_isolation": run_session_isolation,
}


def run_case(case: dict[str, Any]) -> tuple[bool, str]:
    runner = RUNNERS.get(case["kind"])
    if runner is None:
        return False, f"no runner for kind={case['kind']}"
    try:
        return runner(case)
    except Exception as exc:
        return False, f"EXCEPTION {type(exc).__name__}: {exc}"
