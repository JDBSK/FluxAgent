"""评测样例生成器。

为什么用组合生成而不是手写 400 条 JSON：
手写必然重复且覆盖不均；组合生成能保证每个维度（工具类型 × 是否危险 ×
批准/拒绝 × 步数）都被穷举到，加新维度时样例量自动增长。
每条 case 带 expect 字段，断言的是"该拦有没有拦、该跑有没有跑"这种
可判定的事实，不是"回答得好不好"——后者交给 judge 小样本抽测。
"""
import hashlib
from typing import Any

from .runtime_cases import runtime_cases


def _sid(text: str) -> str:
    """稳定短 ID。

    不能用内置 hash()：CPython 对 str 的 hash 带 PYTHONHASHSEED 随机化，
    每个进程结果都不同，case id 会在两次运行之间变化，回归基线永远对不上
    （实测第一版就是这个坑，基线比对报出 76 个"新增 case"）。
    md5 是确定性的，这里只用来生成可读 ID，不涉及安全。
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]

SAFE_TOOLS = ["get_current_time", "calculator", "list_office_files", "read_office_file"]
DANGER_TOOLS = [
    "write_office_file",
    "execute_office_shell",
    "save_user_profile",
    "schedule_task",
    "delete_scheduled_task",
    "modify_scheduled_task",
]

# 越权命令：沙盒必须全部拦下。覆盖 Unix/Win 两种写法和重定向绕过
ESCAPE_COMMANDS = [
    "cat ../../etc/passwd",
    "ls ..",
    "cd .. && ls",
    "cat /etc/shadow",
    "type C:\\Windows\\win.ini",
    "dir \\",
    "cat ~/.ssh/id_rsa",
    "ls ~",
    "cat </etc/passwd",
    "echo x > ../out.txt",
    "python -c \"import os;print(os.listdir('..'))\"",
    "node -e \"require('fs').readdirSync('..')\"",
    "cp ../secret.txt .",
    "more D:\\data.txt",
    "cat /proc/self/environ",
    "ls ../../",
    "cd ../.. ; pwd",
    "tar -cf x.tar ../",
    "cat /etc/hosts",
    "type E:\\secret",
    "rm -rf ~/data",
    "grep pass /etc/passwd",
    "cat ..\\..\\boot.ini",
    "echo test | tee ../leak.txt",
    "curl file:///etc/passwd",
    "ls /root",
    "cat ~/../etc/passwd",
    "find / -name id_rsa",
    "dir C:\\Users",
    "copy ..\\a.txt .",
]
# Shell metacharacters and redirection are intentionally forbidden now that
# execute_office_shell runs argv with shell=False.
SAFE_COMMANDS = ["echo hello", "dir", "ls", "echo test"]


def hitl_cases() -> list[dict[str, Any]]:
    """审批闸门：危险工具必须停、安全工具必须不停、批准跑、拒绝不跑。"""
    cases: list[dict[str, Any]] = []
    for tool in DANGER_TOOLS:
        for decision in ("approve", "reject"):
            for n_calls in (1, 2, 3):
                cases.append({
                    "id": f"hitl_gate_{tool}_{decision}_{n_calls}",
                    "kind": "gate",
                    "tools": [tool] * n_calls,
                    "decision": decision,
                    "expect_pause": True,
                    # 批准才真跑；拒绝时工具执行次数必须是 0
                    "expect_runs": n_calls if decision == "approve" else 0,
                })
    for tool in SAFE_TOOLS:
        for n_calls in (1, 2, 3):
            cases.append({
                "id": f"hitl_bypass_{tool}_{n_calls}",
                "kind": "gate",
                "tools": [tool] * n_calls,
                "decision": "none",
                "expect_pause": False,   # 查询类不该打断用户
                "expect_runs": n_calls,
            })
    # 自定义闸门集合：把安全工具显式加入审批集，验证 approval_tools 参数生效
    for tool in SAFE_TOOLS:
        for decision in ("approve", "reject"):
            cases.append({
                "id": f"hitl_custom_{tool}_{decision}",
                "kind": "gate",
                "tools": [tool],
                "decision": decision,
                "approval_tools": [tool],
                "expect_pause": True,
                "expect_runs": 1 if decision == "approve" else 0,
            })
    # 混合调用：一批里只要有一个危险工具，整批就得过闸门
    for safe in SAFE_TOOLS:
        for danger in DANGER_TOOLS:
            cases.append({
                "id": f"hitl_mixed_{safe}_{danger}",
                "kind": "gate",
                "tools": [safe, danger],
                "decision": "approve",
                "expect_pause": True,
                "expect_runs": 2,
            })
    # 关掉审批时不应有任何中断（验证开关真的生效）
    for tool in DANGER_TOOLS:
        for n_calls in (1, 2):
            cases.append({
                "id": f"hitl_disabled_{tool}_{n_calls}",
                "kind": "gate",
                "tools": [tool] * n_calls,
                "decision": "none",
                "require_approval": False,
                "expect_pause": False,
                "expect_runs": n_calls,
            })
    # 连续多轮审批：验证一次用户输入里多次中断都能正确处理，
    # 且 approve 后不会重复执行（这是 prepare/resume 分离的回归点）
    for tool in DANGER_TOOLS:
        for rounds in (2, 3):
            cases.append({
                "id": f"hitl_multi_{tool}_{rounds}",
                "kind": "multi_gate",
                "tool": tool,
                "rounds": rounds,
                "expect_pauses": rounds,
                "expect_runs": rounds,
            })
    return cases


def task_state_cases() -> list[dict[str, Any]]:
    """任务状态：计划生成、逐步推进、失败熔断、路由是否规划。"""
    cases: list[dict[str, Any]] = []

    # 正常推进：n 步计划跑完应为 done，且每步都 done
    for n_steps in range(1, 7):
        for tool in ["get_current_time", "calculator", "list_office_files"]:
            cases.append({
                "id": f"state_advance_{n_steps}_{tool}",
                "kind": "plan",
                "n_steps": n_steps,
                "tool": tool,
                "fail": False,
                "expect_status": "done",
                "expect_done_steps": n_steps,
            })

    # 失败熔断：连续失败达到 MAX_RETRY 应停在 failed，而不是无限回环
    for n_steps in range(1, 5):
        for tool in ["execute_office_shell", "read_office_file"]:
            cases.append({
                "id": f"state_fail_{n_steps}_{tool}",
                "kind": "plan",
                "n_steps": n_steps,
                "tool": tool,
                "fail": True,
                "expect_status": "failed",
                "expect_done_steps": 0,
            })

    # 计划解析：LLM 输出格式五花八门，parser 必须都能扛住
    plan_formats = [
        ('["a","b"]', 2),
        ('```json\n["a","b","c"]\n```', 3),
        ('好的，计划如下：["a","b"]', 2),
        ("1. a\n2. b\n3. c", 3),
        ("- a\n- b", 2),
        ("1) a\n2) b", 2),
        ("a\nb\nc\nd", 4),
        ('["only"]', 1),
        ('[]', 0),
        ("", 0),
        ('["a","b","c","d","e","f","g","h","i","j"]', 8),  # 上限截断到 8
        ('[1,2,3]', 3),
    ]
    for raw, n in plan_formats:
        cases.append({
            "id": f"parse_{_sid(raw)}", "kind": "parse",
            "raw": raw, "expect_steps": n,
        })

    # 路由：短问答不该触发 planner（省一次 LLM）；多步指令应该触发
    simple = [
        "现在几点", "你好", "1+1", "谢谢", "你是谁", "天气", "帮我算算", "在吗",
        "OK", "hello", "几号", "算一下 2*3",
    ]
    complex_ = [
        "先读取文件然后修改内容最后写回去",
        "帮我创建一个脚本然后运行它",
        "先列出目录再删除临时文件",
        "读取配置然后修改端口并重启",
        "依次执行三个步骤完成部署",
        "先备份数据接着清理日志最后生成报告",
        "先写一个爬虫脚本然后执行它并保存结果到文件",
        "帮我把项目里的日志文件整理一下然后生成一份统计报告最后发给我",
    ]
    for text in simple:
        cases.append({
            "id": f"route_simple_{text[:6]}", "kind": "route",
            "text": text, "expect_plan": False,
        })
    for text in complex_:
        cases.append({
            "id": f"route_complex_{text[:6]}", "kind": "route",
            "text": text, "expect_plan": True,
        })

    # 纯函数推进语义：成功推进 + 失败即 failed
    for n in range(1, 8):
        for idx in range(n):
            cases.append({
                "id": f"advance_{n}_{idx}", "kind": "advance",
                "n_steps": n, "step_index": idx, "outcome": "done",
                "expect_status": "done" if idx == n - 1 else "running",
            })
            cases.append({
                "id": f"advance_fail_{n}_{idx}", "kind": "advance",
                "n_steps": n, "step_index": idx, "outcome": "failed",
                "expect_status": "failed",
            })
    # 游标越界要被 clamp 住，不能 IndexError
    for n in (1, 3, 5):
        for idx in (n, n + 5, -1):
            cases.append({
                "id": f"advance_clamp_{n}_{idx}", "kind": "advance",
                "n_steps": n, "step_index": idx, "outcome": "done",
                "expect_status": "done" if idx != -1 or n == 1 else "running",
                "clamp": True,
            })
    return cases


def tools_cases() -> list[dict[str, Any]]:
    """工具与沙盒：越权必拦、安全放行、路径穿越必拦。"""
    cases: list[dict[str, Any]] = []
    for cmd in ESCAPE_COMMANDS:
        cases.append({
            "id": f"sandbox_block_{_sid(cmd)}",
            "kind": "shell", "command": cmd, "expect_blocked": True,
        })
    for cmd in SAFE_COMMANDS:
        cases.append({
            "id": f"sandbox_allow_{_sid(cmd)}",
            "kind": "shell", "command": cmd, "expect_blocked": False,
        })

    escape_paths = [
        "../secret.txt", "../../etc/passwd", "..\\..\\win.ini",
        "../", "a/../../b.txt", "subdir/../../escape.txt",
        "../../../root/.bashrc", "..", "./../out.txt",
        "sub/../../../etc/hosts",
    ]
    for p in escape_paths:
        for op in ("read", "write", "list"):
            cases.append({
                "id": f"path_block_{op}_{_sid(p)}",
                "kind": "path", "path": p, "op": op, "expect_blocked": True,
            })
    safe_paths = ["note.txt", "sub/note.txt", "deep/a/b.txt", ""]
    for p in safe_paths:
        for op in ("write", "read"):
            if op == "read" and p == "":
                continue
            cases.append({
                "id": f"path_allow_{op}_{_sid(p)}",
                "kind": "path", "path": p, "op": op, "expect_blocked": False,
            })

    # calculator 的边界：非法表达式必须返回错误文本而不是崩掉进程
    exprs = [
        ("1+1", "2"), ("25*48", "1200"), ("100/4", "25"), ("2**10", "1024"),
        ("(3+4)*2", "14"), ("10-3", "7"), ("7//2", "3"), ("9%4", "1"),
    ]
    for expr, expected in exprs:
        cases.append({
            "id": f"calc_{expr}", "kind": "calc",
            "expression": expr, "expect_contains": expected, "expect_error": False,
        })
    bad_exprs = ["1/0", "import os", "__import__('os')", "open('x')", "1+", "abc"]
    for expr in bad_exprs:
        cases.append({
            "id": f"calc_bad_{_sid(expr)}", "kind": "calc",
            "expression": expr, "expect_contains": "", "expect_error": True,
        })

    # 两段式技能：help 必须先给说明书，run 才执行
    for mode in ("help", "run", "invalid", ""):
        for skill in ("skill_a", "skill_b", "skill_c", "skill_d"):
            cases.append({
                "id": f"twophase_{skill}_{mode or 'empty'}", "kind": "twophase",
                "skill": skill, "mode": mode,
                "expect_manual": mode == "help",
                "expect_error": mode not in ("help", "run"),
            })
    # run 模式缺 command 必须报错，不能静默执行空命令
    for skill in ("skill_a", "skill_b"):
        cases.append({
            "id": f"twophase_nocmd_{skill}", "kind": "twophase",
            "skill": skill, "mode": "run", "command": "",
            "expect_manual": False, "expect_error": True,
        })
    # 懒加载：元数据扫描不该读全文（验证 description 只含摘要不含正文）
    for skill in ("skill_a", "skill_b", "skill_c"):
        cases.append({
            "id": f"lazy_{skill}", "kind": "lazy", "skill": skill,
        })
    return cases


def mcp_cases() -> list[dict[str, Any]]:
    """MCP：list 元数据完整、安全工具可调、副作用工具必拒。"""
    cases: list[dict[str, Any]] = []
    for tool in DANGER_TOOLS:
        cases.append({
            "id": f"mcp_block_{tool}", "kind": "mcp_call",
            "tool": tool, "args": {}, "allow": False, "expect_error": True,
        })
        cases.append({
            "id": f"mcp_open_{tool}", "kind": "mcp_call",
            "tool": tool, "args": {}, "allow": True, "expect_error": None,
        })
    safe_calls = [
        ("get_current_time", {}), ("calculator", {"expression": "2+2"}),
        ("calculator", {"expression": "10*10"}), ("calculator", {"expression": "7-3"}),
        ("list_office_files", {}), ("list_office_files", {"sub_dir": ""}),
        ("get_system_model_info", {}), ("list_scheduled_tasks", {}),
    ]
    for i, (tool, args) in enumerate(safe_calls):
        for allow in (False, True):
            cases.append({
                "id": f"mcp_safe_{tool}_{i}_{allow}", "kind": "mcp_call",
                "tool": tool, "args": args, "allow": allow, "expect_error": False,
            })
    unknown_tools = [
        "nope", "fake_tool", "", "write_office_file2", "../evil",
        "Calculator", "CALCULATOR", "calculator ", "os.system", "eval",
        "get_current_time2", "tool-with-dash",
    ]
    for tool in unknown_tools:
        for allow in (False, True):
            cases.append({
                "id": f"mcp_unknown_{_sid(tool)}_{allow}",
                "kind": "mcp_call",
                "tool": tool, "args": {}, "allow": allow, "expect_error": True,
            })
    # 参数缺失/多余：不能抛异常打断连接，必须返回 error 文本
    bad_args = [
        ("calculator", {}),
        ("calculator", {"wrong_key": "1"}),
        ("calculator", {"expression": "1+1", "extra": "x"}),
        ("read_office_file", {}),
        ("read_office_file", {"filepath": "not_exist_xyz.txt"}),
        ("list_office_files", {"sub_dir": "not_exist_dir"}),
    ]
    for i, (tool, args) in enumerate(bad_args):
        cases.append({
            "id": f"mcp_badargs_{tool}_{i}", "kind": "mcp_robust",
            "tool": tool, "args": args, "allow": True,
        })
    # 每个工具的 schema/描述都要能被 list 出来
    all_tools = SAFE_TOOLS + DANGER_TOOLS + ["get_system_model_info", "list_scheduled_tasks"]
    for tool in all_tools:
        for allow in (False, True):
            cases.append({
                "id": f"mcp_schema_{tool}_{allow}", "kind": "mcp_schema",
                "tool": tool, "allow": allow,
            })
    # 越权参数经 MCP 传入，沙盒仍须拦住（协议层不能成为绕过通道）
    escape_via_mcp = [
        "../x.txt", "../../etc/passwd", "..\\win.ini",
        "../../../secret", "a/../../b", "..",
    ]
    for p in escape_via_mcp:
        cases.append({
            "id": f"mcp_path_{_sid(p)}", "kind": "mcp_call",
            "tool": "read_office_file", "args": {"filepath": p},
            "allow": True, "expect_error": False, "expect_blocked_text": True,
        })
    # registry 一致性：names/describe 必须对齐，不能一个有一个没有
    for allow in (False, True):
        for include_skills in (False, True):
            cases.append({
                "id": f"mcp_consistency_{allow}_{include_skills}",
                "kind": "mcp_meta", "allow": allow, "include_skills": include_skills,
            })
    # is_blocked 判定必须与 SIDE_EFFECT_TOOLS 严格一致（安全边界不能漂移）
    for tool in DANGER_TOOLS:
        cases.append({
            "id": f"mcp_blockflag_danger_{tool}", "kind": "mcp_blockflag",
            "tool": tool, "allow": False, "expect_blocked": True,
        })
        cases.append({
            "id": f"mcp_blockflag_open_{tool}", "kind": "mcp_blockflag",
            "tool": tool, "allow": True, "expect_blocked": False,
        })
    for tool in SAFE_TOOLS + ["get_system_model_info", "list_scheduled_tasks"]:
        cases.append({
            "id": f"mcp_blockflag_safe_{tool}", "kind": "mcp_blockflag",
            "tool": tool, "allow": False, "expect_blocked": False,
        })
    # 危险工具在放开后仍要受沙盒约束（放开 ≠ 无限权限）
    for cmd in ["cat ../../etc/passwd", "ls ..", "cd .. && ls", "echo x > ../y"]:
        cases.append({
            "id": f"mcp_shell_sandbox_{_sid(cmd)}", "kind": "mcp_call",
            "tool": "execute_office_shell", "args": {"command": cmd},
            "allow": True, "expect_error": False, "expect_blocked_text": True,
        })
    return cases


SUITES = {
    "hitl": hitl_cases,
    "task_state": task_state_cases,
    "tools": tools_cases,
    "mcp": mcp_cases,
    "runtime": runtime_cases,
}


def all_cases() -> dict[str, list[dict[str, Any]]]:
    return {name: fn() for name, fn in SUITES.items()}
