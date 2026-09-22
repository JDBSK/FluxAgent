# -*- coding: utf-8 -*-
"""审计日志实时监控终端：tail -f JSONL 日志文件并格式化显示。
"""

import time
import json
import os
from rich.console import Console
from rich.theme import Theme
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich import box
from datetime import datetime

from fluxagent.core.session import SessionError, SessionStore


cyber_theme = Theme({
    "info": "dim cyan",
    "warning": "color(141)",
    "error": "bold red",
    "llm_input": "dim white",
    "tool_call": "bold yellow",
    "tool_result": "bold green",
    "ai_message": "bold bright_magenta",
    "graph_transition": "bold bright_cyan",
    "timestamp": "dim white"
})  # 解析每行事件，按照类型进行渲染

console = Console(theme=cyber_theme)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

def clear_screen():
    """清空当前终端屏幕。"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """渲染 简约斜体版·FluxAgent 监控面板"""
    clear_screen()
    monster = (
        "  ▄█▄▄█▄  \n"
        " ▀██████▀ \n"
        " ██▄██▄██ \n"
        "  ▀    ▀  "
    )
    

    content = Text(justify="center")
    content.append("\n  Live Stream  \n\n", style="bold white italic")
    content.append(monster + "\n\n", style="color(120)")
    content.append("   What is FluxAgent doing?    \n", style="dim white italic") 

    panel = Panel(
        Align.center(content),  
        title="[bold color(74)] FluxAgent [/bold color(74)]",
        title_align="left",
        border_style="color(74)",
        box=box.ROUNDED,
        width=42,               
        padding=0
    )

    console.print(Align.center(panel))
    console.print()

def resolve_log_file(session_ref: str | None = None) -> str:
    """解决显式或当前活动会话的 JSONL 日志路径。找到会话记录后，返回其日志文件路径；找不到会话则抛出 SessionError。"""
    store = SessionStore()  # 初始化会话存储
    active = store.bootstrap()
    record = store.resolve(session_ref, include_archived=True) if session_ref else active
    safe_id = "".join(c for c in record.id if c.isalnum() or c in "-_") or "default"
    return os.path.join(LOG_DIR, f"{safe_id}.jsonl")


def tail_f(path_provider):
    """尾随一个日志文件,并在活动会话更改时重新打开它。类似tail -f从日志尾只读取新产生的JSON事件,但支持动态切换文件。"""
    current_path = ""
    stream = None
    waiting_for = ""
    try:
        while True:
            filepath = path_provider() if callable(path_provider) else str(path_provider)
            if filepath != current_path:
                if stream is not None:
                    stream.close()
                stream = None
                current_path = filepath
                waiting_for = ""
            if stream is None:
                if not os.path.exists(filepath):
                    if waiting_for != filepath:
                        console.print(f"[warning]等待日志文件生成: {filepath}[/warning]")
                        waiting_for = filepath
                    time.sleep(0.5)
                    continue
                stream = open(filepath, "r", encoding="utf-8")
                stream.seek(0, 2)
                console.print(f"[info]监听会话日志: {filepath}[/info]")
            line = stream.readline()
            if not line:
                time.sleep(0.1)
                continue
            yield line
    finally:
        if stream is not None:
            stream.close()

def _short_id(value) -> str:
    value = str(value or "-")
    return value[:8] if value != "-" else value


def _display_text(value, limit=420) -> str:
    text = str(value or "-").replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def _subagent_identity(data: dict) -> str:
    parts = [f"角色={data.get('agent_name', '?')}", f"run={_short_id(data.get('run_id'))}"]
    parent = _short_id(data.get("parent_run_id"))
    if parent != "-":
        parts.append(f"parent={parent}")
    return "  ".join(parts)


def _subagent_line(ts: str, data: dict, message: str, style="bright_cyan"):
    line = Text()
    line.append(f"[ {ts} ] ", style="dim white")
    line.append("子 Agent ", style="bold bright_cyan")
    line.append(_subagent_identity(data) + "  ", style="cyan")
    line.append(message, style=style)
    console.print(line)


def _render_subagent_event(ts: str, data: dict):
    """提交给子 Agent 的事件统一走独立的 Rich 视觉分支，避免与主 Agent 混淆。"""
    event = data.get("event")
    action = str(data.get("action") or "")
    if event == "system_action" and action == "subagent_started":
        body = Text()
        body.append(f"[ {ts} ] ", style="dim white")
        body.append(_subagent_identity(data) + "\n", style="cyan")
        body.append("任务  ", style="bold white")
        body.append(_display_text(data.get("task"), 700) + "\n", style="white")
        body.append("工具  ", style="bold white")
        body.append(", ".join(data.get("tools") or []) or "无（纯分析）", style="white")
        console.print(Panel(
            body,
            title="✦ 子 Agent 已启动 · 主 Agent → 子 Agent",
            title_align="left",
            border_style="bright_blue",
            box=box.ROUNDED,
            padding=(0, 1),
            expand=False,
        ))
        return

    if event == "system_action" and action == "subagent_completed":
        result = _display_text(data.get("result_summary"), 900)
        suffix = f"  · {data.get('duration_ms')}ms" if data.get("duration_ms") is not None else ""
        body = Text(_subagent_identity(data) + "\n", style="cyan")
        body.append(result + "\n", style="bright_white")
        body.append("已返回主 Agent" + suffix, style="dim cyan")
        console.print(Panel(
            body,
            title="✓ 子 Agent 完成",
            title_align="left",
            border_style="green",
            box=box.ROUNDED,
            padding=(0, 1),
            expand=False,
        ))
        return

    if event == "system_action" and action.startswith("subagent_"):
        suffix = f"  · {data.get('duration_ms')}ms" if data.get("duration_ms") is not None else ""
        _subagent_line(ts, data, _display_text(data.get("content") or action) + suffix, "yellow")
        return

    if event == "llm_input":
        return

    if event == "tool_call":
        args = json.dumps(data.get("args", {}), ensure_ascii=False, separators=(",", ":"))
        _subagent_line(ts, data, f"调用工具 {data.get('tool', 'unknown')}  参数={_display_text(args, 260)}", "yellow")
        return

    if event == "tool_result":
        _subagent_line(ts, data, f"工具结果 {data.get('tool', 'unknown')}  {_display_text(data.get('result_summary'), 320)}", "green")
        return

    if event == "ai_message":
        _subagent_line(ts, data, f"阶段输出  {_display_text(data.get('content'), 420)}", "magenta")
        return


def render_event(line: str):
    """解析并渲染监控日志 (100% 中文还原)"""
    try:
        data = json.loads(line.strip())
        event = data.get("event")
        ts_str = data.get("ts", "") 
        try:
            if ts_str.endswith('Z'): ts_str = ts_str[:-1] + '+00:00'
            dt_local = datetime.fromisoformat(ts_str).astimezone()
            ts = dt_local.strftime("%H:%M:%S")
        except (TypeError, ValueError):
            ts = ts_str.split("T")[-1][:8]
            
        prefix = f"[timestamp][ {ts} ][/timestamp] "

        # 子 Agent 事件统一走独立的 Rich 视觉分支，避免与主 Agent 混淆。
        if data.get("run_id") and data.get("agent_name"):
            _render_subagent_event(ts, data)
            return

        if event == "llm_input":
            # 解析 LLM 输入事件，显示上下文记忆条数
            count = data.get("message_count", 0)
            console.print(f"{prefix}[llm_input]🧠 神经元唤醒：发送了 {count} 条上下文记忆...[/llm_input]")
            
        elif event == "tool_call":
            # 解析工具调用事件，显示工具名称和传入参数
            tool_name = data.get("tool", "unknown")
            args_str = json.dumps(data.get("args", {}), ensure_ascii=False, indent=2) 
            content = f"[bold white]● 使用工具: [/bold white][bold color(141)]{tool_name}[/bold color(141)]\n传入参数:\n{args_str}"
            console.print(Panel(content, title=f"✦ 意图决断 [ {ts} ]", title_align="left", border_style="color(141)", width=60))
            
        elif event == "tool_result":
            # 截断过长的结果，避免终端显示过长
            tool_name = data.get("tool", "unknown")
            result = data.get("result_summary", "")
            display_result = result[:300] + "\n...[截断]..." if len(result) > 300 else result
            content = f"[bold white] ● 执行结果: [/bold white][bold cyan]{tool_name}[/bold cyan]\n{display_result}"
            console.print(Panel(content, title=f"✦ 环境回传 [ {ts} ]", title_align="left", border_style="cyan", width=60))
            
        elif event == "system_action":
            action = data.get("action", "")
            console.print(f"{prefix}[warning]✦ 底层状态机：{action}[/warning]")

        elif event == "ai_message":
            content = str(data.get("content", ""))
            console.print(f"{prefix}[ai_message]{content}[/ai_message]")

    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        console.print(f"[error]日志解析失败: {exc}[/error]")

def main(session_ref: str | None = None):
    """Start the monitor for one session or follow the active session."""
    try:
        console.clear()
        print_header()
        provider = (
            (lambda: resolve_log_file(session_ref))
            if session_ref
            else (lambda: resolve_log_file())
        )
        for line in tail_f(provider):
            # 解析并渲染每一行日志，每行包含的信息是表示一个事件的 JSON 对象
            render_event(line)
    except SessionError as exc:
        console.print(f"[error]会话不存在: {exc}[/error]")
    except KeyboardInterrupt:
        console.print("\n[warning]✦ 监控网络已断开。[/warning]")

if __name__ == "__main__":
    main()
