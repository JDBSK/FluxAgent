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
})

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
    """Resolve an explicit or currently active session to its JSONL log."""
    store = SessionStore()
    active = store.bootstrap()
    record = store.resolve(session_ref, include_archived=True) if session_ref else active
    safe_id = "".join(c for c in record.id if c.isalnum() or c in "-_") or "default"
    return os.path.join(LOG_DIR, f"{safe_id}.jsonl")


def tail_f(path_provider):
    """Follow a log file and reopen it when the active session changes."""
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

        elif event == "graph_transition":
            source = data.get("from_node", "?")
            target = data.get("to_node", "?")
            status = data.get("task_status") or "-"
            step = data.get("step_index", 0)
            console.print(
                f"{prefix}[graph_transition]↪ 图转移：{source} -> {target}"
                f"  [状态={status} 步骤={step}][/graph_transition]"
            )

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
