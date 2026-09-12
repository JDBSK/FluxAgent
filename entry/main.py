# -*- coding: utf-8 -*-
"""主程序入口：装配所有模块并启动异步主循环。

async_main 把各模块拼起来：
  - AsyncSqliteSaver 做 checkpointer（SQLite 持久化 + 多会话 thread_id 隔离）
  - 启 3 个 asyncio task：user_input_loop（收输入）/ agent_worker（跑 astream）/ pacemaker_loop（心跳）
  - app.astream(stream_mode="updates") 流式拿每个节点更新，UI 实时显示工具调用进度
  - 审批闸门：图停在 approval 节点 -> 等 y/n -> 续跑

⚠️ checkpointer 是 AsyncSqliteSaver，全链路必须用 async 接口
（aget_state / aupdate_state / ainvoke），不能混用同步版。
"""

import os
import time
import asyncio
import random
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.styles import Style
from prompt_toolkit.application import get_app

from fluxagent.core.agent import create_agent_app
from fluxagent.core.approval import (
    apending_approval,
    approve_prepare,
    areject_prepare,
    describe_calls,
)
from fluxagent.core.config import DB_PATH
from fluxagent.core.bus import task_queue, stop_workers
from fluxagent.core.heartbeat import pacemaker_loop
from fluxagent.core.context import render_plan
from fluxagent.core.permissions import PermissionManager
from fluxagent.core.policy import build_policy_config
from fluxagent.core.session import SessionStore
from fluxagent.core.skill_loader import load_dynamic_skills
from fluxagent.core.tools.builtins import BUILTIN_TOOLS
from fluxagent.mcp.client import load_installed_mcp_tools
from entry.session_cmd import handle_slash_command, sync_config_to_active

def clear_screen():
    """清空当前终端屏幕。"""
    os.system('cls' if os.name == 'nt' else 'clear')

def type_line(text: str, delay: float = 0.008):
    """按字符延迟输出一行文本。"""
    for ch in text:
        print(ch, end='', flush=True)
        time.sleep(delay)
    print()

def print_banner():
    """清屏并输出启动横幅。"""
    clear_screen()

    CYAN = '\033[38;5;51m'
    PURPLE = '\033[38;5;141m'
    SILVER = '\033[38;5;250m'
    DIM = '\033[2m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    WHITE = '\033[37m'

    logo = f"""{CYAN}{BOLD}
    ███████╗██╗     ██╗   ██╗██╗  ██╗ 
    ██╔════╝██║     ██║   ██║╚██╗██╔╝
    █████╗  ██║     ██║   ██║ ╚███╔╝
    ██╔══╝  ██║     ██║   ██║ ██╔██╗ 
    ██║     ███████╗╚██████╔╝██╔╝ ██╗ 
    ╚═╝     ╚══════╝ ╚═════╝ ╚═╝  ╚═╝      
     █████╗  ██████╗ ███████╗███╗   ██╗████████╗
    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║
    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║
    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║ 
    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝         
{RESET}"""

    sub_title = f"{WHITE}{BOLD} 😺 Welcome to the {PURPLE}{BOLD}FluxAgent{RESET}{WHITE}{BOLD} !  {RESET}"

    quotes = [
        "It works on my machine.",
        "It compiles! Ship it.",
        "Git commit, push, pray.",
        "There's no place like 127.0.0.1.",
        "sudo make me a sandwich.",
        "Works fine in dev.",
        "May the source be with you.",
        "Ctrl+C, Ctrl+V, Deploy.",
        "Hello, World."
    ]
    quote = random.choice(quotes)
    meta = f" {SILVER}✦{RESET} {CYAN}{quote}{RESET}"

    tip = (
        f"{PURPLE} ✦ {RESET}"
        f"{SILVER}{PURPLE}{BOLD}FluxAgent{RESET} 已完成启动。/help 查看会话命令，/sessions 列会话，/exit 退出。{RESET}\n"
    )

    print(logo)
    print(sub_title)
    print() 
    time.sleep(0.12)
    print(meta)
    print() 
    type_line(tip, delay=0.004)


def cprint(text="", end="\n"):
    """输出包含 ANSI 样式的终端文本。"""
    print_formatted_text(ANSI(str(text)), end=end)


def render_terminal_status(values: dict) -> str:
    """为没有最终 AIMessage 的失败/取消终态生成明确的 CLI 收尾。"""
    status = str(values.get("task_status") or "")
    if status == "cancelled":
        return "任务已取消：审批被拒绝，未继续执行。"
    if status != "failed":
        return ""

    budget_reason = str(values.get("budget_exceeded_reason") or "")
    failure_note = str(values.get("last_failure_note") or "")
    if budget_reason.startswith("wall_timeout:"):
        return f"任务未完成：执行时间超过运行预算（{budget_reason}）。"
    if budget_reason:
        return f"任务未完成：运行预算已触发（{budget_reason}）。"
    if failure_note:
        return f"任务未完成：{failure_note}"
    return "任务未完成：状态机以 failed 结束，但没有生成最终回复。"


async def async_main():
    """初始化 Agent 会话并运行异步终端交互。"""
    print_banner()
    
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    load_dotenv(env_path)
    
    current_provider = os.getenv("DEFAULT_PROVIDER", "aliyun")
    current_model = os.getenv("DEFAULT_MODEL", "glm-5")

    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as memory:
        permission_manager = PermissionManager()
        mcp_result = await load_installed_mcp_tools()

        # ---------- 统一能力注册表：builtin + skill + mcp 全部走同一协议 ----------
        from fluxagent.capability.providers import (
            builtin_descriptors,
            skill_descriptors,
        )
        from fluxagent.capability.registry import CapabilityRegistry, RegistryError
        from fluxagent.core.skill_loader import read_skill_manual, scan_skill_infos
        from fluxagent.core.logger import audit_logger

        capability_registry = CapabilityRegistry()

        def _register(descriptor):
            try:
                capability_registry.register(descriptor)
                return True
            except RegistryError as exc:
                # fail-closed：坏描述的能力不进入系统，但不阻断启动
                audit_logger.log_event(
                    thread_id="system_default",
                    event="system_action",
                    action="capability_register_failed",
                    capability=descriptor.capability_id,
                    error=str(exc)[:200],
                )
                return False

        for descriptor in builtin_descriptors(BUILTIN_TOOLS):
            _register(descriptor)

        skill_tools_by_name = {str(tool.name): tool for tool in load_dynamic_skills()}
        for info in scan_skill_infos():
            descriptors = skill_descriptors(
                [info],
                content_loader=read_skill_manual,
                tool_loader=lambda item: skill_tools_by_name.get(str(item.get("name") or "")),
            )
            for descriptor in descriptors:
                _register(descriptor)

        for descriptor in getattr(mcp_result, "descriptors", []):
            _register(descriptor)

        policy = build_policy_config(tool_base_overrides=mcp_result.risk_overrides)
        app = create_agent_app(
            provider_name=current_provider,
            model_name=current_model,
            checkpointer=memory,
            permission_manager=permission_manager,
            capability_registry=capability_registry,
            policy=policy,
        )
        session_store = SessionStore()
        config = {"configurable": {"thread_id": "local_geek_master"}}
        active_id = sync_config_to_active(session_store, config)
        active = session_store.get(active_id)
        title = active.title if active else active_id
        cprint("  \033[38;5;141m* session: %s (%s)\033[0m" % (title, active_id))
        cprint("  \033[38;5;245m/help | /sessions | /new | /switch | /history\033[0m")
        if mcp_result.tools:
            cprint(
                "  \033[38;5;108m* MCP: 已连接 %s 个外部工具\033[0m"
                % len(mcp_result.tools)
            )
        for error in mcp_result.errors:
            cprint("  \033[38;5;208m* MCP 连接失败: %s\033[0m" % error)
        cprint()

        class SpinnerState:
            """保存底部状态栏动画的运行状态。"""

            action_words = [
                "Thinking...",              
                "Working...",               
                "Beep boop...",             
                "Eating bugs...",           
                "Charging battery...",      
                "Brewing coffee...",        
                "Blinking lights...",       
                "Polishing pixels...",      
                "Scanning matrix...",       
                "Warming up circuits...",   
                "Syncing data...",          
                "Pinging server..."         
            ]
            current_words = [] 
            is_spinning = False
            start_time = 0
            frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            is_tool_calling = False 
            tool_msg = ""           

        spinner = SpinnerState()


        def get_bottom_toolbar():
            """根据当前运行状态生成底部工具栏文本。"""
            if not spinner.is_spinning:
                return ANSI("") 
            
            elapsed = time.time() - spinner.start_time
            if spinner.is_tool_calling:
                display_msg = spinner.tool_msg
            else:
                idx_word = int(elapsed) % len(spinner.current_words)
                display_msg = f"😺 {spinner.current_words[idx_word]}"

            idx_frame = int(elapsed * 12) % len(spinner.frames)
            frame = spinner.frames[idx_frame]
            

            return ANSI(f"  \033[38;5;51m{frame}\033[0m \033[38;5;250m{display_msg}\033[0m \033[38;5;141m[{elapsed:.1f}s]\033[0m")

        prompt_message = ANSI("  \033[38;5;51m❯\033[0m ")
        placeholder_text = ANSI("\033[3m\033[38;5;242minput...\033[0m")

        # Main input waits for each queued task to finish, so approval is the
        # only active terminal prompt while the graph is interrupted.
        approval_session = PromptSession()

        async def agent_worker():
            """消费任务队列并驱动 Agent、工具调用及审批流程。"""
            # 这个协程负责从消息总线 task_queue 取出用户输入或心跳触发的任务，调用 app.astream() 执行图，并处理审批闸门
            while True:
                user_input = await task_queue.get()
                if user_input is stop_worker:
                    task_queue.task_done()
                    break
                raw_in = (user_input or "").strip()
                if raw_in.lower() in ["/exit", "/quit"]:
                    task_queue.task_done()
                    break

                # Session slash commands stay outside the agent graph.
                if raw_in.startswith("/"):
                    handled, msg = await handle_slash_command(
                        raw_in, session_store, app, config, permission_manager
                    )
                    if handled:
                        spinner.is_spinning = False
                        if msg == "__EXIT__":
                            task_queue.task_done()
                            break
                        if msg:
                            for line in str(msg).splitlines() or [msg]:
                                cprint("  \033[38;5;250m%s\033[0m" % line)
                            cprint()
                        task_queue.task_done()
                        continue

                spinner.current_words = spinner.action_words.copy()
                random.shuffle(spinner.current_words)
                
                spinner.start_time = time.time()
                spinner.is_spinning = True
                spinner.is_tool_calling = False
                
                inputs = {"messages": [HumanMessage(content=user_input)]}
                try:
                    # 一次用户输入可能被审批闸门打断多次（多个副作用工具），进入LangGraph状态图
                    # 所以外层循环：跑到中断 -> 等审批 -> 续跑 -> 直到没有待批
                    stream_input = inputs
                    while True:
                        pending_agent_text = ""
                        async for event in app.astream(stream_input, config=config, stream_mode="updates"):
                            for node_name, node_data in event.items():
                                if node_name == "planner":
                                    plan = node_data.get("task_plan") or []
                                    if plan:
                                        spinner.is_spinning = False
                                        cprint("  \033[38;5;141m● 任务计划\033[0m")
                                        for line in render_plan(plan, node_data.get("step_index", 0)).split("\n"):
                                            cprint(f"    \033[38;5;250m{line}\033[0m")
                                        cprint('')
                                        spinner.is_spinning = True

                                elif node_name == "progress":
                                    plan = node_data.get("task_plan")
                                    if plan:
                                        idx = node_data.get("step_index", 0)
                                        done = sum(1 for s in plan if s.get("status") == "done")
                                        cprint(f"  \033[38;5;108m● 进度: {done}/{len(plan)} 步\033[0m")
                                        if node_data.get("task_status") == "done" and pending_agent_text:
                                            spinner.is_spinning = False
                                            lines = pending_agent_text.strip().split('\n')
                                            if lines:
                                                formatted_out = f"  \033[38;5;141m❯\033[0m \033[38;5;250m{lines[0]}"
                                                for line in lines[1:]:
                                                    formatted_out += f"\n    {line}"
                                                formatted_out += "\033[0m"
                                                cprint(formatted_out)
                                            pending_agent_text = ""
                                        elif node_data.get("task_status") == "running":
                                            # Only the final planned answer is shown.
                                            pending_agent_text = ""

                                elif node_name == "agent":
                                    messages = node_data.get("messages") or []
                                    if not messages:
                                        continue
                                    last_msg = messages[-1]

                                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                                        for tc in last_msg.tool_calls:
                                            spinner.is_tool_calling = True
                                            spinner.tool_msg = f"唤醒内置工具 : {tc['name']}..."
                                            cprint(f"  ●\033[38;5;51m Tool Call: \033[0m{tc['name']}")
                                            cprint('')

                                    elif last_msg.content:
                                        if node_data.get("task_status") == "running":
                                            pending_agent_text = str(last_msg.content)
                                            continue

                                        spinner.is_spinning = False

                                        lines = last_msg.content.strip().split('\n')
                                        if lines:
                                            formatted_out = f"  \033[38;5;141m❯\033[0m \033[38;5;250m{lines[0]}"
                                            for line in lines[1:]:
                                                formatted_out += f"\n    {line}"
                                            formatted_out += "\033[0m"
                                            cprint(formatted_out)

                                elif node_name not in ("agent", "planner", "progress"):
                                    spinner.is_tool_calling = False

                        # 图停下来了：要么真的结束，要么卡在审批闸门
                        calls = await apending_approval(app, config)
                        if not calls:
                            try:
                                snap = await app.aget_state(config)
                                values = snap.values or {}
                                terminal_status = render_terminal_status(values)
                                if terminal_status:
                                    spinner.is_spinning = False
                                    cprint(
                                        "  \033[38;5;196m● "
                                        f"{terminal_status}\033[0m"
                                    )
                                summary = values.get("last_run_summary") or {}
                                if summary:
                                    cprint(
                                        "  \033[38;5;245m● run metrics: "
                                        f"status={summary.get('task_status')} "
                                        f"llm={summary.get('llm_calls')} "
                                        f"tools={summary.get('tool_rounds')} "
                                        f"replan={summary.get('replan_count')} "
                                        f"retry={summary.get('retry_count')} "
                                        f"approve={summary.get('approval_interrupts')} "
                                        f"deny={summary.get('hard_deny_count')} "
                                        f"trim={summary.get('context_trim_count')} "
                                        f"ctx={summary.get('context_turns_before_trim')}"
                                        f"->{summary.get('context_turns_kept')} "
                                        f"summary_llm={summary.get('summary_llm_calls')} "
                                        f"t={summary.get('elapsed_seconds')}s"
                                        "\033[0m"
                                    )
                            except Exception:
                                pass
                            break

                        spinner.is_spinning = False
                        cprint("  \033[38;5;208m⚠ 需要审批：以下操作有副作用\033[0m")
                        for line in describe_calls(calls).split("\n"):
                            cprint(f"    \033[38;5;208m{line}\033[0m")
                        cprint(
                            "  \033[38;5;250m输入 y 单次 / s 本次记住相同调用 / "
                            "a 永久记住相同调用 / n 拒绝\033[0m"
                        )

                        valid_answers = {
                            "y", "yes", "是", "s", "session", "本次", "记住",
                            "a", "always", "永久", "总是", "n", "no", "否",
                        }
                        while True:
                            answer = (await approval_session.prompt_async(
                                ANSI("  \033[38;5;208m批准? (y/s/a/n)\033[0m ")
                            )).strip().lower()
                            if answer in valid_answers:
                                break
                            cprint("  \033[38;5;208m请输入 y、s、a 或 n。\033[0m")

                        spinner.start_time = time.time()
                        spinner.is_spinning = True
                        if answer in ("s", "session", "本次", "记住"):
                            permission_manager.grant_calls(calls, persistent=False)
                            cprint("  \033[38;5;108m● 已批准，本次运行记住相同工具和参数\033[0m")
                            approve_prepare(app, config, calls)
                        elif answer in ("a", "always", "永久", "总是"):
                            permission_manager.grant_calls(calls, persistent=True)
                            cprint("  \033[38;5;108m● 已批准，永久记住相同工具和参数\033[0m")
                            approve_prepare(app, config, calls)
                        elif answer in ("y", "yes", "是"):
                            cprint("  \033[38;5;108m● 已批准\033[0m")
                            approve_prepare(app, config, calls)
                        elif answer in ("n", "no", "否"):
                            cprint("  \033[38;5;196m● 已拒绝\033[0m")
                            await areject_prepare(app, config, calls)
                        # prepare 只改状态不推进图；下一轮 astream(None) 负责续跑，
                        # 这样审批后的执行过程同样能流式显示出来
                        stream_input = None
                        continue

                except Exception as e:
                    spinner.is_spinning = False
                    cprint(f"  \033[31m[ ⚠️ 引擎异常 : {e} ]\033[0m")

                spinner.is_spinning = False
                try:
                    sid = config.get("configurable", {}).get("thread_id")
                    if sid:
                        session_store.touch(
                            sid,
                            preview=raw_in[:100],
                            message_delta=1,
                            auto_title_from=raw_in,
                        )
                except Exception:
                    pass
                cprint() # 空出舒适的行距
                task_queue.task_done()

        async def user_input_loop():
            """读取用户输入并依次提交到任务队列。"""
            custom_style = Style.from_dict({
                'bottom-toolbar': 'bg:default fg:default noreverse',
            })
            
            session = PromptSession(
                bottom_toolbar=get_bottom_toolbar,
                style=custom_style,
                erase_when_done=True,
                reserve_space_for_menu=0  
            )
            
            async def redraw_timer():
                """定时刷新终端底部状态栏动画。"""
                while True:
                    if spinner.is_spinning:
                        try:
                            get_app().invalidate()
                        except Exception:
                            pass
                    await asyncio.sleep(0.08)
                    
            redraw_task = asyncio.create_task(redraw_timer())
            
            while True:
                try:
                    user_input = await session.prompt_async(prompt_message, placeholder=placeholder_text)

                    user_input = user_input.strip()
                    if not user_input:
                        continue
                    

                    padded_bubble = f"  ❯ {user_input}    "
                    cprint(f"\033[48;2;38;38;38m\033[38;5;255m{padded_bubble}\033[0m\n")
                    
                    if user_input.lower() in ["/exit", "/quit"]:
                        cprint("  \033[38;5;141m✦ 记忆已固化，FluxAgent 进入休眠。\033[0m")
                        break
                    await task_queue.put(user_input)
                    # Keep terminal ownership single-threaded. While the graph
                    # runs, agent_worker may use the terminal for approval.
                    await task_queue.join()
                        
                except (KeyboardInterrupt, EOFError):
                    cprint("\n  \033[38;5;141m✦ 强制中断，FluxAgent 进入休眠。\033[0m")
                    break

            redraw_task.cancel() 

        with patch_stdout():
            # 启动异步任务
            worker = asyncio.create_task(agent_worker())
            heartbeat_worker = asyncio.create_task(pacemaker_loop(task_queue=task_queue, check_interval=10))
            await user_input_loop()
            # 退出排空：先停心跳生产者（取消后可能补投最后一个任务），
            # 再发退出哨兵并 join，保证已入队工作全部处理完，消费者才退出。
            await stop_workers(worker, heartbeat_worker, task_queue)

def main():
    """启动异步终端应用。"""
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
