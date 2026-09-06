# -*- coding: utf-8 -*-
"""内置工具集：时间查询、安全计算、用户画像管理、任务调度、沙盒文件与 Shell。

所有工具用 @fluxagent_tool 装饰器注册，BUILTIN_TOOLS 列表汇总后传入 ToolNode。
任务调度相关工具用 threading.Lock 保护 tasks.json 的并发读写。
"""

from datetime import datetime
from .base import fluxagent_tool, FluxAgentBaseTool
import os
import json
import uuid
import threading
from ..config import MEMORY_DIR, TASKS_FILE
from ..installer import (
    InstallError,
    install_mcp_source,
    install_skill_source,
    list_installed_mcp_servers as load_installed_mcp_servers,
)
from ..safe_calc import safe_calculate
from .sandbox_tools import (
    list_office_files,
    read_office_file,
    write_office_file,
    execute_office_shell
)


tasks_lock = threading.Lock()
PROFILE_PATH = os.path.join(MEMORY_DIR, "user_profile.md")


@fluxagent_tool
def get_system_model_info() -> str:
    """
    获取当前 FluxAgent 正在运行的底层大模型（LLM）型号和提供商信息。
    当用户询问“你是基于什么模型”、“你的底层大模型是什么”、“你是GPT还是GLM”、“现在用的什么模型”等身份问题时，调用此工具。
    """
    provider = os.getenv("DEFAULT_PROVIDER", "unknown")
    model = os.getenv("DEFAULT_MODEL", "unknown")
    
    if provider == "unknown" or model == "unknown":
        return "无法获取当前的系统模型配置，可能是环境变量未正确加载。"
        
    return f"当前使用的模型提供商(Provider)是: {provider}，具体型号(Model)是: {model}。"


@fluxagent_tool
def save_user_profile(new_content: str) -> str:
    """
    更新用户的全局显性记忆档案。
    当你发现用户的偏好发生改变，或者有新的重要事实需要记录时：
    1.请先调用 read_user_profile 获取当前的完整档案。
    2.在你的上下文中，将新信息融入档案，并删去冲突或过时的旧信息。
    3.将修改后的一整篇完整 Markdown 文本作为 new_content 参数传入此工具。
    注意：此操作将完全覆盖旧文件！请确保传入的是完整的最新档案。
    """
    os.makedirs(MEMORY_DIR, exist_ok=True)
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)

    return "记忆档案已成功覆写更新。新的人设画像已生效。"


@fluxagent_tool
def get_current_time() -> str:
    """
    获取当前的系统时间和日期。
    当用户询问“现在几点”、“今天星期几”、“今天几号”等与当前时间相关的问题时，调用此工具。
    """
    now = datetime.now()
    return f"当前本地系统时间是: {now.strftime('%Y-%m-%d %H:%M:%S')}"


@fluxagent_tool
def calculator(expression: str) -> str:
    """Safe arithmetic calculator (AST whitelist, no eval)."""
    return safe_calculate(expression)


@fluxagent_tool
def install_skill(
    source_type: str,
    source: str,
    name: str = "",
    subdirectory: str = "",
) -> str:
    """安装一个扩展 Skill，支持 GitHub Clone 或本地目录/ZIP 导入。

    这是受控安装器，不是任意 Shell。远程来源默认只接受 GitHub HTTPS 地址，
    本地来源可传用户明确提供的目录或 ZIP。若仓库中有多个 SKILL.md，必须传入
    subdirectory 指定真正的技能目录。安装会写入 office/skills，完成后需重启
    FluxAgent 以重新绑定新工具。

    Args:
        source_type: 'git'（GitHub Clone）或 'local'（本地目录/ZIP）。
        source: GitHub HTTPS 仓库地址，或本地绝对路径。
        name: 安装后的技能目录名，可选；只允许字母、数字、下划线和连字符。
        subdirectory: 仓库内包含 SKILL.md 的相对目录，可选。
    """
    try:
        destination = install_skill_source(source_type, source, name, subdirectory)
        return (
            f"Skill 安装成功: {destination.name}\n"
            f"位置: {destination}\n"
            "请重启 FluxAgent 后先用 mode='help' 阅读该 Skill 的说明。"
        )
    except InstallError as exc:
        return f"❌ Skill 安装失败: {exc}"


@fluxagent_tool
def install_mcp_server(
    source_type: str,
    source: str,
    name: str,
    command: str = "",
    launch_args: list[str] | None = None,
    read_only_tools: list[str] | None = None,
    subdirectory: str = "",
) -> str:
    # command改为可选参数，允许自动识别 Node/Python 项目并生成启动命令。
    """自动安装并登记一个外部 stdio MCP Server，支持 GitHub 或本地目录/ZIP。

    安装器会根据项目清单识别 Node/Python，安装隔离依赖、执行必要构建并完成
    MCP 握手；只有验证成功才写入 index.json。第三方安装/构建脚本会在用户批准
    本次 EXEC 操作后运行。重启 FluxAgent 后会发现并绑定该 Server 的工具。

    Args:
        source_type: 'git' 或 'local'。
        source: GitHub HTTPS 仓库地址，或本地绝对路径。
        name: MCP Server 唯一名称。
        command: 可选兜底启动命令；常见 Node/Python 项目无需提供。
        launch_args: 与兜底 command 配套的可选启动参数。
        read_only_tools: 明确声明为只读、可免审批调用的 MCP 工具名列表。
        subdirectory: 仓库内 MCP Server 根目录，可选。
    """
    try:
        record = install_mcp_source(
            source_type=source_type,
            source=source,
            name=name,
            command=command,
            args=launch_args,
            read_only_tools=read_only_tools,
            subdirectory=subdirectory,
        )
        config = {
            "mcpServers": {
                record.name: {
                    "command": record.command,
                    "args": record.args,
                    "cwd": record.installed_path,
                }
            }
        }
        return (
            f"MCP Server 已安装、验证并登记: {record.name}\n"
            f"源码目录: {record.installed_path}\n"
            f"自动识别启动方式: {record.command} {' '.join(record.args)}\n"
            "重启 FluxAgent 即会自动发现其工具；也可以将以下配置加入其他 MCP 客户端：\n"
            + json.dumps(config, ensure_ascii=False, indent=2)
        )
    except InstallError as exc:
        return f"❌ MCP Server 安装失败: {exc}"


@fluxagent_tool
def list_installed_mcp_servers() -> str:
    """查看由 install_mcp_server 安装并登记过的外部 MCP Server。"""
    servers = load_installed_mcp_servers()
    if not servers:
        return "当前没有已登记的外部 MCP Server。"
    lines = ["已登记 MCP Server:"]
    for item in servers:
        lines.append(
            f"- {item.get('name', '?')}: {item.get('command', '')} "
            f"{' '.join(item.get('args') or [])}\n"
            f"  cwd: {item.get('installed_path', '')}"
        )
    return "\n".join(lines)



@fluxagent_tool
def schedule_task(target_time: str, description: str, repeat: str = None, repeat_count: int = None) -> str:
    """
    为一个未来的任务设定闹钟或提醒。
    参数 target_time 必须是严格的格式："YYYY-MM-DD HH:MM:SS"（请先调用 get_current_time 获取当前时间，并在其基础上推算）。
    参数 description 是需要执行的动作或要说的话。
    
    【高级循环功能】：
    - repeat (可选): 设置重复频率。可选值为 "hourly", "daily", "weekly"。如果不重复请留空。
    - repeat_count (可选): 结合 repeat 使用，表示一共需要触发几次。
    
    【案例教学】：
    1. 用户说："以后每天8点提醒我喝牛奶" -> repeat="daily", repeat_count=None (无限循环)
    2. 用户说："接下来的3天，每天提醒我吃药" -> repeat="daily", repeat_count=3 (有限循环)
    3. 用户说："明早8点叫我起床" -> repeat=None, repeat_count=None (单次任务)

    【时间歧义严格确认协议 (AM/PM Ambiguity CRITICAL)】：
    当用户说出的时间存在 12 小时制的模糊性时（例如：只说了“7点”，没明确说早上还是晚上）：
    1. 你必须向用户提问确认是上午还是下午。
    2. 【死命令】：在用户明确回复“上午”或“下午”（或改为24小时制）之前，本工具处于【绝对锁定状态】！
    3. 就算用户发省略号（如“。。”）、发脾气、或者说无关内容，你也【绝对禁止】为了讨好用户而自行猜测时间！
    4. 严禁出现“抱歉多问了”、“默认早上”这种妥协行为。
    5. 如果用户不明确回答，你必须坚定地回复：“抱歉，没有明确上下午，我无权为您设置闹钟。请明确告知时间段。”并立即中止工具调用。
    """
    try:
        target_dt = datetime.strptime(target_time, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return "设定失败：时间格式错误，必须严格遵循 'YYYY-MM-DD HH:MM:SS' 格式。"
    
    now = datetime.now()
    if target_dt <= now:
        return (
            "设定失败：target_time 必须晚于当前时间。"
            f" 当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')}，"
            f" 你传入的是：{target_time}"
        )

    with tasks_lock:
        tasks = []
        if os.path.exists(TASKS_FILE):
            try:
                with open(TASKS_FILE, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        tasks = json.loads(content)
            except Exception as e:
                return f"设定失败：读取任务队列异常 {str(e)}"

        new_task = {
            "id": str(uuid.uuid4())[:8],
            "target_time": target_time,
            "description": description,
            "repeat": repeat,
            "repeat_count": repeat_count
        }
        tasks.append(new_task)

        try:
            with open(TASKS_FILE, "w", encoding="utf-8") as f:
                json.dump(tasks, f, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"设定失败：写入任务队列异常 {str(e)}"

    msg = f" 任务已成功加入队列。首发时间：{target_time} | 任务：{description}"
    if repeat:
        msg += f" | 循环模式：{repeat} (共 {repeat_count if repeat_count else '无限'} 次)"
    return msg


@fluxagent_tool
def list_scheduled_tasks() -> str:
    """
    查看当前所有待处理的定时任务列表。
    当用户询问“我都有哪些任务”、“查一下闹钟”、“刚才定了什么”时调用此工具。
    """
    with tasks_lock:
        if not os.path.exists(TASKS_FILE):
            return "当前没有任何定时任务。"
        
        try:
            with open(TASKS_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return "任务列表为空。"
                tasks = json.loads(content)
            
            if not tasks:
                return "当前没有任何定时任务。"
            
            tasks.sort(key=lambda x: x['target_time'])
            
            res = " 当前待执行任务列表：\n"
            for t in tasks:
                res += f"- [ID: {t['id']}] 时间: {t['target_time']} | 任务: {t['description']}\n"
            return res
        except Exception as e:
            return f"查询失败：{str(e)}"
    

@fluxagent_tool
def delete_scheduled_task(task_id: str) -> str:
    """
    根据任务 ID 取消或删除一个定时任务。
    
    【强制性风险控制协议 (CRITICAL)】：
    删除操作具有不可逆性。
    1. 只要匹配到符合描述的任务数量 > 1。
    2. 无论用户语气多么确定，只要他没提供具体的任务 ID。
    
    【你必须执行的动作】：
    【禁止】在单次回复中针对同一个模糊描述发起多个删除工具调用。
    你必须先列出所有匹配的任务（1. 2. 3.），并询问用户：
    “发现了多个符合条件的提醒（列出列表），为了安全起见，请问是要全部删除，还是只删除其中几个？”
    必须要用户明确给出编号或者说确定全部删除，才能调用此工具！！
    严禁自作主张执行批量删除。
    """

    with tasks_lock:
        if not os.path.exists(TASKS_FILE):
            return "删除失败：任务列表文件不存在。"

        try:
            with open(TASKS_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                tasks = json.loads(content) if content else []
            
            new_tasks = [t for t in tasks if t['id'] != task_id]
            
            if len(new_tasks) == len(tasks):
                return f"删除失败：未找到 ID 为 {task_id} 的任务。"
            
            with open(TASKS_FILE, "w", encoding="utf-8") as f:
                json.dump(new_tasks, f, ensure_ascii=False, indent=2)
            
            return f" 任务 [ID: {task_id}] 已成功取消。"
        except Exception as e:
            return f"操作异常：{str(e)}"
    

@fluxagent_tool
def modify_scheduled_task(task_id: str, new_time: str = None, new_description: str = None) -> str:
    """
    修改现有定时任务的时间或内容。
    
    【强制性风险控制协议 (CRITICAL)】：
    1. 只要用户通过“模糊描述”（如：那个5天的任务、洗澡的任务）来要求修改，而没有直接提供 ID。
    2. 无论用户的话语看起来是单数还是复数（如：“把5天的任务全改了”）。
    3. 只要系统中匹配到的任务数量 > 1。
    
    【你必须执行的动作】：
    禁止直接调用本工具！你必须向用户展示匹配到的所有任务列表，并强制询问：
    “我发现有 [N] 个任务符合描述（列出列表），请问你是要【全部修改】，还是修改其中【某几个】？（请告诉我编号或确认全部）”
    
    必须在用户回复“全部”或者指定了具体编号后，你才能继续操作！修改任务并非小事,这是为了安全！！
    """

    with tasks_lock:
        if not os.path.exists(TASKS_FILE):
            return "修改失败：任务列表为空。"

        try:
            with open(TASKS_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                tasks = json.loads(content) if content else []
            
            found = False
            for t in tasks:
                if t['id'] == task_id:
                    if new_time:
                        parsed_new_time = datetime.strptime(new_time, "%Y-%m-%d %H:%M:%S")
                        now = datetime.now()
                        if parsed_new_time <= now:
                            return (
                                "修改失败：new_time 必须晚于当前时间。"
                                f" 当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')}，"
                                f" 你传入的是：{new_time}"
                            )
                        t['target_time'] = new_time
                    if new_description:
                        t['description'] = new_description
                    found = True
                    break
            
            if not found:
                return f"修改失败：未找到 ID 为 {task_id} 的任务。"
            
            with open(TASKS_FILE, "w", encoding="utf-8") as f:
                json.dump(tasks, f, ensure_ascii=False, indent=2)
                
            return f" 任务 [ID: {task_id}] 已成功更新。"
        except ValueError:
            return "修改失败：时间格式错误。"
        except Exception as e:
            return f"操作异常：{str(e)}"

# 构建的工具
BUILTIN_TOOLS = [
    get_current_time,
    calculator,
    install_skill,
    install_mcp_server,
    list_installed_mcp_servers,
    save_user_profile,
    list_office_files,
    read_office_file,
    write_office_file,
    execute_office_shell,
    get_system_model_info,
    schedule_task,
    list_scheduled_tasks,
    delete_scheduled_task,
    modify_scheduled_task
]
