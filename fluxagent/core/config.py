# -*- coding: utf-8 -*-
"""路径与目录配置。

定义 workspace 下各子目录的绝对路径（记忆/沙盒/技能/任务/会话），
并自动创建缺失目录。启动信息走 stderr，因为 MCP stdio 模式把 stdout
当协议通道，任何多余 print 都会让客户端握手失败。
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()


def _positive_int_env(name: str, default: int) -> int:
    """Read a positive integer setting and fail fast on invalid config."""
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(CORE_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)

WORKSPACE_DIR = os.getenv("FLUXAGENT_WORKSPACE", os.path.join(PROJECT_ROOT, "workspace"))


DB_PATH = os.path.join(WORKSPACE_DIR, "state.sqlite3")     # 状态机：潜意识与短期记忆
MEMORY_DIR = os.path.join(WORKSPACE_DIR, "memory")         # 显性记忆：Markdown 画像
PERSONAS_DIR = os.path.join(WORKSPACE_DIR, "personas")     # 人设区：系统 Prompt
SCRIPTS_DIR = os.path.join(WORKSPACE_DIR, "scripts")       # 脚本区：自动化武器库
OFFICE_DIR = os.path.join(WORKSPACE_DIR, "office")         # 沙盒工位 唯一被允许执行文件与shell操作的空间
SKILLS_DIR = os.path.join(OFFICE_DIR, "skills")            # 技能卡槽
MCP_SERVERS_DIR = os.path.join(WORKSPACE_DIR, "mcp_servers")
MCP_SERVERS_INDEX = os.path.join(MCP_SERVERS_DIR, "index.json")
TASKS_FILE = os.path.join(WORKSPACE_DIR, "tasks.json")
SESSIONS_DIR = os.path.join(WORKSPACE_DIR, "sessions")     # 多会话元数据与会话笔记
SESSIONS_INDEX = os.path.join(SESSIONS_DIR, "index.json")  # 会话索引（与 checkpoint 分离）

# 高低水位设计，前10轮压缩成摘要，后10轮保留+用户画像+输入
# 避免每轮都要重新摘要、降低频率
# 上下文压缩使用高/低水印。达到20圈会触发压缩，直到最后10圈。这两个值都是可配置的.
CONTEXT_TRIGGER_TURNS = _positive_int_env("CONTEXT_TRIGGER_TURNS", 20)  # 触发压缩轮数，保留更多原始近期上下文
CONTEXT_KEEP_TURNS = _positive_int_env("CONTEXT_KEEP_TURNS", 10)  # 压缩后保留轮数
if CONTEXT_KEEP_TURNS >= CONTEXT_TRIGGER_TURNS:
    raise ValueError("CONTEXT_KEEP_TURNS must be smaller than CONTEXT_TRIGGER_TURNS")

for d in [
    WORKSPACE_DIR,
    MEMORY_DIR,
    PERSONAS_DIR,
    SCRIPTS_DIR,
    OFFICE_DIR,
    SKILLS_DIR,
    MCP_SERVERS_DIR,
    SESSIONS_DIR,
]:
    os.makedirs(d, exist_ok=True)

# 启动信息走 stderr，不能走 stdout。
# MCP stdio 传输把 stdout 当 JSON-RPC 通道，任何多余的 print 都会让客户端
# 解析失败（实测：这一行 print 直接把 MCP 握手打挂）。
# 同时 Windows 控制台默认 GBK，emoji 会抛 UnicodeEncodeError，所以只用 ASCII。
try:
    print(f"[Config] Workspace ready: {WORKSPACE_DIR}", file=sys.stderr)
except UnicodeEncodeError:
    pass
