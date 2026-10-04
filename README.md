<h1 align="center">FluxAgent</h1>

<p align="center">
  <strong>一款面向真实工具执行的 Agent 应用，支持任务规划、失败恢复、安全确认与会话持久化。</strong>
</p>
<p align="center">
  <strong>An open-source Agent application for planning, tool execution, recovery, and controllable automation.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version">
  <img src="https://img.shields.io/badge/Engine-LangGraph-FF6F00?style=for-the-badge" alt="LangGraph">
  <img src="https://img.shields.io/badge/Tooling-LangChain-D85CFF?style=for-the-badge" alt="LangChain">
  <img src="https://img.shields.io/badge/Storage-SQLite-817A91?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite">
  <img src="https://img.shields.io/badge/Extension-MCP-FF9A62?style=for-the-badge" alt="MCP">
  <img src="https://img.shields.io/badge/License-MIT-49C9A5?style=for-the-badge" alt="License">
</p>
<p align="center">
  <a href="#概述">概述</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#系统架构">系统架构</a> ·
  <a href="#核心设计">核心设计</a> 
</p>


---

## 📌 概述

🤐普通 ReAct Agent 很容易在一次工具调用中完成 Demo，却很难稳定处理真实任务：复杂请求没有清晰步骤，工具失败后只能盲目重试，写文件或执行命令前用户看不到风险，进程重启后也难以恢复上下文。

🔗FluxAgent 将这些行为放进一张**显式的 LangGraph 状态图**中。它可以根据请求复杂度选择直接回答或先生成计划；工具调用会经过 Policy 检查，并在出现任务失败时，会被分类并选择重试、重新规划或结束；运行状态和对话历史通过 SQLite checkpoint 保存。

🔓FluxAgent 的目标很直接：让 Agent 不只是“能调用工具”，还能够把一次任务完整地跑完，并让用户知道它正在做什么、为什么暂停以及失败后做了什么调整的一个**透明可控智能体**。

### 🌟 核心能力

- 🔄 **任务规划与执行**：简单请求直接回答，复杂请求进入 Planner → Agent → Tool → Progress流程。
- ♻️ **失败处理**：区分瞬时错误、文件或参数错误、策略错误，分别处理 retry、replan 和终止。
- 🛡️ **执行前确认**：写操作工具进入 Human-in-the-loop（HITL），高风险参数由 Policy 直接拦截。
- **🧠 选择性记忆**：工作记忆、短期上下文和长期记忆分层；真实事件先落 SQLite，再检索相关内容，避免把全部画像塞给模型。
- 💾 **状态可恢复**：使用 AsyncSqliteSaver保存图状态，多会话通过thread_id进行**会话隔离**。
- 🧩 **Skill 与 MCP 扩展**：支持 Github开源工具SKILL和 MCP 安装与懒加载，两段式调用（help → run），先看说明书再执行。
- 💻 **运行时监控**：JSONL 审计日志记录 LLM 输入、工具调用、工具结果、AI 回复和系统动作。

## 🚀 快速开始

### **1️⃣**安装

```bash
git clone https://github.com/JDBSK/FluxAgent
cd FluxAgent

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e .
```

当前代码入口命令为 `fluxagent`，包名、CLI、MCP 命令和环境变量已统一使用 FluxAgent 品牌。

### **2️⃣**配置

有两种方式：使用交互式配置向导，或手动编辑 `.env`。

#### 方式一：配置向导

```bash
# 配置向导会引导进行Provider配置、API Key和Base URL等信息填写，并测试连接。
fluxagent config
```

#### 方式二：手动配置

```bash
cp .env.example .env
vim .env # 编辑 .env配置
```

编辑 `.env`，至少配置 Provider、模型和对应 API Key：

```bash
DEFAULT_PROVIDER=other  	# 模型提供商 (openai, anthropic, aliyun...)
DEFAULT_MODEL=your-model	# 模型名称 (如 gpt-5.6-sol, glm-5.2...)
OPENAI_API_KEY=your-api-key # OpenAI 或兼容接口的 API Key
OPENAI_API_BASE=https://your-compatible-endpoint/v1  # 兼容接口的 Base URL
```

### **3️⃣**运行

```bash
# 启动主程序
fluxagent run
```

<p align="center">
  <img src="docs/welcome.png" alt="FluxAgent Banner" width="70%">
</p>

### **4️⃣**基本用法

<p align="center">
  <img src="docs/chat.png" alt="FluxAgent Banner" width="70%">
</p>

#### 🧭 常用命令示例

启动后，可以直接用自然语言让 Agent 调用工具：

| 类型      | 示例                                    | 作用             |
| --------- | --------------------------------------- | ---------------- |
| 时间查询  | `现在是北京时间几点？`                  | 调用时间工具     |
| 数学计算  | `帮我算一下 (128+64)/4`                 | 调用安全计算器   |
| 文件查找  | `列出 office/demo 目录下的文件`         | 查看工作区内容   |
| 读取读取  | `读取 demo/project_brief.md`            | 读取项目说明     |
| 文件写入  | `把分析结果写入 demo/release_report.md` | 触发写入审批     |
| Shell命令 | `运行校验脚本检查报告格式`              | 进入受控命令执行 |
| 定时任务  | `每周五下午五点提醒我整理周报`          | 创建循环任务     |
| 用户画像  | `记住我更喜欢简短回复`                  | 更新长期画像     |

CLI 会话控制命令：

| 指令                  | 作用           |
| --------------------- | -------------- |
| /help                 | 查看命令帮助   |
| /new                  | 新建并切换会话 |
| /switch <id \| 标题>  | 切换会话       |
| /sessions             | 列出会话       |
| /history 10           | 查看最近历史   |
| /notes                | 追加会话笔记   |
| /rename <id \| 标题>  | 重命名当前会话 |
| /pin                  | 置顶当前会话   |
| /archive <id \| 标题> | 归档会话       |
| /exit                 | 退出程序       |

#### **⏰ 心跳任务系统**

**FluxAgent 内置心跳任务系统，用于让 Agent 在设定时间主动执行任务：**

- **自动触发：心跳进程每秒检查任务队列，到点自动触发任务**
- **定时循环：支持 hourly、daily、weekly、monthly 循环任务**
- **任务持久化：任务写入 JSON 文件，程序重启后仍可读取。**

**示例：**

```bash
> 每周一上午 10 点提醒我检查本周发布计划
✅ 任务已加入队列 | 循环模式：weekly
```

## **🏗️ 系统架构**

FluxAgent 把控制流组织为循环有向图。简单请求可以跳过 Planner；复杂任务先拆成步骤，再根据工具结果继续执行或重新规划。

<p align="center">
  <img src="docs/architecture.png" alt="FluxAgent 架构图" width="78%">
</p>

**架构说明：**

- **输入与调度层（浅蓝）**：接收用户输入和 Heartbeat心跳任务，由 agent_worker统一进入任务队列，负责驱动 LangGraph、流式输出和人工审批交互。
- **初始化与规划层（浅紫）**：initialize初始化模块本次任务的运行状态；**简单请求**直接进入 Agent，**复杂请求**先由 planner拆分为多个可执行步骤。
- **智能决策层（黄色）**：agent 调用 LLM，结合当前任务、上下文和计划，决定下一步是直接回答、调用普通工具、请求人工审批，还是触发安全拦截。
- **工具执行与安全层（橙色）**：  
  - `approval`：对文件写入、Shell、外部 MCP 等副作用操作进行人工确认；  
  - `tools`：执行内置工具、Skill 和 MCP 工具；  
  - `policy_block`：拦截高危参数、越权路径和违反安全策略的调用。
- **进度与恢复层（绿色）**：progress模块检查工具结果并更新任务状态、步骤游标、重试次数和预算信息。任务可以继续执行、有限重试，或进入 replan 生成新的执行方案。
- **任务终止层（灰色）**：当任务完成、失败、取消或超出预算时进入END，结束本次任务执行。

**运行逻辑：**

```txt
输入任务
  → 初始化 / 规划
  → Agent 决策
  → 审批 / 安全检查 / 工具执行
  → Progress 更新任务状态
  → 继续执行、重试、重规划或结束
```

### **①状态图节点**

| **节点** | **作用** |
|---|---|
| initializ | 初始化本轮任务状态、计数器和运行预算 |
| planne | 将复杂请求拆分为可执行步骤；简单请求跳过 |
| agent | 读取上下文和计划，决定直接回答或发起 Tool Call |
| approva | 在写操作工具真正执行前暂停，等待用户确认 |
| policy_block | 对 critical 或越权参数生成阻断结果 |
| tool | 使用 LangChain `ToolNode` 执行工具 |
| progress | 推进计划、分类失败、处理 retry/replan 和预算 |
| replan | 根据失败原因生成替代计划并重新进入 Agent |

### ②显式节点状态

**`AgentState` 除了消息历史，还保存任务执行过程中无法从历史可靠推导的状态：**

- 任务计划`task_plan`、运行步骤索引`step_index`、任务状态`task_status`
- 重试次数`retry_count`、重规划次数`replan_count`、失败分类和失败说明
- LLM 调用次数、工具轮次、审批中断次数
- 上下文裁剪前后回合数和字符数、摘要调用次数
- 运行预算`RunBudget` 的 LLM、工具轮次、重规划和墙钟时间限制

`progress` 是任务推进的集中处理点：成功时推进步骤，瞬时错误有限重试，文件或参数错误重新规划，策略错误不盲目重复，预算超过限制时结束任务。

## **🔬 核心设计**

### **1. 任务规划、失败恢复与预算**

FluxAgent 不把任务进度只放在 Prompt 里，而是将计划和游标写进状态。每个计划步骤包含动作、状态和备注，`step_index` 指向当前步骤。

工具返回错误后，`recovery` 模块会先判断错误类型：

- Transient：允许在当前步骤有限重试；
- Permanent / input error：记录原因并生成新计划；
- Policy error：不重复执行相同危险动作，优先换方案或结束；
- Budget exceeded：停止继续调用，输出运行摘要。

重规划采用整表替换，不把旧计划追加到新计划后面，避免失败动作残留和重复执行。

### **2. 工具安全与 HITL**（human-in-the-loop）

**工具调用路径为：**

```text
LLM Tool Call
    ↓
Policy：READ / WRITE / MUTATE / EXEC / CRITICAL
    ├── READ：直接执行
    ├── WRITE / MUTATE / EXEC：进入 approval，等待一次性 y / n
    └── CRITICAL：直接阻断，不进入审批
```

审批由 SQLite `PermissionBroker` 管理。每个副作用调用生成一次性 `action_id`，状态按
`pending → approved/rejected → executing → succeeded/failed/unknown` 推进。SQLite 条件更新保证同一动作只能进入一次 `executing`；参数变化会生成新动作并重新审批。所有审批权限统一由 `PermissionBroker` 管理。

用户拒绝时，系统会补齐与原调用相同 `tool_call_id` 的 `ToolMessage`，保持模型消息协议完整，避免下一轮请求因为缺少工具结果而失败。

### **3. 文件与命令执行**

文件和命令操作集中在工作空间 `workspace/office` 内。Shell 工具使用命令白名单、参数数组、`shell=False`、`realpath/commonpath` 路径检查，并设置超时、输出上限和进程树终止，提供应用层的执行约束。

以下输入会被拒绝：URL、绝对路径、UNC 路径、`..` 路径穿越、重定向、命令链、环境变量展开和解释器内联代码。

### **4. 上下文与状态恢复**

- `AsyncSqliteSaver` 全链路使用 `astream`、`ainvoke`、`aget_state`、`aupdate_state`。
- checkpoint 随图节点和状态更新保存，进程重启后可以读取原有控制器运行状态。
- 上下文按完整用户回合裁剪，避免拆散 `AIMessage.tool_calls` 与 `ToolMessage`。
- 默认高/低水位为 20/10：达到 20 个用户回合后压缩旧内容，保留最近 10 个完整回合。
- 旧回合摘要保存于 checkpoint，全局用户画像保存为 Markdown，会话笔记按会话隔离。

### **5. 可恢复主从 Agent**

- 主 Agent 可通过 `delegate_task` 将独立复杂任务交给子 Agent；同一轮只批量委派互不依赖的平级任务，默认同时运行 2 个，其余进入最多 8 个任务的有界等待队列。
- 主图在当前用户任务内保存已完成子任务账本；重规划再次发出完全相同的委派参数时直接复用结果，只执行未完成项。
- 子 Agent 使用独立 `thread_id/run_id` 和短期 SQLite 检查点，不继承用户画像、完整主对话或主图摘要。
- 只读工具直接执行；副作用调用暂停子图并在主会话显示审批。批准后恢复原子图、原工具和原参数，不由主 Agent 重新调用。
- 子 Agent 不能递归委派，工具权限只能继承或收窄。拒绝、超时、步数耗尽和工具失败不会被标成 `completed`。
- 文件覆盖使用路径锁、临时文件、`fsync` 和 `os.replace`；审批期间目标发生变化时拒绝覆盖。

真实 `Deepseek-v4-flash` 安全/机制 A/B 共 240 次：单 Agent 正确率 100%、P50 5.704s；层级 Agent 正确率 97.5%、P50 6.781s。两组未授权副作用、重复副作用和失败误报完成均为 0。

新的自主委派 A/B 使用上线评审、事故定位、接口变更审查和简单值班查询，共 8 个正常用户任务、16 次运行；提示词不要求委派，成功只看答案和证据。两组正确率和证据获取率均为 100%，6 个复杂任务中自主委派 1 次，2 个简单任务误委派 0 次。唯一实际委派样本耗时为单 Agent 的 1.11 倍，但输入 Token 降至 47.07%。因此当前实现默认由主 Agent 直接完成；只有多条独立调查线需要多步判断、专业分工或上下文隔离时才考虑委派。完整结果见 `artifacts/subagent_autonomy_benchmark_final.json`。

### **6. Skill 两阶段懒加载**

Skill 启动时只扫描 `SKILL.md` 前部的 `name` 和 `description`，正文在 `help` 时按需读取，并通过 LRU + mtime 缓存避免重复 I/O。

1. `mode="help"`：读取技能说明，帮助 Agent 理解参数和使用方式；
2. `mode="run"`：执行技能命令，复用同一套 Policy 和安全执行环境；
3. “先 help 后 run”目前主要由提示词引导，给LLM调用工具的后悔机会，并非状态机硬强制，可灵活调用。

### **7. MCP 扩展**

**外部 MCP 的实际使用流程**

外部 MCP 不是“复制进 Skill 目录就能用”，也不是 FluxAgent 未经授权主动搜索网络。通过 `install_mcp_server` 安装外部服务后，FluxAgent 会依次完成：

```text
下载或导入源码
    ↓
识别 Node / Python 运行时
    ↓
准备依赖并执行构建
    ↓
initialize + list_tools 握手
    ↓
健康检查通过后登记
    ↓
启动时发现并绑定工具 mcp_<server>_<tool>
```

​	已登记的 MCP Server 保存在`workspace/mcp_servers/index.json`,外部工具会以`mcp_<server>_<tool>`形式绑定到 Agent。例如：

```
mcp_howtocook_get_menu
mcp_stock_get_quotes
```

## **🔌 技能生态兼容**

**FluxAgent 可以读取符合 `SKILL.md` 约定的 OpenClaw 和 Claude Code 风格技能，将其转换为 LangChain Tool 使用。**

**兼容重点包括：**

- **技能目录结构；**
- **`SKILL.md` 元数据和说明格式；**
- **`help/run` 调用参数；**
- **技能命令在 office 工作区内的受控执行。**

**第三方技能是否能直接运行，还取决于其脚本、依赖和命令是否符合当前安全规则。**

## **🔧 内置工具**

FluxAgent 当前内置 15 个工具，覆盖时间、计算、画像、文件、Shell、任务和扩展登记：

| **工具** | **功能** | **示例** |
|---|---|---|
| `get_current_time` | 获取当前时间 | “现在是几点？” |
| `calculator` | 安全数学计算 | “计算 (128 + 64) / 4” |
| `get_system_model_info` | 查看 Provider 和模型 | “当前使用什么模型？” |
| `propose_memory_change` | 提交单条长期记忆变更提案，需审批 | “记住我偏好简洁回答” |
| `list_office_files` | 列出 office 文件 | “查看 demo 目录” |
| `read_office_file` | 读取 office 文件 | “读取 project_brief.md” |
| `write_office_file` | 写入 office 文件 | “生成 release_report.md” |
| `execute_office_shell` | 执行白名单 Shell | “运行校验脚本” |
| `schedule_task` | 创建一次性/循环任务 | “每周一提醒我整理周报” |
| `list_scheduled_tasks` | 查看任务列表 | “列出我的提醒” |
| `modify_scheduled_task` | 修改任务 | “把周报提醒改到下午三点” |
| `delete_scheduled_task` | 删除任务 | “取消这条提醒” |
| `install_skill` | 安装并登记 Skill | “安装这个技能包” |
| `install_mcp_server` | 安装、验证并登记 MCP Server | “安装这个 GitHub MCP 服务” |
| `list_installed_mcp_servers` | 查看已登记 MCP | “列出已安装的 MCP” |

## **💻 实时动态监控**

```bash
# 启动交互式会话
fluxagent run

# 查看当前会话列表
fluxagent sessions list

# 在另一个终端跟随活动会话
fluxagent monitor
```

<p align="center">
  <img src="docs/monitor.png" alt="FluxAgent monitor" width="50%">
</p>
实时查看：

- ✍LLM输入内容
- 🧮工具调用状况
- 🛜LangGraph状态转移及运行状态
- 🤖AI回复内容
- ✅人工审批状况

**查看审计日志**

```bash
# 实时查看某个会话的日志文件
tail -f logs/<thread_id>.jsonl

# 搜索最近的工具调用
grep "tool_call" logs/<thread_id>.jsonl | tail -20
```

## **👤 自定义用户画像**

**FluxAgent 使用 Markdown 保存全局用户画像。可以通过工具更新，也可以直接编辑：**

```markdown
# 用户档案 workspace/memory/user_profile.md：

- **姓名**: Tom
- **个人偏好**:
  - 喜欢美食，不喜欢高糖视频
  - 每天7点起床
- **特殊要求**:
  - 回复简洁
```

**画像用于后续对话的系统上下文；单个会话的临时记录可以使用 `/notes` 保存。**

## **🧠 记忆系统**

FluxAgent 将记忆和运行状态分层保存：

| **数据** | **存储位置** | **用途** |
|---|---|---|
| **任务状态与对话摘要** | `SQLite checkpoint` | **断点恢复、任务计划、运行计数、历史摘要** |
| **长期记忆查询快照** | `workspace/memory/structured_memory.json` | **由记忆事件投影生成，跨会话提供稳定偏好和事实** |
| **记忆事件与投影任务** | `workspace/memory/memory_events.sqlite3` | **追加、重试、幂等投影和重放** |
| **会话信息** | `workspace/sessions/` | **会话列表、标题、笔记和元数据** |

长期记忆进入模型前经过会话/能力边界过滤、BM25 检索和 Token 限制；配置 embedding 且关键词命中不足时增加向量检索，失败自动回退 BM25。

旧 `user_profile.md` 只作为一次性迁移输入并保留备份，不再是正常运行时的长期记忆主存储。

当前没有让 LLM 直接整理或修改长期记忆：`MemoryWorker` 只处理已验证事件的程序投影；存储和画像解析在 `fluxagent/capability/memory_store.py`，Prompt 拼装在 `context_compiler.py`。

### **✂️上下文裁剪**

一个完整回合包含用户输入、LLM回答以及后续的工具调用信息，不会把工具调用和工具结果拆开。

本项目按完整用户回合对对话进行分组，当用户回合达到阈值时：

1. 始终保留系统提示；
2. 保留最近的完整回合；
3. 将较早回合压缩为摘要，并保存到 SQLite checkpoint；
4. 记录裁剪回合数、字符数和摘要调用次数。

## **📂 仓库结构**

```text
FluxAgent/
├── fluxagent/
│   ├── core/
│   │   ├── agent.py              # LangGraph 状态图与路由
│   │   ├── approval.py           # HITL 审批与 ToolMessage 配对
│   │   ├── budget.py             # RunBudget 与运行摘要
│   │   ├── context.py            # AgentState 与上下文裁剪
│   │   ├── policy.py             # 风险分级与策略判断
│   │   ├── recovery.py           # 失败分类、retry、replan
│   │   ├── session.py            # 多会话索引与会话笔记
│   │   ├── skill_loader.py       # Skill 元数据扫描与懒加载
│   │   └── tools/                # 内置工具与安全执行
│   └── mcp/                      # 对外 MCP Server、ToolRegistry 与外部 MCP Client
├── entry/
│   ├── cli.py                    # Typer 配置和命令入口
│   ├── main.py                   # AsyncIO worker、checkpoint、heartbeat
│   ├── monitor.py                # Rich 审计监控
│   └── session_cmd.py            # 会话控制命令
├── tests/
│   ├── eval/                    # 确定性、回归和真实模型评测
│   └── test_*.py                # 单元、集成和安全回归测试
├── docs/                         # 架构图、使用说明与评测文档
├── .env.example                  # 环境配置模板
├── setup.py
└── README.md
```

## **📜 许可证**

**本项目使用 MIT License，详见 [LICENSE](LICENSE)。**

## **🙏 致谢**

**感谢 LangChain、LangGraph、MCP 和开源 Agent 社区提供的工具与实践。**
