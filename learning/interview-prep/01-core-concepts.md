# 01｜核心概念深挖

## 一、工具调用：模型只决策，程序才执行

### 1. 完整链路

一次工具调用不是“模型执行了函数”，而是：

```text
用户请求
  ↓
系统把工具列表 + JSON Schema 交给模型
  ↓
模型输出 tool_call：工具名 + 参数
  ↓
运行时校验：工具是否存在、参数是否合法、风险是否允许
  ↓
运行时真正执行函数 / 命令 / MCP 调用
  ↓
把 ToolMessage 写回上下文
  ↓
模型根据结果继续推理或生成最终回答
```

面试关键点：模型输出的是结构化意图，不是代码执行权。

### 2. 为什么要 JSON Schema

Schema 有三个作用：

1. **选择工具**：description 告诉模型什么场景该用它。
2. **生成参数**：properties 告诉模型必须传什么字段、什么类型。
3. **运行前校验**：少字段、错类型、越界参数可以在执行前拦截。

FluxAgent 的工具通过 `@fluxagent_tool` 包装，最终都会进入统一的 ToolNode 执行。最简单的例子：

- `get_current_time`：无参数，只读。
- `calculator(expression)`：一个必填参数，只做安全算术。
- `install_mcp_server(...)`：多个参数，执行级别高风险。

### 3. 失败类型必须分开

| 类型 | 例子 | 正确处理 |
|---|---|---|
| 参数错误 | 缺字段、路径不存在、格式错误 | 修正输入或重新规划，不该重试 |
| 瞬时错误 | 网络抖动、服务暂时不可用、超时 | 有限重试，最好带退避 |
| 策略错误 | 越权路径、危险命令、未批准写操作 | 硬拒绝或人工审批 |
| 资源耗尽 | 预算、上下文、次数用尽 | 熔断并结束 |
| 程序异常 | 代码 bug、未处理异常 | 捕获、审计、避免污染上下文 |

FluxAgent 的 `classify_failure` 会识别 policy、permanent、transient，并结合 retry/replan 预算决定下一步。

### 4. 重试设计的边界

无限重试是事故，不是鲁棒性。合理设计至少包含：

- 最大尝试次数；
- 退避时间；
- 幂等性判断；
- 失败分类；
- 重试用尽后的 replan；
- 仍然失败则 abort。

一句话：**transient 可 retry，permanent 要 replan，policy 不能盲 retry。**

## 二、智能循环：Agent 和普通问答的本质区别

### 1. 普通问答 vs RAG vs Agent

| 类型 | 核心动作 | 局限 |
|---|---|---|
| 普通问答 | 模型直接生成答案 | 没有真实世界动作 |
| RAG | 检索资料后生成答案 | 主要是读，不做复杂执行 |
| Agent | 规划、调用工具、观察结果、修正执行 | 必须解决安全、成本、循环、恢复 |

Agent 的强项是能处理多步、动态、有反馈的任务；它的难点是失控风险也变大了。

### 2. FluxAgent 的核心循环

```text
initialize
  ↓
planner
  ↓
agent：决定下一步或回答
  ↓
需要工具？
  ├── 否：结束
  └── 是：policy 评估
        ├── 高危：policy_block
        └── 可审批：approval → tools
tools 执行
  ↓
progress：更新计划、观察结果
  ↓
route：
  ├── 完成 → END
  ├── 失败可重试 → agent
  ├── 需要换计划 → replan → agent
  └── 失败不可恢复 → END
```

### 3. 为什么要显式状态

只靠 messages 让模型“自己悟”是不够的，因为有些状态必须跨轮保存：

- 当前执行到第几步；
- 当前步重试了几次；
- 已经重规划几次；
- LLM 调用多少次；
- 工具轮次多少；
- 是否等待审批；
- 上下文裁剪了多少次。

FluxAgent 的 `AgentState` 明确保存这些字段，避免从对话历史里低质量反推状态。

### 4. Planner 的价值

复杂任务先拆成步骤，优点是：

1. 执行有顺序，不靠模型每轮临时发挥；
2. 失败时知道是哪一步失败；
3. 可以对失败步骤做 replan；
4. 用户能看到任务进度。

但也不是所有任务都要 Planner。简单问题直接回答，复杂任务才规划，避免无意义消耗。

## 三、MCP：Agent 的通用工具接口

### 1. MCP 解决什么问题

没有 MCP 时，每个 Agent 框架接工具的方式都不一样：

```text
Agent A → 自己定义工具协议
Agent B → 另一套函数调用协议
Agent C → 又一套插件协议
```

MCP 的意义是统一：

```text
Agent / MCP Client
  ↕ 标准协议
MCP Server / 工具提供方
```

类比 REST：REST 让 Web 系统之间用统一 HTTP 接口通信；MCP 让 AI 客户端和工具服务方用统一协议发现和调用能力。

### 2. 关键流程

1. 启动或连接 MCP Server。
2. `initialize` 完成握手。
3. `list_tools` 拿到工具名、description、input schema。
4. Agent 把 MCP 工具包装成本地可调用工具。
5. 模型输出 tool_call。
6. 客户端调用 MCP 的 `call_tool`。
7. 服务端返回文本或结构化内容。
8. 客户端把结果转成 ToolMessage 给模型。

FluxAgent 的 `mcp/client.py` 就是这个模式：发现远端工具、生成安全暴露名、包装成 LangChain Tool，并根据 `read_only_tools` 标记风险级别。

### 3. MCP 不等于安全

MCP 只是协议，不天然保证安全。接第三方 MCP 时要处理：

- 工具描述投毒；
- 参数注入；
- 越权路径；
- 命令执行；
- 网络访问；
- 数据泄露；
- 供应链风险。

FluxAgent 的做法：

1. 安装器只接受受控来源；
2. MCP 工具默认按写风险处理；
3. 明确声明只读的工具才降低风险；
4. 执行前进入 Policy；
5. 审计日志记录工具调用。

### 4. MCP 和普通函数工具的区别

| 维度 | 本地函数工具 | MCP 工具 |
|---|---|---|
| 部署位置 | 同进程或本机 | 独立进程 / 远端服务 |
| 发现方式 | 代码注册 | initialize + list_tools |
| 调用方式 | 直接函数调用 | 标准 MCP call_tool |
| 风险来源 | 本地函数逻辑 | 外部服务 + 协议 + 供应链 |
| 扩展性 | 修改代码扩展 | 安装/配置服务扩展 |

## 四、LangGraph：为什么 Agent 需要状态图

### 1. 普通 LLM Chain 的问题

线性 Chain 只适合固定流程。Agent 的真实流程是动态的：

- 可能要规划；
- 可能直接回答；
- 可能等待审批；
- 可能重试；
- 可能重规划；
- 可能预算耗尽；
- 可能中途恢复。

LangGraph 把这些变成显式图结构：节点、边、条件边、状态、checkpointer、interrupt。

### 2. FluxAgent 图中的关键节点

| 节点 | 作用 |
|---|---|
| `initialize` | 初始化运行状态和计数 |
| `planner` | 复杂任务拆步骤 |
| `agent` | 模型决策下一步 |
| `approval` | 人工审批闸门 |
| `policy_block` | 高危调用硬拒绝 |
| `tools` | 真正执行工具 |
| `progress` | 观察结果、推进计划、分类失败 |
| `replan` | 重新生成计划 |

### 3. Checkpointer 的价值

Checkpointer 把状态持久化：

1. 进程重启后能恢复；
2. 多会话用 thread_id 隔离；
3. 审批等待后能继续执行；
4. 长任务可以断点续跑。

FluxAgent 支持内存 checkpoint 和 SQLite checkpoint。

### 4. Interrupt 的价值

`interrupt_before` 让图在工具执行前停下来，等用户批准。这个不是提示词层面的“请小心”，而是运行时级别的硬控制。
