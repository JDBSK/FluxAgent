# 02｜FluxAgent 架构深挖

## 一、整体架构一句话

FluxAgent 是一个用 LangGraph 构建的显式状态机 Agent：任务进入后先初始化，再按需规划，然后 Agent 决策 → 策略审批 → 工具执行 → 观察进度 → 失败恢复/重规划/完成，全过程都有预算熔断、上下文管理和审计日志。

## 二、安全体系：三层防线

面试时讲安全不能只说“有沙箱”，要讲出层次。

### 第 1 层：策略层（执行前拦截）

`policy.py` 按「工具类型 × 参数信号」定级：

- 只读工具默认放行；
- 写操作、变更、执行进入审批；
- 明显越权参数硬拒绝，例如路径穿越、盘符逃逸、危险命令、危险安装来源。

示例判断：

- `read_office_file("workspace/a.md")` → 只读放行。
- `write_office_file(...)` → 写入审批。
- `execute_office_shell("rm -rf ...")` → 硬拒绝。
- `install_mcp_server(...)` → 执行级 + 来源校验。

### 第 2 层：人工审批（Human-in-the-loop）

- 副作用工具在 `approval` 节点等待用户确认；
- LangGraph `interrupt_before=approval` 真正暂停图执行；
- 用户可以批准、拒绝；
- 拒绝后任务进入 cancelled，不会继续乱做。

### 第 3 层：沙箱执行

即使前面放行了，沙箱还要兜底：

- 文件路径用「词法路径 + 解析路径」双重校验，防止 symlink/junction 逃逸；
- 命令不交给 shell 字符串执行，而是解析成 argv，用 `shell=False`；
- 只允许白名单可执行文件；
- 禁止 shell 链式、重定向、变量展开、命令替换；
- 限制读取/写入/输出大小；
- 限制执行时间，超时杀进程树。

## 三、失败恢复：不靠模型悟，靠状态机

### 失败怎么被识别

工具执行后，`progress` 节点检查结果：

- 结构化错误：`ok=false` + error code；
- 文本错误：以错误标记开头；
- 策略拒绝标记；
- 永久错误标记（文件不存在、参数无效等）。

### 动作决策

`classify_failure` 输出三种动作：

| 动作 | 含义 | 什么时候用 |
|---|---|---|
| `retry` | 同一动作再试 | transient，且未超过重试预算 |
| `replan` | 换一套计划 | 参数/策略/资源原因，重试无意义 |
| `abort` | 终止任务 | 预算耗尽、重试+重规划都耗尽、致命错误 |

关键点：

- 失败不等于结束，先分类；
- 放弃不等于失败传播，必须写清楚原因；
- 所有失败路径都要有预算上限；
- 重规划会替换整张计划，而不是追加步骤，避免把旧计划残留下来。

## 三、记忆系统：双水位

FluxAgent 用两层记忆解决“上下文爆炸”和“长期偏好”：

| 层次 | 载体 | 生命周期 | 例子 |
|---|---|---|---|
| 长期记忆 | `user_profile.md` | 跨会话长期 | 用户偏好、身份、习惯 |
| 短期记忆 | state.summary | 随会话 | 最近对话的关键事实、未完成事项 |

### 为什么需要摘要

长对话直接全量塞给模型会：

1. 超上下文窗口；
2. 关键信息被淹没；
3. 成本随 token 线性上涨。

FluxAgent 的裁剪策略：

- 按完整用户回合裁剪；
- `HumanMessage → 后续 AI/Tool 消息` 作为一个回合，保证 tool_call 和 ToolMessage 不会被拆散；
- 达到阈值才压缩；
- 旧的裁剪消息交给模型生成摘要，摘要进入后续提示词。

这样既保留短期上下文，又稳定支撑长期偏好。

## 4. 上下文安全

真实运行中会出现“AIMessage 有 tool_calls 但缺少对应 ToolMessage”的情况，通常由异常中断或 checkpoint 不完整导致。此时把这段历史直接送给模型会报协议错误。FluxAgent 在送模型之前做修复：补一个占位 ToolMessage，明确告诉模型“上一次调用中断，请重新判断，不要盲重试原参数”。

这是入库级细节，却能体现真实生产经验，适合面试重点讲。

## 5. 成本与预算控制

### 为什么用计数而不是 token 数

真实场景里：

- 不同模型 tokenizer 不同；
- 运行前很难精确预测 token；
- 面试/评测需要可确定性。

FluxAgent 用可读计数作为成本代理：

- `max_llm_calls`
- `max_tool_rounds`
- `max_replans`
- `max_retries_per_step`
- `max_wall_seconds`

### 熔断怎么工作

在每次可能花成本的节点前，先“预留”计数，再检查是否超限：

```text
当前次数 + 即将发生的本轮消耗 <= 预算上限
```

如果超限：

- 写 `budget_exceeded_reason`；
- 设置 `task_status = failed`；
- 输出可读原因，例如 `llm_calls_exceeded:41>40`；
- 审计日志记录。

### 为什么要在消耗前检查

如果不预留，就有“刚好到上限还多跑一轮”的问题。先在动作前检查，能保证不超过预算。

## 6. 可观测性

FluxAgent 使用 JSONL 审计日志记录：

- LLM 输入/输出；
- 工具名、参数、结果；
- 审批动作；
- 系统动作；
- 失败分类；
- 预算超限；
- 上下文裁剪。

每次运行还有 `build_run_summary` 生成结构化指标：

```text
task_status, llm_calls, tool_rounds, replan_count,
retry_count, approval_interrupts, hard_deny_count,
context_trim_count, elapsed_seconds, budget_exceeded_reason
```

面试讲这个：**Agent 不是黑盒，而是可以复盘、可以审计、可以定位成本来源的系统。**

## 7. 会话与恢复

- SQLite checkpoint 保存图状态；
- session.id 与 LangGraph thread_id 一一对应；
- 多会话天然隔离；
- 程序重启后可恢复；
- CLI 有 `/sessions`、`/switch`、`/history` 等会话操作。

### 8. 扩展机制：Skill + MCP

- Skill：两段式调用，`help → run`，先看说明书再执行，降低“乱用外部技能”的风险；
- MCP：可通过安装器注册第三方工具，启动时自动发现并包装；
- 因此 FluxAgent 内置能力 + 外部协议能力统一进一张图。
