# FluxAgent 面试准备总入口

这个阶段的目标不是“看过”，而是能面试现场把项目讲清楚、把原理讲准、把风险讲透。

## 学习顺序

1. `01-core-concepts.md`：把工具调用、智能循环、MCP、LangGraph 打穿。
2. `02-architecture-deep-dive.md`：把 FluxAgent 的架构、安全、恢复、记忆、成本串起来。
3. `03-qa-bank.md`：25 个高频面试问答，每题都结合项目证据。
4. `04-interview-script.md`：1 分钟项目介绍、3 分钟架构讲解、追问应对。

## 回答公式

面试回答建议固定成 4 段：

1. **结论先行**：先给出判断，不要绕概念。
2. **机制展开**：说清楚数据流、状态流、控制流。
3. **项目证据**：明确说 FluxAgent 里怎么做的。
4. **权衡取舍**：说清楚为什么不用更复杂方案。

示例：

> Agent 的关键不是多调几个工具，而是有可控的状态机。FluxAgent 用 LangGraph 把任务初始化、规划、执行、审批、恢复、预算熔断都显式建模；每次工具失败会先分类，再决定 retry、replan 或 abort，而不是让模型无限制试错。这样能解决真实任务里的死循环、越权和成本失控问题。

## 证据文件地图

| 面试主题 | 关键文件 |
|---|---|
| Agent 状态图 | `fluxagent/core/agent.py` |
| 状态定义与上下文裁剪 | `fluxagent/core/context.py` |
| 失败恢复 | `fluxagent/core/recovery.py` |
| 预算控制 | `fluxagent/core/budget.py` |
| 风险审批 | `fluxagent/core/policy.py` |
| 沙箱执行 | `fluxagent/core/tools/sandbox_tools.py` |
| 内置工具 | `fluxagent/core/tools/builtins.py` |
| MCP 客户端 | `fluxagent/mcp/client.py` |
| MCP 服务端注册 | `fluxagent/mcp/registry.py` |
| Skill 两段式调用 | `fluxagent/core/skill_loader.py` |
| 审计日志 | `fluxagent/core/logger.py` |
