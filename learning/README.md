# FluxAgent 深度学习路线

这不是“背概念”的路线，而是把你手上的 FluxAgent 当教材：每学一个概念，就看项目里对应的真实实现，再跑一个最小 Demo，最后回到源码里找答案。

## 学习目标

学完后，你应该能回答 4 个问题：

1. AI 是怎么知道该调用哪个工具、参数从哪里来的？
2. Agent 为什么比普通问答强？它的循环边界在哪里？
3. MCP 和普通函数调用有什么区别？为什么说它像 REST？
4. 线上 Agent 死循环、脏数据、上下文爆炸时，分别怎么防？

## 总路线

| 阶段 | 主题 | 对应项目文件 | 结果 |
|---|---|---|---|
| 01 | 工具调用与失败重试 | `fluxagent/core/tools/base.py`、`fluxagent/core/tools/builtins.py` | 跑通一个“无大模型版工具调用器” |
| 02 | 智能循环与状态机 | `fluxagent/core/agent.py`、`fluxagent/core/recovery.py` | 画出 Planner → Agent → Tool → Progress 的流转图 |
| 03 | MCP 协议 | `fluxagent/mcp/client.py`、`fluxagent/mcp/server.py` | 明白 discover / call / schema / risk 的完整链路 |
| 04 | LangGraph 工具链 | `fluxagent/core/agent.py`、`examples/basic_usage.py` | 能讲清 StateGraph、checkpointer、interrupt 的作用 |
| 05 | 沙箱、记忆、成本 | `policy.py`、`context.py`、`budget.py`、`sandbox_tools.py` | 做一个安全与成本评估清单 |
| 06 | 故障演练 | `tests/test_recovery.py`、`test_redteam_sandbox.py` | 亲手制造三类故障并解释防线 |

## 学习方式

每个阶段固定 4 步：

1. **先跑 Demo**：不看理论，直接观察输入、输出、失败。
2. **再读源码**：只读和本阶段相关的函数，不贪多。
3. **写大白话解释**：把源码逻辑翻译成生活例子。
4. **做面试表达**：把结论压缩成 3 分钟回答。

## 当前进度

- [ ] 01 工具调用：AI 的“手”
- [ ] 02 智能循环：AI 的“工作流程”
- [ ] 03 MCP：AI 的“通用插座”
- [ ] 04 LangGraph：AI 的“流程引擎”
- [ ] 05 安全与成本：AI 的“刹车和油门”
- [ ] 06 故障演练：把坑踩成经验

## 第 1 课源码地图

- `learning/01-tool-calling/README.md`：本课讲解、源码对照和作业。
- `learning/01-tool-calling/demo_tool_call.py:78`：工具调用、参数校验、失败分类的最小实现。
- `fluxagent/core/tools/builtins.py:67`：真实项目里的时间工具。
- `fluxagent/core/tools/builtins.py:77`：真实项目里的安全计算器。
- `fluxagent/core/recovery.py:143`：失败后决定 retry / replan / abort。
- `fluxagent/core/agent.py:423`：Agent 状态图入口，第 2 课重点。
- `fluxagent/mcp/client.py:170`：MCP 工具发现和加载，第 3 课重点。

## 下一步

完成第 1 课作业后，直接说“开始第 2 课”。我会带你从单次工具调用升级到完整的智能循环，并用一个可运行 Demo 模拟：接任务 → 制定计划 → 执行 → 观察 → 失败重试/重规划 → 输出结果。
