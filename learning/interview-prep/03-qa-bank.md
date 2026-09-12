# 03｜高频面试问答库

## 基础概念区

### Q1：Agent 和普通对话模型有什么区别？和 RAG 呢？

结论：Agent 多了“行动 + 观察 + 迭代”的闭环。

- 普通对话：模型根据上下文直接生成回复。
- RAG：先从外部知识库检索资料，再生成，本质还是“读”。
- Agent：能制定计划、调用外部工具、观察结果、失败后修正，形成可迭代的执行闭环。

FluxAgent 证据：任务进入 LangGraph 状态图后，会经过 planner → agent → approval/policy → tools → progress，完成闭环；灌进脏数据或长任务时，靠恢复和预算兜底。

### Q2：一次工具调用，从模型到执行，完整链路是什么？

1. 系统把工具列表（名称、描述、JSON Schema）放进提示词。
2. 模型输出结构化的 `tool_calls`：工具名 + 参数。
3. 运行时校验工具存在、参数合法。
4. Policy 评估风险：放行 / 审批 / 拒绝。
5. 执行工具（本地函数、沙箱命令或 MCP 调用）。
6. 结果包装成 ToolMessage 写回上下文。
7. 模型看到工具结果后继续推理，直到能回答用户。

### Q3. 模型是怎么决定调用哪个工具的？会选错吗？

主要靠工具 description 和参数 schema 的匹配。模型把用户意图和工具说明做语义匹配，会选错，原因有：

- description 写得含糊；
- 工具太多、互相像；
- 指令注入/提示里带着假的调用信号；
- 工具本身设计有歧义。

FluxAgent 的做法：给工具注入别名和场景词（`aliases`、description 的场景说明），再让模型通过统一路由能力选择；测试里还有“显式后端约束错了就拦下来”的封闭用例。

### Q4. 模型输出的工具参数不合法怎么办？

两种处理：

1. **执行前校验**：缺必填、类型不匹配、越界路径，直接拦截，不让执行。
2. **喂回错误信息**：把校验失败消息作为 ToolMessage 交给模型，让它修正。

FluxAgent 里 `validate` 由 FluxAgentBaseTool / Pydantic 校验完成；面试需要重点讲的是：**绝不直接把 payload 丢给 `eval`/shell；用白名单 + 失败分类**。

### Q5. 工具调用失败后，为什么不能无脑重试？

因为失败原因不同：

- 瞬时失败（网络、超时）→ 重试有意义；
- 参数错误/资源不存在（重试无意义）；
- 策略/权限拒绝（重试会让权限变化吗？不会，应该终止或 replan）；
- 预算/上下文耗尽（必须熔断而不是重试）。

### Q6. 你会怎么设计重试机制？

三个维度：

1. **次数上限**：例如 `max_retries_per_step=3`；
2. **退避策略**：线性/指数退避，避免雪崩；
3. **失败分类后的动作**：transient retry，permanent/replan，policy abort。

FluxAgent 具体是先识别失败类型，再查 retry_count 和 replan_count 预算，决定动作，并且每次都会写审计日志。

### Q7. ReAct 和“先规划再执行”有什么区别？

- ReAct：模型边推理边动作，只有循环没有显式计划。
- Planner + Execution：先拆步骤再逐步执行，步骤可复盘、进度可展示、失败可定位。

FluxAgent 用的是有显式任务计划的执行循环，Planner 只负责拆解计划，执行按 `step_index` 推进，失败时明确知道是哪一步。

### Q8. 为什么要 Planning？简单任务也要规划吗？

不需要。规划本身有成本（多一次 LLM 调用）。FluxAgent 按请求复杂度决定是否进入 Planner：简单请求直接回，复杂请求才规划。这也是“把成本放进图结构”的体现。

### Q9. Agent 状态到底要存哪些东西？

不能只看 messages。需要跨节点保存的：

- 当前任务、计划、执行游标、任务状态；
- 重试次数、重规划次数、LLM 调用次数、工具轮次；
- 开始时间；
- 是否等待审批；
- 上下文裁剪指标；
- 预算耗尽原因。

FluxAgent `AgentState` 就把这些显式建模，因为从 messages 里重推这些信息既慢又不可靠。

### Q10. 没有 LangGraph 之前，你会怎么实现循环？

一个最简 Agent 循环其实可以是：

```python
while budget_ok:
    decision = llm(messages, tools)
    if decision.done: break
    result = execute(decision.tool, decision.args)
    messages.append(result)
```

但问题是：状态不显式、恢复困难、难插审批、难调试、难预算。LangGraph 把它们变成节点、边、checkpoint、interrupt，本质是工程化的状态机。

## 工程实现区

### Q11. 为什么选择 LangGraph 而不是手写循环？

- 显式状态：状态对象跨节点自动分发；
- 条件边：规划/执行/审批/失败路径清晰；
- checkpoint：会话隔离 + 断点恢复；
- interrupt：HITL 审批原生支持；
- 调试/测试：整张图可控、可回放。

### Q12. 图中的 Planner 为什么单独建节点？

让“规划”和“执行”解耦。规划失败不用执行；执行失败可以回到 replan；用户能看到的任务进度，状态可以给模型“我在第几步”。FluxAgent 用 `task_plan + step_index + task_status` 表达推进，避免模型来自历史里猜。

### Q13. 什么是 Human-in-the-loop？怎么落地？

HITL = 让高风险动作在关键节点前停下来等人工确认。FluxAgent 用三件事落地：

1. Policy 把写/变更/执行级别工具标记为 `needs_approval`；
2. LangGraph `interrupt_before=approval` 暂停图；
3. 用户批准/拒绝进入不同分支；拒绝则进入 cancelled。

### Q14. 沙箱怎么防止越权？

三层防：

1. **路径校验**：词法+解析路径需要双重校验，防止 `..`、盘符、symlink/junction；
2. **命令白名单**：只允许白名单命令，阻止任意 shell，`shell=False`，拒绝链式/重定向/变量展开；
3. **资源限制**：输出字节、读字节、执行时间上限。

### Q15. 你处理过哪些 Agent 安全问题？

1. 绝对路径越权：`C:/...`、UNC、`cd ..` 一律拒绝；
2. 安装来源：只接受 GitHub HTTPS 或本地显式路径，禁止任意 URL 拉取；
3. 命令注入：`shell=False` + 白名单可执行文件 + 防内联解释器参数（`python -c`、`node -e` 拒绝）；
4. 第三方 MCP 工具：默认按写风险，声明只读才降低；
5. 审批疲劳：工具静态定级 + 参数信号定级，放行 / 审批 / 拒绝三档。

### Q16. 上下文爆炸了怎么办？

FluxAgent 的三层策略：

1. **裁剪**：按完整用户回合裁剪，保留最近 N 回合，保证 tool_call 与 ToolMessage 不成对拆散；
2. **摘要压缩**：被裁掉的消息交给 LLM 生成摘要，写进 state.summary；
3. **长期记忆分离**：`user_profile.md` 保存跨会话偏好，不占会话窗口。

### Q17. 为什么按回合裁剪而不是按 token 裁剪？

因为剪到一半会导致：工具调用协议要求每一对 `AIMessage.tool_calls` 都有对应 `ToolMessage`，如果按 token 掐断，下轮会查不到对应结果从而导致协议错误。按完整回合保证不变串。

### Q18. 记忆系统有几种类型？

- 短期：state.summary，本轮总结，跟随 checkpoint 持久化；
- 长期：user_profile.md，跨会话用户画像；
- 显式注意：`advance_plan`、`step_index`、`retry_count` 属于运行时状态，也能被称之为记忆。

### Q19. 成本控制怎么“具体落地”？

- 用计数器 `llm_calls / tool_rounds` 做预算，而非 token 数；
- 预设最大值：40 次 LLM、24 轮工具、2 replan、3 retry；
- 执行前「预留+检查」，防止刚好超限；
- 超限后 `task_status=failed` + `budget_exceeded_reason` 可读；
- 每次运行用 `build_run_summary` 输出指标，监控。

### Q20. 预算检查和超额熔断的具体步骤？

1. 在 LLM 调用前 `budget_violation(state, budget, pending_llm_calls=1)`，若返回 reason 则停止；
2. 控制器不允许下一轮，直接结束；
3. 写审计：`action=budget_exceeded`；
4. CLI 输出 `budget_exceeded_reason` 给用户看得到；
5. 运行结束后 `build_run_summary` 可以统计 approve/deny/retry/trim。

### Q21. 审计日志里都记什么？

工具调用名、参数、结果、LLM 输入/输出摘要、系统动作、失败分类、审批动作、预算熔断、上下文裁剪。全部 JSONL，可复盘、可定位、可用于评测。

### Q22. 多会话隔离怎么做？

用 LangGraph `thread_id` 区分会话，`AsyncSqliteSaver` 持久化，`session.id == thread_id`，天然隔离。重启后可以从 checkpoint / metadata 恢复。

### Q23. Agent 测试怎么测？

FluxAgent 的测试策略：

1. 纯函数单元测试：分类、预算、路径校验、上下文裁剪都有单测（无网络、无需真实模型）；
2. 用 ScriptedLLM 录制模型输出，跑整个图断言状态和最终消息；
3. eval 用例库：把“要规划但拒绝工具”“预算压到最小”“replan 路径”“沙箱拦截”做成一致可复现的评估；
4. 红蓝测试：对计算器、shell、MCP 做对抗污染。

### Q24. 谁动权限？谁动风险？怎么编排？

- Policy 引擎先执行前拦截；
- 高风险硬拒，中风险审批；
- 沙箱执行、审计日志兜底；
- 所有走一条链，不各自漂移（`policy.py` 单一链路复用）。

### Q25. MCP 常见风险有哪些？

1. **外部服务 恶意**：要校验安装者；
2. **工具描述投毒**：description 可能引导模型错误调用；
3. **参数逃逸度高**：MCP 工具能传递任意参数，需参数校验；
4. **只读/写权限语义需要声明**：默认高风险；
5. **供应链依赖**：NPM/Python 依赖不可信。

### Q26. 项目中 MCP 的接入链路过一遍？

`install_mcp_server` → 安装器验证 → 写 index → 重启 FluxAgent → `load_installed_mcp_tools` 发现所有已注册 server → `initialize` → `list_tools` → 包装成 LangChain Tool → 加入 ToolNode → 调用时通过 MCP client `call_tool` → 结果渲染成 TextToolCall 返回模型。

### Q27. 为什么说 MCP 是 Agent 的 USB？

MCP = Agent 与工具之间的标准接口。发行一次，所有 AI 客户端都能用。类比 REST

```text
REST：HTTP 客户端 <-> 服务端
MCP：AI 客户端 <-> 工具服务端
```

好处：工具提供方写一次；AI 生态都能用；坏处：标准可能还不够统一。

### Q28. 遇到工具结果超长怎么办？

- 截断展示；
- 只保留关键片段当 ToolMessage；
- 摘要大结果；
- 但必须保持数据完整或做截断提示，不让模型無中生有。
- FluxAgent 有 `MAX_READ_BYTES` / `MAX_SHELL_OUTPUT_BYTES` 上限。

### Q29. 无限循环怎么防？

Policies 组合：

1. Budget：LLM 次数、tool 次数、时间上限；
2. Retry 要设上限和分类；
3. Replan 上限；
4. 图里的 conditional edge 条件，禁止无条件的回环；
5. Replan 替换旧计划；
6. 审计日志能发现循环再干预。

### Q30. 说出你当前版本不足/技术债？

诚实且有理是可以的，例如：

- 预算用调用次数代理 token，无法精确卡 token 成本；
- 会话恢复尚需人工 run 恢复入口；
- planner 步骤为固定顺序，复杂依赖关系（并行/条件）仍有限；
- MCP 生态版本迭代寿命，需要持续升级。

