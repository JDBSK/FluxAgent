# 统一能力路由与 Context Compiler 实现总结

日期：2026-09-12 18:43 UTC ｜ 分支：`feature/capability-routing-context`（基线 `11f0aa5`）

## 1. 解决的问题

旧架构把全部已安装 Skill/MCP 工具整表绑定给模型，隐式检索（词面交集打分）无匹配时回退"全量工具开放"；模型在此条件下产生无关工具调用（如菜谱/情感问答中调用股票工具），叠加"未知工具默认 WRITE → 每次精确参数都要审批"的策略，形成频繁授权与预算终止回环。

## 2. 新架构（四层控制面）

```text
Provider 发现 -> Capability Registry（全量登记，不给模型）
             -> Hybrid RAG Router（能力级检索与决策）
             -> RouteDecision / InvocationContract（最小候选集合）
             -> Context Compiler（选择性记忆 + 编译 prompt）
             -> 只绑定契约允许的工具
             -> Execution Gateway（执行前机械校验）
             -> 策略/审批/工具执行
```

核心原则：模型不负责从全量工具中自由选择；无足够证据时 abstain（`answer_without_tools`/`clarify`/`unavailable`），绝不回退全量工具。

## 3. 新增模块（fluxagent/capability/）

| 模块 | 职责 | 关键事实 |
|---|---|---|
| `descriptors.py` | 统一能力/操作描述 | 来源无关（builtin/mcp/prompt_skill/executable_skill）；prompt_skill 禁止携带操作（构造期校验）；不可变 + 内容指纹 |
| `registry.py` | 全量能力登记 | register/refresh/unregister/enable/disable/update_health；重名 ID、跨能力别名冲突、坏 JSON Schema（jsonschema 2020-12 check_schema）一律 fail-closed；快照版本 = 内容指纹 + 健康 + 启用态 |
| `providers.py` | 三来源适配 | builtin_descriptors / mcp_descriptors / skill_descriptors / descriptors_from_tools（metadata fallback，复刻 `mcp_<server>_` 命名降级规则）；暴露名清理与 `mcp.client._exposed_name` 一致；bind_operation_tools 绑定执行工具 |
| `retrieval.py` | 检索底层 | 纯 Python Okapi BM25（JSON 持久化）；EmbeddingProvider 接口（DeterministicHashEmbedding 离线默认 / OpenAICompatibleEmbedding 可配置）；DenseIndex 余弦检索；RRF 融合（只用户次，不相加异构分数） |
| `router.py` | 能力级路由 | 显式别名解析（最长命中，并列→clarify）> Hybrid（BM25+dense→RRF→HeuristicReranker）；准入=证据覆盖−反例惩罚−健康惩罚−缺槽位惩罚，无单一"置信度"裁决；输出不可变 RouteDecision，契约含 retry_policy（schema_error/route_mismatch/approval_rejected→stop） |
| `context_compiler.py` | 选择性记忆与编译 | MemoryRecord（scope/user_fact、user_preference、task_state、episodic、skill_context；TTL/敏感级/来源/置信度）；ProfileCompatProvider 把既有 Markdown 画像导入为结构化记录（核心不依赖 Markdown）；ContextSnapshot 按优先级（规则>契约>相关事实>任务态>失败事实>skill 片段>摘要）+ token 预算（CJK≈1 token/字，其他 4 字符/token）裁剪；注入内容全部带 record_id 与原因 |
| `gateway.py` | 执行边界 | 校验链：契约成员→快照版本→能力健康→熔断→真实 JSON Schema；越权=route_mismatch（不进审批不执行）；统一错误协议 10 态 + envelope（retryable/retry_policy 字段）；CircuitBreaker closed→open→half_open（单探测、成功复位、失败重开、按 capability 隔离） |

## 4. Agent 集成改动（fluxagent/core/agent.py）

- `create_agent_app` 新增 `capability_registry / memory_store / context_compiler` 参数；传入 registry 时执行工具集完全由 descriptor 的 operation 推导（prompt_skill 不产生工具）。
- 每用户回合：`_turn_contract()` 做显式解析（Registry 别名，含 CJK 片段，与旧 `_explicit_tool_scope` 语义一致）→ Hybrid 检索，产出 (RouteDecision, InvocationContract, scoped_tools, label)，会话内缓存；新回合在 `initialize_run_node` 重置并重载画像记忆。
- 常驻 builtin 工具并入契约：路由只决定外部能力最小集合，本地动作（时间/计算/文件）始终可用且过同一网关。
- `tools_node`：每次调用先过 ExecutionGateway；全部被拒时零执行直接返回结构化失败；部分被拒时裁剪 tool_calls 再执行；执行后按结果回报熔断器。
- 系统提示改由 ContextCompiler 生成（规则+契约+相关记忆+失败事实+skill 片段+摘要）；用户画像不再整文件注入；`[Recent context]` 由编译器注入，不再重复拼接。
- `_relevant_tool_subset` 重写为 Hybrid RAG，**删除"无匹配返回全量工具"fallback**，abstain 时只保留非外部工具。
- 失败分类（recovery.py）：`route_mismatch`/`stale_contract` 归入 policy 类、`schema_error` 归入 permanent 类——均不盲重试。

## 5. CLI 主路径（entry/main.py + fluxagent/mcp/client.py）

- `entry/main.py`：builtin + skill（含 prompt/executable 区分，doc-skill 只注册 context_provider）+ mcp descriptor 全部注册进统一 Registry，`create_agent_app` 只接收 registry；单条注册失败 fail-closed（审计 + 跳过该能力，不阻断启动）。
- `MCPToolLoadResult` 增加 `descriptors` 字段：发现工具的同一趟生成绑定工具的 CapabilityDescriptor；schema 单一事实源。
- `skill_loader` 新增公共 `scan_skill_infos()` / `read_skill_manual()`。

## 6. 兼容性与边界

- `_explicit_tool_scope` / `_tool_scope_for_turn` 保留（函数级测试与旧调用方兼容）；`_tool_scope_for_turn` 隐式分支已是 Hybrid 实现。
- 显式 `tools=[...]` 注入路径行为不变（tests/eval stub 全部兼容）。
- 审批/策略仍由 policy 层单一裁决；Gateway 只做执行边界机械校验，二者职责不重叠。
- embedding 未配置时使用确定性本地向量（索引 tag 明确标注），不伪装成语义模型；接入远程 embedding 只需配置，不改路由代码。
- 已知限制：HeuristicReranker 为无模型重排（覆盖/反例/健康/槽位四信号）；槽位缺失检查是保守包含性匹配，最终以 Gateway 的 JSON Schema 校验为准。

## 7. 测试

新增 5 个测试文件共 66 项：registry 契约 14、hybrid router/检索 19、context compiler 13、gateway/熔断 14、agent 集成 6。全量回归 289 passed / 3 skipped（含全部旧用例）。所有 fixture 使用合成能力 ID，不绑定任何真实安装项。
