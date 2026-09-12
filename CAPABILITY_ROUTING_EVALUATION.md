# 统一能力路由定向测评总结

日期：2026-09-12 18:43 UTC ｜ 模型：z-ai/glm-5.3-free（other）｜ 报告：`artifacts/post_change_eval.json`

## 0. 基线声明（不重跑旧测评）

- 旧全量测评（L1 确定性 610 例、L2 真实模型 278 trials、L3 AB、L4 校准）**未重跑**，以 `PROJECT_SUMMARY.md`（提交 `11f0aa5` 前）为冻结基线：任务通过率 96.04%（Wilson 95% CI [93.1%, 97.8%]）。
- 本次为新能力（Router/Context/Gateway）的**新增定向测评**，与旧基线分母不同，不构成 paired A/B；对比仅描述能力新增，不声称旧指标因果提升。
- 旧基线场景本身无股票类无关调用防护（这正是本次修改动机）；定向测评证明该类失败模式已被结构性消除。

## 1. 结果总览（本次实测 11 例，通过 11）

| 轨道 | 类型 | 用例 | 结果 |
|---|---|---|---|
| A router_correctness | 真实模型 | 5（repeat=1） | 5/5/5 |
| B context_isolation | 确定性（0 API） | 4 | 4/4 |
| C end_to_end（schema 熔断） | 真实模型 | 1 | 1/1 |
| C judge | LLM-as-judge | 1 | pass |

- A1 菜谱请求→只执行 `mcp_cooking_recommendMeals`；A2 股票请求→只执行 `mcp_market_getQuotes`；A3 通用问答→零外部工具；A4 天气/A5 翻译同样精确路由，禁用工具零命中。
- B1 菜谱意图注入饮食事实且排除敏感记录与跨能力任务态；B2 市场意图注入市场任务态；B3 无工具路由注入"未授权任何外部工具"契约；B4 token 预算裁剪后规则与契约段完整保留（实测 226/300 tokens）。

## 2. 关键行为证据（两次真实运行）

1. **首修前运行（工具 schema 声明漂移缺陷）**：4/5 路由用例中模型每次调用都被 Gateway 以 `schema_error` 拦截——拦截机制本身正确生效；模型的后续行为符合契约：**没有换参数重试**，而是如实向用户说明失败原因并基于上下文直接回答（A1 甚至正确引用了画像中的忌口/过敏原）。
2. **修复后运行（op schema 从工具单一事实源推导）**：11/11 通过；模型首次调用即符合 `queries` 数组契约，工具真实执行，禁用工具零命中。
3. 两次运行合并证明：路由隔离、最小绑定、schema 校验、失败停止四个机制在真实模型下按设计工作。

## 3. API 消耗

本次定向测评两次完整运行（A/C 真实调用 + judge）合计约 60-80 次模型调用，远低于 5 小时 1200 次限额；未设置人为硬上限（按用户要求移除）。确定性轨道不消耗 API。

## 4. 残余风险与后续

- HeuristicReranker 依赖描述文本质量；上线远程 embedding/reranker 后应补充跨语言与同义改写用例。
- C1 本轮模型一次命中正确 schema，"错误后停止"行为由首修前运行与确定性网关测试（`test_execution_gateway.py` 14 项）共同覆盖。
- 旧 eval 套件（tests/eval）与本次新架构的适配未做：旧 stub 场景以 `tools=` 注入路径保持兼容，未迁移到 registry 显式路径。
- `_tool_scope_for_turn` 等旧函数为兼容保留，后续可在确认无外部依赖后合并入 Router。

## 5. 运行方式

```cmd
run_overnight.cmd   :: 回归 -> 定向测评 -> 总结，任一阶段失败即停
```
产物均在项目内：`artifacts/post_change_eval.json`、`artifacts/*.log`、两份本总结。
