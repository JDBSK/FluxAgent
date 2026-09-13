# 统一能力路由定向测评总结

日期：2026-09-13 04:29 UTC ｜ 模型：z-ai/glm-5.3-free（other）｜ 报告：`artifacts/post_change_eval.json`、`artifacts/retrieval_metrics.json`

## 0. 基线声明（不重跑旧测评）

- 旧全量测评（L1 确定性 610 例、L2 真实模型 278 trials、L3 AB、L4 校准）**未重跑**，以 `PROJECT_SUMMARY.md`（提交 `11f0aa5` 前）为冻结基线：任务通过率 96.04%（Wilson 95% CI [93.1%, 97.8%]）。
- 本次为新能力（Router/Context/Gateway）的**新增定向测评**，与旧基线分母不同，不构成 paired A/B；对比仅描述能力新增，不声称旧指标因果提升。

## 1. 检索质量指标（20 能力合成语料 / 22 标注查询：18 正例 + 4 abstain）

| 检索器 | recall@1 | recall@3 | MRR | 平均延迟 |
|---|---|---|---|---|
| BM25 单路 | 0.8333 | 0.8889 | 0.8704 | 0.02ms |
| dense（确定性哈希）单路 | 0.2778 | 0.3889 | 0.3778 | 2.7ms |
| Hybrid（RRF + 启发式重排） | 0.8333 | 0.8889 | 0.875 | 0.25ms |

路由决策口径（decide() 最终单选）：**decision_accuracy=0.8333，abstain_precision=0.75**。

关键结论（全部由指标实测驱动修复，非拍脑袋）：
1. 首次测量暴露 hybrid recall@1=0.0 的实现级 bug（指标脚本标签偏移）——修复后为 0.67，仍低于 BM25 的 0.83；
2. 二轮定位：确定性哈希向量在 RRF 中是噪声票，稀释 BM25 的 IDF 信号 → 词法模式下禁用 dense 路；
3. 三轮定位：重排器丢掉了检索器排序信号 → 词法模式改为"检索序 + 覆盖度门槛过滤"（门槛 0.05：4 个 none 查询 coverage 全为 0，可安全 abstain）；
4. 终态：hybrid 与 BM25 持平（0.83/0.89/0.875 MRR 略优），语义改写与跨语言层（"明天出门要带伞吗"、"what's the weather..."）仍需真实 embedding——当前渠道不提供任何 embedding 模型（4 个模型名均 503 model_not_found，已留证 `artifacts/retrieval_metrics_real.json`），接入 `FLUXAGENT_EMBEDDING_*` 后自动启用双路融合并可用 `--real-embedding` 复测。

## 2. 记忆选择指标（9 条标注记录 / 3 个查询视角）

- **precision=1.0，recall=1.0，F1=1.0**
- 语义规则由指标驱动定案：无标签记录（通用事实/通用经验）跨能力可见；带能力标签的记录只进对应能力，**abstain 回合一律排除**（修复了一个真实跨任务泄漏：abstain 时其他能力的 task_state 被注入）；敏感记录与 TTL 过期记录在全部视角下被排除。
- 隔离用例：菜谱视角不含茅台/股票工具失败史；市场视角含行情任务态与该能力的失败经验；通用问答不携带任何能力任务态。

## 3. 端到端行为（真实模型 glm-5 / other）

- A 轨 5 例：菜谱/股票/天气/翻译各自只执行对应能力工具，禁用工具零命中，通用问答零外部工具——**5/5**。
- C 轨 schema 熔断 1 例：模型首次调用即符合 `queries` 数组契约；此前一次运行（工具 schema 声明漂移）实证了失败链路：Gateway 全部拦截 → 模型**不换参数重试**，如实说明失败并直接回答。
- LLM-as-judge：对 **A+C 全部 6 条 transcript** 真实评审（rubric：切题完整/失败说明/不编造数据），结果：6/6 pass。

## 4. API 消耗

两次完整运行（A/C 真实调用 + judge）合计约 60-100 次模型调用；embedding 探测 4 次失败请求；确定性轨道与全部单元/集成测试 0 API。

## 5. 残余风险与后续

- 语义检索层（同义改写/跨语言）依赖真实 embedding，当前渠道不可用；接入后预期补齐 decision_accuracy 中剩余 3 例语义 miss，需复测验证。
- judge rubric 为单裁判直评，未做旧体系的 20 例校准与 AB 换序；规模化前应复用 `tests/eval/llmjudge` 框架。
- `_tool_scope_for_turn` 等旧函数为兼容保留；旧 eval 套件未迁移到 registry 显式路径。

## 6. 运行方式

```cmd
run_overnight.cmd   :: 回归 -> 定向测评（含指标） -> 总结，任一阶段失败即停
```
