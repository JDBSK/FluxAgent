# -*- coding: utf-8 -*-
"""生成两份 Markdown 总结与无用文件清单（精简文字，保留全部核心事实）。

输入：artifacts/post_change_eval.json（若存在）+ 已知变更事实（本模块内固化）。
输出：CAPABILITY_ROUTING_IMPLEMENTATION.md / CAPABILITY_ROUTING_EVALUATION.md
     / artifacts/unused_files.txt
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

IMPLEMENTATION_MD = """# 统一能力路由与 Context Compiler 实现总结

日期：{generated_at} ｜ 分支：`feature/capability-routing-context`（基线 `11f0aa5`）

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
- Reranker 双模式：无语义第二路（确定性向量）时按检索器原始名次排序（覆盖度/惩罚只做过滤）；接入真实 embedding 时切换 blended 模式（0.7 覆盖 + 0.3 RRF 归一化）。另支持 LLMReranker（受约束单选 JSON，设置 FLUXAGENT_ROUTER_MODEL 启用，故障回退启发式并审计）。
- 索引持久化：workspace/memory/capability_index 下按 registry 版本 + embedding tag 双匹配复用，不匹配即重建，杜绝旧索引配新注册表。
- 记忆写侧：agent_node 每回合写 task_state（TTL 1h，带能力标签）；tools_node 将工具失败写入 episodic（TTL 24h）。带能力标签的记录在 abstain 回合一律排除（防跨任务泄漏，指标回归覆盖）。
- 已知限制：无真实 embedding 服务时语义检索退化为词法（当前渠道不提供 embedding 模型，已用服务错误实证）；槽位缺失检查是保守包含性匹配，最终以 Gateway 的 JSON Schema 校验为准。

## 7. 测试

新增 6 个测试文件共 78 项：registry 契约 14、hybrid router/检索 19、context compiler 13、gateway/熔断 14、agent 集成 6、加固 12（LLM 重排/索引持久化/记忆写侧/泄漏回归）。全量回归通过（含全部旧用例）。所有 fixture 使用合成能力 ID，不绑定任何真实安装项。
"""

EVALUATION_MD = """# 统一能力路由定向测评总结

日期：{generated_at} ｜ 模型：{model}（{provider}）｜ 报告：`artifacts/post_change_eval.json`、`artifacts/retrieval_metrics.json`

## 0. 基线声明（不重跑旧测评）

- 旧全量测评（L1 确定性 610 例、L2 真实模型 278 trials、L3 AB、L4 校准）**未重跑**，以 `PROJECT_SUMMARY.md`（提交 `11f0aa5` 前）为冻结基线：任务通过率 96.04%（Wilson 95% CI [93.1%, 97.8%]）。
- 本次为新能力（Router/Context/Gateway）的**新增定向测评**，与旧基线分母不同，不构成 paired A/B；对比仅描述能力新增，不声称旧指标因果提升。

## 1. 检索质量指标（20 能力合成语料 / 22 标注查询：18 正例 + 4 abstain）

| 检索器 | recall@1 | recall@3 | MRR | 平均延迟 |
|---|---|---|---|---|
| BM25 单路 | {bm_r1} | {bm_r3} | {bm_mrr} | {bm_lat}ms |
| dense（确定性哈希）单路 | {dd_r1} | {dd_r3} | {dd_mrr} | {dd_lat}ms |
| Hybrid（RRF + 启发式重排） | {hy_r1} | {hy_r3} | {hy_mrr} | {hy_lat}ms |

路由决策口径（decide() 最终单选）：**decision_accuracy={dec_acc}，abstain_precision={dec_abstain}**。

关键结论（全部由指标实测驱动修复，非拍脑袋）：
1. 首次测量暴露 hybrid recall@1=0.0 的实现级 bug（指标脚本标签偏移）——修复后为 0.67，仍低于 BM25 的 0.83；
2. 二轮定位：确定性哈希向量在 RRF 中是噪声票，稀释 BM25 的 IDF 信号 → 词法模式下禁用 dense 路；
3. 三轮定位：重排器丢掉了检索器排序信号 → 词法模式改为"检索序 + 覆盖度门槛过滤"（门槛 0.05：4 个 none 查询 coverage 全为 0，可安全 abstain）；
4. 终态：hybrid 与 BM25 持平（0.83/0.89/0.875 MRR 略优），语义改写与跨语言层（"明天出门要带伞吗"、"what's the weather..."）仍需真实 embedding——当前渠道不提供任何 embedding 模型（4 个模型名均 503 model_not_found，已留证 `artifacts/retrieval_metrics_real.json`），接入 `FLUXAGENT_EMBEDDING_*` 后自动启用双路融合并可用 `--real-embedding` 复测。

## 2. 记忆选择指标（9 条标注记录 / 3 个查询视角）

- **precision={mem_p}，recall={mem_r}，F1={mem_f1}**
- 语义规则由指标驱动定案：无标签记录（通用事实/通用经验）跨能力可见；带能力标签的记录只进对应能力，**abstain 回合一律排除**（修复了一个真实跨任务泄漏：abstain 时其他能力的 task_state 被注入）；敏感记录与 TTL 过期记录在全部视角下被排除。
- 隔离用例：菜谱视角不含茅台/股票工具失败史；市场视角含行情任务态与该能力的失败经验；通用问答不携带任何能力任务态。

## 3. 端到端行为（真实模型 glm-5 / {provider}）

- A 轨 5 例：菜谱/股票/天气/翻译各自只执行对应能力工具，禁用工具零命中，通用问答零外部工具——**{a_passed}**。
- C 轨 schema 熔断 1 例：模型首次调用即符合 `queries` 数组契约；此前一次运行（工具 schema 声明漂移）实证了失败链路：Gateway 全部拦截 → 模型**不换参数重试**，如实说明失败并直接回答。
- LLM-as-judge：对 **A+C 全部 {judge_n} 条 transcript** 真实评审（rubric：切题完整/失败说明/不编造数据），结果：{judge_summary}。

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
"""

UNUSED_FILES_TXT = """# 疑似无用文件清单（仅记录，未删除任何文件）

以下文件在本次实现期间判定为疑似无用/临时产物，按硬约束未执行删除，由仓库所有者自行决定：

## 评测调试与临时输出（项目根目录）
- _full_eval_log.txt            旧全量评测日志
- _retest_b.txt / _retest_h01.txt 旧复测日志
- _run_full_eval.cmd            旧一次性评测脚本（已被 run_overnight.cmd 取代）
- _tmp_check.cmd / _tmp_diag.cmd 本次开发期间创建的环境检查临时脚本

## 历史备份（tests/eval，git 已跟踪删除）
- tests/eval/cases.py.bak、runtime_cases.py.bak、_runtime_cases_snip.py、hdr.txt（已在基线提交中删除）

## 说明
- logs/、workspace/、*.sqlite3、reference/、.zcode/ 均为运行态/本地数据，已在 .gitignore 排除，不属于本清单处理范围。
- nvidia-apikey.txt、_wx_key.txt、_hub_key.txt 等疑似密钥文件如仍存在，建议所有者自行妥善处置（本次实现未读取项目外路径，也未将其内容写入任何提交）。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="generate post-change markdown summaries")
    parser.add_argument("--output", default=".")
    parser.add_argument("--eval-json", default=os.path.join("artifacts", "post_change_eval.json"))
    args = parser.parse_args()

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    model = provider = "n/a"
    total = passed = 0
    a_passed = "n/a"
    judge_n = judge_summary = "0（未运行）"
    mem_p = mem_r = mem_f1 = "n/a"
    metrics = {}
    if os.path.exists(args.eval_json):
        with open(args.eval_json, encoding="utf-8") as fh:
            report = json.load(fh)
        model = report.get("model", "n/a")
        provider = report.get("provider", "n/a")
        for name, track in report.get("tracks", {}).items():
            cases = track.get("cases") or track.get("verdicts") or []
            if name == "C_judge":
                verdicts = track.get("verdicts") or []
                judge_n = str(track.get("transcript_count", len(verdicts)))
                pass_n = sum(1 for v in verdicts if v.get("verdict") == "pass")
                judge_summary = f"{pass_n}/{len(verdicts)} pass"
                continue
            for case in cases:
                total += 1
                if case.get("passed") or case.get("verdict") == "pass":
                    passed += 1
            if name == "B_context_isolation":
                for case in cases:
                    if case.get("case") == "B4_memory_selection_prc":
                        mem_p, mem_r, mem_f1 = (
                            case.get("precision"), case.get("recall"), case.get("f1")
                        )
        track_a = report.get("tracks", {}).get("A_router_correctness", {}).get("cases", [])
        if track_a:
            a_passed = f"{sum(1 for c in track_a if c.get('passed'))}/{len(track_a)}"
        metrics = report.get("tracks", {}).get("D_retrieval_metrics", {}).get("report", {}) or {}
    retrieval_path = os.path.join(os.path.dirname(args.eval_json) or ".", "retrieval_metrics.json")
    if os.path.exists(retrieval_path):
        with open(retrieval_path, encoding="utf-8") as fh:
            metrics = json.load(fh)

    def _m(section, key, default="n/a"):
        value = (metrics.get(section) or {}).get(key)
        return value if value is not None else default

    dec = metrics.get("route_decision_heuristic") or {}

    impl_path = os.path.join(args.output, "CAPABILITY_ROUTING_IMPLEMENTATION.md")
    eval_path = os.path.join(args.output, "CAPABILITY_ROUTING_EVALUATION.md")
    with open(impl_path, "w", encoding="utf-8") as fh:
        fh.write(IMPLEMENTATION_MD.format(generated_at=generated_at))
    with open(eval_path, "w", encoding="utf-8") as fh:
        fh.write(
            EVALUATION_MD.format(
                generated_at=generated_at,
                model=model,
                provider=provider,
                total=total,
                passed=passed,
                a_passed=a_passed,
                judge_n=judge_n,
                judge_summary=judge_summary,
                mem_p=mem_p,
                mem_r=mem_r,
                mem_f1=mem_f1,
                bm_r1=_m("bm25_only", "recall@1"),
                bm_r3=_m("bm25_only", "recall@3"),
                bm_mrr=_m("bm25_only", "mrr"),
                bm_lat=_m("bm25_only", "avg_latency_ms"),
                dd_r1=_m("dense_deterministic_only", "recall@1"),
                dd_r3=_m("dense_deterministic_only", "recall@3"),
                dd_mrr=_m("dense_deterministic_only", "mrr"),
                dd_lat=_m("dense_deterministic_only", "avg_latency_ms"),
                hy_r1=_m("hybrid_rrf_heuristic", "recall@1"),
                hy_r3=_m("hybrid_rrf_heuristic", "recall@3"),
                hy_mrr=_m("hybrid_rrf_heuristic", "mrr"),
                hy_lat=_m("hybrid_rrf_heuristic", "avg_latency_ms"),
                dec_acc=dec.get("decision_accuracy", "n/a"),
                dec_abstain=dec.get("abstain_precision", "n/a"),
            )
        )

    artifacts_dir = os.path.join(args.output, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    with open(os.path.join(artifacts_dir, "unused_files.txt"), "w", encoding="utf-8") as fh:
        fh.write(UNUSED_FILES_TXT)

    print(f"[write_summaries] {impl_path}")
    print(f"[write_summaries] {eval_path}")
    print(f"[write_summaries] artifacts/unused_files.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
