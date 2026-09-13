# -*- coding: utf-8 -*-
"""检索质量指标：标注合成语料上的 recall@1/@3、MRR 与延迟。

语料全部使用通用合成能力（5 域 × 4 能力 × 2 操作），查询带标注标签，
包含：精确术语、同义改写、跨语言表达、易混淆对、无能力应答五类。
指标分别测量：BM25 单路、dense 单路（确定性 / 可选真实 embedding）、
Hybrid（RRF+启发式重排）——用数字回答"路由检索是否比单路更好"。

用法：
  python -m fluxagent.experiments.retrieval_metrics --output artifacts
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import Any

from fluxagent.capability.descriptors import (
    CapabilityDescriptor,
    OperationDescriptor,
    RiskLevel,
)
from fluxagent.capability.registry import CapabilityRegistry
from fluxagent.capability.retrieval import (
    BM25Index,
    DenseIndex,
    DeterministicEmbedding,
    Document,
    load_embedding_provider,
    tokenize,
)
from fluxagent.capability.router import HeuristicReranker, HybridRouter, Candidate

# ---------- 合成语料：5 域，每域 4 能力，每能力 2 操作 ----------

DOMAINS: list[tuple[str, str, tuple[tuple[str, str, str, tuple[str, ...], tuple[str, ...]], ...]]] = [
    # (domain, description_core, capabilities: (capability_id, name, desc, examples, negatives))
    ("cooking", "菜谱膳食烹饪食材", (
        ("cap.cook.meal", "MealPlanner", "按人数忌口过敏原推荐中餐菜谱和一周膳食计划",
         ("给我推荐中餐菜谱", "三个人一周菜谱", "推荐两人早餐"), ("查股票",)),
        ("cap.cook.recipe", "RecipeLookup", "查询具体菜品的做法步骤和所需食材",
         ("红烧肉怎么做", "水煮鱼的做法步骤"), ("天气如何",)),
        ("cap.cook.substitute", "IngredientSwap", "给出食材替代方案和过敏原替换建议",
         ("没有生抽用什么替代", "对虾过敏换什么"), ("翻译句子",)),
        ("cap.cook.nutrition", "NutritionCalc", "估算一餐的卡路里和营养成分",
         ("这顿饭多少卡路里", "计算蛋白质摄入"), ("查股价",)),
    )),
    ("market", "股票行情证券投资", (
        ("cap.market.quote", "QuoteService", "查询股票实时行情价格涨跌幅",
         ("查询贵州茅台行情", "AAPL 现在多少钱"), ("推荐菜谱",)),
        ("cap.market.kline", "KlineService", "获取历史K线和均线技术指标",
         ("看茅台近一月K线", "600519 的历史走势"), ("今天温度",)),
        ("cap.market.flow", "FlowService", "查询资金流向和北向资金数据",
         ("北向资金今天净流入多少", "主力资金流向"), ("菜谱推荐",)),
        ("cap.market.etf", "EtfService", "查询 ETF 基金净值和跟踪指数",
         ("机器人ETF净值", "510300 跟踪什么指数"), ("翻译文本",)),
    )),
    ("weather", "天气温度降雨空气质量", (
        ("cap.weather.now", "CurrentWeather", "查询城市当前天气温度湿度",
         ("今天南京天气", "北京现在多少度"), ("查股价",)),
        ("cap.weather.forecast", "Forecast", "查询未来几天天气预报和降雨概率",
         ("明天会下雨吗", "周末上海天气预报"), ("红烧肉做法",)),
        ("cap.weather.air", "AirQuality", "查询城市空气质量指数和污染物",
         ("今天空气质量怎么样", "南京 AQI 多少"), ("查行情",)),
        ("cap.weather.history", "WeatherHistory", "查询历史同期天气数据",
         ("去年今天南京多少度", "历史气温对比"), ("翻译",)),
    )),
    ("translate", "翻译多语言文本转换", (
        ("cap.tr.text", "TextTranslate", "把中文文本翻译成英文或目标语言",
         ("把你好翻译成英文", "翻译这段话"), ("查天气",)),
        ("cap.tr.doc", "DocTranslate", "翻译长文档并保持格式",
         ("翻译这份文档", "整篇翻成日文"), ("查股价",)),
        ("cap.tr.detect", "LangDetect", "检测文本语言并给出语种代码",
         ("这句话是什么语言", "检测语种"), ("做菜步骤",)),
        ("cap.tr.gloss", "Glossary", "查询专业术语的双语对照表",
         ("术语表里 quantum 怎么翻", "查术语对照"), ("天气",)),
    )),
    ("docs", "文件读写文档处理", (
        ("cap.doc.read", "FileRead", "读取项目内文本文件内容",
         ("读一下 readme 的内容", "打开这个文件看看"), ("查天气",)),
        ("cap.doc.write", "FileWrite", "把内容写入项目内文件",
         ("把总结写进 notes.txt", "保存这段文字"), ("查行情",)),
        ("cap.doc.list", "FileList", "列出目录下的文件清单",
         ("看看目录里有什么文件", "列一下文件"), ("翻译",)),
        ("cap.doc.search", "FileSearch", "在项目文件中搜索关键词",
         ("搜一下哪里用到了 router", "找包含关键字的文件"), ("菜谱",)),
    )),
]


def build_corpus() -> tuple[CapabilityRegistry, dict[str, int]]:
    """构建注册表语料；返回 (registry, doc_count 记录)。"""
    registry = CapabilityRegistry()
    op_seq = 0
    for domain, _core, capabilities in DOMAINS:
        for capability_id, name, desc, examples, negatives in capabilities:
            operations = []
            for suffix, risk in (("query", RiskLevel.READ), ("submit", RiskLevel.WRITE)):
                op_seq += 1
                op_name = f"mcp_{name.lower()}_{suffix}"
                operations.append(
                    OperationDescriptor(
                        capability_id=capability_id,
                        name=op_name,
                        operation_id=f"{capability_id}/{op_name}",
                        description=f"{desc}（{suffix}）",
                        input_schema={
                            "type": "object",
                            "properties": {"value": {"type": "string"}},
                            "required": ["value"],
                        },
                        required_slots=("value",),
                        side_effect_level=risk,
                        tool=object(),
                    )
                )
            registry.register(
                CapabilityDescriptor(
                    capability_id=capability_id,
                    version="1",
                    kind="mcp",
                    name=name,
                    description=desc,
                    aliases=(name.lower(),),
                    intents=(domain,),
                    examples=examples,
                    negative_examples=negatives,
                    operations=tuple(operations),
                )
            )
    return registry, {"capabilities": op_seq // 2, "operations": op_seq}


# ---------- 标注查询集 ----------

# (query, 期望能力 capability_id 或 "none")
LABELED_QUERIES: list[tuple[str, str]] = [
    # 精确术语
    ("帮我推荐中餐菜谱，四个人吃", "cap.cook.meal"),
    ("查询贵州茅台的股票行情", "cap.market.quote"),
    ("今天南京天气怎么样", "cap.weather.now"),
    ("把'晚上好'翻译成英文", "cap.tr.text"),
    ("读一下项目里的 readme 文件内容", "cap.doc.read"),
    # 同义改写（无字面重叠的关键词）
    ("明天出门要带伞吗", "cap.weather.forecast"),
    ("这顿饭热量高不高", "cap.cook.nutrition"),
    ("看看最近资金都在往哪个板块流", "cap.market.flow"),
    ("家里鸡蛋没了做蛋糕能用什么代替", "cap.cook.substitute"),
    ("这段话是哪个国家的文字", "cap.tr.detect"),
    # 跨语言 / 混合表达
    ("what's the weather like in Beijing today", "cap.weather.now"),
    ("帮我写进 todo.txt 文件里", "cap.doc.write"),
    ("A股大盘今天涨了吗", "cap.market.quote"),
    # 易混淆对（泛化词重叠，考反例与重排）
    ("分析一下茅台最近的走势好不好", "cap.market.kline"),
    ("搜索一下文件里的配置项", "cap.doc.search"),
    ("这道菜适合小孩吃吗有没有营养", "cap.cook.nutrition"),
    ("把合同文档翻成英文保持排版", "cap.tr.doc"),
    ("周末去杭州玩天气适合吗", "cap.weather.forecast"),
    # 无能力应答（应 abstain）
    ("1加1等于几", "none"),
    ("帮我写一首关于春天的诗", "none"),
    ("你觉得人生的意义是什么", "none"),
    ("讲个笑话听听", "none"),
]


def _candidate_labels_to_capability(doc_id: str) -> str:
    if doc_id.startswith("op:"):
        return doc_id[3:].split("/", 1)[0]
    if doc_id.startswith("cap:"):
        return doc_id[4:]
    return doc_id


# ---------- 规模扫描：任意能力集的词法检索质量（与已安装项无关） ----------

# 合成词库：域词池 + 修饰词，组合生成任意规模的"看起来真实"的能力卡。
# 每个能力独占若干词 + 与同域能力共享域词——复刻真实语料的混淆结构。
_SCALE_DOMAIN_WORDS = [
    "行情", "菜谱", "天气", "翻译", "文件", "日程", "库存", "工单", "账单", "物流",
    "审批", "报表", "画像", "推荐", "检索", "监控", "告警", "回执", "档案", "预约",
    "排班", "计费", "合同", "积分", "问卷", "课时", "药品", "航段", "赛事", "票据",
]
_SCALE_ACTION_WORDS = [
    "查询", "创建", "更新", "导出", "校验", "汇总", "分发", "匹配", "追踪", "归档",
]
_SCALE_FILLER = [
    "服务", "数据", "记录", "详情", "列表", "摘要", "状态", "配置", "结果", "报告",
]


def build_scale_corpus(num_capabilities: int):
    """生成通用合成能力语料（与当前安装项完全无关）。

    返回 (registry, labeled_queries, doc_count)：labeled_queries 为
    (query, capability_id) 的词法可达标注——查询由该能力独占词+域词构成。
    """
    registry = CapabilityRegistry()
    labeled: list[tuple[str, str]] = []
    per_domain = max(2, num_capabilities // len(_SCALE_DOMAIN_WORDS) + 1)
    made = 0
    for domain_index, domain_word in enumerate(_SCALE_DOMAIN_WORDS):
        for item in range(per_domain):
            if made >= num_capabilities:
                break
            action = _SCALE_ACTION_WORDS[(domain_index + item) % len(_SCALE_ACTION_WORDS)]
            filler_a = _SCALE_FILLER[(domain_index * 3 + item) % len(_SCALE_FILLER)]
            filler_b = _SCALE_FILLER[(domain_index * 7 + item * 2) % len(_SCALE_FILLER)]
            unique_token = f"u{domain_index:02d}{item:02d}"
            capability_id = f"cap.gen{domain_index:02d}.{item:02d}"
            name = f"Gen{domain_index:02d}Svc{item:02d}"
            description = f"{action}{domain_word}{filler_a}与{filler_b}的{unique_token}处理"
            op_name = f"mcp_gen{domain_index:02d}_{item:02d}_run"
            op = OperationDescriptor(
                capability_id=capability_id,
                name=op_name,
                operation_id=f"{capability_id}/{op_name}",
                description=description,
                input_schema={"type": "object", "properties": {"value": {"type": "string"}}},
                side_effect_level=RiskLevel.READ,
                tool=object(),
            )
            registry.register(
                CapabilityDescriptor(
                    capability_id=capability_id,
                    version="1",
                    kind="mcp",
                    name=name,
                    description=description,
                    intents=(domain_word,),
                    examples=(f"{action}{domain_word}{filler_a}",),
                    operations=(op,),
                )
            )
            # 词法可达查询：独占词 + 域词 + 动作
            labeled.append((f"{action}{unique_token}{domain_word}{filler_a}", capability_id))
            made += 1
        if made >= num_capabilities:
            break
    return registry, labeled, {"capabilities": made, "operations": made}


def evaluate_scale(scale_num: int) -> dict[str, Any]:
    """在指定能力规模下测词法路由质量与延迟（确定性，0 API）。

    两组指标：
    - decision_accuracy_lexical：带独占 nonce 的标注查询（可达性上界）；
    - domain_discrimination_accuracy：无 nonce 的域内混淆查询——只要求路由到
      同域任一兄弟能力，度量规模增大时域间/域内的区分度退化。
    """
    registry, labeled, counts = build_scale_corpus(scale_num)
    snapshot = registry.snapshot()
    router = HybridRouter(embedding_provider=DeterministicEmbedding(dim=256))
    started = time.perf_counter()
    router.rebuild_index(snapshot)
    index_seconds = round(time.perf_counter() - started, 3)

    correct = 0
    latencies: list[float] = []
    for query, label in labeled:
        began = time.perf_counter()
        decision = router.decide(query, snapshot)
        latencies.append((time.perf_counter() - began) * 1000.0)
        if decision.route == "route" and list(decision.capability_ids) == [label]:
            correct += 1

    # 域内混淆查询：action+domain+filler（无 nonce），期望路由到该域任一能力
    domain_groups: dict[str, set[str]] = {}
    for _query, cap_label in labeled:
        domain = cap_label.split(".")[1]
        domain_groups.setdefault(domain, set()).add(cap_label)
    discrimination_correct = 0
    discrimination_total = 0
    for domain, siblings in domain_groups.items():
        domain_index = _SCALE_DOMAIN_WORDS.index(domain)
        for item in range(max(1, len(siblings) // 2)):
            action = _SCALE_ACTION_WORDS[(domain_index + item) % len(_SCALE_ACTION_WORDS)]
            filler = _SCALE_FILLER[(domain_index * 3 + item) % len(_SCALE_FILLER)]
            query = f"{action}{domain}{filler}"
            began = time.perf_counter()
            decision = router.decide(query, snapshot)
            latencies.append((time.perf_counter() - began) * 1000.0)
            discrimination_total += 1
            if decision.route == "route" and decision.capability_ids and (
                decision.capability_ids[0] in siblings
            ):
                discrimination_correct += 1
    return {
        "capabilities": counts["capabilities"],
        "decision_accuracy_lexical": round(correct / len(labeled), 4) if labeled else 0.0,
        "domain_discrimination_accuracy": (
            round(discrimination_correct / discrimination_total, 4)
            if discrimination_total
            else None
        ),
        "avg_decision_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "index_build_seconds": index_seconds,
        "queries": len(labeled),
        "discrimination_queries": discrimination_total,
    }


def run_scale_sweep(scales=(20, 50, 100)) -> list[dict[str, Any]]:
    return [evaluate_scale(scale) for scale in scales]


def evaluate_retriever(rank_fn, k_values=(1, 3)) -> dict[str, Any]:
    """对标注查询集计算 recall@k、MRR、平均延迟。rank_fn(query)->有序 doc_id 列表。"""
    recalls = {f"recall@{k}": [] for k in k_values}
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []
    abstain_correct = 0
    abstain_total = 0
    for query, label in LABELED_QUERIES:
        started = time.perf_counter()
        ranking = rank_fn(query)
        latencies.append((time.perf_counter() - started) * 1000.0)
        if label == "none":
            abstain_total += 1
            # 检索层无候选即视为正确 abstain（路由层还有阈值兜底）
            if not ranking:
                abstain_correct += 1
            continue
        hit_rank = 0
        for rank, doc_id in enumerate(ranking, start=1):
            if _candidate_labels_to_capability(doc_id) == label:
                hit_rank = rank
                break
        for k in k_values:
            recalls[f"recall@{k}"].append(1.0 if 0 < hit_rank <= k else 0.0)
        reciprocal_ranks.append(1.0 / hit_rank if hit_rank else 0.0)
    metrics = {
        **{
            name: round(statistics.mean(values), 4) if values else 0.0
            for name, values in recalls.items()
        },
        "mrr": round(statistics.mean(reciprocal_ranks), 4) if reciprocal_ranks else 0.0,
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "abstain_precision": (
            round(abstain_correct / abstain_total, 4) if abstain_total else None
        ),
        "labeled_positive_queries": sum(1 for _, l in LABELED_QUERIES if l != "none"),
        "abstain_queries": abstain_total,
    }
    return metrics


def evaluate_route_decisions(decide_fn) -> dict[str, Any]:
    """路由决策口径：decide() 必须精确选中标注能力；none 标注必须 abstain。"""
    correct = 0
    wrong: list[dict] = []
    abstain_correct = 0
    positive = 0
    for query, label in LABELED_QUERIES:
        decision = decide_fn(query)
        if label == "none":
            if decision.route in {"answer_without_tools", "clarify", "unavailable"}:
                abstain_correct += 1
            else:
                wrong.append({"query": query, "expected": "none", "got": decision.capability_ids})
            continue
        positive += 1
        if decision.route == "route" and list(decision.capability_ids) == [label]:
            correct += 1
        else:
            wrong.append(
                {"query": query, "expected": label,
                 "got": {"route": decision.route, "capabilities": list(decision.capability_ids)}}
            )
    return {
        "decision_accuracy": round(correct / positive, 4) if positive else 0.0,
        "abstain_precision": round(abstain_correct / max(1, len(LABELED_QUERIES) - positive), 4),
        "misrouted": wrong[:10],
    }


def run_metrics(include_real_embedding: bool = False) -> dict[str, Any]:
    registry, counts = build_corpus()
    snapshot = registry.snapshot()

    # BM25 单路
    bm25 = BM25Index()
    docs = []
    for capability in snapshot.enabled_capabilities():
        docs.append(Document(f"cap:{capability.capability_id}", capability.searchable_text()))
        for operation in capability.operations:
            # 与 router.rebuild_index 的 op 文档组装保持一致（消除语料差异因素）
            op_text = "\n".join(
                part
                for part in (
                    operation.name,
                    operation.description,
                    " ".join(operation.required_slots),
                    " ".join(operation.examples),
                    capability.name,
                    " ".join(capability.intents),
                )
                if part
            )
            docs.append(Document(f"op:{operation.operation_id}", op_text))
    bm25.rebuild(docs)

    def bm25_rank(query: str) -> list[str]:
        return [doc_id for doc_id, _ in bm25.search(query, top_k=10)]

    # dense 单路（确定性）
    provider = DeterministicEmbedding(dim=256)
    dense = DenseIndex()
    dense.rebuild([(d.doc_id, d.text) for d in docs], provider)

    def dense_rank(query: str) -> list[str]:
        return [doc_id for doc_id, _ in dense.search(query, provider, top_k=10)]

    # hybrid（RRF + 启发式重排）：复用真实路由器。
    # hybrid_rank 返回路由器融合重排后的"能力级"排序（与 BM25/dense 的 top-10
    # 排序口径可比）；decide() 的最终单选行为单独用 evaluate_route_decisions 度量。
    router = HybridRouter(embedding_provider=provider)
    router.rebuild_index(snapshot)

    def hybrid_rank(query: str) -> list[str]:
        decision = router.decide(query, snapshot)
        reranked = decision.audit.get("reranked") or []
        ordered: list[str] = []
        for capability_id, _op_id, _score in reranked:
            if capability_id not in ordered:
                ordered.append(f"cap:{capability_id}")
        if not ordered and decision.route == "route":
            ordered = [f"cap:{cap_id}" for cap_id in decision.capability_ids]
        return ordered

    report: dict[str, Any] = {
        "corpus": counts,
        "queries": len(LABELED_QUERIES),
        "bm25_only": evaluate_retriever(bm25_rank),
        "dense_deterministic_only": evaluate_retriever(dense_rank),
        "hybrid_rrf_heuristic": evaluate_retriever(hybrid_rank),
        "route_decision_heuristic": evaluate_route_decisions(
            lambda q: router.decide(q, snapshot)
        ),
        "dense_real_embedding": None,
    }

    # 真实 embedding 可选轨：仅在配置了 FLUXAGENT_EMBEDDING_* 时执行
    if include_real_embedding:
        real_provider = load_embedding_provider({
            "base_url": os.getenv("FLUXAGENT_EMBEDDING_BASE_URL", ""),
            "api_key": os.getenv("FLUXAGENT_EMBEDDING_API_KEY", ""),
            "model": os.getenv("FLUXAGENT_EMBEDDING_MODEL", ""),
        })
        if type(real_provider).__name__ == "OpenAICompatibleEmbedding":
            try:
                real_dense = DenseIndex()
                real_dense.rebuild([(d.doc_id, d.text) for d in docs], real_provider)

                def real_rank(query: str) -> list[str]:
                    return [doc_id for doc_id, _ in real_dense.search(query, real_provider, top_k=10)]

                report["dense_real_embedding"] = evaluate_retriever(real_rank)
                report["dense_real_embedding"]["model_tag"] = real_provider.model_tag
            except Exception as exc:  # noqa: BLE001 - 真实服务失败如实记录为 skipped
                report["dense_real_embedding"] = {"skipped": True, "error": str(exc)[:200]}
        else:
            report["dense_real_embedding"] = {
                "skipped": True,
                "reason": "FLUXAGENT_EMBEDDING_* 未配置",
            }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="retrieval quality metrics")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--real-embedding", action="store_true")
    parser.add_argument("--scale-sweep", action="store_true", help="运行任意能力集规模扫描")
    args = parser.parse_args()
    report = run_metrics(include_real_embedding=args.real_embedding)
    if args.scale_sweep:
        report["scale_sweep"] = run_scale_sweep((20, 50, 100))
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "retrieval_metrics.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    for name in ("bm25_only", "dense_deterministic_only", "hybrid_rrf_heuristic"):
        metrics = report[name]
        print(
            f"[retrieval] {name}: recall@1={metrics['recall@1']} "
            f"recall@3={metrics['recall@3']} mrr={metrics['mrr']} "
            f"latency={metrics['avg_latency_ms']}ms"
        )
    for entry in report.get("scale_sweep", []):
        print(
            f"[scale] capabilities={entry['capabilities']}: "
            f"decision_accuracy={entry['decision_accuracy_lexical']} "
            f"latency={entry['avg_decision_latency_ms']}ms "
            f"index_build={entry['index_build_seconds']}s"
        )
    print(f"[retrieval] report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
