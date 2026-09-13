# -*- coding: utf-8 -*-
"""能力级 Hybrid RAG Router 与 InvocationContract。

核心语义：
- 模型不负责从全量工具中自由选择；Router 从 Registry 全集中检索候选能力，
  生成系统授权的最小候选集合（InvocationContract）；
- 显式指定优先于检索；显式指定也必须通过健康/意图/槽位校验；
- 无显式指定时走 BM25 + dense + RRF + reranker；证据不足、候选并列歧义、
  索引不可用时 abstain（answer_without_tools / clarify / unavailable），
  绝不回退"全量工具开放"；
- RouteDecision 不可变、带 Registry/Index 版本，模型不能扩大其集合。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .descriptors import CapabilityDescriptor, OperationDescriptor, ProviderHealth
from .registry import RegistrySnapshot
from .retrieval import (
    BM25Index,
    DenseIndex,
    Document,
    EmbeddingError,
    EmbeddingProvider,
    rrf_fuse,
)

ROUTE_DECISIONS = frozenset({"route", "answer_without_tools", "clarify", "unavailable"})

# 索引构建逻辑版本：语料选择规则或索引头语义变化（如 v3 头部改记
# content_version——索引只随内容失效，不随健康翻转重建）时递增，
# 使旧持久化索引整体失效——否则代码升级后仍会加载带旧语料的索引。
INDEX_FORMAT_VERSION = "3"

_STOPWORDS = frozenset({
    "the", "and", "for", "with", "please", "help", "me", "my", "a", "an", "to",
    "的", "了", "吗", "呢", "吧", "我", "你", "他", "她", "它", "请", "帮我",
})

_EXPLICIT_PATTERN = re.compile(
    r"(?:\buse\b|\bcall\b|\bswitch\s+to\b|用|使用|调用|切换到|选择)\s*"
    r"([\w./:\-\u4e00-\u9fff]+(?:\s+[\w./:\-\u4e00-\u9fff]+){0,2})\s*"
    r"(?:\bmcp\b|\bskill\b|mcp|skill|服务|后端)",
    re.IGNORECASE,
)


# ---------- 数据结构 ----------

@dataclass(frozen=True)
class RouteDecision:
    route: str
    intent: str
    capability_ids: Tuple[str, ...] = ()
    operation_ids: Tuple[str, ...] = ()
    allowed_tool_ids: Tuple[str, ...] = ()
    evidence_ids: Tuple[str, ...] = ()
    missing_slots: Tuple[str, ...] = ()
    memory_scopes: Tuple[str, ...] = ()
    registry_version: str = ""
    index_version: str = ""
    reason: str = ""
    audit: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.route not in ROUTE_DECISIONS:
            raise ValueError(f"非法路由决策: {self.route}")


@dataclass
class InvocationContract:
    """系统授权的最小候选集合；Gateway 按它做执行前校验。"""

    decision: RouteDecision
    tools: Dict[str, Any] = field(default_factory=dict)  # tool_name -> BaseTool
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float = 600.0

    @property
    def allowed_tool_ids(self) -> Tuple[str, ...]:
        return self.decision.allowed_tool_ids

    def is_expired(self, now: Optional[float] = None) -> bool:
        return (now if now is not None else time.time()) - self.created_at > self.ttl_seconds

    def allows(self, tool_name: str) -> bool:
        return tool_name in self.tools and not self.is_expired()


# ---------- Reranker ----------

@dataclass(frozen=True)
class Candidate:
    doc_id: str
    capability_id: str
    kind: str
    name: str
    text: str
    score: float
    capability: CapabilityDescriptor
    operation: Optional[OperationDescriptor] = None


class HeuristicReranker:
    """无模型重排：绝对证据门槛 + RRF 排序信号 + 反例/健康/缺槽位惩罚。

    v3 语义：
    - 准入（evidence gate）用绝对量——查询词对全量能力卡的覆盖率；低于
      min_coverage 的候选不可路由，全部低于则整体 abstain（防"垃圾查询
      也给高分"的相对分陷阱）；
    - 排序用 RRF 归一化分参与加权，不再丢弃检索器的排序信号；
    - 能力级去重，同能力 cap/op 文档不自相拥挤。
    """

    def __init__(self, min_coverage: float = 0.05, order_by: str = "blended") -> None:
        if order_by not in {"blended", "retrieval"}:
            raise ValueError("order_by 必须是 blended 或 retrieval")
        self.min_coverage = min_coverage
        # retrieval 模式：无语义第二路时，排序尊重检索器原始名次（覆盖度/惩罚
        # 只做过滤），避免词面重打分破坏 BM25 的 IDF 排序（指标实测会掉 11 个点）
        self.order_by = order_by

    def rerank(self, query: str, candidates: Sequence[Candidate], context_turns: Optional[Sequence[str]] = None) -> List[Tuple[Candidate, float]]:
        from .retrieval import tokenize

        if not candidates:
            return []
        query_tokens = {
            token for token in tokenize(query) if len(token) >= 2 and token not in _STOPWORDS
        }
        # 覆盖度分母取信息量最高的前 12 个查询词（按长度）——长查询的 filler
        # 词会把覆盖率稀释到门槛之下，造成长请求假 abstain
        denom_tokens = sorted(query_tokens, key=len, reverse=True)[:12]
        max_fused = max(candidate.score for candidate in candidates) or 1.0

        # 能力级去重：每个能力保留融合分最高的代表文档
        best_by_capability: Dict[str, Candidate] = {}
        for candidate in candidates:
            incumbent = best_by_capability.get(candidate.capability_id)
            if incumbent is None or candidate.score > incumbent.score:
                best_by_capability[candidate.capability_id] = candidate

        scored: List[Tuple[Candidate, float]] = []
        for candidate in best_by_capability.values():
            cap = candidate.capability
            # 覆盖度信息面 = 全量能力卡 ∪ 候选 op 文本（查询词可能只命中 op 卡）
            text_tokens = set(tokenize(cap.searchable_text()))
            if candidate.operation is not None:
                text_tokens.update(tokenize(candidate.operation.description))
            coverage = len(query_tokens & text_tokens) / max(1, len(denom_tokens))
            if coverage < self.min_coverage:
                continue  # 证据不足：该候选不可路由
            rrf_norm = candidate.score / max_fused
            negative_hits = sum(
                1
                for example in cap.negative_examples
                if _significant_overlap(query, example)
            )
            health_penalty = {
                ProviderHealth.HEALTHY: 0.0,
                ProviderHealth.UNKNOWN: 0.05,
                ProviderHealth.DEGRADED: 0.25,
                ProviderHealth.UNAVAILABLE: 1.0,
            }[cap.provider_health]
            if health_penalty >= 1.0:
                continue  # 不可用能力在检索序模式下也无条件排除（与显式路径一致）
            missing = _missing_required_slots(candidate, query)
            score = (
                0.7 * coverage
                + 0.3 * rrf_norm
                - 0.5 * negative_hits
                - health_penalty
                - 0.1 * len(missing)
            )
            scored.append((candidate, score))
        if self.order_by == "retrieval":
            # 保持 RRF 融合名次（即 BM25 名次），只保留通过门槛/惩罚的候选；
            # 返回的分数仍是启发式分，供 _decide_hybrid 的阈值判定使用
            scored.sort(key=lambda kv: (-kv[0].score, kv[0].doc_id))
            return scored
        scored.sort(key=lambda kv: (-kv[1], kv[0].doc_id))
        return scored


def _significant_overlap(text_a: str, text_b: str) -> bool:
    from .retrieval import tokenize

    tokens_a = {t for t in tokenize(text_a) if len(t) >= 2}
    tokens_b = {t for t in tokenize(text_b) if len(t) >= 2}
    if not tokens_a or not tokens_b:
        return False
    overlap = tokens_a & tokens_b
    return len(overlap) / min(len(tokens_a), len(tokens_b)) >= 0.5


class LLMReranker:
    """受约束 LLM 路由重排：只看 Top-K 能力卡片，输出结构化决策。

    与"把全量工具交给模型"的本质区别：这里模型只在 RRF 已召回的少数候选
    中做受约束选择，输出 JSON；解析失败/超时/异常一律回退启发式重排并在
    audit 中标注 fallback，绝不扩大候选集。
    """

    def __init__(self, chat_model: Any, fallback: Optional[Any] = None) -> None:
        self.chat_model = chat_model
        self.fallback = fallback or HeuristicReranker()
        self.last_fallback_reason = ""

    def rerank(self, query: str, candidates: Sequence[Candidate], context_turns: Optional[Sequence[str]] = None) -> List[Tuple[Candidate, float]]:
        if not candidates:
            return []
        prompt = self._build_prompt(query, candidates, context_turns=context_turns)
        try:
            response = self.chat_model.invoke(prompt)
            text = str(getattr(response, "content", response))
            selection = self._parse_selection(text, candidates)
            if selection is None:
                raise ValueError("无法解析路由器 JSON 输出")
            self.last_fallback_reason = ""
            return selection
        except Exception as exc:  # noqa: BLE001 - 任何 LLM 故障回退启发式，不阻断路由
            self.last_fallback_reason = f"{type(exc).__name__}: {exc}"[:200]
            return self.fallback.rerank(query, candidates)

    @staticmethod
    def _build_prompt(
        query: str,
        candidates: Sequence[Candidate],
        context_turns: Optional[Sequence[str]] = None,
    ) -> str:
        lines = [
            "你是严格的能力路由器。根据用户请求，从候选能力中选择至多一个，"
            "只在证据充分时选择；不确定必须选 none。\n",
            f"用户请求: {query}\n",
        ]
        if context_turns:
            # 会话近期请求：帮助解析"回到刚才的话题"等指代表达
            recent = " | ".join(str(t)[:80] for t in context_turns[-2:])
            lines.append(f"会话近期请求（仅供理解指代，不作为独立依据）: {recent}\n")
        lines.append("候选能力（编号|ID|描述|适用意图|输入必需槽位）:")
        for index, candidate in enumerate(candidates):
            operation = candidate.operation
            slots = ",".join(operation.required_slots) if operation else ""
            # 能力边界必须让 LLM 可见：负例（明确排除的场景）是路由判定的
            # 关键依据，缺失它会导致"比特币→股票"类边界外请求被错配
            negatives = ",".join(candidate.capability.negative_examples[:4])
            negative_note = f" | 排除(应选none): {negatives}" if negatives else ""
            lines.append(
                f"{index}|{candidate.capability_id}|{candidate.text[:120]}|"
                f"{','.join(candidate.capability.intents)}|{slots}{negative_note}"
            )
        lines.append(
            '只输出 JSON：{"selected": <编号或-1>, "reason": "<40字内>"}'
        )
        return "\n".join(lines)

    def _parse_selection(
        self, text: str, candidates: Sequence[Candidate]
    ) -> Optional[List[Tuple[Candidate, float]]]:
        import json as _json

        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        payload = _json.loads(text[start : end + 1])
        selected = int(payload.get("selected", -1))
        if selected < 0 or selected >= len(candidates):
            return []  # 模型明确 abstain
        ordered = [(candidates[selected], 1.0)]
        ordered.extend(
            (candidate, 0.5 - rank * 0.01)
            for rank, candidate in enumerate(candidates)
            if candidate is not candidates[selected]
        )
        return ordered


def _missing_required_slots(candidate: Candidate, query: str) -> Tuple[str, ...]:
    """必需槽位无法从请求文本中推断时视为缺失（简单包含性检查，Gateway 仍做强校验）。"""
    if candidate.operation is None:
        return ()
    missing = []
    for slot in candidate.operation.required_slots:
        slot_variants = {slot.casefold(), slot.casefold().replace("_", "")}
        if not any(variant in query.casefold().replace(" ", "") for variant in slot_variants):
            missing.append(slot)
    return tuple(missing)


# ---------- Router ----------

class HybridRouter:
    """显式解析 + BM25/dense/RRF + 重排 → RouteDecision/InvocationContract。"""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        reranker: Optional[Any] = None,
        top_k: int = 8,
        min_rerank_score: float = 0.05,
        contract_ttl_seconds: float = 600.0,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k 必须大于 0")
        self.embedding_provider = embedding_provider
        from .retrieval import DeterministicEmbedding

        self.reranker = reranker or HeuristicReranker(
            order_by=(
                "retrieval"
                if isinstance(embedding_provider, DeterministicEmbedding)
                else "blended"
            )
        )
        self.top_k = top_k
        self.min_rerank_score = min_rerank_score
        self.contract_ttl_seconds = contract_ttl_seconds
        self._bm25 = BM25Index()
        self._dense = DenseIndex()
        self._index_registry_version = ""
        self._index_lock = __import__("threading").RLock()
        self._docs_by_id: Dict[str, Tuple[CapabilityDescriptor, Optional[OperationDescriptor]]] = {}

    # ---------- 索引维护 ----------

    def rebuild_index(self, snapshot: RegistrySnapshot) -> None:
        """原子重建：新索引在本地构建（含远程 embedding 调用），全部成功后
        在锁内一次性换装——失败不撕裂状态，并发读者不见中间态。"""
        docs: List[Document] = []
        mapping: Dict[str, Tuple[CapabilityDescriptor, Optional[OperationDescriptor]]] = {}
        for capability in snapshot.enabled_capabilities():
            if capability.kind == "builtin":
                # builtin 内置工具每回合无条件可用，不是路由目标；
                # 进入检索索引只会让"工具不存在"的请求错配到内置工具上
                # （真实语料实测：查行情→路由到 list_scheduled_tasks）
                continue
            docs.append(
                Document(
                    doc_id=f"cap:{capability.capability_id}",
                    text=capability.searchable_text(),
                    payload={"capability_id": capability.capability_id, "kind": capability.kind},
                )
            )
            mapping[f"cap:{capability.capability_id}"] = (capability, None)
            for op in capability.operations:
                op_text = "\n".join(
                    part
                    for part in (
                        op.name,
                        op.description,
                        " ".join(op.required_slots),
                        " ".join(op.examples),
                        capability.name,
                        " ".join(capability.intents),
                    )
                    if part
                )
                docs.append(
                    Document(
                        doc_id=f"op:{op.operation_id}",
                        text=op_text,
                        payload={"capability_id": capability.capability_id, "operation_id": op.operation_id},
                    )
                )
                mapping[f"op:{op.operation_id}"] = (capability, op)
        new_bm25 = BM25Index()
        new_bm25.rebuild(docs)
        new_dense = DenseIndex()
        new_dense.rebuild([(d.doc_id, d.text) for d in docs], self.embedding_provider)
        with self._index_lock:
            self._bm25 = new_bm25
            self._dense = new_dense
            self._docs_by_id = mapping
            # 索引只随内容失效：健康翻转推进 snapshot.version（契约/网关语义）
            # 但不推进 content_version，此处记录内容版本
            self._index_registry_version = snapshot.content_version

    @property
    def index_version(self) -> str:
        with self._index_lock:
            return f"{self._index_registry_version}/{self._bm25_knowledge_len()}"

    def _bm25_knowledge_len(self) -> int:
        return len(self._docs_by_id)

    # ---------- 索引持久化 ----------

    def save_index(self, directory: str) -> None:
        """按 registry 版本持久化两路索引（原子写 + 每文件内嵌校验头）。"""
        import os

        os.makedirs(directory, exist_ok=True)
        with self._index_lock:
            header = {
                "format": INDEX_FORMAT_VERSION,
                "content_version": self._index_registry_version,
                "embedding_tag": self.embedding_provider.model_tag,
            }
            self._bm25.save(os.path.join(directory, "bm25.json"), header)
            self._dense.save(os.path.join(directory, "dense.json"), header)
            with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as fh:
                import json

                json.dump(
                    {
                        **header,
                        "doc_count": len(self._docs_by_id),
                    },
                    fh,
                    ensure_ascii=False,
                )

    def load_index(self, directory: str, snapshot: RegistrySnapshot) -> bool:
        """尝试加载与当前 registry 内容版本和 embedding tag 匹配的持久化索引。

        成功返回 True（不再重建）；索引格式版本、内容版本、embedding tag
        任一不匹配或文件损坏时返回 False（调用方必须 rebuild），绝不允许旧
        索引配新代码/新注册表。健康翻转不改变内容版本，因此不会导致重建。
        """
        import json
        import os

        meta_path = os.path.join(directory, "meta.json")
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            return False
        if meta.get("format") != INDEX_FORMAT_VERSION:
            return False
        if meta.get("content_version") != snapshot.content_version:
            return False
        if meta.get("embedding_tag") != self.embedding_provider.model_tag:
            return False
        try:
            bm25 = BM25Index.load(os.path.join(directory, "bm25.json"))
            dense = DenseIndex.load(os.path.join(directory, "dense.json"))
            # 逐文件校验头：防止多进程交错写出的"meta 新 + 数据旧"组合
            for payload_header in (bm25._header, dense._header):
                if payload_header.get("format") != INDEX_FORMAT_VERSION \
                        or payload_header.get("content_version") != snapshot.content_version \
                        or payload_header.get("embedding_tag") != self.embedding_provider.model_tag:
                    return False
        except (OSError, ValueError, KeyError, AttributeError):
            return False
        with self._index_lock:
            self._bm25 = bm25
            self._dense = dense
            self._index_registry_version = snapshot.content_version
            # doc_id -> descriptor 映射从 snapshot 重建（与 rebuild 的语料选择
            # 规则一致：排除 builtin；描述可内存推导，无需持久化）
            mapping: Dict[str, Tuple[CapabilityDescriptor, Optional[OperationDescriptor]]] = {}
            for capability in snapshot.enabled_capabilities():
                if capability.kind == "builtin":
                    continue
                mapping[f"cap:{capability.capability_id}"] = (capability, None)
                for operation in capability.operations:
                    mapping[f"op:{operation.operation_id}"] = (capability, operation)
            self._docs_by_id = mapping
        return True

    # ---------- 决策 ----------

    def decide(
        self,
        user_text: str,
        snapshot: RegistrySnapshot,
        *,
        explicit_capability: Optional[str] = None,
        context_turns: Optional[Sequence[str]] = None,
        hint_fallback_to_retrieval: bool = False,
    ) -> RouteDecision:
        text = str(user_text or "").strip()
        # 索引失效只看内容版本：健康翻转走契约/Gateway 失效路径，不重建索引
        if self._index_registry_version != snapshot.content_version:
            try:
                self.rebuild_index(snapshot)
            except EmbeddingError as exc:
                # 重建失败（如远程 embedding 不可用）：明确 unavailable，
                # 不抛出、不留下撕裂的中间索引状态
                return self._decision(
                    route="unavailable",
                    intent="retrieval_provider_error",
                    snapshot=snapshot,
                    reason=f"索引重建失败: {exc}",
                    audit={"path": "decide", "error": str(exc)[:200]},
                )

        resolved = explicit_capability or self._resolve_explicit(
            text, snapshot, hint_fallback_to_retrieval=hint_fallback_to_retrieval
        )
        if resolved is not None:
            return self._decide_explicit(resolved, text, snapshot)

        decision = self._decide_hybrid(text, snapshot)
        # 会话上下文救援：主决策 abstain/clarify 且本会话有近期请求时，
        # 用近期文本辅助检索（解决"回到刚才的话题"类指代路由）。
        # 只在证据不足时启用，不扰动已有明确路由。
        if (
            decision.route in {"answer_without_tools", "clarify"}
            and context_turns
        ):
            rescue = self._decide_hybrid(text, snapshot, context_turns=context_turns)
            if rescue.route == "route":
                rescue.audit["context_rescue"] = True
                return rescue
            return decision
        return decision

    # ---------- 显式路由 ----------

    @staticmethod
    def _alias_in_text(alias: str, lowered_text: str) -> bool:
        """别名命中判定：ASCII 别名要求词边界（避免 search 命中 research），
        CJK 别名保持子串匹配。"""
        import re as _re

        if all(ord(ch) < 128 for ch in alias):
            return _re.search(
                rf"(?<![a-z0-9_]){ _re.escape(alias)}(?![a-z0-9_])", lowered_text
            ) is not None
        return alias in lowered_text

    def _resolve_explicit(
        self,
        text: str,
        snapshot: RegistrySnapshot,
        *,
        hint_fallback_to_retrieval: bool = False,
    ) -> Optional[str]:
        """解析'使用 X mcp/skill'语法和别名包含；最长匹配，并列视为歧义。

        hint_fallback_to_retrieval：别名无命中但出现"用/调用 ... 服务"句式时，
        True 返回 None（回落检索，供短步骤文本等生成式输入使用）；默认 False
        返回 ""（fail-closed clarify，用户显式点名未注册后端必须澄清）。
        """
        alias_map: Dict[str, List[str]] = {}
        for capability in snapshot.enabled_capabilities():
            if capability.kind == "builtin":
                continue
            aliases = set()
            aliases.add(capability.capability_id.casefold())
            aliases.add(capability.name.strip().casefold())
            for alias in capability.aliases:
                normalized = alias.strip().casefold()
                if len(normalized) >= 2:
                    aliases.add(normalized)
            for alias in aliases:
                alias_map.setdefault(alias, []).append(capability.capability_id)

        lowered = text.casefold()
        matches: List[Tuple[int, str]] = []
        for alias, capability_ids in alias_map.items():
            if len(alias) >= 2 and self._alias_in_text(alias, lowered):
                if len(capability_ids) > 1:
                    # 同名别名跨能力共享：无法唯一解析，必须 clarify 而非静默取第一个
                    return ""
                matches.append((len(alias), capability_ids[0]))
        if matches:
            matches.sort(reverse=True)
            best_len = matches[0][0]
            best = {cap_id for length, cap_id in matches if length == best_len}
            if len(best) == 1:
                return next(iter(best))
            return ""  # 歧义：显式信号存在但无法唯一解析 → clarify

        hint = _EXPLICIT_PATTERN.search(text)
        if hint:
            return "" if not hint_fallback_to_retrieval else None
        return None

    def _decide_explicit(
        self, capability_id: str, text: str, snapshot: RegistrySnapshot
    ) -> RouteDecision:
        base_audit = {"path": "explicit", "requested": capability_id or "(ambiguous)"}
        if capability_id == "":
            return self._decision(
                route="clarify",
                intent="explicit_backend_unresolved",
                snapshot=snapshot,
                reason="用户显式指定了能力，但无法唯一解析或该能力未注册",
                audit=base_audit,
            )
        capability = snapshot.get(capability_id)
        if capability is None or not capability.enabled:
            return self._decision(
                route="clarify",
                intent="explicit_backend_missing",
                snapshot=snapshot,
                reason=f"显式指定的能力未注册或已禁用: {capability_id}",
                audit=base_audit,
            )
        if capability.provider_health in {ProviderHealth.UNAVAILABLE, ProviderHealth.DEGRADED}:
            return self._decision(
                route="unavailable",
                intent="explicit_backend_unhealthy",
                snapshot=snapshot,
                reason=f"能力当前不可用: {capability_id} ({capability.provider_health.value})",
                audit=base_audit,
            )
        return self._build_route(capability, text, snapshot, audit=base_audit, path="explicit")

    # ---------- Hybrid 路由 ----------

    def _decide_hybrid(
        self,
        text: str,
        snapshot: RegistrySnapshot,
        *,
        context_turns: Optional[Sequence[str]] = None,
    ) -> RouteDecision:
        if not self._docs_by_id:
            return self._decision(
                route="answer_without_tools",
                intent="no_registered_capability",
                snapshot=snapshot,
                reason="注册表为空：没有可路由的外部能力",
                audit={"path": "hybrid"},
            )
        # 会话上下文追加进检索查询（仅救援路径传入），提升指代轮的词法召回
        retrieval_text = text
        if context_turns:
            retrieval_text = text + " " + " ".join(str(t) for t in context_turns[-2:])
        try:
            bm25_ranking = self._bm25.search(retrieval_text, top_k=self.top_k)
            rankings = [bm25_ranking]
            # 双路融合只在第二路具备真实语义信号时启用；确定性哈希向量本质
            # 是词面信息的向量重排，参与 RRF 只会稀释 BM25 的 IDF 信号
            # （检索指标实测：0.83→0.44），故此场景退化为单路词法检索。
            from .retrieval import DeterministicEmbedding

            if not isinstance(self.embedding_provider, DeterministicEmbedding):
                rankings.append(
                    self._dense.search(retrieval_text, self.embedding_provider, top_k=self.top_k)
                )
        except EmbeddingError as exc:
            return self._decision(
                route="unavailable",
                intent="retrieval_provider_error",
                snapshot=snapshot,
                reason=f"检索组件不可用: {exc}",
                audit={"path": "hybrid", "error": str(exc)[:200]},
            )
        fused = rrf_fuse(rankings, top_k=self.top_k)
        if not fused:
            return self._decision(
                route="answer_without_tools",
                intent="no_retrieval_evidence",
                snapshot=snapshot,
                reason="两路检索均无候选",
                audit={"path": "hybrid"},
            )

        candidates: List[Candidate] = []
        for doc_id, fused_score in fused:
            entry = self._docs_by_id.get(doc_id)
            if entry is None:
                continue
            capability, operation = entry
            candidates.append(
                Candidate(
                    doc_id=doc_id,
                    capability_id=capability.capability_id,
                    kind=capability.kind,
                    name=capability.name,
                    text=(operation.description if operation else capability.searchable_text()),
                    score=fused_score,
                    capability=capability,
                    operation=operation,
                )
            )
        ranked = self.reranker.rerank(
            retrieval_text if context_turns else text,
            candidates,
            context_turns=list(context_turns) if context_turns else None,
        )
        if not ranked:
            return self._decision(
                route="answer_without_tools",
                intent="no_reranked_candidate",
                snapshot=snapshot,
                reason="重排后无候选",
                audit={"path": "hybrid"},
            )

        best_candidate, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        ambiguous = (
            len(ranked) > 1
            and abs(best_score - second_score) < 1e-9
            and ranked[1][0].capability_id != best_candidate.capability_id
        )
        # B2 补丁：词法模式下检索器原始分同分（不同能力）也是并列歧义——
        # RRF 名次化会把原始同分变成相邻名次差，启发式分也可能不再相等，
        # 仅靠 ranked 分数判歧义会漏检
        if not ambiguous and len(bm25_ranking) >= 2:
            (top_id, top_raw), (second_id, second_raw) = bm25_ranking[0], bm25_ranking[1]
            if abs(top_raw - second_raw) < 1e-9:
                top_cap = self._docs_by_id.get(top_id)
                second_cap = self._docs_by_id.get(second_id)
                if (
                    top_cap is not None
                    and second_cap is not None
                    and top_cap[0].capability_id != second_cap[0].capability_id
                ):
                    ambiguous = True
        audit = {
            "path": "hybrid",
            "fused": [(doc_id, round(score, 6)) for doc_id, score in fused],
            "reranked": [
                (c.capability_id, c.operation.operation_id if c.operation else None, round(s, 4))
                for c, s in ranked
            ],
            "best_score": round(best_score, 4),
            "second_score": round(second_score, 4),
        }
        if best_score < self.min_rerank_score or ambiguous:
            return self._decision(
                route="clarify" if ambiguous else "answer_without_tools",
                intent="insufficient_routing_evidence",
                snapshot=snapshot,
                reason=(
                    "候选能力并列歧义，需要澄清"
                    if ambiguous
                    else f"重排得分不足: {best_score:.3f} < {self.min_rerank_score}"
                ),
                audit=audit,
            )
        return self._build_route(
            best_candidate.capability, text, snapshot, audit=audit, path="hybrid"
        )

    # ---------- 契约构建 ----------

    def _build_route(
        self,
        capability: CapabilityDescriptor,
        text: str,
        snapshot: RegistrySnapshot,
        *,
        audit: Mapping[str, Any],
        path: str,
    ) -> RouteDecision:
        if capability.provider_health in {ProviderHealth.UNAVAILABLE, ProviderHealth.DEGRADED}:
            # B4 补丁：hybrid 路径与显式路径的健康门槛保持一致（fail-closed）
            return self._decision(
                route="unavailable",
                intent="capability_unhealthy",
                snapshot=snapshot,
                reason=f"能力当前不可用: {capability.capability_id} ({capability.provider_health.value})",
                audit=audit,
            )
        operations = list(capability.operations)
        tool_ids: Tuple[str, ...] = tuple(op.name for op in operations if op.tool is not None)
        operation_ids = tuple(op.operation_id for op in operations)
        # S3：missing_slots 按最优匹配 operation 计算（取缺口最小的 op），
        # 不再把用户未涉及操作的槽位全部报缺失
        per_op_missing = [
            _missing_required_slots(
                Candidate(
                    doc_id="",
                    capability_id=capability.capability_id,
                    kind=capability.kind,
                    name=capability.name,
                    text=op.description,
                    score=0.0,
                    capability=capability,
                    operation=op,
                ),
                text,
            )
            for op in operations
        ]
        missing = sorted(min(per_op_missing, key=len)) if per_op_missing else []
        intent = capability.intents[0] if capability.intents else capability.capability_id
        memory_scopes = ("user_fact", "user_preference", "task_state", "episodic")
        return self._decision(
            route="route",
            intent=intent,
            snapshot=snapshot,
            capability_ids=(capability.capability_id,),
            operation_ids=operation_ids,
            allowed_tool_ids=tool_ids,
            missing_slots=tuple(missing),
            memory_scopes=memory_scopes,
            reason=f"路由到能力 {capability.capability_id}（{path}）",
            audit={**audit, "missing_slots": missing},
        )

    def _decision(
        self,
        *,
        route: str,
        intent: str,
        snapshot: RegistrySnapshot,
        reason: str = "",
        capability_ids: Tuple[str, ...] = (),
        operation_ids: Tuple[str, ...] = (),
        allowed_tool_ids: Tuple[str, ...] = (),
        missing_slots: Tuple[str, ...] = (),
        memory_scopes: Tuple[str, ...] = (),
        audit: Mapping[str, Any] = None,
    ) -> RouteDecision:
        return RouteDecision(
            route=route,
            intent=intent,
            capability_ids=capability_ids,
            operation_ids=operation_ids,
            allowed_tool_ids=allowed_tool_ids,
            missing_slots=missing_slots,
            memory_scopes=memory_scopes,
            registry_version=snapshot.version,
            index_version=self.index_version,
            reason=reason,
            audit=dict(audit or {}),
        )

    # ---------- 契约 ----------

    def build_contract(
        self, decision: RouteDecision, snapshot: RegistrySnapshot
    ) -> InvocationContract:
        tools: Dict[str, Any] = {}
        if decision.route == "route":
            for capability_id in decision.capability_ids:
                capability = snapshot.get(capability_id)
                if capability is None:
                    continue
                for op in capability.operations:
                    if op.tool is not None and op.name in decision.allowed_tool_ids:
                        tools[op.name] = op.tool
        return InvocationContract(
            decision=decision,
            tools=tools,
            ttl_seconds=self.contract_ttl_seconds,
        )
