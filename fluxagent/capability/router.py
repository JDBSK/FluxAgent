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

_STOPWORDS = frozenset({
    "the", "and", "for", "with", "please", "help", "me", "my", "a", "an", "to",
    "的", "了", "吗", "呢", "吧", "我", "你", "他", "她", "它", "请", "帮我",
})

_EXPLICIT_PATTERN = re.compile(
    r"(?:用|使用|调用|切换到|选择|use|call|switch\s+to)\s*"
    r"([\w./:\-\u4e00-\u9fff]+(?:\s+[\w./:\-\u4e00-\u9fff]+){0,2})\s*"
    r"(mcp|skill|服务|后端)",
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
    """无模型重排：证据覆盖 + 反例惩罚 + 健康惩罚 + 缺槽位惩罚。

    显式服务于"置信度不是唯一准入"的原则：只有多信号齐备才允许 route。
    """

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> List[Tuple[Candidate, float]]:
        from .retrieval import tokenize

        query_tokens = {
            token for token in tokenize(query) if len(token) >= 2 and token not in _STOPWORDS
        }
        scored: List[Tuple[Candidate, float]] = []
        for candidate in candidates:
            cap = candidate.capability
            text_tokens = set(tokenize(candidate.text))
            coverage = len(query_tokens & text_tokens) / max(1, len(query_tokens))
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
            missing = _missing_required_slots(candidate, query)
            score = coverage - 0.5 * negative_hits - health_penalty - 0.1 * len(missing)
            scored.append((candidate, score))
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
        top_k: int = 4,
        min_rerank_score: float = 0.05,
        contract_ttl_seconds: float = 600.0,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k 必须大于 0")
        self.embedding_provider = embedding_provider
        self.reranker = reranker or HeuristicReranker()
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
        docs: List[Document] = []
        mapping: Dict[str, Tuple[CapabilityDescriptor, Optional[OperationDescriptor]]] = {}
        for capability in snapshot.enabled_capabilities():
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
        with self._index_lock:
            self._bm25.rebuild(docs)
            self._dense.rebuild([(d.doc_id, d.text) for d in docs], self.embedding_provider)
            self._docs_by_id = mapping
            self._index_registry_version = snapshot.version

    @property
    def index_version(self) -> str:
        with self._index_lock:
            return f"{self._index_registry_version}/{self._bm25_knowledge_len()}"

    def _bm25_knowledge_len(self) -> int:
        return len(self._docs_by_id)

    # ---------- 决策 ----------

    def decide(
        self,
        user_text: str,
        snapshot: RegistrySnapshot,
        *,
        explicit_capability: Optional[str] = None,
    ) -> RouteDecision:
        text = str(user_text or "").strip()
        if self._index_registry_version != snapshot.version:
            self.rebuild_index(snapshot)

        resolved = explicit_capability or self._resolve_explicit(text, snapshot)
        if resolved is not None:
            return self._decide_explicit(resolved, text, snapshot)

        return self._decide_hybrid(text, snapshot)

    # ---------- 显式路由 ----------

    def _resolve_explicit(self, text: str, snapshot: RegistrySnapshot) -> Optional[str]:
        """解析'使用 X mcp/skill'语法和别名包含；最长匹配，并列视为歧义。"""
        alias_map: Dict[str, List[str]] = {}
        for capability in snapshot.enabled_capabilities():
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
            if alias in lowered and len(alias) >= 2:
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
            return ""  # 有显式路由意图但没有已注册能力匹配 → fail closed（clarify）
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

    def _decide_hybrid(self, text: str, snapshot: RegistrySnapshot) -> RouteDecision:
        if not self._docs_by_id:
            return self._decision(
                route="answer_without_tools",
                intent="no_registered_capability",
                snapshot=snapshot,
                reason="注册表为空：没有可路由的外部能力",
                audit={"path": "hybrid"},
            )
        try:
            bm25_ranking = self._bm25.search(text, top_k=self.top_k)
            dense_ranking = self._dense.search(
                text, self.embedding_provider, top_k=self.top_k
            )
        except EmbeddingError as exc:
            return self._decision(
                route="unavailable",
                intent="retrieval_provider_error",
                snapshot=snapshot,
                reason=f"检索组件不可用: {exc}",
                audit={"path": "hybrid", "error": str(exc)[:200]},
            )
        fused = rrf_fuse([bm25_ranking, dense_ranking], top_k=self.top_k)
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
        ranked = self.reranker.rerank(text, candidates)
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
        operations = list(capability.operations)
        tool_ids: Tuple[str, ...] = tuple(op.name for op in operations if op.tool is not None)
        operation_ids = tuple(op.operation_id for op in operations)
        missing = sorted(
            {
                slot
                for op in operations
                for slot in _missing_required_slots(
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
            }
        )
        intent = capability.intents[0] if capability.intents else capability.capability_id
        memory_scopes = ("user_fact", "user_preference", "task_state")
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
