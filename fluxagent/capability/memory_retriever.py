# -*- coding: utf-8 -*-
"""Memory retrieval: hard boundaries first, lexical retrieval by default."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .retrieval import BM25Index, DenseIndex, EmbeddingError, EmbeddingProvider, Document, rrf_fuse


@dataclass(frozen=True)
class MemoryContext:
    """The bounded memory result passed from retrieval to prompt rendering."""

    records: tuple[Any, ...]
    query: str
    stats: dict[str, Any]


class _CountingEmbeddingProvider(EmbeddingProvider):
    """Count provider calls for one retrieval without changing the provider API."""

    def __init__(self, delegate: EmbeddingProvider) -> None:
        self.delegate = delegate
        self.calls = 0
        self.model_tag = getattr(delegate, "model_tag", "")

    def embed_texts(self, texts):
        self.calls += 1
        return self.delegate.embed_texts(texts)


class LegacyMemoryRetriever:
    """Pre-change selector used only for paired A/B evaluation."""

    def __init__(self, memory_store: Any):
        self.memory_store = memory_store
        self.last_stats: dict[str, Any] = {}

    def retrieve(self, **kwargs) -> list[Any]:
        started = time.perf_counter()
        records = self.memory_store.query(
            scopes=kwargs.get("scopes"),
            capability_ids=kwargs.get("capability_ids"),
            intent=kwargs.get("intent") or None,
            include_sensitive=kwargs.get("include_sensitive", False),
            session_id=kwargs.get("session_id"),
            now=kwargs.get("now"),
        )
        self.last_stats = {
            "candidate_count": len(records),
            "bm25_hits": None,
            "dense_used": False,
            "dense_error": "",
            "returned_count": len(records),
            "retrieval_latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "embedding_calls": 0,
        }
        return records


class MemoryRetriever:
    """选择小的、相关的记忆集合，而不改变权限。

    BM25总是第一步。只有当存在显式配置的embedding模型且词汇证据较弱时，才会尝试密集检索。
    减少embedding调用次数的主要目的是降低成本和延迟，而不是提高准确性。
    """

    def __init__(
        self,
        memory_store: Any,
        *,
        embedding_provider: Optional[EmbeddingProvider] = None,
        top_k: int = 5,  # 默认返回的最大记忆条目数
        dense_min_candidates: int = 8,  # 只有当候选记忆条目数大于等于该值时，才会尝试密集检索
        dense_min_lexical_hits: int = 2,  # 只有当BM25检索命中数小于该值时，才会尝试密集检索
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k 必须大于 0")
        if dense_min_candidates < 1 or dense_min_lexical_hits < 1:
            raise ValueError("dense 检索阈值必须大于 0")
        self.memory_store = memory_store
        self.embedding_provider = embedding_provider
        self.top_k = top_k
        self.dense_min_candidates = dense_min_candidates
        self.dense_min_lexical_hits = dense_min_lexical_hits
        self.last_stats: dict[str, Any] = {}

    def retrieve(
        self,
        *,
        query: str,
        scopes: Sequence[Any],
        capability_ids: Optional[Sequence[str]] = None,
        intent: str = "",
        include_sensitive: bool = False,
        session_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> list[Any]:
        started = time.perf_counter()
        embedding_calls = 0

        def finish_stats(**stats: Any) -> None:
            stats["retrieval_latency_ms"] = round(
                (time.perf_counter() - started) * 1000, 3
            )
            stats["embedding_calls"] = embedding_calls
            self.last_stats = stats

        if any(
            phrase in str(query or "")
            for phrase in ("不需要结合我的个人信息", "不要结合我的个人信息", "无需个人信息")
        ):
            finish_stats(
                candidate_count=0,
                bm25_hits=0,
                dense_used=False,
                dense_error="explicit_memory_exclusion",
                returned_count=0,
            )
            return []
        candidates = self.memory_store.query(
            scopes=scopes,
            capability_ids=capability_ids,
            intent=intent or None,
            include_sensitive=include_sensitive,
            session_id=session_id,
            now=now,
        )
        if not candidates:
            finish_stats(
                candidate_count=0,
                bm25_hits=0,
                dense_used=False,
                dense_error="",
                returned_count=0,
            )
            return []

        documents = [
            Document(
                doc_id=record.record_id,
                text=" ".join(
                    (
                        record.record_id,
                        record.value,
                        *record.capability_tags,
                        *record.intent_tags,
                    )
                ),
            )
            for record in candidates
        ]
        lexical = BM25Index()
        lexical.rebuild(documents)
        retrieval_query = " ".join(
            part for part in (query, intent) if str(part or "").strip()
        )
        lexical_ranking = lexical.search(
            retrieval_query,
            top_k=min(self.top_k, len(candidates)),
        )

        dense_used = False
        dense_error = ""
        rankings = [lexical_ranking]
        should_use_dense = (
            self.embedding_provider is not None
            and len(candidates) >= self.dense_min_candidates
            and len(lexical_ranking) < self.dense_min_lexical_hits
        )
        if should_use_dense:
            dense_used = True
            dense = DenseIndex()
            counting_provider = _CountingEmbeddingProvider(self.embedding_provider)
            try:
                dense.rebuild(
                    [(document.doc_id, document.text) for document in documents],
                    counting_provider,
                )
                rankings.append(
                    dense.search(
                        retrieval_query,
                        counting_provider,
                        top_k=min(self.top_k, len(candidates)),
                    )
                )
                embedding_calls = counting_provider.calls
            except EmbeddingError as exc:
                embedding_calls = counting_provider.calls
                dense_error = str(exc)[:200]
                dense_used = False

        if len(rankings) == 1:
            ranked_ids = [doc_id for doc_id, _score in lexical_ranking]
        else:
            ranked_ids = [doc_id for doc_id, _score in rrf_fuse(rankings, top_k=self.top_k)]

        # 没有检索命中时不能按置信度随便塞入记忆。通过安全边界不等于
        # 与当前请求相关，宁可不注入，也不能制造上下文噪声。

        by_id = {record.record_id: record for record in candidates}
        selected = [by_id[record_id] for record_id in ranked_ids if record_id in by_id]
        finish_stats(
            candidate_count=len(candidates),
            bm25_hits=len(lexical_ranking),
            dense_used=dense_used,
            dense_error=dense_error,
            returned_count=len(selected),
        )
        return selected[: self.top_k]

    def retrieve_context(self, **kwargs: Any) -> MemoryContext:
        records = tuple(self.retrieve(**kwargs))
        return MemoryContext(
            records=records,
            query=str(kwargs.get("query") or ""),
            stats=dict(self.last_stats),
        )
