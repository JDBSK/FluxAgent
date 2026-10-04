# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

from fluxagent.capability.context_compiler import MemoryScope, MemoryStore
from fluxagent.capability.memory_retriever import MemoryContext, MemoryRetriever
from fluxagent.capability.retrieval import EmbeddingError, EmbeddingProvider


class CountingEmbedding(EmbeddingProvider):
    model_tag = "test-embedding"

    def __init__(self, fail: bool = False):
        self.calls = 0
        self.fail = fail

    def embed_texts(self, texts):
        self.calls += 1
        if self.fail:
            raise EmbeddingError("test embedding unavailable")
        return [[float(len(str(text)) + index), 1.0] for index, text in enumerate(texts)]


class TestMemoryRetriever(unittest.TestCase):
    def _store(self, count: int = 2) -> MemoryStore:
        store = MemoryStore()
        for index in range(count):
            store.add(
                MemoryScope.USER_FACT,
                f"用户事实 {index}：喜欢阅读第 {index} 类资料",
                key=f"profile.fact_{index}",
            )
        return store

    def test_small_memory_does_not_call_embedding(self):
        provider = CountingEmbedding()
        retriever = MemoryRetriever(self._store(), embedding_provider=provider)
        records = retriever.retrieve(
            query="没有关键词的请求",
            scopes=(MemoryScope.USER_FACT,),
        )
        self.assertEqual(len(records), 0)
        self.assertEqual(provider.calls, 0)
        self.assertFalse(retriever.last_stats["dense_used"])

    def test_configured_embedding_is_used_only_when_lexical_recall_is_weak(self):
        provider = CountingEmbedding()
        retriever = MemoryRetriever(
            self._store(8), embedding_provider=provider, dense_min_candidates=8
        )
        retriever.retrieve(query="完全不同的语义", scopes=(MemoryScope.USER_FACT,))
        self.assertGreater(provider.calls, 0)
        self.assertTrue(retriever.last_stats["dense_used"])

    def test_embedding_failure_falls_back_to_bounded_results(self):
        retriever = MemoryRetriever(
            self._store(8),
            embedding_provider=CountingEmbedding(fail=True),
            dense_min_candidates=8,
        )
        records = retriever.retrieve(
            query="完全不同的语义", scopes=(MemoryScope.USER_FACT,)
        )
        self.assertLessEqual(len(records), 5)
        self.assertTrue(retriever.last_stats["dense_error"])

    def test_retrieve_context_freezes_records_and_stats(self):
        retriever = MemoryRetriever(self._store())
        context = retriever.retrieve_context(
            query="阅读资料", scopes=(MemoryScope.USER_FACT,)
        )
        self.assertIsInstance(context, MemoryContext)
        self.assertIsInstance(context.records, tuple)
        self.assertEqual(context.stats["returned_count"], len(context.records))

    def test_semantic_key_is_searchable_for_delete_or_update(self):
        store = MemoryStore()
        store.add(
            MemoryScope.USER_PREFERENCE,
            "通勤方式偏好：旧记录",
            key="profile.commute.preference",
        )
        retriever = MemoryRetriever(store)
        records = retriever.retrieve(
            query="commute.preference",
            scopes=(MemoryScope.USER_PREFERENCE,),
        )
        self.assertEqual(
            [record.record_id for record in records],
            ["user_preference:profile.commute.preference"],
        )


if __name__ == "__main__":
    unittest.main()
