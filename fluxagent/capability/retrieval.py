# -*- coding: utf-8 -*-
"""Hybrid 检索底层：BM25 全文索引 + dense 向量索引 + RRF 融合。

设计要点：
- 检索文档是能力卡/操作卡（Registry 输出），不是原始工具名；
- BM25 纯 Python 实现，可持久化为项目内 JSON；依赖少、确定性可测；
- EmbeddingProvider 是可替换接口：测试用确定性本地实现，运行时可配置
  OpenAI 兼容 embedding 服务；provider 不可用时上层必须 abstain，不允许
  降级成"全量工具开放"；
- 两路得分不可直接比较，统一用 Reciprocal Rank Fusion 合并。
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# ---------- 分词 ----------

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """语言无关分词：英文按 token，中文按 bigram（保留整词 + 原文片段）。"""
    lowered = str(text or "").casefold()
    tokens = _TOKEN_RE.findall(lowered)
    for run in _CJK_RE.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
            continue
        tokens.append(run)
        if len(run) == 2:
            # 2 字 run 的 bigram 与整词相同，重复计入会使该词 tf 翻倍
            continue
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


# ---------- BM25 ----------

@dataclass
class Document:
    doc_id: str
    text: str
    payload: Mapping = field(default_factory=dict)


class BM25Index:
    """Okapi BM25 的确定性实现，支持 JSON 持久化（项目内路径）。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._lock = threading.RLock()
        self._docs: Dict[str, Document] = {}
        self._postings: Dict[str, Dict[str, float]] = {}
        self._doc_len: Dict[str, float] = {}
        self._avg_len: float = 0.0
        self._header: Mapping = {}

    def rebuild(self, documents: Iterable[Document]) -> None:
        with self._lock:
            self._docs = {}
            self._postings = {}
            self._doc_len = {}
            for doc in documents:
                tokens = tokenize(doc.text)
                self._docs[doc.doc_id] = doc
                self._doc_len[doc.doc_id] = float(len(tokens))
                counts: Dict[str, float] = {}
                for token in tokens:
                    counts[token] = counts.get(token, 0.0) + 1.0
                for token, tf in counts.items():
                    self._postings.setdefault(token, {})[doc.doc_id] = tf
            total = sum(self._doc_len.values())
            self._avg_len = total / len(self._doc_len) if self._doc_len else 0.0

    def search(self, query: str, top_k: int = 8) -> List[Tuple[str, float]]:
        if top_k < 1:
            raise ValueError("top_k 必须大于 0")
        with self._lock:
            if not self._docs:
                return []
            n_docs = len(self._docs)
            query_tokens = tokenize(query)
            scores: Dict[str, float] = {}
            for token in query_tokens:
                postings = self._postings.get(token)
                if not postings:
                    continue
                idf = math.log(1.0 + (n_docs - len(postings) + 0.5) / (len(postings) + 0.5))
                for doc_id, tf in postings.items():
                    dl = self._doc_len.get(doc_id, 0.0) or 1.0
                    norm = tf * (self.k1 + 1.0) / (
                        tf + self.k1 * (1.0 - self.b + self.b * dl / (self._avg_len or 1.0))
                    )
                    scores[doc_id] = scores.get(doc_id, 0.0) + idf * norm
            ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
            return ranked[:top_k]

    # ---------- 持久化 ----------

    def save(self, path: str, header: Optional[Mapping] = None) -> None:
        with self._lock:
            payload = {
                "k1": self.k1,
                "b": self.b,
                "header": dict(header or {}),
                "docs": [
                    {"doc_id": d.doc_id, "text": d.text, "payload": dict(d.payload)}
                    for d in self._docs.values()
                ],
            }
        _atomic_write_json(path, payload)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        index = cls(k1=payload.get("k1", 1.5), b=payload.get("b", 0.75))
        index._header = dict(payload.get("header", {}))
        index.rebuild(
            Document(d["doc_id"], d["text"], d.get("payload", {})) for d in payload["docs"]
        )
        return index


# ---------- Embedding ----------

class EmbeddingError(RuntimeError):
    """embedding provider 不可用/失败；调用方必须 abstain，不能降级全量。"""


class EmbeddingProvider:
    """可替换的向量提供接口。"""

    model_tag: str = "abstract"

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError


class DeterministicEmbedding(EmbeddingProvider):
    """确定性哈希 embedding（测试/离线默认）：不依赖网络，结果可复现。"""

    def __init__(self, dim: int = 256) -> None:
        if dim < 16:
            raise ValueError("dim 至少为 16")
        self.dim = dim
        self.model_tag = f"deterministic-hash-{dim}"

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            tokens = tokenize(text)
            if not tokens:
                vectors.append(vec)
                continue
            for token in tokens:
                digest = int.from_bytes(
                    __import__("hashlib").sha256(token.encode("utf-8")).digest()[:8],
                    "little",
                )
                idx = digest % self.dim
                sign = 1.0 if (digest >> 63) & 1 == 0 else -1.0
                vec[idx] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class OpenAICompatibleEmbedding(EmbeddingProvider):
    """OpenAI 兼容 embedding 适配器；base_url/key/model 全部由调用方注入。"""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 30.0) -> None:
        if not base_url or not model:
            raise EmbeddingError("embedding base_url/model 必须配置")
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.model = model
        self.model_tag = f"openai-compatible-{model}"
        self.timeout = timeout

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        from openai import OpenAI

        client = OpenAI(
            api_key=self._api_key, base_url=self.base_url, timeout=self.timeout
        )
        try:
            response = client.embeddings.create(
                model=self.model, input=list(texts)
            )
        except Exception as exc:  # noqa: BLE001 - 上层据 EmbeddingError abstain
            raise EmbeddingError(f"embedding 调用失败: {exc}") from exc
        return [item.embedding for item in response.data]


class DenseIndex:
    """余弦相似度向量索引；持久化为 JSON（模型 tag 一并记录）。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._ids: List[str] = []
        self._payloads: Dict[str, Mapping] = {}
        self._vectors: Dict[str, List[float]] = {}
        self.model_tag: str = ""
        self._header: Mapping = {}

    def rebuild(
        self, items: Sequence[Tuple[str, Mapping]], provider: EmbeddingProvider
    ) -> None:
        if not items:
            with self._lock:
                self._ids, self._payloads, self._vectors, self.model_tag = [], {}, [], provider.model_tag
            return
        vectors = provider.embed_texts([text for _, text in items])
        with self._lock:
            self.model_tag = provider.model_tag
            self._ids = [doc_id for doc_id, _ in items]
            self._payloads = {doc_id: payload for doc_id, payload in items}
            self._vectors = {doc_id: vec for (doc_id, _), vec in zip(items, vectors)}

    def search(self, query: str, provider: EmbeddingProvider, top_k: int = 8) -> List[Tuple[str, float]]:
        if top_k < 1:
            raise ValueError("top_k 必须大于 0")
        with self._lock:
            if not self._vectors:
                return []
            query_vec = provider.embed_texts([query])[0]
            scored: List[Tuple[str, float]] = []
            for doc_id, vec in self._vectors.items():
                scored.append((doc_id, _cosine(query_vec, vec)))
            scored.sort(key=lambda kv: (-kv[1], kv[0]))
            return scored[:top_k]

    def save(self, path: str, header: Optional[Mapping] = None) -> None:
        with self._lock:
            payload = {
                "model_tag": self.model_tag,
                "header": dict(header or {}),
                "items": [
                    {
                        "doc_id": doc_id,
                        "payload": str(self._payloads.get(doc_id, "")),
                        "vector": self._vectors[doc_id],
                    }
                    for doc_id in self._ids
                ],
            }
        _atomic_write_json(path, payload)

    @classmethod
    def load(cls, path: str) -> "DenseIndex":
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        index = cls()
        index.model_tag = payload.get("model_tag", "")
        index._header = dict(payload.get("header", {}))
        index._ids = [item["doc_id"] for item in payload["items"]]
        index._payloads = {item["doc_id"]: str(item.get("payload", "")) for item in payload["items"]}
        index._vectors = {item["doc_id"]: list(item["vector"]) for item in payload["items"]}
        return index


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _atomic_write_json(path: str, payload: Mapping) -> None:
    """临时文件 + os.replace 原子写，避免多进程读到半截索引。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    os.replace(temp_path, path)


# ---------- 融合 ----------

def rrf_fuse(
    rankings: Sequence[Sequence[Tuple[str, float]]],
    top_k: int = 8,
    k: int = 60,
) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion：输入是各路 (doc_id, 原始分) 排名，输出融合排名。

    原始分不参与相加——不同检索器的分数量纲不可比，RRF 只用名次。
    """
    if top_k < 1:
        raise ValueError("top_k 必须大于 0")
    fused: Dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1.0)
    ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:top_k]


def load_embedding_provider(settings: Optional[Mapping] = None) -> EmbeddingProvider:
    """按配置选择 embedding provider；未配置时回落确定性实现（显式标注 tag）。"""
    if not settings:
        return DeterministicEmbedding()
    base_url = str(settings.get("base_url") or "").strip()
    model = str(settings.get("model") or "").strip()
    if base_url and model:
        return OpenAICompatibleEmbedding(
            base_url=base_url,
            api_key=str(settings.get("api_key") or ""),
            model=model,
            timeout=float(settings.get("timeout", 30.0)),
        )
    return DeterministicEmbedding(dim=int(settings.get("dim", 256)))
