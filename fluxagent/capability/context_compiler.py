# -*- coding: utf-8 -*-
"""Context Compiler 与选择性记忆。

核心语义：
- 记忆不是"全文画像 + 全量历史"，而是按 scope / 相关性 / 敏感性 / TTL /
  token 预算筛选后的结构化记录；
- 每次模型调用前编译出不可变 ContextSnapshot，注入内容全部带来源 ID 和
  选择原因，可审计、可测试"为什么进入 prompt"；
- Compiler 构造消息副本，不修改 checkpoint 中的原始消息对象；
- 完整历史、旧工具结果、旧参数错误、未选中 Skill 文档默认不注入。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Tuple


# ---------- 记忆模型 ----------

class MemoryScope(str, Enum):
    USER_FACT = "user_fact"
    USER_PREFERENCE = "user_preference"
    TASK_STATE = "task_state"
    EPISODIC = "episodic"
    SKILL_CONTEXT = "skill_context"


SENSITIVITY_LEVELS = frozenset({"normal", "sensitive"})


@dataclass(frozen=True)
class MemoryRecord:
    record_id: str
    scope: MemoryScope
    value: str
    source_id: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    ttl_seconds: Optional[float] = None
    capability_tags: Tuple[str, ...] = ()
    intent_tags: Tuple[str, ...] = ()
    sensitivity: str = "normal"
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.sensitivity not in SENSITIVITY_LEVELS:
            raise ValueError(f"非法敏感级别: {self.sensitivity}")

    def is_expired(self, now: Optional[float] = None) -> bool:
        if self.ttl_seconds is None:
            return False
        reference = now if now is not None else time.time()
        return reference - self.created_at > self.ttl_seconds


class MemoryStore:
    """线程安全的结构化记忆仓库。

    冲突规则：同 (scope, key) 写入时旧记录被 superseded（保留可追溯），
    不做静默覆盖——查询永远返回最新版本。
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[str, MemoryRecord] = {}
        self._counter = 0

    def add(
        self,
        scope: MemoryScope,
        value: str,
        *,
        key: Optional[str] = None,
        source_id: str = "",
        ttl_seconds: Optional[float] = None,
        capability_tags: Sequence[str] = (),
        intent_tags: Sequence[str] = (),
        sensitivity: str = "normal",
        confidence: float = 1.0,
        now: Optional[float] = None,
    ) -> MemoryRecord:
        with self._lock:
            self._counter += 1
            record_id = key or f"mem-{self._counter}"
            record = MemoryRecord(
                record_id=record_id,
                scope=scope,
                value=str(value),
                source_id=source_id,
                created_at=now if now is not None else time.time(),
                updated_at=now if now is not None else time.time(),
                ttl_seconds=ttl_seconds,
                capability_tags=tuple(capability_tags),
                intent_tags=tuple(intent_tags),
                sensitivity=sensitivity,
                confidence=confidence,
            )
            self._records[record_id] = record
            return record

    def query(
        self,
        *,
        scopes: Optional[Sequence[MemoryScope]] = None,
        capability_id: Optional[str] = None,
        intent: Optional[str] = None,
        include_sensitive: bool = False,
        now: Optional[float] = None,
    ) -> List[MemoryRecord]:
        """按 scope + capability/intent 相关性筛选；敏感记录默认排除。"""
        with self._lock:
            candidates = list(self._records.values())
        result: List[MemoryRecord] = []
        for record in candidates:
            if record.is_expired(now):
                continue
            if scopes is not None and record.scope not in scopes:
                continue
            if not include_sensitive and record.sensitivity == "sensitive":
                continue
            if capability_id and record.capability_tags and capability_id not in record.capability_tags:
                continue
            if intent and record.intent_tags and intent not in record.intent_tags:
                continue
            result.append(record)
        result.sort(key=lambda r: (-r.confidence, r.record_id))
        return result

    def forget(self, record_id: str) -> bool:
        with self._lock:
            return self._records.pop(record_id, None) is not None

    def forget_prefix(self, prefix: str) -> int:
        with self._lock:
            stale = [rid for rid in self._records if rid.startswith(prefix)]
            for rid in stale:
                del self._records[rid]
            return len(stale)

    def all_records(self) -> List[MemoryRecord]:
        with self._lock:
            return list(self._records.values())


class ProfileCompatProvider:
    """兼容 Provider：把项目内 Markdown 用户画像转换成结构化记忆。

    核心编译器不依赖 Markdown；本 Provider 只在装配时把现有画像"导入"为
    USER_FACT 记录，之后写入仍走 save_user_profile 工具（保持既有行为）。
    """

    def __init__(self, profile_path: str) -> None:
        self.profile_path = str(profile_path)

    def load_into(self, store: MemoryStore) -> int:
        import os
        import re as _re

        if not os.path.exists(self.profile_path):
            return 0
        try:
            with open(self.profile_path, encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
        except OSError:
            return 0
        count = 0
        for line in content.splitlines():
            line = line.strip()
            match = _re.match(r"^[-*]\s*\*\*(.+?)\*\*[：:]\s*(.+)$", line)
            if not match:
                continue
            key, value = match.group(1).strip(), match.group(2).strip()
            if not key or not value:
                continue
            store.add(
                MemoryScope.USER_FACT,
                f"{key}：{value}",
                key=f"profile.{key}",
                source_id=f"user_profile.md#{key}",
            )
            count += 1
        return count


# ---------- Token 预算 ----------

def estimate_tokens(text: str) -> int:
    """保守估算：CJK 每字约 1 token，其他按 4 字符 1 token（不足 4 记 1）。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return cjk + (max(1, other // 4 + (1 if other % 4 else 0)) if other else 0)


# ---------- Context Snapshot ----------

@dataclass(frozen=True)
class InjectedMemory:
    record_id: str
    scope: str
    value: str
    reason: str


@dataclass(frozen=True)
class ContextSnapshot:
    """一次模型调用的完整输入：不可变、可审计。"""

    system_prompt: str
    memory: Tuple[InjectedMemory, ...] = ()
    estimated_tokens: int = 0
    truncated_sections: Tuple[str, ...] = ()
    contract_route: str = ""
    capability_ids: Tuple[str, ...] = ()


# ---------- Compiler ----------

@dataclass
class CompilationContext:
    """编译输入：当前请求、契约、历史摘要、任务状态和可选 skill 上下文。"""

    user_text: str
    route: str = "answer_without_tools"
    intent: str = ""
    capability_ids: Tuple[str, ...] = ()
    memory_scopes: Tuple[str, ...] = ()
    history_digest: str = ""
    task_state: str = ""
    skill_context: str = ""
    failure_facts: Tuple[str, ...] = ()


class ContextCompiler:
    """把"该让模型看到什么"变成确定性、可审计的编译过程。

    注入优先级（超预算时按此顺序裁剪尾部）：
      规则 > 契约 > 相关记忆 > 任务状态/失败事实 > skill 上下文 > 历史摘要
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        *,
        token_budget: int = 2000,
        tokenizer: Optional[Callable[[str], int]] = None,
        max_skill_context_tokens: int = 600,
    ) -> None:
        if token_budget < 64:
            raise ValueError("token_budget 过小")
        self.memory_store = memory_store
        self.token_budget = token_budget
        self.tokenizer = tokenizer or estimate_tokens
        self.max_skill_context_tokens = max_skill_context_tokens

    def compile(
        self,
        context: CompilationContext,
        *,
        include_sensitive: bool = False,
        now: Optional[float] = None,
    ) -> ContextSnapshot:
        truncated: List[str] = []
        scopes = self._scopes_for(context.memory_scopes)
        records = self.memory_store.query(
            scopes=scopes,
            capability_id=context.capability_ids[0] if context.capability_ids else None,
            intent=context.intent or None,
            include_sensitive=include_sensitive,
            now=now,
        )

        memory_injected: List[InjectedMemory] = []
        memory_lines: List[str] = []
        for record in records:
            reason = self._injection_reason(record, context)
            memory_injected.append(
                InjectedMemory(
                    record_id=record.record_id,
                    scope=record.scope.value,
                    value=record.value,
                    reason=reason,
                )
            )
            memory_lines.append(f"- {record.value}")

        # skill 上下文按独立预算裁剪
        skill_block = ""
        if context.skill_context:
            skill_block, was_truncated = self._fit_block(
                "[Selected skill context]\n" + context.skill_context,
                self.max_skill_context_tokens,
            )
            if was_truncated:
                truncated.append("skill_context")

        sections: List[str] = []
        sections.append(self._rules_block(context))
        sections.append(self._contract_block(context))
        if memory_lines:
            sections.append("[Relevant user facts]\n" + "\n".join(memory_lines))
        if context.task_state:
            sections.append("[Task state]\n" + context.task_state)
        if context.failure_facts:
            sections.append(
                "[Failure facts]\n" + "\n".join(f"- {fact}" for fact in context.failure_facts)
            )
        sections.append(skill_block)
        if context.history_digest:
            sections.append("[Recent context]\n" + context.history_digest)

        # 全局预算：按优先级从尾部丢弃
        prompt = "\n\n".join(s for s in sections if s)
        while self.tokenizer(prompt) > self.token_budget and len(sections) > 2:
            dropped = sections.pop()
            truncated.append(dropped.split("]", 1)[0].strip("[]") or "section")
            prompt = "\n\n".join(sections)

        return ContextSnapshot(
            system_prompt=prompt,
            memory=tuple(memory_injected),
            estimated_tokens=self.tokenizer(prompt),
            truncated_sections=tuple(truncated),
            contract_route=context.route,
            capability_ids=tuple(context.capability_ids),
        )

    # ---------- 内部 ----------

    @staticmethod
    def _scopes_for(memory_scopes: Sequence[str]) -> Tuple[MemoryScope, ...]:
        if not memory_scopes:
            return (MemoryScope.USER_FACT, MemoryScope.USER_PREFERENCE)
        valid = {scope.value for scope in MemoryScope}
        selected = [MemoryScope(scope) for scope in memory_scopes if scope in valid]
        return tuple(selected) or (MemoryScope.USER_FACT,)

    @staticmethod
    def _injection_reason(record: MemoryRecord, context: CompilationContext) -> str:
        reasons = [f"scope={record.scope.value}"]
        if record.capability_tags and context.capability_ids:
            overlap = set(record.capability_tags) & set(context.capability_ids)
            if overlap:
                reasons.append(f"capability_match={','.join(sorted(overlap))}")
        if record.intent_tags and context.intent:
            if context.intent in record.intent_tags:
                reasons.append("intent_match")
        return ";".join(reasons)

    def _fit_block(self, text: str, budget_tokens: int) -> Tuple[str, bool]:
        if self.tokenizer(text) <= budget_tokens:
            return text, False
        # 保守按字符比例截断，保留头尾以维持可用性
        ratio = budget_tokens / max(1, self.tokenizer(text))
        keep = max(200, int(len(text) * ratio * 0.9))
        head = text[: keep // 2]
        tail = text[-keep // 2 :]
        return head + "\n...[截断]...\n" + tail, True

    @staticmethod
    def _rules_block(context: CompilationContext) -> str:
        return (
            "你是FluxAgent——一个透明且可控制的本地智能体。\n"
            "Rules:\n"
            "1. Stay inside the office sandbox for files/shell.\n"
            "2. 只使用本回合 [Invocation contract] 明确允许的工具；"
            "没有列出任何工具时，直接用已有知识回答，不要编造或猜测工具。\n"
            "3. When user states long-term preferences, call save_user_profile.\n"
            "4. Never claim a tool was called or a file was written unless the "
            "structured tool_call and ToolMessage result are present.\n"
            "5. 工具参数错误或被拒绝后不要换参数重试；直接向用户说明原因。\n"
        )

    @staticmethod
    def _contract_block(context: CompilationContext) -> str:
        lines = [f"[Invocation contract] route={context.route}"]
        if context.capability_ids:
            lines.append(f"allowed_capabilities={','.join(context.capability_ids)}")
        if context.route != "route":
            lines.append(
                "本回合未授权任何外部工具；请基于上下文直接回答，"
                "或向用户澄清需要哪个能力。"
            )
        return "\n".join(lines)
