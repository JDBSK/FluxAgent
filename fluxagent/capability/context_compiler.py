# -*- coding: utf-8 -*-
"""Context Compiler 与选择性记忆。

核心语义：
- 记忆不是"全文画像 + 全量历史"，而是按 scope / 相关性 / 敏感性 /TTL/token 预算筛选后的结构化记录；  
- 每次模型调用前编译出不可变 ContextSnapshot,注入内容全部带来源 ID 和选择原因，可审计、可测试"为什么进入 prompt";
- Compiler 构造消息副本，不修改 checkpoint 中的原始消息对象；
- 完整历史、旧工具结果、旧参数错误、未选中 Skill 文档默认不注入。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Sequence, Tuple


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
    """一条记忆的数据结构。不可变，便于快照和审计。"""
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
    # 会话归属：空串 = 全局记录（画像/通用偏好）；非空 = 仅本会话可见。
    # 这是跨会话数据边界的硬边界：会话 A 的任务态/失败经验不得进入会话 B。
    session_id: str = ""

    def __post_init__(self) -> None:
        if self.sensitivity not in SENSITIVITY_LEVELS:
            raise ValueError(f"非法敏感级别: {self.sensitivity}")

    def is_expired(self, now: Optional[float] = None) -> bool:
        if self.ttl_seconds is None:
            return False
        reference = now if now is not None else time.time()
        return reference - self.created_at > self.ttl_seconds


# 记忆持久化文件格式版本：序列化字段或语义变化时递增，旧文件整份拒绝
# （fail-open 起步为空记忆，不会把旧结构误读进新代码）。
MEMORY_FILE_FORMAT = "1"


class MemoryStore:
    """线程安全的结构化记忆仓库。保存、查询、删除记忆

    冲突规则：同 (scope, key) 写入时旧记录被 superseded(保留可追溯),
    不做静默覆盖——查询永远返回最新版本。

    持久化：构造时传入 persist_path 后,所有变更(add/forget/forget_prefix)
    自动全量快照落盘（原子写），重启后 load() 恢复。未传 persist_path 时
    行为与纯内存仓库完全一致（测试与子 Agent 场景）。
    """

    def __init__(self, persist_path: Optional[str] = None) -> None:
        self._lock = threading.RLock()
        self._records: dict[str, MemoryRecord] = {}
        self._counter = 0
        self.persist_path = str(persist_path) if persist_path else ""
        self.last_persist_error: str = ""

    def _auto_save(self) -> None:
        """变更后落盘。失败不抛出(记忆写失败不阻断任务流),错误留存在last_persist_error 供审计/测试检查。"""
        if not self.persist_path:
            return
        try:
            self.save(self.persist_path)
            self.last_persist_error = ""
        except OSError as exc:
            self.last_persist_error = f"{type(exc).__name__}: {exc}"[:200]

    def save(self, path: str) -> None:
        """全量快照落盘（临时文件 + os.replace 原子写）。

        敏感记录也写入：文件位于本地 workspace(与画像 markdown 同级），
        注入侧仍由 query 的敏感过滤把关，落盘不改变可见性语义。
        """
        import json

        from .retrieval import _atomic_write_json

        with self._lock:
            payload = {
                "format": MEMORY_FILE_FORMAT,
                "counter": self._counter,
                "records": [
                    {
                        "record_id": r.record_id,
                        "scope": r.scope.value,
                        "value": r.value,
                        "source_id": r.source_id,
                        "created_at": r.created_at,
                        "updated_at": r.updated_at,
                        "ttl_seconds": r.ttl_seconds,
                        "capability_tags": list(r.capability_tags),
                        "intent_tags": list(r.intent_tags),
                        "sensitivity": r.sensitivity,
                        "confidence": r.confidence,
                        "session_id": r.session_id,
                    }
                    for r in self._records.values()
                ],
            }
        _atomic_write_json(path, payload)

    def load(self, path: str) -> int:
        """启动恢复。文件缺失/损坏/格式不符返回 0(fail-open:记忆是可用性
        数据而非安全边界，坏文件不阻断启动）；单条坏记录跳过不拖垮整体。
        过期记录在加载时丢弃(不复活);counter 恢复高水位防 record_id 碰撞。
        """
        import json

        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, ValueError):
            return 0
        if payload.get("format") != MEMORY_FILE_FORMAT:
            return 0
        raw_records = payload.get("records")
        if not isinstance(raw_records, list):
            return 0
        now = time.time()
        loaded = 0
        max_counter = 0
        with self._lock:
            for item in raw_records:
                if not isinstance(item, dict):
                    continue
                try:
                    record = MemoryRecord(
                        record_id=str(item["record_id"]),
                        scope=MemoryScope(item["scope"]),
                        value=str(item.get("value", "")),
                        source_id=str(item.get("source_id", "")),
                        created_at=float(item.get("created_at", 0.0)),
                        updated_at=float(item.get("updated_at", 0.0)),
                        ttl_seconds=(
                            float(item["ttl_seconds"])
                            if item.get("ttl_seconds") is not None
                            else None
                        ),
                        capability_tags=tuple(
                            str(t) for t in item.get("capability_tags", ())
                        ),
                        intent_tags=tuple(
                            str(t) for t in item.get("intent_tags", ())
                        ),
                        sensitivity=str(item.get("sensitivity", "normal")),
                        confidence=float(item.get("confidence", 1.0)),
                        session_id=str(item.get("session_id", "")),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                if record.is_expired(now):
                    continue
                self._records[record.record_id] = record
                loaded += 1
                if record.record_id.startswith("mem-"):
                    try:
                        max_counter = max(max_counter, int(record.record_id[4:]))
                    except ValueError:
                        continue
            self._counter = max(self._counter, max_counter)
        return loaded

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
        session_id: str = "",
        now: Optional[float] = None,
    ) -> MemoryRecord:
        
        if "@@" in str(key or "") or "@@" in str(session_id or ""):
            # fail-closed：@@ 是会话命名空间分隔符，出现即拒绝，
            # 防止 (key="k@@a", session="b") 与 (key="k", session="a@@b") 碰撞
            raise ValueError('memory key/session_id 不允许包含 "@@"')
        
        with self._lock:
            self._counter += 1
            # record_id 含 scope 前缀 + 会话命名空间：同 key 跨 scope 不互相覆盖，
            # 不同会话写同名 key 不互相覆盖
            if key and session_id:
                record_id = f"{scope.value}:{key}@@{session_id}"
            elif key:
                record_id = f"{scope.value}:{key}"
            else:
                record_id = f"mem-{self._counter}"
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
                session_id=str(session_id or ""),
            )
            self._records[record_id] = record
        self._auto_save()
        return record

    def query(
        self,
        *,
        scopes: Optional[Sequence[MemoryScope]] = None,
        capability_id: Optional[str] = None,
        capability_ids: Optional[Sequence[str]] = None,
        intent: Optional[str] = None,
        include_sensitive: bool = False,
        session_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> List[MemoryRecord]:
        """按 scope + capability/intent 相关性 + 会话边界筛选；敏感记录默认排除。

        会话边界(fail-closed):
        - 全局记录(session_id 为空，如画像/通用偏好）任何会话可见；
        - 会话级记录只在 session_id 完全一致时可见；调用方未提供会话上下文
          (session_id=None)时会话级记录一律排除,防止跨会话泄漏。
        """
        with self._lock:
            candidates = list(self._records.values())
            # 惰性驱逐：过期记录在查询时清理，防止长驻进程内存无界增长。
            # 锁内重验再删（防 TOCTOU 误删并发覆盖写的新记录）；
            # 仅墙钟驱动（now=None）——传入历史时间戳的回放不得破坏全局状态
            if now is None:
                expired_ids = {
                    record.record_id
                    for record in candidates
                    if record.is_expired(None)
                }
                for record_id in expired_ids:
                    record = self._records.get(record_id)
                    if record is not None and record.is_expired(None):
                        del self._records[record_id]
            else:
                expired_ids = {
                    record.record_id for record in candidates if record.is_expired(now)
                }

        result: List[MemoryRecord] = []
        for record in candidates:
            if record.record_id in expired_ids:
                continue
            if scopes is not None and record.scope not in scopes:
                continue
            if not include_sensitive and record.sensitivity == "sensitive":
                continue
            if record.session_id:
                if session_id is None or record.session_id != session_id:
                    continue
            if record.capability_tags:
                # 带能力标签的记录只进对应能力（多能力回合按交集判断）；
                # abstain 回合（无能力）一律排除，
                # 防止其他任务的工具上下文泄漏进通用问答
                allowed = set(capability_ids or ()) or (
                    {capability_id} if capability_id else set()
                )
                if not allowed or not (allowed & set(record.capability_tags)):
                    continue
            if intent and record.intent_tags and intent not in record.intent_tags:
                continue
            result.append(record)
        result.sort(key=lambda r: (-r.confidence, r.record_id))
        return result

    def forget(self, record_id: str) -> bool:
        with self._lock:
            removed = self._records.pop(record_id, None) is not None
        if removed:
            self._auto_save()
        return removed

    def forget_prefix(self, prefix: str) -> int:
        """删除所有 record_id 以 prefix 开头的记录，返回删除数量.
        作用是清理同一类记忆（如某个能力的全部记忆）或清理会话级记忆。"""
        with self._lock:
            stale = [rid for rid in self._records if rid.startswith(prefix)]
            for rid in stale:
                del self._records[rid]
            removed = len(stale)
        if removed:
            self._auto_save()
        return removed

    def all_records(self) -> List[MemoryRecord]:
        with self._lock:
            return list(self._records.values())


class ProfileCompatProvider:
    """兼容 Provider:把项目内 Markdown 用户画像转换成结构化记忆。

    核心编译器不依赖 Markdown;本 Provider 只在装配时把现有画像"导入"为USER_FACT 记录,
    之后写入仍走 save_user_profile 工具（保持既有行为）。
    """

    def __init__(self, profile_path: str) -> None:
        self.profile_path = str(profile_path)

    def load_into(self, store: MemoryStore) -> int:
        """把 Markdown 用户画像导入 MemoryStore,返回导入的记录数。"""
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
                f"{key}:{value}",
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
    # 会话上下文：为空时编译器排除所有会话级记忆（fail-closed 数据边界）
    session_id: str = ""
    # 本回合实际绑定的工具名（渲染进契约块，避免提示与机制不一致）
    allowed_tool_names: Tuple[str, ...] = ()
    skill_is_context_only: bool = False


class ContextCompiler:
    """决定本轮哪些记忆能进入模型上下文

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
        max_memory_value_chars: int = 200,
    ) -> None:
        if token_budget < 64:
            raise ValueError("token_budget 过小")
        if max_skill_context_tokens < 64:
            raise ValueError("max_skill_context_tokens 过小")
        self.memory_store = memory_store
        self.token_budget = token_budget
        self.tokenizer = tokenizer or estimate_tokens
        self.max_skill_context_tokens = max_skill_context_tokens
        self.max_memory_value_chars = max_memory_value_chars

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
            capability_ids=context.capability_ids or None,
            intent=context.intent or None,
            include_sensitive=include_sensitive,
            session_id=context.session_id if context.session_id else None,
            now=now,
        )

        memory_injected: List[InjectedMemory] = []
        memory_lines: List[str] = []
        for record in records:
            reason = self._injection_reason(record, context)
            # 注入侧截断（防御纵深）：不信任写入方的值长度，
            # 超长内容（含可能的注入文本）在这里被截为数据摘要
            value = (
                record.value[: self.max_memory_value_chars] + "…[截断]"
                if len(record.value) > self.max_memory_value_chars
                else record.value
            )
            memory_injected.append(
                InjectedMemory(
                    record_id=record.record_id,
                    scope=record.scope.value,
                    value=value,
                    reason=reason,
                )
            )
            memory_lines.append(f"- {value}")

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
        """生成内存注入的原因说明。"""
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
        keep = max(64, min(int(len(text) * ratio * 0.9), len(text)))
        head = text[: keep // 2]
        tail = text[-keep // 2 :]
        return head + "\n...[截断]...\n" + tail, True

    @staticmethod
    def _rules_block(context: CompilationContext) -> str:
        return (
            "你是FluxAgent——一个透明且可控制的本地智能体。\n"
            "Rules:\n"
            "1. 对于文件/Shell 操作，请始终在办公沙箱内执行。\n"
            "2. 只使用本回合 [Invocation contract] 明确允许的工具；没有列出任何工具时，直接用已有知识回答，不要编造或猜测工具.\n"
            "3. 当用户指定长期偏好时，调用 `save_user_profile` 函数。\n"
            "4. 除非存在结构化的 `tool_call` 和 `ToolMessage` 结果，否则切勿声称已调用某工具或已写入某文件。\n"
            "5. 工具参数错误或被拒绝后不要换参数重试；直接向用户说明原因。\n"
        )

    @staticmethod
    def _contract_block(context: CompilationContext) -> str:
        lines = [f"[Invocation contract] route={context.route}"]
        if context.capability_ids:
            lines.append(f"allowed_capabilities={','.join(context.capability_ids)}")
        if context.allowed_tool_names:
            shown = ", ".join(context.allowed_tool_names[:12])
            more = " ..." if len(context.allowed_tool_names) > 12 else ""
            lines.append(f"allowed_tools={shown}{more}")
        if context.skill_is_context_only:
            lines.append(
                "选中的 Skill 以本提示中的上下文注入，不产生任何工具调用；"
                "请基于其内容直接回答。"
            )
        if context.route != "route":
            lines.append(
                "本回合未授权任何外部能力工具（内置本地工具仍可用）；"
                "请基于上下文直接回答，或向用户澄清需要哪个能力。"
            )
        return "\n".join(lines)
