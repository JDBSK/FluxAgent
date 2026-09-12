# -*- coding: utf-8 -*-
"""help→run 一次性凭据门禁回归测试。

契约：run 必须携带本会话 help 签发的一次性 help_token；凭据绑定
会话 + 技能 + 说明版本（SHA-256），300 秒有效、锁内单次消费。
任何校验失败，执行器（execute_office_shell）都不会被调用。
"""
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from fluxagent.core.skill_loader import LazySkillLoader

CFG_A = {"configurable": {"thread_id": "session-a"}}
CFG_B = {"configurable": {"thread_id": "session-b"}}


class FakeClock:
    """受控单调时钟：TTL 边界测试不真的等待 300 秒。"""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now


def _write_script_skill(root: Path, name: str, body: str = "print('ok')\n") -> Path:
    skill = root / name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: gate fixture\n---\n\nManual.\n",
        encoding="utf-8",
    )
    (skill / "main.py").write_text(body, encoding="utf-8")
    return skill


def _make_loader(root: Path, shell: Mock) -> tuple:
    if not any(root.iterdir()):
        # 未预置夹具的用例默认获得一个带脚本的 demo 技能
        _write_script_skill(root, "demo")
    patcher_skills = patch("fluxagent.core.skill_loader.SKILLS_DIR", str(root))
    patcher_shell = patch("fluxagent.core.skill_loader.execute_office_shell", shell)
    patcher_skills.start()
    patcher_shell.start()
    loader = LazySkillLoader()
    tool = loader.get_all_tools(force_rescan=True)[0]
    return loader, tool, patcher_skills, patcher_shell


def _help_token(tool, cfg=CFG_A) -> str:
    manual = tool.invoke({"mode": "help"}, config=cfg)
    match = re.search(r"help_token: ([A-Za-z0-9_-]+)", manual)
    assert match, f"help 未签发凭据：{manual[:200]}"
    return match.group(1)


def test_run_without_token_never_reaches_executor():
    with TemporaryDirectory() as tmp:
        shell = Mock()
        _loader, tool, p1, p2 = _make_loader(Path(tmp), shell)
        try:
            result = tool.invoke(
                {"mode": "run", "command": "python {baseDir}/main.py"}, config=CFG_A
            )
            assert "权限拒绝" in result
            shell.invoke.assert_not_called()
        finally:
            p1.stop()
            p2.stop()


def test_forged_token_is_rejected():
    with TemporaryDirectory() as tmp:
        shell = Mock()
        _loader, tool, p1, p2 = _make_loader(Path(tmp), shell)
        try:
            result = tool.invoke(
                {"mode": "run", "command": "python {baseDir}/main.py",
                 "help_token": "forged-token-value"},
                config=CFG_A,
            )
            assert "权限拒绝" in result
            shell.invoke.assert_not_called()
        finally:
            p1.stop()
            p2.stop()


def test_token_is_single_use_and_execution_failure_does_not_refund():
    with TemporaryDirectory() as tmp:
        shell = Mock()
        # 首次执行失败（执行器返回错误），凭据已在派发前消费，不退还
        shell.invoke.return_value = "❌ 工具错误: command_failed"
        _loader, tool, p1, p2 = _make_loader(Path(tmp), shell)
        try:
            token = _help_token(tool)
            first = tool.invoke(
                {"mode": "run", "command": "python {baseDir}/main.py", "help_token": token},
                config=CFG_A,
            )
            assert "工具错误" in first
            shell.invoke.assert_called_once()
            # 重放：即使上次执行失败，凭据也不退还
            replay = tool.invoke(
                {"mode": "run", "command": "python {baseDir}/main.py", "help_token": token},
                config=CFG_A,
            )
            assert "权限拒绝" in replay
            assert shell.invoke.call_count == 1
        finally:
            p1.stop()
            p2.stop()


def test_cross_session_rejection_keeps_parent_receipt_valid():
    with TemporaryDirectory() as tmp:
        shell = Mock()
        shell.invoke.return_value = "executed"
        _loader, tool, p1, p2 = _make_loader(Path(tmp), shell)
        try:
            parent_token = _help_token(tool, CFG_A)
            # 子会话拿父凭据执行：拒绝，且父凭据未被消费
            child = tool.invoke(
                {"mode": "run", "command": "python {baseDir}/main.py",
                 "help_token": parent_token},
                config=CFG_B,
            )
            assert "权限拒绝" in child
            shell.invoke.assert_not_called()
            # 父会话随后仍可用原凭据执行一次
            ok = tool.invoke(
                {"mode": "run", "command": "python {baseDir}/main.py",
                 "help_token": parent_token},
                config=CFG_A,
            )
            assert ok == "executed"
        finally:
            p1.stop()
            p2.stop()


def test_cross_skill_token_is_rejected():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_script_skill(root, "alpha")
        _write_script_skill(root, "beta")
        shell = Mock()
        patcher_skills = patch("fluxagent.core.skill_loader.SKILLS_DIR", str(root))
        patcher_shell = patch("fluxagent.core.skill_loader.execute_office_shell", shell)
        patcher_skills.start()
        patcher_shell.start()
        try:
            loader = LazySkillLoader()
            tools = {t.name: t for t in loader.get_all_tools(force_rescan=True)}
            token = _help_token(tools["alpha"])
            result = tools["beta"].invoke(
                {"mode": "run", "command": "python {baseDir}/main.py",
                 "help_token": token},
                config=CFG_A,
            )
            assert "权限拒绝" in result
            shell.invoke.assert_not_called()
        finally:
            patcher_skills.stop()
            patcher_shell.stop()


def test_ttl_expiry_rejects_without_real_wait():
    with TemporaryDirectory() as tmp:
        clock = FakeClock()
        shell = Mock()
        shell.invoke.return_value = "executed"
        _loader, tool, p1, p2 = _make_loader(Path(tmp), shell)
        try:
            with patch("fluxagent.core.skill_loader.time", clock):
                token = _help_token(tool)
                clock.now += 299  # 有效期内
                ok = tool.invoke(
                    {"mode": "run", "command": "python {baseDir}/main.py",
                     "help_token": token},
                    config=CFG_A,
                )
                assert ok == "executed"
                token2 = _help_token(tool)
                clock.now += 301  # 超过 300 秒 TTL
                expired = tool.invoke(
                    {"mode": "run", "command": "python {baseDir}/main.py",
                     "help_token": token2},
                    config=CFG_A,
                )
                assert "权限拒绝" in expired
        finally:
            p1.stop()
            p2.stop()


def test_manual_change_invalidates_outstanding_token():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        skill = _write_script_skill(root, "demo")
        shell = Mock()
        _loader, tool, p1, p2 = _make_loader(root, shell)
        try:
            token = _help_token(tool)
            # 说明被修改（版本哈希变化）：旧凭据立即失效
            skill.joinpath("SKILL.md").write_text(
                "---\nname: demo\ndescription: gate fixture v2\n---\n\nManual changed.\n",
                encoding="utf-8",
            )
            result = tool.invoke(
                {"mode": "run", "command": "python {baseDir}/main.py",
                 "help_token": token},
                config=CFG_A,
            )
            assert "权限拒绝" in result
            shell.invoke.assert_not_called()
        finally:
            p1.stop()
            p2.stop()


def test_help_without_thread_id_reads_manual_but_issues_no_token():
    with TemporaryDirectory() as tmp:
        shell = Mock()
        _loader, tool, p1, p2 = _make_loader(Path(tmp), shell)
        try:
            manual = tool.invoke({"mode": "help"})
            assert "完整说明书" in manual
            assert "help_token:" not in manual
            shell.invoke.assert_not_called()
        finally:
            p1.stop()
            p2.stop()


def test_doc_only_skill_run_does_not_need_or_consume_credentials():
    """纯文档技能没有执行器：run 在凭据校验前即返回 skill_not_executable。"""
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        skill = root / "manual_only"
        skill.mkdir()
        (skill / "SKILL.md").write_text(
            "---\nname: manual_only\ndescription: doc fixture\n---\n\nRead me.\n",
            encoding="utf-8",
        )
        shell = Mock()
        _loader, tool, p1, p2 = _make_loader(root, shell)
        try:
            result = json.loads(
                tool.invoke({"mode": "run", "command": "python {baseDir}/x.py"}, config=CFG_A)
            )
            assert result["error"]["code"] == "skill_not_executable"
            shell.invoke.assert_not_called()
        finally:
            p1.stop()
            p2.stop()


def test_oversized_manual_is_rejected_not_truncated():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        skill = _write_script_skill(root, "big")
        skill.joinpath("SKILL.md").write_text(
            "---\nname: big\ndescription: big\n---\n\n" + "x" * (64 * 1024 + 1),
            encoding="utf-8",
        )
        shell = Mock()
        _loader, tool, p1, p2 = _make_loader(root, shell)
        try:
            manual = tool.invoke({"mode": "help"}, config=CFG_A)
            assert "权限拒绝" in manual
            assert "help_token:" not in manual
        finally:
            p1.stop()
            p2.stop()


def test_repeated_help_replaces_previous_grant():
    """同会话同技能重复 help：旧凭据作废，只有最新凭据可用。"""
    with TemporaryDirectory() as tmp:
        shell = Mock()
        shell.invoke.return_value = "executed"
        _loader, tool, p1, p2 = _make_loader(Path(tmp), shell)
        try:
            old_token = _help_token(tool)
            new_token = _help_token(tool)
            assert old_token != new_token
            stale = tool.invoke(
                {"mode": "run", "command": "python {baseDir}/main.py",
                 "help_token": old_token},
                config=CFG_A,
            )
            assert "权限拒绝" in stale
            ok = tool.invoke(
                {"mode": "run", "command": "python {baseDir}/main.py",
                 "help_token": new_token},
                config=CFG_A,
            )
            assert ok == "executed"
        finally:
            p1.stop()
            p2.stop()


def test_help_token_redaction_pattern_covers_manual_output():
    """凭据格式必须匹配脱敏正则，保证日志侧能完整遮蔽。"""
    from fluxagent.core.agent import _REDACT_HELP_TOKEN_RE

    token = "Abc123_-xyz"
    sample = f"help_token: {token}\nrun now"
    assert token not in _REDACT_HELP_TOKEN_RE.sub("help_token: [REDACTED]", sample)
