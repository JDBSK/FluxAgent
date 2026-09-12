import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from fluxagent.core.skill_loader import LazySkillLoader


def _write_skill(root: Path, name: str, scripts: dict[str, str] | None = None) -> Path:
    skill = root / name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test skill\n---\n\nInstructions.\n",
        encoding="utf-8",
    )
    for filename, content in (scripts or {}).items():
        path = skill / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return skill


def test_manual_only_skill_reports_non_executable_instead_of_spawning_python():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_skill(root, "manual_skill")
        with patch("fluxagent.core.skill_loader.SKILLS_DIR", str(root)):
            tool = LazySkillLoader().get_all_tools(force_rescan=True)[0]
            result = json.loads(tool.invoke({"mode": "run", "command": "python {baseDir}/ask.py"}))

    assert result["error"]["code"] == "skill_not_executable"


def test_unknown_skill_script_is_rejected_with_available_scripts():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_skill(root, "script_skill", {"scripts/main.py": "print('ok')\n"})
        with patch("fluxagent.core.skill_loader.SKILLS_DIR", str(root)):
            tool = LazySkillLoader().get_all_tools(force_rescan=True)[0]
            cfg = {"configurable": {"thread_id": "tester"}}
            manual = tool.invoke({"mode": "help"}, config=cfg)
            token = re.search(r"help_token: ([A-Za-z0-9_-]+)", manual).group(1)
            result = json.loads(
                tool.invoke(
                    {
                        "mode": "run",
                        "command": "python {baseDir}/scripts/missing.py",
                        "help_token": token,
                    },
                    config=cfg,
                )
            )

    assert result["error"]["code"] == "skill_script_not_found"
    assert "scripts/main.py" in result["error"]["message"]


def test_known_skill_script_keeps_base_dir_replacement():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_skill(root, "script_skill", {"scripts/main.py": "print('ok')\n"})
        shell = Mock()
        shell.invoke.return_value = "ran"
        with patch("fluxagent.core.skill_loader.SKILLS_DIR", str(root)), patch(
            "fluxagent.core.skill_loader.execute_office_shell", shell
        ):
            tool = LazySkillLoader().get_all_tools(force_rescan=True)[0]
            cfg = {"configurable": {"thread_id": "tester"}}
            # 新契约：脚本技能 run 必须先 help 获取本会话一次性 help_token。
            manual = tool.invoke({"mode": "help"}, config=cfg)
            token = re.search(r"help_token: ([A-Za-z0-9_-]+)", manual).group(1)
            result = tool.invoke(
                {
                    "mode": "run",
                    "command": "python {baseDir}/scripts/main.py --question hi",
                    "help_token": token,
                },
                config=cfg,
            )

    assert result == "ran"
    shell.invoke.assert_called_once()
    assert shell.invoke.call_args.args[0] == {
        "command": "python skills/script_skill/scripts/main.py --question hi"
    }


def test_skill_metadata_exposes_declared_trigger_aliases():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        skill = root / "persona"
        skill.mkdir()
        (skill / "SKILL.md").write_text(
            "---\n"
            "name: persona\n"
            "description: |\n"
            "  A response persona.\n"
            "触发词：「景辰」「persona mode」\n"
            "---\n",
            encoding="utf-8",
        )
        with patch("fluxagent.core.skill_loader.SKILLS_DIR", str(root)):
            tool = LazySkillLoader().get_all_tools(force_rescan=True)[0]

    assert tool.metadata["capability"] == "persona"
    assert "景辰" in tool.metadata["aliases"]
    assert "persona mode" in tool.metadata["aliases"]
