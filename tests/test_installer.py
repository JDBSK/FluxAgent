import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from fluxagent.core.installer import (
    InstallError,
    _install_tree,
    _prepare_mcp,
    _prepare_python_mcp,
    install_mcp_source,
    install_skill_source,
    list_installed_mcp_servers,
)
from fluxagent.core.context import repair_unpaired_tool_calls
from fluxagent.core.policy import RiskLevel, assess_call
from fluxagent.core.tools.builtins import BUILTIN_TOOLS, install_mcp_server


class TestControlledInstaller(unittest.TestCase):
    def test_installs_local_skill_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source_skill"
            source.mkdir()
            (source / "SKILL.md").write_text(
                "---\nname: demo\ndescription: demo skill\n---\n# Demo\n",
                encoding="utf-8",
            )
            (source / "main.py").write_text("print('ok')\n", encoding="utf-8")
            skills = root / "skills"

            destination = install_skill_source(
                "local", str(source), target_root=skills
            )

            self.assertEqual(destination, skills / "source_skill")
            self.assertTrue((destination / "SKILL.md").is_file())
            self.assertTrue((destination / "main.py").is_file())

    def test_rejects_unapproved_git_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(InstallError, "允许主机"):
                install_skill_source(
                    "git",
                    "https://example.com/org/repo.git",
                    target_root=Path(tmp) / "skills",
                )

    @patch("fluxagent.core.installer._probe_mcp", return_value=["demo_tool"])
    @patch(
        "fluxagent.core.installer._prepare_mcp",
        return_value=("python", ["server.py"], "Python/venv"),
    )
    def test_installs_and_registers_local_mcp_server(self, prepare, probe):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source_mcp"
            source.mkdir()
            (source / "server.py").write_text("print('server')\n", encoding="utf-8")
            servers_root = root / "servers"
            index_path = servers_root / "index.json"

            record = install_mcp_source(
                "local",
                str(source),
                name="demo-mcp",
                command="python",
                args=["server.py"],
                target_root=servers_root,
                index_path=index_path,
            )

            self.assertEqual(record.name, "demo-mcp")
            self.assertTrue((servers_root / "demo-mcp" / "server.py").is_file())
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["servers"][0]["command"], "python")
            self.assertEqual(
                list_installed_mcp_servers(index_path)[0]["name"], "demo-mcp"
            )
            prepare.assert_called_once()
            probe.assert_called_once()

    @patch("fluxagent.core.installer._run_setup")
    def test_node_manifest_overrides_incorrect_model_launch_guess(self, run_setup):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "package.json").write_text(
                json.dumps(
                    {
                        "main": "./dist/index.js",
                        "scripts": {"build": "tsup"},
                    }
                ),
                encoding="utf-8",
            )

            def finish_build(command, cwd, label):
                if command == ["npm", "run", "build"]:
                    (cwd / "dist").mkdir()
                    (cwd / "dist" / "index.js").write_text("", encoding="utf-8")

            run_setup.side_effect = finish_build

            command, args, runtime = _prepare_mcp(root, "python", ["-m", "server"])

            self.assertEqual(command, "node")
            self.assertEqual(args, ["dist/index.js"])
            self.assertEqual(runtime, "Node/npm")

    @patch("fluxagent.core.installer._run_setup")
    def test_node_install_retries_known_npm_peer_graph_failure(self, run_setup):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "package.json").write_text(
                json.dumps({"main": "index.js"}), encoding="utf-8"
            )
            (root / "index.js").write_text("", encoding="utf-8")
            run_setup.side_effect = [
                InstallError("Cannot read properties of null (reading 'edgesOut')"),
                None,
            ]

            command, args, _ = _prepare_mcp(root, "", [])

            self.assertEqual((command, args), ("node", ["index.js"]))
            self.assertEqual(run_setup.call_count, 2)
            self.assertIn("--legacy-peer-deps", run_setup.call_args_list[1].args[0])

    def test_rejects_linux_only_python_wheel_on_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text(
                '[project]\nrequires-python = ">=3.12"\n'
                '[tool.uv.sources]\nqtf = { path = "libs/qtf-manylinux.whl" }\n',
                encoding="utf-8",
            )
            with patch("fluxagent.core.installer.os.name", "nt"):
                with self.assertRaisesRegex(InstallError, "Linux 专用 wheel"):
                    _prepare_python_mcp(root)

    def test_install_tree_reports_existing_target_without_deleting_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "new.txt").write_text("new", encoding="utf-8")
            target_root = root / "servers"
            existing = target_root / "same"
            existing.mkdir(parents=True)
            (existing / "keep.txt").write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(InstallError, "目标已存在"):
                _install_tree(source, target_root, "same")

            self.assertTrue((existing / "keep.txt").is_file())

    @patch(
        "fluxagent.core.installer._prepare_mcp",
        side_effect=InstallError("Node MCP 构建失败"),
    )
    def test_failed_setup_rolls_back_source_and_registry(self, prepare):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source_mcp"
            source.mkdir()
            (source / "package.json").write_text("{}", encoding="utf-8")
            servers_root = root / "servers"
            index_path = servers_root / "index.json"

            with self.assertRaisesRegex(InstallError, "构建失败"):
                install_mcp_source(
                    "local",
                    str(source),
                    name="broken-mcp",
                    target_root=servers_root,
                    index_path=index_path,
                )

            self.assertFalse((servers_root / "broken-mcp").exists())
            self.assertFalse(index_path.exists())

    @patch(
        "fluxagent.core.installer._probe_mcp",
        side_effect=InstallError("MCP 启动检查失败: Connection closed"),
    )
    @patch(
        "fluxagent.core.installer._prepare_mcp",
        return_value=("node", ["dist/index.js"], "Node/npm"),
    )
    def test_failed_probe_rolls_back_source_and_registry(self, prepare, probe):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source_mcp"
            source.mkdir()
            (source / "package.json").write_text("{}", encoding="utf-8")
            servers_root = root / "servers"
            index_path = servers_root / "index.json"

            with self.assertRaisesRegex(InstallError, "Connection closed"):
                install_mcp_source(
                    "local",
                    str(source),
                    name="unhealthy-mcp",
                    target_root=servers_root,
                    index_path=index_path,
                )

            self.assertFalse((servers_root / "unhealthy-mcp").exists())
            self.assertFalse(index_path.exists())

    @patch("fluxagent.core.installer._probe_mcp", return_value=["quote"])
    @patch(
        "fluxagent.core.installer._prepare_mcp",
        return_value=("node", ["dist/index.js"], "Node/npm"),
    )
    def test_reinstall_same_source_repairs_existing_registration(self, prepare, probe):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            servers_root = root / "servers"
            installed = servers_root / "stock-sdk-mcp"
            installed.mkdir(parents=True)
            (installed / "package.json").write_text("{}", encoding="utf-8")
            index_path = servers_root / "index.json"
            index_path.write_text(
                json.dumps(
                    {
                        "servers": [
                            {
                                "name": "stock-sdk-mcp",
                                "source_type": "git",
                                "source": "https://github.com/example/stock-sdk-mcp",
                                "installed_path": str(installed),
                                "command": "python",
                                "args": ["-m", "server"],
                                "read_only_tools": [],
                                "installed_at": "2026-09-05T00:00:00Z",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            record = install_mcp_source(
                "git",
                "https://github.com/example/stock-sdk-mcp",
                name="stock-sdk-mcp",
                command="python",
                args=["-m", "server"],
                target_root=servers_root,
                index_path=index_path,
            )

            self.assertEqual(record.command, "node")
            self.assertEqual(record.args, ["dist/index.js"])
            saved = list_installed_mcp_servers(index_path)[0]
            self.assertEqual(saved["command"], "node")
            self.assertEqual(saved["args"], ["dist/index.js"])
            prepare.assert_called_once()
            probe.assert_called_once()

    def test_install_tools_require_approval_but_are_not_hard_denied(self):
        for tool_name in ("install_skill", "install_mcp_server"):
            assessment = assess_call(tool_name, {"source": "https://github.com/org/repo"})
            self.assertEqual(assessment.level, RiskLevel.EXEC)
            self.assertFalse(assessment.hard_deny)

    def test_install_tools_are_registered_for_agent_and_mcp(self):
        tool_names = {tool.name for tool in BUILTIN_TOOLS}
        self.assertTrue(
            {"install_skill", "install_mcp_server", "list_installed_mcp_servers"}
            .issubset(tool_names)
        )

    def test_install_mcp_server_schema_does_not_rename_launch_args(self):
        schema = install_mcp_server.args_schema.model_json_schema()
        properties = schema["properties"]
        self.assertIn("launch_args", properties)
        self.assertNotIn("v__args", properties)
        self.assertNotIn("command", schema.get("required", []))

    def test_repairs_orphan_tool_calls_for_next_llm_request(self):
        messages = [
            HumanMessage(content="安装 MCP"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "install_mcp_server",
                        "args": {"v__args": ["-y", "demo"]},
                        "id": "call_legacy",
                        "type": "tool_call",
                    }
                ],
            ),
            HumanMessage(content="继续处理"),
        ]

        repaired = repair_unpaired_tool_calls(messages)

        self.assertIsInstance(repaired[2], ToolMessage)
        self.assertEqual(repaired[2].tool_call_id, "call_legacy")
        self.assertEqual(repaired[3].content, "继续处理")
