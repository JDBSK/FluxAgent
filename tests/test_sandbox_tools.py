import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fluxagent.core.tools import sandbox_tools


def error_payload(value: str) -> dict:
    return json.loads(value)


class TestSandboxPaths(unittest.TestCase):
    def test_safe_path_and_sibling_prefix_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            office = root / "office"
            sibling = root / "office_backup"
            office.mkdir()
            sibling.mkdir()
            with patch.object(sandbox_tools, "OFFICE_DIR", str(office)):
                safe = sandbox_tools._get_safe_path("sub/file.txt")
                self.assertTrue(safe.startswith(os.path.realpath(str(office))))
                with self.assertRaises(PermissionError):
                    sandbox_tools._get_safe_path("../office_backup/secret.txt")

    def test_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            office = root / "office"
            outside = root / "outside"
            office.mkdir()
            outside.mkdir()
            link = office / "link"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("当前 Windows 配置不允许创建测试符号链接")
            with patch.object(sandbox_tools, "OFFICE_DIR", str(office)):
                with self.assertRaises(PermissionError):
                    sandbox_tools._get_safe_path("link/secret.txt")


class TestSandboxFiles(unittest.TestCase):
    def test_read_write_and_size_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            office = Path(tmp)
            with patch.object(sandbox_tools, "OFFICE_DIR", str(office)):
                written = sandbox_tools.write_office_file.invoke(
                    {"filepath": "docs/a.txt", "content": "hello", "mode": "w"}
                )
                self.assertIn("成功", written)
                self.assertEqual(
                    sandbox_tools.read_office_file.invoke({"filepath": "docs/a.txt"}),
                    "hello",
                )
                with patch.object(sandbox_tools, "MAX_WRITE_BYTES", 4):
                    payload = error_payload(
                        sandbox_tools.write_office_file.invoke(
                            {"filepath": "large.txt", "content": "12345"}
                        )
                    )
                    self.assertEqual(payload["error"]["code"], "payload_too_large")
                (office / "large.txt").write_text("12345", encoding="utf-8")
                with patch.object(sandbox_tools, "MAX_READ_BYTES", 4):
                    payload = error_payload(
                        sandbox_tools.read_office_file.invoke({"filepath": "large.txt"})
                    )
                    self.assertEqual(payload["error"]["code"], "file_too_large")

    def test_invalid_mode_and_missing_file_are_structured(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(sandbox_tools, "OFFICE_DIR", tmp):
                invalid = error_payload(
                    sandbox_tools.write_office_file.invoke(
                        {"filepath": "a.txt", "content": "x", "mode": "x"}
                    )
                )
                missing = error_payload(
                    sandbox_tools.read_office_file.invoke({"filepath": "missing.txt"})
                )
                self.assertFalse(invalid["ok"])
                self.assertEqual(invalid["error"]["code"], "invalid_mode")
                self.assertEqual(missing["error"]["code"], "not_found")


class TestSandboxShell(unittest.TestCase):
    def test_safe_command_uses_argv_runner(self):
        with patch.object(sandbox_tools.shutil, "which", return_value="C:\\Python\\python.exe"), patch.object(
            sandbox_tools, "_run_command", return_value=(0, "ok", "")
        ) as run:
            result = sandbox_tools.execute_office_shell.invoke(
                {"command": "python skills/demo.py"}
            )
        self.assertIn("ok", result)
        argv = run.call_args.args[0]
        self.assertEqual(argv[0], "C:\\Python\\python.exe")
        self.assertEqual(argv[1], "skills/demo.py")

    def test_quoted_drive_unc_url_and_shell_syntax_are_rejected(self):
        commands = [
            'python "C:\\private\\secret.py"',
            'python "\\\\server\\share\\secret.py"',
            "python a.py && python b.py",
            "python https://example.com/a.py",
            "python -c print(1)",
        ]
        for command in commands:
            with self.subTest(command=command):
                payload = error_payload(
                    sandbox_tools.execute_office_shell.invoke({"command": command})
                )
                self.assertEqual(payload["error"]["code"], "sandbox_violation")

    def test_timeout_and_output_limit_are_structured(self):
        with patch.object(sandbox_tools, "_parse_command", return_value=["python"]), patch.object(
            sandbox_tools,
            "_run_command",
            side_effect=subprocess.TimeoutExpired(["python"], 60),
        ):
            timeout = error_payload(
                sandbox_tools.execute_office_shell.invoke({"command": "python script.py"})
            )
        self.assertEqual(timeout["error"]["code"], "command_timeout")
        self.assertTrue(timeout["error"]["retryable"])

        with patch.object(sandbox_tools, "_parse_command", return_value=["python"]), patch.object(
            sandbox_tools, "_run_command", side_effect=OverflowError("too much")
        ):
            overflow = error_payload(
                sandbox_tools.execute_office_shell.invoke({"command": "python script.py"})
            )
        self.assertEqual(overflow["error"]["code"], "output_limit")

    def test_process_launch_disables_shell(self):
        process = Mock()
        process.stdout = io.BytesIO(b"ok")
        process.stderr = io.BytesIO(b"")
        process.poll.return_value = 0
        process.returncode = 0
        with patch.object(sandbox_tools.subprocess, "Popen", return_value=process) as popen:
            code, stdout, stderr = sandbox_tools._run_command(["python", "script.py"])
        self.assertEqual((code, stdout, stderr), (0, "ok", ""))
        self.assertFalse(popen.call_args.kwargs["shell"])


if __name__ == "__main__":
    unittest.main()
