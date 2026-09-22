"""Focused regression tests for policy approval and command hardening."""

import unittest

from fluxagent.core.policy import RiskLevel, assess_call, assess_tool_calls, build_policy_config


class TestPolicyHardening(unittest.TestCase):
    def test_legacy_gate_respects_exact_approval_checker(self):
        call = {
            "name": "write_office_file",
            "args": {"filepath": "ok.txt", "content": "x"},
        }
        checker = lambda name, args: name == call["name"] and args == call["args"]
        decision = assess_tool_calls(
            [call],
            config=build_policy_config(approval_tools={"write_office_file"}),
            approval_checker=checker,
        )
        changed = assess_tool_calls(
            [{"name": "write_office_file", "args": {"filepath": "other.txt", "content": "x"}}],
            config=build_policy_config(approval_tools={"write_office_file"}),
            approval_checker=checker,
        )
        self.assertFalse(decision.needs_approval)
        self.assertTrue(changed.needs_approval)

    def test_critical_call_cannot_be_approved(self):
        decision = assess_tool_calls(
            [{"name": "execute_office_shell", "args": {"command": "rm -rf data"}}],
            approval_checker=lambda _name, _args: True,
        )
        self.assertTrue(decision.hard_deny)
        self.assertFalse(decision.needs_approval)

    def test_empty_overwrite_cannot_bypass_delete_denial(self):
        assessment = assess_call(
            "write_office_file",
            {"filepath": "protected.txt", "content": "", "mode": "w"},
        )
        self.assertTrue(assessment.hard_deny)
        self.assertIn("write_empty_overwrite", assessment.reasons)

        append = assess_call(
            "write_office_file",
            {"filepath": "audit.log", "content": "", "mode": "a"},
        )
        self.assertFalse(append.hard_deny)

    def test_installer_source_and_subdirectory_signals(self):
        safe = assess_call(
            "install_mcp_server",
            {
                "source_type": "git",
                "source": "https://github.com/modelcontextprotocol/servers",
                "subdirectory": "src/filesystem",
                "command": "node",
                "launch_args": ["dist/index.js", "."],
            },
        )
        unsafe_url = assess_call(
            "install_skill", {"source_type": "git", "source": "file:///tmp/repo"}
        )
        traversal = assess_call(
            "install_skill",
            {
                "source_type": "git",
                "source": "https://github.com/example/repo",
                "subdirectory": "../outside",
            },
        )
        self.assertEqual(safe.level, RiskLevel.EXEC)
        self.assertFalse(safe.hard_deny)
        self.assertTrue(unsafe_url.hard_deny)
        self.assertTrue(traversal.hard_deny)

    def test_windows_paths_and_delete_commands_are_critical(self):
        commands = [
            'type "D:\\private\\secret.txt"',
            'type "\\\\server\\share\\secret.txt"',
            "echo done && rmdir cache",
            'powershell -Command "Remove-Item cache -Recurse"',
        ]
        for command in commands:
            with self.subTest(command=command):
                assessment = assess_call("execute_office_shell", {"command": command})
                self.assertTrue(assessment.hard_deny)

    def test_interpreter_name_not_substring_false_positive(self):
        interpreter = assess_call(
            "execute_office_shell", {"command": "python -m pytest"}
        )
        harmless = assess_call(
            "execute_office_shell", {"command": 'echo "rm -rf is documentation"'}
        )
        self.assertIn("shell_interpreter:python", interpreter.reasons)
        self.assertFalse(harmless.hard_deny)

    def test_malformed_and_unknown_calls_are_conservative(self):
        malformed = assess_tool_calls(["not-a-dict"])
        unknown = assess_call("new_read_tool", {})
        self.assertEqual(malformed.max_level, RiskLevel.WRITE)
        self.assertTrue(malformed.needs_approval)
        self.assertIn("unknown_tool", malformed.assessments[0].reasons)
        self.assertIn("unknown_tool", unknown.reasons)


if __name__ == "__main__":
    unittest.main()
