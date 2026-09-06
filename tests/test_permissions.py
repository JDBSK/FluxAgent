import json
import tempfile
import unittest
from pathlib import Path

from fluxagent.core.permissions import PermissionManager
from fluxagent.core.policy import RiskLevel, assess_tool_calls


class TestPermissions(unittest.TestCase):
    def test_session_grant_matches_exact_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = PermissionManager(Path(tmp) / "permissions.json")
            call = {
                "name": "install_mcp_server",
                "args": {"source_type": "git", "source": "https://github.com/a/b"},
            }
            manager.grant_calls([call])
            self.assertTrue(manager.is_allowed(call["name"], call["args"]))
            self.assertFalse(
                manager.is_allowed(
                    call["name"],
                    {"source_type": "git", "source": "https://github.com/other/repo"},
                )
            )

    def test_persistent_grant_uses_v2_hashes_and_reloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "permissions.json"
            call = {
                "name": "execute_office_shell",
                "args": {"command": "python skills/demo.py"},
            }
            manager = PermissionManager(path)
            manager.grant_calls([call], persistent=True)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["version"], 2)
            self.assertNotIn("python skills/demo.py", path.read_text(encoding="utf-8"))
            self.assertEqual(payload["always_allow_calls"][0]["tool"], call["name"])
            self.assertEqual(len(payload["always_allow_calls"][0]["args_sha256"]), 64)
            reloaded = PermissionManager(path)
            self.assertTrue(reloaded.is_allowed(call["name"], call["args"]))
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_legacy_and_non_object_json_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "permissions.json"
            path.write_text(
                json.dumps({"version": 1, "always_allow_tools": ["execute_office_shell"]}),
                encoding="utf-8",
            )
            legacy = PermissionManager(path)
            self.assertFalse(legacy.is_allowed("execute_office_shell", {"command": "python a.py"}))
            path.write_text("[]", encoding="utf-8")
            malformed_shape = PermissionManager(path)
            self.assertEqual(malformed_shape.allowed_tools(), set())

    def test_policy_uses_exact_checker_but_never_bypasses_critical(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = PermissionManager(Path(tmp) / "permissions.json")
            safe_call = {
                "name": "execute_office_shell",
                "args": {"command": "python skills/demo.py"},
            }
            manager.grant_calls([safe_call])
            allowed = assess_tool_calls([safe_call], approval_checker=manager.is_allowed)
            blocked = assess_tool_calls(
                [{"name": "execute_office_shell", "args": {"command": "rm -rf x"}}],
                approval_checker=lambda _name, _args: True,
            )
            self.assertFalse(allowed.needs_approval)
            self.assertFalse(allowed.hard_deny)
            self.assertEqual(blocked.max_level, RiskLevel.CRITICAL)
            self.assertTrue(blocked.hard_deny)

    def test_old_persistent_grant_api_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = PermissionManager(Path(tmp) / "permissions.json")
            with self.assertRaises(ValueError):
                manager.grant(["execute_office_shell"], persistent=True)

    def test_mcp_registry_listing_is_read_only(self):
        decision = assess_tool_calls([{"name": "list_installed_mcp_servers", "args": {}}])
        self.assertEqual(decision.max_level, RiskLevel.READ)
        self.assertFalse(decision.needs_approval)


if __name__ == "__main__":
    unittest.main()
