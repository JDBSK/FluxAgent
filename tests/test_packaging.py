# -*- coding: utf-8 -*-
"""打包完整性：find_packages 必须覆盖所有含 __init__.py 的包目录，
纯 wheel 安装后在仓库外也必须能导入核心模块（L03 发布契约的离线部分）。
"""
import os
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_IMPORTS = [
    "fluxagent",
    "fluxagent.core",
    "fluxagent.core.tools",
    "fluxagent.mcp",
    "entry",
]


class PackagingTests(unittest.TestCase):
    def test_every_python_package_has_init(self):
        """任何含 __init__.py 的目录都必须能被 find_packages() 发现，
        否则 wheel 安装后会出现"模块存在但导入失败"的隐性缺包。"""
        discovered = set()
        for dirpath, dirnames, filenames in os.walk(PROJECT_ROOT):
            dirnames[:] = [d for d in dirnames if d not in {
                "__pycache__", ".git", "reference", "learning", "node_modules",
                ".venv", "build", "dist", ".pytest_cache", "logs", "workspace",
                "docs", "examples",
            }]
            if "__init__.py" in filenames:
                discovered.add(Path(dirpath).resolve())
        self.assertTrue(discovered)

        import setuptools  # noqa: F401  确认构建依赖存在
        from setuptools import find_packages
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        found = set(find_packages(where=str(PROJECT_ROOT)))
        for package_dir in discovered:
            relative = package_dir.relative_to(PROJECT_ROOT)
            parts = relative.parts
            if parts and parts[0] in {"fluxagent", "entry", "tests"}:
                name = ".".join(parts)
                self.assertIn(
                    name, found,
                    f"包 {name} 含 __init__.py 但 find_packages 未覆盖（打包漏文件）",
                )

    def test_expected_imports_are_real_packages(self):
        for name in EXPECTED_IMPORTS:
            path = PROJECT_ROOT.joinpath(*name.split("."))
            self.assertTrue((path / "__init__.py").is_file(), f"{name} 缺少 __init__.py")


if __name__ == "__main__":
    unittest.main()
