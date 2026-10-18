"""Protect source coverage and failure propagation on every host shell."""

import contextlib
import io
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.check_python_sources import check_sources


class PythonSourceChecksTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="radiance-source-check-")
        self.addCleanup(temp.cleanup)
        self.repo = Path(temp.name)
        self.paths = (
            "_patchlib.py", "install_radiance_hooks.py", "patch_example.py",
            "radiance_example.py", "benchmarks/bin/nested/check.py", "tools/check.py",
        )
        for name in self.paths:
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            # Compilation must never import unavailable runtime dependencies.
            path.write_text("import nonexistent_gpu_runtime\n", encoding="utf-8")

    def check(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return check_sources(self.repo)

    def test_valid_sources_compile_without_imports(self):
        self.assertTrue(self.check())

    def test_syntax_error_in_each_source_group_fails(self):
        for name in self.paths:
            with self.subTest(path=name):
                path = self.repo / name
                source = path.read_text(encoding="utf-8")
                path.write_text("def broken(:\n", encoding="utf-8")
                self.assertFalse(self.check())
                path.write_text(source, encoding="utf-8")

    def test_unmatched_glob_fails(self):
        for name in ("patch_example.py", "radiance_example.py"):
            with self.subTest(path=name):
                path = self.repo / name
                source = path.read_text(encoding="utf-8")
                path.unlink()
                self.assertFalse(self.check())
                path.write_text(source, encoding="utf-8")

    def test_missing_required_file_fails(self):
        (self.repo / "_patchlib.py").unlink()
        self.assertFalse(self.check())

    def test_missing_required_directory_fails(self):
        (self.repo / "tools/check.py").unlink()
        (self.repo / "tools").rmdir()
        self.assertFalse(self.check())

    def test_command_exits_nonzero_for_invalid_runtime_source(self):
        script = self.repo / "tools/check_python_sources.py"
        shutil.copyfile(Path(__file__).resolve().parents[1] / "tools/check_python_sources.py", script)
        (self.repo / "radiance_example.py").write_text("def broken(:\n", encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("radiance_example.py", result.stdout)


if __name__ == "__main__":
    unittest.main()
