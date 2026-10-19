"""CPU-only fail-closed boundary checks; native source/build checks run separately."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rocr_idle_backport", ROOT / "tools/rocr_idle_backport.py")
backport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backport)


class BackportBoundaryTests(unittest.TestCase):
    def git_fixture(self, directory):
        root = Path(directory)
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        path = root / "source.cpp"
        path.write_bytes(b"int original;\n")
        subprocess.run(["git", "add", "."], cwd=root, check=True)
        subprocess.run(["git", "-c", "user.name=CPU Test", "-c", "user.email=cpu@example.invalid",
                        "commit", "-qm", "baseline"], cwd=root, check=True)
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        path.write_bytes(b"int patched;\n")
        return root, path, {"source_commit": revision, "patched_files": {"source.cpp": backport.source_digest(path)}}

    def test_recorded_series_is_complete_and_unchanged(self):
        manifest = backport.verify_series()
        self.assertEqual(len(manifest["patches"]), 3)
        self.assertEqual(len({p["upstream_commit"] for p in manifest["patches"]}), 3)
        self.assertEqual(len(manifest["patched_files"]), 8)

    def test_corrupt_patch_is_rejected_before_git(self):
        with patch.object(backport, "digest", return_value="0" * 64), patch.object(backport, "run") as command:
            with self.assertRaisesRegex(ValueError, "checksum"):
                backport.verify_series()
            command.assert_not_called()

    def test_existing_checkout_is_never_reused(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(backport, "run") as command:
            marker = Path(directory) / "keep.txt"
            marker.write_text("previous failed attempt")
            with self.assertRaises(FileExistsError):
                backport.prepare(Path(directory))
            command.assert_not_called()
            self.assertEqual(marker.read_text(), "previous failed attempt")

    def test_wrong_source_revision_is_rejected(self):
        with patch.object(backport, "run", return_value="0" * 40):
            with self.assertRaisesRegex(ValueError, "pinned"):
                backport.verify_source(ROOT)

    def test_modified_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, path, expected = self.git_fixture(directory)
            with patch.object(backport, "verify_series", return_value=expected):
                backport.verify_source(root)
                path.write_bytes(b"int unintended_change;\n")
                with self.assertRaisesRegex(ValueError, "source mismatch"):
                    backport.verify_source(root)

    def test_unrelated_source_edit_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, path, expected = self.git_fixture(directory)
            (root / "unreviewed.cpp").write_bytes(b"int extra;\n")
            with patch.object(backport, "verify_series", return_value=expected):
                with self.assertRaisesRegex(ValueError, "Unexpected"):
                    backport.verify_source(root)

    def test_only_line_endings_may_differ(self):
        with tempfile.TemporaryDirectory() as directory:
            root, path, expected = self.git_fixture(directory)
            path.write_bytes(b"int patched;\r\n")
            with patch.object(backport, "verify_series", return_value=expected):
                backport.verify_source(root)
                path.write_bytes(b"int  patched;\r\n")
                with self.assertRaisesRegex(ValueError, "source mismatch"):
                    backport.verify_source(root)

    def test_unknown_original_binary_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "libhsa-runtime64.so.1.21.0"
            path.write_bytes(b"a different runtime with the same filename")
            with self.assertRaisesRegex(ValueError, "re-audit"):
                backport.check_runtime(path)
            self.assertEqual(path.read_bytes(), b"a different runtime with the same filename")

    def test_default_build_has_no_backport_activation(self):
        self.assertNotIn("rocr_idle", (ROOT / "Dockerfile").read_text())
        self.assertNotIn("rocr-idle", (ROOT / "Dockerfile.patch").read_text())
        self.assertNotIn("HSA_TOOLS_DISABLE_REGISTER", (ROOT / "Dockerfile.rocr-idle").read_text())


if __name__ == "__main__":
    unittest.main()
