"""Use real Git histories to protect local and CI changed-test selection."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import radiance


class ChangedFilesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="radiance-cli-")
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.git("init", "-b", "main")
        self.git("config", "user.name", "Radiance test")
        self.git("config", "user.email", "radiance-test@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        self.git("config", "core.hooksPath", str(self.repo / "no-hooks"))
        self.write("tracked.py", "baseline\n")
        self.write(".gitignore", "ignored/\n")
        self.commit()
        self.base = self.git("rev-parse", "HEAD").strip()
        self.git("update-ref", "refs/remotes/origin/main", self.base)
        self.repo_patch = patch.object(radiance, "REPO", self.repo)
        self.repo_patch.start()
        self.addCleanup(self.repo_patch.stop)

    def git(self, *args):
        return subprocess.check_output(
            ["git", "-C", str(self.repo), *args], text=True, stderr=subprocess.STDOUT
        )

    def write(self, name, text):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def commit(self):
        self.git("add", "-A")
        self.git("commit", "-m", "test fixture")

    def test_clean_checkout(self):
        self.assertEqual(radiance.changed_files(None), [])

    def test_unstaged_and_staged_changes_are_deduplicated(self):
        self.write("tracked.py", "staged\n")
        self.git("add", "tracked.py")
        self.assertEqual(radiance.changed_files(None), ["tracked.py"])
        self.write("tracked.py", "unstaged\n")
        self.assertEqual(radiance.changed_files(None), ["tracked.py"])
        self.git("reset", "--", "tracked.py")
        self.assertEqual(radiance.changed_files(None), ["tracked.py"])

    def test_new_files_are_selected_without_git_add_but_ignored_files_are_not(self):
        self.write("new regression.py", "new\n")
        self.write("ignored/private.py", "ignored\n")
        self.assertEqual(radiance.changed_files(None), ["new regression.py"])
        self.git("add", "-N", "new regression.py")
        self.assertEqual(radiance.changed_files(None), ["new regression.py"])

    def test_explicit_base_includes_committed_staged_and_untracked_changes(self):
        self.write("committed.py", "committed\n")
        self.commit()
        self.write("tracked.py", "staged\n")
        self.git("add", "tracked.py")
        self.write("untracked.py", "untracked\n")
        expected = ["committed.py", "tracked.py", "untracked.py"]
        self.assertEqual(radiance.changed_files(self.base), expected)
        self.assertEqual(radiance.changed_files(None), expected)

    def test_rename_preserves_both_source_and_destination_areas(self):
        self.git("mv", "tracked.py", "renamed.py")
        self.assertEqual(radiance.changed_files(None), ["renamed.py", "tracked.py"])
        self.commit()
        self.assertEqual(radiance.changed_files(self.base), ["renamed.py", "tracked.py"])

    def test_deleted_file_is_selected(self):
        (self.repo / "tracked.py").unlink()
        self.assertEqual(radiance.changed_files(None), ["tracked.py"])

    def test_without_remote_base_falls_back_to_last_commit_and_local_edits(self):
        self.git("update-ref", "-d", "refs/remotes/origin/main")
        self.write("committed.py", "committed\n")
        self.commit()
        self.write("local.py", "local\n")
        self.assertEqual(radiance.changed_files(None), ["committed.py", "local.py"])

    def test_invalid_base_fails_instead_of_silently_running_baseline(self):
        with self.assertRaises(SystemExit):
            radiance.changed_files("missing-ref")

    def test_local_runtime_file_selects_its_area(self):
        self.write("radiance_kv_offload.py", "local change\n")
        areas = radiance.areas_for_files(radiance.load(), radiance.changed_files(None))
        self.assertIn("kv-offload", areas)


if __name__ == "__main__":
    unittest.main()
