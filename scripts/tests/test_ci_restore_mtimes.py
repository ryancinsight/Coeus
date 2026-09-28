"""Value-semantic tests for the git-derived mtime restoration used to keep
the workspace-crate build cache content-addressed across fresh checkouts."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.ci_restore_mtimes import last_commit_mtimes, restore, tracked_paths

_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "Coeus CI Fixture",
    "GIT_AUTHOR_EMAIL": "ci-fixture@example.invalid",
    "GIT_COMMITTER_NAME": "Coeus CI Fixture",
    "GIT_COMMITTER_EMAIL": "ci-fixture@example.invalid",
}


def _run(args: list[str], cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, env=_ENV)


def _commit(cwd: Path, message: str, timestamp: int) -> None:
    env = {
        **_ENV,
        "GIT_AUTHOR_DATE": f"{timestamp} +0000",
        "GIT_COMMITTER_DATE": f"{timestamp} +0000",
    }
    subprocess.run(
        ["git", "commit", "-m", message, "--no-gpg-sign"],
        cwd=cwd,
        check=True,
        capture_output=True,
        env=env,
    )


class RestoreMtimesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)
        _run(["git", "init", "--initial-branch=main"], self.repo)
        _run(["git", "config", "commit.gpgsign", "false"], self.repo)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_untouched_file_gets_its_original_commit_time(self) -> None:
        (self.repo / "stable.txt").write_text("unchanged\n")
        (self.repo / "changed.txt").write_text("v1\n")
        _run(["git", "add", "-A"], self.repo)
        _commit(self.repo, "initial", 1_700_000_000)

        (self.repo / "changed.txt").write_text("v2\n")
        _run(["git", "add", "-A"], self.repo)
        _commit(self.repo, "update changed.txt only", 1_700_100_000)

        mtimes = last_commit_mtimes(self.repo)
        self.assertEqual(mtimes["stable.txt"], 1_700_000_000)
        self.assertEqual(mtimes["changed.txt"], 1_700_100_000)

    def test_restore_sets_disk_mtimes_from_history(self) -> None:
        (self.repo / "a.txt").write_text("a\n")
        _run(["git", "add", "-A"], self.repo)
        _commit(self.repo, "add a", 1_700_000_500)

        # Simulate a fresh checkout: every file's mtime jumps to "now".
        now = 1_800_000_000
        os.utime(self.repo / "a.txt", (now, now))
        self.assertEqual(int(os.stat(self.repo / "a.txt").st_mtime), now)

        applied = restore(self.repo)

        self.assertEqual(applied, 1)
        self.assertEqual(int(os.stat(self.repo / "a.txt").st_mtime), 1_700_000_500)

    def test_two_checkouts_of_identical_content_agree(self) -> None:
        (self.repo / "shared.txt").write_text("shared\n")
        _run(["git", "add", "-A"], self.repo)
        _commit(self.repo, "add shared.txt", 1_700_000_777)

        first = restore(self.repo)
        os.utime(self.repo / "shared.txt", (1_900_000_000, 1_900_000_000))
        second = restore(self.repo)

        self.assertEqual(first, second)
        self.assertEqual(
            int(os.stat(self.repo / "shared.txt").st_mtime), 1_700_000_777
        )

    def test_tracked_paths_matches_git_ls_files(self) -> None:
        (self.repo / "one.txt").write_text("1\n")
        (self.repo / "two.txt").write_text("2\n")
        _run(["git", "add", "-A"], self.repo)
        _commit(self.repo, "add two files", 1_700_000_900)

        self.assertEqual(set(tracked_paths(self.repo)), {"one.txt", "two.txt"})


if __name__ == "__main__":
    unittest.main()
