"""Execute the installed lock hooks against real Git and Cargo fixtures."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
HOOKS = ("pre-commit", "pre-push")


class HookInstallationTests(unittest.TestCase):
    def test_git_entry_points_are_executable_in_the_index(self) -> None:
        result = subprocess.run(
            ["git", "ls-files", "--stage", "--", *[f".githooks/{hook}" for hook in HOOKS]],
            cwd=REPOSITORY, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=10, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        modes = {}
        for entry in result.stdout.splitlines():
            metadata, path = entry.split("\t", 1)
            mode, _object, stage = metadata.split()
            self.assertEqual(stage, "0", f"unmerged hook index entry: {entry}")
            modes[path] = mode
        self.assertEqual(modes, {f".githooks/{hook}": "100755" for hook in HOOKS})


class LockHookTests(unittest.TestCase):
    def setUp(self) -> None:
        # Cargo's Git cache requires a full path on Windows, not the shortened
        # user-profile path returned by the system temporary directory.
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.environment = os.environ.copy()
        self.environment.pop("SKIP_LOCKFILE_CHECK", None)
        self.environment["PYTHON"] = Path(sys.executable).as_posix()
        self.environment["GIT_CONFIG_NOSYSTEM"] = "1"
        self.environment["GIT_CONFIG_GLOBAL"] = str(self.root / "gitconfig")
        self.git = shutil.which("git")
        self.assertIsNotNone(self.git, "hook tests require Git")
        self.bash = shutil.which("bash")
        if os.name == "nt":
            executable_root = Path(self.run_command([self.git, "--exec-path"]).stdout.strip())
            self.bash = str(executable_root.parents[2] / "bin" / "bash.exe")
        self.assertIsNotNone(self.bash, "hook tests require Bash")
        self.run_command([self.git, "init", "-q"])
        for directory in (".githooks", "scripts", "src"):
            (self.root / directory).mkdir()
        for hook in (*HOOKS, "lockfile.sh"):
            destination = self.root / ".githooks" / hook
            shutil.copyfile(REPOSITORY / ".githooks" / hook, destination)
            destination.chmod(0o755)
        shutil.copyfile(REPOSITORY / "scripts/lockfile.py", self.root / "scripts/lockfile.py")
        self.run_command([self.git, "config", "core.hooksPath", ".githooks"])

    def run_command(self, command, *, environment=None, expected=0):
        result = subprocess.run(
            command, cwd=self.root, env=self.environment if environment is None else environment,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=60, check=False,
        )
        if expected is not None:
            self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def hook(self, name, **variables):
        environment = self.environment | variables
        return self.run_command(
            [self.bash, "--noprofile", "--norc", f".githooks/{name}"],
            environment=environment, expected=None,
        )

    def test_missing_checker_rejects_both_hooks(self) -> None:
        (self.root / "scripts/lockfile.py").unlink()
        for hook in HOOKS:
            with self.subTest(hook=hook):
                result = self.hook(hook)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"{hook}: scripts/lockfile.py not present", result.stderr)

    def test_missing_shared_entry_rejects_both_hooks(self) -> None:
        (self.root / ".githooks/lockfile.sh").unlink(missing_ok=True)
        for hook in HOOKS:
            with self.subTest(hook=hook):
                result = self.hook(hook)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(hook, result.stderr)
                self.assertIn("lockfile.sh", result.stderr)

    def test_missing_interpreter_rejects_both_hooks(self) -> None:
        # Absolute Bash starts the actual hook while PATH exposes only Git.
        # No replacement checker or interpreter can manufacture success.
        git_directory = Path(self.git).parent
        if os.name != "nt":
            git_directory = self.root / "git-only"
            git_directory.mkdir()
            (git_directory / "git").symlink_to(self.git)
        for hook in HOOKS:
            with self.subTest(hook=hook):
                result = self.hook(hook, PYTHON="", PATH=str(git_directory))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"{hook}: no python interpreter found", result.stderr)

    def test_explicit_missing_interpreter_rejects_both_hooks(self) -> None:
        interpreter = (self.root / "absent-python").as_posix()
        for hook in HOOKS:
            with self.subTest(hook=hook):
                result = self.hook(hook, PYTHON=interpreter)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(interpreter, result.stderr)

    def test_skip_variable_cannot_hide_checker_failures(self) -> None:
        # Both real checks reject a lock without the required first-party source.
        (self.root / "Cargo.lock").write_text("version = 4\n", encoding="utf-8")
        self.run_command([self.git, "add", "Cargo.lock"])
        for hook in HOOKS:
            with self.subTest(hook=hook):
                result = self.hook(hook, SKIP_LOCKFILE_CHECK="1")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("contains no first-party git sources", result.stderr)

    def test_real_lock_passes_and_stale_lock_retains_cargo_diagnostic(self) -> None:
        dependency = self.root / "dependency"
        (dependency / "src").mkdir(parents=True)
        (dependency / "src/lib.rs").write_text("//! Hook fixture dependency.\n", encoding="utf-8")
        (dependency / "Cargo.toml").write_text(
            '[package]\nname = "hook-dependency"\nversion = "0.1.0"\nedition = "2024"\n',
            encoding="utf-8",
        )
        self.run_command([self.git, "init", "-q", str(dependency)])
        self.run_command([self.git, "-C", str(dependency), "add", "Cargo.toml", "src/lib.rs"])
        self.run_command([
            self.git, "-C", str(dependency), "-c", "user.name=Coeus tests",
            "-c", "user.email=tests@localhost", "commit", "-qm", "Add hook fixture",
        ])
        url = "https://github.com/ryancinsight/hook-fixture"
        # Git resolves this first-party-shaped source locally; Cargo still
        # generates and verifies the actual Git revision and dependency graph.
        self.run_command([
            self.git, "config", "--global", f"url.{dependency.as_uri()}.insteadOf", url,
        ])
        self.environment["CARGO_NET_GIT_FETCH_WITH_CLI"] = "true"
        self.environment["CARGO_HOME"] = str(self.root / "cargo-home")
        (self.root / "src/lib.rs").write_text("//! Hook fixture consumer.\n", encoding="utf-8")
        manifest = self.root / "Cargo.toml"
        manifest.write_text(
            '[package]\nname = "hook-consumer"\nversion = "0.1.0"\nedition = "2024"\n'
            '[workspace]\nexclude = ["dependency"]\n'
            f'[dependencies]\nhook-dependency = {{ git = "{url}" }}\n',
            encoding="utf-8",
        )
        self.run_command([sys.executable, "scripts/lockfile.py", "--regenerate"])
        before = (self.root / "Cargo.lock").read_bytes()
        self.run_command([self.git, "add", "Cargo.lock"])
        for hook in HOOKS:
            result = self.run_command([self.git, "hook", "run", hook])
            if hook == "pre-push":
                self.assertIn(
                    "resolves under --locked; 1 first-party git sources",
                    result.stdout + result.stderr,
                )
        self.assertEqual((self.root / "Cargo.lock").read_bytes(), before)
        manifest.write_text(
            manifest.read_text(encoding="utf-8").replace('version = "0.1.0"', 'version = "0.2.0"'),
            encoding="utf-8",
        )
        result = self.hook("pre-push")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("locked dependency hydration failed", result.stderr)
        self.assertIn("lock file", result.stderr)
        self.assertEqual((self.root / "Cargo.lock").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
