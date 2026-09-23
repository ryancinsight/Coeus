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
ZERO = "0" * 40
# `pre-push` reads the pushed range from stdin to decide whether the range
# touches Cargo.lock/Cargo.toml at all (a legitimate skip, not a bypass) and
# to select the revision it exports and checks. These fixtures have no
# `origin` remote, so `default_branch_base` cannot resolve one and the hook
# falls back to its conservative default: run the check anyway. The SHA does
# not need to resolve to a real commit for that fallback to trigger -- it
# only needs to be a non-zero placeholder for a "new branch" update, so one
# probe line serves every test below regardless of what the fixture has
# committed.
PROBE_PUSH_LINE = f"refs/heads/probe {'1' * 40} refs/heads/probe {ZERO}\n"


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
        for hook in HOOKS:
            destination = self.root / ".githooks" / hook
            shutil.copyfile(REPOSITORY / ".githooks" / hook, destination)
            destination.chmod(0o755)
        shutil.copyfile(REPOSITORY / "scripts/lockfile.py", self.root / "scripts/lockfile.py")
        self.run_command([self.git, "config", "core.hooksPath", ".githooks"])
        # `pre-push` exports and checks a *committed* revision (never the
        # bare working tree), so it needs a real HEAD to export from the
        # first invocation onward. Seed one now, with the checker present,
        # so every test below starts from a valid, checker-carrying history
        # and only has to commit the state it specifically wants to vary.
        self.run_command([self.git, "add", "-A"])
        self.run_command([self.git, "commit", "-qm", "Seed hooks and checker"])

    def run_command(self, command, *, environment=None, expected=0, input_text=None):
        env = self.environment if environment is None else environment
        if input_text is None:
            result = subprocess.run(
                command, cwd=self.root, env=env,
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=60, check=False,
            )
        else:
            # Bytes, not text: a text-mode pipe on Windows translates `\n` to
            # `\r\n` while writing the child's stdin, which corrupts the
            # push-line protocol `pre-push` parses byte-for-byte -- a `\r`
            # riding along in `remote_sha` makes it compare unequal to the
            # all-zeros sentinel, so the "new branch" case is never taken
            # (same fix as atlas's scripts/tests/test_atlas_pre_push_gate.py
            # GateFixture.run_hook).
            raw = subprocess.run(
                command, cwd=self.root, env=env,
                input=input_text.encode("utf-8"),
                capture_output=True, timeout=60, check=False,
            )
            result = subprocess.CompletedProcess(
                raw.args, raw.returncode,
                stdout=raw.stdout.decode("utf-8", errors="replace"),
                stderr=raw.stderr.decode("utf-8", errors="replace"),
            )
        if expected is not None:
            self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def hook(self, name, **variables):
        environment = self.environment | variables
        # `pre-push` alone reads a push range from stdin; feeding it to
        # `pre-commit` too would be inert (the commit hook never reads stdin)
        # but stays scoped to the hook that needs it for clarity.
        input_text = PROBE_PUSH_LINE if name == "pre-push" else None
        return self.run_command(
            [self.bash, "--noprofile", "--norc", f".githooks/{name}"],
            environment=environment, expected=None, input_text=input_text,
        )

    def test_missing_checker_rejects_both_hooks(self) -> None:
        (self.root / "scripts/lockfile.py").unlink()
        result = self.hook("pre-commit")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pre-commit: scripts/lockfile.py not present", result.stderr)

        # `pre-push` checks the checker in the *pushed* revision, not the
        # working tree, so the removal must be committed before it is
        # visible there. The removal commit would itself be refused by the
        # now-installed pre-commit (which reads the same working-tree
        # absence) -- `--no-verify` constructs the adversarial history this
        # sub-test needs pre-push to catch, the same way a commit made
        # before the hook existed, or with `--no-verify` for real, would.
        self.run_command([self.git, "add", "-A"])
        self.run_command([self.git, "commit", "--no-verify", "-qm", "Remove the checker"])
        result = self.hook("pre-push")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pre-push: scripts/lockfile.py not present", result.stderr)

    # test_missing_shared_entry_rejects_both_hooks is retired: the shared
    # `.githooks/lockfile.sh` entry point it exercised is gone. The new hooks
    # (adopted whole from the atlas stack, ATLAS-PREPUSH-HOOK-FORKED-ACROSS-
    # MEMBERS-2026-09-09) are each self-contained -- no shared entry file for
    # either to be missing.

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
        # Both real checks reject a lock without the required first-party
        # source. `pre-commit` checks the staged blob; `pre-push` exports and
        # checks the committed revision, never the bare working tree, so the
        # bad lock is checked staged first and then committed before the
        # push side of this is exercised.
        (self.root / "Cargo.lock").write_text("version = 4\n", encoding="utf-8")
        self.run_command([self.git, "add", "Cargo.lock"])
        result = self.hook("pre-commit", SKIP_LOCKFILE_CHECK="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("contains no first-party git sources", result.stderr)
        self.assertIn("SKIP_LOCKFILE_CHECK is no longer honoured", result.stderr)
        # `pre-push` checks the committed revision; `--no-verify` gets the
        # same bad lock into history without going through the pre-commit
        # this fixture just proved refuses it (a commit made with
        # `--no-verify` for real, or before this hook existed, is exactly
        # what `pre-push` exists to still catch).
        self.run_command([self.git, "commit", "--no-verify", "-qm", "Add a flattened lock"])
        result = self.hook("pre-push", SKIP_LOCKFILE_CHECK="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("contains no first-party git sources", result.stderr)
        self.assertIn("SKIP_LOCKFILE_CHECK is no longer honoured", result.stderr)

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

        # `pre-commit` checks the staged blob; `pre-push` exports and checks
        # a committed revision, never the bare working tree. Stage the good
        # manifest/lock pair, prove `pre-commit` passes it staged, then
        # commit so `pre-push` has a real revision to export and check.
        self.run_command([self.git, "add", "-A"])
        self.run_command([self.git, "hook", "run", "pre-commit"])
        self.run_command([
            self.git, "commit", "-qm", "Add hook fixture consumer",
        ])
        result = self.hook("pre-push")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "resolves under --locked; 1 first-party git sources",
            result.stdout + result.stderr,
        )
        self.assertEqual((self.root / "Cargo.lock").read_bytes(), before)

        # A manifest/lock mismatch must be part of the checked revision: the
        # hook never reads the bare working tree, so the version bump is
        # committed (without regenerating the lock) before the push side is
        # exercised again.
        manifest.write_text(
            manifest.read_text(encoding="utf-8").replace('version = "0.1.0"', 'version = "0.2.0"'),
            encoding="utf-8",
        )
        self.run_command([self.git, "add", "Cargo.toml"])
        self.run_command([self.git, "commit", "-qm", "Bump the consumer version"])
        result = self.hook("pre-push")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("locked dependency hydration failed", result.stderr)
        self.assertIn("lock file", result.stderr)
        self.assertEqual((self.root / "Cargo.lock").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
