"""Execute the installed lock hooks against real Git and Cargo fixtures.

The hooks are the stack's single owned copies. Their lock stage does not run a
member checker: it extracts `scripts/lockfile.py` from the fetched default
(`origin/HEAD`, else `origin/main`) of the stack this checkout belongs to --
the nearest ancestor that registers its members under `repos/` in its
`.gitmodules` and has one of them share this checkout's object store -- and
runs that. Outside a stack checkout there is no checker to run, and the stage
defers to the `lockfile-guard` CI job; a push from there is still refused,
since the trusted credential scanner is just as unreachable and a credential
cannot be taken back once pushed.

The fixtures therefore build a stack (`stack/`, one commit carrying
`.gitmodules` and `scripts/`, published as `refs/remotes/origin/main`) with
the member at `stack/repos/member`, and a clone outside any stack (`alone/`).
The checker the fixture stack carries is a recording stand-in
(`STACK_CHECKER`), and its credential scanner a clean-range stand-in
(`SECRET_SCANNER`) so the stage that refuses an unscanned push does not decide
the lock verdicts these tests judge. What these tests judge is how the hooks
locate, invoke, and obey a checker, and what that checker's verdicts are is
judged by the checker's own tests (atlas `scripts/tests/test_lockfile_*.py`).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


# Every external command the hooks run before and inside their lock stage,
# including the stack-tool extraction that precedes it (`mkdir`, `mv`, `touch`).
# The missing-interpreter fixture must keep these on a stripped PATH: a hook
# that fails earlier, on a missing `seq`/`mktemp`/`mv`, never reaches the
# interpreter search and the test would judge the wrong refusal.
_HOOK_COMMANDS = (
    "bash", "git", "seq", "sed", "grep", "awk", "cat", "head", "sort",
    "tr", "mktemp", "rm", "tar", "mkdir", "mv", "touch",
)

REPOSITORY = Path(__file__).resolve().parents[2]
HOOKS = ("pre-commit", "pre-push")
ZERO = "0" * 40

# A first-party git source in the lock, and a manifest declaring the matching
# dependency: the fixture member is the shape the checker's flattened-lock
# diagnosis applies to.
LOCK = (
    '[[package]]\nname = "provider"\nversion = "0.1.0"\n'
    'source = "git+https://github.com/ryancinsight/provider.git?branch=main#abc123"\n'
)
MANIFEST = (
    '[package]\nname = "member"\nversion = "0.1.0"\n\n[dependencies]\n'
    'provider = { git = "https://github.com/ryancinsight/provider", version = "0.1" }\n'
)

# The fixture stack's `scripts/lockfile.py`. It appends one JSON record per call
# (arguments, working directory, the `Cargo.lock` beside a `--manifest-path`,
# and the skip variable it inherited) to `LOCKFILE_CALLS_LOG`, then answers by
# `LOCKFILE_STUB`: `pass` (default), `fail`, or `cargo` -- `--check` then runs
# real `cargo metadata --locked` on the manifest and forwards its stderr and
# status, as the stack checker's resolution step does, so a stale lock is
# refused with cargo's own diagnostic; `--check-staged` runs no cargo.
STACK_CHECKER = '''\
import json, os, pathlib, subprocess, sys

arguments = sys.argv[1:]
lock = None
manifest = None
if "--manifest-path" in arguments:
    manifest = pathlib.Path(arguments[arguments.index("--manifest-path") + 1])
    beside = manifest.with_name("Cargo.lock")
    lock = beside.read_text(encoding="utf-8") if beside.is_file() else None
record = {
    "arguments": arguments,
    "cwd": os.getcwd(),
    "lock": lock,
    "skip": os.environ.get("SKIP_LOCKFILE_CHECK"),
}
with open(os.environ["LOCKFILE_CALLS_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps(record) + "\\n")

mode = os.environ.get("LOCKFILE_STUB", "pass")
if mode == "fail":
    print("stack checker: refused", file=sys.stderr)
    sys.exit(1)
if mode == "cargo" and "--check" in arguments:
    resolved = subprocess.run(
        ["cargo", "metadata", "--locked", "--format-version", "1",
         "--manifest-path", str(manifest)],
        capture_output=True, encoding="utf-8", errors="replace", check=False,
    )
    sys.stderr.write(resolved.stderr)
    sys.exit(resolved.returncode)
sys.exit(0)
'''

# The credential scanner a fixture stack publishes when the tests do not judge
# it: `pre-push` refuses a push it could not scan, so every stack that reaches
# that stage carries one. This stand-in reports a clean range.
SECRET_SCANNER = "import sys\nsys.exit(0)\n"


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
        # The synthetic stack must not sit inside the Coeus repository: the
        # hooks reject a candidate stack nested in another repository because
        # it may be a member carrying a misleading `.gitmodules`. Keep the
        # Windows path long-form for Cargo's Git cache while placing the
        # fixture under the operating system temporary root, as Atlas's owner
        # hook tests do.
        fixture_root = None
        if os.name == "nt":
            local_app_data = Path(
                os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
            )
            fixture_root = local_app_data / "Temp"
        self.directory = tempfile.TemporaryDirectory(
            prefix="coeus-hooks-", dir=fixture_root
        )
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.stack = self.root / "stack"
        self.member = self.stack / "repos" / "member"
        self.alone = self.root / "alone"
        self.calls_log = self.root / "checker-calls.jsonl"
        self.environment = os.environ.copy()
        for variable in ("SKIP_LOCKFILE_CHECK", "LOCKFILE_STUB"):
            self.environment.pop(variable, None)
        self.environment["PYTHON"] = Path(sys.executable).as_posix()
        self.environment["GIT_CONFIG_NOSYSTEM"] = "1"
        self.environment["GIT_CONFIG_GLOBAL"] = str(self.root / "gitconfig")
        # `pre-push` reaches its code gate only after the lock stage; the gate
        # is not under test, so the lock verdict is the hook's exit status.
        self.environment["SKIP_LOCAL_GATE"] = "1"
        self.environment["LOCKFILE_CALLS_LOG"] = str(self.calls_log)
        # The isolated global config above carries no identity, and CI runners
        # (unlike a developer machine) have none in any wider scope either.
        self.environment["GIT_AUTHOR_NAME"] = "Coeus tests"
        self.environment["GIT_AUTHOR_EMAIL"] = "tests@localhost"
        self.environment["GIT_COMMITTER_NAME"] = "Coeus tests"
        self.environment["GIT_COMMITTER_EMAIL"] = "tests@localhost"
        self.git = shutil.which("git")
        self.assertIsNotNone(self.git, "hook tests require Git")
        self.bash = shutil.which("bash")
        if os.name == "nt":
            executable_root = Path(
                self.run_command([self.git, "--exec-path"], cwd=self.root).stdout.strip()
            )
            self.bash = str(executable_root.parents[2] / "bin" / "bash.exe")
        self.assertIsNotNone(self.bash, "hook tests require Bash")

    def run_command(self, command, *, cwd, environment=None, expected=0, input_text=None):
        env = self.environment if environment is None else environment
        if input_text is None:
            result = subprocess.run(
                command, cwd=cwd, env=env,
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=60, check=False,
            )
        else:
            # Bytes, not text: a text-mode pipe on Windows translates `\n` to
            # `\r\n` while writing the child's stdin, which corrupts the
            # push-line protocol `pre-push` parses byte-for-byte -- a `\r`
            # riding along in `remote_sha` makes it compare unequal to the
            # all-zeros sentinel, so the "new branch" case is never taken.
            raw = subprocess.run(
                command, cwd=cwd, env=env,
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

    def run_git(self, repository, *arguments, **options):
        return self.run_command([self.git, *arguments], cwd=repository, **options)

    def publish_stack(self, checker: str | None) -> None:
        """Make a one-commit stack whose fetched default carries `checker`.

        `None` publishes a default whose `scripts/` lacks `lockfile.py`, as a
        stack default cut before the checker existed does. The `.gitmodules`
        registering `repos/member` is what makes this directory a stack to the
        hooks (stacks are found by that identity, not by depth), and the
        credential scanner stand-in is what carries a push past the stage that
        refuses an unscanned one.
        """
        self.run_git(self.stack.parent, "init", "-q", str(self.stack))
        (self.stack / "scripts").mkdir()
        (self.stack / ".gitmodules").write_text(
            '[submodule "member"]\n\tpath = repos/member\n\turl = ./member\n',
            encoding="utf-8", newline="\n",
        )
        (self.stack / "scripts" / "atlas-secret-scan.py").write_text(
            SECRET_SCANNER, encoding="utf-8", newline="\n"
        )
        if checker is None:
            (self.stack / "scripts" / "other.py").write_text("pass\n", encoding="utf-8")
        else:
            (self.stack / "scripts" / "lockfile.py").write_text(
                checker, encoding="utf-8", newline="\n"
            )
        self.run_git(self.stack, "add", ".gitmodules", "scripts")
        self.run_git(self.stack, "commit", "-qm", "Publish the stack scripts")
        head = self.run_git(self.stack, "rev-parse", "HEAD").stdout.strip()
        self.run_git(self.stack, "update-ref", "refs/remotes/origin/main", head)

    def make_member(self, path: Path) -> Path:
        """A member repository carrying the installed hooks and no lock checker."""
        path.mkdir(parents=True)
        self.run_git(path, "init", "-q")
        (path / ".githooks").mkdir()
        for hook in HOOKS:
            destination = path / ".githooks" / hook
            shutil.copyfile(REPOSITORY / ".githooks" / hook, destination)
            destination.chmod(0o755)
        (path / "Cargo.toml").write_text(MANIFEST, encoding="utf-8", newline="\n")
        self.commit_all(path, "Seed the member")
        # The member's fetched default: `pre-push` measures its range against
        # it, and the secret scan's trusted allowlist reads its revision. It
        # stops just before the lock, so the lock-seeding commit is always
        # inside the pushed range and the lock stage is never skipped as
        # touching no dependency state (a range that changes no manifest and
        # no lock cannot make the lock worse, and is left alone).
        head = self.run_git(path, "rev-parse", "HEAD").stdout.strip()
        self.run_git(path, "update-ref", "refs/remotes/origin/main", head)
        (path / "Cargo.lock").write_text(LOCK, encoding="utf-8", newline="\n")
        self.commit_all(path, "Seed the lock")
        return path

    def commit_all(self, repository: Path, message: str) -> None:
        self.run_git(repository, "add", "-A")
        self.run_git(repository, "commit", "-qm", message)

    def stage_lock(self, text: str) -> None:
        (self.member / "Cargo.lock").write_text(text, encoding="utf-8", newline="\n")
        self.run_git(self.member, "add", "Cargo.lock")

    def hook(self, name, *, member=None, **variables):
        member = self.member if member is None else member
        environment = self.environment | variables
        # `pre-push` alone reads a push range from stdin. The member's fetched
        # default stops just before its lock (see `make_member`), so the pushed
        # range carries dependency state and is judged as a range -- never
        # `HEAD` in its place.
        input_text = None
        if name == "pre-push":
            head = self.run_git(member, "rev-parse", "HEAD").stdout.strip()
            input_text = f"refs/heads/probe {head} refs/heads/probe {ZERO}\n"
        return self.run_command(
            [self.bash, "--noprofile", "--norc", f".githooks/{name}"],
            cwd=member, environment=environment, expected=None, input_text=input_text,
        )

    def prepare(self, hook: str) -> None:
        """Put the member where `hook` judges its lock.

        `pre-commit` judges the index and returns before any checker work when
        no `Cargo.lock` is staged; `pre-push` judges the pushed range, which
        the lock-seeding commit already carries.
        """
        if hook == "pre-commit":
            self.stage_lock(LOCK + "# touched\n")

    def calls(self) -> list[dict]:
        if not self.calls_log.is_file():
            return []
        lines = self.calls_log.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines]

    def test_pre_commit_runs_the_stack_checker_on_the_index(self) -> None:
        self.publish_stack(STACK_CHECKER)
        self.make_member(self.member)
        self.assertFalse((self.member / "scripts").exists(), "the member carries no checker")
        self.prepare("pre-commit")

        result = self.hook("pre-commit")

        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual([call["arguments"] for call in calls], [["--check-staged"]])
        self.assertEqual(Path(calls[0]["cwd"]).resolve(), self.member.resolve())

    def test_pre_commit_without_a_staged_lock_never_reaches_the_stack(self) -> None:
        # The stack default lacks the checker, and the commit still passes: the
        # staged-path test runs before the stack is touched.
        self.publish_stack(None)
        self.make_member(self.member)
        (self.member / "notes.txt").write_text("x\n", encoding="utf-8")
        self.run_git(self.member, "add", "notes.txt")

        result = self.hook("pre-commit")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls(), [])

    def test_pre_push_checks_an_export_of_the_pushed_revision(self) -> None:
        self.publish_stack(STACK_CHECKER)
        self.make_member(self.member)
        # The working lock is not the push: a peer's tree, or a cargo run inside
        # a stack, leaves it flattened while the pushed commit stays sound.
        (self.member / "Cargo.lock").write_text("# flattened\n", encoding="utf-8")

        result = self.hook("pre-push")

        self.assertEqual(result.returncode, 0, result.stderr)
        (call,) = self.calls()
        self.assertEqual(call["arguments"][:2], ["--check", "--manifest-path"])
        manifest = Path(call["arguments"][2])
        self.assertEqual(manifest.name, "Cargo.toml")
        self.assertEqual(Path(call["cwd"]).resolve(), manifest.parent.resolve())
        self.assertFalse(
            manifest.resolve().is_relative_to(self.member.resolve()),
            "the manifest is the export's, never the checkout's",
        )
        self.assertEqual(call["lock"], LOCK, "the checker read the pushed lock")
        self.assertEqual(
            (self.member / "Cargo.lock").read_text(encoding="utf-8"), "# flattened\n"
        )

    def test_a_member_copy_is_never_the_checker_that_runs(self) -> None:
        # A copy that accepts every lock cannot overrule the stack's verdict,
        # and it is never executed: it leaves a witness if it ever is.
        self.publish_stack(STACK_CHECKER)
        self.make_member(self.member)
        (self.member / "scripts").mkdir()
        witness = self.member / "member-copy-ran"
        (self.member / "scripts" / "lockfile.py").write_text(
            f"import pathlib, sys\npathlib.Path({witness.as_posix()!r}).write_text('ran')\n"
            "sys.exit(0)\n",
            encoding="utf-8",
        )
        self.commit_all(self.member, "Add a member copy")
        for hook in HOOKS:
            with self.subTest(hook=hook):
                self.prepare(hook)
                result = self.hook(hook, LOCKFILE_STUB="fail")
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("stack checker: refused", result.stderr)
        self.assertFalse(witness.exists())
        self.assertEqual(len(self.calls()), 2)

    def test_stack_default_without_the_checker_rejects_both_hooks(self) -> None:
        self.publish_stack(None)
        self.make_member(self.member)
        for hook in HOOKS:
            with self.subTest(hook=hook):
                self.prepare(hook)
                result = self.hook(hook)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn(f"{hook}: the stack's scripts at ", result.stderr)
                self.assertIn("carry no lockfile.py; lockfile not verified", result.stderr)
        self.assertEqual(self.calls(), [])

    def _path_without_python(self) -> str:
        """A PATH carrying the hook's own commands and no python interpreter.

        The hook resolves its repository with git and runs coreutils before it
        searches for an interpreter, so an empty PATH (or one holding only
        `git`) tests the wrong failure: it fails on a missing `seq`/`mktemp`
        first, never reaching the interpreter search at all. Windows keeps its
        interpreter in its own directory outside the coreutils Git bundles, so
        filtering only python-holding entries out of the inherited PATH is
        correct there; POSIX needs the named commands linked into a fresh
        directory instead, since `/usr/bin` there also holds a system python
        (mirrors atlas's scripts/tests/test_atlas_pre_push_gate.py
        `_path_without_python`).
        """
        if os.name == "nt":
            entries = []
            for entry in os.environ.get("PATH", "").split(os.pathsep):
                if not entry or entry in entries:
                    continue
                probe = Path(entry)
                if any(
                    (probe / name).exists()
                    for name in ("python.exe", "python3.exe", "python", "python3")
                ):
                    continue
                entries.append(entry)
            return os.pathsep.join(entries)

        tools = self.root / "interpreter-free-tools"
        tools.mkdir(exist_ok=True)
        for name in _HOOK_COMMANDS:
            found = shutil.which(name)
            if found is None:
                continue
            link = tools / name
            if not link.exists():
                link.symlink_to(found)
        absent = [name for name in ("bash", "git") if not (tools / name).exists()]
        self.assertFalse(
            absent,
            f"interpreter-free PATH is missing {absent}; the fixture would "
            "test a missing shell rather than a missing interpreter",
        )
        return str(tools)

    def test_missing_interpreter_rejects_both_hooks(self) -> None:
        # Absolute Bash starts the actual hook; PATH carries its coreutils but
        # no python, and the stack checker is present and would pass. No
        # replacement interpreter can manufacture success.
        self.publish_stack(STACK_CHECKER)
        self.make_member(self.member)
        path = self._path_without_python()
        for hook in HOOKS:
            with self.subTest(hook=hook):
                self.prepare(hook)
                result = self.hook(hook, PYTHON="", PATH=path)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn(
                    f"{hook}: no python interpreter found; lockfile not verified",
                    result.stderr,
                )
        self.assertEqual(self.calls(), [])

    def test_explicit_missing_interpreter_rejects_both_hooks(self) -> None:
        self.publish_stack(STACK_CHECKER)
        self.make_member(self.member)
        interpreter = (self.root / "absent-python").as_posix()
        for hook in HOOKS:
            with self.subTest(hook=hook):
                self.prepare(hook)
                result = self.hook(hook, PYTHON=interpreter)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(interpreter, result.stderr)
        self.assertEqual(self.calls(), [])

    def test_skip_variable_cannot_hide_checker_failures(self) -> None:
        self.publish_stack(STACK_CHECKER)
        self.make_member(self.member)
        for hook in HOOKS:
            with self.subTest(hook=hook, checker="refuses"):
                self.prepare(hook)
                result = self.hook(hook, SKIP_LOCKFILE_CHECK="1", LOCKFILE_STUB="fail")
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn(f"{hook}: SKIP_LOCKFILE_CHECK is no longer honoured", result.stderr)
                self.assertIn("stack checker: refused", result.stderr)
        # The checker ran under the skip variable: it was not skipped.
        self.assertEqual([call["skip"] for call in self.calls()], ["1", "1"])

    def test_skip_variable_cannot_hide_an_absent_checker(self) -> None:
        self.publish_stack(None)
        self.make_member(self.member)
        for hook in HOOKS:
            with self.subTest(hook=hook):
                self.prepare(hook)
                result = self.hook(hook, SKIP_LOCKFILE_CHECK="1")
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn(f"{hook}: SKIP_LOCKFILE_CHECK is no longer honoured", result.stderr)
                self.assertIn("carry no lockfile.py; lockfile not verified", result.stderr)

    def test_real_lock_passes_and_stale_lock_retains_cargo_diagnostic(self) -> None:
        dependency = self.root / "dependency"
        (dependency / "src").mkdir(parents=True)
        (dependency / "src/lib.rs").write_text("//! Hook fixture dependency.\n", encoding="utf-8")
        (dependency / "Cargo.toml").write_text(
            '[package]\nname = "hook-dependency"\nversion = "0.1.0"\nedition = "2024"\n',
            encoding="utf-8",
        )
        self.run_git(dependency, "init", "-q")
        self.run_git(dependency, "add", "Cargo.toml", "src/lib.rs")
        self.run_git(dependency, "commit", "-qm", "Add hook fixture")
        url = "https://github.com/ryancinsight/hook-fixture"
        # Git resolves this first-party-shaped source locally; Cargo still
        # generates and verifies the actual Git revision and dependency graph.
        self.run_git(self.root, "config", "--global", f"url.{dependency.as_uri()}.insteadOf", url)
        self.environment["CARGO_NET_GIT_FETCH_WITH_CLI"] = "true"
        self.environment["CARGO_HOME"] = str(self.root / "cargo-home")
        self.environment["LOCKFILE_STUB"] = "cargo"

        self.publish_stack(STACK_CHECKER)
        self.make_member(self.member)
        (self.member / "src").mkdir()
        (self.member / "src/lib.rs").write_text("//! Hook fixture consumer.\n", encoding="utf-8")
        manifest = self.member / "Cargo.toml"
        manifest.write_text(
            '[package]\nname = "hook-consumer"\nversion = "0.1.0"\nedition = "2024"\n'
            '[workspace]\n'
            f'[dependencies]\nhook-dependency = {{ git = "{url}" }}\n',
            encoding="utf-8",
        )
        # A neutral working directory keeps the stack's cargo configuration
        # (and any above this checkout) out of the lock the fixture generates.
        self.run_command(
            ["cargo", "generate-lockfile", "--manifest-path", str(manifest)],
            cwd=Path(tempfile.gettempdir()),
        )
        before = (self.member / "Cargo.lock").read_bytes()
        self.assertIn(b"git+" + url.encode(), before)

        # `pre-commit` judges the staged blob; `pre-push` an export of the
        # committed revision, never the bare working tree.
        self.run_git(self.member, "add", "-A")
        result = self.hook("pre-commit")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.run_git(self.member, "commit", "-qm", "Add hook fixture consumer")
        result = self.hook("pre-push")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.member / "Cargo.lock").read_bytes(), before)
        pushed = self.calls()[-1]
        self.assertEqual(pushed["arguments"][:2], ["--check", "--manifest-path"])
        self.assertEqual(pushed["lock"].encode("utf-8"), before)

        # A manifest/lock mismatch must be part of the checked revision: the
        # hook never reads the bare working tree, so the version bump is
        # committed (without regenerating the lock) before pushing again.
        manifest.write_text(
            manifest.read_text(encoding="utf-8").replace('version = "0.1.0"', 'version = "0.2.0"'),
            encoding="utf-8",
        )
        self.run_git(self.member, "add", "Cargo.toml")
        self.run_git(self.member, "commit", "-qm", "Bump the consumer version")
        result = self.hook("pre-push")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("does not resolve under --locked", result.stderr)
        # Cargo's own diagnostic, forwarded by the checker and kept by the hook.
        self.assertIn("lock file", result.stderr)
        self.assertIn("--locked was passed", result.stderr)
        self.assertEqual((self.member / "Cargo.lock").read_bytes(), before)

    def test_outside_a_stack_the_lock_is_left_to_ci_and_a_push_is_refused(self) -> None:
        # No stack above this clone: there is no checker to run, and the one arm
        # that lets a lock through unchecked says so and names its enforcer.
        # `pre-push` says the same and is then refused: the trusted credential
        # scanner is just as unreachable, and a credential cannot be taken back
        # once pushed, so an unscanned range never leaves the machine.
        self.make_member(self.alone)
        (self.alone / "Cargo.lock").write_text("# flattened\n", encoding="utf-8")
        self.run_git(self.alone, "add", "Cargo.lock")
        for hook in HOOKS:
            with self.subTest(hook=hook):
                result = self.hook(hook, member=self.alone, LOCKFILE_STUB="fail")
                self.assertIn("no Atlas stack above this clone", result.stderr)
                self.assertIn("is not verified here", result.stderr)
                if hook == "pre-commit":
                    self.assertEqual(result.returncode, 0, result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0, result.stderr)
                    self.assertIn(
                        "credential scanner is not reachable from this clone",
                        result.stderr,
                    )
        self.assertEqual(self.calls(), [])
        self.assertEqual(self.calls(), [])


if __name__ == "__main__":
    unittest.main()
