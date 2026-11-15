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

import contextlib
import hashlib
import http.client
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import traceback
import unittest
import urllib.error
import urllib.request
from collections.abc import Mapping
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
ATLAS_RUNTIME_REVISION = "d59e295ac41dffc34084126583b259dfa49a3117"
ATLAS_RUNTIME_FILES = {
    "process_tree.py": "f03b2d94caced534ce6dab89b040a9efb4c89d87",
    "windows_process.py": "3072bd44412a463ed5eb5a11f3cab5f7170ddcd3",
    "darwin_process.py": "60f0ff8f7dedbdfc22d9d5f9fcb0d601322434f1",
}

ATLAS_RUNTIME_BASE = (
    "https://api.github.com/repos/ryancinsight/atlas/contents/scripts"
)


class AtlasRuntimeFetchError(RuntimeError):
    """The pinned Atlas runtime could not be fetched from the hosting API."""


def atlas_runtime_token(environment: Mapping[str, str] = os.environ) -> str | None:
    """Return the hosting token `environment` provides, if any."""

    return environment.get("GITHUB_TOKEN") or environment.get("GH_TOKEN") or None


# Characters an Authorization token cannot carry: C0 controls other than tab,
# DEL, and anything outside latin-1, the encoding `http.client.putheader`
# applies. That is a superset of what `putheader` rejects (a CR or LF, or a
# non-latin-1 character); a test checks it against the running interpreter, so
# no header error from `http.client` can echo a token that passed.
_UNCARRIABLE_TOKEN_CHARACTER = re.compile(r"[\x00-\x08\x0a-\x1f\x7f]|[^\x00-\xff]")


def fetch_atlas_runtime_file(
    name: str, token: str | None, urlopen=urllib.request.urlopen
) -> bytes:
    """Fetch one runtime file at `ATLAS_RUNTIME_REVISION`.

    Hosted CI does not call this: it checks the pinned revision out and sets
    `ATLAS_HOOK_RUNTIME`. A local run without a checkout reaches the hosting
    API, which admits 60 unauthenticated requests per hour per client address;
    a token raises the quota to the token's own.

    Every failure raises `AtlasRuntimeFetchError` naming the file, the revision
    and the authentication state: a token an HTTP header cannot carry (refused
    before any request is built), an HTTP status, a connection or timeout error
    (`OSError`, which covers `URLError`), a truncated or malformed response
    (`http.client.HTTPException`), and a malformed redirect `Location` or an
    unencodable header (`ValueError`, which covers `UnicodeError`).

    No error this function raises carries the token, in its message or its
    chain: the token is validated first, so no `http.client` header error
    echoes it, and the messages are built from the file, the revision, the
    state and the exception's own text. Text a server chooses (a status reason,
    a `Location`) is embedded as received; a server holds the token already.
    The token is an unredirected header: `redirect_request` builds a redirect's
    headers from `req.headers` only (CPython 3.13.12
    `Lib/urllib/request.py:650-651`), so a redirect target never receives it.
    """

    state = "authenticated" if token else "unauthenticated (set GITHUB_TOKEN)"
    if token and _UNCARRIABLE_TOKEN_CHARACTER.search(token):
        raise AtlasRuntimeFetchError(
            f"Atlas hook runtime {name} at {ATLAS_RUNTIME_REVISION}: the token "
            "(GITHUB_TOKEN or GH_TOKEN) holds a character an HTTP header "
            f"cannot carry, {state}, no request sent"
        )
    request = urllib.request.Request(
        f"{ATLAS_RUNTIME_BASE}/{name}?ref={ATLAS_RUNTIME_REVISION}",
        headers={
            "Accept": "application/vnd.github.raw+json",
            "User-Agent": "coeus-hook-tests",
        },
    )
    if token:
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    try:
        with urlopen(request, timeout=30) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        error.close()
        failure = f"HTTP {error.code} {error.reason}"
        raise AtlasRuntimeFetchError(
            f"Atlas hook runtime {name} at {ATLAS_RUNTIME_REVISION}: "
            f"{failure}, {state}"
        ) from error
    except (OSError, http.client.HTTPException, ValueError) as error:
        failure = (
            error.reason
            if isinstance(error, urllib.error.URLError)
            else f"{type(error).__name__}: {error}"
        )
        raise AtlasRuntimeFetchError(
            f"Atlas hook runtime {name} at {ATLAS_RUNTIME_REVISION}: "
            f"{failure}, {state}"
        ) from error


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
# (arguments, working directory, every lock in a `--check-tree`, its immutable
# source, and the skip variable it inherited) to `LOCKFILE_CALLS_LOG`, then
# answers by
# `LOCKFILE_STUB`: `pass` (default), `fail`, or `cargo` -- `--check` then runs
# real `cargo metadata --locked` on the manifest and forwards its stderr and
# status, as the stack checker's resolution step does, so a stale lock is
# refused with cargo's own diagnostic; `--check-staged` runs no cargo.
STACK_CHECKER = '''\
import json, os, pathlib, subprocess, sys

arguments = sys.argv[1:]
lock = None
manifest = None
locks = {}
tree_source = None
if "--check-tree" in arguments:
    tree = pathlib.Path(arguments[arguments.index("--check-tree") + 1])
    manifest = tree / "Cargo.toml"
    locks = {
        path.relative_to(tree).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(tree.rglob("Cargo.lock"))
    }
    lock = locks.get("Cargo.lock")
elif "--manifest-path" in arguments:
    manifest = pathlib.Path(arguments[arguments.index("--manifest-path") + 1])
    beside = manifest.with_name("Cargo.lock")
    lock = beside.read_text(encoding="utf-8") if beside.is_file() else None
if "--tree-source" in arguments:
    source_index = arguments.index("--tree-source")
    tree_source = arguments[source_index + 1:source_index + 3]
record = {
    "arguments": arguments,
    "cwd": os.getcwd(),
    "lock": lock,
    "locks": locks,
    "skip": os.environ.get("SKIP_LOCKFILE_CHECK"),
    "tree_source": tree_source,
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


class AtlasRuntimeFetchTests(unittest.TestCase):
    """The runtime fetch authenticates when it can and fails closed when not."""

    class _Response:
        def __init__(self, body: bytes) -> None:
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *exception) -> None:
            return None

        def read(self) -> bytes:
            return self.body

    def _recording_opener(self, body: bytes):
        seen = []

        def opener(request, timeout):
            seen.append((request, timeout))
            return self._Response(body)

        return opener, seen

    def test_token_is_sent_as_bearer_authorization_for_the_pinned_revision(
        self,
    ) -> None:
        opener, seen = self._recording_opener(b"payload")
        body = fetch_atlas_runtime_file("process_tree.py", "secret-token", opener)
        self.assertEqual(body, b"payload")
        ((request, timeout),) = seen
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-token")
        self.assertEqual(
            request.full_url,
            f"{ATLAS_RUNTIME_BASE}/process_tree.py?ref={ATLAS_RUNTIME_REVISION}",
        )
        self.assertEqual(timeout, 30)

    def test_absent_token_sends_no_authorization_header(self) -> None:
        opener, seen = self._recording_opener(b"payload")
        fetch_atlas_runtime_file("process_tree.py", None, opener)
        self.assertIsNone(seen[0][0].get_header("Authorization"))

    def test_quota_refusal_is_a_typed_error_naming_file_and_auth_state(self) -> None:
        def refuse(request, timeout):
            raise urllib.error.HTTPError(
                request.full_url, 403, "rate limit exceeded", {}, io.BytesIO()
            )

        with self.assertRaises(AtlasRuntimeFetchError) as raised:
            fetch_atlas_runtime_file("windows_process.py", None, refuse)
        message = str(raised.exception)
        self.assertIn("windows_process.py", message)
        self.assertIn(ATLAS_RUNTIME_REVISION, message)
        self.assertIn("HTTP 403 rate limit exceeded", message)
        self.assertIn("unauthenticated", message)

    def test_refusal_closes_the_error_response_body(self) -> None:
        body = io.BytesIO(b"rate limit exceeded")

        def refuse(request, timeout):
            raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, body)

        # `raised` keeps the error chain, and so the response body, alive:
        # the body is closed by the fetch, not by collecting the chain.
        with self.assertRaises(AtlasRuntimeFetchError) as raised:
            fetch_atlas_runtime_file("process_tree.py", None, refuse)
        self.assertIsInstance(raised.exception.__cause__, urllib.error.HTTPError)
        self.assertTrue(body.closed)

    def test_error_never_echoes_the_token(self) -> None:
        def refuse(request, timeout):
            raise urllib.error.HTTPError(
                request.full_url, 401, "Bad credentials", {}, io.BytesIO()
            )

        with self.assertRaises(AtlasRuntimeFetchError) as raised:
            fetch_atlas_runtime_file("process_tree.py", "secret-token", refuse)
        message = str(raised.exception)
        self.assertNotIn("secret-token", message)
        self.assertIn("authenticated", message)
        self.assertNotIn("unauthenticated", message)

    def test_network_failure_is_a_typed_error(self) -> None:
        def unreachable(request, timeout):
            raise urllib.error.URLError("name resolution failed")

        with self.assertRaises(AtlasRuntimeFetchError) as raised:
            fetch_atlas_runtime_file("darwin_process.py", "t", unreachable)
        self.assertIn("name resolution failed", str(raised.exception))

    @contextlib.contextmanager
    def _local_server(self, reply: bytes | None):
        """Serve one connection on loopback; `None` never answers it.

        Yields the port and the request texts the server read. The connection
        stays open until the context exits, so a `None` reply makes the
        client's wait time out. Exit connects to the listener itself, so a
        client that never connected does not hold the test for the accept
        timeout.
        """

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        # Backstop only: it ends the thread should the exit connection fail.
        listener.settimeout(30)
        release = threading.Event()
        received: list[str] = []

        def serve() -> None:
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                return
            with connection:
                try:
                    request = connection.recv(65536)
                    if request:
                        received.append(request.decode("latin-1"))
                    if reply is None:
                        release.wait(timeout=30)
                    else:
                        connection.sendall(reply)
                except OSError:
                    return

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        try:
            yield port, received
        finally:
            release.set()
            if thread.is_alive():
                with contextlib.suppress(OSError):
                    socket.create_connection(("127.0.0.1", port), timeout=5).close()
            thread.join(timeout=30)
            listener.close()

    def _loopback_opener(self, port: int, wait: float):
        """Send the fetch's own request, headers kept as built, to loopback."""

        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def open_loopback(request, timeout):
            local = urllib.request.Request(
                f"http://127.0.0.1:{port}/", headers=request.headers
            )
            for key, value in request.unredirected_hdrs.items():
                local.add_unredirected_header(key, value)
            return opener.open(local, timeout=wait)

        return open_loopback

    def _assert_token_absent(self, error: BaseException, token: str) -> None:
        """The token is absent from the message and the whole rendered chain."""

        self.assertNotIn(token, str(error))
        self.assertNotIn(token, "".join(traceback.format_exception(error)))

    def _assert_typed_failure(self, reply, cause: type[BaseException]) -> None:
        with self._local_server(reply) as (port, _):
            with self.assertRaises(AtlasRuntimeFetchError) as raised:
                fetch_atlas_runtime_file(
                    "process_tree.py", None, self._loopback_opener(port, 0.5)
                )
        message = str(raised.exception)
        self.assertIn("process_tree.py", message)
        self.assertIn(ATLAS_RUNTIME_REVISION, message)
        self.assertIn(cause.__name__, message)
        self.assertIn("unauthenticated", message)
        self.assertIsInstance(raised.exception.__cause__, cause)

    def test_unanswered_request_times_out_as_a_typed_error(self) -> None:
        self._assert_typed_failure(None, TimeoutError)

    def test_connection_closed_without_a_response_is_a_typed_error(self) -> None:
        self._assert_typed_failure(b"", http.client.RemoteDisconnected)

    def test_truncated_body_is_a_typed_error(self) -> None:
        reply = b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nshort"
        self._assert_typed_failure(reply, http.client.IncompleteRead)

    def test_token_a_header_cannot_carry_is_refused_before_any_request(
        self,
    ) -> None:
        stem = "fake-secret-stem"
        # CR, LF, NUL and DEL, then characters outside latin-1.
        for suffix in ("\r\n", "\r", "\n", "\x00", "\x7f", "’", "\U0001f512"):
            with self.subTest(suffix=repr(suffix)):
                with self._local_server(b"") as (port, received):
                    with self.assertRaises(AtlasRuntimeFetchError) as raised:
                        fetch_atlas_runtime_file(
                            "process_tree.py",
                            stem + suffix,
                            self._loopback_opener(port, 0.5),
                        )
                error = raised.exception
                for part in (
                    "process_tree.py",
                    ATLAS_RUNTIME_REVISION,
                    "a character an HTTP header cannot carry",
                    "authenticated",
                ):
                    self.assertIn(part, str(error))
                self.assertEqual(received, [])
                self.assertIsNone(error.__cause__)
                self.assertIsNone(error.__context__)
                self._assert_token_absent(error, stem)

    def test_token_validation_rejects_everything_http_client_rejects(self) -> None:
        connection = http.client.HTTPConnection("127.0.0.1")
        connection.putrequest("GET", "/")
        for codepoint in (*range(0x300), 0x2019, 0xD800, 0xFFFF, 0x10FFFF):
            token = f"a{chr(codepoint)}b"
            if not _UNCARRIABLE_TOKEN_CHARACTER.search(token):
                with self.subTest(codepoint=hex(codepoint)):
                    connection.putheader("Authorization", f"Bearer {token}")
        for carried in ("\xe9", "\xff", "\t", " "):
            self.assertIsNone(_UNCARRIABLE_TOKEN_CHARACTER.search(carried))

    def test_malformed_redirect_location_is_a_typed_error(self) -> None:
        token = "fake-redirect-secret"
        reply = (
            b"HTTP/1.1 302 Found\r\nLocation: http://[::1/x\r\n"
            b"Content-Length: 0\r\n\r\n"
        )
        with self._local_server(reply) as (port, received):
            with self.assertRaises(AtlasRuntimeFetchError) as raised:
                fetch_atlas_runtime_file(
                    "process_tree.py", token, self._loopback_opener(port, 2)
                )
        error = raised.exception
        self.assertIn("ValueError", str(error))
        self.assertIn(ATLAS_RUNTIME_REVISION, str(error))
        self.assertIsInstance(error.__cause__, ValueError)
        # The first hop carried the token, so its absence from the error is
        # evidence about the error and not about a request never made.
        self.assertIn(f"authorization: bearer {token}", received[0].lower())
        self._assert_token_absent(error, token)

    def test_redirect_target_never_receives_the_token(self) -> None:
        token = "fake-forwarded-secret"
        final = (
            b"HTTP/1.1 200 OK\r\nContent-Length: 7\r\n"
            b"Connection: close\r\n\r\npayload"
        )
        with self._local_server(final) as (target_port, target_received):
            redirect = (
                "HTTP/1.1 302 Found\r\n"
                f"Location: http://127.0.0.1:{target_port}/next\r\n"
                "Content-Length: 0\r\nConnection: close\r\n\r\n"
            ).encode("ascii")
            with self._local_server(redirect) as (origin_port, origin_received):
                body = fetch_atlas_runtime_file(
                    "process_tree.py", token, self._loopback_opener(origin_port, 2)
                )
        self.assertEqual(body, b"payload")
        self.assertIn(f"authorization: bearer {token}", origin_received[0].lower())
        self.assertEqual(len(target_received), 1)
        self.assertNotIn("authorization", target_received[0].lower())

    def test_environment_token_prefers_github_token_over_gh_token(self) -> None:
        self.assertEqual(
            atlas_runtime_token({"GITHUB_TOKEN": "a", "GH_TOKEN": "b"}), "a"
        )
        self.assertEqual(atlas_runtime_token({"GH_TOKEN": "b"}), "b")
        self.assertEqual(
            atlas_runtime_token({"GITHUB_TOKEN": "", "GH_TOKEN": "b"}), "b"
        )
        self.assertIsNone(atlas_runtime_token({}))

    def test_workflow_checks_out_the_revision_the_tests_pin(self) -> None:
        workflow = (REPOSITORY / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            f"ATLAS_RUNTIME_REVISION: {ATLAS_RUNTIME_REVISION}\n", workflow
        )


class LockHookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        configured = os.environ.get("ATLAS_HOOK_RUNTIME")
        if configured:
            cls.runtime = Path(configured).resolve()
        else:
            directory = tempfile.TemporaryDirectory(prefix="coeus-hook-runtime-")
            cls.addClassCleanup(directory.cleanup)
            cls.runtime = Path(directory.name)
            token = atlas_runtime_token()
            for name in ATLAS_RUNTIME_FILES:
                (cls.runtime / name).write_bytes(
                    fetch_atlas_runtime_file(name, token)
                )
        missing = [
            name
            for name in ATLAS_RUNTIME_FILES
            if not (cls.runtime / name).is_file()
        ]
        if missing:
            raise RuntimeError(f"Atlas hook runtime is missing {', '.join(missing)}")
        for name, expected in ATLAS_RUNTIME_FILES.items():
            content = (cls.runtime / name).read_bytes().replace(b"\r\n", b"\n")
            header = f"blob {len(content)}\0".encode("ascii")
            observed = hashlib.sha1(
                header + content, usedforsecurity=False
            ).hexdigest()
            if observed != expected:
                raise RuntimeError(
                    f"Atlas hook runtime {name} is {observed}, expected {expected}"
                )
        probe_code = (
            "import sys;"
            f"sys.path.insert(0, {str(cls.runtime)!r});"
            "import process_tree;"
            "result=process_tree.run("
            "[sys.executable,'-c',\"print('coeus-runtime')\"],timeout=30);"
            "sys.stdout.buffer.write(result.stdout or b'');"
            "sys.stderr.buffer.write(result.stderr or b'');"
            "raise SystemExit(result.returncode)"
        )
        probe = subprocess.run(
            [sys.executable, "-I", "-c", probe_code],
            capture_output=True,
            timeout=30,
            check=False,
        )
        output = probe.stdout.replace(b"\r\n", b"\n")
        if probe.returncode != 0 or output != b"coeus-runtime\n":
            raise RuntimeError(
                "Atlas process-tree runtime probe failed: "
                f"exit {probe.returncode}, stdout {probe.stdout!r}, "
                f"stderr {probe.stderr!r}"
            )

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
        # The hook, git and cargo processes these fixtures spawn never need the
        # hosting credential. Other test modules' children are not covered.
        for variable in (
            "SKIP_LOCKFILE_CHECK", "LOCKFILE_STUB", "GITHUB_TOKEN", "GH_TOKEN"
        ):
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
            for name in ATLAS_RUNTIME_FILES:
                shutil.copyfile(
                    self.runtime / name, self.stack / "scripts" / name
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

    def assert_pushed_tree(self, call: dict, member: Path | None = None) -> None:
        member = self.member if member is None else member
        arguments = call["arguments"]
        self.assertEqual(arguments[:2], ["--check", "--check-tree"])
        exported = Path(arguments[2])
        self.assertEqual(Path(call["cwd"]).resolve(), exported.resolve())
        self.assertFalse(
            exported.resolve().is_relative_to(member.resolve()),
            "the checked tree is the export's, never the checkout's",
        )
        source_repository, source_commit = call["tree_source"]
        self.assertEqual(Path(source_repository).resolve(), member.resolve())
        expected = self.run_git(member, "rev-parse", "HEAD").stdout.strip()
        self.assertEqual(source_commit, expected)

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
        nested = self.member / "nested"
        nested.mkdir()
        (nested / "Cargo.lock").write_text("# nested lock\n", encoding="utf-8")
        self.commit_all(self.member, "Add a nested lock owner")
        # The working lock is not the push: a peer's tree, or a cargo run inside
        # a stack, leaves it flattened while the pushed commit stays sound.
        (self.member / "Cargo.lock").write_text("# flattened\n", encoding="utf-8")

        result = self.hook("pre-push")

        self.assertEqual(result.returncode, 0, result.stderr)
        (call,) = self.calls()
        self.assert_pushed_tree(call)
        self.assertEqual(call["lock"], LOCK, "the checker read the pushed lock")
        self.assertEqual(
            call["locks"],
            {"Cargo.lock": LOCK, "nested/Cargo.lock": "# nested lock\n"},
        )
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
        self.assert_pushed_tree(pushed)
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
