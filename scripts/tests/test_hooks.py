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
import gc
import hashlib
import http.client
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
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple


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
# Largest body accepted: 1 MiB against the largest pinned file's 29,490 bytes.
ATLAS_RUNTIME_MAX_BYTES = 1 << 20
_READ_STEP = 1 << 16


class AtlasRuntimeFetchError(RuntimeError):
    """The pinned Atlas runtime could not be fetched from the hosting API."""


class AtlasRuntimeToken(NamedTuple):
    """A hosting token and the environment variable that held it."""

    variable: str
    value: str


def atlas_runtime_token(
    environment: Mapping[str, str] = os.environ,
) -> AtlasRuntimeToken | None:
    """Return the first non-blank `GITHUB_TOKEN`, then `GH_TOKEN`, in `environment`.

    A value that is empty or only whitespace, a space or a bare line ending
    included, counts as unset: it is skipped and never reaches a request header.
    """

    for variable in ("GITHUB_TOKEN", "GH_TOKEN"):
        value = environment.get(variable)
        if value is not None and value.strip():
            return AtlasRuntimeToken(variable, value)
    return None


# Characters an Authorization token cannot carry: C0 controls other than tab,
# DEL, and anything outside latin-1, the encoding `http.client.putheader`
# applies. A superset of what `putheader` rejects, checked by a test against
# the running interpreter, so no header error can echo a token that passed.
_UNCARRIABLE_TOKEN_CHARACTER = re.compile(r"[\x00-\x08\x0a-\x1f\x7f]|[^\x00-\xff]")


def _read_bounded(response) -> bytes:
    """Read a body in bounded steps, so no declared length sizes a read."""

    body = bytearray()
    while chunk := response.read(_READ_STEP):
        body += chunk
        if len(body) > ATLAS_RUNTIME_MAX_BYTES:
            raise ValueError(f"body exceeds {ATLAS_RUNTIME_MAX_BYTES} bytes")
    if response.length:
        raise http.client.IncompleteRead(bytes(body), response.length)
    return bytes(body)


def _close_unreturned_responses(error: BaseException) -> None:
    """Close each HTTP response still referenced by a frame `error` crossed.

    `urllib` leaves the first-hop response open when its redirect handler
    rejects a malformed `Location` with a `ValueError`: the caller never
    receives the response, so only the traceback's frames (`fp` of
    `http_error_302`, `response` of its callers) still reach its socket.
    """

    frame_trace = error.__traceback__
    while frame_trace is not None:
        for value in frame_trace.tb_frame.f_locals.values():
            if isinstance(value, http.client.HTTPResponse):
                value.close()
        frame_trace = frame_trace.tb_next


def fetch_atlas_runtime_file(
    name: str, token: AtlasRuntimeToken | None, urlopen=urllib.request.urlopen
) -> bytes:
    """Fetch one runtime file at `ATLAS_RUNTIME_REVISION`.

    Hosted CI checks the pinned revision out and never calls this. A local run
    reaches the hosting API, which admits 60 unauthenticated requests per hour
    per client address; a token raises the quota.

    Raises `AtlasRuntimeFetchError` naming file, revision and authentication
    state for: a token an HTTP header cannot carry, whose message also names
    the variable that held it (before any request); an
    HTTP status; an `OSError`; an `http.client.HTTPException`; a `ValueError`
    (malformed redirect `Location`, a body above `ATLAS_RUNTIME_MAX_BYTES`, a
    redirect `urllib` followed to a non-HTTP(S) URL such as ftp, refused
    unread). Other exceptions propagate. A malformed redirect `Location`
    leaves no socket open: `urllib` abandons the first-hop response, which
    the fetch closes through the exception's traceback. The body is read in
    64 KiB steps, so a
    declared length or chunk size never sizes a read: five oversized witnesses
    raise `OverflowError` or `MemoryError` from an unbounded read on CPython
    3.13.3.

    The token is validated first, so no `http.client` header error echoes it,
    and is an unredirected header: `urllib` copies it to no redirect target
    (tested for a redirect to another port). With a token set the message
    carries no text the server chose: it names file, revision, authentication
    state and the failure kind (the exception type name, or `HTTP` and the
    status code). `__cause__` is the original `ValueError`, scrubbed of its
    arguments, notes and chain, when the failure is exactly a `ValueError`, and
    `None` otherwise; its traceback names code lines only. Without a token the message carries the server text
    and the original cause. Not held: the traceback's frame locals hold the
    token, which a local-variable dump shows.
    """

    state = "authenticated" if token is not None else "unauthenticated (set GITHUB_TOKEN)"
    where = f"Atlas hook runtime {name} at {ATLAS_RUNTIME_REVISION}"
    if token is not None and _UNCARRIABLE_TOKEN_CHARACTER.search(token.value):
        raise AtlasRuntimeFetchError(
            f"{where}: the token in {token.variable} holds a character "
            f"an HTTP header cannot carry, {state}, no request sent"
        )
    request = urllib.request.Request(
        f"{ATLAS_RUNTIME_BASE}/{name}?ref={ATLAS_RUNTIME_REVISION}",
        headers={
            "Accept": "application/vnd.github.raw+json",
            "User-Agent": "coeus-hook-tests",
        },
    )
    if token is not None:
        request.add_unredirected_header("Authorization", f"Bearer {token.value}")
    try:
        with urlopen(request, timeout=30) as response:
            if not response.geturl().startswith(("http://", "https://")):
                raise ValueError("redirected to a non-HTTP(S) URL")
            return _read_bounded(response)
    except urllib.error.HTTPError as error:
        error.close()
        failure, kind, cause = f"HTTP {error.code} {error.reason}", f"HTTP {error.code}", error
    except (OSError, http.client.HTTPException, ValueError) as error:
        _close_unreturned_responses(error)
        failure, kind, cause = f"{type(error).__name__}: {error}", type(error).__name__, error
    if token is not None:
        failure = kind
        if type(cause) is ValueError:
            cause.args, cause.__cause__, cause.__context__ = (), None, None
            vars(cause).pop("__notes__", None)
        else:
            cause = None
    raise AtlasRuntimeFetchError(f"{where}: {failure}, {state}") from cause

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
    """The runtime fetch's request, its token, and what the token stays out of.

    Every case drives the real `urllib` stack against loopback sockets, except
    four. `test_exceptions_outside_the_typed_kinds_propagate` and
    `test_a_token_clears_what_a_value_error_chains_to` inject openers that
    raise directly. `test_token_validation_rejects_everything_http_client_rejects`
    needs only an unconnected connection, and
    `test_environment_token_is_the_first_non_blank_of_github_token_then_gh_token`
    reads no network.
    """

    SECRET = "fake-secret-token"
    # A 302 whose `Location` a test fills in.
    REDIRECT = "HTTP/1.1 302 Moved\r\nContent-Length: 0\r\nLocation: {}\r\n\r\n"

    def setUp(self) -> None:
        self.reads: list[tuple] = []

    @contextlib.contextmanager
    def _local_server(self, reply: bytes | tuple[bytes | None, ...] | None):
        """Answer loopback connections with `reply`, one connection per reply.

        `reply` is bytes, or a tuple of them answering consecutive connections;
        `None` holds that connection open. Yields the port and the request
        texts read.
        """

        replies = reply if isinstance(reply, tuple) else (reply,)
        received: list[str] = []
        release = threading.Event()
        with socket.create_server(("127.0.0.1", 0)) as listener:
            listener.settimeout(30)  # Backstop only: exit's connections end it.

            def serve() -> None:
                with contextlib.suppress(OSError):
                    for answer in replies:
                        connection, _ = listener.accept()
                        with connection:
                            connection.settimeout(30)
                            request = connection.recv(65536)
                            if request:
                                received.append(request.decode("latin-1"))
                            if answer is None:
                                release.wait(timeout=30)
                            else:
                                connection.sendall(answer)

            thread = threading.Thread(target=serve, daemon=True)
            thread.start()
            port = listener.getsockname()[1]
            try:
                yield port, received
            finally:
                release.set()
                for _ in replies:
                    if not thread.is_alive():
                        break
                    with contextlib.suppress(OSError):
                        socket.create_connection(("127.0.0.1", port), timeout=5).close()
                thread.join(timeout=30)

    @staticmethod
    def _environment_token(token):
        """`token` as the environment lookup returns it, for a `str` value."""

        if isinstance(token, str):
            return atlas_runtime_token({"GITHUB_TOKEN": token})
        return token

    def _fetch(self, reply, token=None, wait=0.5, name="process_tree.py"):
        """Fetch from loopback (timeout `wait`): the body or error, and the requests.

        A `str` token is read as the value of `GITHUB_TOKEN`.
        """

        token = self._environment_token(token)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with self._local_server(reply) as (port, received):

            def open_loopback(request, timeout):
                self.assertEqual(timeout, 30)
                url = request.full_url.replace(ATLAS_RUNTIME_BASE, f"http://127.0.0.1:{port}")
                local = urllib.request.Request(url, headers=request.headers)
                for key, value in request.unredirected_hdrs.items():
                    local.add_unredirected_header(key, value)
                response = opener.open(local, timeout=wait)
                read = response.read
                response.read = lambda *args: self.reads.append(args) or read(*args)
                return response

            try:
                outcome = fetch_atlas_runtime_file(name, token, open_loopback)
            except AtlasRuntimeFetchError as error:
                outcome = error
        return outcome, received

    def _assert_token_absent(self, error: BaseException, token: str) -> None:
        """No 8-character window of the token is in the message, repr or either traceback form.

        A window of eight characters that holds no character `repr` or a parser
        rewrites appears wherever the token does, whatever escaping is applied.
        """

        renderings = (
            str(error),
            repr(error),
            "".join(traceback.format_exception(error)),
            "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        )
        for start in range(len(token) - 7):
            for text in renderings:
                self.assertNotIn(token[start : start + 8], text)

    # The headers of a request without a token: the two the fetch sets, the
    # three `urllib` and `http.client` add. A token adds `authorization` only.
    REQUEST_HEADERS = ("accept", "accept-encoding", "connection", "host", "user-agent")

    @staticmethod
    def _header_names(request: str) -> list[str]:
        return sorted(line.split(":")[0].lower() for line in request.split("\r\n")[1:] if ":" in line)

    def _assert_request_headers(self, request: str, token: str | None) -> None:
        expected = sorted((*self.REQUEST_HEADERS, *(("authorization",) if token else ())))
        self.assertEqual(self._header_names(request), expected)

    def _assert_typed_failure(self, reply, cause: type[BaseException], **kwargs):
        error, received = self._fetch(reply, **kwargs)
        self.assertIsInstance(error, AtlasRuntimeFetchError)
        for part in ("process_tree.py", ATLAS_RUNTIME_REVISION, cause.__name__):
            self.assertIn(part, str(error))
        self.assertIsInstance(error.__cause__, cause)
        return error, received

    def test_request_names_the_pinned_file_and_sends_the_token_only_as_bearer(self) -> None:
        reply = b"HTTP/1.1 200 OK\r\nContent-Length: 7\r\n\r\npayload"
        for token in (None, self.SECRET):
            with self.subTest(authenticated=token is not None):
                body, received = self._fetch(reply, token)
                self.assertEqual(body, b"payload")
                request = received[0].lower()
                self.assertTrue(request.startswith(f"get /process_tree.py?ref={ATLAS_RUNTIME_REVISION} "))
                self._assert_request_headers(received[0], token)
                self.assertEqual(f"authorization: bearer {self.SECRET}" in request, token is not None)

    def test_exceptions_outside_the_typed_kinds_propagate(self) -> None:
        for token in (None, self.SECRET):
            for failure in (KeyError("k"), RuntimeError("r"), TypeError("t")):
                with self.subTest(authenticated=token is not None, failure=type(failure).__name__):

                    def opener(request, timeout):
                        raise failure

                    # `AtlasRuntimeFetchError` is a `RuntimeError`, so identity decides.
                    with self.assertRaises(type(failure)) as raised:
                        fetch_atlas_runtime_file(
                            "process_tree.py", self._environment_token(token), opener
                        )
                    self.assertIs(raised.exception, failure)

    def test_quota_refusal_names_file_status_and_authentication(self) -> None:
        reply = b"HTTP/1.1 403 rate limit exceeded\r\nContent-Length: 0\r\n\r\n"
        for token, state in ((None, "unauthenticated"), (self.SECRET, "authenticated")):
            with self.subTest(state=state):
                error, received = self._fetch(reply, token, name="windows_process.py")
                self.assertIsInstance(error, AtlasRuntimeFetchError)
                for part in ("windows_process.py", "HTTP 403", state):
                    self.assertIn(part, str(error))
                self.assertEqual("unauthenticated" in str(error), token is None)
                self._assert_request_headers(received[0], token)
                self.assertEqual("rate limit exceeded" in str(error), token is None)
                if token is None:
                    # The fetch closes the error response, not the collector.
                    self.assertIsInstance(error.__cause__, urllib.error.HTTPError)
                    self.assertTrue(error.__cause__.fp.closed)
                else:
                    self.assertIsNone(error.__cause__)
                    self._assert_token_absent(error, token)

    def test_transport_failures_are_typed_with_their_cause(self) -> None:
        incomplete = http.client.IncompleteRead
        cases = (
            (None, TimeoutError),
            (b"", http.client.RemoteDisconnected),
            (b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nshort", incomplete),
            (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nshort", incomplete),
        )
        for reply, cause in cases:
            with self.subTest(cause=cause.__name__, reply=reply):
                error, _ = self._assert_typed_failure(reply, cause)
                self.assertIn("unauthenticated", str(error))

    def test_declared_sizes_never_size_a_read(self) -> None:
        # CPython 3.13.3 raises `OverflowError` or `MemoryError` from an
        # unbounded read of these; 3.13.12 reads in bounded steps already.
        lengths = ("9223372036854775808", "99999999999999999999999999999", "4611686018427387904")
        sizes = ("FFFFFFFFFFFFFFFFFFFFFFFFFFFF", "10000000000000000")
        framings = [f"Content-Length: {length}\r\n\r\n" for length in lengths] + [
            f"Transfer-Encoding: chunked\r\n\r\n{size}\r\n" for size in sizes
        ]
        for framing in framings:
            with self.subTest(framing=framing):
                reply = f"HTTP/1.1 200 OK\r\n{framing}short".encode()
                self._assert_typed_failure(reply, http.client.IncompleteRead)

    def test_body_bound_is_inclusive(self) -> None:
        def reply(body: bytes) -> bytes:
            return f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body

        at_bound = b"x" * ATLAS_RUNTIME_MAX_BYTES
        for token in (None, self.SECRET):
            with self.subTest(authenticated=token is not None):
                self.assertEqual(self._fetch(reply(at_bound), token, wait=5)[0], at_bound)
                error, _ = self._assert_typed_failure(
                    reply(at_bound + b"x"), ValueError, token=token, wait=5
                )
                # The bound is the fetch's own text, which a token drops with the rest.
                self.assertEqual(str(ATLAS_RUNTIME_MAX_BYTES) in str(error), token is None)
                if token is not None:
                    self._assert_scrubbed_cause(error, ValueError)
                    self._assert_token_absent(error, token)

    def test_a_token_keeps_server_text_and_the_cause_out_of_the_error(self) -> None:
        # Each server text can hold the token; the chunk size holds it in the
        # chain only. Beside a plain token, the tokens hold U+00A0, a backslash,
        # both quote characters, and U+0085: characters `repr` escapes or a
        # parser rewrites, and whose escaping depends on the text around them.
        tokens = (
            "fake-secret-token-0123",
            "fake-secret-\xa0-token-0123",
            "fake-secret\\token-0123",
            "fake-secret-'\"-token-0123",
            "fake-secret-\x85-token-0123",
        )
        for secret in tokens:
            cases = (
                (f"HTTP/1.1 401 {secret}\r\nContent-Length: 0\r\n\r\n", "HTTP 401,"),
                (self.REDIRECT.format(f"gopher://x/{secret}"), "HTTP 302,"),
                (self.REDIRECT.format(f"http://[{secret}]/x"), "ValueError,"),
                (self.REDIRECT.format(f'http://["{secret}]/x'), "ValueError,"),
                (f"{secret}\r\n\r\n", "BadStatusLine,"),
                (f"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n{secret}\r\n", "IncompleteRead,"),
            )
            for reply, kind in cases:
                with self.subTest(secret=secret, kind=kind, reply=reply[:60]):
                    error, received = self._fetch(reply.encode("latin-1"), secret)
                    self.assertIsInstance(error, AtlasRuntimeFetchError)
                    self.assertIn(f"authorization: bearer {secret}".lower(), received[0].lower())
                    expected = f"process_tree.py at {ATLAS_RUNTIME_REVISION}: {kind} authenticated"
                    self.assertTrue(str(error).endswith(expected), str(error))
                    self.assertIsNone(error.__context__)
                    self._assert_scrubbed_cause(error, ValueError if kind == "ValueError," else None)
                    self._assert_token_absent(error, secret)

    def _assert_scrubbed_cause(self, error, cause_type) -> None:
        """`__cause__` is the raised `cause_type` with no arguments or chain, or absent."""

        cause = error.__cause__
        if cause_type is None:
            self.assertIsNone(cause)
            return
        self.assertIs(type(cause), cause_type)
        self.assertEqual(cause.args, ())
        self.assertIsNotNone(cause.__traceback__, "the cause was replaced by a new exception")
        self.assertIsNone(cause.__cause__)
        self.assertIsNone(cause.__context__)
        self.assertFalse(hasattr(cause, "__notes__"))

    def test_a_token_clears_what_a_value_error_chains_to(self) -> None:
        # No server in the matrix above raises a chained or annotated
        # `ValueError`, so the opener does: the exception, its note and the
        # `OSError` it chains to all hold the token.
        secret = self.SECRET
        held = ValueError(secret)
        held.add_note(secret)

        def chained(request, timeout):
            try:
                raise OSError(secret)
            except OSError as inner:
                raise held from inner

        def subclass(request, timeout):
            raise UnicodeError(secret)

        for opener, cause_type in ((chained, ValueError), (subclass, None)):
            with self.subTest(opener=opener.__name__):
                with self.assertRaises(AtlasRuntimeFetchError) as raised:
                    fetch_atlas_runtime_file(
                        "process_tree.py", self._environment_token(secret), opener
                    )
                self.assertIsNone(raised.exception.__context__)
                self._assert_scrubbed_cause(raised.exception, cause_type)
                if cause_type is not None:
                    # The cause is the exception the opener raised, scrubbed in place.
                    self.assertIs(raised.exception.__cause__, held)
                self._assert_token_absent(raised.exception, secret)

    @contextlib.contextmanager
    def _ftp_server(self, payload: bytes):
        """Serve `payload` to one FTP login and `RETR` on loopback.

        Yields the port and the control lines received, complete once the block exits.
        """

        commands: list[str] = []

        with socket.create_server(("127.0.0.1", 0)) as control, socket.create_server(("127.0.0.1", 0)) as data:
            control.settimeout(10)
            data.settimeout(10)

            def serve() -> None:
                with contextlib.suppress(OSError):
                    connection, _ = control.accept()
                    with connection:
                        connection.sendall(b"220 ready\r\n")
                        for line in connection.makefile("rb"):
                            commands.append(line.decode("latin-1"))
                            if line.startswith(b"PASV"):
                                high, low = divmod(data.getsockname()[1], 256)
                                reply = f"227 Passive Mode (127,0,0,1,{high},{low})\r\n".encode()
                            elif line.startswith(b"RETR"):
                                connection.sendall(b"150 ok\r\n")
                                with data.accept()[0] as peer:
                                    peer.sendall(payload)
                                reply = b"226 done\r\n"
                            else:
                                reply = b"200 ok\r\n"
                            connection.sendall(reply)

            thread = threading.Thread(target=serve, daemon=True)
            thread.start()
            yield control.getsockname()[1], commands
            thread.join(timeout=10)

    def test_redirect_to_a_served_ftp_file_is_typed_and_unread(self) -> None:
        for token in (None, self.SECRET):
            with self.subTest(authenticated=token is not None):
                with self._ftp_server(b"print('x')\n") as (ftp_port, commands):
                    reply = self.REDIRECT.format(f"ftp://127.0.0.1:{ftp_port}/process_tree.py")
                    error, received = self._assert_typed_failure(
                        reply.encode(), ValueError, token=token, wait=5
                    )
                self.assertEqual(self.reads, [], "the ftp response body was read")
                self.assertEqual("non-HTTP(S)" in str(error), token is None)
                # urllib has contacted the ftp host; the token went to the first hop only.
                self.assertTrue(any(line.startswith("RETR") for line in commands), commands)
                self.assertEqual("authorization" in received[0].lower(), token is not None)
                for line in commands:
                    self.assertNotIn("authorization", line.lower())
                    self.assertNotIn(self.SECRET, line)
                if token is not None:
                    self._assert_scrubbed_cause(error, ValueError)
                    self._assert_token_absent(error, token)

    def test_token_a_header_cannot_carry_is_refused_before_any_request(self) -> None:
        stem = "fake-secret-stem"
        suffixes = (
            "\r\n", "\r", "\n", "\x00", "\x1b", "\x7f", "\u2019", "\U0001f512", "\ud800", "\u0100"
        )
        for suffix in suffixes:
            with self.subTest(suffix=repr(suffix)):
                error, received = self._fetch(b"", stem + suffix)
                self.assertIsInstance(error, AtlasRuntimeFetchError)
                for part in ("process_tree.py", ATLAS_RUNTIME_REVISION, "no request sent"):
                    self.assertIn(part, str(error))
                self.assertIn("a character an HTTP header cannot carry", str(error))
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

    MALFORMED_LOCATIONS = ("http://[::1/x", "http://[zz]/x", "http://a]b/x", "http://[::1]x/y")

    @staticmethod
    def _resource_warnings_after(attempt) -> int:
        """Run `attempt`, collect garbage, and count the `ResourceWarning`s emitted.

        `attempt` must return nothing: a leaked socket warns only once the
        objects holding it are collected, so none may outlive the call. Garbage
        an earlier test left is collected first, outside the recording.
        """

        gc.collect()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            attempt()
            gc.collect()
        return sum(issubclass(item.category, ResourceWarning) for item in caught)

    def test_malformed_redirect_location_leaves_no_socket_open(self) -> None:
        def abandon_a_socket() -> None:
            socket.socket()

        # The recorder is live: it sees a socket dropped unclosed.
        self.assertEqual(self._resource_warnings_after(abandon_a_socket), 1)
        for location in self.MALFORMED_LOCATIONS:
            reply = self.REDIRECT.format(location).encode()
            for token in (None, self.SECRET):
                with self.subTest(location=location, authenticated=token is not None):

                    def fetch() -> None:
                        self._assert_typed_failure(reply, ValueError, token=token, wait=2)

                    self.assertEqual(self._resource_warnings_after(fetch), 0)

    def test_malformed_redirect_location_is_typed_with_a_value_error_cause(self) -> None:
        for location in self.MALFORMED_LOCATIONS:
            with self.subTest(location=location):
                reply = self.REDIRECT.format(location).encode()
                for token in (None, self.SECRET):
                    error, _ = self._assert_typed_failure(reply, ValueError, token=token, wait=2)
                    if token is not None:
                        self._assert_scrubbed_cause(error, ValueError)
                        self._assert_token_absent(error, token)

    def test_redirect_to_another_origin_never_receives_the_token(self) -> None:
        final = b"HTTP/1.1 200 OK\r\nContent-Length: 7\r\n\r\npayload"
        for code in (301, 302, 303, 307, 308):
            with self.subTest(code=code):
                with self._local_server(final) as (target_port, target_received):
                    location = f"http://127.0.0.1:{target_port}/next"
                    redirect = self.REDIRECT.replace("302", str(code)).format(location)
                    body, origin_received = self._fetch(redirect.encode(), self.SECRET, wait=2)
                self.assertEqual(body, b"payload")
                self.assertIn(f"authorization: bearer {self.SECRET}", origin_received[0].lower())
                self._assert_request_headers(origin_received[0], self.SECRET)
                self.assertEqual(len(target_received), 1)
                self._assert_request_headers(target_received[0], None)

    def test_same_origin_redirect_drops_the_token(self) -> None:
        # `urllib` copies no `Authorization` header to a redirect, same origin
        # or not: the fetch's request sets it as an unredirected header.
        final = b"HTTP/1.1 200 OK\r\nContent-Length: 7\r\n\r\npayload"
        for code in (301, 302, 303, 307, 308):
            with self.subTest(code=code):
                redirect = self.REDIRECT.replace("302", str(code)).format("/next")
                body, received = self._fetch((redirect.encode(), final), self.SECRET, wait=2)
                self.assertEqual(body, b"payload")
                self.assertEqual(len(received), 2)
                self.assertTrue(received[1].startswith("GET /next "), received[1])
                self.assertIn(f"authorization: bearer {self.SECRET}", received[0].lower())
                self._assert_request_headers(received[0], self.SECRET)
                self._assert_request_headers(received[1], None)

    def test_blank_environment_token_sends_no_authorization(self) -> None:
        reply = b"HTTP/1.1 200 OK\r\nContent-Length: 7\r\n\r\npayload"
        for variable in ("GITHUB_TOKEN", "GH_TOKEN"):
            for blank in (" ", "\r\n"):
                with self.subTest(variable=variable, blank=repr(blank)):
                    token = atlas_runtime_token({variable: blank})
                    body, received = self._fetch(reply, token)
                    self.assertEqual(body, b"payload")
                    self._assert_request_headers(received[0], None)
        with self.subTest(case="blank GITHUB_TOKEN beside GH_TOKEN"):
            token = atlas_runtime_token({"GITHUB_TOKEN": " ", "GH_TOKEN": self.SECRET})
            body, received = self._fetch(reply, token)
            self.assertEqual(body, b"payload")
            self.assertIn(f"authorization: bearer {self.SECRET}", received[0].lower())
            self._assert_request_headers(received[0], self.SECRET)

    def test_refused_token_names_the_variable_that_held_it(self) -> None:
        stem = "fake-secret-stem"
        refused = stem + "\x00"
        cases = (
            ("GITHUB_TOKEN", {"GITHUB_TOKEN": refused}),
            ("GH_TOKEN", {"GH_TOKEN": refused}),
            ("GITHUB_TOKEN", {"GITHUB_TOKEN": refused, "GH_TOKEN": "fake-secret-other"}),
            ("GH_TOKEN", {"GITHUB_TOKEN": " ", "GH_TOKEN": refused}),
        )
        for variable, environment in cases:
            other = "GH_TOKEN" if variable == "GITHUB_TOKEN" else "GITHUB_TOKEN"
            with self.subTest(variable=variable, environment=sorted(environment)):
                error, received = self._fetch(b"", atlas_runtime_token(environment))
                self.assertIsInstance(error, AtlasRuntimeFetchError)
                self.assertIn("no request sent", str(error))
                self.assertIn(variable, str(error))
                self.assertNotIn(other, str(error))
                self.assertEqual(received, [])
                self._assert_token_absent(error, stem)

    def test_environment_token_is_the_first_non_blank_of_github_token_then_gh_token(self) -> None:
        both = {"GITHUB_TOKEN": "a", "GH_TOKEN": "b"}
        self.assertEqual(atlas_runtime_token(both), AtlasRuntimeToken("GITHUB_TOKEN", "a"))
        self.assertEqual(atlas_runtime_token({"GH_TOKEN": "b"}), AtlasRuntimeToken("GH_TOKEN", "b"))
        for blank in ("", " ", "\r\n", "\t \n"):
            with self.subTest(blank=repr(blank)):
                self.assertEqual(
                    atlas_runtime_token({"GITHUB_TOKEN": blank, "GH_TOKEN": "b"}),
                    AtlasRuntimeToken("GH_TOKEN", "b"),
                )
                self.assertIsNone(atlas_runtime_token({"GITHUB_TOKEN": blank}))
                self.assertIsNone(atlas_runtime_token({"GH_TOKEN": blank}))
                self.assertIsNone(atlas_runtime_token({"GITHUB_TOKEN": blank, "GH_TOKEN": blank}))
        # A token with a trailing line ending is not blank: it is kept, and the
        # fetch refuses it naming its variable.
        self.assertEqual(
            atlas_runtime_token({"GITHUB_TOKEN": "a\n"}), AtlasRuntimeToken("GITHUB_TOKEN", "a\n")
        )
        self.assertIsNone(atlas_runtime_token({}))


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
