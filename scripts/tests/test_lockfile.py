"""Exercise lock resolution against real temporary Cargo workspaces."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import tarfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "lockfile", Path(__file__).resolve().parents[1] / "lockfile.py"
)
assert SPEC is not None and SPEC.loader is not None
lockfile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lockfile)


class CargoResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = self.root / "Cargo.toml"
        self.manifest.write_text(
            '[package]\nname = "resolution-fixture"\nversion = "0.1.0"\n'
            'edition = "2024"\n[workspace]\nexclude = ["optional"]\n'
            '[dependencies]\noptional = { path = "optional", optional = true }\n',
            encoding="utf-8",
        )
        for directory in (self.root, self.root / "optional"):
            (directory / "src").mkdir(parents=True)
            (directory / "src" / "lib.rs").write_text(
                "//! Cargo resolution fixture.\n", encoding="utf-8"
            )
        (self.root / "optional" / "Cargo.toml").write_text(
            '[package]\nname = "optional"\nversion = "0.1.0"\nedition = "2024"\n',
            encoding="utf-8",
        )
        result = lockfile.run_outside_the_overlay(
            ["generate-lockfile", "--offline"], self.manifest
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_both_activation_graphs_accept_one_unchanged_lock(self) -> None:
        before = self.manifest.with_name("Cargo.lock").read_bytes()
        self.assertEqual(lockfile.check_resolution(self.manifest), 0)
        for activation, expected in (
            ([], {"resolution-fixture"}),
            (["--all-features"], {"resolution-fixture", "optional"}),
        ):
            result = lockfile.run_outside_the_overlay(
                ["metadata", "--offline", "--locked", "--format-version", "1", *activation],
                self.manifest,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads(result.stdout)
            names = {package["id"]: package["name"] for package in metadata["packages"]}
            active = {names[node["id"]] for node in metadata["resolve"]["nodes"]}
            self.assertEqual(active, expected)
        self.assertEqual(self.manifest.with_name("Cargo.lock").read_bytes(), before)

    def test_stale_lock_is_rejected_then_stabilized_idempotently(self) -> None:
        path = self.manifest.with_name("Cargo.lock")
        before = path.read_bytes()
        self.manifest.write_text(
            self.manifest.read_text(encoding="utf-8").replace('version = "0.1.0"', 'version = "0.2.0"'),
            encoding="utf-8",
        )
        diagnostic = io.StringIO()
        with contextlib.redirect_stderr(diagnostic):
            self.assertEqual(lockfile.check_resolution(self.manifest), 1)
        self.assertIn("default features", diagnostic.getvalue())
        self.assertIn("lock file", diagnostic.getvalue())
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(lockfile.stabilize_resolution(self.manifest), 0)
        after = path.read_bytes()
        self.assertIn(b'version = "0.2.0"', after)
        self.assertEqual(lockfile.stabilize_resolution(self.manifest), 0)
        self.assertEqual(path.read_bytes(), after)

    def test_invalid_manifest_releases_its_drive_mapping(self) -> None:
        self.manifest.write_text("[package\n", encoding="utf-8")
        result = lockfile.run_outside_the_overlay(
            ["metadata", "--offline", "--format-version", "1"], self.manifest
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unclosed table", result.stderr)
        if os.name == "nt":
            mappings = lockfile.run_command(["subst"], timeout=10)
            self.assertEqual(mappings.returncode, 0, mappings.stderr)
            self.assertNotIn(
                str(self.manifest.resolve().parent).casefold(), mappings.stdout.casefold()
            )

    @unittest.skipUnless(os.name == "nt", "subst mappings are Windows-specific")
    def test_cleanup_failure_preserves_the_cargo_failure(self) -> None:
        self.manifest.write_text("[package\n", encoding="utf-8")
        result = lockfile.run_outside_the_overlay(
            ["metadata", "--offline", "--format-version", "1"], self.manifest
        )
        self.assertNotEqual(result.returncode, 0)
        # Invalid drive syntax forces a real cleanup error without risking a
        # mapping that another task acquired after our command finished.
        drive = "invalid-drive"
        with self.assertRaises(RuntimeError) as failure:
            lockfile.remove_drive(drive, result)
        self.assertIn(f"cargo failed with status {result.returncode}", str(failure.exception))
        self.assertIn(result.stderr, str(failure.exception))
        self.assertIn(f"failed to remove repository drive {drive}", str(failure.exception))

    def test_uncached_optional_package_fails_only_all_features(self) -> None:
        # Use a full workspace path: libgit2 cannot initialize a registry cache
        # through the shortened Windows user-profile path on the CI host.
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory).resolve()
            index = root / "index"
            (index / "op" / "ti").mkdir(parents=True)
            archive = root / "downloads" / "optional" / "0.1.0" / "download"
            archive.parent.mkdir(parents=True)
            with tarfile.open(archive, "w:gz") as package:
                for name in ("Cargo.toml", "src/lib.rs"):
                    package.add(
                        self.root / "optional" / name,
                        arcname=f"optional-0.1.0/{name}",
                    )
            (index / "config.json").write_text(
                json.dumps({"dl": (root / "downloads").as_uri() + "/{crate}/{version}/download"}),
                encoding="utf-8",
            )
            (index / "op" / "ti" / "optional").write_text(
                json.dumps({
                    "name": "optional", "vers": "0.1.0", "deps": [],
                    "cksum": hashlib.sha256(archive.read_bytes()).hexdigest(),
                    "features": {}, "yanked": False,
                }) + "\n",
                encoding="utf-8",
            )
            for arguments in (
                ["git", "init", "-q", str(index)],
                ["git", "-C", str(index), "add", "config.json", "op/ti/optional"],
                ["git", "-C", str(index), "-c", "user.name=Coeus tests",
                 "-c", "user.email=tests@localhost", "commit", "-qm", "Add fixture index"],
            ):
                result = lockfile.run_command(arguments, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
            variables = {
                "CARGO_HOME": str(root / "cargo-home"),
                "CARGO_REGISTRIES_FIXTURE_INDEX": index.as_uri(),
            }
            previous = {name: os.environ.get(name) for name in variables}
            os.environ.update(variables)
            try:
                self.manifest.write_text(
                    self.manifest.read_text(encoding="utf-8").replace(
                        'path = "optional", optional = true',
                        'version = "0.1.0", registry = "fixture", optional = true',
                    ), encoding="utf-8",
                )
                result = lockfile.run_outside_the_overlay(["generate-lockfile"], self.manifest)
                self.assertEqual(result.returncode, 0, result.stderr)
                before = self.manifest.with_name("Cargo.lock").read_bytes()
                result = lockfile.run_outside_the_overlay(
                    ["metadata", "--offline", "--locked", "--format-version", "1"],
                    self.manifest,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                diagnostic = io.StringIO()
                with contextlib.redirect_stderr(diagnostic):
                    self.assertEqual(lockfile.check_resolution(self.manifest), 1)
                self.assertIn("all features", diagnostic.getvalue())
                self.assertIn("failed to download `optional v0.1.0", diagnostic.getvalue())
                self.assertEqual(self.manifest.with_name("Cargo.lock").read_bytes(), before)
            finally:
                for name, value in previous.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value


class CommandDeadlineTests(unittest.TestCase):
    def test_blocked_child_is_terminated_at_its_deadline(self) -> None:
        result = lockfile.run_command(
            [sys.executable, "-c", "import threading; threading.Event().wait()"],
            timeout=1,
        )
        self.assertEqual(result.returncode, 124)
        self.assertIn("command exceeded 1s", result.stderr)


if __name__ == "__main__":
    unittest.main()
