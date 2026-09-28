"""Value-semantic tests for CI changed-scope classification."""

from __future__ import annotations

import subprocess
import unittest

from scripts.ci_scope import (
    changed_paths,
    classify,
    is_doc_only_path,
    is_hook_contract_path,
)


class ChangedScopeTests(unittest.TestCase):
    def test_hook_contract_paths_select_the_fast_path(self) -> None:
        paths = (
            ".githooks/pre-push",
            "scripts/lockfile.py",
            "scripts/tests/test_hooks.py",
        )
        self.assertEqual(classify(paths), "hook-only")

    def test_doc_only_paths_skip_native_coverage(self) -> None:
        paths = (
            "backlog.md",
            "gap_audit.md",
            "CHANGELOG.md",
            "README.md",
            "docs/adr/0037-uninitialized-cow-consumer.md",
            "docs/adr/README.md",
            "LICENSE-APACHE",
            "LICENSE-MIT",
        )
        for path in paths:
            with self.subTest(path=path):
                self.assertEqual(classify((path,)), "skip")
        self.assertEqual(classify(paths), "skip")

    def test_rust_and_workflow_paths_retain_native_coverage(self) -> None:
        for path in (
            "Cargo.toml",
            "Cargo.lock",
            "crates/coeus-core/src/lib.rs",
            ".github/workflows/ci.yml",
            "scripts/tests/test_lockfile.py",
            "scripts/ci_scope.py",
            "rust-toolchain.toml",
        ):
            with self.subTest(path=path):
                self.assertEqual(classify((path,)), "native")

    def test_mixed_doc_and_native_paths_are_conservative(self) -> None:
        self.assertEqual(
            classify(("backlog.md", "crates/coeus-core/src/lib.rs")),
            "native",
        )
        self.assertEqual(
            classify(("backlog.md", ".githooks/pre-push")),
            "native",
        )

    def test_empty_or_mixed_ranges_are_conservative(self) -> None:
        self.assertEqual(classify(()), "native")
        self.assertEqual(
            classify((".githooks/pre-push", "Cargo.toml")),
            "native",
        )

    def test_hook_contract_boundary_is_explicit(self) -> None:
        self.assertTrue(is_hook_contract_path(".githooks/pre-commit"))
        self.assertTrue(is_hook_contract_path("scripts/lockfile.py"))
        self.assertFalse(is_hook_contract_path("scripts/lockfile.pyc"))
        self.assertFalse(is_hook_contract_path("scripts/tests/test_lockfile.py"))

    def test_doc_only_boundary_is_explicit(self) -> None:
        self.assertTrue(is_doc_only_path("backlog.md"))
        self.assertTrue(is_doc_only_path("docs/adr/0001-example.md"))
        self.assertTrue(is_doc_only_path("LICENSE-APACHE"))
        self.assertFalse(is_doc_only_path("crates/coeus-core/README.md.rs"))
        self.assertFalse(is_doc_only_path("scripts/ci_scope.py"))
        self.assertFalse(is_doc_only_path("Cargo.toml"))

    def test_revision_resolution_failure_is_explicit(self) -> None:
        with self.assertRaises(subprocess.CalledProcessError):
            changed_paths("missing-ci-base", "missing-ci-head")
