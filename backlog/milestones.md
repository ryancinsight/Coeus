# Milestone ladder ([ADR 0083](../docs/adr/0083-release-scope-1-0.md))
<a id="M-core-correctness"></a> **M-core-correctness** (0.13.0) needs: (none) — core tensor and backend surface.
- exit: pass core tensor tests (`cargo test -p coeus-tensor --locked --all-features` passes).
- exit: pass ops tests (`cargo test -p coeus-ops --locked --all-features` passes).
- standing: no open correctness item.
<a id="M-autograd-verify"></a> **M-autograd-verify** (0.14.0) needs: M-core-correctness — autograd and verification.
- exit: pass autograd tests (`cargo test -p coeus-autograd --locked --all-features` passes).
- exit: pass nn and optim tests (`cargo test -p coeus-nn --locked --all-features` and `cargo test -p coeus-optim --locked --all-features` pass).
- standing: no open correctness item.
<a id="M-ecosystem"></a> **M-ecosystem** (1.0.0) needs: M-autograd-verify — stable release (authority-gated).
- standing: `cargo fmt --all -- --check`, `cargo clippy --workspace --locked --all-targets --all-features -- -D warnings`, `cargo test --workspace --locked --all-features`, `cargo test --workspace --locked --doc --all-features`, and `cargo doc --workspace --locked --no-deps --all-features` pass; SemVer and publish dry-run gates required for declared releases pass; README and rustdoc are current; no open correctness item.
