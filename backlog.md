# Coeus Backlog

Ready (Definition-of-Ready) items. Each carries a change-class tag, a testable
acceptance criterion, and identified dependencies/blockers.

## G-FFT-CONSOLIDATE — invert FFT tensor-bridge ownership `[arch]`

**Status:** todo · **Blocker:** `coeus-autograd` under active concurrent editing (loss
re-export work) at the time `coeus-fft` landed — refactoring `coeus-autograd/ops/fft.rs`
while a peer mutates sibling modules risks verifying/committing against a transiently
broken crate.

**Context:** The Apollo→Coeus tensor FFT primitive (`FftScalar` trait + tensor-level
`fft_1d`/`ifft_1d`) currently exists in **two** places: the SSOT in
[`coeus-fft`](coeus-fft/src/lib.rs) (non-autograd, deps `coeus-core`/`coeus-tensor`/`apollo-fft`)
and a duplicate in [`coeus-autograd/ops/fft.rs`](coeus-autograd/src/ops/fft.rs) (which also
hosts the `Var`-level nodes). This is a transient DRY violation accepted only because the
autograd crate was blocked; it must be consolidated.

**Task:** Make `coeus-autograd` depend on `coeus-fft`:
1. Add `coeus-fft = { workspace = true }` to `coeus-autograd/Cargo.toml`.
2. In `coeus-autograd/ops/fft.rs`, delete the local `FftScalar`/`fft_1d`/`ifft_1d`
   definitions and replace with `pub use coeus_fft::{FftScalar, fft_1d, ifft_1d};`
   (keeps `ops::mod`/`lib.rs` re-export names stable — no downstream call-site churn).
   Retain only the `Var`-level nodes (`Fft1DNode`, `Ifft1DNode`, `fft_energy`, `*_var`).

**Acceptance criteria:**
- No duplicate `FftScalar`/`fft_1d`/`ifft_1d` definitions remain in the workspace
  (single grep hit each, in `coeus-fft`).
- `cargo nextest run -p coeus-autograd` and `-p coeus-nn` pass unchanged (Var-level FFT
  autograd behavior and gradients identical).
- `cargo doc --no-deps` warning-clean; the `coeus-fft` module doc's forward reference to
  this item is satisfied.

**Change-class:** `[arch]` (crate dependency-direction change; no public API surface change,
so no version-major implication — re-export names are preserved).
