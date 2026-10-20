# ADR 0085: Versioned Hephaestus 0.27.1 admission

Status: Proposed  \
Date: 2026-10-20  \
Change class: [major]  \
Delivery: Item `COEUS-PROVIDER-0271-COMPATIBILITY`

## Context

The accepted 0.13 source requires Hephaestus ^0.25 and Apollo FFT ^0.29.3.
Those requirements exclude the reviewed Hephaestus 0.27.1 numerical repair.
The separate admission diagnostic changes requirements only and uses external
path overlays. Its source-less lock entries are not canonical Git identities.

This crosses a public dependency boundary: `HephaestusProvider::Device` is
bounded by `hephaestus_core::ComputeDevice`, device buffers occur in public
storage interfaces, and `WgpuContext::hephaestus_device` exposes
`hephaestus_wgpu::WgpuDevice`. Different source/version identities are
different Rust types even when consumer Rust text is unchanged.

## Decision proposed for exact-head review

Prepare the existing 16-package lockstep cohort at 0.14.0. Require all five
Hephaestus dependencies at ^0.27.1 and Apollo FFT at ^0.30.0. Update every
internal path+version edge atomically. Preserve canonical Git URLs, features,
default-features, provider ownership, scalar contracts, numerical assertions,
test inventory and CI gates. Keep the 1.95 MSRV and pinned test toolchain.

The public break uses the next pre-1.0 compatibility line, not a 1.0 stability
commitment. This is authorized private preparation; the ADR remains Proposed.
ADR sign-off and independent review of the final exact candidate precede
adoption, opening release implementation items, and any public update.

Keep the provider floor distinct from the selected source commit. ^0.27.1
means >=0.27.1,<0.28.0; the Git lock identifies the reviewed commit. Preserve
real Git source identities, with one common type-bearing dependency graph.
Private upstream objects may be imported into an isolated Cargo source cache
for locked checks without claiming that a remote default or registry serves
them. Do not commit a diagnostic path overlay or widen/downlevel admission.

## Migration

Update all Coeus requirements to ^0.14.0 and Apollo FFT to ^0.30.0. Direct
Hephaestus users move core and backend requirements together to ^0.27.1 and
renew their locks to one reviewed source graph. Implementors of
`HephaestusProvider` and callers exchanging device/buffer/error types use
the same provider source identity as Coeus. Do not bridge old provider types
with aliases, wrappers or casts. Python reads the same workspace version;
wheel runtime/ABI and complete platform acceptance remain separate gates.

## Alternatives and failure modes

- Retaining old provider requirements excludes the repair.
- A 0.13 patch would hide the public provider-type compatibility boundary.
- Path overlays erase ordinary Git provenance and cannot establish admission.
- Lowering the floor or widening to untested provider lines evades the contract.
- A manifest version does not publish an upstream package or select its commit.

Hephaestus PR #509 remains draft and unmerged/unpublished. The private Apollo
compatibility preparation is not a remote or registry release. Default-source
fresh resolution, normalized registry packages and ordinary registry consumers
must be qualified at the separately authorized upstream lifecycle step.

## Verification and retained holds

Run actual-source baseline/new admission controls, locked default/all-feature
resolution, current CPU/native and strict/docs/doctest/MSRV checks, package
normalization and actual before/after API graphs. Attribute every result to
its source graph and exact head; unavailable or unrun checks stay explicit.
No assertion, skip, backend feature policy, baseline or budget changes.

Preserve #611 and #585 reviewed evidence and all existing numerical, mean,
negative-power, NN-gradient and native/wheel-linkage repairs. This candidate
does not integrate #585 or alter its branch. All normal-final/subnormal-erfc,
six portable countermodel, performance, physical-hardware, broader-cohort,
deferred Coeus Miri and service holds remain. Main/scheduled publish-pending
side effects require release coordination; private preparation authorizes no
push, PR edit, dispatch, acquisition retry, merge or publication.
