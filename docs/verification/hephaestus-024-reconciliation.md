# Hephaestus 0.24 and repaired-native reconciliation

Historical checkpoint: exact head `38e014b3` is independently accepted within
its recorded 0.24 graph. Current source admission and verification are in the
[0.25 renewal](hephaestus-025-renewal.md); this note does not claim latest-provider
readiness.

[arch] The unpublished Coeus 0.13 workspace reconciles its existing repairs
with actual merged Coeus PR #614 (e8b429ffd8c48a2fd3daf8289d079edb6b3c1a7c,
feature 9b76889437b86d6be035b1371196bf91f0d2b459). Actual PR #613
(e4650b638f569f69f94429f0736dc92c3e20a330) remains an ancestor. This supersedes
the earlier provisional Hephaestus >=0.22.2,<0.24 admission.

## Intent and constraints

Hephaestus PR #505 introduces the canonical dialect-parametric
`FusionScalar<L>` in core at 0.24.0; older core releases do not export it.
Use that provider-owned capability without a Coeus compatibility trait or
another fusion dispatch implementation. Retain the owner-authored CUDA,
disabled-CUDA and WGPU source changes byte-for-byte. Preserve private 0.13
crate identities, Moirai runtime/async 0.10 floors, Eunomia 0.10/0.11 admission,
Hermes/Apollo safety floors, the native/Python linkage arrangement and all
numerical and NN test-cycle repairs. Do not change workflows, features,
assertions, timeouts or profiles to obtain a passing graph.

## Selected design and migration

All five Hephaestus edges require ^0.24.0. `CudaScalar` requires
`FusionScalar<CudaC>` and `WgpuScalar` requires `FusionScalar<Wgsl>`;
existing f32/f64 and WGSL scalar implementations use the matching provider
implementations. Dialect variation stays in the type parameter. The shared
generic fusion dispatcher remains the single implementation; no dynamic
adapter, alias, forwarding wrapper or duplicate kernel is introduced.
Downstream custom scalar implementations implement the provider trait for
their selected dialect. The unpublished 0.13 cohort versions this source change.

Both Leto packages require >=0.46.3,<0.48, excluding defective 0.46.2.
The accepted repair candidate is b3d744a91f331061d0e3eb40b5126cc1fa7782c2,
tree 63be55806f50a253718644a9d98e65ea50b4d2fd. Its exact-head independent
numerical and matched-performance receipt is SHA256
2c7570ede0b949ec109df7be01968aec7554fef7e676ed4d88cbe29c57580d1c.
That acceptance resolves the native guard-cost finding within its measured
scope; it does not accept absolute application performance, Hephaestus 0.24,
this Coeus graph, a release, or the separate future 0.47 stored-order policy.

Resolve the manifest field by field and regenerate Cargo.lock from the
preserved private/#613 records against current coherent providers. Do not
copy #614's defective 0.46.2 lock or sixteen 0.12 workspace entries. The repaired native merge 5051cd9577650a5712aa6a7effcd8955c9d8bcc7 and both
0.46.3 registry archives are verified against the accepted source. Ordinary
Cargo renewal selects that current default and Hephaestus 0.24 at 746553d0.
All 235 registry and 34 unrelated Git records remain unchanged. No permanent
branch/rev/path override substitutes for canonical provider resolution.

## Alternatives and failure modes

Retaining the former <0.24 interval after importing the new core trait would
advertise unsupported source compatibility. Copying the complete owner
manifest would remove private provider floors, versioning and Python linkage
repairs. Copying its lock would reintroduce the known native regression.
Duplicating its source edits would split ownership. These options are rejected.
A source-compatible build does not establish device behavior, normalized
registry resolution, or a single common Python FFI package identity.

## Verification and remaining boundaries

At the actual reconciled head, check exact owner Rust bytes, every other
private Rust file, both histories, provider identities and unchanged gates.
Run the applicable native/backend suites, scalar/layout/seed regressions,
strict lint/docs/fmt, Rust 1.95, conformance/API and ordinary verified packages
against the canonical graph. Keep prior registry/Git records unless an
assessed current provider renewal requires a change. Final exact-head
independent review and applicable hosted checks remain gates.

Coeus PyO3 0.23 versus common PyO3 0.30, future Leto 0.47 numerical-policy
reconciliation, #497/#500 owner work, first-publication prerequisites, physical
GPU/platform coverage and absolute application performance remain separate.
Coeus Miri is deferred and unverified. The denied push remains paused;
no merge, release, authentication or unrelated service hold is relaxed.
