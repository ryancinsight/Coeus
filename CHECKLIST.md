# Coeus Development Roadmap Checklist

<!-- Compacted 2026-09-21 under the 1,000-line board budget: a board is a queue, not a ledger, so closed sections and closed item bodies are gone -- their record is the PR that closed them and its `Item:` trailer. Open items, anchors and live-marked residuals are kept. Recover any removed narrative with `git log -p -- <this file>`. -->
## Lockfile resolution

- [x] [Verify both activation sets and bound resolution](docs/backlog.md#coeus-lockfile-script-narrow).
- [ ] [Restore fresh generation after upstream version alignment](docs/backlog.md#coeus-provider-resolution-2026-09-07).

## COEUS-BYTEMUCK-RESIDUE — one concept, two markers [patch] — todo <a id="coeus-bytemuck-residue"></a>

- **Finding:** `coeus_core::Scalar` now carries both `bytemuck::Pod` and Eunomia's `Pod` as supertraits. Eunomia is the stack's datatype law and owns this concept; bytemuck is the third-party equivalent, retained only because `Scalar::has_zero_bit_pattern`'s default body calls `bytemuck::bytes_of`.
- **Scope:** move that body to `eunomia::layout::bytes_of` and drop the bytemuck supertrait, so one marker states the contract. Check the blast radius first: the supertrait is public surface, so removing it is `[major]` for anyone implementing `Scalar` outside the workspace.
- **Last-update:** 2026-09-06.

## COEUS-STAGGERED-VENDOR-WIRING — Bind the remaining vendor backends [minor] — todo <a id="coeus-staggered-vendor-wiring"></a>

- **Outcome:** bind `StaggeredPairOps<f32>` for CUDA and Metal through the existing provider-owned kernels and shared staggered dispatch; both bindings remain open.
- **CUDA:** `coeus-cuda` now uses the Hephaestus CUDA provider and carries no Cutile or `cuda-bindings` dependency; `libclang` is not a prerequisite for this binding. Add the missing provider/backend binding without duplicating dispatch.
- **Metal:** `HephaestusBackend<MetalProvider>` already uses the generic bridge. Add `StaggeredProvider` for `MetalProvider` to obtain its existing generic staggered implementation.
- **Provider capability:** `CudaStaggered3DOps` and `MetalStaggered3DOps` exist upstream; ROCm still requires upstream staggered kernels.
- **Verification:** exercise the shared gradient/divergence, axis, adjoint, and invalid-grid oracles on real CUDA and Apple Metal devices; preserve typed acquisition failures. CUDA needs a working NVIDIA driver/device, and Metal device execution needs Apple hardware.
- **Last-update:** 2026-09-07.

## COEUS-SIBLING-NAMED-CRATES — Crates named after their dependencies [major] [arch] — todo <a id="coeus-sibling-named-crates"></a>

- **Finding:** `coeus-leto` and `coeus-hephaestus` name stack members. The agent instruction set now states the rule directly: "A stack member's name never appears in another member's crate, module, feature, or item names: `<host>-<sibling>` names an adapter after its dependency — implementation-source naming — and reuses the sibling's identity for a second concept, so `git grep <sibling>` matches both, and the name rots on the sibling's rename or replacement." A crate is named for its concern; the dependency is a manifest fact.
- **Scope:** the two sibling-named crates. `coeus-cuda`, `coeus-rocm`, `coeus-metal`, and `coeus-wgpu` name vendors rather than stack members, and their thinning is already decided by [ADR 0071](docs/adr/0071-provider-owned-accelerator-backends.md) — this item does not re-open that; it addresses the names ADR 0071 leaves standing.
- **Recommended target (for the ADR this item drafts):** name each crate for the concern it owns — the CPU array adaptation and the accelerator adaptation — so the manifest, not the crate name, records which provider supplies it. A rename is a mechanical transform over every call site in one change (no re-export bridge). This is `[major]`: published package/import names are consumer-facing contracts, so the rename requires a migration note.
- **Why filed rather than executed now:** it is `[arch]`, so an ADR with a recommended option is its first planning step, and it touches the crate topology containing the completed staggered binding (a closed item this board has since compacted away); the rename remains separate from that delivered behavior.
- **Last-update:** 2026-09-06.

## COEUS-ALLOC-BUDGET-INTERMITTENT — the scatter_add allocation budget fails intermittently on Linux CI [patch] — todo

- **Observed:** `scatter_add_allocation_count_is_independent_of_index_size` failed once on a pull-request run with `small=7, large=11`. The same commit passes: 12 of 12 local runs on Windows, and the `Tests` job on `main` three runs in a row. The pull request it failed on changed one `timeout-minutes` line in `python-release.yml`, which that job does not run, so the change cannot be the cause.
- **Why this is not "just flaky":** the test is built to be deterministic and documents the reasoning. `CountingAllocator` leaves `realloc` and `alloc_zeroed` to the `GlobalAlloc` defaults so that collection growth routes through `alloc` and is counted rather than hidden by the system's in-place paths -- "slightly pessimistic and never optimistic". Under that design the count should not vary by platform, and it did.
- **What the numbers rule out:** the difference is 4 allocations for a 64x larger workload (128 elements against 8192). Per-slice allocation, the defect the test exists to catch, would give roughly 16 against 256, not 4. A logarithmic growth pattern fits better -- log2(64) is 6 -- which points at a `Vec` grown by pushing somewhere on the path rather than a returned per-element allocation.
- **Not reproducible from Windows.** Twelve runs, no failure. This needs a Linux runner, ideally the failing job re-run with the counter printed at each step of the measured window rather than only at its ends.
- **Two candidates, neither confirmed:** a size-dependent `Vec` growth inside the kernel that the local allocator absorbs and glibc's does not; or an allocation from outside the kernel entering the measured window, which the global counter cannot distinguish from the kernel's own.

## ATLAS-COEUS-BACKEND-045 [minor] [arch]

Outcome: remove the forked matmul implementation and unseal `ComputeBackend`.
Acceptance: matmul reaches every provider through one generic dispatch; no
vendor crate carries a matmul kernel. ADR-0066.

- [x] Unseal `ComputeBackend`: delete `coeus_core::backend::private`, the supertrait, and all five `Sealed` impls. The implementor set is cross-crate (one per vendor), and the marker was publicly re-exported, so the seal was both wrong and inoperative.
- [x] Add the `coeus-hephaestus` `matmul` family (`MatmulProvider`, `MatmulBackend`, `matmul`) over `hephaestus_core::DenseProductOps`. No Hephaestus change was needed: the seam and all four vendor impls already existed.
- [x] Migrate `CudaBackend` and `WgpuBackend` onto the seam; delete `coeus-cuda/src/kernels/launch_matmul.rs`, `backend/ops/math/`, `coeus-wgpu/src/kernels/matmul.rs`, and `backend/ops/matmul.rs`. Route the duplicate public `coeus_wgpu::matmul` through the same dispatch.
- [x] Declare `MatmulProvider<f32>` for `MetalProvider` and `RocmProvider`.
- [ ] **Blocked — needs upstream Hephaestus seams.** `HephaestusBackend<P>` still lacks `PoolOps` and `UnfoldFoldOps`, so it does not satisfy `BackendOps<f32>` and Metal/ROCm remain partial. `hephaestus-core` has no pooling or sliding-window device trait; matmul was completable only because `DenseProductOps` already existed. Re-open trigger: a `PoolOps<D, T>` and `UnfoldFoldOps<D, T>` seam in `hephaestus-core` with CUDA/WGPU/Metal/ROCm impls. Until then ~2.8k lines of pool and ~0.9k–1.0k lines of unfold/fold per vendor crate stay forked; the device primitive both dialects already agree on is a 1-D launch over `numel(output)` at block/workgroup 256 taking per-operand layout descriptors plus 4 (pool) or 9 (unfold/fold) `u32` params.
- [ ] Device-side equivalence of the provider matmul kernel against the deleted consumer kernel is unverified: no GPU adapter on the development host. Re-open trigger: a GPU-capable CI run of the unmodified matmul parity suites.
- [ ] `cargo semver-checks` reports a pre-existing major break in `coeus-hephaestus` against the published 0.10.0 baseline — five associated types on `ElementwiseProvider`, `ScalarPowerProvider`, and `ReductionProvider` added since publish. Unrelated to this item; needs a version decision before the next release.

## COEUS-HEPHAESTUS-CUDA-001 [major] [arch]

- [x] Route CUDA elementwise/scalar-power/reduction/scan through the generic Coeus-Hephaestus bridge with provider operation bundles on `CudaBackend` (`ElementwiseProvider<f32|i32>`, `ScalarPowerProvider<f32|f64>`, `ReductionProvider`, plus the existing parameterized/random/rotate-half/ cross-entropy bundles).
- [x] Add the zero-copy `HephaestusStorage::from_arc` seam so `CudaStorage<T>` (`Arc<CudaBuffer<T>>`) shares the identical device handle with the bridge; add `From<HephaestusBackendError> for CudaBackendError` that preserves the historical `UnsupportedRank` contract for axis reductions.
- [x] Delete the cloned NVRTC elementwise fallback layer (`backend/ops/math/elementwise/*`) and the contiguous/strided launchers (`kernels/launch_ops/*`); remove the public launch helper re-exports.
- [x] Run the strict gate under the overlay with the `cuda` feature: check (0 warnings), 25 lib + 99 parity + 2 doctests pass, strict clippy `-D warnings` rc=0, fmt + `git diff --check` clean, stub path green.
- [ ] Restore `f64` elementwise once hephaestus-cuda implements the six comparison `TypedBinaryExpr<CudaC, f64>` operations (recorded in `docs/gap_audit.md`); `f64` scalar-power already routes through the bridge.

## COEUS-HEPHAESTUS-WGPU-001 [major] [arch]

- [x] Route WGPU reduction/scan through the generic Coeus-Hephaestus bridge: `WgpuBackend` declares `ReductionProvider` with `type AxisOperations = WgpuAxisReductionOps` and `type ScanOperations = WgpuScanOps`, and its `coeus_ops::ReductionOps` impls delegate through `HephaestusBackend<WgpuBackend>` over the same `Arc<WgpuBuffer<T>>` handles via `HephaestusStorage::from_arc`.
- [x] Delete the duplicated rank-2 layout/axis conversion and free-function dispatch helpers (`provider_layout`, `provider_axis`, `dispatch_scan`, `dispatch_reduction`) from `backend/ops/impls/reduction.rs` (301 → 84 lines); the fused reduction path (`kernels/reduce.rs` + `evaluate_fused_reduce`) is unchanged.
- [x] Add `From<HephaestusBackendError> for WgpuBackendError` preserving the historical `Validation(BackendError::UnsupportedRank { operation: "reduction", max_rank: 2 })` wire contract via label normalization.
- [x] Run the strict gate under the overlay: check rc=0 with 0 code warnings (lib, tests, benches), strict clippy `-D warnings` rc=0, fmt + `git diff --check` clean, doc tests 5/5. The 5 storage unit tests and the device integration suite fail only with the pre-existing `AdapterUnavailable` hardware gate (verified identical on the parent commit; GPU contract-test execution is external).
- [x] Record the replacement and external migration contract in ADR 0065.
- [x] Run focused native Nextest (17/17), doctests (0/0), format, warning-denied Clippy, and diff hygiene; physical-device execution is not claimed.
- [ ] Run exact-head hosted provider contracts after Hephaestus lands; report physical-device and local overlay limitations.

## COEUS-AUTOGRAD-LP-NORM-PROVIDER-001 [major] [arch]

- [ ] Collect workspace-doctest evidence against the delivered revision for [COEUS-AUTOGRAD-LP-NORM-PROVIDER-001](docs/backlog.md#coeus-autograd-lp-norm-provider-001).
- [ ] Run SemVer against the intended baseline, classify existing versus item-specific breaks, and record the result in the [backlog item](docs/backlog.md#coeus-autograd-lp-norm-provider-001).

## COEUS-AUTOGRAD-PROD-PROVIDER-001 [minor] [arch]

- [x] Confirm `coeus_ops::prod` copies the full input to host and `ProdNode::backward` rebuilds an input-sized host gradient.
- [x] Confirm existing Leto and Hephaestus product-axis providers cover CPU, WGPU, CUDA, ROCm, and Metal dispatch.
- [x] Record the provider-owned product contract in ADR 0057.
- [x] Route global product through provider axis reductions and one final scalar read.
- [x] Rewrite exact zero-aware product backward using provider elementwise and reduction composition; retain only provider tensors in the node.
- [x] Add zero-free, one-zero, multi-zero, non-unit-seed, strided, and COW differential coverage plus host-residue checks.
- [x] Run format, warning-denied Clippy, focused/full Nextest, and doctests.
- [x] Run the public-surface check; it reports the same three pre-existing 0.9.0 baseline failures and no product-specific failure.
- [x] Run the exact hosted backend provider gates on the merged head: Backend parity run `31052471989` passes Metal, ROCm, CUDA, and WGPU.
- [x] Restore the committed Git-source lock and pass locked provider-package checks and warning-denied provider Clippy.
- [ ] Run the remaining workspace doctests and exact SemVer check. The exact shared-target full Ops Nextest run passes 208/208 after the initial wrapper timeout was isolated to compilation under concurrent Cargo locks.

Status: merged at `d775cd90`; locked provider verification and full Ops
Nextest are green. Workspace doctests and the documented SemVer baseline
residual remain open.

## COEUS-WGPU-ELEMENTWISE-TREE-001 [patch] [arch]

- [x] Move WGPU elementwise and scalar-power provider dispatch out of the `ops` manifest into the named `elementwise` module.
- [x] Move provider activation metadata, strided capability checks, and the dynamic-to-const-rank Leto layout bridge into the named routing leaf.
- [x] Replace the duplicated activation classification with one metadata table and reject signed-stride narrowing before Leto dispatch.
- [x] Remove the hidden parent-import dependency from the WGPU `matmul` and `pool` leaves by declaring their required crate modules explicitly.
- [x] Record the module boundary and preserved Leto/Hephaestus dispatch in ADR-0058.
- [x] Record the routing leaf and signed-stride safety boundary in ADR-0059.
- [x] Pass format, locked all-target WGPU compilation, and warning-denied WGPU Clippy.
- [x] Pass the exact-head hosted provider contract gate and merge the architectural increment in PR #303 at `af2c86ee`.

Status: local refactor is complete and statically preserves provider routing.
The routing leaf now owns one activation metadata table and checked
const-generic Leto layout conversion. The affected WGPU package passes locked
all-targets check, warning-denied Clippy, and five doctests. The native WGPU
nextest collection exceeded the local wrapper bound while adapter-dependent
tests were still running; no local native-test pass is claimed for this
increment.
The local WGPU suite compiled 151 tests; 35 host-side tests passed and 116
adapter-dependent tests were blocked by `AdapterUnavailable` on this host.
Final-head provider run `31113333932` on `5034d8f0` passes WGPU job
`92656480115`, CUDA job `92656480141`, ROCm job `92656480117`, and Metal job
`92656480221`; required-device CUDA and ROCm jobs were skipped because
hardware execution was not requested. The recurring `recurseml/analysis`
status failed independently of the provider-contract workflow.

## COEUS-FROBENIUS-NORM-PROVIDER-001 [patch] [arch]

- [x] Replace the rank-3-and-higher host fold in `coeus_ops::frobenius_norm_batched` with provider elementwise, reduction, and square-root composition.
- [x] Preserve rank-2 scalar behavior, batched output shape, strided-input materialization, and source-storage immutability.
- [x] Record provider ownership and the no-host-staging boundary in ADR-0060.
- [x] Add analytical contiguous, rank-4, and strided CPU coverage.
- [x] Pass locked workspace all-targets check, warning-denied `coeus-ops` Clippy, full `coeus-ops` Nextest, focused Frobenius Nextest, and `coeus-ops` doctests.
- [ ] Run exact-head hosted WGPU, CUDA, ROCm, and Metal provider contracts.

Status: implementation is complete locally and the parent host-staging queue
remains in progress for other loss and norm families. The provider-composed
slice passes `coeus-ops` Nextest 209/209, including 7 Frobenius tests, and 23
doctests. No performance or resident-memory claim is made without controlled
measurements.

## COEUS-REGISTRY-PACKAGE-1 [patch] — Owner: Codex `/root`

- [x] Bind Moirai, Mnemosyne, and Themis imports to their published packages.
- [x] Refresh the exact external lock graph after Apollo `816a5c89`, Hephaestus `48fba669`, Moirai `b7988419`, and Mnemosyne `213fead9` merge.
- [ ] Pass exact-head provider CI and merge the release-preparation branch.
- [ ] Publish reusable crates in dependency order through Trusted Publishing.

## ATLAS-COEUS-SAFETY-002 Native COW seam consolidation [arch][patch]

- [x] Route native Coeus WGPU and CUDA COW detachment through `ComputeDevice::copy_buffer`.
- [x] Add WGPU and CUDA value-semantic regressions covering detached and retained device buffers.
- [x] Synchronize the changelog, gap audit, and ADR 0036.

Acceptance: the native WGPU and CUDA storage paths contain no provider-local
COW transfer implementation, and their focused backend contract suites verify
that detachment preserves values in both buffers. Local WGPU compilation,
warning-denied Clippy, doctests (3/3), and Nextest (104/104) pass. Local CUDA
feature compilation and warning-denied library Clippy pass; CUDA Nextest and
doctests are blocked by the Windows MinGW linker error `cannot find -lcuda`.
Hosted exact-head run `30339683483` passed the CUDA provider contracts job
`90212208770`, WGPU provider contracts job `90212208755`, ROCm provider
contracts job `90212208702`, and Metal provider contracts job `90212208797`.
The required-device ROCm job `90212209211` was skipped because no hosted AMD
runner was dispatched; no physical-device execution claim is made.

## CUDA attention kernel tree [arch]

Historical topology increment. ADR-0047 and
`COEUS-ATTENTION-PROVIDER-001` supersede it by deleting the complete local
CUDA attention kernel tree and routing the backend through Hephaestus.

## CUDA attention launch ABI [patch] [arch]

Historical safety increment. ADR-0047 and
`COEUS-ATTENTION-PROVIDER-001` supersede its consumer-owned launch ABI by
deleting the local launcher and using Hephaestus validation and dispatch.
