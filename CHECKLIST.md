# Coeus Development Roadmap Checklist

## ATLAS-COEUS-HEPHAESTUS-CUDA-GELU-PARITY-001 [minor]

- [x] Route CUDA `Gelu` and `GeluGrad` through the existing Hephaestus exact-
      erf marker kernels for contiguous and runtime-shaped strided layouts.
- [x] Select the existing CUDA/Leto forward and gradient parity tests in the
      backend-parity workflow.
- [ ] Record exact-head WGPU, CUDA, ROCm, and Metal CI evidence without making
      an unmeasured runtime or resident-memory claim.

Owner: Codex on `codex/coeus-cuda-common-activation-parity`; claimed scope:
`crates/coeus-cuda/src/backend/ops/math.rs`,
`crates/coeus-cuda/tests/cuda/parity/unfold_fold.rs`,
`.github/workflows/backend-parity.yml`, `docs/backlog.md`, `CHECKLIST.md`, and
`docs/gap_audit.md`. The peer-owned WGPU pool change currently owns the dirty
`CHANGELOG.md`; it remains outside this increment. Peer reduction /
backend-error files also remain outside this claim.

Implementation head `a8dcc51c` is complete; hosted exact-head evidence is
pending. The local locked CUDA package check is blocked before compilation by
the pre-existing Atlas Eunomia repository/worktree package collision.

## ATLAS-COEUS-SAFETY-003 Uninitialized COW replacement [arch][minor][perf]

- [x] Keep ordinary WGPU, CUDA, and generic Hephaestus storage construction
      zero-initialized.
- [x] Allocate only COW replacement buffers through
      `ComputeDevice::alloc_uninitialized_with_hint` before the complete
      device-local copy.
- [x] Extend the generic Hephaestus test device and preserve detached and
      retained value semantics.
- [x] Run exact-head WGPU, CUDA, ROCm, and Metal Coeus provider/consumer CI
      after the Hephaestus provider seam merges.

Acceptance: every changed COW path fully overwrites the replacement through
`ComputeDevice::copy_buffer` before it can be read; ordinary defined-content
allocation remains zero-initialized; focused value-semantic tests pass. This is
static allocation-path and structural memory-bandwidth evidence, not a runtime
speed or resident-memory claim. The local CUDA feature check passes; hosted
provider CI is required for device execution.

Implementation owner: Codex on `codex/coeus-uninitialized-cow-copy`; ADR 0037.

Provider prerequisite: Hephaestus PR #136 merged at `da785b53`; the consumer
branch now targets the merged `ComputeDevice::alloc_uninitialized_with_hint`
contract in hosted CI.

Hosted exact-head Coeus run `30345002409` passed CUDA job `90229046185`, WGPU
job `90229046271`, ROCm job `90229046258`, and Metal job `90229046242`. The
required-device ROCm job `90229047328` was skipped because no hosted AMD
runner was dispatched; no physical-device execution claim is made.

## ATLAS-COEUS-SAFETY-002 Native COW seam consolidation [arch][patch]

- [x] Route native Coeus WGPU and CUDA COW detachment through
      `ComputeDevice::copy_buffer`.
- [x] Add WGPU and CUDA value-semantic regressions covering detached and
      retained device buffers.
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

## ATLAS-COEUS-SAFETY-001 Device-local COW increment [patch]

- [x] Replace Hephaestus storage COW host staging with one source-tier device
      allocation and the shared completed device-local copy seam.
- [x] Add a generic storage regression proving copied values, retained memory
      tier, one device copy, and zero COW downloads.
- [x] Synchronize the changelog, gap audit, and ADR 0036.

Evidence: `cargo check` and the focused `coeus-hephaestus` Nextest storage
contract pass under the local Atlas overlay. This is a static allocation-path
and value-semantic result; no runtime speedup claim is made without a matched
device benchmark. The infallible `StorageMut::make_unique` boundary remains a
separate typed-error migration item. Hosted exact-head run `30336317894`
passed the CUDA provider contracts job `90201872163`, WGPU provider contracts
job `90201872262`, ROCm provider contracts job `90201872299`, and Metal
provider contracts job `90201872213`. The required-device ROCm job
`90201873084` was skipped because no hosted AMD runner was dispatched; no
physical-device execution claim is made. The external `recurseml/analysis`
status returned its recurring analyzer error and is not repository-owned
verification.

## ATLAS-COEUS-HEPHAESTUS-CUDA-ACTIVATION-PARITY-001 [minor]

- [x] Route CUDA `GeluTanh`, `GeluTanhGrad`, `Softplus`, and `SoftplusGrad`
      through the existing Hephaestus marker kernels for contiguous and
      runtime-shaped strided layouts.
- [x] Add CUDA/Leto value-semantic forward and gradient parity coverage and
      select the tests in the CUDA CI contract job.
- [x] Record exact-head CUDA, WGPU, ROCm, and Metal CI evidence and preserve
      the existing typed unsupported-operation behavior for operations outside
      the common marker seam.

Owner: Codex on `codex/coeus-cuda-common-activation-parity`; completed at
`8a38f392`. Claimed scope:
`crates/coeus-cuda/src/backend/ops/math.rs`,
`crates/coeus-cuda/tests/cuda/parity/unfold_fold.rs`,
`.github/workflows/backend-parity.yml`, `CHECKLIST.md`, and
`docs/gap_audit.md`. Peer-owned reduction and backend-error files remain
outside this claim.

Evidence: docs-head run `30359324025` passed CUDA job `90274888940`, WGPU job
`90274889041`, ROCm job `90274889047`, and Metal job `90274888991`.
Required-device ROCm job `90274889835` was skipped because no hosted AMD
runner was dispatched. The CUDA selector executed the four new forward and
gradient tests; WGPU selected the GELU-tanh contract. No runtime speedup or
resident-memory delta is claimed without a matched benchmark.

## ATLAS-COEUS-HEPHAESTUS-LGAMMA-PARITY-001 [arch]

- [x] Route `UnaryOp::Lgamma` through the provider-owned Hephaestus WGPU,
      CUDA, ROCm, and Metal f32 implementations.
- [x] Extend CUDA, ROCm, and Metal backend suites with Leto CPU differential
      cases covering positive inputs, reflection, and gamma poles.
- [x] Replace the WGPU unsupported-operation assertion with a provider
      expression contract assertion.
- [x] Run and record exact-head WGPU, CUDA, ROCm, and Metal provider CI for the
      Hephaestus expression seam and the Coeus consumer head.

Status: complete for f32 forward dispatch. The WGPU and Metal paths use the
provider-owned Lanczos/reflection expression; CUDA and ROCm use their native
`lgammaf`/`lgamma` device functions. Hephaestus PR #118 passed WGPU
`90086428952`, CUDA `90086430178`, ROCm `90086430143`, and Metal `90086428160`.
Coeus PR #231 merged at `971fab9614b97bd708a716d01684da58fd1331ba`; its
consumer jobs passed WGPU `90088836682`, CUDA `90088836688`, ROCm `90088836731`,
and Metal `90088836675`. Required-device ROCm `90088837591` was skipped because
no hosted AMD runner was dispatched. No digamma gradient or non-f32 contract is
implied.

## ATLAS-COEUS-HEPHAESTUS-GELU-PARITY-001 [arch]

- [x] Route `UnaryOp::Gelu` and `UnaryOp::GeluGrad` through the shared
      Hephaestus ROCm and Metal f32 dispatch arms.
- [x] Extend both backend elementwise suites with Leto CPU differential cases
      over the existing activation input domain.
- [x] Run and record exact-head WGPU, CUDA, ROCm, and Metal provider CI for the
      Hephaestus expression seam and the Coeus consumer head.

Evidence: Hephaestus provider jobs CUDA `90048504061`, ROCm `90048505968`,
WGPU `90048506717`, and Metal `90048504635` passed; Coeus consumer jobs CUDA
`90061390565`, ROCm `90061390546`, WGPU `90061390522`, and Metal `90061390499`
passed. Hardware-device jobs skipped because no registered runner was
available. ADR 0028 owns the exact GELU contract.

## ATLAS-COEUS-HEPHAESTUS-ACTIVATION-TAIL-PARITY-001 [arch][minor]

- [x] Route `UnaryOp::Mish`, `MishGrad`, `Elu`, and `EluGrad` through the
      provider-owned Hephaestus ROCm and Metal f32 strided kernels.
- [x] Extend the CUDA contiguous and strided launch expressions with ELU and
      its gradient, preserving the existing Mish expressions.
- [x] Extend WGPU, CUDA, ROCm, and Metal contracts with Leto CPU differential
      coverage for forward and gradient activation paths.
- [x] Run and record exact-head WGPU, CUDA, ROCm, and Metal provider CI for
      the provider and consumer revisions.

Acceptance: all four backends expose the same unparameterized f32 Mish and ELU
forward/gradient operations; integer providers retain typed unsupported
operation errors; backend values match the Leto CPU oracle over signed inputs
including the zero branch boundary. The existing strided/device-resident
kernel paths remain in use; no runtime performance or resident-memory delta is
claimed without a controlled benchmark. ADR 0038 owns the contract.

Status: complete for the unparameterized f32 scope. Targeted exact-head Coeus
run `30353984154` passed CUDA job `90257861209`, WGPU job `90257861154`, ROCm
job `90257861218`, and Metal job `90257861119`; required-device ROCm job
`90257861858` was skipped because no hosted AMD runner was dispatched. The
WGPU and CUDA selectors execute the new ELU forward and gradient contracts.
The external `recurseml/analysis` status returned its recurring analyzer error
and is not repository-owned verification.

## ATLAS-COEUS-HEPHAESTUS-ERROR-FUNCTION-PARITY-001 [arch]

- [x] Route `UnaryOp::Erf` and `UnaryOp::Erfc` through the provider-owned
      Hephaestus ROCm and Metal f32 dispatch arms.
- [x] Extend both backend elementwise suites with Leto CPU differential cases
      over the existing bounded real-valued input domain.
- [x] Run and record exact-head WGPU, CUDA, ROCm, and Metal provider CI for the
      Hephaestus expression seam and the Coeus consumer head.

Evidence: local Coeus test-target compilation and `cargo nextest run -p
coeus-rocm -p coeus-metal` pass 6/6 with the Hephaestus error-function branch
and the merged Leto comparison-marker revision temporarily overlaid. The
temporary manifest and lock overlays are restored. Coeus run `30282267102`
passed CUDA job `90031346303`, Metal job `90031346354`, ROCm job `90031346411`,
and WGPU job `90031346421`; required-device ROCm job `90031346992` skipped.
Hardware-device execution remains a separate evidence tier and is not claimed
when the required-device lane skips.

## ATLAS-COEUS-BUILD-001 Locked provider source graph [patch]

- [x] Verify the current manifest graph and active peer provider declarations.
- [x] Regenerate or retain only the lockfile entries produced by those
      declarations; remove transient path-source identity if present.
- [x] Run locked metadata against the current provider graph; the affected
      package gates remain owned by the source items below.
- [x] Record exact resolver, build, test, and documentation coverage limits.

Evidence: the current Coeus manifest declares Cutile from the NVlabs Git
source, while the checked-in lock previously lacked the corresponding source
identity. The live lock delta records that Git source and current local Hermes
provider versions. Hephaestus currently declares the same Cutile Git source in
its clean peer worktree; the previous `/tmp/cutile-rs` path blocker is no
longer present. `cargo metadata --locked --offline --no-deps` resolves 13
workspace members and full `cargo metadata --locked --offline` resolves 373
packages. The broad `cargo fmt --all -- --check` remains red on existing
formatting drift in Coeus and sibling path workspaces. Root patches now force
 Git-sourced Eunomia, Aequitas, Themis, and Hermes dependencies onto the local
 Atlas instances; the locked `coeus-wgpu` library check passes after removing
 the duplicate Themis identity. Full offline metadata now resolves the
 workspace graph with the Hermes patch; the all-target compile remains
 unverified while concurrent MSYS2 jobs mix stable and nightly artifacts in
 the shared target directory.

## ATLAS-STRUCTURE-001 Workspace crate hierarchy [arch]

- [x] Move all 13 workspace crates from the repository root into `crates/`
      with `git mv`, preserving package names and source contents.
- [x] Update the root workspace members, local dependency paths, release
      workflow manifest path, documentation, PM paths, and source references.
- [x] Confirm `cargo metadata --locked --no-deps` resolves every workspace
      manifest under `crates/` and the stale-path scan is empty.
- [x] Confirm the staged diff contains only content-preserving crate renames,
      path consumers, and synchronized architecture records.
- [x] Run the applicable structural, formatting, compilation, and test
      discovery gates; record exact peer-owned blockers for the remaining
      full matrix.

Evidence: all 13 packages resolved under `crates/`, the locked workspace check
finished successfully before the peer CUDA dependency cutover, and no root
crate directories or stale repository-local crate paths remain. The earlier
format run reached existing peer formatting drift; the current format,
metadata, and nextest discovery attempts stop at the peer `coeus-cuda` manifest
because its optional `cutile-rs` paths are absent. The earlier broad nextest
attempt also recorded the missing CUDA linker and existing WGPU test-compilation
errors. The peer `.cargo/config.toml`, `Cargo.lock`,
`crates/coeus-cuda/Cargo.toml`, and `crates/coeus-cuda/build.rs` changes remain
outside this relocation.

## ATLAS-ATTENTION-PERF-001 CPU attention scratch reuse [perf]

- [x] Replace the per-query `d_attn_row` allocation with the corresponding
      disjoint row of the existing `d_scores` scratch buffer.
- [x] Preserve the analytical backward formula and existing value-semantic
      differential coverage without changing the public API.
- [x] Run the focused package gates or record the peer dependency blocker; do
      not claim a measured speedup without a controlled benchmark baseline.

Evidence: the current backward kernel allocates `Vec<T>` once per query task;
the existing `d_scores` buffer already owns one disjoint row per task and can
hold the intermediate dot products before the softmax derivative overwrites
them. The allocation is removed without changing the reduction order. Direct
rustfmt and staged diff checks pass. `cargo check --locked -p coeus-ops` is
blocked before compilation by the peer CUDA manifest's absent
`/tmp/cutile-rs/cuda-async/Cargo.toml`; no package test or benchmark result is
claimed.

## ATLAS-CORE-SAFETY-001 Parallel pointer auto-trait bounds [patch]

- [x] Constrain `SendPtr` and `SendPtrMut` unsafe `Send`/`Sync` impls to the
      pointee capabilities required by cross-thread use.
- [x] Preserve existing Coeus scalar pointer users and document the remaining
      disjoint-access safety obligation.
- [x] Run direct formatting and diff checks; record the peer Cargo blocker if
      package compilation remains unavailable.

Evidence: `crates/coeus-core/src/ptr.rs` currently marks both public raw-pointer
wrappers `Send + Sync` for every `T`, which overstates the thread-safety
contract for non-thread-safe pointee types. `SendPtr<T>` now requires `T: Send`
for movement and `T: Sync` for shared reads; `SendPtrMut<T>` requires `T: Send`
for movement and shared disjoint writes. Direct rustfmt and diff checks pass.
The focused Cargo check is blocked before compilation by the peer CUDA manifest's
absent `/tmp/cutile-rs/cuda-async/Cargo.toml`.

## WGPU layout and dispatch failure boundary [arch]
- [x] Add the checked `GpuLayoutInfo` SSOT constructor with typed rank,
      stride-rank, offset, shape, and stride overflow errors.
- [x] Add one backend error associated type and make operation dispatch
      return `Result` through the shared trait seam.
- [x] Convert `GpuLayoutInfo` and the WGPU dispatch-grid/ABI conversions to
      typed checked constructors; migrate elementwise and matmul operation
      families without adapters or silent no-ops.
- [x] Update CPU, CUDA, WGPU, and high-level callers together; verify value
      semantics, typed negative broadcast failure, and public API documentation.
- [ ] Run the full affected package matrix after the peer fallible-operation
      migration completes; no green matrix result is claimed while peer callers
      remain incomplete.

Decision: ADR-0020 selects a backend-associated typed error plus fallible
operation traits. An operation-local `Option`/early return is rejected because
it can leave output storage uninitialized or stale while hiding the violated
WGSL ABI contract.

Increment evidence: `ComputeBackend::Error` now carries the shared typed
validation conversion; elementwise and matmul traits return `Result`; CPU
maps Leto failures, CUDA and WGPU preserve provider-specific errors, and
high-level arithmetic, unary, shape, and matmul callers propagate them.
`ReductionOps::reduce` now returns the backend-associated result; CPU, CUDA,
and WGPU implementations plus public core callers use it. The existing
infallible autograd/NN boundary consumes validated results with explicit
invariant messages; an error-valued graph/module API is separate breaking
work. The focused `coeus-ops` gate passes 110/110 nextest tests, 22/22
doctests, warning-denied Clippy, locked compilation, and no-deps Rustdoc. The
public WGPU matmul wrapper now returns the typed result, validates ranks and
inner dimensions, and checks output element-count overflow; the public WGPU
add wrapper now returns a typed shape error instead of panicking. Coeus root
patches now collapse Git-sourced Aequitas/Eunomia/Themis/Hermes identities onto
the local Atlas providers, so the locked provider graph compiles. The locked
WGPU library check passes. The full WGPU
all-target matrix remains gated by the incomplete peer `coeus-nn`/
`coeus-autograd` fallible-operation migration.

Test-target increment: WGPU layout tests now construct `Shape`/`SmallVec`
values through their supported conversions and assert typed error fields with
guarded `matches!` patterns; WGPU parity tests handle fallible unary and direct
backend calls explicitly, and tensor parity tests handle fallible assign
operations. Direct nightly rustfmt and diff checks pass. The Coeus provider
graph no longer stops at the peer Leto bound failure; the locked `coeus-ops`
check, 110-test nextest run, 22 doctests, warning-denied Clippy, and no-deps
Rustdoc pass. WGPU all-target verification remains outside this manifest/lock
integration increment.

## Axis-reduction error propagation [major] [arch]
- [x] Change `ReductionOps::reduce` to return the backend-associated typed
      `Result`, then migrate CPU, CUDA, and WGPU implementations without a
      unit-returning adapter or silent fallback.
- [x] Replace the CPU reduction `expect` with the existing Leto-to-backend
      error mapping and validate WGPU layout, axis, output count, and dispatch
      conversions before queue submission.
- [x] Migrate direct public `sum`, `mean`, `sum_axis`, `mean_axis`, `max_axis`,
      and `min_axis` callers across autograd/NN with explicit invariant
      boundaries; the current infallible graph/module contract is recorded as
      separate breaking work. Fused reduction and default index/cumulative
      reductions remain separate follow-up items.
- [x] Run direct format/diff checks and the affected package gates; direct
      checks, locked metadata, `coeus-ops` compilation/tests/docs, and
      warning-denied Clippy pass after the provider-identity cutover.

Claim: Codex `/coeus`; scope is the shared axis-reduction seam and its direct
CPU/CUDA/WGPU/public callers. The Coeus root provider patches unify the local
Aequitas/Eunomia identities that previously caused the Leto
`Quantity<T>::in_unit` trait-bound failure. The locked `coeus-ops` check,
110/110 nextest tests, 22/22 doctests, warning-denied Clippy, and no-deps
Rustdoc now pass; the remaining WGPU all-target and infallible autograd/NN
residuals are separate migration work.

Unary dispatch increment: both unary kernel entry points now return typed
backend errors, use checked layout conversion, route `lgamma` through the
provider-owned Hephaestus marker, and validate workgroup rounding before
converting to the WGPU `u32` dispatch ABI.
The new unit tests cover supported rounding, arithmetic overflow, ABI range,
and unsupported-operation behavior without a device. Direct nightly rustfmt
and `git diff --check` pass; the locked `coeus-ops` check and focused tests pass
after the provider-identity cutover.

Binary dispatch increment: contiguous and broadcasting paths now return typed
results, validate all three layout descriptors before device initialization,
and use the checked workgroup-count helper. The public WGPU `add` wrapper and
the shared elementwise backend seam propagate the failure. Direct nightly
rustfmt and `git diff --check` pass; the locked `coeus-ops` check and focused
tests pass after the provider-identity cutover.

## WGPU pool1d dispatch mode ownership [patch]
- [x] Replace the forward dispatcher’s mixed forward/backward mode enum with
      a forward-only mode type.
- [x] Preserve shader source selection and public pool1d launch functions
      without adapters or duplicate shader bodies.
- [x] Run format, diff, static residual, and package gates; record the
      preserved peer dependency-resolution blocker if it remains.

Evidence: `ForwardPoolKind` removes the forward dispatcher’s backward-only
states without changing shader source generation or public pool1d launch
functions. The pool1d residual scan is clean; format and diff checks pass.
The current locked WGPU library check and warning-denied Clippy pass. The
all-targets check reaches compilation and is blocked later by the peer
`coeus-nn` fallible-operation migration; that residual is tracked in the
WGPU layout and dispatch failure boundary item above.

## Fused operation-tag tree [arch]
- [x] Split the 625-line operation-tag module into an op-tags manifest and a
      unary trait subtree with elementary, transcendental, and activation leaves.
- [x] Preserve all public tag names, `UnaryOpTag` generic dispatch, WGSL
      rendering, and existing binary/leaky-relu ownership without adapters.
- [x] Verify all leaves remain below 500 lines and run format and diff checks;
      record package-gate limitations without claiming compiled or test output.

Evidence: the operation-tag manifest is 9 lines and unary leaves are 27, 125,
180, and 294 lines. Format and diff checks pass. Package gates remain blocked
by the unrelated dirty provider dependency manifest; no compiled or test result
is claimed for this slice.

## CUDA attention kernel tree [arch]
- [x] Split the 567-line attention kernel module into a manifest and cohesive
      validation, source, forward, backward, and test leaves.
- [x] Preserve the public launch functions, checked dimensions, device-buffer
      ownership, and explicit CPU capability boundary without adapters.
- [x] Verify all leaves remain below 500 lines and run format and diff checks;
      record the package-gate blocker without claiming compiled or test output.

Evidence: leaves are 12, 81, 92, 101, 135, and 149 lines. Format and diff
checks pass. The prior package-gate dependency-resolution blocker is resolved
in the current locked graph; no new CUDA feature-test result is claimed by
this topology slice.

## CUDA convolution backend tree [arch]
- [x] Split the former 614-line convolution backend into a manifest and
      forward, backward, and transposed-convolution leaves.
- [x] Preserve the existing checked-count, layout, fallback, and storage
      ownership paths without compatibility modules or duplicate implementations.
- [x] Verify every leaf remains below 500 lines and run format, diff,
      feature-enabled check/Clippy, default Nextest, feature rustdoc, and the
      CUDA-feature linker boundary.

Evidence: convolution leaves are 36, 186, 236, and 181 lines. Feature check,
warning-denied Clippy, and rustdoc pass; default package Nextest passes 3/3
with zero skipped in 0.054 seconds. CUDA-feature Nextest reaches the Windows
GNU linker but cannot link because `-lcuda` is absent from
`/usr/local/cuda-11.3/lib64/`; no feature test execution is claimed.

## CUDA elementwise backend count and failure boundary [patch]
- [x] Replace unary/binary output `Iterator::product()` with the shared
      checked-count SSOT before native dispatch or fallback.
- [x] Convert Hephaestus contiguous and strided `Result` failures into the
      existing explicit CPU capability path; remove provider-side panics.
- [x] Verify format, diff, feature-enabled check, warning-denied Clippy,
      default Nextest, feature rustdoc, and the feature-linker boundary.

Evidence: feature-enabled check and warning-denied Clippy pass; default
package Nextest passes 3/3 with zero skipped in 0.114 seconds; feature
rustdoc passes in 3.55 seconds. CUDA-feature Nextest reaches the Windows GNU
linker but cannot link because `-lcuda` is absent from
`/usr/local/cuda-11.3/lib64/`; no feature test execution is claimed.

## CUDA fused-dispatch ABI [patch] [arch]
- [x] Validate checked output counts/grids, contiguous output indexing,
      broadcast contracts, input/output storage bounds, and null inputs before
      dynamic CUDA source compilation and launch.
- [x] Move layout storage-length validation into the shared kernel-validation
      SSOT and retain zero-copy layout serialization with explicit safety proof.
- [x] Replace local block/grid narrowing with the canonical checked grid seam;
      remove the input-dependent `CString` unwrap at kernel lookup.
- [x] Add overflow and layout-storage boundary regressions; verify format,
      diff, feature-enabled check, warning-denied Clippy, default Nextest, and
      feature rustdoc.

Evidence: feature-enabled check and warning-denied Clippy pass; default
package Nextest passes 3/3 with zero skipped in 0.055 seconds; feature
rustdoc passes in 3.09 seconds. CUDA-feature Nextest reaches the Windows GNU
linker but cannot link because `-lcuda` is absent from
`/usr/local/cuda-11.3/lib64/`; no feature test execution is claimed.

## CUDA transposed-convolution launch ABI [patch] [arch]
- [x] Validate 1-D and 2-D transposed-convolution dimensions, checked input/
      weight/output products, optional bias capacity, and `u32` ABI values
      before native compilation or dispatch.
- [x] Reuse the shared checked 1-D grid launcher while preserving native
      gather kernels, output-shape ownership, and device-buffer ownership.
- [x] Restrict native dispatch to rank-correct, contiguous, offset-zero
      layouts with matching batch/channel contracts; use overflow-safe device
      arithmetic for dilation products and coordinate subtraction.
- [x] Add pure checked-product regressions and the co-located ADR.
- [x] Verify format, diff, feature-enabled check, warning-denied Clippy,
      default Nextest, and feature rustdoc.

Evidence: the transposed-convolution launchers contain no input-dependent
dimension or work-count narrowing and no unchecked storage product. Feature
check and warning-denied Clippy pass; pure product tests compile with the
feature build. CUDA-feature Nextest remains subject to the Windows GNU linker
environment and is not claimed unless it executes.

## CUDA unfold/fold launch ABI [patch] [arch]
- [x] Split the launcher into manifest, shared dispatch/source, validation,
      and 1-D/2-D leaves; move the CUDA source into a co-located asset.
- [x] Replace panic-based parameter narrowing and unchecked products with
      checked formulas, exact shape contracts, layout/storage bounds, output
      alias checks, and the shared 1-D grid seam.
- [x] Add pure validation coverage for sliding-window formulas and invalid or
      overflowing parameters; preserve native kernels and device ownership.
- [x] Add the co-located ADR and verify format, diff, feature-enabled check,
      warning-denied Clippy, default Nextest, and feature rustdoc.

Evidence: all unfold/fold Rust leaves are below the 500-line target; no local
`as u32`, unchecked output product, panic-based parameter conversion, or
unchecked output-width derivation remains in the dispatch tree. Feature
check, warning-denied Clippy, and rustdoc pass; default package Nextest passes
3/3 with zero skipped in 0.193 seconds. CUDA-feature Nextest reaches the
Windows GNU linker and fails before execution because `-lcuda` is absent from
`/usr/local/cuda-11.3/lib64/`.

## CUDA attention launch ABI [patch] [arch]
- [x] Validate positive attention dimensions, checked element counts, mask
      contracts, and device-buffer lengths before native compilation or
      transient backward allocation.
- [x] Restrict native dispatch to contiguous offset-zero rank-three tensors
      with compatible shapes and supported contiguous rank-one/rank-two masks;
      route unsupported layouts through the explicit CPU capability path.
- [x] Reuse checked shared 1-D grid launch validation and add pure boundary
      tests for zero, overflow, and mask-shape cases.
- [x] Add the co-located ADR and verify format, diff, feature-enabled check,
      warning-denied Clippy, and default Nextest.

Evidence: attention launch dimensions and buffer lengths are checked before
kernel compilation or transient allocation; the shared `launch_1d` path has no
input-dependent grid narrowing. Pure boundary tests cover valid rank-two mask
counts, zero dimensions, mask-rank inconsistency, non-divisible heads, and
product overflow. Feature-enabled package check and warning-denied Clippy
pass; default package Nextest passes 3/3 with zero skipped in 0.171 seconds;
default doctests pass 4/4 in 14.21 seconds. CUDA-feature Nextest reaches the
Windows GNU linker and fails before execution because `-lcuda` is absent from
`/usr/local/cuda-11.3/lib64/`.

## CUDA matmul launch ABI [patch] [arch]
- [x] Reject non-rank-two, zero-sized, incompatible, or output-mismatched
      matmul layouts before native kernel compilation.
- [x] Replace both unchecked 16-wide grid conversions with the shared checked
      arbitrary-block grid helper; retain the tiled f32 kernel and buffers.
- [x] Add the co-located ADR and verify format, diff, feature-enabled check,
      warning-denied Clippy, and default Nextest.

Evidence: `launch_matmul.rs` contains no input-dependent grid narrowing or
unchecked rank indexing. Shared validation tests cover custom block widths;
feature-enabled package check and warning-denied Clippy pass; default package
Nextest passes 3/3 with zero skipped. CUDA-feature Nextest remains blocked
before execution because the Windows GNU linker cannot find `-lcuda` at
`/usr/local/cuda-11.3/lib64/`.

## CUDA pool3d launch ABI [patch] [arch]
- [x] Apply the pool-owned validation seam to 3-D average/max forward and
      backward dispatch with rank-five nonempty layouts and shape contracts.
- [x] Remove 3-D parameter, work-count, grid, and block-size narrowing while
      retaining native kernels and device-buffer ownership.
- [x] Add the co-located ADR and verify format, diff, feature-enabled check,
      warning-denied Clippy, and default Nextest.

Evidence: all 1-D, 2-D, and 3-D pooling sources contain no input-dependent
parameter/count/grid/block narrowing or unchecked shape product. Feature-
enabled package check and warning-denied Clippy pass; default package Nextest
passes 3/3 with zero skipped. CUDA-feature Nextest remains blocked before
execution because the Windows GNU linker cannot find `-lcuda` at
`/usr/local/cuda-11.3/lib64/`.

## CUDA pool2d launch ABI [patch] [arch]
- [x] Promote pooling parameter, work, layout, prefix, shape, and block-size
      validation into `crates/coeus-cuda/src/kernels/pool/validation.rs`.
- [x] Apply the shared seam to 2-D average/max forward and backward leaves and
      migrate pool1d without retaining duplicate helpers.
- [x] Add the co-located ADR and verify format, diff, feature-enabled check,
      warning-denied Clippy, and default Nextest.

Evidence: 1-D and 2-D pooling sources contain no input-dependent parameter,
count, grid, or block narrowing and no unchecked shape product. Feature-
enabled package check and warning-denied Clippy pass; default package Nextest
passes 3/3 with zero skipped. CUDA-feature Nextest remains blocked before
execution because the Windows GNU linker cannot find `-lcuda` at
`/usr/local/cuda-11.3/lib64/`.

## CUDA pool1d launch ABI [patch] [arch]
- [x] Validate positive representable pooling parameters, checked work counts
      and grids, rank-three nonempty layouts, and operation shape contracts.
- [x] Remove pool1d-local narrowing and grid/block derivation; reuse the
      shared CUDA validation seam and canonical block size.
- [x] Add the co-located ADR and verify format, diff, feature-enabled check,
      warning-denied Clippy, and default Nextest.

Evidence: `pool1d.rs` contains no input-dependent parameter/count/grid
narrowing or unchecked shape product. Feature-enabled package check and
warning-denied Clippy pass; default package Nextest passes 3/3 with zero
skipped. CUDA-feature Nextest remains blocked before execution because the
Windows GNU linker cannot find `-lcuda` at `/usr/local/cuda-11.3/lib64/`.

## CUDA optimizer launch ABI [patch] [arch]
- [x] Apply shared checked element-count, `u32`, grid, layout, and same-shape
      validation to AdaGrad, Adam, AdamW, RMSprop, and SGD.
- [x] Remove optimizer-local block-size and narrowing casts from contiguous
      and strided launches; reject unrepresentable Adam step exponents.
- [x] Add the co-located ADR and verify format, diff, feature-enabled check,
      warning-denied Clippy, and default Nextest.

Evidence: all five optimizer leaves contain no input-dependent `as u32`,
`as i32`, unchecked `numel`, or local grid/block derivation. Shared
validation tests cover shape mismatch; feature-enabled package check and
warning-denied Clippy pass; default package Nextest passes 3/3 with zero
skipped. CUDA-feature Nextest remains blocked before execution because the
Windows GNU linker cannot find `-lcuda` at `/usr/local/cuda-11.3/lib64/`.

## CUDA elementwise launch tree and ABI [patch] [arch]
- [x] Split the 530-line `launch_ops.rs` into a manifest plus contiguous and
      strided leaves, preserving all four public helper names.
- [x] Validate element counts, launch grids, layouts, and broadcast rank at
      the shared boundary; replace raw strided layout serialization with
      `bytemuck::cast_slice`; reject zero-stride output layouts before the
      generated kernel can divide by zero or alias writes.
- [x] Add the co-located ADR and verify format, diff, feature-enabled check,
      warning-denied Clippy, and default Nextest.

Evidence: `launch_ops.rs` is a manifest; contiguous and strided leaves are
219 and 327 lines. Shared validation tests cover zero-stride output layouts,
zero work, and overflow. The affected source contains no input-dependent
`as u32`, raw layout slice, unchecked elementwise grid, or family-local
validator.
Feature-enabled package check and warning-denied Clippy pass; default package
Nextest passes 3/3 with zero skipped. CUDA-feature Nextest remains blocked
before execution because the Windows GNU linker cannot find `-lcuda` at
`/usr/local/cuda-11.3/lib64/`.

## CUDA reduction launch-boundary validation [patch] [arch]
- [x] Promote shared `u32`, checked element-count, layout-fit, and grid-size
      helpers to `kernels/validation.rs`; delete the convolution-local copy.
- [x] Validate standard and fused reduction axis, expression rank, layout,
      output count, and launch grid before kernel dispatch.
- [x] Replace fused reduction's input-shape panic, unchecked casts/products,
      and raw layout serialization while preserving the native kernel path.
- [x] Add overflow regressions and the co-located ADR; verify format, diff,
      feature-enabled check, warning-denied Clippy, and default Nextest.

Evidence: reduction source contains no input-dependent `as u32`, unchecked
output product, expression-shape indexing, or panic. Feature-enabled package
check and warning-denied Clippy pass; default package Nextest passes 3/3 with
zero skipped. CUDA-feature Nextest remains blocked before execution because
the Windows GNU linker cannot find `-lcuda` at `/usr/local/cuda-11.3/lib64/`.

## CUDA layout ABI boundary [major] [arch]
- [x] Replace the public truncating `GpuLayoutInfo` conversion with one
      crate-private `TryFrom<&Layout>` seam that rejects rank mismatch, rank
      overflow, and values outside the CUDA `u32` ABI.
- [x] Serialize the `Pod` descriptor with `bytemuck::cast_slice` and migrate
      every CUDA layout consumer without retaining a compatibility wrapper.
- [x] Replace forward convolution shape products with one checked element
      count seam and retain native dispatch plus the established failure
      result.
- [x] Add boundary tests and the co-located ADR; verify format, diff,
      feature-enabled check, warning-denied Clippy, and default Nextest.

Evidence: feature-enabled package check and warning-denied Clippy pass. The
boundary tests compile and cover representable layouts, unsupported rank,
shape/stride rank mismatch, and an offset above `u32::MAX`. Default package
Nextest passes 3/3 with zero skipped in 0.053 seconds. CUDA-feature Nextest
remains blocked before execution because the Windows GNU linker cannot find
`-lcuda` in `/usr/local/cuda-11.3/lib64/`. `cargo semver-checks` against the
pre-change `HEAD` reports the two intentional removed public items and
classifies the change as major.

## CUDA convolution launch-boundary validation [patch]
- [x] Reject convolution layouts, stride/padding/dilation values, element
      counts, channel counts, and derived grid sizes that cannot be represented
      by the CUDA `u32` launch ABI.
- [x] Replace unchecked backward shape products and rank-0 channel indexing
      with checked conversion and failure-result paths while preserving the
      native device dispatch and allocation behavior.
- [x] Keep `launch_conv.rs` as an eight-line manifest and place validation,
      forward, and per-dimensional backward launch families in leaves from 32
      to 268 lines under `kernels/launch_conv/`.
- [x] Verify format, diff, CUDA-feature all-targets check, warning-denied
      Clippy, and the default package Nextest gate.

Evidence: the launch boundary contains no input-dependent `as u32` casts,
unchecked shape products, or input-dependent indexing/panics. CUDA-feature
all-targets check and warning-denied Clippy pass; the default package Nextest
passes 3/3 with zero skipped in 0.072 seconds. CUDA-feature Nextest remains
blocked before execution because the Windows GNU linker cannot find `-lcuda`
at `/usr/local/cuda-11.3/lib64/`. Shared `GpuLayoutInfo` serialization and
the caller-side forward element-count product remain separate residuals.

## CUDA convolution launch failure propagation [patch]
- [x] Replace the CUDA 1D convolution grad-input launch panic with the
      established `false` failure result so the operation boundary can take
      its fallback path.
- [x] Synchronize the launcher's Rustdoc with the error contract and verify
      format, diff, CUDA-feature all-targets check, warning-denied Clippy, and
      the default package Nextest gate.

Evidence: CUDA-feature all-targets check and warning-denied Clippy pass;
default package Nextest passes 3/3 with zero skipped in 0.072 seconds. The
CUDA-feature Nextest remains blocked at link time because the Windows GNU
linker cannot find `-lcuda` at `/usr/local/cuda-11.3/lib64/`. The remaining
unchecked `usize` to CUDA `u32` launch-parameter conversions are recorded in
`docs/gap_audit.md` and are outside this narrow panic-propagation fix.

## CUDA backend operation impl hierarchy [patch]
- [x] Move the eight generic CUDA operation trait impl blocks out of the
      993-line backend operation manifest into
      `backend/ops/impls/` leaves for elementwise, matmul, reduction,
      convolution, pooling, attention, optimizer, and unfold/fold.
- [x] Retain the public CUDA operation helper modules in the manifest and
      keep each moved impl leaf responsible only for its operation-family
      trait forwarding.
- [x] Keep the manifest at 11 lines and every impl leaf below 301 lines;
      verify format, diff, default and CUDA-feature checks, warning-denied
      Clippy, locked metadata, and the default package Nextest gate.

Evidence: locked metadata remains one library, one `cuda_ops` integration
target, and two benchmark targets. Default package Nextest passes 3/3 with
zero skipped in 0.059 seconds. Default and `--features cuda` package checks
and warning-denied Clippy pass. The CUDA-feature Nextest link step is blocked
by the environment's missing `-lcuda` linker library at
`/usr/local/cuda-11.3/lib64/`; no feature-test result is claimed. This is a
module-topology and maintainability change only; no runtime, memory, or
performance delta is claimed.

## CPU backend operation impl hierarchy [patch]
- [x] Move the eight generic CPU operation trait impl blocks out of the
      1,151-line backend implementation file into `backend_ops/cpu_impl/impls/`
      leaves for elementwise, matmul, reduction, convolution, pooling,
      optimizer, attention, and unfold/fold.
- [x] Retain `CpuBackend` and its execution-policy marker impls in the
      manifest; do not duplicate or alter provider dispatch logic.
- [x] Keep the manifest at 56 lines and every operation-family leaf below 325
      lines; verify format, diff, package check, warning-denied Clippy,
      metadata, and exact package Nextest.

Evidence: locked metadata retains one `ops` integration target. Package check
and warning-denied Clippy pass. Exact CPU package Nextest passes 196/196 with
zero skipped in 4.325 seconds across two binaries. This is a module-topology
and maintainability change only; no runtime, memory, or performance delta is
claimed.

## WGPU backend operation impl hierarchy [patch]
- [x] Move the seven non-elementwise WGPU trait impl blocks out of the
      1,357-line operation manifest into `backend/ops/impls/` leaves for
      matmul, reduction, convolution, pooling, attention, optimizer, and
      unfold/fold.
- [x] Retain shared routing helpers and elementwise dispatch in the manifest;
      do not duplicate or alter provider dispatch logic.
- [x] Keep the manifest at 450 lines and every impl leaf below 315 lines;
      verify format, diff, package check, warning-denied Clippy, metadata, and
      exact package Nextest.

Evidence: locked metadata remains unchanged; package check and warning-denied
Clippy pass. Exact WGPU package Nextest passes 89/89 with zero skipped in
90.167 seconds. This is a module-topology and maintainability change only; no
runtime, memory, or performance delta is claimed.

## Coeus-NN attention parity oracle split [patch]
- [x] Move the large attention parity oracle out of the operation test leaf
      into `tensor/nn_parity/attention/expected.rs`.
- [x] Preserve all 11 parity test attributes, numerical values, tolerances,
      transposition logic, and assertion sites.
- [x] Keep the operational attention test at 182 lines and the expected-value
      leaf at 91 lines; verify format, diff, package check, warning-denied
      Clippy, focused parity, and exact package Nextest.

Evidence: the exact package Nextest run passes 268/268 with zero skipped in
2.405 seconds; focused `test_mha_parity` passes 1/1. Locked metadata reports
one `nn_ops` integration target and the `nn_bench` benchmark target. The
11-test source census is unchanged. This is a test-topology and oracle
maintainability change only; no production runtime, memory, or performance
delta is claimed.

## WGPU native unfold/fold and pool1d closure [patch]
- [x] Replace the four no-op WGPU `UnfoldFoldOps` methods with native WGSL
      unfold/fold dispatches and add 1D max/average pooling forward/backward
      kernels.
- [x] Keep device buffers on the WGPU path; upload only bounded layout and
      parameter metadata, with no host fallback or silent CPU degradation.
- [x] Split the pool1d kernel family into manifest, shader, forward, and
      backward leaves, each below 250 lines, and add Sequential differential
      tests for padded/dilated forward and backward behavior.
- [x] Verify format, diff, package check, warning-denied Clippy, and the exact
      package Nextest gate.

Evidence: the WGPU package check and `cargo clippy --locked -p coeus-wgpu
--tests -- -D warnings` pass. Focused pool1d Nextest passes 2/2 with zero
skipped. Exact package Nextest passes 89/89 with zero skipped in 79.311
seconds; the two new pool1d tests and the two unfold/fold tests pass. The
previous no-op paths now perform input-sensitive device computation. No
latency, throughput, or allocation reduction claim is made without a benchmark
baseline.

## Backend-generic host extraction [minor]
- [x] Materialize tensor views through the selected backend's host-copy seam.
- [x] Preserve logical row-major values for offset and strided layouts.
- [x] Verify 58/58 tensor tests and warning-denied Clippy.

## Coeus-dist hierarchical integration harness [patch]
- [x] Replace the 1,262-line `crates/coeus-dist/tests/dist_tests.rs` leaf with one
      `dist_ops` manifest and local/TCP transport subtrees under
      `crates/coeus-dist/tests/distributed/`.
- [x] Preserve all 64 test functions, 64 `#[test]` attributes, panic contracts,
      collective assertions, and extracted Rust function bodies.
- [x] Verify one integration target, warning-denied Clippy, package check,
      format, diff checks, and the exact package Nextest gate.

Evidence: locked metadata reports one `dist_ops` integration target; the
pre/post source census remains 64 unique test functions and all 64 extracted
Rust function bodies compare equal. The largest test-family leaf is
`distributed/tcp/errors/collective.rs` at 464 lines; every leaf is below 500
lines. Exact package Nextest passes 64/64 with zero skipped in 0.444 seconds,
with no slow tests. Production distributed code and test assertions are
unchanged; active documentation now references the `dist_ops` manifest.

## Coeus-NN loss-contract family split [patch]
- [x] Split the live 902-line `nn_ops/losses/nn_loss_tests.rs` leaf into nested
      binary, classification, distance, and distribution leaves under
      `nn_ops/losses/nn_loss/`.
- [x] Preserve all 24 test functions, 24 `#[test]` attributes, analytical
      assertions, tolerances, and extracted Rust function bodies.
- [x] Verify format, package check, warning-denied Clippy, diff checks, and the
      exact package Nextest gate.

Evidence: the pre/post source census remains 24 unique test functions and all
24 extracted Rust function bodies compare equal. The largest leaf is
`distance.rs` at 315 lines; every new leaf is below 500. Exact package Nextest
passes 268/268 with 0 skipped in 2.270 seconds. Package check, warning-denied
Clippy, format, and diff checks pass. Production NN code, fixtures, tolerances,
and sibling loss test files are unchanged. This is a test-topology and
maintainability change only; no production kernel or runtime/memory delta is
claimed.

## Coeus-optim contract-family harness split [patch]
- [x] Split the live 676-line `crates/coeus-optim/tests/optim_tests.rs` leaf into
      optimizer, scheduler, convergence, and gradient-clipping modules under
      `crates/coeus-optim/tests/optim_ops/`.
- [x] Preserve all 20 test functions, 20 `#[test]` attributes, analytical
      comments, tolerances, and extracted Rust function bodies.
- [x] Verify one integration target, format, package check, warning-denied
      Clippy, diff checks, and the exact package Nextest gate.

Evidence: the pre/post source census remains 20 unique test functions and all
20 extracted Rust function bodies compare equal. Locked metadata reports one
`optim_ops` integration target. The largest new leaf is `convergence.rs` at
239 lines; every new leaf is below 250. Exact package Nextest passes 20/20 with
0 skipped in 0.188 seconds. Package check, warning-denied Clippy, format, and
diff checks pass. Production optimizer code and all test oracles are unchanged.
This is a test-topology and maintainability change only; no production
optimizer runtime or memory delta is claimed.

## Coeus-NN extended activation contract-family split [patch]
- [x] Split the live 648-line `nn_ops/activations/act_extended_tests.rs` leaf
      into piecewise, parameterized, module-smoke, and smooth leaves under
      `nn_ops/activations/act_extended/`.
- [x] Preserve all 17 test functions, 17 `#[test]` attributes, analytical
      derivatives, tolerances, and extracted Rust test function bodies.
- [x] Keep the `close`/slice assertion helpers single-sourced and verify
      format, package check, warning-denied Clippy, diff checks, and exact
      package Nextest.

Evidence: the pre/post source census remains 17 unique test functions and all
17 extracted Rust function bodies compare equal. The largest new leaf is
`piecewise.rs` at 354 lines; every new leaf is below 360. Exact package Nextest
passes 268/268 with 0 skipped in 3.155 seconds. Package check, warning-denied
Clippy, format, and diff checks pass. Production NN code, fixtures, formulas,
and tolerances are unchanged. This is a test-topology and maintainability
change only; no production activation runtime or memory delta is claimed.

## Coeus-ops hierarchical integration harness [patch]
- [x] Move the 36 flat `coeus-ops` integration-test files into ten
      operation-family directories under `crates/coeus-ops/tests/ops/`.
- [x] Add one `crates/coeus-ops/tests/ops.rs` integration target with explicit module
      manifests; preserve every leaf test body and assertion.
- [x] Verify the target census reduced from 36 integration binaries to 1,
      the harness exposes 87 integration tests, and package Nextest passes
      196/196 with warning-denied Clippy.

Evidence: `cargo metadata --locked --no-deps` reports one `coeus-ops` test
target (`ops`); `cargo nextest list --locked -p coeus-ops --all-features`
reports 87 `ops` tests; the exact package run passes 196/196. This is a test
topology and build-artifact change only; production operation code is
unchanged.

## Coeus-NN hierarchical integration harness [patch]
- [x] Move the 33 flat `crates/coeus-nn/tests/*.rs` leaf files into ten
      operation-family directories under `crates/coeus-nn/tests/nn_ops/`.
- [x] Use one `crates/coeus-nn/tests/nn_ops.rs` integration target for the established
      `tests/nn/` module tree and the operation-family modules.
- [x] Verify the target census reduced from 2 to 1 and the exact package
      Nextest run passes 268/268 with warning-denied Clippy and package check.

Evidence: locked Cargo metadata reports one `nn_ops` integration target and the
`nn_bench` benchmark target; the exact package run reports 268/268 tests with
0 skipped in 4.463 seconds. Package check, warning-denied Clippy, format, and
diff checks pass. The established NN module tree and all operation-family test
bodies remain unchanged. This is a test topology and build-artifact change
only; it does not claim a production NN speedup, memory reduction, or
whole-workspace debug-tree size delta.

## Coeus-NN tensor parity-family split [patch]
- [x] Split the 1,317-line `tensor/nn_parity.rs` leaf into a manifest plus
      attention, convolution, embedding, linear/normalization, losses, and
      regularization operation-family modules.
- [x] Preserve every live parity test, expected value, tolerance, and
      CPU/autograd assertion; production NN code and fixtures remain unchanged.
- [x] Keep the new operation-family leaves cohesive and verify format, check,
      warning-denied Clippy, diff checks, and the exact package Nextest gate.

Evidence: the pre/post source-name census remains 11 unique parity test
functions; exact package Nextest passes 268/268 with zero skipped in 2.405
seconds. The attention operation leaf is 182 lines and its expected-value
oracle leaf is 91 lines; all six operation-family leaves remain below 250
lines. Package check, warning-denied Clippy, format, and diff checks pass.
This is a test-topology and maintainability change only; it does not claim a
production-kernel speedup, memory reduction, or whole-workspace debug-tree
delta.

## Coeus-autograd hierarchical integration harness [patch]
- [x] Move `grid_sample_3d.rs`, `linear_interpolation.rs`, and
      `selective_scan.rs` into three operation-family directories under
      `crates/coeus-autograd/tests/autograd_ops/`.
- [x] Use one `crates/coeus-autograd/tests/autograd_ops.rs` target for the established
      `tests/autograd/` tree and the standalone operation-family modules.
- [x] Verify the target census reduced from 2 to 1 and the exact package
      Nextest run passes 94/94 with warning-denied Clippy and package check.

Evidence: locked Cargo metadata reports one `autograd_ops` integration target;
the exact package run reports 94/94 tests with 0 skipped in 1.535 seconds.
Package check, warning-denied Clippy, format, and diff checks pass. The existing
autograd module tree and all standalone operation-family test bodies are
unchanged. This is a test topology and build-artifact change only; it does not
claim a production autograd speedup, memory reduction, or whole-workspace
debug-tree size delta.

## Coeus-tensor hierarchical integration harness [patch]
- [x] Move the 13 flat `crates/coeus-tensor/tests/*.rs` leaf files into six
      operation-family directories under `crates/coeus-tensor/tests/tensor_ops/`.
- [x] Add one `crates/coeus-tensor/tests/tensor_ops.rs` integration target with
      explicit backend, checkpoint, constructor, layout, operation, and
      property manifests; preserve every leaf test body and assertion.
- [x] Verify the target census reduced from 13 to 1 and the exact package
      Nextest run passes 58/58 with warning-denied Clippy and package check.

Evidence: locked Cargo metadata reports one `tensor_ops` integration target;
the source census remains 53 annotated integration tests and the exact package
run reports 58/58 tests with 0 skipped, including five library unit tests.
Production tensor code is unchanged. This is a test topology and build-artifact
change only; it does not claim a whole-workspace debug-tree size reduction.

## Coeus-sparse hierarchical integration harness [patch]
- [x] Move the three flat `crates/coeus-sparse/tests/*.rs` leaf files into conversion,
      differential, and invariant directories under `tests/sparse_ops/`.
- [x] Add one `crates/coeus-sparse/tests/sparse_ops.rs` integration target with
      explicit operation-family manifests; preserve every leaf test body and
      value-semantic assertion.
- [x] Verify the target census reduced from 3 to 1 and the exact package
      Nextest run passes 19/19 with warning-denied Clippy and package check.

Evidence: locked Cargo metadata reports one `sparse_ops` integration target;
the exact package run reports 19/19 tests with 0 skipped in 0.713 seconds.
Production sparse code is unchanged. This is a test topology and build-artifact
change only; it does not claim a whole-workspace debug-tree size reduction.

## Coeus-core hierarchical integration harness [patch]
- [x] Move the four flat `crates/coeus-core/tests/*.rs` leaf files into storage,
      dependency-policy, and scalar directories under `tests/core_ops/`.
- [x] Add one `crates/coeus-core/tests/core_ops.rs` integration target with explicit
      operation-family manifests; preserve every leaf assertion and retain the
      seven existing library unit tests in `src`.
- [x] Verify the target census reduced from 4 to 1 and the exact package
      Nextest run passes 21/21 with warning-denied Clippy and package check.

Evidence: locked Cargo metadata reports one `core_ops` integration target; the
exact package run reports 21/21 tests with 0 skipped, comprising 14 integration
cases and seven unchanged library unit tests. Production core code is unchanged.
This is a test topology and build-artifact change only; it does not claim a
whole-workspace debug-tree size reduction.

## Coeus-CUDA feature-aware integration harness [patch]
- [x] Move the three flat CUDA test files into device and fallback directories
      under `crates/coeus-cuda/tests/cuda_ops/` and add one `cuda_ops.rs` harness.
- [x] Preserve the `cuda` and `not(feature = "cuda")` gates, including the
      existing nested `tests/cuda/` module tree used by `cuda_tests.rs`.
- [x] Verify metadata reports one integration target; default Nextest passes
      3/3, and all-features check plus warning-denied Clippy pass.

Evidence: exact default package Nextest passes 3/3 with 0 skipped in 0.053
seconds. All-features execution is not claimable on this host: the linker
fails to find `/usr/local/cuda-11.3/lib64/libcuda`; this is an environment
linker dependency, not a test assertion failure.

## Coeus-CUDA parity-family split [patch]
- [x] Split the live 1,672-line `crates/coeus-cuda/tests/cuda/parity.rs` leaf into
      seven operation-family modules under `tests/cuda/parity/`.
- [x] Preserve all 29 parity test functions, shared CPU/CUDA oracle helpers,
      production CUDA code, fixtures, and tolerance contracts.
- [x] Verify format, default and CUDA-feature checks, warning-denied Clippy,
      diff checks, and the default Nextest gate.

Evidence: the pre/post source-name census remains 29 unique parity test
functions; every new parity leaf is below 500 lines, with `convolution.rs` the
largest at 365 lines. Default package Nextest passes 3/3 with zero skipped;
default and `--features cuda` package Clippy pass with `-D warnings`. The
feature-enabled Nextest target cannot link on this host because
`x86_64-w64-mingw32-gcc` cannot find `-lcuda` while searching
`/usr/local/cuda-11.3/lib64/`. This is an external CUDA linker limitation, so
no live CUDA parity execution is claimed. The slice changes test topology and
maintainability only; production kernels are unchanged.

## Coeus-Python hierarchical integration harness [patch]
- [x] Move the six flat `crates/coeus-python/tests/*.rs` leaf files into activation,
      distributed, NN, operation, optimizer, and autodiff directories under
      `tests/binding_ops/`.
- [x] Add one `crates/coeus-python/tests/binding_ops.rs` target with a single shared
      `tests/common` lock module; preserve every binding assertion and Python
      parity file.
- [x] Verify the target census reduced from 6 to 1 and the exact package
      Nextest run passes 75/75 with warning-denied Clippy and package check.

Evidence: locked Cargo metadata reports one `binding_ops` integration target;
the exact all-features package run reports 75/75 tests with 0 skipped in
6.585 seconds. Production PyO3 and Python parity code is unchanged. This is a
test topology and build-artifact change only; it does not claim Python wheel or
external-interpreter coverage.

## Coeus-Python operation binding-family split [patch]
- [x] Split the live 3,160-line `binding_ops/operations/binding_tests_ops.rs`
      leaf into fourteen operation-family leaves plus nested NN module
      manifests under `binding_ops/operations/`.
- [x] Move the Python interpreter setup into one shared support module and
      preserve all 61 test functions, embedded scripts, assertions, and the
      thin PyO3 boundary.
- [x] Verify exact function-body parity, format, package check, warning-denied
      Clippy, diff checks, and the exact package Nextest gate.

Evidence: the pre/post source census remains 61 unique test functions and all
61 extracted Rust function bodies compare equal. The largest new leaf is
`reductions.rs` at 391 lines; every test-family leaf is below 400 lines. Exact
package Nextest passes 75/75 with zero skipped in 8.079 seconds. Package check,
warning-denied Clippy, format, and diff checks pass. Production PyO3 code,
Python parity scripts, and generated artifacts are unchanged. This is a test
topology and maintainability change only; it does not claim a Python-wheel,
production-kernel, memory, or runtime-performance delta.

## Coeus-WGPU hierarchical integration harness [patch]
- [x] Move the two flat `crates/coeus-wgpu/tests/*.rs` integration targets under one
      `tests/wgpu_ops.rs` harness with `fusion` and `backend/wgpu` ownership.
- [x] Preserve the existing backend operation modules and every WGPU/fused
      assertion; no production kernel, fixture, or tolerance changed.
- [x] Verify locked metadata reports one integration target and the exact
      package Nextest run remains 85/85 with zero skipped.

Evidence: the moved source files are content-identical renames; locked Cargo
metadata reports one `wgpu_ops` integration target instead of two. The exact
package Nextest run passes 85/85 with zero skipped in 84.155 seconds, including
the three unchanged library tests. Package check, warning-denied Clippy, format,
and diff checks pass. The remaining `backend/wgpu/parity.rs` file is 808 lines
and spans multiple operation families; its family split is a separate claimed
follow-up, not part of this target-count relocation.

## Coeus-WGPU parity-family split [patch]
- [x] Split the 808-line parity leaf into a shared oracle manifest plus
      elementwise, reduction, matmul, convolution/pooling, optimizer, and
      strided operation-family modules.
- [x] Preserve every parity test name, generated unary test, tolerance, and
      CPU/GPU differential assertion; production kernels and fixtures remain
      unchanged.
- [x] Keep each parity leaf below 500 lines and rerun format, check, Clippy,
      and package Nextest.

Evidence: the pre/post source-name census remains 47 unique parity identifiers;
locked package Nextest passes 85/85 with zero skipped in 80.113 seconds. The
largest new parity leaf is `elementwise.rs` at 287 lines; all seven leaves are
below 500 lines. Package check and warning-denied Clippy pass after removing two
unused imports exposed by the split.

## Coeus-Leto hierarchical integration harness [patch]
- [x] Move `contract.rs` and `sparse_dispatch.rs` under one
      `crates/coeus-leto/tests/leto_ops.rs` harness with contract and sparse-dispatch
      operation-family modules.
- [x] Preserve all 28 listed integration tests and their cross-provider
      contract assertions; production APIs, fixtures, and tolerances remain
      unchanged.
- [x] Verify locked metadata reports one integration target and exact package
      Nextest preserves the full test count with warning-denied Clippy.

Evidence: locked metadata reports one `leto_ops` integration target instead of
two. The exact package Nextest run passes 28/28 with zero skipped in 1.064
seconds. Package check, warning-denied Clippy, format, and diff checks pass.
The live census corrected the prior 26-test tracking claim: `contract` contains
26 tests and `sparse_dispatch` contains 2. This is a test topology and
maintainability change only; it does not claim a production-kernel speedup,
memory reduction, or whole-workspace debug-tree delta.

## Coeus-Leto cross-provider contract-family split [patch]
- [x] Split the live 505-line `leto_ops/contract.rs` leaf into arithmetic,
      reductions, matmul, layout, and accumulation modules under
      `crates/coeus-leto/tests/leto_ops/contract/`.
- [x] Preserve all 26 contract tests, 26 `#[test]` attributes, shared layout
      oracle behavior, and extracted Rust test function bodies.
- [x] Keep one `leto_ops` integration target and verify package check, format,
      diff checks, warning-denied Clippy, and exact package Nextest.

Evidence: the pre/post source census remains 26 unique contract tests and all
26 extracted Rust test function bodies compare equal. The largest new leaf is
`layout.rs` at 197 lines; every new leaf is below 200 lines. Exact package
Nextest passes 28/28 with 0 skipped in 0.325 seconds. Locked metadata reports
one `leto_ops` integration target. Production Leto dispatch code, APIs,
fixtures, formulas, and tolerances are unchanged. This is a test-topology and
maintainability change only; no production runtime, memory, or zero-copy delta
is claimed.

This document tracks the high-level roadmap and feature validation checklist for the Coeus tensor library. For detailed task lists, sprint archives, and progress tracking, see:
- [docs/checklist.md](file:///d:/coeus/docs/checklist.md)
- [docs/backlog.md](file:///d:/coeus/docs/backlog.md)

---

## Named Optimizer Ownership [major]
- [x] Move the canonical `Parameter` carrier below NN and optimizer consumers.
- [x] Make every optimizer retain stable names through steps and gradient clipping.
- [x] Validate complete path inventories before loading updated values into modules.
- [x] Require explicit `(name, tensor)` pairs at the PyO3 boundary.
- [x] Verify optimizer nextest 20/20, cross-boundary nextest 21/21,
      affected NN parity 144/144, Clippy, Rustdoc, and doctests.

## RITK Stable Named Parameters [minor]
- [x] Add canonical named parameter collection to the `Module` seam.
- [x] Preserve optimizer ordering while assigning semantic leaf names.
- [x] Prefix dynamic/static containers, recurrent modules, and transformer trees hierarchically.
- [x] Verify exact decoder paths, full-transformer uniqueness, shared gradient identity, nextest 410/410, and warning-denied Clippy.

## RITK Dimension-Complete Interpolation [minor]
- [x] Replace dimension-specific entry points with one const-dimension linear interpolation family.
- [x] Encode replicated-border behavior as a sealed zero-sized policy.
- [x] Provide shared 2-D/3-D forward and reverse-mode kernels without hot-loop allocation.
- [x] Verify analytical values/gradients, every coordinate by central difference,
      malformed contracts, Sequential/Moirai agreement, and affected nextest 282/282.

## RITK Bounded Archived State Support [minor]
- [x] Replace the eager bespoke `StateDict` format with validated rkyv archives.
- [x] Expose zero-copy borrowed tensor names, shapes, and payload bytes before materialization.
- [x] Enforce archive/tensor/name/rank/payload limits, scalar identity, byte order, duplicate-name rejection, and deterministic ordering.
- [x] Verify package nextest 56/56, warning-denied Clippy, Rustdoc, and doctests.

## RITK VMamba Depthwise Convolution Support [minor]
- [x] Add a canonical Coeus depthwise 3-D convolution module with one learned kernel per channel.
- [x] Preserve input, kernel, and bias reverse-mode paths without a consumer-owned grouped-convolution adapter.
- [x] Verify exact two-channel values and analytical input gradients under nextest.

## RITK Attention Matmul Support [patch]
- [x] Preserve rank-N logical batch axes across batched matmul.
- [x] Dispatch backward accumulation through an explicit rank-3 kernel layout.
- [x] Verify exact rank-4 values and both operand gradients, affected nextest
      689/689, and warning-denied Clippy.

## RITK TransMorph Provider Support [minor]
- [x] Generalized `coeus_nn::Linear` to project the final feature axis of
      rank-2 and higher-rank inputs through one autograd-preserving path.
- [x] Verified exact rank-3/rank-5 forward values and rank-3
      input/weight/bias gradients, all 409 `coeus-nn` tests,
      warning-denied Clippy, and rustdoc.

## Default Parallel Memory Features [patch]
- [x] Restored Moirai default features for workspace consumers so parallel execution, Mnemosyne-backed memory surfaces, and Mellinoe branding are active by default.
- [x] Restored Leto and Leto Ops defaults so Coeus consumes Mnemosyne-backed Leto storage and default parallel ops without local feature suppression.
- [x] Verification: `cargo check --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo nextest run --workspace`, and `cargo test --doc --workspace` pass in the default no-CUDA-provider configuration. Real cutile CUDA tests remain under `coeus-cuda --features cuda`.

## 🟩 Phase 1: CPU Workspace Stabilization (100% Complete)
Established a clean, warning-free compiler baseline and resolved lifetime and borrow checker conflicts across all workspace crates.

- [x] **Zero-Copy Layout Traversal**: Refactored `coeus-ops` kernels (unary, binary, matmul, sum/mean reductions, SpMV, SpMM) to perform direct strided index math without calling `to_contiguous()`.
- [x] **Thread-Safe Pointers**: Created `SendPtr` and `SendPtrMut` raw pointer wrappers in `crates/coeus-ops/src/ptr.rs` to allow safe, thread-safe capture of raw pointers in `Moirai` parallel closures.
- [x] **Apollo FFT Integration**: Decoupled FFT operations from coeus crates, supporting them directly inside the `apollo-fft` crate in the `apollo` workspace.
- [x] **Autograd & Optimizers**: Resolved lifetime and borrow checker errors (SGD, Adam, RMSProp step loops, and LayerNorm/BatchNorm backward closures).
- [x] **Numerical Parity Validation**: Verified mathematical outputs (relu, matmul, reductions, FFT, sparse operations) against `ndarray` and PyTorch references inside `crates/coeus-tensor/tests/parity_tests.rs`.
- [x] **Autodiff PyTorch Comparison**: Implemented integration benchmarks in `crates/coeus-python/tests/autodiff_comparison.rs` verifying 100% mathematical gradient parity (X, weight, and bias gradients) and measuring step time comparison.

---

## 🟩 Phase 2: Associated-Type Backend Overhaul (GPU Abstractions) (100% Complete)
Evolve the backend trait to support heterogeneous device compilation (CPU and GPU backends) with zero abstraction overhead.

- [x] **ComputeBackend Trait**: Define a new generalized backend trait containing associated types:
  - `type DeviceBuffer<T>`: Handle for device-allocated memory buffers.
  - `type KernelDescriptor`: Configuration and dispatch params for compute pipelines.
  - `type DispatchFuture<T>`: Non-blocking future wrapping GPU execution handles.
- [x] **Unified Memory Management**: Update the `Storage` and `StorageMut` traits to support non-CPU-addressable memory buffers.
- [x] **Explicit Memory Transfers**: Add explicit copy APIs (`Tensor::to_backend::<NewB>(&self, backend: &NewB)`) to handle CPU-to-GPU, GPU-to-CPU, and GPU-to-GPU memory boundaries.

---

## 🟩 Phase 3: WebGPU Backend Crate (coeus-wgpu) (100% Complete)
Implement a cross-platform GPU backend powered by `wgpu`.

- [x] **Crate Setup**: Create `coeus-wgpu` and add it to the workspace members.
- [x] **Device Memory Wrapper**: Implement `WgpuStorage<T>` managing raw GPU buffer allocations.
- [x] **WGSL Compute Kernels**: Write WebGPU Shading Language compute shaders for element-wise ops, matrix multiplication, sum reductions, and Conv1D/Conv2D forward and backward passes.
- [x] **Compilation Caching**: Implement automated compute pipeline caching using `wgpu::ComputePipeline`.

---

## 🟪 Phase 4: CUDA Backend Crate (coeus-cuda) (100% Complete)
Implemented a native NVIDIA GPU backend dynamically loading the CUDA driver.

- [x] **Crate Setup**: Created `coeus-cuda` and added it to the workspace members.
- [x] **Context Management**: Dynamically binds to the native CUDA Driver API (`cuInit`, `cuCtxCreate`, etc.) via `libloading` at runtime.
- [x] **Staging Memory Transfers**: Implemented FFI copies between host CPU and device GPU memory (`cuMemcpyHtoD`, `cuMemcpyDtoH`).
- [x] **Thread-Safe Context**: Implemented a thread-safe static wrapper for the loaded context pointer.
- [x] **Hephaestus CUDA Substrate**: Allocations and shared primitive
  contiguous elementwise kernels route through `hephaestus-cuda`, matching the
  WGPU provider boundary; Coeus-local CUDA kernels remain for aliasing,
  strided/dynamic layout coverage, and NN-specific convolution, pooling,
  optimizer, and activation formulas.

