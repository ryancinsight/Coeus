# Coeus Gap Audit

## COEUS-ATTENTION-PROVIDER-001: Consumer-owned attention dispatch

**Location**: `coeus-ops`, `coeus-wgpu`, `coeus-cuda`, `coeus-rocm`, and
`coeus-metal` attention implementations.
**Gap**: Coeus duplicated scaled dot-product attention mathematics for CPU,
WGPU, and CUDA, retained a CUDA host fallback, and exposed no shared ROCm/Metal
provider dispatch or typed operation failure.
**Resolution**: call Leto directly for CPU storage, bind all accelerator storage
through one monomorphized Coeus-Hephaestus bridge, make the public operation
fallible, migrate every caller, and delete superseded implementations.
**Residual**: exact-head hosted device lanes remain the closure gate. No
performance or memory delta is claimed without matched measurements.
**Status**: local implementation, provider differential verification, and
independent architecture/correctness review complete under ADR-0047. Review
findings consolidated all accelerator request assembly in `AttentionBackend`,
made Coeus's scalar marker provider-neutral, and closed Python mask validation.

## ATLAS-COEUS-HEPHAESTUS-006: Activation-tail provider parity

**Location**: `crates/coeus-wgpu/src/backend/ops/mod.rs`,
`crates/coeus-cuda/src/backend/ops/math.rs`, and the ROCm/Metal elementwise
provider leaves and parity suites.
**Gap**: Hephaestus PR #123 provides `Mish`, `MishGrad`, `Elu`, and `EluGrad`
markers, but Coeus consumer dispatch did not route the four operations through
the provider-owned APIs on every applicable backend path.
**Resolution**: add direct contiguous/strided WGPU and CUDA marker dispatches,
extend the existing ROCm and Metal f32 activation matches, and compare all
forward/gradient results with the Leto CPU oracle.
**Residual**: parameterized activations and reduced/vector precision remain
separate parity scopes. Provider rank
and aliasing contracts are not expanded; out-of-contract activation-tail
requests return typed errors without local-kernel or CPU fallback.
**Evidence**: focused WGPU, CUDA, ROCm, and Metal nextest plus doctest and
rustdoc gates; exact-head backend run `30623370603` passes WGPU/CPU/Leto
(`91132945368`), Metal (`91132945402`), ROCm (`91132945438`), and CUDA
(`91132945439`). The manual-only hardware ROCm job (`91132945943`) is skipped
by design.
**Status**: source integration and differential coverage complete. Locked
metadata, focused non-CUDA nextest (307/307), warning-denied Clippy, workspace
doctests (153 passed, 2 ignored), warning-denied rustdoc, and the MSVC CUDA
feature compile check pass. Focused CUDA nextest passes 6/6 with real
contiguous and transposed device execution; the focused CPU/WGPU/ROCm/Metal
activation-tail lane passes 10/10. No fallback path was added. Exact-head
backend CI passes all four software provider lanes.
## ATLAS-COEUS-NN-SAFETY-019: Infallible module execution

**Location**: `crates/coeus-nn/src/module/`, 44 module implementation files,
seven normalization files, and direct Rust/Python consumers.
**Gap**: `Module::forward` cannot represent backend or module-state failure.
An all-target check succeeds but emits exactly 54 ignored-result warnings
across BatchNorm, GroupNorm, InstanceNorm, LayerNorm, and RMSNorm. BatchNorm
can also partially update running mean before a later variance operation
fails.
**Resolution target**: ADR-0045 changes the canonical trait to return
`ModuleError<B::Error>`, migrates all 85 implementations and every caller
atomically, replaces input-dependent normalization panics with typed errors,
and stages both BatchNorm running-stat tensors before committing either.
**Evidence target**: warning-denied all-target Clippy, failure-injection and
state-integrity regressions, analytical normalization and gradient parity,
Rust/Python consumer tests, doctests, SemVer classification, and exact-head
provider CI.
**Status**: in progress. The call graph is audited and the module bounded
context is split into manifest, trait, and typed-error leaves. No runtime,
allocation, or binary-size improvement is claimed without matched
measurements.

## ATLAS-COEUS-WGPU-008: Duplicate ordinary reduction provider

**Location**: `crates/coeus-wgpu/src/backend/ops/impls/reduction.rs` and the
superseded `backend/ops/reduction.rs` plus ordinary shader path in
`kernels/reduce.rs`.
**Gap**: scans called Hephaestus, while sum, product, mean, minimum, and
maximum generated and submitted a second Coeus-owned WGSL implementation.
**Resolution**: ADR-0044 routes rank-one and rank-two ordinary reductions
directly through Hephaestus and deletes the consumer-owned dispatcher, shader,
metadata staging, and validation tests.
**Evidence**: package all-target check passes. Focused Nextest run
`a3a70d2f-37ff-4d75-9754-a6b029850c16` passes all 11 reduction contracts in
9.143 seconds, including five ordinary operation families, rank-one sum and
scan, and exact typed rank-three rejection. Warning-denied package Clippy and
all three WGPU doctests pass. Implementation-head run `30407395047` passed
Metal `90435767524`, ROCm `90435767607`, WGPU `90435767627`, and CUDA
`90435767639`; required-device ROCm `90435768426` was skipped because no
hosted AMD runner was dispatched.
**Residual**: fused-expression reduction retains its distinct Coeus kernel
pending a provider expression contract. Terminal run `30408820242` passed
Metal `90440235821`, CUDA `90440235825`, WGPU `90440235878`, and ROCm
`90440236008`; required-device ROCm `90440236263` was skipped because no
hosted AMD runner was dispatched. PR #246 merged as `7a9811f4`. No runtime,
memory, or binary-size improvement is claimed without matched measurements.

## ATLAS-COEUS-CUDA-007: Backend identity changed execution identity

**Location**: `crates/coeus-cuda/src/backend/ops/math/`,
`src/fallback/ops/math.rs`, and the disabled-provider backend.
**Gap**: CUDA mathematical dispatch misses downloaded device storage, executed
with `SequentialBackend`, and uploaded results. Disabled-provider builds also
implemented CPU mathematics while reporting a CUDA backend.
**Resolution**: ADR-0043 deletes those mathematical fallback paths, routes
rank-two reductions through Hephaestus, and leaves disabled-provider builds
without mathematical backend traits.
**Residual**: CUDA optimizer capability paths still copy through host memory.
WGPU/CUDA aliased elementwise operations still
require provider-owned in-place Hephaestus contracts.
**Evidence**: no-default all-target compilation and all three disabled-provider
identity/error tests pass. Exact-head run `30405547693` passed ROCm
(`90430046827`), Metal/Leto (`90430046863`), WGPU/CPU (`90430046874`), and
CUDA (`90430046879`); required-device ROCm (`90430047556`) was intentionally
skipped. PR #245 merged as `77834e37`.

## ATLAS-AUTOGRAD-SAFETY-018: Infallible backward traversal

**Location**: `crates/coeus-autograd/src/node.rs`, `var.rs`, and operation
node implementations.
**Gap**: 143 fallible backend mutation or direct-dispatch calls discarded
their `Result` because the autograd graph contract returned `()`. A backend
failure could leave partial gradients while the caller observed success.
**Resolution**: ADR-0042 makes the graph traversal and every node return the
backend's typed error, with no compatibility or fallback path.
**Evidence**: warning-denied `coeus-autograd` all-target Clippy passes. The
failure-injection regression observes the exact `BackendError::Storage`.
Nextest passes 102 autograd/FFT and 268 NN tests. All 24 executable doctests
pass; two pre-existing NN doctests remain ignored. SemVer checks against
`origin/main` report the `BackwardNode` and `BinaryAutogradOp` return changes
as requiring a major release. Exact-head run `30397554467` attempt 2 passes
WGPU (`90407664433`), ROCm
(`90407664470`), CUDA (`90407664479`), and Metal (`90407664482`);
required-device ROCm (`90407665417`) is intentionally skipped. Attempt 1
failed before Coeus in Leto's missing `T: UnitScalar` stencil contract; Leto
PR #77 repaired that provider-owned bound before the successful rerun.
**Residual**: compilation exposes 54 ignored fallible normalization mutations
in `coeus-nn` and one ignored distributed mutation outside this increment;
they remain separate typed-propagation work. No runtime, allocation, or
binary-size improvement is claimed.
**Status**: complete at `81eeec09`; merge delivery pending.

## ATLAS-CUDA-SAFETY-017: Pooling physical-index contract

**Location**: `crates/coeus-cuda/src/kernels/validation.rs` and
`crates/coeus-cuda/src/kernels/pool/`.
**Gap**: pooling checked layout fields and parameters independently against
the CUDA unsigned ABI, but did not prove the derived physical offset or the
signed window-coordinate expressions representable. Storage capacity and
writable aliasing were also unchecked at the pooling boundary.
**Resolution**: ADR-0041 centralizes the physical layout/storage proof shared
by fusion, unfold/fold, and pooling. Every pooling dimension now validates
allocation bounds, writable non-aliasing, and complete forward/backward signed
coordinate extrema before compilation.
**Evidence**: pure boundary regressions cover exact and undersized strided
storage, physical-offset overflow, writable zero-stride aliasing, and signed
coordinate overflow. Feature-enabled all-target check and warning-denied
package Clippy pass. Local native test linking remains blocked by MinGW
`cannot find -lcuda`. Exact-head run `30391721824` passes CUDA
(`90384681039`), WGPU (`90384681127`), Metal (`90384681124`), and ROCm
(`90384681137`); required-device ROCm (`90384681768`) is intentionally
skipped.
**Residual**: the broader warning-denied graph exposes 143 pre-existing
ignored `Result` errors in `coeus-autograd`. The live Atlas overlay later
resolved Leto with missing `T: UnitScalar` bounds in
`application/stencil.rs`, blocking repeat Coeus compilation before the
touched crate. No runtime, bandwidth, or resident-memory delta is claimed.
**Status**: complete at `8fe4da78`; merge delivery pending.

## ATLAS-COEUS-SAFETY-003: Uninitialized COW replacement allocation

**Location**: `crates/coeus-hephaestus/src/storage.rs`,
`crates/coeus-wgpu/src/storage.rs`, and `crates/coeus-cuda/src/storage.rs`.
**Gap**: device-local COW allocates a replacement and immediately overwrites
every element through `ComputeDevice::copy_buffer`, but the previous consumer
path requested zero-initialized storage. CUDA and ROCm therefore paid a full
initialization pass before the full device-to-device copy.
**Resolution**: keep public storage construction on the zeroed allocation
contract and route only COW replacements through the explicit Hephaestus
overwrite-before-read allocation seam. The generic Hephaestus test device
implements the seam without changing the value-semantic copy regression.
**Residual**: the Hephaestus provider seam prerequisite is satisfied by PR #136
merged at `da785b53`. Hosted exact-head Coeus run `30345002409` passed CUDA job
`90229046185`, WGPU job `90229046271`, ROCm job `90229046258`, and Metal job
`90229046242`; required-device ROCm job `90229047328` was skipped because no
hosted AMD runner was dispatched. No runtime bandwidth, latency, or
resident-memory delta is claimed without a controlled benchmark. The
infallible `StorageMut::make_unique` failure boundary remains the separate
`ATLAS-COEUS-SAFETY-001` item.
**Local evidence**: Coeus WGPU and generic Hephaestus all-target checks pass
against the provider branch; the CUDA feature all-target check also passes.
The focused generic Hephaestus Nextest storage contract passes. Temporary
provider path overlays were restored after verification.
**Status**: complete for the consumer allocation path and hosted provider
matrix; physical-device execution and runtime performance measurement remain
explicit residuals.

## ATLAS-COEUS-DISPATCH-SAFETY-020: Provider-owned convolution

**Location**: `crates/coeus-ops/src/backend_ops/`,
`crates/coeus-hephaestus/src/convolution/`, accelerator `ConvOps`
implementations, and convolution consumers in autograd, NN, Python, tests, and
benchmarks.
**Gap**: Coeus owned CPU and accelerator convolution mathematics, exposed
infallible dispatch, retained a separate 3-D transposed capability, downloaded
unsupported CUDA requests for CPU execution, and implemented transposed
backward loops in autograd.
**Resolution**: ADR-0046 consolidates regular/transposed forward and backward
into four fallible const-generic `ConvOps` methods. CPU dispatch borrows
storage into Leto; CUDA/WGPU/ROCm/Metal dispatch device buffers into
Hephaestus through one generic static dispatch implementation; vendor modules
bind only their device, buffer, and error types. Leto regular and transposed
operations share one borrowed-view construction path. Rank-specific methods
are default adapters over that SSOT. Consumer-owned CUDA/WGPU kernels, CUDA
host fallbacks, the generic transposed host default, `ConvTranspose3dOps`, and
autograd host backward loops are deleted.
**Evidence**: warning-denied all-target Clippy passes for the consolidated
Leto, Hephaestus, WGPU, CUDA, and operation-contract scope. CPU/autograd/NN
Nextest passes 592/592, including regular/transposed rank-one through
rank-three differentials and exact transposed gradients. Final-review
Leto/Hephaestus/autograd/WGPU Nextest passes 214/214, including
regular/transposed parity, exact gradient accumulation, and COW storage. All
46 executable affected-package doctests pass; two pre-existing NN doctests
remain ignored.
`cargo-semver-checks` confirms the changed failure contract and removed
capability seam require a major release. Residue scans find no convolution
`SequentialBackend`, host transfer, fallback, or consumer kernel path.
Exact-head provider run `30545333101` passed WGPU job `90880014492`, CUDA job
`90880014608`, ROCm job `90880014606`, and Metal job `90880014508`;
required-device ROCm job `90880015294` was skipped because no AMD hardware
runner was dispatched.
**Residual**: no implementation, migration, build, test, or merge residual
remains in the authorized convolution scope. No runtime, memory, or
binary-size delta is claimed without controlled measurements.
**Status**: complete. PR #250 merged as `0dfab53e`.

## ATLAS-COEUS-DISPATCH-001: Unsupported reduction selection fallback

**Location**: `crates/coeus-ops/src/backend_ops/traits/reduction.rs`,
`crates/coeus-ops/src/backend_ops/defaults/reductions.rs`, and the public
selection callers under `crates/coeus-ops/src/reduction`.
**Gap**: generic `argmax`, `argmin`, and `topk` defaults copied device buffers
to host memory and executed the Leto CPU path when an accelerator did not
provide a native operation. This made unsupported ROCm, Metal, WGPU, CUDA,
and generic Hephaestus calls look available while violating provider ownership
and zero-copy dispatch.
**Resolution**: the defaults and public/autograd selection entry points now
require `CpuBackend`. CPU backends retain direct Leto dispatch; accelerator
reduction and scan methods remain provider-owned. Native selection kernels are
not added downstream and remain a separate Hephaestus/provider item.
**Evidence**: pinned formatting, metadata, and staged-diff checks passed
locally. The local package compile, Nextest, and doctest gates remain blocked
before Coeus compilation because the peer-owned Leto path lacks
`EqOp`/`GeOp`/`GtOp`/`LeOp`/`LtOp`/`NeOp`. Exact-head provider matrix
`30278852605` passed WGPU `90019911397`, CUDA `90019911331`, ROCm `90019911264`,
and Metal `90019911476`; required-device ROCm `90019912082` was skipped because
no hosted AMD runner was dispatched.
**Status**: complete for the CPU/provider capability boundary; native
accelerator arg-reduction and top-k kernels remain a separate provider item.

## ATLAS-COEUS-HEPHAESTUS-005: Unary math provider parity

**Location**: `crates/coeus-rocm/src/backend/elementwise.rs`,
`crates/coeus-metal/src/backend/elementwise.rs`, and their elementwise
contracts.
**Gap**: Coeus/Leto defined 19 unparameterized f32 unary math operations that
were available to the WGPU/CUDA shader paths but were rejected by the native
ROCm and Metal provider matches.
**Resolution**: route each operation through the shared Hephaestus marker and
strided kernel, with valid-domain Leto differential coverage. Keep integer
providers on their typed arithmetic-only rejection path.
**Residual**: `erf`, `erfc`, parameterized activations, and f64/vector contracts
remain separate capability slices; `lgamma` is closed by the dedicated item
below.
**Evidence target**: exact-head WGPU, CUDA, ROCm, and Metal CI; hardware lanes
are reported independently from adapterless provider compilation.
**Status**: complete for the 19-operation f32 scope. Hephaestus PR #112 merged
as `e6ba1c14`. Coeus exact-head run `30273987046` passed WGPU `90003264732`,
CUDA `90003264777`, ROCm `90003265014`, and Metal `90003264805`; required-device
ROCm `90003265412` was skipped because no registered AMD runner was available.

## ATLAS-COEUS-HEPHAESTUS-ACTIVATION-TAIL-PARITY-001: Mish and ELU

**Location**: the Coeus accelerator activation dispatch tables and the four
backend contract suites.
**Gap**: Hephaestus already exports native f32 `Mish`, `MishGrad`, `Elu`, and
`EluGrad` markers for every accelerator provider. Coeus ROCm and Metal
originally rejected all four operations, while CUDA and WGPU later computed
ELU through consumer-owned expressions instead of the Hephaestus markers.
**Resolution**: dispatch ROCm and Metal through the existing Hephaestus
strided marker seam; dispatch CUDA and WGPU ELU forward and gradient through
Hephaestus contiguous and strided markers; delete the superseded consumer
expressions. Extend the backend suites with Leto CPU differential checks for
contiguous and transposed-strided forward and gradient operations.
**Residual**: parameterized activations, f64/reduced/vector contracts, and
physical-device execution remain separate evidence scopes.
**Evidence target**: exact-head WGPU, CUDA, ROCm, and Metal provider/consumer
CI; required-device ROCm is reported independently from adapterless provider
compilation.
**Status**: complete for the unparameterized f32 scope. CUDA and WGPU reject
ELU fallthrough rather than entering local-kernel or CPU paths. Exact-head run
`30387168252` passed CUDA job `90369248008`, WGPU job `90369248023`, ROCm job
`90369247910`, and Metal job `90369248013`; required-device ROCm job
`90369248641` was skipped because no hosted AMD runner was dispatched. The
external `recurseml/analysis` status returned its recurring analyzer error and
is not repository-owned verification. ADR 0038 owns the provider contract.

## ATLAS-COEUS-HEPHAESTUS-CUDA-ACTIVATION-PARITY-001: GELU-tanh and Softplus

**Location**: `crates/coeus-cuda/src/backend/ops/math.rs`, the CUDA parity
suite, and the WGPU/CUDA backend-parity selectors.
**Gap**: Coeus ROCm and Metal already routed `GeluTanh`, `GeluTanhGrad`,
`Softplus`, and `SoftplusGrad` through Hephaestus strided markers, while CUDA
had no shared-provider arms for these operations. Contiguous and runtime-shaped
strided CUDA tensors could therefore reach the legacy capability boundary.
**Resolution**: route the four operations through Hephaestus's existing CUDA
marker kernels for both allocation-returning contiguous dispatch and caller-
owned dynamic-rank strided dispatch. Add CUDA/Leto forward and gradient
differential cases and a WGPU GELU-tanh contract selection.
**Residual**: parameterized activations, f64/reduced/vector contracts, and
physical-device execution remain separate evidence scopes. No runtime
performance or resident-memory delta is claimed without a controlled benchmark.
**Evidence target**: exact-head WGPU, CUDA, ROCm, and Metal provider/consumer
CI; required-device ROCm is reported independently from adapterless provider
compilation.
**Status**: docs head `8a38f392` passed run `30359324025`: CUDA
`90274888940`, WGPU `90274889041`, ROCm `90274889047`, and Metal `90274888991`.
Required-device ROCm `90274889835` was skipped because no hosted AMD runner was
dispatched. The external `recurseml/analysis` status returned its recurring
analyzer error and is not repository-owned verification.

## ATLAS-COEUS-CUDA-ELEMENTWISE-STORAGE-BOUNDARY-001

**Location**: `crates/coeus-cuda/src/backend/ops/math/elementwise.rs` and its
`elementwise/validation.rs` leaf.
**Gap**: the safe elementwise backend accepted layouts whose maximum physical
offset exceeded the associated CUDA allocation. The contiguous branch also
accepted nonzero offsets while its raw kernel always indexed from element zero.
**Resolution**: validate every input and writable output layout against the
actual allocation at the backend and public raw-launch boundaries; reject
writable zero-stride layouts and remapped aliases; route offset-bearing
contiguous views through the offset-aware strided kernel.
**Evidence target**: allocation-free validation unit tests for each operand,
offset routing, writable aliasing, and exact-layout in-place operation; package
check, warning-denied Clippy, Nextest, and exact-head CUDA provider CI. Empty
layouts remain valid and complete without a device launch.
**Residual**: CUDA convolution, optimizer, matmul, module-loading, and
device-acquisition boundaries remain separate accepted audit findings. No
runtime performance or resident-memory delta is claimed without controlled
measurements.
**Status**: implementation complete under
`ATLAS-COEUS-DISPATCH-SAFETY-020`. CUDA feature test targets compile,
warning-denied all-target Clippy passes, and disabled-provider Nextest passes
3/3. Exact implementation-head run `30426667552` passed CUDA `90494509271`,
Metal `90494509298`, ROCm `90494509264`, and WGPU `90494509247`;
required-device ROCm `90494509665` was skipped because no AMD runner was
registered. Local feature-enabled execution is blocked by the GNU CUDA import
library and shared-cache MSVC host-artifact collision.

## ATLAS-COEUS-CUDA-LINALG-CONV-STORAGE-BOUNDARY-002

**Location**: `crates/coeus-cuda/src/kernels/launch_matmul.rs` and
`kernels/launch_conv/`.
**Gap**: raw CUDA matmul and convolution launchers checked layout ABI
representability but not whether each logical layout fit its physical device
allocation. Convolution also trusted caller-provided output counts and gradient
buffer capacities. Its embedded PTX computes physical addresses with signed
32-bit arithmetic while the generic CUDA descriptor admits unsigned values.
**Resolution**: reuse the shared CUDA storage-bound validator for matmul and
centralize rank-specialized convolution forward/backward contracts in
`launch_conv/validation.rs`; reject undersized source, output, gradient, and
bias allocations plus incompatible batch/channel/spatial shapes, checked
stride/padding/dilation extent mismatches, count mismatches, and writable
zero-stride layouts before pointer acquisition. Convolution composes that
generic proof with PTX-specific signed-field, convolution-parameter,
derived-coordinate, and maximum-physical-index bounds; NVRTC kernels retain the
wider unsigned descriptor contract.
**Evidence**: Rust 1.95 warning-denied feature-enabled Clippy; focused signed,
shape, and modular-coordinate Nextest 5/5; disabled-provider Nextest 3/3; and
exact source-head run `30488454769`, which passed CUDA `90700098613`, Metal
`90700098624`, ROCm `90700098669`, and WGPU `90700098570`. Required-device
ROCm `90700098948` skipped because no AMD runner was registered.
**Residual**: none in this storage-boundary item. ADR-0046 and
`ATLAS-COEUS-DISPATCH-SAFETY-020` close the former convolution ownership and
fallibility residual. No runtime performance or resident-memory delta is
claimed without controlled measurements.
**Status**: resolved for the raw matmul and historical convolution storage and
signed-PTX boundaries.

## ATLAS-COEUS-HEPHAESTUS-CUDA-GELU-PARITY-001: exact GELU forward and gradient

**Location**: `crates/coeus-cuda/src/backend/ops/math/elementwise.rs`, the CUDA
unary parity tests, and the backend-parity CUDA selector.
**Gap**: Hephaestus already exposes exact-erf `GeluOp` and `GeluGradOp`, and
ROCm/Metal route both operations through the shared strided provider seam, but
CUDA's generic math dispatch left them to the legacy kernel table. The CUDA
selector also omitted the existing forward and gradient parity tests.
**Resolution**: route contiguous and runtime-shaped strided CUDA dispatch
through the Hephaestus exact-erf markers and select
`test_cuda_parity_gelu`/`test_cuda_parity_gelu_grad` in backend CI.
**Residual**: parameterized activations, reduced/vector scalar contracts, and
physical-device execution remain separate evidence scopes. No runtime
performance or resident-memory delta is claimed without a controlled
benchmark.
**Status**: complete at `f861cea6`. Exact-head run `30379272710` passed CUDA
`90342897802`, WGPU `90342897872`, ROCm `90342897673`, and Metal
`90342897752`. Required-device ROCm `90342898718` was skipped because no hosted
AMD runner was dispatched. Local locked CUDA package checking remains blocked
before compilation by the peer-owned provider-overlay lockfile. No
physical-device, runtime-performance, or resident-memory claim is made.

## ATLAS-COEUS-HEPHAESTUS-LGAMMA-PARITY-001: f32 forward log-gamma

**Location**: the WGPU unary expression, CUDA/ROCm/Metal provider elementwise
dispatch, and their backend contract suites.
**Resolution**: route `UnaryOp::Lgamma` through the Hephaestus provider marker
on all four backends. WGPU and Metal use the shared Lanczos/reflection
expression; CUDA and ROCm use native device functions. Backend tests compare
positive, reflected non-integer, and non-positive integer pole inputs with the
Leto CPU oracle, requiring positive infinity at poles.
**Evidence**: Hephaestus PR #118 passed WGPU `90086428952`, CUDA `90086430178`,
ROCm `90086430143`, and Metal `90086428160`. Coeus PR #231 merged at
`971fab9614b97bd708a716d01684da58fd1331ba`; its consumer jobs passed WGPU
`90088836682`, CUDA `90088836688`, ROCm `90088836731`, and Metal `90088836675`.
Required-device ROCm `90088837591` was skipped because no hosted AMD runner was
dispatched; physical-device execution is not claimed.
**Residual**: digamma gradients, f64/reduced/vector contracts, and complete
non-elementwise Leto parity remain outside this item.
**Status**: complete for the f32 forward provider/consumer boundary.

## ATLAS-COEUS-SAFETY-001: Hephaestus provider failure boundary

**Location**: `crates/coeus-hephaestus/src/reduction.rs` and the ROCm/Metal
provider leaves under `crates/coeus-{rocm,metal}/src/backend/provider.rs`.
**Gap**: `HephaestusProvider::device()` returns a shared device reference, so
ROCm and Metal acquire devices through `OnceLock::get_or_init(...expect(...))`.
The generic Hephaestus `ComputeBackend` also uses `expect` for fill and host/
device transfers. Device absence and provider transfer failures therefore
panic inside a library boundary instead of reaching callers as typed errors.
This is independent of comparison-kernel correctness and is not repaired by
the backend topology split.
**Resolution target**: introduce a fallible provider-initialization and
transfer contract, migrate every implementor and caller in dependency order,
and preserve native dispatch without a CPU fallback or silent degradation.
Because the public backend contract changes, the migration requires an ADR,
in-repo caller conversion, negative no-device/transfer tests, and a full
backend matrix before closure.
**Evidence target**: warning-denied compilation, value-semantic typed-error
tests for unavailable devices and transfer failures, a production panic scan,
and provider feature gates on hosts with and without the required hardware.
**Status**: open; deliberately outside the native comparison-provider item
because it is a separate public failure-boundary migration.

### Device-local COW increment

`crates/coeus-hephaestus/src/storage.rs` now acquires the provider once,
allocates the replacement with the source buffer's `MemoryTier`, and invokes
`ComputeDevice::copy_buffer`. The old full-size host allocation and download /
upload round trip are removed. A generic storage contract verifies copied
values, tier preservation, one device copy, and no COW-triggered download.
The `StorageMut::make_unique` method is still infallible, so allocation and
copy failures remain the explicit residual of this broader failure-boundary
migration rather than being reclassified as closed here. Hosted exact-head run
`30336317894` passed the CUDA provider contracts job `90201872163`, WGPU
provider contracts job `90201872262`, ROCm provider contracts job `90201872299`,
and Metal provider contracts job `90201872213`. The required-device ROCm job
`90201873084` was skipped because no hosted AMD runner was dispatched; no
physical-device execution claim is made. The external `recurseml/analysis`
status returned its recurring analyzer error and is not repository-owned
verification.

### Native COW seam consolidation

The native Coeus WGPU and CUDA storage implementations now call the shared
Hephaestus `ComputeDevice::copy_buffer` contract instead of duplicating a
WGPU command encoder or a raw CUDA driver copy. Each provider storage test
detaches a shared device buffer, downloads both the detached and retained
buffers, and asserts value preservation. This consolidates the transfer
primitive without claiming a runtime speedup; matched device benchmarks remain
outside this increment. Hosted exact-head run `30339683483` passed CUDA
(`90212208770`), WGPU (`90212208755`), ROCm (`90212208702`), and Metal
(`90212208797`) provider contracts. Required-device ROCm (`90212209211`) was
skipped because no hosted AMD runner was dispatched.

## ATLAS-COEUS-DISPATCH-003: CUDA fused dispatch failure boundary

**Location**: coeus-cuda/src/lib.rs, coeus-cuda/src/kernels/{fuse,reduce}.rs,
and coeus-cuda/src/error.rs.
**Gap**: CUDA fused elementwise and fused-reduction helpers represented driver,
context, layout, compilation, cache, transfer, and launch failures as false.
The public entry points then evaluated the expression through the CPU path,
which silently changed the selected backend and discarded provider diagnostics.
**Resolution target**: return typed CudaBackendError values from the native
helpers and public fused entry points. The CUDA-feature path must either finish
on CUDA or return an error; the explicitly CPU-backed no-CUDA feature remains a
separate construction-time backend choice.
**Invariant**: selecting CudaBackend with the CUDA feature never silently
executes the Leto CPU evaluator after a native dispatch failure. Fused layout
metadata remains copied directly into device storage.
**Evidence target**: no boolean fused-dispatch residual, no CUDA-feature CPU
fallback residual, updated no-CUDA and CUDA-feature callers, warning-denied
package checks, focused Nextest, and CUDA differential tests when a device and
linker are available.
**Status**: implementation complete for the claimed files. Rustfmt, locked
metadata, diff hygiene, and residual scans pass. Exact-head run `30379272710`
passed the CUDA package check, warning-denied Clippy, selected provider
contracts, and doctests in job `90342897802`; WGPU, ROCm, and Metal provider
jobs also passed. The local locked package check remains blocked before
compilation because the Atlas-overlay-generated lockfile requires regeneration;
the lockfile remains outside this increment. Stale reduction and backend-error
edits were reconciled to the merged implementation. No CUDA runtime or
performance claim is made.

## ATLAS-CUDA-SAFETY-016: Remaining CUDA launch-parameter narrowing

**Location**: remaining non-convolution launchers under
`crates/coeus-cuda/src/kernels` and CUDA backend math dispatch.
**Gap**: the shared `GpuLayoutInfo` conversion, convolution forward output
count seam, standard/fused reduction launch boundaries, and elementwise
launches are closed by `ATLAS-CUDA-SAFETY-003`, `ATLAS-CUDA-SAFETY-004`, and
`ATLAS-CUDA-SAFETY-005`; optimizer launches are closed by
`ATLAS-CUDA-SAFETY-006`; the canonical 1-D pooling dispatcher is closed by
`ATLAS-CUDA-SAFETY-007`; 2-D pooling and the shared pooling seam are closed by
`ATLAS-CUDA-SAFETY-008`, `ATLAS-CUDA-SAFETY-009`, and
`ATLAS-CUDA-SAFETY-010`; unfold/fold is closed by
`ATLAS-CUDA-SAFETY-012`, and fused dispatch is closed by
`ATLAS-CUDA-SAFETY-014`, and the elementwise backend count/failure boundary is
closed by `ATLAS-CUDA-SAFETY-015`, but other CUDA kernel families still narrow
dimensions and derived counts with unchecked casts or products. These are
separate operation-family boundaries and were not silently folded into the
layout, reduction, elementwise, optimizer, pooling, matmul, attention,
unfold/fold, or transposed-convolution migrations.
**Resolution target**: migrate each operation family to checked, allocation-
free `u32` conversion and checked element counts, deleting local narrowing
paths while preserving native device dispatch and explicit failure results.
**Evidence target**: per-family feature-enabled check and Clippy, value-
semantic no-device regressions, and CUDA differential tests when the linker
and device environment are available.
**Status**: open; attention is closed by ATLAS-CUDA-SAFETY-011, unfold/fold
by ATLAS-CUDA-SAFETY-012, and transposed convolution by
ATLAS-CUDA-SAFETY-013; fused dispatch is closed by
ATLAS-CUDA-SAFETY-014 and elementwise backend counts/failures by
ATLAS-CUDA-SAFETY-015. The current environment cannot execute CUDA-feature
Nextest because its Windows GNU linker cannot resolve `-lcuda` from
`/usr/local/cuda-11.3/lib64/`.

## MS-446: provider identity and TCP teardown

**Location**: workspace `Cargo.toml`/`Cargo.lock` and
`crates/coeus-dist/src/tcp/mesh.rs`.
**Gap**: stale provider requirements allowed Cargo to retain parallel Git
source identities for the same Atlas contracts. `TcpMesh` also constructed a
default host-sized runtime per rank and stopped that runtime before dropping
its reactor-backed sockets; under concurrent Nextest processes, teardown could
wait for the 45-second debug I/O timeout.
**Resolution**: declare the current versioned Git contracts once and let the
lockfile own their exact commits. Construct the serialized mesh I/O runtime
with one worker in each pool, consolidate both construction sites on that
factory, and declare streams before the runtime so Rust's field-order drop
semantics close sockets before runtime destruction.
**Invariant**: one provider role resolves to one Cargo source identity, so
trait and type ownership cannot split across duplicate package instances.
Each mesh owns only the execution capacity its serial `block_on` contract can
use, and teardown reverses construction order.
**Evidence tier**: locked dependency metadata plus value-semantic real-socket
integration. The complete 64-test `coeus-dist` Nextest gate passes in 0.385 s;
the full workspace gate passes 938/938 in 82.449 s with no slow tests.

## MS-441: tensor benchmark still owns a Burn path

**Location**: `crates/coeus-tensor/Cargo.toml` and
`crates/coeus-tensor/benches/tensor_bench.rs`.
**Gap**: the production tensor provider is Coeus/Leto, but the benchmark still
declares a legacy NdArray dependency and duplicates every comparison through a
second backend. This keeps an obsolete dependency in the development graph and
creates a second benchmark vocabulary.
**Resolution target**: delete the Burn dependency and comparison functions;
retain provider-owned Coeus Sequential/Moirai and Leto dispatch measurements.
**Theorem**: for each retained benchmark input `x` and operation `f`, every
timed row evaluates a provider-owned path `P_f(x)`; removing the legacy row
does not alter the measured Coeus/Leto computation, and the benchmark has one
shape/layout/input SSOT per operation group.
**Evidence target**: manifest residue scan, committed-lock package
compilation, value-semantic Nextest, warning-denied diagnostics, doctests,
rustdoc, and the dependency-policy contract.

**Closure evidence**: the targeted residue scan is clean; locked package
compilation, 56/56 Nextest, warning-denied Clippy, five doctests,
warning-clean rustdoc, and locked metadata pass. The committed lock graph
selects Hephaestus `0.16.1` and Apollo `0.25.0` from local provider heads.

**Residual**: `coeus-nn` still owns a separate benchmark-only legacy dependency
and comparison suite; MS-442 is the next bounded deletion. This residual is
outside MS-441's tensor package scope and remains visible in the lock graph.

## MS-442: NN benchmark-only legacy path closed

**Location**: `crates/coeus-nn/Cargo.toml`, `crates/coeus-nn/benches/nn_bench.rs`, and the
committed workspace lock graph.
**Gap**: the NN Criterion target mixed 424 native provider measurements with
Burn comparison rows, keeping the obsolete provider in the development graph.
Deleting the target would also delete the canonical performance instrument and
is prohibited benchmark contraction.
**Resolution**: deleted only Burn setup, comparison rows, and the dependency;
retained all 211 operation groups and their Sequential/Moirai measurements.
No wrapper or compatibility path recreates the removed provider.
**Theorem**: removing one provider dimension preserves each native operation
scenario and shape exactly when the operation-group set and native rows remain
invariant. Mechanical census before and after establishes 211 groups and 424
native rows; the lock graph loses the benchmark-owned Burn packages.
**Evidence tier**: type-checked package boundary,
value-semantic native tests, and warning-denied diagnostics. Configured
Nextest is 268/268, doctests are 8/8 with two intentionally ignored, rustdoc is
warning-clean, and locked metadata resolves Eunomia 0.4.0, Leto 0.38.2, and
Hephaestus 0.17.0. Historical Burn references in CHANGELOG entries remain
archival records and are not dependency or runtime paths.

## ATLAS-PROVIDER-004: TCP loopback cluster isolation

**Location**: `crates/coeus-dist/src/tcp/mesh.rs`, `crates/coeus-dist/tests/dist_ops.rs`,
and `crates/coeus-python/{src/dist.rs,tests/binding_tests_dist.rs}`.
**Gap**: distributed tests selected an ephemeral port, released it, then asked
each rank to bind it. Concurrent nextest processes could claim that address in
the intervening interval and make a peer receive time out.
**Resolution**: `TcpMesh::create_loopback_cluster` keeps every OS-selected
loopback listener bound until the complete mesh is connected. Rust and PyO3
collective tests use that cluster directly; no shared temp-file lock, port
counter, or timing retry remains. Its `NonZeroUsize` Rust boundary and PyO3
`ValueError` conversion reject an invalid zero-sized cluster before socket
creation.
**Evidence tier**: real-socket Rust/PyO3 integration; 1008/1008 all-feature
nextest (real CUDA enabled); warning-denied workspace Clippy; 153 passing
doctests with 2 intentionally ignored; and warning-clean workspace Rustdoc.

## MS-440: Burn live-parity target removed

**Location**: `crates/coeus-nn/Cargo.toml`,
`crates/coeus-nn/tests/burn_live_parity.rs`, and `crates/coeus-nn/tests/pool1d_parity.rs`.
**Gap**: the Burn 0.16-only target stopped compiling while the workspace still
claimed it as a current oracle.
**Resolution**: removed the obsolete test target. Native pooling contracts now
instantiate the exact multi-channel 1-D oracle for Sequential and Moirai; the
existing 3-D contract already covers both providers. Burn remains limited to
dev-only Criterion comparisons, never a runtime or test compatibility path.
**Evidence tier**: analytical value oracles, provider-conformance nextest, and
the 1008/1008 all-feature workspace nextest gate with real CUDA enabled.

## MS-439: Named optimizer ownership closed

**Location**: `crates/coeus-autograd/src/parameter.rs`, `coeus-optim`,
`crates/coeus-nn/src/module.rs`, and `crates/coeus-python/src/optim.rs`.
**Gap**: optimizer construction flattened named module inventories into
`Vec<Var>`, severing persistence identity and forcing Burn-like positional
reload behavior.
**Resolution**: the parameter carrier moved to the deepest common owner;
optimizers retain names, module reload validates the complete inventory, and
the Python boundary requires names from the caller. No unnamed constructor or
compatibility re-export remains.
**Evidence tier**: type-level optimizer storage, typed count/name failures,
analytical and convergence tests across all five algorithms (20/20), and real
NN/PyO3 update integration (21/21), affected NN parity 144/144, Clippy,
Rustdoc, and doctests.

## MS-438: Stable hierarchical module parameters closed

**Location**: `crates/coeus-nn/src/{module,parameter}.rs` and composite module impls.
**Consumer driver**: RITK ADR 0004 requires stable parameter names for archived
displacement components and native optimizer updates without Burn visitors.
**Resolution**: the canonical `Module` seam owns named reflection. Leaf
weight/bias conventions are semantic, wider layouts must override rather than
falling back to ordinals, and composite paths encode child ownership. Plain and
named inventories retain identical ordering and shared gradient buffers.
**Evidence tier**: exact hierarchical name oracle, uniqueness over a two-layer
encoder/two-layer decoder, Arc pointer identity for gradient storage, full
nextest 410/410, and warning-denied Clippy.

## MS-437: Dimension-complete interpolation closed

**Location**: `crates/coeus-ops/src/interpolation.rs` and
`crates/coeus-autograd/src/ops/interpolation.rs`.
**Consumer driver**: RITK ADR 0004 requires one differentiable 2-D/3-D field
sampling contract before Burn displacement-field deletion.
**Resolution**: one const-dimension operation family implements both forward
and reverse mode, with sealed compile-time dimension evidence, sealed ZST
border policy selection, and no allocation in the per-point corner traversal.
The dimension-specific API is deleted.
**Evidence tier**: exact analytical values and gradients, independent central
differences across all 2-D/3-D coordinate axes with an epsilon/step-derived
bound, typed negative contracts, Sequential/Moirai differential agreement,
and affected nextest 282/282.

`cargo-semver-checks` could not construct the historical baseline: the cloned
Coeus revision loses sibling path dependencies, and its remote Mnemosyne pin
requires Themis `^0.8.0` while the tracked repository exposes 0.9.17. The
breaking surface is therefore classified explicitly by source inspection and
the workspace version is advanced to 0.6.0; no semver-clean claim is made.

The first affected-suite build exhausted the 584.73 GiB shared debug target,
including 276.64 GiB of inactive incremental artifacts. With no active Rust
processes, only `D:/atlas/target/debug/incremental` was removed and recreated,
recovering approximately 280 GB. The unchanged full suite then passed 282/282.

## MS-436: Bounded archived tensor state closed

**Location**: `crates/coeus-tensor/src/checkpoint.rs`.
**Consumer driver**: RITK ADR 0004 requires named, bounded persistence for
trainable displacement fields without retaining Burn records.
**Resolution**: `StateDict` owns a deterministic validated rkyv archive with a
zero-copy borrowed inspection view and explicit bounded materialization.
**Evidence tier**: exact tensor round trips, pointer-range proof for borrowed
payloads, deterministic insertion-order invariance, typed truncation/scalar/
limit/duplicate failures, package nextest 56/56, Clippy, Rustdoc, and doctests.

## MS-435: Depthwise 3-D convolution closed

**Location**: `crates/coeus-nn/src/conv/depthwise3d.rs`.
**Consumer driver**: RITK VMamba requires `groups == channels` for its local
volumetric feature kernel.
**Resolution**: Coeus now owns a depthwise 3-D module with a single
`[channels, 1, kernel, kernel, kernel]` parameter tensor and differentiable
channel slicing, canonical convolution, bias broadcast, and concatenation.
**Evidence tier**: exact two-channel output values, analytical input gradient,
and non-empty weight/bias gradients under nextest.

## MS-434: Rank-preserving batched matmul closed

**Location**: `crates/coeus-ops/src/matmul/kernel.rs` and
`crates/coeus-autograd/tests/autograd/linalg.rs`.
**Gap**: rank-N matmul flattened logical batch axes into one output axis, and
backward accumulation passed the restored rank-N gradient destination directly
to a rank-3 Leto kernel. Rank-4 Swin attention therefore panicked during real
backward execution.
**Resolution**: output tensors retain their logical batch shape, while forward
and accumulating backward construct explicit flattened dispatch layouts.
**Evidence tier**: exact rank-4 forward values and analytical gradients for
both operands; affected Coeus nextest 689/689; warning-denied Clippy clean.

## MS-433: Rank-generic linear projection closed

**Location**: `crates/coeus-nn/src/linear.rs`.
**Consumer driver**: RITK TransMorph window attention and MLP operate on
rank-3 and rank-5 tensors whose feature width is the last axis.
**Resolution**: `Linear::forward` validates the last-axis contract, flattens
all leading axes, applies the single canonical matrix projection, and restores
the leading shape. Rank-2 inputs retain the direct path. No consumer adapter
or parallel linear implementation is required.
**Evidence tier**: exact analytical rank-3/rank-5 forward values and
reverse-mode rank-3 input/weight/bias gradients; full package nextest 409/409;
warning-denied Clippy and rustdoc clean.

## Known Gaps & Residual Risks

### G-049: special-function unary lane
**Location**: `crates/coeus-core/src/dtype/traits.rs`,
`crates/coeus-ops/src/unary/math.rs`, `crates/coeus-python/src/ops/elementwise.rs`,
`crates/coeus-python/tests/test_pytorch_parity.py`.
**Compared against**: PyTorch `torch.erf`, `torch.special.erfc`,
`torch.nn.functional.gelu(approximate="none")`, and
`torch.special.gammaln`.
**Resolution**: MS-237 routes `erf`/`erfc` through Eunomia, adds `lgamma`
through CPU/Leto unary dispatch, and exposes forward-only Python
`gammaln`/`lgamma`. Exact GELU remains the default Python `gelu` surface and
is rechecked against PyTorch at f64.
**Residual risk**: `gammaln` backward is intentionally unavailable because
`d/dx lgamma(x) = digamma(x)`, and Eunomia does not expose `digamma` yet.
Python raises `NotImplementedError` for grad-tracked `gammaln` inputs instead
of emitting a fake or zero gradient.
**Evidence tier**: value-semantic Rust tests plus f64 differential PyTorch
parity.

### ~~G-047: Apollo-backed FFT autograd/Python parity missing~~ **CLOSED**
**Location**: `crates/coeus-autograd/src/ops/fft.rs`,
`crates/coeus-python/src/ops/fft.rs`, `crates/coeus-python/tests/test_pytorch_parity.py`.
**Gap**: FFT lived outside the public autograd/Python surface after the
Burn dev-dependency migration, so Apollo-backed signal transforms had no
value-semantic gradient parity in Coeus.
**Resolution**: MS-218 added public autograd FFT wrappers and a thin PyO3
complex tensor binding. The Python parity test compares `pycoeus.fft` against
`torch.fft.fft` and verifies `fft_energy` input gradients against PyTorch
autograd at f64.
**Evidence tier**: analytical/value-semantic Rust tests plus differential
PyTorch parity.

### G-046: Python-binding parity closure for AdaptiveMaxPool
**Location**: `crates/coeus-python/src/nn/pool.rs`, `crates/coeus-python/src/nn/mod.rs`,
`crates/coeus-python/src/lib.rs`, `crates/coeus-python/tests/test_pytorch_parity.py`,
`crates/coeus-python/tests/test_jax_parity.py`.
**Compared against**: PyTorch `torch.nn.AdaptiveMaxPool1d/2d` and JAX.
**Gap**: After PR #109 (AdaptiveAvgPool differentiable), PR #110
(AdaptiveAvgPool dx parity in PyTorch+JAX), and PR #111
(`b3e993b` AdaptiveMaxPool1d/2d differentiable in Rust core), the Python
binding surface had not been extended for the Max variant. The
`test_adaptive_max_pool_backward_matches_pytorch` JIT-imported
`pycoeus.AdaptiveMaxPool1d/2d` from `test_pytorch_parity.py:2555`,
which would have raised `AttributeError` against the old binding.
**Resolution (PR #112 = `d1ad9d2`, peer merge)**: Added
`PyAdaptiveMaxPool1d` and `PyAdaptiveMaxPool2d` thin PyO3 wrappers
(mirroring the `PyAdaptiveAvgPool*` pattern), with `m.add_class::<>`
registrations and `pool.rs` re-exports. PR #112 also added the JAX
parity fixture (`test_adaptive_max_pool_matches_jax`) using a per-region
`jnp.max` reference plus `jax.value_and_grad`. Three-way parity
(Rust core \u2194 PyTorch \u2194 JAX) now holds for forward + input gradient.
**Evidence tier**: differential/value-semantic pytest outcomes (PyTorch
parity file 2/2 for the adaptive-max family) plus 379/379 passing Rust nn
tests.
**Acceptance**: closed. Future MS work may add a Burn benchmark row to
match G-043 expansion (already has AvgPool families).

### G-045 forward-only modules sweep:
### G-043: Burn/PyTorch NN benchmark matrix remains partial
**Location**: `crates/coeus-nn/benches/nn_bench.rs`,
`crates/coeus-python/tests/test_pytorch_parity.py`
**Compared against**: Burn `burn::nn` module families and PyTorch `torch.nn`
module families.
**Gap**: Current Coeus-vs-Burn benchmarks cover selected forward rows
(Linear, LayerNorm, RMSNorm, LSTM, GRU, InstanceNorm2d, CrossEntropyLoss, MSELoss,
HuberLoss, ReLU, GeLU, PReLU, Sigmoid, Tanh, SiLU, LeakyReLU, Mish, SwiGLU,
Softmax, Dropout eval, Conv1d/2d/2d-backward/3d, ConvTranspose1d/3d, MHA self-attention,
Transformer encoder layer, Embedding lookup, EmbeddingBag sum, AdaptiveAvgPool2d(1,1),
AdaptiveMaxPool2d(1,1), BatchNorm1d/2d/3d eval forward, GroupNorm forward,
MaxPool1d/2d/3d forward, AvgPool1d/2d/3d forward),
not the full NN family set needed to claim Burn-level performance parity.
PyTorch differential coverage similarly remains module-family selective.
**Acceptance**: Add a benchmark/parity manifest keyed by module family, then add
rows for every newly implemented G-035..G-042 family with Coeus sequential,
Moirai, WGPU/CUDA where applicable, Burn NdArray where comparable, and PyTorch
Python differential tests at f64. Report median/confidence intervals for
benchmarks and analytical tolerance derivations for numerical comparisons.
**Evidence tier**: source-surface audit plus external API documentation audit.
**2026-07-08 update**: added `bench_maxpool3d_forward`/`bench_avgpool3d_forward`
(Coeus Sequential vs Moirai only — no Burn 0.16.1 `max_pool3d`/`avg_pool3d` op
exists to compare against, confirmed against the pinned `burn-tensor` source).
**2026-07-08 update 2**: added `bench_interpolate2d_nearest_forward`/
`bench_interpolate2d_bilinear_forward` (full 3-way: Burn's
`nn::interpolate::Interpolate2d` exists at the pinned version, unlike 3D
pooling). Remaining open surface: MHA cross-attention, vanilla RNN/RNNCell,
`Bidirectional` wrapper — all confirmed to have no Burn 0.16 equivalent
(vanilla RNN: `nn::rnn` has only `lstm.rs`/`gru.rs`/`gate_controller.rs`, no
plain-RNN or bidirectional wrapper) — these would be Coeus-only benches
(Sequential vs Moirai), same pattern as MaxPool3d/AvgPool3d.
**2026-07-08 update 3**: added `bench_bilinear_forward` (Coeus Sequential vs
Moirai only, two distinct inputs via `bilinear_forward`; confirmed no
`nn::Bilinear`/`BilinearConfig` in `burn-core` 0.16.0). Remaining open
surface: MHA cross-attention, vanilla RNN/RNNCell, `Bidirectional` wrapper.
**2026-07-11 update**: added `bench_rnn_forward` for the vanilla RNN sequence
path (Sequential vs Moirai, `[4, 32, 64] -> hidden 128`). The pinned Burn 0.16
`nn::rnn` source has LSTM and GRU only, so no Burn baseline is fabricated.
**2026-07-11 update 2**: added `bench_rnn_cell_forward` for one vanilla RNN
transition (Sequential vs Moirai, batch 4, input 64, hidden 128), separating
cell cost from sequence unrolling. Burn remains inapplicable for this family.
**2026-07-11 update 3**: added `bench_bidirectional_rnn_forward` (Sequential
vs Moirai, `[4, 32, 64] -> 2×hidden 128`), measuring the wrapper's real
reverse-and-concatenate path. Burn has neither required vanilla-RNN primitive.
**2026-07-11 update 4**: added `bench_mha_cross_attention_forward` (Sequential
vs Moirai, query `[8, 32, 256]`, memory `[8, 64, 256]`, 8 heads). This closes
the identified no-Burn MHA cross-attention row; G-043 remains open only for the
broader every-family manifest audit.
**2026-07-11 update 5**: added `bench_local_response_norm_forward`
(Sequential vs Moirai, `[8, 32, 16, 16]`, size 5). The pinned Burn 0.16 source
has no `LocalResponseNorm` module surface, so no synthetic Burn row is used.
**2026-07-11 update 6**: extended the EmbeddingBag workload with mean-mode
rows, comparing Burn `Embedding::forward + mean_dim` to Coeus Sequential and
Moirai on the same indexed bags.

### ~~G-042: Quantized and lazy module parity policy missing~~ **CLOSED (non-goal)**
**Location**: `crates/coeus-nn/src/lib.rs`, `crates/coeus-python/src/lib.rs`
**Compared against**: PyTorch quantized/lazy NN module families.
**Closed by**: MS-212 — Recorded as an explicit non-goal for Coeus v0.x.
The typed `Scalar` + `BackendOps<T>` design provides a natural extension point for
quantized numerics (e.g., a `QuantizedBackend` implementing `BackendOps<i8>`) without
dedicated lazy-module infrastructure. Coeus v0.x targets f32/f64 parity with Burn and
PyTorch for the standard NN module families; quantized/lazy support is deferred to
a future typed-dtype extension documented in `docs/roadmap.md`.
**Evidence tier**: design decision / non-goal record.

### ~~G-041: Regularization, sparse, and local-response modules incomplete~~ **CLOSED**
**Location**: `crates/coeus-nn/src/dropout.rs`, `crates/coeus-nn/src/embedding.rs`,
`crates/coeus-nn/src/normalization/`, `crates/coeus-python/src/nn/`
**Compared against**: Burn `GaussianNoise`/`LocalResponseNorm` and PyTorch
`AlphaDropout`, `FeatureAlphaDropout`, `EmbeddingBag`, and
`LocalResponseNorm`.
**Closed by**: MS-209 — Added `coeus_nn::EmbeddingBag` with `sum`/`mean`/`max`
aggregation and offsets semantics, value-semantic + backward tests
(`crates/coeus-nn/tests/embeddingbag_tests.rs`), and thin PyO3 wrapper
`pycoeus.EmbeddingBag` delegating to Rust (`crates/coeus-python/src/nn/embedding.rs`,
registered in `crates/coeus-python/src/{nn/mod.rs,lib.rs}`). Together with MS-208
(`AlphaDropout`, `FeatureAlphaDropout`, `GaussianNoise`, `LocalResponseNorm`),
this closes the Rust/Python module-surface gap for G-041.
**Evidence tier**: source-surface audit plus external API documentation audit.

### ~~G-040: Recurrent parity lacks vanilla and bidirectional sequence variants~~ **CLOSED**
**Location**: `crates/coeus-nn/src/rnn/`, `crates/coeus-python/src/nn/rnn.rs`
**Closed by**: MS-206/MS-219 — Vanilla RNN/RNNCell, GRU/GRUCell, LSTM/LSTMCell
with a generic `Bidirectional<M: Module>` wrapper and thin PyO3 wrappers
(`PyBidirectional`, `PyGRUCell`, `PyLSTMCell`, `PyRNNCell`). Bidirectional
wrapper reverses along the time axis via `flip`, applies the backward module,
and concatenates via `cat` — no per-cell code duplication.
**Evidence tier**: source-surface audit plus external documentation audit.

### ~~G-039: Python loss wrappers lag existing Rust loss surface~~ **CLOSED**
**Location**: `crates/coeus-nn/src/loss.rs`, `crates/coeus-python/src/losses.rs`,
`crates/coeus-python/src/lib.rs`, `crates/coeus-python/tests/test_pytorch_parity.py`
**Closed by**: MS-182 — Added thin PyO3 wrappers
`pycoeus.{kl_divergence,margin_ranking_loss}` delegating directly to
`coeus_nn::loss::{kl_divergence,margin_ranking_loss}`, exported both in the
module registration/stub surface, and added PyTorch differential tests
`test_kl_divergence_matches_pytorch` and
`test_margin_ranking_loss_matches_pytorch` asserting scalar forward and input
gradients at f64. Evidence tier: differential/empirical.

### ~~G-037: Activation surface remains incomplete versus Burn/PyTorch~~ **CLOSED**
**Location**: `crates/coeus-core/src/dtype/{traits.rs,float/cpu_unary.rs,int.rs}`,
`crates/coeus-wgpu/src/kernels/unary.rs`,
`crates/coeus-autograd/src/ops/activation/{ext.rs,relu.rs,mod.rs}`,
`crates/coeus-nn/src/{activation.rs,lib.rs}`,
`crates/coeus-python/src/{activations.rs,lib.rs}`,
`crates/coeus-nn/tests/nn_ops/activations/act_extended/`,
`crates/coeus-python/tests/test_pytorch_parity.py`
**Closed by**: MS-186 — Added nine new activation functions end-to-end:
**Hardtanh** (`coeus_nn::hardtanh` / `Hardtanh` Module, default `[-1, 1]`),
**Hardsigmoid**, **Hardswish**, **Hardshrink** (default λ=0.5),
**Softshrink** (default λ=0.5), **Softsign**, **Threshold** (default `threshold=0, value=0`),
**Celu** (default α=1.0), and **PReLU** (single scalar α default 0.25). Implementation extends
`coeus-core::CpuUnaryOp` with 18 new variants (forward + gradient pairs, single-parameter scalars
packed via `f64::to_bits` following the `LeakyRelu` precedent, pair parameters
packed as little-endian `f32` lanes inside one `u64`), adds the corresponding
float dispatcher in `coeus-core::dtype::float::cpu_unary`, and extends the WGSL
codegen emitter in `coeus-wgpu::kernels::unary` for GPU parity. Tracked
autograd nodes follow the existing `LeakyReluNode` manual-node pattern for
parameterized ops, and the generic `unary_op<T,B,Op>` ZST template for
parameter-free ops (Hardsigmoid, Hardswish, Softsign).
PReLU's α is exposed as a single scalar α in the tracked functional
(`coeus_autograd::prelu(x, alpha)`); per-channel PReLU composes via
`coeus_ops::broadcast_to`. Kink/subgradient points documented inline and excluded from
the PyTorch differential tests per PyTorch's convention (e.g. Hardtanh at
x=±min/max → 0, Hardsigmoid at x=±3 → 0, Hardshrink/Softshrink at |x|=λ → 0,
Threshold at x=threshold → 0). Evidence tier: value-semantic Rust analytical
backward tests (`crates/coeus-nn/tests/nn_ops/activations/act_extended/` covering 9 ops at f64
with closed-form formula oracles) plus PyTorch f64 differential tests
(`crates/coeus-python/tests/test_pytorch_parity.py` adds 9 new tests using the
existing `_assert_activation_parity` helper). MS-187 corrected the regression
where gradient operators evaluated on `grad_out` instead of the saved input and
where pair-parameter decoding treated truncated halves as `f64` bit patterns.

### ~~G-038: Loss and distance surface remains below PyTorch coverage~~ **CLOSED**
**Location**: `crates/coeus-nn/src/loss.rs`, `crates/coeus-python/src/losses.rs`,
`crates/coeus-autograd/src/ops/nn/loss/`
**Compared against**: PyTorch loss and distance families.
**Closed by**: MS-219 (22/23 losses) + **MS-225** (CTCLoss — final item).
- MS-225: Added `CtcLossNode` autograd node with log-space forward-backward DP
  (α and β tables), exposed as `coeus_nn::ctc_loss` and `pycoeus.ctc_loss`.
  5 analytical Rust tests (single-frame oracle, two-frame oracle, batch oracle,
  gradient propagation, nn/autograd consistency). PyTorch parity test at f64
  atol=1e-6 against `torch.nn.functional.ctc_loss(reduction='mean')`.
- **All 23/23 PyTorch loss/distance families now have Coeus parity.**
**Evidence tier**: analytical/value-semantic Rust tests + differential PyTorch
parity (1/1 CTC test at f64).

### ~~G-037: Activation surface remains incomplete versus Burn/PyTorch~~ **CLOSED**
**Location**: `crates/coeus-nn/src/activation.rs`, `crates/coeus-python/src/activation.rs`
**Compared against**: Burn activations and PyTorch activation modules/functions.
**Gap**: Coeus covers common activations, but lacks Rust module/API parity for
PReLU, CELU, Hardshrink, Hardsigmoid, Hardtanh, Hardswish, Softshrink,
Softsign, Threshold, and a Rust `nn` GLU/SwiGLU family surface matching
framework module expectations.
**Acceptance**: Add one generic Rust activation implementation per operation
family with analytical derivative tests; expose PyO3 wrappers as delegation
only; add PyTorch/Burn differential tests at f64, with kink/subgradient points
handled by documented analytical contracts.
**Evidence tier**: source-surface audit plus external API documentation audit.
**Closed by**: Cumulative work across MS-79 (GLU), MS-131 (leaky_relu/softplus/mish), MS-211 (checklist note), MS-217 (PReLU/LeakyReLU); all target families (PReLU, CELU, hardshrink, softshrink, softsign, threshold, GLU, SwiGLU) now have autograd backward, nn wrappers, Python bindings, and PyTorch/JAX differential parity. Hardswish/hardsigmoid backward routing verified correct (this session). Hardtanh covered by existing clamp/clamp_min/clamp_max path.

### ~~G-036: Pooling, adaptive pooling, and unfold/fold coverage incomplete~~ **CLOSED**
**Location**: `crates/coeus-nn/src/pool.rs`, `crates/coeus-python/src/nn/pool.rs`,
`crates/coeus-nn/src/conv/unfold_fold.rs`, `crates/coeus-ops/src/backend_ops/traits/unfold_fold.rs`
**Compared against**: Burn `Unfold4d` and PyTorch pooling/unfold/fold module
families.
**Gap**: Coeus exposes 2D/3D average and max pooling plus selected global
pooling wrappers, but lacks 1D pooling modules, adaptive pooling surfaces beyond
global wrappers, and Unfold/Fold/Unfold4d parity surfaces.
**Closed by**: MS-206 (pool1d) and MS-211 (unfold/fold):
- MS-206: `MaxPool1d`/`AvgPool1d` with forward+backward, autograd, Python bindings.
- MS-211: `UnfoldFoldOps` sub-trait added to `BackendOps` (8th concern); CPU,
  CUDA, and WGPU kernels for `unfold1d`/`fold1d`/`unfold2d`/`fold2d`;
  `coeus_nn::{Unfold1d, Fold1d, Unfold2d, Fold2d}` NN modules; and parity tests
  covering shape, value semantics, and round trips. ATLAS-WGPU-CORRECTNESS-001
  removed the WGPU no-op methods and added native device coverage for the
  unfold/fold family plus padded/dilated 1D pooling forward/backward paths.
**Evidence tier**: analytical/value-semantic Rust tests.

### ~~G-035: ConvTranspose3d parity missing~~ **CLOSED**
**Location**: `crates/coeus-nn/src/conv/`, `crates/coeus-python/src/nn/conv.rs`
**Compared against**: PyTorch `ConvTranspose3d` and the existing Coeus
ConvTranspose1d/2d family.
**Closed by**: MS-185 — Added `coeus-ops` forward operation, backend default
method, tracked autograd backward node, `coeus-nn::ConvTranspose3d`,
Sequential/Moirai value-semantic module tests, `pycoeus.ConvTranspose3d`, and
PyTorch f64 differential coverage for forward output plus input, weight, and bias
gradients. WGPU/CUDA acceleration is deferred to a future GPU-sprint milestone;
the CPU parity surface is complete and matches PyTorch at f64.
**Evidence tier**: analytical/value-semantic Rust tests (conv_transpose_nn_parity.rs) +
differential PyTorch parity (`test_conv_transpose3d_matches_pytorch` at f64).

### ~~G-034: Linear/loss tests only checked gradient existence~~ **CLOSED**
**Location**: `crates/coeus-nn/tests/nn/linear_activation_loss.rs`
**Closed by**: MS-179 — Replaced Linear, MSE, and CrossEntropy
gradient-existence checks with value-semantic assertions. Linear now pins
input, weight, and bias gradients for a deterministic all-ones layer; MSE pins
the mean-reduction derivative; CrossEntropy pins the stable
softmax-minus-onehot mean-reduction gradient. Evidence tier:
analytical/value-semantic Rust tests.

### ~~G-033: Conv module tests only checked gradient existence~~ **CLOSED**
**Location**: `crates/coeus-nn/tests/nn/conv1d.rs`,
`crates/coeus-nn/tests/nn/conv2d.rs`,
`crates/coeus-nn/tests/nn/conv3d_pool3d.rs`
**Closed by**: MS-178 — Replaced Conv1d/Conv2d/Conv3d module backward smoke checks with
exact analytical assertions for input, weight, and bias gradients under
deterministic all-ones kernels. Evidence tier: analytical/value-semantic Rust
tests.

### ~~G-032: TCP collectives could hang past nextest timeout~~ **CLOSED**
**Location**: `crates/coeus-dist/src/tcp/mesh.rs`,
`crates/coeus-dist/tests/dist_ops.rs`
**Closed by**: MS-177 — Added deterministic TCP test port reservation through a
file-backed cross-process port allocator lock, and debug-mode mesh timeouts around
connect, accept, peer-rank read, send, and recv paths. Connect retry backoff
remains async through `moirai_async::sleep`, so the debug diagnostics do not
introduce executor-blocking sleep. The lock creation path also treats Windows
`PermissionDenied` as lock contention rather than a distinct fatal failure,
preserving the stale-lock timeout diagnostic under nextest process contention.
Evidence tier: empirical/value-semantic through the `coeus-dist` package gate.

### ~~G-031: JAX harness lacked regression/binary loss parity~~ **CLOSED**
**Location**: `crates/coeus-python/tests/test_jax_parity.py`
**Closed by**: MS-175 — Added `test_{mse_loss,binary_cross_entropy,huber_loss}_matches_jax`,
asserting forward loss and prediction gradient against inline JAX references at f64
(Huber δ=1.0 spans both regions; BCE probabilities in (0,1)). Completes the
regression/binary loss parity against JAX, symmetric with PyTorch. Evidence tier:
differential/empirical.

### ~~G-030: JAX harness lacked LayerNorm/RMSNorm parity~~ **CLOSED**
**Location**: `crates/coeus-python/tests/test_jax_parity.py`
**Closed by**: MS-174 — Added `test_{layernorm,rmsnorm}_matches_jax`,
asserting forward output and gradients against inline f64 JAX references.
LayerNorm covers input/gamma/beta gradients; RMSNorm covers input/gamma
gradients. Evidence tier: differential/empirical.

### ~~G-029: JAX harness lacked softmax/log-softmax/cross-entropy parity~~ **CLOSED**
**Location**: `crates/coeus-python/tests/test_jax_parity.py`
**Closed by**: MS-173 — Added `test_{softmax,log_softmax,cross_entropy_loss}_matches_jax`,
asserting forward output and gradient against `jax.nn.{softmax,log_softmax}` and a
fused log-softmax+NLL mean reference at f64. Extends the JAX harness to the
classification/softmax path, symmetric with the PyTorch coverage. Evidence tier:
differential/empirical.

### ~~G-028: `BackendOps` mixed every operation concern in one trait~~ **CLOSED**
**Location**: `crates/coeus-ops/src/backend_ops/trait_def.rs`,
`crates/coeus-ops/src/backend_ops/cpu_impl.rs`
**Closed by**: MS-171 — Added single-concern operation traits and made
`BackendOps` an aggregate super-trait with a blanket impl. CPU dispatch now
implements one operation trait per concern, preserving the existing kernel leaf
modules while eliminating duplicate blanket-impl coherence failures. Evidence
tier: compile/lint/docs plus value-semantic `coeus-ops` nextest coverage.

### ~~G-027: JAX harness lacked elementwise activation parity~~ **CLOSED**
**Location**: `crates/coeus-python/tests/test_jax_parity.py`
**Closed by**: MS-168 — Added `_assert_activation_matches_jax` (`jax.grad` for
backward) and `test_{silu,mish,elu,softplus,leaky_relu}_matches_jax`, asserting
forward output and input gradient against `jax.nn.*` at f64. Extends the JAX
harness beyond Linear/MHA/decoder to the elementwise activations, symmetric with
the PyTorch coverage of MS-167. Evidence tier: differential/empirical.

### ~~G-026: Elementwise activation differential parity missing (only GELU covered)~~ **CLOSED**
**Location**: `crates/coeus-python/tests/test_pytorch_parity.py`
**Closed by**: MS-167 — Added a shared `_assert_activation_parity` helper and
`test_{silu,mish,elu,softplus,leaky_relu}_matches_pytorch`, asserting forward
output and input gradient against `torch.nn.functional.*` at f64 on mixed-sign
inputs. LeakyReLU excludes the `x=0` kink (implementation-defined subgradient);
the C1 activations include it. Evidence tier: differential/empirical.

### ~~G-025: GlobalAvg/MaxPool2d differential parity missing~~ **CLOSED**
**Location**: `crates/coeus-python/tests/test_pytorch_parity.py`
**Closed by**: MS-166 — Added `test_global_avg_pool2d_matches_pytorch` and
`test_global_max_pool2d_matches_pytorch` (input `[2,3,4,4]` → `[N,C,1,1]`),
asserting forward output and input gradient against
`torch.nn.functional.adaptive_{avg,max}_pool2d(x, 1)` at f64, `atol=1e-10`.
Covers the uniform-distribution (avg) and argmax-routing (max) backward paths,
replacing prior existence-only binding coverage. Evidence tier: differential/empirical.

### ~~G-024: Zero-numel collectives skipped per-rank numel validation~~ **CLOSED**
**Location**: `crates/coeus-dist/src/local.rs`,
`crates/coeus-dist/src/tcp/collectives.rs`
**Closed by**: MS-165 — Local and TCP `all_gather`, rooted `gather`, and rooted
`scatter` now validate per-rank output/input tensor element counts before
zero-numel early returns. Evidence tier: panic-contract nextest coverage.

### ~~G-023: Conv2d canonical CPU path retained dot-per-output overhead~~ **CLOSED**
**Location**: `crates/coeus-ops/src/backend_ops/cpu_impl/conv/conv2d.rs`,
`crates/coeus-core/src/dtype/traits.rs`
**Closed by**: MS-164 — Added the `Scalar::axpy_slice` seam and rewrote the
canonical contiguous Conv2d forward path as an output-stationary AXPY row
kernel, with coarser row-block partitioning for Moirai execution. Evidence
tier: value-semantic scalar/Conv2d tests plus Criterion Conv2d row.

### ~~G-022: Local collective staging mutex covered payload work~~ **CLOSED**
**Location**: `crates/coeus-dist/src/local.rs`
**Closed by**: MS-163 — Local collectives now snapshot staged rank payloads
under the shared staging mutex and perform reductions/output copies after the
lock is released; root scatter extracts tensor host data before publishing
payloads. Evidence tier: value-semantic local communicator tests.

### ~~G-021: KL/MarginRanking tracked loss coverage missing~~ **CLOSED**
**Location**: `crates/coeus-autograd/src/ops/nn/loss`,
`crates/coeus-nn/tests/burn_live_parity.rs`, `crates/coeus-nn/tests/loss_parity.rs`
**Closed by**: MS-161 — Added tracked KL divergence and margin ranking loss
entry points, NN wrappers, analytical forward/backward tests, and
sequential/Moirai loss parity checks. Evidence tier: analytical Rust tests plus
package nextest.

### ~~G-020: BCE/Huber Python differential parity missing~~ **CLOSED**
**Location**: `crates/coeus-python/tests/test_pytorch_parity.py`
**Closed by**: MS-156 — Added `test_binary_cross_entropy_matches_pytorch`
and `test_huber_loss_matches_pytorch`, asserting scalar losses and prediction
gradients against `torch.nn.functional.binary_cross_entropy` and
`torch.nn.functional.huber_loss` at f64. Evidence tier:
differential/empirical.

### ~~G-019: SiLU/Mish tests still had existence-only gradient checks~~ **CLOSED**
**Location**: `crates/coeus-nn/tests/nn_silu_tests.rs`,
`crates/coeus-nn/tests/nn_mish_tests.rs`
**Closed by**: MS-154 — Module and non-contiguous SiLU/Mish paths now assert
analytical forward and backward values instead of only checking that gradients
exist. Evidence tier: analytical value-semantic Rust tests.

### ~~G-018: CrossEntropy/NLL loss differential parity missing~~ **CLOSED**
**Location**: `crates/coeus-python/tests/test_pytorch_parity.py`
**Closed by**: MS-153 — Added `test_cross_entropy_loss_matches_pytorch` and
`test_nll_loss_matches_pytorch` (logits `[3,4]`, class-index targets), asserting
the scalar loss and logit gradient against `torch.nn.functional.cross_entropy`
and `nll_loss(log_softmax(x))` at f64, `atol=1e-10` (both mean reduction). Pins
the fused log-softmax+NLL forward and the softmax-minus-onehot backward — the
classification training signal. Evidence tier: differential/empirical.

### ~~G-017: FeedForward binding monolith~~ **CLOSED**
**Location**: `crates/coeus-python/src/nn/feedforward.rs`
**Closed by**: MS-152 — Replaced the flat binding file with
`crates/coeus-python/src/nn/feedforward/mod.rs`, `feedforward/positional.rs`, and
`feedforward/transformer/*` leaf modules while preserving `pycoeus` `nn`
registration exports. Evidence tier: compile/lint/docs plus Rust and Python
binding tests.

### ~~G-016: MaxPool2d/AvgPool2d differential parity missing~~ **CLOSED**
**Location**: `crates/coeus-python/tests/test_pytorch_parity.py`
**Closed by**: MS-151 — Added `test_maxpool2d_matches_pytorch` and
`test_avgpool2d_matches_pytorch` (kernel=2, stride=2 on `[1,2,4,4]`), asserting
forward output and input gradient against `torch.nn.functional.{max,avg}_pool2d`
at f64, `atol=1e-10`. Exercises the max-routing (gradient to argmax) and
average-distribution (uniform 1/window) backward paths, previously covered only
by binding smoke tests. Evidence tier: differential/empirical.

### ~~G-015: Scalar identity still depended on num-traits/libm~~ **CLOSED**
**Location**: `Cargo.toml`, `crates/coeus-core/src/dtype/traits.rs`,
`crates/coeus-core/src/dtype/float/erf.rs`, `crates/coeus-ops/src/sparse/ops.rs`
**Closed by**: MS-150 — Removed Coeus' direct `num-traits`/`libm` dependency
path from the scalar contract, added canonical `Scalar::zero()`/`one()`, and
routed GELU/erf through a Coeus-owned piecewise rational implementation.
Evidence tier: compile/lint/docs plus value-semantic Rust tests.

### ~~G-014: GroupNorm Python differential parity missing~~ **CLOSED**
**Location**: `crates/coeus-python/tests/test_pytorch_parity.py`
**Closed by**: MS-149 — Added `test_groupnorm_matches_pytorch`, asserting
GroupNorm forward output plus input, weight, and bias gradients against
`torch.nn.functional.group_norm` at f64, `atol=1e-10`.
Evidence tier: differential/empirical.

### ~~G-013: Duplicate einsum implementation under shape::util~~ **CLOSED**
**Location**: `crates/coeus-ops/src/shape/einsum.rs`,
`crates/coeus-ops/src/shape/util/einsum.rs`
**Closed by**: MS-148 — Deleted the byte-identical utility copy and routed
`shape::util::{einsum,einsum3}` through the canonical parent implementation.
Evidence tier: compile/lint/docs plus value-semantic tests (`coeus-ops` full
nextest 189/189, focused einsum nextest 12/12).

### ~~G-001: PyTransformerEncoderLayer stateless binding~~ **CLOSED**
**Location**: `crates/coeus-python/src/nn/feedforward.rs` — `PyTransformerEncoderLayer`
**Closed by**: MS-127 — Refactored to stateful `Py<PyLayerNorm>` + `Py<PyMultiHeadAttention>` +
`Py<PyFeedForward>` sub-module fields; `parameters()` returns 16 params; forward replaces
dummy weights from Python sub-objects; `test_transformer_encoder_layer_matches_pytorch` PASSES.

### ~~G-002: PyTransformerEncoder stateless binding~~ **CLOSED**
**Location**: `crates/coeus-python/src/nn/feedforward.rs` — `PyTransformerEncoder`
**Closed by**: MS-128 — Refactored to stateful `Vec<Py<PyTransformerEncoderLayer>>` field;
`parameters()` returns `16 × N` params; `forward()` chains layer-wise Pre-LN forwards without
re-creating Rust encoder; `build_from_layer`/`from_rust_layer` inherent methods eliminate code
duplication with `PyTransformerEncoderLayer::new()`. Tests:
`transformer_encoder_stack_2layer_self_consistent` (structural, 111/111 Rust),
`transformer_encoder_stack_2layer_forward_matches_burn` (differential, Burn NdArray),
`test_transformer_encoder_stack_matches_pytorch` (differential, PyTorch, 8/8 Python).

### ~~G-003: PyTransformerDecoderLayer stateless binding~~ **CLOSED**
**Location**: `crates/coeus-python/src/nn/feedforward.rs` — `PyTransformerDecoderLayer`
**Closed by**: MS-129 — Refactored to stateful `Py<PyLayerNorm>×3` + `Py<PyMultiHeadAttention>×2`
(self_attn + cross_attn) + `Py<PyFeedForward>` sub-module fields; `parameters()` returns 26 params;
`forward(tgt, memory)` injects stored weights into Rust forward; `build_from_layer<H>` /
`from_rust_layer<H>` inherent methods (SSOT, shared with `PyTransformerDecoder`).

### ~~G-004: PyTransformerDecoder missing~~ **CLOSED**
**Location**: `crates/coeus-python/src/nn/feedforward.rs` — class did not exist
**Closed by**: MS-129 — Added `PyTransformerDecoder` with `Vec<Py<PyTransformerDecoderLayer>>`
layers; `parameters()` returns `26 × N`; `forward(tgt, memory)` chains layer-wise Pre-LN
cross-attention forwards; `num_layers` getter; `zero_grad()`. Tests:
`transformer_decoder_layer_forward_is_deterministic` (determinism),
`transformer_decoder_stack_2layer_self_consistent` (structural, 277/277 Rust),
`transformer_decoder_forward_uses_self_as_memory` (API contract),
`test_transformer_decoder_layer_matches_pytorch` (differential, PyTorch, 10/10 Python),
`test_transformer_decoder_stack_matches_pytorch` (differential, PyTorch, 10/10 Python).

### ~~G-005: PyTransformer (full seq2seq) missing~~ **CLOSED**
**Location**: `crates/coeus-python/src/nn/feedforward.rs` — class did not exist
**Closed by**: MS-131 — Added `PyTransformer` wrapping `Py<PyTransformerEncoder>` +
`Py<PyTransformerDecoder>`; `forward(src, tgt)` chains encoder→decoder; `parameters()`
returns `16×N_enc + 26×N_dec`; `num_enc_layers`/`num_dec_layers` getters; validation
`d_model % num_heads == 0` at constructor boundary. Test:
`test_transformer_seq2seq_composition` (structural composition identity, atol=1e-12).

### ~~G-006: RNN and positional-encoding Burn parity tests missing~~ **CLOSED**
**Location**: `crates/coeus-nn/tests/burn_live_parity.rs` — 0 tests for LSTM, GRU, RoPE, Sinusoidal
**Closed by**: MS-131 — Added 8 tests: `lstm_zero_input_zero_output_analytical` (analytical,
zero-bias+zero-input→zero; evidence tier: compile-time proof via docstring invariant),
`lstm_output_shape_contract`, `lstm_forward_seq_matches_module_forward`,
`gru_zero_input_zero_output_analytical`, `gru_output_shape_contract`,
`gru_forward_seq_matches_module_forward`, `sinusoidal_encoding_output_shape_matches_input`,
`sinusoidal_encoding_pos0_equals_analytical` (PE[0]=[0,1,0,1,...] analytically derived),
`rope_zero_input_zero_output`, `rope_output_shape_matches_input`. 292/292 Rust tests pass.

### ~~G-007: Transformer seq2seq structural parity tests missing~~ **CLOSED**
**Location**: `crates/coeus-nn/tests/burn_live_parity.rs` — no `forward_seq2seq` structural tests
**Closed by**: MS-136 — Added `transformer_seq2seq_self_consistent` (proves `forward_seq2seq`
== manual encoder+decoder chain; f32::EPSILON*4 tolerance) and
`transformer_module_forward_routes_to_seq2seq_self` (proves `Module::forward(x)` ==
`forward_seq2seq(x,x)`). Both use `Transformer<f32, SequentialBackend, 2, 1, 1>` with
dropout_p=0. Evidence tier: structural/deterministic. 294/294 Rust tests pass.

### ~~G-008: LSTM/GRU PyTorch parity tests missing~~ **CLOSED**
**Location**: `crates/coeus-python/tests/test_pytorch_parity.py` — 0 tests for LSTMCell/GRUCell step
**Closed by**: MS-136 — Added `test_lstm_cell_step_matches_pytorch`: copies w_ih/b_ih/w_hh/b_hh
from pycoeus LSTMCell(4,6) into torch.nn.LSTMCell.double(); verifies h_new and c_new at
atol=1e-10 after one step on zero-init hidden state. Gate order [i,f,g,o] matches between coeus
and PyTorch. Added `test_gru_cell_step_matches_pytorch`: same weight-injection approach for
GRUCell, verifying h_new; n=tanh(ih_n+r*hh_n) formula is consistent between implementations.
Evidence tier: differential/empirical.

### ~~G-009: JAX and MLX Python parity harnesses missing~~ **CLOSED**
**Location**: `crates/coeus-python/tests/` — no JAX or MLX parity harness existed
**Closed by**: MS-138 — Added `test_jax_parity.py` for f64
`Linear + ReLU + MSELoss` forward/backward parity against JAX, and
`test_mlx_parity.py` for MLX-native f32 forward-loss parity when MLX is
installed. Evidence tier: JAX differential/empirical; MLX optional-framework
collection behavior verified on this Windows environment (1 collected skip,
MLX not installed).

### ~~G-010: Optimizer step correctness unverified~~ **CLOSED**
**Location**: `crates/coeus-optim/src/{sgd,adam,adamw}.rs` — SGD, Adam, AdamW step implementations
had zero tests (no analytical derivation, no differential parity).
**Closed by**: MS-139 — Existing Rust analytical tests cover first-step SGD,
Adam, and AdamW formulas in `burn_live_parity.rs`; MS-139 added 3 Python
PyTorch differential tests:
`test_sgd_step_matches_pytorch`, `test_adam_step_matches_pytorch`,
`test_adamw_step_matches_pytorch` — each sets up mse_loss→backward→step and compares
against torch.optim at atol=1e-10. Evidence tier: analytical (Rust) + differential/empirical (Python).

### ~~G-011: Bilinear per-output indexing parity gap~~ **CLOSED**
**Location**: `crates/coeus-nn/tests/bilinear_parity.rs`,
`crates/coeus-python/tests/test_pytorch_parity.py` — Bilinear had all-ones analytical
coverage but lacked a per-output weight-indexing oracle and direct PyTorch
parity check.
**Closed by**: MS-140 — Added a Rust analytical identity/swap weight oracle
that verifies `[out, in1, in2]` indexing on Sequential and Moirai backends, and
added `test_bilinear_forward_matches_pytorch` against `torch.nn.Bilinear`.
Evidence tier: analytical (Rust) + differential/empirical (Python).

### ~~G-012: Python `Tensor.sum`/`.mean` reduction + InstanceNorm parity missing~~ **CLOSED**
**Location**: `crates/coeus-python/src/tensor/pyimpl.rs` — the Python `Tensor` exposed only
axis reductions (`sum_axis`/`mean_axis`), no full-reduction `sum()`/`mean()`, so the
idiomatic scalar-loss path `out.sum().backward()` was inexpressible and InstanceNorm
{1,2,3}d had no PyTorch parity coverage.
**Closed by**: MS-145 — Added `PyTensor::sum`/`PyTensor::mean` (GIL-released,
autograd-preserving, delegating to `coeus_autograd::{sum,mean}`); added
`test_instancenorm{1,2,3}d_matches_pytorch` (forward + dx + dγ + dβ at atol=1e-10)
and `test_{rmsprop,adagrad}_step_matches_pytorch`. Corrected the InstanceNorm oracle
to set `requires_grad=True` on the reference affine params. Removed stale
`tests/pycoeus*.pyd` artifacts that shadowed the installed extension during pytest.
Evidence tier: differential/empirical (PyTorch f64).

## Slop Pattern Library

- **Stale local `*.pyd` shadowing the installed extension**: pytest prepends the
  test directory to `sys.path`, so a leftover `crates/coeus-python/tests/pycoeus*.pyd`
  build artifact silently overrides the freshly `maturin develop`-installed module,
  pinning an out-of-date binary and producing spurious `AttributeError`s for
  newly-added bindings. Mitigation: keep built extensions out of `tests/`; the
  canonical module is the site-packages install. (Detected MS-145.)

## Residual Risks

| Risk | Evidence Tier | Status |
|------|--------------|--------|
| G-036 pool1d/adaptive pooling/unfold/fold family gaps — **CLOSED** via MS-206, MS-211, MS-212, MS-213. Pool1d (Max/Avg), adaptive pooling (Avg/Max 1d/2d), unfold/fold 1d/2d all implemented with autograd backward, Rust value-semantic tests, and Python bindings with PyTorch parity. | value-semantic + differential | **closed** |
| G-038 loss and distance surface — **CLOSED** via MS-219 (22/23) + MS-225 (CTCLoss). All 23 PyTorch loss/distance families now have Coeus parity. | analytical/value-semantic + differential | **closed** |
| G-040 recurrent parity — **CLOSED** via MS-206/MS-219. Vanilla RNN/RNNCell, GRU/GRUCell, LSTM/LSTMCell and Bidirectional wrapper with PyO3 bindings and parity tests. | source-surface + differential | **closed** |
| G-041 regularization/sparse/local-response — **CLOSED** via MS-208/MS-209. AlphaDropout, FeatureAlphaDropout, EmbeddingBag, GaussianNoise, LocalResponseNorm with PyO3 bindings. | source-surface + differential | **closed** |
| G-042 quantized/lazy parity policy — **CLOSED** via MS-212. Recorded as explicit non-goal for v0.x; natural extension point via typed `Scalar` + `BackendOps<T>` for quantized numerics. | design decision | **closed** |
| G-043 Burn/PyTorch NN benchmark matrix remains partial | source-surface + empirical | **open** |
| G-044 LocalResponseNorm — forward+backward fixed via band-matrix matmul; forward+dx parity with torch.nn.LocalResponseNorm verified. | differential | **closed** |
| G-045 forward-only modules sweep — **CLOSED**. AdaptiveAvgPool1d/2d, AdaptiveMaxPool1d/2d, Unfold1d/2d, Fold1d/2d all have full autograd backward implementations with Rust value-semantic and PyTorch parity tests. | value-semantic + differential | **closed** |
| G-001 stateless PyTransformerEncoderLayer binding | structural | **closed MS-127** |
| G-002 stateless PyTransformerEncoder binding | structural | **closed MS-128** |
| G-003 stateless PyTransformerDecoderLayer binding | structural | **closed MS-129** |
| G-004 PyTransformerDecoder missing | structural | **closed MS-129** |
| G-005 PyTransformer (full seq2seq) missing | structural | **closed MS-131** |
| G-006 RNN/PE Burn parity tests missing | structural | **closed MS-131** |
| G-007 Transformer seq2seq structural parity tests missing | structural | **closed MS-136** |
| G-008 LSTM/GRU PyTorch parity tests missing | differential | **closed MS-136** |
| G-009 JAX/MLX Python parity harnesses missing | differential/optional empirical | **closed MS-138** |
| G-010 Optimizer step correctness unverified | analytical + differential | **closed MS-139** |
| G-011 Bilinear per-output indexing parity gap | analytical + differential | **closed MS-140** |
| G-012 Python `Tensor.sum`/`.mean` reduction + InstanceNorm parity missing | differential | **closed MS-145** |
| G-013 duplicate einsum implementation under shape::util | compile/lint/docs + value-semantic tests | **closed MS-148** |
| G-014 GroupNorm Python differential parity missing | differential/empirical | **closed MS-149** |
| G-015 Scalar identity still depended on num-traits/libm | compile/lint/docs + value-semantic tests | **closed MS-150** |
| G-016 MaxPool2d/AvgPool2d differential parity missing | differential | **closed MS-151** |
| G-018 CrossEntropy/NLL loss differential parity missing | differential | **closed MS-153** |
| G-020 BCE/Huber loss differential parity missing | differential | **closed MS-156** |
| G-025 GlobalAvg/MaxPool2d differential parity missing | differential | **closed MS-166** |
| G-026 Elementwise activation differential parity missing | differential | **closed MS-167** |
| G-027 JAX harness lacked elementwise activation parity | differential | **closed MS-168** |
| G-029 JAX harness lacked softmax/log-softmax/cross-entropy parity | differential | **closed MS-173** |
| G-030 JAX harness lacked LayerNorm/RMSNorm parity | differential | **closed MS-174** |
| G-031 JAX harness lacked regression/binary loss parity | differential | **closed MS-175** |
| ConvTranspose backward WGPU/CUDA coverage | empirical GPU/CPU autograd differential | **closed MS-176** |
| mnemosyne-backend lib.rs docstring stale | documentation | **closed 87da068** |
| `test_hardswish_matches_pytorch` PyTorch differential parity | differential | **closed** — backward routing verified correct (evaluates on saved input, formulas match PyTorch); tests exist and run |
| `test_hardsigmoid_matches_pytorch` PyTorch differential parity | differential | **closed** — backward routing verified correct (evaluates on saved input, formulas match PyTorch); tests exist and run |
| `test_prelu_matches_pytorch` PyTorch differential parity | differential | **closed MS-217** — tightened shared `LeakyReluGrad` predicate from `x >= 0 ? 1 : α` to `x > 0 ? 1 : α` across `coeus-core` (float + int) and the `LeakyReluGradTag` fuse path; corrected the `act_extended/parameterized.rs` oracle + `nn_activation_tests.rs::test_leaky_relu_activation` expected gradient; added `test_prelu_matches_jax` and `leaky_relu_kink_at_zero_returns_slope`. Three-way Rust ↔ PyTorch ↔ JAX parity at the kink position. |
| `test_tcp_scatter_zero_numel_mismatched_target_numel_panics` slow | empirical | **closed MS-446** — socket-first `TcpMesh` teardown and bounded dedicated runtimes remove the 45 s wait; the test passes in 0.124 s, the 64-test `coeus-dist` lane in 0.385 s, and the 938-test workspace lane in 82.449 s with no slow tests |
| `coeus-cuda` clippy errors under `--all-features` | lint | **pre-existing peer crate dependency** — not addressed in MS-215 (out of coeus scope) |
