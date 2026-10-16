# ADR 0082: Unified generic fusion dispatch for CUDA and WGPU

Status: Accepted  \
Date: 2026-10-16  \
Change class: [arch]  \
Board item: [COEUS-FUSION-UNIFICATION](../../backlog.md#coeus-fusion-unification)
Revision 2026-10-16: implemented and accepted as decided. Both vendor
`fusion.rs` files thin to `FusionBackend` implementations plus entry
wrappers (172/173 lines to ~60 each); the CUDA and WGPU suites pass
unchanged on hardware through the shared generic, and the converged
error variants carry exact-error unit coverage.

## Context

`crates/coeus-cuda/src/fusion.rs` (172 lines) and
`crates/coeus-wgpu/src/fusion.rs` (173 lines) implement the same fused
expression dispatch twice: the same `ExpressionAdapter` string rewriting
(`val_` to `input_`), the same `provider_layout` rank/stride adaptation,
the same `input_views` projection, and the same acquire-validate-dispatch
sequence in `dispatch_fused`/`dispatch_fused_reduce`. They differ only in
six vendor bindings: dialect (`CudaC`/`Wgsl`), scalar bound
(`CudaScalar`/`WgpuScalar`), backend (`CudaBackend`/`WgpuBackend`),
fusion operations (`CudaFusionOps`/`WgpuFusionOps`), device acquisition,
and error constructors.

The duplication has already drifted semantically. `provider_layout`
reports a rank mismatch as `BackendError::Storage` with a formatted
reason on CUDA but as the precise `BackendError::LayoutRankMismatch` on
WGPU, and an out-of-range stride as `Storage` on CUDA but as
`BackendError::Overflow` on WGPU. The operation label for validation
failures is `"fused elementwise"` on CUDA but `"fused expression"` on
WGPU (which labels its own dispatch `"fused elementwise"`). No test
pins the CUDA `Storage`-reason spellings; the WGPU `Storage` spelling
for empty input expressions is pinned by
`coeus-wgpu/tests/wgpu_ops/fusion.rs`, and the CUDA `"fused elementwise"`
label is pinned by the disabled-provider stub tests.

Constraints: both vendors use concrete backends (`CudaBackend`,
`WgpuBackend`) with their own error types — never the generic
`HephaestusBackend<P>` spelling — so the generic dispatch must be
parameterized over the vendor backend and monomorphized per vendor.
Vendor public API (`evaluate_fused`, `evaluate_fused_reduce`) and the
`jit` suites stay green and behavior-identical except where this ADR
explicitly converges error precision.

## Decision

1. Add a `FusionBackend<T>` trait in `coeus-hephaestus` (the vendor-neutral
   home both vendors already depend on), implemented once per vendor
   backend in its own crate:
   ```text
   trait FusionBackend<T>: HephaestusSeamBackend
       type FusionOps: FusedElementwiseOps<Self::Device, T>
           + FusedReductionOps<Self::Device, T>
           + Default;
       fn validation_error(error: BackendError) -> Self::Error;
   ```
   Device acquisition, buffer borrowing, and dispatch-error mapping reuse
   `HephaestusSeamBackend` (whose `try_device` landed under ADR 0079);
   the trait adds only the fusion operation bundle and the typed
   validation mapping. Per-vendor scalar requirements
   (`CudaFusionScalar`/`WgpuFusionScalar`) are enforced by each
   `FusionOps` implementation's own bounds at monomorphization, so the
   generic signatures carry no vendor scalar trait.
2. Move `ExpressionAdapter`, `provider_layout`, `input_views`,
   `dispatch_fused`, and `dispatch_fused_reduce` into a shared
   `coeus-hephaestus/src/fusion.rs`, generic over `B: FusionBackend<T>`,
   `T`, and `E: ExprNode<T, B>`. The adapter implements
   `FusedExpression<B::FusionOps::Dialect>`; its body is
   dialect-independent string rewriting, so one implementation serves
   both dialects. Both vendor `fusion.rs` files thin to entry wiring
   that instantiates the shared generic at the concrete backend; the
   `fuse_api.rs`/`api.rs` call sites keep their shapes.
3. Converge error precision on the WGPU spellings: rank mismatch reports
   `BackendError::LayoutRankMismatch` and stride overflow reports
   `BackendError::Overflow` on both vendors. The empty-expression case
   keeps `BackendError::Storage` with reason
   `"expression contains no tensor inputs"` on both vendors. This changes
   CUDA's validation-error variants for two conditions from generic to
   precise; the information content is unchanged and no test pins the old
   spellings.
4. Converge operation labels on `"fused elementwise"` and
   `"fused reduction"` everywhere, matching the dispatch labels both
   vendors already emit. This renames WGPU's validation label from
   `"fused expression"`.
5. Shared `provider_layout` returns `BackendError` and each dispatch maps
   it once through `FusionBackend::validation_error`, so no vendor
   stringifies a typed failure.

## Rejected alternatives

- Transplanting the sweep's `FusionProvider`
  (`rescue/coeus-green-sweep-20261012:crates/coeus-hephaestus/src/fusion/`):
  its dispatch is typed over `Tensor<T, HephaestusBackend<P>>`, a spelling
  neither vendor uses; the transplant would land dead code. Its
  adapter/layout logic is subsumed by the shared module here.
- Macro-generated per-vendor copies: preserves two maintained copies and
  the drift that motivated this item.
- Dynamic dispatch over a boxed fusion backend: introduces a vtable into a
  per-dispatch path and contradicts the repository's generics and
  monomorphization directive.

## Consequences

CUDA validation failures for rank mismatch and stride overflow change
variants (generic `Storage` to precise typed variants); the SemVer
review treats this as a minor behavioral precision fix. Metal and ROCm
are unaffected (no fusion dispatch). The two vendor `fusion.rs` files
delete ~110 duplicated lines net of the new trait implementations.

## Verification

The CUDA and WGPU `jit` suites pass on hardware CI; the
disabled-provider unavailable tests pass; new exact-error tests pin the
converged `LayoutRankMismatch`/`Overflow` variants and the
`"fused elementwise"`/`"fused reduction"` labels on both vendors. A
final search proves no second fusion dispatch path remains.
