# Coeus Development Backlog

Open-work queue only. Delivered items are compacted away at the commit/PR that
closed them (`git log --grep='^Item:'` or the cited PR); this file never
restates their play-by-play. Priority is one of {correctness, architecture,
verification, tightening, feature}.

<a id="coeus-count-conv-spelling"></a>
## COEUS-COUNT-CONV-SPELLING — Count-typed spelling for `from_f64(n as f64)` sites

- Status: todo; priority: tightening; [patch].
- Finding: the COEUS-SCALAR-VOCAB slices (closed) retargeted count conversions to `<T as FloatElement>::from_f64(n as f64)`; the item's "Also" clause (migrate them to `from_count`/`try_from_count`) was not executed — `from_count` does not exist, and the exactness contract above 2^53 (exact vs rounded vs rejected) is undecided. Values are correct today (`n as f64` is exact below 2^53); only the spelling misstates intent.
- Scope: decide the contract, add the spelling (upstream if it belongs to eunomia), migrate the ~50 count sites.
- Acceptance: every count-typed conversion names the count spelling; values unchanged; gates green.
- Next step: decide exact-vs-rounded-vs-rejected for counts above 2^53.

<a id="coeus-rescue-queue"></a>
## COEUS-RESCUE-QUEUE — Complete or close the stranded rescue PRs

- Status: todo; priority: correctness; [patch].
- Outcome: each listed rescue PR is completed onto current `main` (ported, verified, merged) or closed once its diff resolves empty against `main`.
- Scope: head refs of the PRs below; no board or code change until a PR is claimed.
- PRs (draft parking records, unclaimed; head ref is the only durable copy):
  - `ryancinsight/coeus#469` (`rescue/audit-coeus-20260928`): CLOSED as landed — takeover port #512 merged as `39ad8054` (verified on main); 2 audit commits (module-root splits, crate-level allow removal, RNG unit fix) now on main.
  - `ryancinsight/coeus#530` (`rescue/coeus-green-sweep-20261012`, base `89cb08b0`): 304 files, +6689/-9415; unmergeable whole (3 add/add, 41 deletions vs landed work, 210 files also touched by main). Slices landed: ADRs 0079/0080 (#535), activation `math/` split (#536), dtype `traits/` split (#537). Drops with rationale: cuda/wgpu backend rewrite (superseded — main's concrete `CudaBackend` won), book hunks (stale API spelling), `coeus-leto` pooling dispatch as a single file (callers live in the pool-chain item below), `FusionProvider` transplant (typed for the `HephaestusBackend<P>` spelling vendors don't use; see fusion item). Residual sampling: COEUS-SWEEP-RESIDUAL-SAMPLING.
  - `ryancinsight/coeus#531` (`fix/coeus-s4b-sparse-wiring`): PRESERVATION DRAFT, do not close (peer #533 declares it intact). Sound part (allocation-free borrowed dispatch) recovered by peer-live #533. Forward spmv/spmm cutover proven unsound: Coeus accepts unsorted/duplicate CSR columns (stored-order accumulation), leto's `CsrView::from_slices` requires strictly increasing columns; leto kernels are serial vs `parallel_for` and use SIMD axpy (FMA) vs separate mul/add. Own takeover #534 closed with the proof.
- Acceptance: no open `rescue/` PR for this repository remains unaccounted for; a port's resolved diff against `main` is either integrated and verified by the committed gate or empty (landed-work proof), never closed on resemblance.
- Needs: none.
- Next step: `git diff $(git merge-base origin/main <head>) <head>` per PR, port onto a fresh branch from `main`, run the package gates.

<a id="coeus-fusion-unification"></a>
## COEUS-FUSION-UNIFICATION — One generic dispatch for vendor fusion

- Status: todo; priority: architecture; [arch] — ADR first.
- Finding: `crates/coeus-cuda/src/fusion.rs` (163 lines) and `crates/coeus-wgpu/src/fusion.rs` (160 lines) are ~65% identical (54+/57- diff): same `ExpressionAdapter`, same `provider_layout`, same empty-axis semantics, differing only in dialect/bundle/backend/error (`CudaC`/`CudaFusionOps`/`CudaBackend`/`CudaBackendError` vs WGPU counterparts).
- Non-solution: transplanting the sweep's `FusionProvider` (`rescue/coeus-green-sweep-20261012:crates/coeus-hephaestus/src/fusion/`) lands dead code — its dispatch is typed over `Tensor<T, HephaestusBackend<P>>`, a spelling main's vendors don't use (concrete `CudaBackend`/`WgpuBackend` with own error types). Reuse its adapter/layout logic, not its tensor spelling.
- Outcome: one generic dispatch (generic over the vendor backend, monomorphized per vendor) serves both vendors; the duplicated ~110 lines delete; vendor public API and `jit` tests unchanged and green.
- Acceptance: ADR selects the generic spelling; both `fusion.rs` files thin to bundle+entry wiring; no parallel fusion paths remain.
- Next step: draft the ADR.

<a id="coeus-pool-leto-chain"></a>
## COEUS-POOL-LETO-CHAIN — Delegate CPU pooling to leto (contract-gated)

- Status: todo; priority: architecture; [minor].
- Finding: main's `cpu_impl/pool/` (`avg.rs`, `max.rs`, `pool1d.rs`, `validation.rs`) is hand-rolled with no leto delegation. The sweep rewrites all three kernels onto `coeus_leto::pooling_forward_into`/`pooling_backward_into` (`crates/coeus-leto/src/dispatch/pooling.rs`, 259 lines) over `leto_ops::pooling_forward_into` — following main's established cpu_impl→coeus_leto dispatch pattern (8 call sites).
- Contract gate (prove before porting): padding/dilation/ceil-mode/`count_include_pad`/divisor-override parity between leto's pooling contract and current CPU behavior, including empty-output and backward identities; plus the `T: leto_ops::Scalar` bound ripple audit across callers (SemVer).
- Scope: forward + backward + validation + differential tests with exact values; vendor `pool.rs` files stay (separate backends, out of scope).
- Acceptance: bitwise-or-epsilon-justified parity vs current kernels on a fixed fixture grid; bound ripple compiles workspace-wide; package gates green.
- Next step: write the contract comparison (leto pin's pooling params vs current semantics).

<a id="coeus-sweep-residual-sampling"></a>
## COEUS-SWEEP-RESIDUAL-SAMPLING — Sample the unsampled #530 remainder

- Status: todo; priority: verification; [patch].
- Finding: of #530's 304 files, ~200 modified files remain unsampled after slices S1–S3 and the fusion/pool/einsum verdicts.
- Batch 1 verdicts (2026-10-15, `89cb08b0..16a73a81` vs main `f0d01021`): `dtype/float/cpu_unary.rs` — `ZERO`/`ONE` landed (64 sites, zero `zero()`/`one()` left); its per-site `FloatElement::from_f64` rewrite folds into COEUS-SCALAR-VOCAB (no slice — pre-empts the ADR). `cpu_impl/unfold_fold.rs` + `validation.rs` deletion — unique, filed as COEUS-UNFOLD-LETO-CHAIN. `tests/alloc_budget.rs` attribution machinery — unique, filed as COEUS-ALLOC-ATTRIBUTION. Gradcheck `GradcheckConfig`/tolerance-scale — landed independently (`gradcheck_with`, error/tests split; 431-line file under budget); the sweep's exact 5-way split dropped as stylistic.
- Next samples by diff size: vendor `pool.rs` ×2 (436/433), vendor `unfold_fold.rs` ×2 (227/224), `coeus-optim/src/least_squares/tests.rs` (211). `coeus-leto/src/dispatch/unfold_fold.rs` (210) folds into COEUS-UNFOLD-LETO-CHAIN.
- Batch 2 verdicts (2026-10-15): vendor `pool.rs` ×2 and `unfold_fold.rs` ×2 deletions — dropped as superseded (main's concrete vendor spelling won; sweep `pooling/`+`unfold_fold/` bridge logic verified line-identical to main modulo landed renames and fallible `make_unique`). `least_squares/tests.rs` — `ZERO` landed; its `+ FloatElement` bounds fold into COEUS-SCALAR-VOCAB (option-2 direction, no piecemeal port).
- Batch 3 verdicts (2026-10-15): `coeus-wgpu/src/lib.rs` rewrite, `backend/mod.rs` deletion, `coeus-cuda/src/backend/provider.rs` addition — all dropped as the superseded backend rewrite (re-export/doc-only or generic-spelling provider wiring; bundles, bounds, and `CudaScalar` f32/f64/F16/Bf16/i32 coverage verified identical to main's concrete impls).
- Batch 4 verdicts (2026-10-15): autograd `einsum.rs` typed variants — landed in substance (all four concepts present; `ShapeMismatch`/`Backend` cover the sweep's `DimMismatch`/`ForwardFailed`). `coeus-wgpu/src/backend/provider.rs`, `coeus-cuda/src/lib.rs` — dropped (same superseded class; re-export/wiring only). `shape_ops.rs` rejection tests — substance landed in the combined test; the 3 extra cases (empty subscript, unsupported arity, dot length) ported as #542 adapted to main variants.
- Next samples: vendor `reduction.rs` ×2 (147/146), `coeus-cuda/src/backend/mod.rs` (133), `backend/error.rs` ×2 (130/125).
- Batch 5 verdicts (2026-10-15): vendor `reduction.rs` ×2 deletions — dropped as superseded (main's concrete impls carry the same `ReductionProvider` bounds plus later `unsafe` hardening). `coeus-cuda/src/backend/mod.rs` deletion — converged: main independently restructured to `backend.rs`, all 5 public items (`ops`, `CudaScalar`, device fns, `CudaBackend`) verified present. `error.rs` ×2 deletions — dropped (main retains the concrete vendor error types).
- Next samples: vendor `attention.rs` (124), `coeus-python/src/ops/elementwise.rs` (121), `dtype/complex.rs` (120), vendor `optim.rs`/`optimizer.rs` (117/116).
- Batch 6 verdicts (2026-10-15): vendor `attention.rs` deletion — dropped as superseded (main retains the 119-line concrete impls). `elementwise.rs` + `tensor/dispatch.rs` — unique, ported re-cut as #545 (sweep predates fallibility; main already owns fallible `unary_dispatch`, so the slice exposes it crate-wide, adds `binary_dispatch`, converts 34 sites; `scalar_*` keep direct construction — the sweep's `from_var` conversion there would change grad-mode semantics). `dtype/complex.rs` vocab migration — folds into COEUS-SCALAR-VOCAB (option-2 direction).
- Next samples: vendor `optim.rs`/`optimizer.rs` (117/116), vendor `cross_entropy.rs` (114), `dtype/complex.rs` siblings by rank.
- Batch 7 verdicts (2026-10-15): vendor `optim.rs`/`optimizer.rs`/`cross_entropy.rs` deletions — dropped as superseded (main retains all three with concrete provider+backend+ops impls).
- Batch 8 verdicts (2026-10-15): wgpu `cross_entropy.rs`/`attention.rs` deletions and `device_outputs.rs` ×2 rewrites — dropped (superseded spelling; no new tests). `dtype/traits/float.rs` — converged with S3 (identical modulo a module-doc line). `cross_product/host_fallback.rs` restructure — dropped (lateral move of landed ADR-0077 logic; references `validate_cross_product_lengths`, absent from hephaestus 0.22). `interpolation.rs` f32→`T` genericization — unique, ported as #547 with an f64 second-instantiation test.
- Next samples: `conv.rs` ×2 (104/102), `elementwise.rs` ×2 (96/90), `backend_stub.rs` (95).
- Batch 9 verdicts (2026-10-15): vendor `conv.rs` ×2 deletions — dropped as superseded (main retains live const-generic forward/backward/transposed impls). `conv.rs` test — dropped (provider-rename only). ADRs 0079/0080 — landed (blob-identical: `95852799`/`288e6dad`). `dtype/traits/float.rs` — converged (same `Float` surface, doc-only drift). `gradcheck/mod.rs` — content landed inline in `gradcheck.rs`; exact 3-way split dropped as stylistic. `gradcheck/error.rs` — converged (2 doc lines).
- Batch 10 verdicts (2026-10-15): `gradcheck/precision.rs` — content landed inline (`narrow`/`machine_epsilon` with the same eunomia-dev-dep rationale); its `FloatElement` retarget folds into COEUS-SCALAR-VOCAB. `new_ops.rs` — unique, filed as COEUS-GRADCHECK-EMBEDDING (`embedding_case<T: GradcheckScalar>`) and COEUS-GRADCHECK-INTERP-DUAL (generic interpolation case; also the unmet acceptance tail of the interpolation item #547 otherwise closed). `losses.rs` — `cross_entropy_loss` import is #550's ground (in CI); its `FloatElement` retargets fold into COEUS-SCALAR-VOCAB. `float_ops.rs` — converged (25/25 members). `abs.rs` — converged (same `AbsOp` shape, main fallible). `dtype/int.rs` — int half ported as #549 (in CI). `random.rs`/`regularization.rs` — `from_f64`→`FloatElement` + `ZERO`/`ONE`, fold into COEUS-SCALAR-VOCAB.
- Batch 11 verdicts (2026-10-15): `host_fallback.rs` — RE-VERDICT unique, filed as COEUS-CROSS-HOST-SEAM: batch 8's drop rationale is disproven (`validate_cross_product_lengths` + `CrossProductOps` shipped in `crate-hephaestus-core-v0.22.0`, hence in the pinned 0.22.1). `device_outputs.rs`/`fusion.rs`/staggered-test/`unavailable.rs`/`state_updates.rs`/`pool1d.rs`/conv-transpose-test/pooling-test — dropped (sweep `HephaestusBackend<WgpuProvider>`/`CudaProvider` spelling; main has no `*Provider` structs, keeps concrete `WgpuBackend`/`CudaBackend`; unavailable coverage equivalent).
- Batch 12 verdicts (2026-10-15): vendor `elementwise.rs` ×2, `backend_stub.rs`, `staggered.rs`, `cross.rs`, `matmul.rs` ×2 whole-file deletions — dropped as superseded (main's copies live: `CudaScalar`+metadata stub, real `matmul`/`convolution` impls, all `mod`-referenced). `cpu_impl/pool.rs` — main already fallible (`?` propagates); sweep tail-call is stylistic. `elementary.rs` — `ZERO`/`ONE` landed. `neg.rs` — converged (main fallible). `ctc_loss_tests.rs`/`init_leto_diff.rs` — `FloatElement` retargets fold into COEUS-SCALAR-VOCAB. `levenberg_marquardt.rs` — `ZERO` landed; `+ FloatElement` on `is_finite`/`check_finite` is a no-op (`abs_val`/`from_f64` are declared on `Scalar` itself).
- Batch 13 verdicts (2026-10-15): `unfold_fold/validation.rs` `require_signed_coordinates` deletion — reference design for COEUS-UNFOLD-LETO-CHAIN (sweep `cpu_impl/unfold_fold.rs` delegates all four kernels to `coeus_leto::unfold_into`/`fold_into` with `map_leto_error`). `tensor/dispatch.rs` — converged with #545 (same GIL-release+`from_var` semantics; sweep free fns vs #545 `PyTensor` methods). Ops `linear_interpolation.rs` test + autograd `interpolation.rs` + ops `interpolation.rs` — covered by #547 (superset: sweep returbofish only, #547 adds the f64 instantiation). `backend/error.rs` einsum variants — reference design for COEUS-EINSUM-TYPED-ERRORS.
- Next samples by diff size: ranks 92+ (`ctc_loss_tests/layouts.rs`, vendor `random_init.rs` ×2, `op_tags/unary/activation.rs`, `constructors.rs`, metal/rocm `staggered.rs` — all ≤34 lines).
- Batch 14 verdicts (2026-10-15): vendor `random_init.rs` ×2 whole-file deletions — dropped as superseded (main's copies live: real `uniform_random`/`normal_random`). `layouts.rs`/`activation.rs`/`reduction.rs` (autograd) — `FloatElement` retargets, folded into the vocab slices (#555/#557). `constructors.rs` — unique, ported in #552 (3 f64 endpoint tests, adapted to fallible API). Metal/rocm `staggered.rs` — unique, ported in #552 as new test files; required the missing `StaggeredProvider` impls (`MetalStaggered3DOps`/`RocmStaggered3DOps`, both exported at pinned hephaestus `b03e9176`) + `tracing` dev-deps for the path-included oracle helpers (coeus-wgpu precedent).
- Next samples by diff size: ranks 100+ (wgpu `transfers_and_matmul.rs`, cuda `convolution_transpose.rs`, ctc `oracle.rs`, `public_reduction_leto_diff.rs`, `dtype/traits/mod.rs` — all ≤32 lines).
- Batch 15 verdicts (2026-10-15): wgpu `transfers_and_matmul.rs` + cuda `convolution_transpose.rs` — dropped (provider-rename only, superseded spelling, batch-11 rationale). `oracle.rs` — converged (main has `ZERO`/`ONE`, `+ FloatElement` bounds, and the `FloatElement::from_f64` count retarget via the closed vocab slices). `public_reduction_leto_diff.rs` — converged (main has `NumericElement::to_f64` + a single `FloatElement` bound; the sweep's duplicated bound is not reproduced). `dtype/traits/mod.rs` — converged (identical modulo main's newer both-Pod doc line and the `ReductionOp` relocation out of `op_tags`). `CHANGELOG.md` (rank 103, unlisted) — dropped (three [major] entries narrating the rejected provider-spelling rewrite; unshipped, would falsify release history).
- Next samples by diff size: ranks 106–115 skipping 107 — `gradcheck/config.rs`, `dtype/traits/int.rs`, `matmul_leto_diff.rs`, autograd `trig.rs`, `fusion/provider.rs`, `zero_grad.rs`, cuda parity `convolution.rs`, `dtype/float/half.rs`, wgpu `optimizer.rs` (rank 107 cuda `cross.rs` already dropped in batch 12; all ≤28 lines).
- Batch 16 verdicts (2026-10-15): `gradcheck/config.rs` — landed independently (batch-1 rationale; `GradcheckConfig` lives in `gradcheck.rs`). `dtype/traits/int.rs` — converged (identical modulo a 2-line module-doc trim). `matmul_leto_diff.rs` + autograd `trig.rs` — converged (`ZERO`/`ONE`, `+ FloatElement` bounds, `NumericElement::to_f64` all on main via the closed vocab slices). `fusion/provider.rs` — dropped (`FusionProvider` dead-code transplant, rescue-queue rationale; absent on main). `zero_grad.rs` — converged (same 5 ops/bounds modulo import hygiene). cuda parity `convolution.rs` + wgpu `optimizer.rs` — dropped (provider-rename only, batch-11 rationale; optimizer verified line-by-line: all 23 changed lines are renames).
- Batch 17 verdicts (2026-10-16): `dtype/float/native.rs` — converged (`Scalar` zero/one/to_f64/from_f64 deletion + `FloatElement` gelu retarget both on main via the closed vocab slices; zero `fn zero`/`fn from_f64` left, gelu at native.rs:187). wgpu `state_updates.rs`, cuda `optimizer_and_3d.rs`/`jit.rs`/`attention.rs`, cuda parity `pooling.rs`, wgpu `conv3d.rs` — dropped (provider-rename only, batch-11 rationale; `state_updates` even collapses two distinct backends into one duplicated line). `coeus-leto/src/lib.rs` — converged in substance (#559 pooling re-exports present; #566 unfold delegation live via rank-specialized `unfold1d/2d`/`fold1d/2d` consumed by `cpu_impl/unfold_fold.rs`; the sweep's bare-name spelling is not reproduced). `cuda_ops.rs` — dropped (lateral wiring move: the sweep deletes `cuda/mod.rs` and inlines the same 7 modules; main keeps `mod.rs` wired via `cuda_tests.rs`). `batched_matmul_leto_diff.rs` — converged (`ZERO` + `FloatElement` + `NumericElement::to_f64` on main). COEUS-ALLOC-ATTRIBUTION — dropped per reproduce-first (250-run CI census: zero CI/parity failures; `alloc_budget` 5/5 green locally; attribution already landed via #484/#515/#457; the +219 opt-in diagnostic + `backtrace` dep has no flake to chase).
- Batch 18 verdicts (2026-10-16): `reduction/linalg.rs` — converged (`T::ZERO` on main; doc renames are superseded spelling). cuda `README.md` — dropped (rewrite narration: generic-backend framing + feature-gate paragraph describing the sweep's restructured crate; main documents the concrete architecture). metal/rocm `backend/provider.rs` — converged via #552 (`StaggeredProvider`) + #561 (`CrossProductProvider`, main's `impl<T>`+where shape). wgpu/cuda `rotate_half.rs` deletions — dropped (superseded-spelling move: sweep relocates the impls to `*Provider` in `provider.rs`; main retains the concrete files). `ownership/optimizer.rs` — converged (9× `T::ZERO` on main). `adaptive_pool.rs` — converged (`from_count` + `ZERO` + `FloatElement` bounds all on main). autograd `activation/math/mod.rs` — converged (main holds the identical 7-module index). wgpu `conv.rs` — dropped (provider-rename only, batch-11 rationale).
- Next samples by diff size: ranks 136–145 — cuda/wgpu `backend_writes.rs`, wgpu `attention.rs`, cuda `random_init.rs`, `binary_simd_diff.rs`, `docs/book/tensor_backend.md`, cuda `conv_reduce.rs`, wgpu/cuda `attention_bench.rs`, nn `pool/adaptive.rs` (all ≤18 lines).
- Method: per file, extract distinctive added lines and grep current main for them (landed) vs genuine absence (unique); port uniques as slices, record drops with rationale. Never close on resemblance.
- Acceptance: every #530 file is landed, filed as its own item, or dropped with a stated reason; #530 closes with the accounting comment.
- Next step: sample rank 136+ files (≤18 lines each).

<a id="coeus-cross-vendor-bundles"></a>
## COEUS-CROSS-VENDOR-BUNDLES — Name the vendor on-device cross bundles

- Status: todo; priority: architecture; [minor].
- Finding: the COEUS-CROSS-HOST-SEAM port routes metal/rocm/cuda through host-fold math (`HostCrossProductOps` / `CudaBackend`'s own impl), but the pinned hephaestus `b03e9176` ships on-device `CrossProductOps` for CUDA, Metal, ROCm, and WGPU (`CudaCrossProductOps`/`MetalCrossProductOps`/`RocmCrossProductOps`/`WgpuCrossProductOps`; only the last is named coeus-side). Naming the real bundles is a behavior change (host→device dispatch, possible 1-ulp FMA differences, dialect-bound coverage narrower than `T: Scalar`), so it was split out of the values-preserving seam port.
- Scope: per-provider `CrossProductProvider` bundle swap + `CudaBackend::cross_storage` rewire to the wgpu fast-path shape + parity tests on HW; coverage audit per scalar type first (dialect bounds cover f32/u32/i32/f64/F16/Bf16-ish, not all `Scalar`).
- Acceptance: conforming layouts dispatch on-device on all four vendors; parity vs host fold within a derived bound; no `CrossOps` resolution regression for covered types; gates green.
- Next step: audit dialect-scalar coverage vs current `CrossOps` callers per vendor.

<a id="coeus-fallible-unary-execution"></a>
## COEUS-FALLIBLE-UNARY-EXECUTION — Propagate unary provider failures

- Status: todo; priority: correctness; [major] [arch].
- Outcome: provider failures reach callers without input-dependent panics for shared unary autograd execution.
- Scope: `UnaryAutogradOp::forward`/`backward` (`coeus-autograd/src/ops/activation/mod.rs`), `unary_op`, and their ~44 implementors' public wrappers (macro-emitted and hand-written) across `coeus-autograd`, `coeus-nn`, `coeus-python`. Confirmed by falsification, not estimate: an unpushed, since-reverted attempt at widening `coeus-ops`'s unary functions alone broke `coeus-autograd` at exactly these 44 call sites — `UnaryAutogradOp` is infallible by trait contract even though the lower `BackwardNode::backward` already returns `Result`.
- Acceptance: complete caller migration, typed failure/gradient tests, full native/device gates, SemVer classification with an updated ADR.
- Needs: none; the storage allocation/COW caller cutover landed under `Item: COEUS-FALLIBLE-STORAGE-CORE`.
- Non-goal: resurrect superseded dependency pins or storage implementations.
- Next step: add a forced backend-allocation failure fixture for unary forward/backward, then run the gradient and device gates against the migrated signatures in [ADR 0076](docs/adr/0076-fallible-unary-autograd-execution.md).

<a id="coeus-fallible-index-reduction"></a>
## COEUS-FALLIBLE-INDEX-REDUCTION — Return index-reduction failures

- Status: todo; priority: correctness; [major].
- Outcome: malformed argmax/argmin requests return typed errors through their complete caller chain.
- Scope: index-reduction backend trait methods, CPU/provider implementations, mathematical/Python callers; preserve index conventions and tie behavior.
- Evidence: the `COEUS-FALLIBLE-STORAGE-CORE` cutover changes argmax/argmin provider methods and public callers to typed `Result`; failure-injection coverage remains to close the item.
- Acceptance: malformed layouts/axes reject before output writes, retain clone values, propagate typed errors; valid ties/boundaries retain exact expected indices.
- Needs: none; storage allocation propagation landed under `Item: COEUS-FALLIBLE-STORAGE-CORE`.
- Next step: add malformed-layout and forced-provider-failure tests for argmax/argmin, then run the CPU and device parity gates.

<a id="coeus-device-output-ownership"></a>
## COEUS-DEVICE-OUTPUT-OWNERSHIP — Collect the device/hardware verification half

- Status: todo (core landed); priority: correctness; [major] [arch].
- Outcome: every mutable GPU kernel output preserves other tensor clones; CUDA/WGPU own generic provider storage with no temporary `from_arc` owner.
- Delivered: commit `9f897f2e` ("Give CUDA and WGPU generic provider storage") lands the core migration on `main` — vendor storage types and `from_arc` bridges are gone (`grep from_arc crates/coeus-{cuda,wgpu}/src` is empty), CPU scans propagate rejection, 267/267 CPU tests pass.
- Residual: device/hardware matrices still need collecting, and the commit's own message names the open half — "the upstream CUDA scalar compiler correction" — tracked upstream at `hephaestus/backlog.md#heph-cuda-dense-product-scalars`.
- Acceptance: real-device regressions over each supported operation/scalar matrix pass; SemVer confirms the declared major removals (already 222/223 checks pass per the landed commit).
- Next step: run the final workspace/device/SemVer gates once the upstream Hephaestus fix lands, then close.

<a id="atlas-coeus-safety-001"></a>
## ATLAS-COEUS-SAFETY-001 — Hephaestus provider device-acquisition panics

- Status: todo; priority: correctness; [major] [arch].
- Outcome: every provider-backed operation and copy-on-write detachment reports provider acquisition, allocation, and transfer failures instead of panicking inside a library boundary.
- Evidence: the `COEUS-FALLIBLE-STORAGE-CORE` cutover makes storage allocation and `StorageMut::make_unique` fallible. Coeus provider-operation dispatch still calls `P::device()` across elementwise, convolution, pooling, matmul, unfold/fold, cross-entropy, attention, and stateful-update paths; ROCm and Metal `device()` still panic when acquisition fails.
- Acceptance: value-semantic typed-error tests for unavailable devices and transfer failures; a production panic scan; provider feature gates on hosts with and without the required hardware; no CPU fallback or silent degradation.
- Next step: draft the provider-acquisition ADR, then replace remaining operation-dispatch `P::device()` calls with typed `P::try_device()` propagation and add forced acquisition-failure coverage.
- Note: deliberately outside the closed native-comparison-provider items — this is the separate public failure-boundary migration.

<a id="coeus-sibling-named-crates"></a>
## COEUS-SIBLING-NAMED-CRATES — Crates named after their dependencies

- Status: todo; priority: architecture; [major] [arch] — ADR first (Foundation phase).
- Finding: `coeus-leto` and `coeus-hephaestus` name stack members; a stack member's name must never appear in another member's crate/module/feature/item names (naming prohibition).
- Scope: the two sibling-named crates only; `coeus-cuda`/`coeus-rocm`/`coeus-metal`/`coeus-wgpu` name vendors, and their thinning is already decided by [ADR 0071](adr/0071-provider-owned-accelerator-backends.md) — out of scope here.
- Recommended target: name each crate for the concern it owns (the CPU array adaptation, the accelerator adaptation) so the manifest, not the crate name, records the provider. Rename is a mechanical transform over every call site in one change, no re-export bridge.
- Acceptance: an ADR with the recommended option is drafted first ([arch]); the rename then executes as one commit touching every call site, published-name migration note included ([major]).
- Next step: draft the ADR.

<a id="atlas-coeus-backend-045"></a>
## ATLAS-COEUS-BACKEND-045 — Unseal ComputeBackend and unify matmul (residual)

- Status: todo (matmul unification landed); priority: architecture; [minor] [arch].
- Outcome: matmul reaches every provider through one generic dispatch; no vendor crate carries a matmul kernel. Delivered: `ComputeBackend` unsealed, `coeus-hephaestus::matmul` added over `DenseProductOps`, `CudaBackend`/`WgpuBackend` migrated, `MatmulProvider<f32>` declared for Metal/ROCm.
- Residual 1 (external blocker): `HephaestusBackend<P>` still lacks `PoolOps`/`UnfoldFoldOps`, so it does not satisfy `BackendOps<f32>` for Metal/ROCm; re-open trigger is a `PoolOps<D,T>`/`UnfoldFoldOps<D,T>` seam landing upstream in `hephaestus-core`. ~2.8k pool + ~0.9k-1.0k unfold/fold lines per vendor crate stay forked until then.
- Residual 2: device-side equivalence of the provider matmul kernel against the deleted consumer kernel needs a GPU-capable CI run (no adapter on the development host).
- Residual 3: `cargo semver-checks` reports a pre-existing major break in `coeus-hephaestus` against the published 0.10.0 baseline (five associated types added since publish) — needs a version decision before the next release; unrelated to this item's own change.
- Links: [ADR 0066](adr/0066-computebackend-unseal-matmul.md).

<a id="coeus-bytemuck-residue"></a>
## COEUS-BYTEMUCK-RESIDUE — one concept, two markers

- Status: todo; priority: architecture; [patch].
- Finding: `coeus_core::Scalar` carries both `bytemuck::Pod` and Eunomia's `Pod` as supertraits. Eunomia is the stack's datatype law and owns this concept; bytemuck is retained only because `Scalar::has_zero_bit_pattern`'s default body calls `bytemuck::bytes_of`.
- Acceptance: move that body to `eunomia::layout::bytes_of` and drop the bytemuck supertrait, so one marker states the contract.
- Scope: the supertrait is public surface, so removing it is `[major]` for anyone implementing `Scalar` outside the workspace — check the blast radius first.
- Next step: enumerate `Scalar` implementors outside the workspace (if any), then land the swap.

<a id="coeus-ci-hook-scope-2026-09-24"></a>
## COEUS-CI-HOOK-SCOPE-2026-09-24 — Keep hook-only CI within the fast path

- Status: todo; integrator: coeus-hook-ci; priority: verification; [patch].
- Outcome: hook contract changes always report the required `Tests` status without installing Rust or compiling the workspace; Rust, manifest, lockfile, toolchain, test, and workflow changes retain native and doctest coverage.
- Scope: `.github/workflows/ci.yml`, `scripts/ci_scope.py`, its value-semantic tests, and this item; non-goal: changing `.githooks/**` or hook behavior.
- Acceptance: hook-only, mixed, and native selections are tested; the hook path runs existing hook-contract tests; native path keeps both Nextest and doctests.
- Evidence: PR #409 Tests job `107824618717` ended with runner exit 143 after unconditionally entering the 45-minute native job on a hook-only change.

<a id="coeus-device-test-build-cost"></a>
## COEUS-DEVICE-TEST-BUILD-COST — Attribute device test compilation

- Status: todo; priority: verification; [patch].
- Outcome: bound device-test compilation while preserving the full scalar/backend matrix.
- Scope: CPU test placement and monomorphization in existing integration harnesses; no assertion, scalar, workload or runtime-budget reduction.
- Evidence: ownership RED compilation took 13m06s while native execution took 7.965s; `wgpu_ops` rustc was the last compiler in the integrated attempt and also instantiates CPU-only scalar matrices.
- Acceptance: compiler timing/codegen evidence identifies the dominant work; an evidence-backed structural change preserves the enumerated test matrix and lowers its attributed compile cost, or records why the proposed partition does not help.
- Needs: [kernel output correction](#coeus-device-output-ownership); keep compiler flags and shared target policy fixed for comparison.

<a id="coeus-provider-resolution-2026-09-07"></a>
## COEUS-PROVIDER-RESOLUTION-2026-09-07 — Restore fresh provider resolution after upstream alignment

- Status: todo; priority: verification; [arch].
- Outcome: fresh lock generation and default/all-feature locked resolution agree, including against current upstream (not only offline idempotence).
- Evidence: a frozen tree regenerates the same offline lock twice and both activation checks pass — this proves closure/idempotence, not remote freshness against the current upstream state.
- Acceptance: standalone regeneration is idempotent and all configured gates pass against a freshly fetched upstream.
- Next step: re-run `cargo update` against current remotes and confirm the lock still resolves after any upstream version alignment since the last check.

<a id="coeus-autograd-l1-provider-001"></a>
## COEUS-AUTOGRAD-L1-PROVIDER-001 — Collect hosted evidence for the provider-owned L1 loss

- Status: todo (implementation + local verification complete); priority: verification; [patch] [arch].
- Outcome: L1 forward/backward compose provider `sub`/`abs`/`mean_axis`/`sign`/`mul`/`neg`; no host-resident `Vec<T>`. Already true in the merged implementation.
- Residual: hosted WGPU/CUDA/ROCm/Metal evidence remains pending before this child can close; the pre-1.0 `L1LossNode` representation change (`diffs: Vec<T>` → provider-resident `Tensor<T,B>`) needs the package's next SemVer review.
- Local evidence: focused Clippy, Nextest 3/3, doctests (autograd 16/16, nn 8/8 with 2 intentionally ignored), formatting, residue scan, diff hygiene pass.
- Next step: run the exact-head hosted backend matrix and record the SemVer classification.

<a id="coeus-autograd-lp-norm-provider-001"></a>
## COEUS-AUTOGRAD-LP-NORM-PROVIDER-001 — Collect doctest/SemVer evidence for provider-owned Lp norms

- Status: todo; priority: verification; [major] [arch].
- Outcome: `norm_p`/`norm_p_axis` perform provider-resident forward/backward (CPU via Leto `PowfOp`, WGPU/CUDA/ROCm/Metal via Hephaestus scalar-strided `PowOp`) — already merged (`4b915102`, `d775cd90`, provider blocker resolved in `cd36ee64`).
- Residual: collect workspace-doctest evidence against the delivered revision, and run SemVer against the intended baseline (existing published-0.9.0 comparison found three pre-existing cumulative-scan/fusion breaks unrelated to this item; classify Lp-norm-specific breaks separately).
- Evidence so far: `cd36ee64` records passing locked provider checks, provider Clippy, Hephaestus regressions, and backend-parity run `31052471989` for WGPU/CUDA/ROCm/Metal.
- Links: [ADR 0056](adr/0056-provider-owned-lp-norms.md).
- Next step: run the workspace doctest suite and `cargo-semver-checks` against the recorded baseline; record both in this item.

<a id="coeus-frobenius-norm-provider-001"></a>
## COEUS-FROBENIUS-NORM-PROVIDER-001 — Collect hosted provider contracts for provider-owned Frobenius norm

- Status: todo (local implementation complete); priority: verification; [patch] [arch].
- Outcome: `coeus_ops::frobenius_norm_batched` composes provider elementwise/reduction/sqrt instead of a rank-3-and-higher host fold; rank-2 scalar behavior, batched output shape, and strided-input materialization preserved.
- Local evidence: locked workspace all-targets check, warning-denied `coeus-ops` Clippy, full `coeus-ops` Nextest (209/209 including 7 Frobenius tests), 23 doctests pass.
- Residual: exact-head hosted WGPU/CUDA/ROCm/Metal provider contracts not yet run.
- Links: [ADR 0060](adr/0060-provider-owned-frobenius-norm.md).
- Next step: run the hosted backend matrix and record the result here.

<a id="coeus-benchmark-manifest-g043"></a>
## COEUS-BENCHMARK-MANIFEST-G043 — Close remaining partial rows in the every-family evidence manifest

- Status: todo; priority: verification; [patch].
- Outcome: every implemented NN family has a non-partial Criterion and Python-differential disposition in `crates/coeus-nn/benches/nn_bench/evidence.tsv`.
- Evidence (current, re-verified): 22 rows, 12 still `partial`, 0 `missing`, 21 `present` — matches the manifest's own consistency check (`crates/coeus-nn/tests/nn_ops/evidence_manifest.rs`).
- Acceptance: the manifest consistency check passes with zero partial rows, or each remaining partial row has a recorded reason (no comparable Burn/PyTorch API) rather than a silent gap.
- Non-goal: fabricating unsupported external-framework rows (several families have no Burn/PyTorch equivalent and are correctly Coeus-only).
- Next step: enumerate the 12 partial Criterion rows and close each with a real measurement or an explicit inapplicability record.

<a id="coeus-staggered-vendor-wiring"></a>
## COEUS-STAGGERED-VENDOR-WIRING — Bind the remaining vendor backends for StaggeredPairOps

- Status: todo; priority: tightening; [minor].
- Outcome: bind `StaggeredPairOps<f32>` for CUDA and Metal through the existing provider-owned kernels and shared staggered dispatch.
- CUDA: `coeus-cuda` already uses the Hephaestus CUDA provider with no Cutile/`cuda-bindings` dependency; add the missing provider/backend binding without duplicating dispatch.
- Metal: `HephaestusBackend<MetalProvider>` already uses the generic bridge; add `StaggeredProvider` for `MetalProvider` to obtain its existing generic implementation.
- Blocked (external): ROCm requires upstream staggered kernels in Hephaestus (`CudaStaggered3DOps`/`MetalStaggered3DOps` exist upstream; ROCm does not).
- Acceptance: exercise the shared gradient/divergence, axis, adjoint, and invalid-grid oracles on real CUDA and Apple Metal devices; preserve typed acquisition failures.

<a id="coeus-python-comparison-dunders"></a>
## COEUS-PYTHON-COMPARISON-DUNDERS — Add tensor comparison dunders with autograd-aware semantics

- Status: todo; priority: feature; [minor].
- Outcome: `PyTensor` implements `__lt__`/`__le__`/`__gt__`/`__ge__`/`__eq__`/`__ne__`, mirroring the existing scalar-arithmetic dunder pattern (`binop_dispatch` discriminator).
- Evidence: verified absent — no comparison dunder is defined anywhere under `crates/coeus-python/src`; current binding tests avoid the gap by extracting scalars via `.item()`.
- Acceptance: comparisons return a boolean or mask tensor matching `coeus_ops`'s existing `Eq`/`Ne`/`Lt`/`Gt`/`Le`/`Ge` element-wise semantics; binding test coverage replaces the `.item()` workaround.
- Non-goal: new autograd kernels — comparisons are non-differentiable, matching PyTorch.
- Next step: route through the existing `BinaryOp` comparison variants and the `binop_dispatch` discriminator pattern from `COEUS-MS-404`.

<a id="ms-445-python-release-wheels"></a>
## MS-445 — Python release wheels

- Status: todo; priority: feature; [patch]; owner: root.
- Outcome: a GitHub Release tagged `coeus-python-v<version>` builds locked Linux/Windows/macOS wheels for CPython 3.9-3.13, installs and imports each as `pycoeus`, attests and attaches artifacts, then publishes to the `coeus-python` PyPI project through OIDC.
- Delivered: the release workflow (`.github/workflows/python-release.yml`) and distribution contract are implemented; GitHub environment `pypi` accepts only `coeus-python-v*` tags; a locked CPython 3.13 wheel builds, installs, and imports as `pycoeus`.
- Residual: hosted CI on the exact release-automation head, and PyPI pending-trusted-publisher registration, remain open.
- Non-goal: Python binding behavior changes.

<a id="coeus-registry-package-1"></a>
## COEUS-REGISTRY-PACKAGE-1 — Publish reusable crates through Trusted Publishing

- Status: todo; priority: feature; [patch]; owner: root.
- Outcome: Coeus's publishable Rust crates (currently all `publish = false` or unset) release to crates.io in dependency order via OIDC trusted publishing.
- Delivered: Moirai/Mnemosyne/Themis imports bind to their published packages (no `rev =` pin — verified in `Cargo.toml`); the exact external lock graph is refreshed after the Apollo/Hephaestus/Moirai/Mnemosyne merges it depended on.
- Residual: exact-head provider CI on the release-preparation branch, then publish in dependency order through Trusted Publishing (engineering_gates: publish pipelines).
- Next step: run the hosted gate on the release branch, then execute the publish sequence.

<a id="coeus-nlls-004"></a>
## COEUS-NLLS-004 — Batched nonlinear least squares for diffusion fitting

- Status: todo; priority: feature; [minor]; owner: unclaimed.
- Outcome: `coeus-optim` gains a damped Gauss-Newton (or equivalent second-order) nonlinear least-squares optimizer batched over a leading problem axis, for per-voxel diffusion fitting (millions of independent small dense residual problems) where the shipped first-order optimizers (SGD/Adam/AdamW/RMSProp/Adagrad) are the wrong instrument by orders of magnitude.
- Scope: `crates/coeus-optim`; non-goal: log-linear DTI (already routes through `leto-ops`).
- Blocks: DKI, NODDI, IVIM, free-water, and every other nonlinear diffusion model.
- Acceptance: verified against an analytical oracle with a known minimum and against a published test-problem set; convergence criterion is a derived relative-residual bound, never a fixed iteration count.
- Next step: draft the batched Jacobian/normal-equations layout (SoA over the batch axis) and its `Scalar`-generic contract before implementation.
- Links: meta ATLAS-COEUS-NLLS-004.

<a id="coeus-linspace-fake-generic"></a>
## COEUS-LINSPACE-FAKE-GENERIC — Native-T linspace/logspace/geomspace

- Status: todo; priority: correctness; [patch]; owner: unclaimed.
- Outcome: `linspace`/`logspace`/`geomspace` compute in `T: Scalar` natively; no `to_f64`/`from_f64` widen-compute-narrow.
- Scope: `coeus-ops/src/constructors.rs:20-118`, `coeus-tensor/src/constructors.rs:84-155` (duplicated implementation — consolidate to one).
- Acceptance: `cargo asm`/source review shows no `f64` intermediate for `f32` instantiation; value-semantic test against analytical endpoints/ratios per dtype.
- Next step: derive the native-T step/ratio formula, implement once, delete the duplicate.

<a id="coeus-expect-swallowed-results"></a>
## COEUS-EXPECT-SWALLOWED-RESULTS — Propagate typed errors instead of `.expect()`

- Status: todo; priority: correctness; [patch]; owner: unclaimed.
- Outcome: public elementwise ops and device/alloc paths return typed `Result` instead of panicking via `.expect()` on backend/allocator failure.
- Scope: `coeus-ops/src/unary/math.rs` (69 ops via `.expect("<opname>")`), `coeus-autograd/src/ops/activation/{ext,gelu,math}.rs`, `coeus-cuda/src/backend/mod.rs:21,101-120`, `coeus-hephaestus/src/storage.rs:56,63`, `coeus-hephaestus/src/reduction.rs:279,285`, `coeus-tensor/src/tensor.rs:438,452`, `coeus-core/src/storage/cpu.rs:131`.
- Acceptance: no bare `.expect()` on backend-Result in listed files; failure surfaces as a typed error to the caller; existing behavior tests still pass.
- Next step: start with `coeus-ops/src/unary/math.rs` (highest count), thread the `Result` through its callers.

<a id="coeus-gradcheck-generic-scalar"></a>
## COEUS-GRADCHECK-GENERIC-SCALAR — Generic finite-difference gradcheck over f32/f64

- Status: todo; priority: verification; [patch]; owner: unclaimed.
- Outcome: the finite-difference gradient-check harness is one generic function/macro instantiated per scalar type (f32 and f64), each with its own analytically derived tolerance, not an f64-only harness.
- Scope: coeus-autograd's gradcheck test infrastructure.
- Acceptance: existing f64 gradcheck coverage still passes; f32 instantiation runs with an f32-appropriate tolerance derived from step size and precision.
- Next step: parameterize the harness's step size and tolerance by `T::EPSILON`, add the f32 instantiation.

<a id="coeus-gradcheck-missing-ops"></a>
## COEUS-GRADCHECK-MISSING-OPS — Fill finite-difference gradient-check gaps

- Status: todo; priority: verification; [minor]; owner: unclaimed.
- Outcome: every differentiable op has a finite-difference gradient-check test (depends on COEUS-GRADCHECK-GENERIC-SCALAR landing first).
- Scope: add/sub/div/remainder/maximum/minimum/neg, all pooling ops, conv1d/conv3d/conv_transpose1d/2d/3d, fold/unfold, cross_entropy, ctc, dropout (mask-fixed), embedding, sparse_matmul/sparse_matmul_coo, transpose_2d, index_put, rotate_half, linear_interpolation.
- Needs: COEUS-GRADCHECK-GENERIC-SCALAR.
- Acceptance: each listed op has a passing FD gradcheck against its analytical Jacobian within the derived tolerance.
- Next step: work the list in dependency-free batches (e.g. arithmetic ops first, since they share the harness shape).

<a id="coeus-existence-only-nn-tests"></a>
## COEUS-EXISTENCE-ONLY-NN-TESTS — Value-semantic nn test assertions

- Status: todo; priority: verification; [patch]; owner: unclaimed.
- Outcome: nn tests assert value-semantic correctness against an analytical reference, not existence/shape only.
- Scope: `coeus-nn` batch_norm tests (`batch_norm.rs:5-15,35-37,143-145`), `nn_normalization_tests.rs` (group/instance norm), attention `tests.rs:52-75,181-200`.
- Acceptance: each listed test asserts a computed numeric value against a derived reference, not only `Ok`/shape.
- Next step: derive the analytical reference (e.g. hand-computed normalization on a small fixture) for each listed test group.

<a id="coeus-dot-cross-host-copy"></a>
## COEUS-DOT-CROSS-HOST-COPY — Device-resident dot/cross on accelerator backends

- Status: blocked; priority: tightening; [patch]; owner: unclaimed.
- Outcome: `dot`/`cross` on accelerator backends stay device-resident instead of copying both operands to host.
- Scope: `coeus-ops/src/reduction/linalg.rs`.
- Blocker: hephaestus is adding device dot/cross; bind to it once it lands (re-open trigger: hephaestus device dot/cross merges).
- Acceptance: no host round-trip for accelerator dot/cross; differential test against the CPU reference.

<a id="coeus-python-elementwise-dedup"></a>
## COEUS-PYTHON-ELEMENTWISE-DEDUP — Consolidate duplicated Python elementwise bindings

- Status: todo; priority: tightening; [patch]; owner: unclaimed.
- Outcome: one generic entry point for elementwise op bindings instead of 28 duplicated wrappers.
- Scope: `coeus-python/src/ops/elementwise.rs` vs `pytensor.rs`.
- Acceptance: one shared dispatch path; binding-level pytest suite still passes.
- Next step: extract the common per-op call pattern into a macro or generic helper (pytensor.rs is already the split target: 933 lines, see COEUS-OVERSIZED-FILES).

<a id="coeus-oversized-files"></a>
## COEUS-OVERSIZED-FILES — Split files past the 500-line target

- Status: todo; priority: tightening; [patch]; owner: unclaimed.
- Outcome: the 14 files currently over 500 lines (worst: `pytensor.rs` 933) split into leaf modules by operation family.
- Scope: repo-wide scan (`coeus-python/src/tensor/pyimpl/pytensor.rs` first).
- Acceptance: each split file's module retains domain cohesion; no file regresses over 500 lines.
- Next step: split `pytensor.rs` first since COEUS-PYTHON-ELEMENTWISE-DEDUP touches it anyway.

<a id="coeus-cuda-safety-comment"></a>
## COEUS-CUDA-SAFETY-COMMENT — Correct misleading SAFETY comment on CudaBackend::parallel_for

- Status: todo; priority: correctness; [patch]; owner: unclaimed.
- Outcome: the `# Safety`/`// SAFETY:` comment on `CudaBackend::parallel_for` accurately states the invariants it relies on.
- Scope: coeus-cuda backend `parallel_for`.
- Acceptance: comment reviewed against the actual unsafe preconditions; miri/sanitizer coverage unaffected.
- Next step: re-derive the actual safety obligation from the call site and rewrite the comment.

<a id="coeus-hephaestus-padops-adoption"></a>
## COEUS-HEPHAESTUS-PADOPS-ADOPTION — Adopt hephaestus's PadOps<D, T> seam

- Status: blocked; priority: architecture; [patch]; owner: unclaimed.
- Outcome: `coeus-hephaestus` consumes hephaestus-core's `PadOps<D, T>` seam once merged, instead of any local padding duplication.
- Blocker: hephaestus PadOps seam not yet merged upstream (re-open trigger: hephaestus-core PadOps<D,T> lands).
- Scope: `coeus-hephaestus` padding call sites.
- Acceptance: coeus-hephaestus padding routes through the upstream seam; no duplicated padding logic remains locally.

<a id="coeus-scatter-add-alloc-ci-only"></a>
## COEUS-SCATTER-ADD-ALLOC-CI-ONLY — scatter_add allocation-count regression on hosted CI only

- Status: todo; priority: correctness; [patch]; owner: unclaimed.
- Outcome: `scatter_add_allocation_count_is_independent_of_index_size` (coeus-ops/tests/alloc_budget.rs) passes on hosted CI, not only locally.
- Evidence: hosted `Tests` job on PR #425 (run 36285453725, job 108526331877) fails with small=6, large=10 allocations for shapes [4,8,4]/[16,32,16]; reproduced locally on Windows with the identical revision and the same nextest invocation and it PASSES (1 passed; 0 failed). This is a landed, pre-existing defect on `main` (predates every PR in this session's reconciliation) — `main` at the failing revision was `db144251`, before any of PRs 421-428 merged.
- Investigated and ruled out: `scatter_add`'s own flat-index loop is allocation-free by design (no per-element or per-slice buffer, per its own code comment); `Tensor::alloc_on`/`CpuStorage::new` are single-call allocations independent of `numel`; `to_contiguous()` short-circuits to `self.clone()` for already-contiguous inputs (both test tensors are freshly built and contiguous), so it never reaches `coeus_leto::contiguous_values`. Falsified the "parallel-dispatch thread count crosses a core-count threshold" hypothesis: pinned the local repro to a 2-core `ProcessorAffinity` mask (matching the hosted runner's core count) via a `Start-Process`-launched `cargo test` — still passes locally (1 passed; 0 failed). `leto-ops`'s documented parallel thresholds (`PARALLEL_MIN_ELEMENTS` 65536, `PARALLEL_MIN_REDUCTION_OUTPUTS` 32768, `PARALLEL_MIN_MATMUL_MACS` 262144) are all far above this test's tensor sizes (max 8192 elements), ruling out its elementwise/reduction/matmul parallel paths regardless of core count.
- Critical finding: this is FLAKY on hosted CI, not a deterministic regression. `main`'s own push-triggered `CI` run at the identical commit (`db144251`, run 36286664346, job 108528635313, 2026-09-27T02:03) shows this exact test **PASS**, while PR #425's `pull_request`-triggered run at essentially the same tree (run 36285453725, ~2026-09-27T01:46) shows it **FAIL** with small=6/large=10. Six concurrent copies of the compiled test binary, each pinned to a 2-core `ProcessorAffinity` mask, all pass locally (`multi_{0..5}.log`) — a scheduler-race reproduction attempt per the "two-core pinning reproduces CI scheduler flakes" pattern also did not reproduce. Per policy, a flaky test is root-caused, never silently retried: the counting allocator (`alloc_budget.rs`'s `#[global_allocator]`) is process-wide, so any concurrent or lazily-initialized background allocation (a `OnceLock`/thread-pool first-touch in a dependency, unrelated to `scatter_add` itself) racing with the small/large calls would produce exactly this nondeterministic +4 without any code-path difference in `scatter_add`.
- Next step: add allocation-site attribution (capture a backtrace or a monotonic sequence id per counted allocation) to `alloc_budget.rs`'s global allocator so a failing run identifies *which* allocation is extra, rather than only the count; run that instrumented binary repeatedly on the hosted runner (matching its exact core count and load) until it flakes again.
- Blocks: PR #425 and any other PR whose `Tests` job runs the full workspace suite on hosted CI.

<a id="coeus-gradcheck-generic-instantiation"></a>
## COEUS-GRADCHECK-GENERIC-INSTANTIATION — Generic-instantiation gradcheck coverage

- Status: todo; priority: verification; [patch]; owner: unclaimed.
- Outcome: every finite-difference check in `crates/coeus-autograd/tests/autograd/gradcheck/` runs at both `f64` (sensitivity oracle) and `f32` (instantiation coverage, per `standards`: Generic Instantiation Coverage), plus FD coverage for the ops that had none.
- Delivered: `mod.rs`'s shared fixtures (`Sampler`, `tensor`, `weighting`, `weighted`) are generic over `GradcheckScalar` (`Float + leto_ops::Scalar + coeus_leto::RealScalar` plus the `CpuAddressableStorage` bound gradcheck itself needs); the module doc states the two-role f64/f32 rationale. New FD coverage, each a generic fn instantiated at both types from a single `#[test]`: `add`/`sub`/`div`/`remainder`/`maximum`/`minimum`/`neg` (`arithmetic.rs`); `max_pool1d/2d/3d`/`avg_pool1d/2d/3d` (`pooling.rs`, via `coeus-nn` `Module` layers, new dev-dependency); `conv1d`/`conv3d`/`conv_transpose1d/2d/3d`/`unfold1d/2d`/`fold1d/2d` (`conv.rs`, via `coeus-nn` `Module` layers). Converted to dual-instantiation: `normalization.rs`, `core_ops.rs`, `attention.rs`.
- Remaining: (1) convert the 4 remaining existing check files (`activation`, `losses`, `reduction`, `shape`) to dual-instantiation; (2) FD coverage still missing: `ctc`, `dropout` (fixed mask), `sparse_matmul(_coo)`, `transpose_2d`, `index_put`, `rotate_half`, `linear_interpolation` (`cross_entropy` and `embedding` already have coverage in `losses.rs`).
- Acceptance: every listed op has a generic FD check instantiated at both types; all 11 existing files are dual-instantiated; a check that fails at `f32` under gradcheck's own derived `ε^(2/3)` bound is root-caused, never given a widened tolerance.
- Next step: convert `activation.rs` next (largest remaining, ~50 tests) then `losses`/`reduction`/`shape`, each its own PR; then the remaining missing-coverage ops as dependency-ordered per-family PRs.

