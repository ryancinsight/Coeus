# Coeus Development Backlog

Open-work queue only. Delivered items are compacted away at the commit/PR that
closed them (`git log --grep='^Item:'` or the cited PR); this file never
restates their play-by-play. Priority is one of {correctness, architecture,
verification, tightening, feature}.

<a id="coeus-scalar-vocab"></a>
## COEUS-SCALAR-VOCAB — Delete `Scalar`'s redeclared element vocabulary

- Status: todo; priority: architecture; [major]; basis: 70e39493.
- Outcome: `coeus_core::Scalar` declares only its slice-kernel surface; `zero`, `one`, `to_f64` and `from_f64` (`crates/coeus-core/src/dtype/traits.rs:346-364`) delete, call sites moving to `NumericElement::ZERO`/`ONE`/`to_f64` and `FloatElement::from_f64` (atlas:0005 decision 1, recorded closed there on 2026-07-04 but never landed).
- Also: the `from_f64(n as f64)` count conversions COEUS-COUNT-CONV-001 left move to `from_count`/`try_from_count`.
- Acceptance: the four methods are absent; values unchanged (the moves are identities); coeus gate green.

<a id="coeus-rescue-queue"></a>
## COEUS-RESCUE-QUEUE — Complete or close the stranded rescue PRs

- Status: todo; priority: correctness; [patch].
- Outcome: each listed rescue PR is completed onto current `main` (ported, verified, merged) or closed once its diff resolves empty against `main`.
- Scope: head refs of the PRs below; no board or code change until a PR is claimed.
- PRs (draft parking records, unclaimed; head ref is the only durable copy):
  - `ryancinsight/coeus#469` (`rescue/audit-coeus-20260928`): 2 audit commits (module-root splits, crate-level allow removal, RNG unit fix); 68 files, +2309/-2430, code across `coeus-ops`, `coeus-nn`, `coeus-autograd`, `coeus-python` and 7 other crates.
  - `ryancinsight/coeus#464` (`rescue/coeus-fallible-storage-core-recovery`): 6 commits of the unverified fallible-storage migration preserved from PR #461 (storage seam, COW failures, caller propagation); 584 files, +19839/-11005, code across 15 crates plus one ADR. Overlaps [COEUS-FALLIBLE-STORAGE-CORE](#coeus-fallible-storage-core) and its dependent leaves; port per those leaves' scopes.
- Acceptance: no open `rescue/` PR for this repository remains unaccounted for; a port's resolved diff against `main` is either integrated and verified by the committed gate or empty (landed-work proof), never closed on resemblance.
- Needs: none for #469; #464 is claimed through the storage leaves above.
- Next step: `git diff $(git merge-base origin/main <head>) <head>` per PR, port onto a fresh branch from `main`, run the package gates.

<a id="coeus-fallible-unary-execution"></a>
## COEUS-FALLIBLE-UNARY-EXECUTION — Propagate unary provider failures

- Status: todo; priority: correctness; [major] [arch].
- Outcome: provider failures reach callers without input-dependent panics for shared unary autograd execution.
- Scope: `UnaryAutogradOp::forward`/`backward` (`coeus-autograd/src/ops/activation/mod.rs`), `unary_op`, and their ~44 implementors' public wrappers (macro-emitted and hand-written) across `coeus-autograd`, `coeus-nn`, `coeus-python`. Confirmed by falsification, not estimate: an unpushed, since-reverted attempt at widening `coeus-ops`'s unary functions alone broke `coeus-autograd` at exactly these 44 call sites — `UnaryAutogradOp` is infallible by trait contract even though the lower `BackwardNode::backward` already returns `Result`.
- Acceptance: complete caller migration, typed failure/gradient tests, full native/device gates, SemVer classification with an updated ADR.
- Needs: [fallible tensor storage](#coeus-fallible-tensor-storage) — its allocation/COW cutover must include the unary, derivative arithmetic and NN/Python callers it makes fallible; landing unary first would touch the same ~44 signatures twice.
- Non-goal: resurrect superseded dependency pins or storage implementations.
- Next step: [ADR 0076](adr/0076-fallible-unary-autograd-execution.md) (Proposed) drafts the recommended migration order and rejected alternatives; implement per it once `COEUS-FALLIBLE-TENSOR-STORAGE` lands.

<a id="coeus-fallible-index-reduction"></a>
## COEUS-FALLIBLE-INDEX-REDUCTION — Return index-reduction failures

- Status: todo; priority: correctness; [major].
- Outcome: malformed argmax/argmin requests return typed errors through their complete caller chain.
- Scope: index-reduction backend trait methods, CPU/provider implementations, mathematical/Python callers; preserve index conventions and tie behavior.
- Evidence: `coeus-ops/src/backend_ops/cpu_impl/reduction.rs` expects fallible Leto argmax/argmin results; current trait methods return no error (source finding, not an executed reproduction).
- Acceptance: malformed layouts/axes reject before output writes, retain clone values, propagate typed errors; valid ties/boundaries retain exact expected indices.
- Needs: coordinate with [fallible storage](#coeus-fallible-tensor-storage) for allocation propagation; reserve the governing ADR and migration before implementation.
- Next step: draft the ADR, then implement after `coeus-fallible-tensor-storage` lands.

<a id="coeus-fallible-tensor-storage"></a>
## COEUS-FALLIBLE-TENSOR-STORAGE — Propagate tensor storage failures

- Status: todo; priority: correctness; [major] [arch].
- Outcome: allocation, zero-fill and copy-on-write failures reach typed callers.
- Scope: `ComputeBackend` storage methods, Tensor constructors/materialization (`alloc_on`/`zeros_on` in `crates/coeus-tensor/src/tensor.rs` still return `Self`, not `Result`), provider implementations and their complete caller closure.
- Acceptance: no provider-error expects on the migrated paths; real malformed-size/layout/device error tests; no partial writes claimed as whole-graph rollback.
- Needs: [CPU ownership correction](#coeus-cpu-storage-ownership) (done); driver for [unary](#coeus-fallible-unary-execution) and [index-reduction](#coeus-fallible-index-reduction) consumers.
- Next step: reserve an ADR; constructor search found 2,781 textual candidates in 406 files at last audit — scope the cutover before implementation.

The migration is decomposed into these dependency ordered leaves:

<a id="coeus-fallible-storage-core"></a>
## COEUS-FALLIBLE-STORAGE-CORE — Make backend storage operations fallible

- Status: todo; priority: correctness; [major] [arch].
- Outcome: `ComputeBackend` allocation, fill, and transfer methods return typed errors and every provider implements the contract.
- Scope: `crates/coeus-core/src/backend`, `crates/coeus-core/src/storage`, and provider backend implementations.
- Acceptance: malformed sizes and provider failures return typed errors; valid CPU/provider operations retain values; no CPU fallback or partial output.
- Needs: [CPU ownership correction](#coeus-cpu-storage-ownership) (done).
- Next step: change the existing trait methods in place and run the provider compile closure.
- Links: [ADR 0078](docs/adr/0078-fallible-tensor-storage.md), parent [COEUS-FALLIBLE-TENSOR-STORAGE](#coeus-fallible-tensor-storage).

<a id="coeus-fallible-storage-tensor"></a>
## COEUS-FALLIBLE-STORAGE-TENSOR — Make tensor constructors fallible

- Status: todo; priority: correctness; [major].
- Outcome: tensor allocation, zero-fill, constant-fill, and host-copy constructors return the backend error.
- Scope: `crates/coeus-tensor/src/tensor.rs` and direct tensor constructor tests.
- Acceptance: shape validation precedes allocation; failed allocation/fill/copy returns the exact error; successful construction preserves values and COW semantics.
- Needs: [COEUS-FALLIBLE-STORAGE-CORE](#coeus-fallible-storage-core).
- Next step: widen the existing constructor names and let the checker enumerate direct callers.
- Links: [ADR 0078](docs/adr/0078-fallible-tensor-storage.md), parent [COEUS-FALLIBLE-TENSOR-STORAGE](#coeus-fallible-tensor-storage).

<a id="coeus-fallible-storage-callers"></a>
## COEUS-FALLIBLE-STORAGE-CALLERS — Propagate tensor construction failures

- Status: todo; priority: correctness; [major].
- Outcome: all operation, autograd, NN, optimizer, distributed, and Python callers propagate tensor storage errors.
- Scope: consumer crates named by the compile closure from the tensor constructor leaf.
- Acceptance: no migrated constructor result is discarded; valid value semantics remain unchanged; provider failure fixtures reach typed public boundaries.
- Needs: [COEUS-FALLIBLE-STORAGE-TENSOR](#coeus-fallible-storage-tensor).
- Next step: run the compile probe and file any crate-level leaves before editing shared callers.
- Links: [ADR 0078](docs/adr/0078-fallible-tensor-storage.md), parent [COEUS-FALLIBLE-TENSOR-STORAGE](#coeus-fallible-tensor-storage).

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
- Outcome: ROCm/Metal provider device acquisition and Hephaestus fill/transfer return typed errors instead of panicking inside a library boundary.
- Evidence (still present): `crates/coeus-rocm/src/backend/provider.rs:34` and `crates/coeus-metal/src/backend/provider.rs:30` call `.expect(...)` inside `OnceLock::get_or_init`; the generic Hephaestus `ComputeBackend` also uses `expect` for fill and host/device transfers.
- Acceptance: value-semantic typed-error tests for unavailable devices and transfer failures; a production panic scan; provider feature gates on hosts with and without the required hardware; no CPU fallback or silent degradation.
- Next step: draft the ADR for the fallible provider-initialization/transfer contract, then migrate every implementor and caller in dependency order.
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

<a id="coeus-float-mean-provider-001"></a>
## COEUS-FLOAT-MEAN-PROVIDER-001 — Make reduced-format means representation-safe

- Status: blocked; priority: correctness; [major] [arch].
- Outcome: forward mean values and gradients remain correct for F16 and other reduced formats when the sample count cannot be represented as a finite storage-format value.
- Scope: provider-owned mean accumulation and reciprocal paths consumed by Coeus; no test-only oracle changes or workload reduction.
- Evidence: `FloatElement::from_count(65_536)` is infinite for F16, so the current Leto `MeanAxis` and Coeus fused/autograd mean paths return zero for an exactly representable `2^-16` mean. The Coeus API migration cannot claim numerical completion until the provider path is fixed and a forward-value regression passes.
- Blocker: upstream Leto `MeanAxis` and Hephaestus accelerator mean kernels require provider changes; no existing Leto or Hephaestus claim currently owns this correction.
- Re-open: an upstream provider claim lands a representation-safe mean implementation and device execution evidence; then add exact F16 forward and backward cases here and rerun the backend matrix.

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

<a id="coeus-einsum-panic"></a>
## COEUS-EINSUM-PANIC — einsum rejects invalid subscripts via typed error

- Status: todo; priority: correctness; [patch]; owner: unclaimed.
- Outcome: caller-supplied einsum subscript strings that are malformed or mismatched no longer panic.
- Scope: `coeus-autograd/src/ops/shape/util/einsum.rs:87,207`.
- Acceptance: a negative test with a malformed subscript returns a typed error, never panics.
- Next step: replace the panicking parse/validation with a typed error path.

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

<a id="coeus-autograd-linear-interpolation-generic-scalar"></a>
## COEUS-AUTOGRAD-LINEAR-INTERPOLATION-GENERIC-SCALAR — Generalize linear_interpolation over T: Float

- Status: todo; priority: tightening; [patch]; owner: unclaimed.
- Outcome: `linear_interpolation`/`linear_interpolation_backward` in `coeus-autograd` and `coeus-ops` take `Var<T, B>`/`Tensor<T, B>` for `T: Float` instead of being pinned to `f32` end-to-end, matching the variation-first-authorship standard every other autograd op in this crate already follows.
- Scope: `crates/coeus-ops/src/interpolation.rs::linear_interpolation`/`linear_interpolation_backward` and `crates/coeus-autograd/src/ops/interpolation.rs::linear_interpolation` hardcode `f32` in every signature (`image: &Var<f32, B>`, `grid: &Var<f32, B>`); `BoundaryPolicy::neighbours` also hardcodes `f32` coordinates. Found while adding this op's finite-difference gradcheck coverage (`COEUS-GRADCHECK-DUAL-COVERAGE`): the check could only be written at `f32`, the sole instantiation that compiles, rather than choosing `f32` as one of two.
- Acceptance: `linear_interpolation` is generic over `T: Float` (or the minimal bound the interpolation math needs); existing `f32` callers and tests are unaffected; `crates/coeus-autograd/tests/autograd/gradcheck/new_ops.rs::linear_interpolation_backward_matches_finite_differences` extends to a generic `<T: GradcheckScalar>` case function instantiated at both `f64` and `f32`, following this module's established pattern.
- Non-goal: changing `BoundaryPolicy`'s trait shape beyond making its coordinate type generic; adding new boundary policies.
- Next step: generalize `BoundaryPolicy::neighbours`'s coordinate parameter over `T: Float` first (the narrowest dependency), then the two `linear_interpolation` call chains, then extend the gradcheck.
