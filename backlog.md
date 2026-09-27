# Coeus Development Backlog

Open-work queue only. Delivered items are compacted away at the commit/PR that
closed them (`git log --grep='^Item:'` or the cited PR); this file never
restates their play-by-play. Priority is one of {correctness, architecture,
verification, tightening, feature}.

<a id="coeus-communicator-fallible-collectives"></a>
## COEUS-COMMUNICATOR-FALLIBLE-COLLECTIVES — TCP collectives panic on peer I/O failure

- Status: todo; priority: correctness; [major] (`Communicator` methods gain a `Result`).
- Outcome: `Communicator` collectives return a typed error, so `TcpCommunicator` propagates `TcpMeshError` from `send`/`recv` instead of panicking; `LocalCommunicator` and `synchronize_gradients` follow.
- Scope: `crates/coeus-dist/src/tcp/collectives.rs` (`TcpCommunicator::send`/`recv` still call `panic!("{}", ErrorChain(&error))` at lines 34/42).
- Acceptance: no panic on an I/O result in `tcp/collectives.rs`; a test drops a peer mid-collective and asserts the typed error; Python collectives raise `ConnectionError`.
- Next step: add the `Result` channel to the `Communicator` trait, migrate `TcpCommunicator`/`LocalCommunicator`, update Python bindings.
- Links: [ADR 0074](adr/0074-fallible-tcp-mesh.md) (successor to the closed COEUS-TCPMESH-FALLIBLE-SETUP).

<a id="coeus-fallible-unary-execution"></a>
## COEUS-FALLIBLE-UNARY-EXECUTION — Propagate unary provider failures

- Status: todo; priority: correctness; [major] [arch].
- Outcome: provider failures reach callers without input-dependent panics for shared unary autograd execution.
- Scope: shared unary autograd execution and its Rust/Python callers.
- Acceptance: complete caller migration, typed failure/gradient tests, full native/device gates, SemVer classification with an updated ADR.
- Needs: [fallible tensor storage](#coeus-fallible-tensor-storage) — its allocation/COW cutover must include the unary, derivative arithmetic and NN/Python callers it makes fallible.
- Non-goal: resurrect superseded dependency pins or storage implementations.
- Next step: reserve a forward ADR (0042/0045 cover backward/module contracts only); wrap unary results after fallible storage lands.

<a id="coeus-fallible-index-reduction"></a>
## COEUS-FALLIBLE-INDEX-REDUCTION — Return index-reduction failures

- Status: todo; priority: correctness; [major].
- Outcome: malformed argmax/argmin requests return typed errors through their complete caller chain.
- Scope: index-reduction backend trait methods, CPU/provider implementations, mathematical/Python callers; preserve index conventions and tie behavior.
- Evidence: `coeus-ops/src/backend_ops/cpu_impl/reduction.rs` expects fallible Leto argmax/argmin results; current trait methods return no error (source finding, not an executed reproduction).
- Acceptance: malformed layouts/axes reject before output writes, retain clone values, propagate typed errors; valid ties/boundaries retain exact expected indices.
- Needs: coordinate with [fallible storage](#coeus-fallible-tensor-storage) for allocation propagation; reserve the governing ADR and migration before implementation.
- Next step: draft the ADR, then implement after `coeus-fallible-tensor-storage` lands.

<a id="coeus-storage-allocation-owner"></a>
## COEUS-STORAGE-ALLOCATION-OWNER — Consolidate backend allocation

- Status: todo; priority: correctness prerequisite; [major].
- Outcome: backend allocation owns construction; storage traits describe existing allocations.
- Scope: remove `Storage::allocate` (still present: `crates/coeus-core/src/storage/traits.rs:31`, `storage/cow.rs:60`, `storage/cpu.rs:241`) and its CPU/Cow/provider implementations; CPU backends call `CpuStorage::new` directly. No Tensor signature or allocation semantics change.
- Acceptance: no static storage factory remains; all callers compile; existing initialization/COW and twelve-scalar backend-write cases preserve exact values.
- Needs: [kernel output ownership](#coeus-device-output-ownership) landed (it has); prepares [fallible storage](#coeus-fallible-tensor-storage).
- Next step: revise [ADR 0037](adr/0037-uninitialized-cow-consumer.md) with construction ownership and migration before implementation.

<a id="coeus-fallible-tensor-storage"></a>
## COEUS-FALLIBLE-TENSOR-STORAGE — Propagate tensor storage failures

- Status: todo; priority: correctness; [major] [arch].
- Outcome: allocation, zero-fill and copy-on-write failures reach typed callers.
- Scope: `ComputeBackend` storage methods, Tensor constructors/materialization (`alloc_on`/`zeros_on` in `crates/coeus-tensor/src/tensor.rs` still return `Self`, not `Result`), provider implementations and their complete caller closure.
- Acceptance: no provider-error expects on the migrated paths; real malformed-size/layout/device error tests; no partial writes claimed as whole-graph rollback.
- Needs: [CPU ownership correction](#coeus-cpu-storage-ownership) (done); driver for [unary](#coeus-fallible-unary-execution) and [index-reduction](#coeus-fallible-index-reduction) consumers.
- Next step: reserve an ADR; constructor search found 2,781 textual candidates in 406 files at last audit — scope the cutover before implementation.

<a id="coeus-device-output-ownership"></a>
## COEUS-DEVICE-OUTPUT-OWNERSHIP — Collect the device/hardware verification half

- Status: in-progress (core landed); priority: correctness; [major] [arch].
- Outcome: every mutable GPU kernel output preserves other tensor clones; CUDA/WGPU own generic provider storage with no temporary `from_arc` owner.
- Delivered: commit `9f897f2e` ("Give CUDA and WGPU generic provider storage") lands the core migration on `main` — vendor storage types and `from_arc` bridges are gone (`grep from_arc crates/coeus-{cuda,wgpu}/src` is empty), CPU scans propagate rejection, 267/267 CPU tests pass.
- Residual: device/hardware matrices still need collecting, and the commit's own message names the open half — "the upstream CUDA scalar compiler correction" — tracked upstream at `hephaestus/backlog.md#heph-cuda-dense-product-scalars`.
- Acceptance: real-device regressions over each supported operation/scalar matrix pass; SemVer confirms the declared major removals (already 222/223 checks pass per the landed commit).
- Next step: run the final workspace/device/SemVer gates once the upstream Hephaestus fix lands, then close.

<a id="coeus-scatter-add-allocation"></a>
## COEUS-SCATTER-ADD-ALLOCATION — Keep scatter-add allocation count independent of shape

- Status: in-progress; priority: correctness; [patch].
- Outcome: `scatter_add` preserves value semantics without shape-scaled allocation calls.
- Scope: `crates/coeus-ops/src/shape/select/scatter.rs`, focused allocation regression; non-goal: changing scatter indexing semantics.
- Acceptance: the allocation-budget test passes for small and large 3-D shapes; strict package Clippy, nextest, docs pass.
- Evidence: merged-main WGPU provider run `35660423617` failed `scatter_add_allocation_count_is_independent_of_index_size` with 7 versus 11 allocations.
- Related: [COEUS-ALLOC-BUDGET-INTERMITTENT](#coeus-alloc-budget-intermittent) — same test now shows platform-dependent counts; investigate together.
- Branch: `fix/coeus-scatter-add-allocation`.

<a id="coeus-alloc-budget-intermittent"></a>
## COEUS-ALLOC-BUDGET-INTERMITTENT — scatter_add allocation budget fails intermittently on Linux CI

- Status: todo; priority: correctness; [patch].
- Outcome: `scatter_add_allocation_count_is_independent_of_index_size` gives the same count on Linux CI as on Windows (12/12 local runs, 3/3 main runs), or the root cause is identified and fixed.
- Observed: one PR run failed `small=7, large=11` (4 allocations more for a 64x larger workload — log2(64)=6, not the ~16 a per-slice defect would give); the changed line in that PR did not touch the exercised job.
- Acceptance: reproduce on a Linux runner with the counter printed at each step of the measured window; identify whether it is size-dependent `Vec` growth inside the kernel or an allocation from outside the kernel entering the measured window.
- Next step: re-run the failing job with `--no-capture` and per-step counter output; not reproducible from Windows (12/12 local runs pass).

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

- Status: in-progress (matmul unification landed); priority: architecture; [minor] [arch].
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

- Status: in-progress; integrator: coeus-hook-ci; priority: verification; [patch].
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

<a id="coeus-workspace-lint-floor"></a>
## COEUS-WORKSPACE-LINT-FLOOR — Recover the inherited lint floor

- Status: todo; priority: verification; [patch].
- Outcome: one workspace `[workspace.lints]` table governs every member (currently absent from `Cargo.toml` — verified).
- Scope: workspace lint inheritance and measured existing suppressions.
- Evidence: unique `74fd5c11` survives on `perf/coeus-ops-index-decode` and `fix/coeus-autograd-honest-cache`; current manifest lacks its floor.
- Acceptance: strict all-target Clippy, non-increasing residual counts, native/doc gates, no blanket suppression growth.
- Next step: add `[workspace.lints]` with `clippy::pedantic` inheritance per `standards`, then integrate the two surviving branches' floor once.

<a id="coeus-provider-resolution-2026-09-07"></a>
## COEUS-PROVIDER-RESOLUTION-2026-09-07 — Restore fresh provider resolution after upstream alignment

- Status: review; priority: verification; [arch].
- Outcome: fresh lock generation and default/all-feature locked resolution agree, including against current upstream (not only offline idempotence).
- Evidence: a frozen tree regenerates the same offline lock twice and both activation checks pass — this proves closure/idempotence, not remote freshness against the current upstream state.
- Acceptance: standalone regeneration is idempotent and all configured gates pass against a freshly fetched upstream.
- Next step: re-run `cargo update` against current remotes and confirm the lock still resolves after any upstream version alignment since the last check.

<a id="coeus-autograd-l1-provider-001"></a>
## COEUS-AUTOGRAD-L1-PROVIDER-001 — Collect hosted evidence for the provider-owned L1 loss

- Status: in-progress (implementation + local verification complete); priority: verification; [patch] [arch].
- Outcome: L1 forward/backward compose provider `sub`/`abs`/`mean_axis`/`sign`/`mul`/`neg`; no host-resident `Vec<T>`. Already true in the merged implementation.
- Residual: hosted WGPU/CUDA/ROCm/Metal evidence remains pending before this child can close; the pre-1.0 `L1LossNode` representation change (`diffs: Vec<T>` → provider-resident `Tensor<T,B>`) needs the package's next SemVer review.
- Local evidence: focused Clippy, Nextest 3/3, doctests (autograd 16/16, nn 8/8 with 2 intentionally ignored), formatting, residue scan, diff hygiene pass.
- Next step: run the exact-head hosted backend matrix and record the SemVer classification.

<a id="coeus-autograd-lp-norm-provider-001"></a>
## COEUS-AUTOGRAD-LP-NORM-PROVIDER-001 — Collect doctest/SemVer evidence for provider-owned Lp norms

- Status: review; priority: verification; [major] [arch].
- Outcome: `norm_p`/`norm_p_axis` perform provider-resident forward/backward (CPU via Leto `PowfOp`, WGPU/CUDA/ROCm/Metal via Hephaestus scalar-strided `PowOp`) — already merged (`4b915102`, `d775cd90`, provider blocker resolved in `cd36ee64`).
- Residual: collect workspace-doctest evidence against the delivered revision, and run SemVer against the intended baseline (existing published-0.9.0 comparison found three pre-existing cumulative-scan/fusion breaks unrelated to this item; classify Lp-norm-specific breaks separately).
- Evidence so far: `cd36ee64` records passing locked provider checks, provider Clippy, Hephaestus regressions, and backend-parity run `31052471989` for WGPU/CUDA/ROCm/Metal.
- Links: [ADR 0056](adr/0056-provider-owned-lp-norms.md).
- Next step: run the workspace doctest suite and `cargo-semver-checks` against the recorded baseline; record both in this item.

<a id="coeus-frobenius-norm-provider-001"></a>
## COEUS-FROBENIUS-NORM-PROVIDER-001 — Collect hosted provider contracts for provider-owned Frobenius norm

- Status: in-progress (local implementation complete); priority: verification; [patch] [arch].
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

- Status: in-progress; priority: feature; [patch]; owner: root.
- Outcome: a GitHub Release tagged `coeus-python-v<version>` builds locked Linux/Windows/macOS wheels for CPython 3.9-3.13, installs and imports each as `pycoeus`, attests and attaches artifacts, then publishes to the `coeus-python` PyPI project through OIDC.
- Delivered: the release workflow (`.github/workflows/python-release.yml`) and distribution contract are implemented; GitHub environment `pypi` accepts only `coeus-python-v*` tags; a locked CPython 3.13 wheel builds, installs, and imports as `pycoeus`.
- Residual: hosted CI on the exact release-automation head, and PyPI pending-trusted-publisher registration, remain open.
- Non-goal: Python binding behavior changes.

<a id="coeus-registry-package-1"></a>
## COEUS-REGISTRY-PACKAGE-1 — Publish reusable crates through Trusted Publishing

- Status: in-progress; priority: feature; [patch]; owner: root.
- Outcome: Coeus's publishable Rust crates (currently all `publish = false` or unset) release to crates.io in dependency order via OIDC trusted publishing.
- Delivered: Moirai/Mnemosyne/Themis imports bind to their published packages (no `rev =` pin — verified in `Cargo.toml`); the exact external lock graph is refreshed after the Apollo/Hephaestus/Moirai/Mnemosyne merges it depended on.
- Residual: exact-head provider CI on the release-preparation branch, then publish in dependency order through Trusted Publishing (engineering_gates: publish pipelines).
- Next step: run the hosted gate on the release branch, then execute the publish sequence.
