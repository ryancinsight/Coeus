# Coeus Development Backlog

Open-work queue only. Delivered items are compacted away at the commit/PR that
closed them (`git log --grep='^Item:'` or the cited PR); this file never
restates their play-by-play. Priority is one of {correctness, architecture,
verification, tightening, feature}.

<a id="milestone-ladder"></a>
## Milestone ladder

The `M-tensor-stack` milestone closed with its nine exits green at
`7db0beba`; its block is deleted. The scope that follows is proposed in
[ADR 0085](docs/adr/0085-release-scope.md). No ladder is drafted until that
proposal is answered, so the repository takes only non-feature items; the open
items carry no milestone until then.

<a id="atlas-coeus-backend-045"></a>
## ATLAS-COEUS-BACKEND-045 — Unseal ComputeBackend and unify matmul (residual)

- Status: todo (matmul unification landed); priority: architecture; [minor] [arch].
- Outcome: matmul reaches every provider through one generic dispatch; no vendor crate carries a matmul kernel. Delivered: `ComputeBackend` unsealed, `coeus-accelerator::matmul` added over `DenseProductOps`, `CudaBackend`/`WgpuBackend` migrated, `MatmulProvider<f32>` declared for Metal/ROCm.
- Residual 1 (external blocker): `HephaestusBackend<P>` still lacks `PoolOps`/`UnfoldFoldOps`, so it does not satisfy `BackendOps<f32>` for Metal/ROCm; re-open trigger is a `PoolOps<D,T>`/`UnfoldFoldOps<D,T>` seam landing upstream in `hephaestus-core`. ~2.8k pool + ~0.9k-1.0k unfold/fold lines per vendor crate stay forked until then.
- Residual 2: device-side equivalence of the provider matmul kernel against the deleted consumer kernel needs a GPU-capable CI run (no adapter on the development host).
- Residual 3: `cargo semver-checks` reports a pre-existing major break in `coeus-accelerator` against the published 0.10.0 baseline (five associated types added since publish) — needs a version decision before the next release; unrelated to this item's own change.
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

<a id="coeus-hook-fetch-residuals"></a>
## COEUS-HOOK-FETCH-RESIDUALS — Close the findings the fetch typed-errors change left open

- Status: todo; priority: correctness; milestone: none (no ladder; ADR 0085); [patch]; basis: 13a568429f68df687eaf7d689475b04272167edb.
- Outcome: the local runtime fetch of `scripts/tests/test_hooks.py` leaves no first-hop socket open after a malformed redirect, skips a blank token, pins its same-origin redirect decision in a test, and names the variable that held a refused token.
- Links: delivered by `COEUS-HOOK-FETCH-TYPED-ERRORS` (`git log --grep='^Item: COEUS-HOOK-FETCH-TYPED-ERRORS'`; PRs #659 and #660, split from #658 at `refs/pull/658/head`).
- Scope: `scripts/tests/test_hooks.py`.
- Oracles, each a loopback test on CPython 3.13.3 and 3.13.12. `fetch_atlas_runtime_file` and `atlas_runtime_token` do not exist at the basis, so each fails against the code those PRs deliver and passes with its fix:
  - F1: a malformed `Location` closes the first-hop socket. A test records warnings (`warnings.catch_warnings(record=True)`, `simplefilter("always")`) across the four malformed `Location` cases (`http://[::1/x`, `http://[zz]/x`, `http://a]b/x`, `http://[::1]x/y`) with `gc.collect()` after each, and asserts none is a `ResourceWarning`; it records 4, one per case, with a token set and without, on both interpreters.
  - F3: a blank `GITHUB_TOKEN` or `GH_TOKEN` (`" "`, `"\r\n"`) counts as unset: `atlas_runtime_token` skips it and the request carries no `Authorization`.
  - F4 (decision): a same-origin 301 drops `Authorization`, as the closed item's clause reads. Keep that and add a test pinning it, or send the token on the same origin and amend the clause.
  - F7: a refusal names the variable that held the token, `GITHUB_TOKEN` or `GH_TOKEN`.
- Next step: F3 and F7 first; both change what `atlas_runtime_token` returns.

<a id="coeus-hook-runtime-pin-test"></a>
## COEUS-HOOK-RUNTIME-PIN-TEST — Test that CI pins one Atlas checkout at the tested revision

- Status: todo; priority: verification; [patch]; basis: 779c14b7.
- Outcome: a parse-based test of `.github/workflows/ci.yml` fails when the `format` or `tests` job does not hold one Atlas checkout at the revision `scripts/tests/test_hooks.py` pins, with `ATLAS_HOOK_RUNTIME` pointing at it. Its docstring, test name and the `ci.yml` comment state exactly what it enforces and list what it does not read.
- Needs: COEUS-HOOK-RUNTIME-FETCH (the pinned-checkout CI change).
- Scope: `scripts/tests/test_hooks.py`, the `ci.yml` comment on `ATLAS_RUNTIME_REVISION`.
- Prior attempt: `refs/pull/652/head` (178 test lines, three reviewed FAILs, whose witness IDs below are defined in the round 2 and 3 verdict comments on PR #652). No YAML library is imported by any script here and the Format job installs none; its small block-mapping reader failed on the cases below.
- Acceptance, each a mutant of `ci.yml` alone that the test fails, or a case the docstring and `ci.yml` comment name as unread: step `env` setting the revision name on the checkout step (R5c, T4); a second checkout of `RyanCinsight/atlas`, `${{ github.repository_owner }}/atlas`, or `uses: Actions/checkout` (R4b, R4c, R4d, T6); job `env` by alias (R2b); revision edited, old line kept as a comment, both refs set to another SHA (M6-M8); runtime value, checkout path, runtime removed, second runtime step, tests-job runtime (C1-C4, T3).
- Block scalars, the round-3 survivors (each AtlasRuntimePinTests-passing at that head): a second atlas checkout with `uses: >-` (N36 format, N36t tests), with a repository expression (N36b, N36bt); `env: >-` and `with: >-` holding an expression that evaluates to a mapping, on the checkout step, the tests-job checkout step and the format job (N37, N37t, N37j, N38). Controls that fail: the one-line forms N37p, N38p.
- Filed with them, each named unread or enforced: step `env` set on the checkout step instead of the test step (F1), key case (F2), a later checkout into the same path (F3), `github-server-url` (F5).
- Rule: no sentence says the test enforces a case it does not read; each docstring, test name, `ci.yml` and PR sentence is checked against a mutant before it is written.

<a id="coeus-device-test-build-cost"></a>
## COEUS-DEVICE-TEST-BUILD-COST — Attribute device test compilation

- Status: todo; priority: verification; [patch].
- Outcome: bound device-test compilation while preserving the full scalar/backend matrix.
- Scope: CPU test placement and monomorphization in existing integration harnesses; no assertion, scalar, workload or runtime-budget reduction.
- Evidence: ownership RED compilation took 13m06s while native execution took 7.965s; `wgpu_ops` rustc was the last compiler in the integrated attempt and also instantiates CPU-only scalar matrices.
- Acceptance: compiler timing/codegen evidence identifies the dominant work; an evidence-backed structural change preserves the enumerated test matrix and lowers its attributed compile cost, or records why the proposed partition does not help.
- Needs: [kernel output correction](#coeus-device-output-ownership); keep compiler flags and shared target policy fixed for comparison.

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

<a id="coeus-benchmark-manifest-g043"></a>
## COEUS-BENCHMARK-MANIFEST-G043 — Close remaining partial rows in the every-family evidence manifest

- Status: todo; priority: verification; [patch].
- Outcome: every implemented NN family has a non-partial Criterion and Python-differential disposition in `crates/coeus-nn/benches/nn_bench/evidence.tsv`.
- Evidence: Criterion lane complete (0 partial, via #593/#594/#596/#597: 52 new run benches; consistency check 2/2). Python lane still partial on 4 rows: conv, dropout, regularization, rnn.
- Acceptance: the manifest consistency check passes with zero partial rows, or each remaining partial row has a recorded reason (no comparable Burn/PyTorch API) rather than a silent gap.
- Non-goal: fabricating unsupported external-framework rows (several families have no Burn/PyTorch equivalent and are correctly Coeus-only).
- Next step: close the 4 partial Python rows (conv transpose/depthwise variants, dropout stochastic train-mode, regularization train-mode, rnn sequence variants) with pytest differential coverage or explicit inapplicability records.

<a id="coeus-staggered-vendor-wiring"></a>
## COEUS-STAGGERED-VENDOR-WIRING — Bind the remaining vendor backends for StaggeredPairOps

- Status: todo; priority: tightening; [minor].
- Outcome: bind `StaggeredPairOps<f32>` for CUDA and Metal through the existing provider-owned kernels and shared staggered dispatch.
- CUDA: `coeus-cuda` already uses the Hephaestus CUDA provider with no Cutile/`cuda-bindings` dependency; add the missing provider/backend binding without duplicating dispatch.
- Metal: `HephaestusBackend<MetalProvider>` already uses the generic bridge; add `StaggeredProvider` for `MetalProvider` to obtain its existing generic implementation.
- Blocked (external): ROCm requires upstream staggered kernels in Hephaestus (`CudaStaggered3DOps`/`MetalStaggered3DOps` exist upstream; ROCm does not).
- Acceptance: exercise the shared gradient/divergence, axis, adjoint, and invalid-grid oracles on real CUDA and Apple Metal devices; preserve typed acquisition failures.

<a id="coeus-release-1-0-0-001"></a>
## COEUS-RELEASE-1-0-0-001 — Release Coeus 1.0.0

- Status: blocked; priority: verification; [major]; owner: root.
- Blocker: release authority. Re-open trigger: the user grants release authority for the 1.0.0 release.
- Outcome: the workspace version becomes 1.0.0, the changelog records the milestone's delivered surface, and the release tag carries the published artifacts.
- Closing commit: `7db0beba` closes `M-tensor-stack`; its nine exits pass there and the `oversized_files` standing target holds at seven files.
- Scope: the workspace version fields, `CHANGELOG.md`, the release tag and the publish pipelines; no source change.
- Acceptance: the SemVer gate and the publish dry run pass at the release commit, the crates.io and PyPI publications succeed under OIDC, and the tag resolves the release notes.
- Links: [ADR 0083](docs/adr/0083-release-scope.md), [ADR 0085](docs/adr/0085-release-scope.md).

<a id="ms-445-python-release-wheels"></a>
## MS-445 — Python release wheels

- Status: blocked; priority: verification; [patch]; owner: root.
- Blocker: release authority; publication is a non-goal of the 1.0 ladder ([ADR 0083](docs/adr/0083-release-scope.md)). Re-open trigger: the user grants release authority after the milestone closes.
- Outcome: a GitHub Release tagged `coeus-python-v<version>` builds locked Linux/Windows/macOS wheels for CPython 3.9-3.13, installs and imports each as `pycoeus`, attests and attaches artifacts, then publishes to the `coeus-python` PyPI project through OIDC.
- Delivered: the release workflow (`.github/workflows/python-release.yml`) and distribution contract are implemented; GitHub environment `pypi` accepts only `coeus-python-v*` tags; a locked CPython 3.13 wheel builds, installs, and imports as `pycoeus`.
- Residual: hosted CI on the exact release-automation head, and PyPI pending-trusted-publisher registration, remain open.
- Non-goal: Python binding behavior changes.

<a id="coeus-registry-package-1"></a>
## COEUS-REGISTRY-PACKAGE-1 — Publish reusable crates through Trusted Publishing

- Status: blocked; priority: verification; [patch]; owner: root.
- Blocker: release authority; publication is a non-goal of the 1.0 ladder ([ADR 0083](docs/adr/0083-release-scope.md)). Re-open trigger: the user grants release authority after the milestone closes.
- Outcome: Coeus's publishable Rust crates (currently all `publish = false` or unset) release to crates.io in dependency order via OIDC trusted publishing.
- Delivered: Moirai/Mnemosyne/Themis imports bind to their published packages (no `rev =` pin — verified in `Cargo.toml`); the exact external lock graph is refreshed after the Apollo/Hephaestus/Moirai/Mnemosyne merges it depended on.
- Residual: exact-head provider CI on the release-preparation branch, then publish in dependency order through Trusted Publishing (engineering_gates: publish pipelines).
- Next step: run the hosted gate on the release branch, then execute the publish sequence.

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
- Outcome: the files over 500 lines are split into leaf modules by operation family; 7 remain.
- Delivered: `coeus-python/src/tensor/pyimpl/pytensor.rs` (985 lines) is now `pytensor/{mod,construct,protocol,arithmetic,unary,reduction,shape,indexing,compare,inplace,cast,grad}.rs`, largest 152, one `#[pymethods]` section per file (pyo3's `multiple-pymethods` feature); `coeus-ops/src/shape/einsum.rs` (688 lines) is now `einsum/{mod,dispatch,subscript,binary,ternary,single}.rs` plus its `tests`. Seven further files are split by operation family: `coeus-autograd/tests/cache_benchmarks.rs` (720 lines) is now `cache_benchmarks/benches/{mod,graphs,plans,lookup}.rs`; `coeus-autograd/tests/autograd/gradcheck/activation.rs` (658) is now `activation/{mod,smooth,elementwise,branch}.rs`; `coeus-wgpu/tests/wgpu_ops/backend/wgpu/parity/elementwise.rs` (616) is now `parity/elementwise/{mod,binary,activations,inplace,buffer,unary}.rs`; `coeus-autograd/src/ops/nn/pool.rs` (610) is now `pool/{mod,max,avg,adaptive}.rs`; `coeus-autograd/tests/autograd/gradcheck/losses.rs` (594) is now `losses/{mod,regression,classification,ranking,similarity}.rs`; `coeus-nn/benches/nn_bench/provider/normalization_forward.rs` (583) keeps six benchmarks and moves the rest to `normalization_forward/extended.rs`; `coeus-optim/src/least_squares/tests.rs` (562) keeps the test bodies and moves the problem fixtures to `tests/fixtures.rs`. The scan now reports 7 oversized files, at the milestone's ceiling.
- Residual: `coeus-ops/src/backend_ops/traits/conv.rs` (646 lines) holds the `ConvOps` trait, whose twelve rank-specific default methods carry long signatures, rustdoc and ratcheted `too_many_arguments` expectations; moving them out of the trait changes the public contract, so it waits for a request-struct migration. `coeus-autograd/src/ops/interpolation.rs` (543), `coeus-nn/benches/nn_bench/provider/pooling_interpolation.rs` (529), `coeus-autograd/src/ops/linalg.rs` (522), `coeus-core/tests/core_ops/policy/dependency_policy.rs` (510), `coeus-ops/src/resize.rs` (502) and `coeus-ops/tests/alloc_budget.rs` (501) remain.
- Scope: repo-wide scan for the rest.
- Acceptance: each split file's module retains domain cohesion; no file regresses over 500 lines.
- Next step: split the six remaining production and test files; the `ConvOps` trait waits for a request-struct migration.

<a id="coeus-accelerator-padops-adoption"></a>
## COEUS-ACCELERATOR-PADOPS-ADOPTION — Adopt hephaestus's PadOps<D, T> seam

- Status: blocked; priority: architecture; [patch]; owner: unclaimed.
- Outcome: `coeus-accelerator` consumes hephaestus-core's `PadOps<D, T>` seam once merged, instead of any local padding duplication.
- Blocker: hephaestus PadOps seam not yet merged upstream (re-open trigger: hephaestus-core PadOps<D,T> lands).
- Scope: `coeus-accelerator` padding call sites.
- Acceptance: coeus-accelerator padding routes through the upstream seam; no duplicated padding logic remains locally.

<a id="coeus-gradcheck-generic-instantiation"></a>
## COEUS-GRADCHECK-GENERIC-INSTANTIATION — Generic-instantiation gradcheck coverage

- Status: todo; priority: verification; [patch]; owner: unclaimed.
- Outcome: every finite-difference check in `crates/coeus-autograd/tests/autograd/gradcheck/` runs at both `f64` (sensitivity oracle) and `f32` (instantiation coverage, per `standards`: Generic Instantiation Coverage), plus FD coverage for the ops that had none.
- Delivered: `mod.rs`'s shared fixtures (`Sampler`, `tensor`, `weighting`, `weighted`) are generic over `GradcheckScalar` (`Float + leto_ops::Scalar + coeus_cpu::RealScalar` plus the `CpuAddressableStorage` bound gradcheck itself needs); the module doc states the two-role f64/f32 rationale. New FD coverage, each a generic fn instantiated at both types from a single `#[test]`: `add`/`sub`/`div`/`remainder`/`maximum`/`minimum`/`neg` (`arithmetic.rs`); `max_pool1d/2d/3d`/`avg_pool1d/2d/3d` (`pooling.rs`, via `coeus-nn` `Module` layers, new dev-dependency); `conv1d`/`conv3d`/`conv_transpose1d/2d/3d`/`unfold1d/2d`/`fold1d/2d` (`conv.rs`, via `coeus-nn` `Module` layers). Converted to dual-instantiation: `normalization.rs`, `core_ops.rs`, `attention.rs`.
- Remaining: (1) convert the 4 remaining existing check files (`activation`, `losses`, `reduction`, `shape`) to dual-instantiation; (2) FD coverage still missing: `ctc`, `dropout` (fixed mask), `sparse_matmul(_coo)`, `transpose_2d`, `index_put`, `rotate_half`, `linear_interpolation` (`cross_entropy` and `embedding` already have coverage in `losses.rs`).
- Acceptance: every listed op has a generic FD check instantiated at both types; all 11 existing files are dual-instantiated; a check that fails at `f32` under gradcheck's own derived `ε^(2/3)` bound is root-caused, never given a widened tolerance.
- Next step: convert `activation.rs` next (largest remaining, ~50 tests) then `losses`/`reduction`/`shape`, each its own PR; then the remaining missing-coverage ops as dependency-ordered per-family PRs.
