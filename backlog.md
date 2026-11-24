# Coeus Development Backlog

Open-work queue only. Delivered items are compacted away at the commit/PR that
closed them (`git log --grep='^Item:'` or the cited PR); this file never
restates their play-by-play. Priority is one of {correctness, architecture,
verification, tightening, feature}.

<a id="coeus-stable-qualification"></a>
## COEUS-STABLE-QUALIFICATION — coeus stable-release qualification [patch] — todo
- priority: verification; basis: `f560da2521b4`.
- Outcome: Qualify the physical-autodiff cohort and settle the meaning of the local tensor milestone. Follow the [release scope](docs/adr/0083-release-scope.md) and `M-tensor-stack`. See the [proposed decision](docs/adr/0088-external-reference-qualification.md). Planning targets for `coeus-core` are 0.15.1 if CPU repairs are needed, then 0.16.0 if reviewed additions are needed, then 1.0.0. Verification alone can advance the current version. Use Hephaestus 0.32.0 for the [verified downstream contract](gap_audit.md#verified-downstream-provider-migration).
- Representative manifest `coeus-core 0.15.0`. Target 1.0.0. The current release plan selects eight CPU packages. Seven other publish-enabled workspace packages need separate qualification.
- [x] [patch] The eight CPU packages are published at 0.15.0. A clean registry consumer verified gradients with physical units enabled and disabled on Rust 1.95. The [release record](docs/adr/0087-physical-quantity-autodiff.md#release-record) holds archive checksums and the source revision.
- [ ] [arch] Review the proposed external-oracle protocol alongside accepted ADR 0083 and proposed ADR 0085. Record which additional operations require provider designs before scheduling implementation.
- [ ] [major] Re-run the nine local tensor exits on a candidate and satisfy the owned Atlas tensor, AD, graph and distributed obligations with reference and native-device evidence. Keep the next-major proposal under its separate approval.
- [ ] [patch] **0.15.1 if CPU repairs are needed.** Use the released eight-package CPU cohort as the comparator. Re-run the physical-gradient quotient with units enabled and the ordinary gradient with units disabled in a fresh caller. [Source and baseline](gap_audit.md#incremental-release-baseline).
- [ ] [minor] **0.16.0 if reviewed additions are needed.** After the external-reference decision, select one owned tensor family, map its operators and shapes to the reference, then compare forward values and gradients on each claimed device. Record unsupported routes and the seven separately qualified packages.
- [x] [patch] Verify Hephaestus 0.32.0 through the real consumer and independent dependency resolution. [Executed coverage](gap_audit.md#verified-downstream-provider-migration).
- [ ] [major] **1.0.0-beta.1 → 1.0.0-rc.1 → 1.0.0.** Freeze the supported surface at beta. The nine tensor exits and owned Atlas evidence identify the same candidate and feature graph. A next-major proposal stays with its own decision. Repeat the applicable exits on the exact RC before stable promotion. Attach exact-revision evidence for the local exits and this member’s [Atlas qualification criteria](https://github.com/ryancinsight/atlas/blob/17b0124a7548ad81e817284afc5cb115c1d1bc78/docs/parity/repositories.json). Require complete implementations and passing product-path acceptance for all critical features.

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

<a id="coeus-hook-runtime-pin-test"></a>
## COEUS-HOOK-RUNTIME-PIN-TEST — Test that CI pins one Atlas checkout at the tested revision

- Status: todo; priority: verification; [patch]; basis: 779c14b7.
- Outcome: a parse-based test of `.github/workflows/ci.yml` fails when the `format` or `tests` job does not hold one Atlas checkout at the revision `scripts/tests/test_hooks.py` pins, with `ATLAS_HOOK_RUNTIME` pointing at it. Its docstring, test name and the `ci.yml` comment state exactly what it enforces and list what it does not read.
- Needs: none.
- Scope: `scripts/tests/test_hooks.py`, the `ci.yml` comment on `ATLAS_RUNTIME_REVISION`.
- Prior attempt: `refs/pull/652/head` (178 test lines, three reviewed FAILs, whose witness IDs below are defined in the round 2 and 3 verdict comments on PR #652). No YAML library is imported by any script here and the Format job installs none; its small block-mapping reader failed on the cases below.
- Acceptance, each a mutant of `ci.yml` alone that the test fails, or a case the docstring and `ci.yml` comment name as unread: step `env` setting the revision name on the checkout step (R5c, T4); a second checkout of `RyanCinsight/atlas`, `${{ github.repository_owner }}/atlas`, or `uses: Actions/checkout` (R4b, R4c, R4d, T6); job `env` by alias (R2b); revision edited, old line kept as a comment, both refs set to another SHA (M6-M8); runtime value, checkout path, runtime removed, second runtime step, tests-job runtime (C1-C4, T3).
- Block scalars, the round-3 survivors (each AtlasRuntimePinTests-passing at that head): a second atlas checkout with `uses: >-` (N36 format, N36t tests), with a repository expression (N36b, N36bt); `env: >-` and `with: >-` holding an expression that evaluates to a mapping, on the checkout step, the tests-job checkout step and the format job (N37, N37t, N37j, N38). Controls that fail: the one-line forms N37p, N38p.
- Filed with them, each named unread or enforced: step `env` set on the checkout step instead of the test step (F1), key case (F2), a later checkout into the same path (F3), `github-server-url` (F5).
- Rule: no sentence says the test enforces a case it does not read; each docstring, test name, `ci.yml` and PR sentence is checked against a mutant before it is written.

<a id="coeus-lockfile-test-tempdir"></a>
## COEUS-LOCKFILE-TEST-TEMPDIR — Build lockfile fixtures outside the checkout

- Status: todo; priority: verification; [patch]; basis: 3c8b71ae4f14bd94b25d62f6196f3049787b2224.
- Outcome: the two `scripts/tests/test_lockfile.py` tests that build a git dependency and a local registry create their fixture trees outside the checkout.
- Evidence: `:72` and `:175` call `tempfile.TemporaryDirectory(dir=Path(__file__).parent)`, so each run writes a git repository, a registry index and an archive under `scripts/tests/`. The `:173-174` comment attributes the full-path need to a libgit2 failure on the shortened Windows profile path "on the CI host"; the only CI step that runs this file is `Lockfile tool contracts` in the `format` job (`ci.yml:99-123`, `python3 -m unittest discover -s scripts/tests -v`, `ubuntu-latest`), so that premise is unreproduced.
- Scope: those two tests; the `:173-174` comment is corrected to what is reproduced.
- Acceptance: a committed test in `test_lockfile.py` runs the two build tests with `tempfile.TemporaryDirectory` replaced by a recorder. `setUp` runs as written, including its `cargo generate-lockfile --offline` call, so the test needs cargo like the file's other tests. A call made from a test body (not `setUp`, not another file) records `Path(dir or tempfile.gettempdir()).resolve()` and raises there, before the body's git, registry or cargo calls. The test asserts both roots are recorded and none `is_relative_to` the repository root (`Path(__file__).resolve().parents[2]`). At the basis both roots are `scripts/tests` and the test fails. It runs in the `ci.yml:123` step above.
- Needs: none.
- Next step: root both fixtures in the system temp directory, resolved with `Path.resolve()` as `:73` and `:176` do, then add the recorder test.

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
## COEUS-REGISTRY-PACKAGE-1 [patch] — Qualify automatic publication for coeus-cpu

- Status: blocked; priority: verification; [patch]; owner: unclaimed.
- Blocker: inspecting the `coeus-cpu` Trusted Publisher registration needs a signed-in crates.io owner of that crate, and the automatic route is qualified only by a release run. Re-open trigger: the owner inspects the registration, or release authority for the next cohort is granted.
- Evidence: the log of [job 137328612002](https://github.com/ryancinsight/Coeus/actions/runs/45060488822/job/137328612002) reads ``the remote server responded with an error (status 403 Forbidden): The provided access token is not valid for crate `coeus-cpu` ``. The 0.14.0 log is not read.
- The 0.15.0 release workflow's crates.io token received HTTP 403 when it
  reached `coeus-cpu`. Publishing that crate from the exact merge revision
  with the existing Cargo credential unblocked the remaining automatic
  uploads. The same failure occurred in 0.14.0.
  [Release run](https://github.com/ryancinsight/Coeus/actions/runs/45060488822)
- Inspect the `coeus-cpu` Trusted Publisher registration for the
  `ryancinsight/Coeus` repository, `rust-release.yml` workflow, and
  `crates-io` environment before the next cohort. Qualify the automatic
  route on the next authorized release.

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
- Outcome: the files over 500 lines are split into leaf modules by operation family; 1 remains, pending a public-contract migration.
- Delivered: `coeus-python/src/tensor/pyimpl/pytensor.rs` (985 lines) is now `pytensor/{mod,construct,protocol,arithmetic,unary,reduction,shape,indexing,compare,inplace,cast,grad}.rs`, largest 152, one `#[pymethods]` section per file (pyo3's `multiple-pymethods` feature); `coeus-ops/src/shape/einsum.rs` (688 lines) is now `einsum/{mod,dispatch,subscript,binary,ternary,single}.rs` plus its `tests`. Thirteen further files are split by operation family: `coeus-autograd/tests/cache_benchmarks.rs` (720 lines) is now `cache_benchmarks/benches/{mod,graphs,plans,lookup}.rs`; `coeus-autograd/tests/autograd/gradcheck/activation.rs` (658) is now `activation/{mod,smooth,elementwise,branch}.rs`; `coeus-wgpu/tests/wgpu_ops/backend/wgpu/parity/elementwise.rs` (616) is now `parity/elementwise/{mod,binary,activations,inplace,buffer,unary}.rs`; `coeus-autograd/src/ops/nn/pool.rs` (610) is now `pool/{mod,max,avg,adaptive}.rs`; `coeus-autograd/tests/autograd/gradcheck/losses.rs` (594) is now `losses/{mod,regression,classification,ranking,similarity}.rs`; `coeus-nn/benches/nn_bench/provider/normalization_forward.rs` (583) keeps six benchmarks and moves the rest to `normalization_forward/extended.rs`; `coeus-optim/src/least_squares/tests.rs` (562) keeps the test bodies and moves the problem fixtures to `tests/fixtures.rs`; `coeus-autograd/src/ops/interpolation.rs` (543) is now `interpolation/{mod,linear,grid_sample}.rs`; `coeus-nn/benches/nn_bench/provider/pooling_interpolation.rs` (529) is now `pooling_interpolation/{mod,pool,interpolation}.rs` plus the existing `extended.rs`; `coeus-autograd/src/ops/linalg.rs` (522) is now `linalg/{mod,dense,sparse}.rs`; `coeus-core/tests/core_ops/policy/dependency_policy.rs` (510) is now `policy/dependency_policy/{forbidden,lockfile,dependency_tree,workspace}.rs`; `coeus-ops/src/resize.rs` (502) is now `resize/{mod,taps,forward,backward}.rs`; `coeus-ops/tests/alloc_budget.rs` (501) keeps the allocator harness and moves the five kernel budgets to `alloc_budget/kernels.rs`. The scan now reports 1 oversized file.
- Residual: `coeus-ops/src/backend_ops/traits/conv.rs` (645 lines) holds the `ConvOps` trait, whose twelve rank-specific default methods carry long signatures, rustdoc and ratcheted `too_many_arguments` expectations; moving them out of the trait changes the public contract, so it waits for a request-struct migration.
- Scope: repo-wide scan for the rest.
- Acceptance: each split file's module retains domain cohesion; no file regresses over 500 lines.
- Next step: migrate the `ConvOps` trait to request structs, then split the trait file; the request-struct migration is a public-contract change and needs its own ADR.

<a id="coeus-accelerator-padops-adoption"></a>
## COEUS-ACCELERATOR-PADOPS-ADOPTION — Adopt hephaestus's PadOps<D, T> seam

- Status: blocked; priority: architecture; [patch]; owner: unclaimed.
- Outcome: `coeus-accelerator` consumes hephaestus-core's `PadOps<D, T>` seam once merged, instead of any local padding duplication.
- Blocker: hephaestus PadOps seam not yet merged upstream (re-open trigger: hephaestus-core PadOps<D,T> lands).
- Scope: `coeus-accelerator` padding call sites.
- Acceptance: coeus-accelerator padding routes through the upstream seam; no duplicated padding logic remains locally.

