# Coeus Development Backlog

Open-work queue only. Delivered items are compacted away at the commit/PR that
closed them (`git log --grep='^Item:'` or the cited PR); this file never
restates their play-by-play. Priority is one of {correctness, architecture,
verification, tightening, feature}.

<a id="coeus-pow-negative-integer-gradient"></a>
## COEUS-POW-NEGATIVE-INTEGER-GRADIENT — Preserve the negative derivative coefficient

- Status: todo (implementation and local verification complete; integration pending); priority: correctness; [patch]; owner: `fix/coeus-independent-numerical-contract`.
- Finding: on main `771f8cf6`, `PowNode::backward` negates an already-negative exponent before division. For finite nonzero x and negative integer n, the derivative is `n / x^(1-n)`, so every nonzero contribution has the opposite sign. Existing power checks cover positive/zero/fractional exponents, not negative integers. No active power repair PR owns this case.
- Executed before: the saved exact-main run executes four f32/f64/F16/Bf16 analytical regressions; all four fail, zero skip. At n=-1, input `[-2,-1/2,1/2,2]` and seed `[2,-1,1/2,-2]` produce `[1/2,-4,2,-1/2]` instead of `[-1/2,4,-2,1/2]`.
- Repair: retain `exp_t` as the coefficient; leave the existing signed denominator and native-T arithmetic intact. No provider, trait, manifest, lock, feature or release-version change belongs to this correction.
- Regression scope: exponents -1/-2/-3, positive/negative finite nonzero bases, nonuniform seeds, tracked/untracked execution, contiguous/transposed layouts and all four scalar formats. Expected values are exact representable dyadics, with no tolerance changes.
- Verification checklist: for the isolated patch, corrected focused cases pass 4/4 with zero skips; complete autograd/downstream NN suites pass 685 tests with eight existing skips, including all existing power controls. Strict workspace all-target Clippy, warning-denied autograd/NN docs, 26 doctests with two existing ignores, formatting and ADR indexing pass. All 30-second slow / 60-second termination settings remain unchanged. Required before integration: the registered unchanged tracked push and exact-head hosted checks; their final-head verdict belongs to the owner PR. Keep this item open until the reviewed repair integrates.
- Residual scope: integer-exponent range/rounding and zero/singular-input conventions are separate unverified contracts; this item does not claim their correctness. Preserve release owner #528, sparse #533 and all held work. Coeus Miri remains deferred and unverified.
- Acceptance: analytical values/seeded derivatives pass without changing coverage or 30-second slow / 60-second termination limits; existing positive, zero and fractional power tests stay green; final draft head and remaining gates are recorded before integration. Main's automatic publisher is a merge side effect, so a passing native patch alone does not authorize a stale-version release.

<a id="coeus-release-package-resolution"></a>
## COEUS-RELEASE-PACKAGE-RESOLUTION — Unblock complete 0.13.0 package verification

- Status: blocked; priority: verification; [patch]; owner: [PR #528](https://github.com/ryancinsight/Coeus/pull/528).
- Outcome: all 15 intended 0.13.0 libraries pass native, lint, docs, blocking API, standard package and required hosted gates before normal integration/publication.
- Current source reconciliation: merge exact main [9fc433b0](https://github.com/ryancinsight/Coeus/commit/9fc433b032134d79a52b9294116714737c787d47), including merged #553–566 and board work through #569, into preserved repair [earlier mean-repair checkpoint](https://github.com/ryancinsight/Coeus/pull/528) in a separate isolated checkout. Also retain later documentation-only main [afb9ba9e](https://github.com/ryancinsight/Coeus/commit/afb9ba9ea0f31ab6f4155193d364d34314f0f997) (#570–577: backlog and book only); executable source and graph stay identical to the validated source. Keep scalar vocabulary deletion, cross-host seams, CPU pooling/unfold delegation, typed einsum errors and new gradient checks. Preserve sparse #533 and other open owners. Main's scalar-vocabulary ADR owns 0081; this unpublished mean ADR now uses 0084 after canonical main allocates 0082 to fusion; 0083 remains reserved for the preserved sparse continuation.
- Ownership and graph check: 151 checked upstream owner files, every Git provider record, all 231 registry records and all incoming backlog entries remain identical to that main. Hephaestus 0.22.1 selects b03e9176; FFT 0.29.1 selects f266bb0b. No unpublished factory or range candidate is consumed.
- Reproduced on exact current main: six unchanged analytical cases run, four pass and two fail, zero skip. F16 length-65536 one-hot mean returns bits 0x0000 instead of 0x0100; F16 minimum positive is 0x0040 instead of canonical normal 0x0400. Integer mean is already rejected upstream after #554/#562; retain that contract and add public/direct MeanOp compile-fail coverage rather than duplicate its owner.
- Numerical repairs: use canonical directly rounded reciprocal-count scaling in forward/backward and the provider's named F16 minimum-normal constant. Ordinary f32/f64/F16/Bf16 tracked/untracked, seeded backward, strides, empty NaN and all 65536 F16 gradients remain covered. All six focused cases pass within the full native suite at [validated mean-repair checkpoint](https://github.com/ryancinsight/Coeus/pull/528); both integer compile-fail doctests also pass.
- Package graph reconciliation: retain the 15 unchanged downstream NN convolution/pooling gradchecks. The current-main build additionally exposes an NN cross-entropy gradcheck import in autograd; that unchanged consumer now lives downstream as well, preserving its dual-scalar closure, fixtures and tolerances without restoring the package cycle. All 16 downstream checks pass in the current-source full native suite; their computed-value assertions remain intact.
- Historical evidence only: earlier aligned source [packaging checkpoint](https://github.com/ryancinsight/Coeus/pull/528) compiles all 15 ordinary normalized 0.13.0 archives after reproducing 453 exact-main #539 errors against stale registry 0.12.0 interfaces. Source [earlier mean-repair checkpoint](https://github.com/ryancinsight/Coeus/pull/528) passes 1510 native tests with eight existing skips, 166 doctests with two existing ignores, strict lint/docs/fmt/ADR and all 15 actual baseline/current API graph pairs. Those results do not establish acceptance of current reconciled source; keep baseline locks and repeat final package/API/hosted gates after factory integration.
- Release checklist: current-source native, lint, docs and integer compile-fail checks complete at [validated mean-repair checkpoint](https://github.com/ryancinsight/Coeus/pull/528): 1539 native passes with WGPU/Metal devices required and eight existing skips; 172 doctest passes with two existing ignores; strict workspace all-target Clippy under the unchanged CI feature selection, warning-denied docs, formatting and ADR indexing pass. All-feature locked metadata resolves; compiling all features on this Mac stops at the provider's explicit Linux ROCm requirement. The first sandboxed full run lacks loopback/device access; the unchanged host-access run passes after normal approval. No test or deadline changes. Canonical factory/version decision and final API, standard-package, exact-head hosted/backend/release checks remain pending. Remote draft stays [preserved draft #528](https://github.com/ryancinsight/Coeus/pull/528) and unpublished; the historical blocked source is not retried.
- Acceptance: preserve fixtures, tolerances, coverage, baselines and the 30-second slow / 60-second termination policy. Coeus Miri is deferred and unverified; CUDA/ROCm hardware and the full Python wheel matrix remain unverified. Check normal merge publisher and approved automatic documentation deployment results before closing this item.

<a id="coeus-landed-cast-gate"></a>
## COEUS-LANDED-CAST-GATE — Reconcile moved main code with the added-line cast guard

- Status: blocked; priority: verification; [patch]; owner: [PR #528](https://github.com/ryancinsight/Coeus/pull/528).
- Outcome: the unchanged tracked push gate accepts the integration without a baseline increase, skipped check, public-const regression or duplicated conversion kernel.
- Standard repairs retained: implicit Arc unsizing replaces the explicit cast; const little-endian byte decoding preserves every packed u64 bit pattern on every target byte order. Both public decoder signatures retain const. Boundary regressions exercise 408 const-decoded pairs, signs, normal/subnormal boundaries, midpoint ties, overflow and NaNs; earlier actual-source const evaluation passes Rust 1.95.
- Remaining conflict: `op_tags.rs:201:52` narrows every admitted binary64 encoding to the provider's f32 ABI inside public const `parameter_pair`. Native narrowing is const, but the unchanged guard rejects that `as` expression outside a sanctioned conversion module. Standard From/TryFrom is absent (E0277), the locked FloatElement runtime factory is non-const (E0015), and the inherent F32 factory is absent (E0599). Do not drop const, restrict inputs, duplicate the kernel or add suppression. This source and the tracked push hook are unchanged from the earlier reproduced blocker; no push retry occurs.
- Provider prerequisite: [Eunomia #182](https://github.com/ryancinsight/eunomia/pull/182) supplies canonical `const F32::from_float(f64)`. Its prior local/hosted native, NumPy, strict API, MSRV and validation-only package/release gates pass, but its version-policy decision remains pending. New-public-item policy selects facade 0.11; a human exception would permit 0.10.1 and avoid coordinated range updates. Do not choose that exception, consume the unpublished factory or merge its automatic publisher before the answer.
- Separate prerequisite resolved: Atlas #542 / Leto #365 and planning closures #543 / #370 are merged; Python 0.46.1 is published with verified registry/provenance/test receipts. That linkage repair does not resolve the Eunomia version decision. Preserve paused range drafts and other provider owners.
- Acceptance: after the prerequisite, rerun native/const, strict lint/docs, actual package/API and exact-final-head hosted gates. The Coeus draft/release stays open, Miri stays deferred/unverified, and no coherent final cohort release is claimed.

<a id="coeus-count-conv-spelling"></a>
## COEUS-COUNT-CONV-SPELLING — Count-typed spelling for `from_f64(n as f64)` sites

- Status: todo; priority: tightening; [patch].
- Finding: the COEUS-SCALAR-VOCAB slices (closed) retargeted count conversions to `<T as FloatElement>::from_f64(n as f64)`; the item's "Also" clause (migrate them to `from_count`/`try_from_count`) was not executed — `from_count` does not exist, and the exactness contract above 2^53 (exact vs rounded vs rejected) is undecided. Values are correct today (`n as f64` is exact below 2^53); only the spelling misstates intent.
- Scope: decide the contract, add the spelling (upstream if it belongs to eunomia), migrate the ~50 count sites.
- Acceptance: every count-typed conversion names the count spelling; values unchanged; gates green.
- Next step: decide exact-vs-rounded-vs-rejected for counts above 2^53.

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

<a id="coeus-device-test-build-cost"></a>
## COEUS-DEVICE-TEST-BUILD-COST — Attribute device test compilation

- Status: todo; priority: verification; [patch].
- Outcome: bound device-test compilation while preserving the full scalar/backend matrix.
- Scope: CPU test placement and monomorphization in existing integration harnesses; no assertion, scalar, workload or runtime-budget reduction.
- Evidence: ownership RED compilation took 13m06s while native execution took 7.965s; `wgpu_ops` rustc was the last compiler in the integrated attempt and also instantiates CPU-only scalar matrices.
- Current attribution: [PR #611](https://github.com/ryancinsight/Coeus/pull/611), run 39325265218, reaches the unchanged 25-minute limit in all three native shards before test execution. Every shard compiles the same graph; the Nextest partition applies after compilation. Sccache reports 68 non-cacheable test/bin compiler calls per shard. Excluding the `coeus-wgpu` workspace member still includes Hephaestus WGPU through the Metal provider's default decomposition features.
- Controlled source for [PR #611](https://github.com/ryancinsight/Coeus/pull/611): the Mac cold native build takes 256 seconds, with the `ops` test compiler peaking at 8.66 GiB. A private three-way operation-harness split preserves all 1,410 test identities and ignored states but takes 274 seconds and peaks at 8.79 GiB; it is not integrated. Mac measurements do not establish Linux memory pressure or OOM.
- Candidate: the four byte-identical negative-power test bodies now load through the documented canonical autograd harness. All 1,410 case identities and ignored states remain present; all 357 autograd native cases pass with eight existing skips. Attributed autograd compiler CPU falls from 452 to 326 seconds, removing one integration compiler invocation. Whole-build wall time is 264 versus 256 seconds and the unchanged `ops` compiler still dominates; no full-build speedup or deadline closure is claimed. Renew independent review and require the actual final-provider head to pass the unchanged 25-minute hosted gate. Keep all compiler profiles, features, test coverage, shard count and limits unchanged.
- Acceptance: compiler timing/codegen evidence identifies the dominant work; an evidence-backed structural change preserves the enumerated test matrix and lowers its attributed compile cost, or records why the proposed partition does not help.
- Needs: [kernel output correction](#coeus-device-output-ownership); keep compiler flags and shared target policy fixed for comparison.

<a id="coeus-provider-resolution-2026-09-07"></a>
## COEUS-PROVIDER-RESOLUTION-2026-09-07 — Restore fresh provider resolution after upstream alignment

- Status: todo; priority: verification; [arch].
- Outcome: fresh lock generation and default/all-feature locked resolution agree, including against current upstream (not only offline idempotence).
- Evidence: a frozen tree regenerates the same offline lock twice and both activation checks pass — this proves closure/idempotence, not remote freshness against the current upstream state.
- Bridge renewal: after the published Apollo 0.29.3 / Leto 0.46.1 / Hephaestus 0.22.2 / Hermes 0.9.1 prerequisites, ordinary targeted Cargo update renews 38 first-party Git records, including Aequitas 0.3.1, Mnemosyne-memory 0.10.1 and Moirai 0.10.0. The subsequent [PR #613](https://github.com/ryancinsight/Coeus/pull/613) reconciliation preserves its Moirai runtime/async 0.10 minimum and complete registry renewal rather than replacing the concurrent owner graph. Acceptance requires final locked default/all-feature resolution, standard packages and actual-head gates; their revision-scoped outcomes belong to the external review packet.
- Current isolated reconciliation: actual merged #614 (`e8b429ff`, feature `9b768894`) is retained in the private 0.13 history. Its CUDA real/stub and WGPU `FusionScalar<L>` edits remain byte-identical; all five Hephaestus edges require 0.24. Preserve actual #613 ancestry, strict Moirai floor, numerical/NN/linkage repairs, every other private Rust file and the owner registry renewal. Leto #376 candidate `b3d744a9` has exact-head numerical/matched-performance acceptance; its actual merge is `5051cd95`. Both 0.46.3 registry archives verify against accepted source; ordinary canonical lock renewal selects that merge and Hephaestus 0.24 at `746553d0`, preserving all 235 registry/34 unrelated Git records. Ordinary package/API verification and material resulting-head review remain gates. Do not copy #614's defective 0.46.2 lock.
- Remaining boundaries: #497/#500 and Athena owner work, first publication, future 0.47 policy, common PyO3 migration, physical CUDA/ROCm/platform and absolute application performance remain separate. The denied push stays paused; Coeus Miri remains deferred and unverified.
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
- Scope: `coeus-ops/src/unary/math.rs` (69 ops via `.expect("<opname>")`), `coeus-autograd/src/ops/activation/{ext,gelu,math}.rs`, `coeus-cuda/src/backend/mod.rs:21,101-120`, `coeus-accelerator/src/storage.rs:56,63`, `coeus-accelerator/src/reduction.rs:279,285`, `coeus-tensor/src/tensor.rs:438,452`, `coeus-core/src/storage/cpu.rs:131`.
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

<a id="coeus-accelerator-padops-adoption"></a>
## COEUS-ACCELERATOR-PADOPS-ADOPTION — Adopt hephaestus's PadOps<D, T> seam

- Status: blocked; priority: architecture; [patch]; owner: unclaimed.
- Outcome: `coeus-accelerator` consumes hephaestus-core's `PadOps<D, T>` seam once merged, instead of any local padding duplication.
- Blocker: hephaestus PadOps seam not yet merged upstream (re-open trigger: hephaestus-core PadOps<D,T> lands).
- Scope: `coeus-accelerator` padding call sites.
- Acceptance: coeus-accelerator padding routes through the upstream seam; no duplicated padding logic remains locally.

<a id="coeus-scatter-add-alloc-ci-only"></a>
## COEUS-SCATTER-ADD-ALLOC-CI-ONLY — scatter_add allocation-count regression on hosted CI only

- Status: todo; priority: correctness; [patch]; owner: unclaimed.
- Outcome: `scatter_add_allocation_count_is_independent_of_index_size` (coeus-ops/tests/alloc_budget.rs) passes on hosted CI, not only locally.
- Evidence: hosted `Tests` job on PR #425 (run 36285453725, job 108526331877) fails with small=6, large=10 allocations for shapes [4,8,4]/[16,32,16]; reproduced locally on Windows with the identical revision and the same nextest invocation and it PASSES (1 passed; 0 failed). This is a landed, pre-existing defect on `main` (predates every PR in this session's reconciliation) — `main` at the failing revision was `db144251`, before any of PRs 421-428 merged.
- Investigated and ruled out: `scatter_add`'s own flat-index loop is allocation-free by design (no per-element or per-slice buffer, per its own code comment); `Tensor::alloc_on`/`CpuStorage::new` are single-call allocations independent of `numel`; `to_contiguous()` short-circuits to `self.clone()` for already-contiguous inputs (both test tensors are freshly built and contiguous), so it never reaches `coeus_cpu::contiguous_values`. Falsified the "parallel-dispatch thread count crosses a core-count threshold" hypothesis: pinned the local repro to a 2-core `ProcessorAffinity` mask (matching the hosted runner's core count) via a `Start-Process`-launched `cargo test` — still passes locally (1 passed; 0 failed). `leto-ops`'s documented parallel thresholds (`PARALLEL_MIN_ELEMENTS` 65536, `PARALLEL_MIN_REDUCTION_OUTPUTS` 32768, `PARALLEL_MIN_MATMUL_MACS` 262144) are all far above this test's tensor sizes (max 8192 elements), ruling out its elementwise/reduction/matmul parallel paths regardless of core count.
- Critical finding: this is FLAKY on hosted CI, not a deterministic regression. `main`'s own push-triggered `CI` run at the identical commit (`db144251`, run 36286664346, job 108528635313, 2026-09-27T02:03) shows this exact test **PASS**, while PR #425's `pull_request`-triggered run at essentially the same tree (run 36285453725, ~2026-09-27T01:46) shows it **FAIL** with small=6/large=10. Six concurrent copies of the compiled test binary, each pinned to a 2-core `ProcessorAffinity` mask, all pass locally (`multi_{0..5}.log`) — a scheduler-race reproduction attempt per the "two-core pinning reproduces CI scheduler flakes" pattern also did not reproduce. Per policy, a flaky test is root-caused, never silently retried: the counting allocator (`alloc_budget.rs`'s `#[global_allocator]`) is process-wide, so any concurrent or lazily-initialized background allocation (a `OnceLock`/thread-pool first-touch in a dependency, unrelated to `scatter_add` itself) racing with the small/large calls would produce exactly this nondeterministic +4 without any code-path difference in `scatter_add`.
- Next step: add allocation-site attribution (capture a backtrace or a monotonic sequence id per counted allocation) to `alloc_budget.rs`'s global allocator so a failing run identifies *which* allocation is extra, rather than only the count; run that instrumented binary repeatedly on the hosted runner (matching its exact core count and load) until it flakes again.
- Blocks: PR #425 and any other PR whose `Tests` job runs the full workspace suite on hosted CI.

<a id="coeus-gradcheck-generic-instantiation"></a>
## COEUS-GRADCHECK-GENERIC-INSTANTIATION — Generic-instantiation gradcheck coverage

- Status: todo; priority: verification; [patch]; owner: unclaimed.
- Outcome: every finite-difference check in `crates/coeus-autograd/tests/autograd/gradcheck/` runs at both `f64` (sensitivity oracle) and `f32` (instantiation coverage, per `standards`: Generic Instantiation Coverage), plus FD coverage for the ops that had none.
- Delivered: `mod.rs`'s shared fixtures (`Sampler`, `tensor`, `weighting`, `weighted`) are generic over `GradcheckScalar` (`Float + leto_ops::Scalar + coeus_cpu::RealScalar` plus the `CpuAddressableStorage` bound gradcheck itself needs); the module doc states the two-role f64/f32 rationale. New FD coverage, each a generic fn instantiated at both types from a single `#[test]`: `add`/`sub`/`div`/`remainder`/`maximum`/`minimum`/`neg` (`arithmetic.rs`); `max_pool1d/2d/3d`/`avg_pool1d/2d/3d` (`pooling.rs`, via `coeus-nn` `Module` layers, new dev-dependency); `conv1d`/`conv3d`/`conv_transpose1d/2d/3d`/`unfold1d/2d`/`fold1d/2d` (`conv.rs`, via `coeus-nn` `Module` layers). Converted to dual-instantiation: `normalization.rs`, `core_ops.rs`, `attention.rs`.
- Remaining: (1) convert the 4 remaining existing check files (`activation`, `losses`, `reduction`, `shape`) to dual-instantiation; (2) FD coverage still missing: `ctc`, `dropout` (fixed mask), `sparse_matmul(_coo)`, `transpose_2d`, `index_put`, `rotate_half`, `linear_interpolation` (`cross_entropy` and `embedding` already have coverage in `losses.rs`).
- Acceptance: every listed op has a generic FD check instantiated at both types; all 11 existing files are dual-instantiated; a check that fails at `f32` under gradcheck's own derived `ε^(2/3)` bound is root-caused, never given a widened tolerance.
- Next step: convert `activation.rs` next (largest remaining, ~50 tests) then `losses`/`reduction`/`shape`, each its own PR; then the remaining missing-coverage ops as dependency-ordered per-family PRs.
