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
- Note (2026-10-08, uncommitted): migration complete — 4 methods deleted from trait + 5 impls; all src/tests/doctests migrated (full-path `coeus_core::FloatElement` bounds, never imported: E0034-proof; counts to `from_count`/`try_from_count`; `i32::from_f64` oracle to integer math). Gates: `check --all-targets` + `clippy -D warnings` + `fmt` clean; nextest 1182/1183 (sole fail = `dependency_policy`, peer bytemuck-manifest dirt, environmental); doctests 99/99. Fixed latent `u64` Sign `ZERO-ONE` const-overflow via `wrapping_sub` (matched release wrap). Excluded peer-broken crates: `coeus-cuda` (stub), `coeus-metal`/`coeus-rocm` (missing `Pod` bound in peer CrossProduct impl).
- Note (2026-10-10, worktree audit — no code change): stack sits on `89cb08b0` (merge #508 `feat/coeus-realscalar-bounds`, 10-07), which predates this work — no mid-session HEAD move, no supersede (#508 = RealScalar bound propagation over 38 files, disjoint concern). Provenance: session-window files (10/9–10/10) carry this work's signatures (`try_from_count`, `wrapping_sub`); pre-10/9 dirt (~81 files: `coeus-cuda`/`coeus-wgpu` clusters + `CHANGELOG.md`) is peer state already present during the green gates — untouched. No file modified after the gate run. Local `main` is 4 behind `origin/main`, but upstream is hooks + `Cargo.lock` only (no source overlap), so landing-time rebase is trivial for these paths.
- Note (2026-10-10 takeover sweep): rescue PR #490 (7 days stale, conflicting, green) verified PARTIAL — deletes zero/one/to_f64 but keeps `from_f64`, no count-conv migration; this lane's completed work supersedes it (all four gone + count-convs + green gates). Posted supersede review recommending close of #490 and #488 (claim-only branch). No competing PR from this lane; item lands once from here.

<a id="coeus-rescue-queue"></a>
## COEUS-RESCUE-QUEUE — Complete or close the stranded rescue PRs

- Status: todo; priority: correctness; [patch].
- Outcome: each listed rescue PR is completed onto current `main` (ported, verified, merged) or closed once its diff resolves empty against `main`.
- Scope: head refs of the PRs below; no board or code change until a PR is claimed.
- PRs (draft parking records, unclaimed; head ref is the only durable copy; inventory refreshed 2026-10-10 — #464 is closed, replaced by the #475/#473 pair):
  - `ryancinsight/coeus#469` (`rescue/audit-coeus-20260928`, 09-30, conflicting, red): 177 files, +3858/-6950 (board's "68 files" count was stale). Consumed only by scoped audit findings, never whole.
  - `ryancinsight/coeus#475` / `#473` (`rescue/coeus-fallible-storage-core-closure/lane`, 10-01, conflicting, red): ~700 files each, ±26k/±16k lines. Full-migration snapshots; consumed leaf-by-leaf by the FALLIBLE board items' owners only — never take over whole.
  - `ryancinsight/coeus#494` (`rescue/coeus-mean-stack`, 10-03, conflicting, was green) + `#493` (`feat/coeus-float-mean-entry`, 10-03, conflicting, was green): float-bound-mean migration; 44 / 15 conflicting files measured read-only 2026-10-10 (nn loss/norm/pool/transformer, autograd losses, reduction traits+impls, leto dispatch, vendor tests). No board item owns it; review posted on #494 (rebase scope, split recommendation, test_hooks.py fallout note). Keep parked until a scoped item with a quiet window takes it.
  - `ryancinsight/coeus#490` (`rescue/coeus-scalar-vocab`, 10-03, conflicting, green) + `#488` (claim-only): superseded — partial (3/4 methods, no count-convs) vs this lane's completed COEUS-SCALAR-VOCAB. Supersede review posted; recommend close.
  - Live, not stale — do not take over: `#458` (`fix/coeus-einsum-panic`, pushed 10-08, earlier claim 09-29; reviewed 2026-10-10 with dot-length gap + sequencing vs this lane's convergent hunks, see COEUS-EINSUM-PANIC note), `#509` (CI sharding, open 10-07, peer-active).
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
- Scope evidence 2026-10-10 (measured, implementation reverted — see below): exactly 4 `ComputeBackend` implementors (`SequentialBackend`, `MoiraiBackend`, generic `HephaestusBackend<P>`, test `CountingBackend`); vendor crates reuse the generic hephaestus impl. 80 backend-method call sites + 449 tensor-ctor sites + 330 materialize sites across 8 crates (coeus-autograd 212 ctor sites alone). The trait change was prototyped through coeus-core green (fallible `CpuStorage` ctors, `Overflow`/`Storage` errors, `Result`-returning tests + 3 new failure tests) and coeus-hephaestus migrated, then reverted: the signature change cannot land green without the full ~1000-site closure, which exceeds one increment. Landable sequence: (1) this leaf's trait+impls, (2) tensor ctors, (3) callers outward — all in one branch, or pre-land the two joint-error enums first. Design notes: `StorageMut::make_unique`/`as_mut_slice` stay infallible (sealed contract; COW detach keeps its documented expect, matching the hephaestus precedent); `to_backend_on` needs a `TransferError<Src, Dst>` joint enum (src+dst errors unrelated); `arange/linspace` need a `Backend+CountRangeError` joint enum; keep the leto-materialization expects and geomspace asserts (orthogonal validation, out of scope).
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
- Next step: migrate every implementor and caller in dependency order (ADR drafted: [0079](docs/adr/0079-fallible-provider-initialization-transfer.md), Proposed — converge onto existing `try_device()`, delete `device()` after its 11-file caller closure empties; 5 in-scope expects, lock/test expects out of scope; CUDA vendor-body deletion needs peer-lane coordination).
- Note: deliberately outside the closed native-comparison-provider items — this is the separate public failure-boundary migration.

<a id="coeus-sibling-named-crates"></a>
## COEUS-SIBLING-NAMED-CRATES — Crates named after their dependencies

- Status: todo; priority: architecture; [major] [arch] — ADR first (Foundation phase).
- Finding: `coeus-leto` and `coeus-hephaestus` name stack members; a stack member's name must never appear in another member's crate/module/feature/item names (naming prohibition).
- Scope: the two sibling-named crates only; `coeus-cuda`/`coeus-rocm`/`coeus-metal`/`coeus-wgpu` name vendors, and their thinning is already decided by [ADR 0071](adr/0071-provider-owned-accelerator-backends.md) — out of scope here.
- Recommended target: name each crate for the concern it owns (the CPU array adaptation, the accelerator adaptation) so the manifest, not the crate name, records the provider. Rename is a mechanical transform over every call site in one change, no re-export bridge.
- Acceptance: an ADR with the recommended option is drafted first ([arch]); the rename then executes as one commit touching every call site, published-name migration note included ([major]).
- ADR drafted 2026-10-10 ([0080](docs/adr/0080-concern-owned-adaptation-crate-names.md), Proposed): `coeus-leto` → `coeus-cpu`, `coeus-hephaestus` → `coeus-accelerator`; measured blast radius ~332 + ~61 code references, 89 doc mentions, 3 out-of-repo manifests (all on `coeus-leto`); no yank, README/CHANGELOG migration notes on the final old-name versions.
- Next step: accept the ADR, then execute the mechanical rename transform.

<a id="atlas-coeus-backend-045"></a>
## ATLAS-COEUS-BACKEND-045 — Unseal ComputeBackend and unify matmul (residual)

- Status: todo (matmul unification landed); priority: architecture; [minor] [arch].
- Outcome: matmul reaches every provider through one generic dispatch; no vendor crate carries a matmul kernel. Delivered: `ComputeBackend` unsealed, `coeus-hephaestus::matmul` added over `DenseProductOps`, `CudaBackend`/`WgpuBackend` migrated, `MatmulProvider<f32>` declared for Metal/ROCm.
- Residual 1 (closed 2026-10-10): the re-open trigger fired — upstream
  `PoolingOps<D,T>`/`SlidingWindowOps<D,T>` seams cover all five
  hephaestus vendors, `coeus-hephaestus` binds them generically
  (`pooling/implementation.rs`, `unfold_fold/implementation.rs`), and all
  four vendor markers (`Cuda`/`Wgpu`/`Metal`/`RocmProvider`) declare
  `PoolingProvider` + `UnfoldFoldProvider`. The forked per-vendor kernels
  are deleted (only the trait, `cpu_impl`, and the hephaestus binding
  define pool ops). Verified: CUDA parity pool 11/11, unfold 34/34,
  fold 35/35 on RTX 5080; WGPU parity pool 11/11, unfold 2/2 on Vulkan;
  `coeus-metal`/`coeus-rocm` compile (hardware runs need Apple/AMD CI).
- Residual 2 (evidence 2026-10-10, CI wiring remains): provider-matmul
  vs CPU-reference parity is green on local hardware — CUDA 2/2
  (`test_cuda_parity_matmul_2d`, `test_cuda_parity_batched_matmul`,
  RTX 5080) and WGPU 6/6 (Vulkan) — so device-side equivalence holds
  where adapters exist. Remainder: (a) a GPU-capable CI job for
  regression coverage, (b) Metal/ROCm equivalence runs (no Apple/AMD
  adapter on this host).
- Residual 3: `cargo semver-checks` reports a pre-existing major break in `coeus-hephaestus` against the published 0.10.0 baseline (five associated types added since publish) — needs a version decision before the next release; unrelated to this item's own change.
- Links: [ADR 0066](adr/0066-computebackend-unseal-matmul.md).

<a id="coeus-bytemuck-residue"></a>
## COEUS-BYTEMUCK-RESIDUE — one concept, two markers

- Status: done 2026-10-10; priority: architecture; [patch].
- Finding: `coeus_core::Scalar` carries both `bytemuck::Pod` and Eunomia's `Pod` as supertraits. Eunomia is the stack's datatype law and owns this concept; bytemuck is retained only because `Scalar::has_zero_bit_pattern`'s default body calls `bytemuck::bytes_of`.
- Acceptance: move that body to `eunomia::layout::bytes_of` and drop the bytemuck supertrait, so one marker states the contract.
- Blast radius (checked first, per scope): zero external implementors possible — `Scalar` is transitively sealed via `CpuUnaryDispatch: private::Sealed`, and all four `impl Scalar` sites live in coeus-core. Sole in-workspace consumer of `Scalar: bytemuck::Pod` is `coeus-tensor` checkpointing (3 call sites), which now states `+ bytemuck::Pod` explicitly at its two generic sites. `cargo semver-checks` reports the identical 3 pre-existing majors on base and after (enum discriminant, missing methods from other in-flight work) — this change adds none.
- Delivered 2026-10-10: body moved to `eunomia::layout::bytes_of` (behavior pinned by the existing `has_zero_bit_pattern` value tests), bytemuck supertrait + `use` + coeus-core dependency dropped, docs synced. Gates: coeus-core lib 13/13, dtype 2/2, coeus-tensor 5+58+5, clippy/fmt clean. One pre-existing failure unrelated: `core_ops policy::dependency_policy::resolved_normal_...` fails on the peer-dirty lockfile (`cargo tree --locked`) identically with the manifest restored.
- Note (escaped defect found 2026-10-10 by workspace coherence check, uncommitted): blast-radius claim "sole in-workspace consumer is checkpointing" missed `coeus-rocm`/`coeus-metal` `CrossProductProvider` impls (`backend/provider.rs`, `type Operations = HostCrossProductOps` requires `T: bytemuck::Pod` via hephaestus `CrossProductOps`) — both fail E0277 since `Scalar` no longer implies it. Remedy is the same explicit `+ bytemuck::Pod` bound (plus dep check); left to the slice owner / peer lanes already dirty in `cross_product/` and both vendor crates. Independent of the hephaestus-cuda peer breakage blocking the rest of the workspace check.

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
- CUDA: `coeus-cuda` already uses the Hephaestus CUDA provider with no Cutile/`cuda-bindings` dependency; add the missing provider/backend binding without duplicating dispatch. (Remainder: untouched — peer-active area.)
- Metal (done 2026-10-10): `StaggeredProvider for MetalProvider`
  (`type Operations = MetalStaggered3DOps`) plus
  `tests/staggered.rs` invoking the shared ownership oracle; fails
  first without the impl (E0277), compiles after; clippy/fmt clean.
  Not run on hardware (no Apple adapter; `require_device` skip
  verified, `HEPHAESTUS_METAL_REQUIRE_DEVICE=1` armed for CI).
- ROCm (done 2026-10-10): the external block below is lifted —
  `RocmStaggered3DOps` exists upstream with its conformance test — so
  `StaggeredProvider for RocmProvider` is bound the same way with its
  own `tests/staggered.rs`; same compile-gated, skip-verified state
  (no AMD adapter; `HEPHAESTUS_ROCM_REQUIRE_DEVICE=1` armed for CI).
- Blocked (external, lifted): ROCm required upstream staggered kernels
  in Hephaestus; they have since landed (`RocmStaggered3DOps` +
  `assert_staggered_3d_contract`), and the ROCm binding above consumes
  them.
- Acceptance: exercise the shared gradient/divergence, axis, adjoint, and invalid-grid oracles on real CUDA and Apple Metal devices; preserve typed acquisition failures. (Metal/ROCm oracle runs still need their adapters; CUDA binding + runs remain.)

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
- Note (2026-10-10, verified + f64 gap closed, uncommitted): the product half predates this turn — `coeus-ops` already delegates to the one `Tensor::{linspace,logspace,geomspace}_on` implementation (no duplicate remains), and the tensor implementation is native-`T` throughout: zero `f64` mentions (grep), `Float::{powf,abs,signum}` are inherent-primitive calls, `coeus_leto::from_shape_fn_values` is generic over `T`, counts via exact `from_count`/`try_from_count`. Only gap was the acceptance's "per dtype": value-semantic tests existed for f32 (endpoints, base-10 ratios, doubling, n=1) and integers (i32 leto-differential + i8 `CountRangeError`) but not f64 — added the three f64 mirrors (`linspace_endpoints_inclusive_f64`, `logspace_base10_f64`, `geomspace_doubling_f64`) in `coeus-ops/src/constructors.rs` tests. Gates: 7/7 constructors, 132/132 `coeus-ops --lib`, clippy `-D warnings` + fmt clean, lock reverted. Ready to close.

<a id="coeus-expect-swallowed-results"></a>
## COEUS-EXPECT-SWALLOWED-RESULTS — Propagate typed errors instead of `.expect()`

- Status: todo; priority: correctness; [patch]; owner: unclaimed.
- Outcome: public elementwise ops and device/alloc paths return typed `Result` instead of panicking via `.expect()` on backend/allocator failure.
- Scope: `coeus-ops/src/unary/math.rs` (69 ops via `.expect("<opname>")`), `coeus-autograd/src/ops/activation/{ext,gelu,math}.rs`, `coeus-cuda/src/backend/mod.rs:21,101-120`, `coeus-hephaestus/src/storage.rs:56,63`, `coeus-hephaestus/src/reduction.rs:279,285`, `coeus-tensor/src/tensor.rs:438,452`, `coeus-core/src/storage/cpu.rs:131`.
- Acceptance: no bare `.expect()` on backend-Result in listed files; failure surfaces as a typed error to the caller; existing behavior tests still pass.
- Next step: start with `coeus-ops/src/unary/math.rs` (highest count), thread the `Result` through its callers.
- Note (2026-10-10, scoping — no code change): this item has no one-turn vertical slice; it needs the FALLIBLE-API migration machinery, not an increment. Measured: `math.rs` holds 43 public fns (30 with `.expect`), called from 126 sites across 30 files (autograd activations/losses/reductions, cuda/wgpu parity tests, ops/tensor tests) — and every consumer signature is bare-`Tensor`/`Var` (e.g. autograd `mul`/`matmul`), so threading `Result` is viral through `coeus-nn` to the Python boundary. The "small" files are no better: `tensor.rs:438,452` sit in bare-`Tensor` `to_backend_on` (callers span peer-held cuda/wgpu/hephaestus), and `cpu.rs:131` is allocator-failure fallibility owned by peer-claimed FALLIBLE-STORAGE-CORE (#464 leaves). Recommend decomposing per-op-family behind the fallible-API migration, not per-file. Untouched (peer overlap + scale).

<a id="coeus-einsum-panic"></a>
## COEUS-EINSUM-PANIC — einsum rejects invalid subscripts via typed error

- Status: todo; priority: correctness; [patch]; owner: unclaimed.
- Outcome: caller-supplied einsum subscript strings that are malformed or mismatched no longer panic.
- Scope: `coeus-autograd/src/ops/shape/util/einsum.rs:87,207` (stale lines; the work below covered both einsum layers — the acceptance required it).
- Acceptance: a negative test with a malformed subscript returns a typed error, never panics.
- Next step: replace the panicking parse/validation with a typed error path.
- Note (2026-10-10, uncommitted): delivered both layers — the tracked fallback delegates to `coeus_ops::einsum`, which itself panicked on unknown patterns, so tracked-only conversion could not meet "never panics". `BackendError` gains `EinsumOperandCount`/`EinsumRank`/`EinsumDim`/`EinsumUnsupportedPattern` (non_exhaustive, additive); the delegate converts all asserts/panics (incl. an `n - 2` swap underflow on empty subscripts) with no signature change (`B::Error: From<BackendError>`), and its false "general elementwise-loop fallback" doc now states typed rejection. Tracked `EinsumError` gains `OperandCountMismatch`/`RankMismatch`/`ForwardFailed` (transparent delegate-failure wrapper; the enum stays non-generic), converts its asserts plus the swap guard and previously-unguarded `shape()[0]` indexing, and maps both `.expect("einsum")` sites. Python already maps `Err` to `PyValueError`, so malformed subscripts from Python now raise instead of panicking across the boundary. Tests: 6 delegate negatives + 5 tracked negatives (malformed-via-fallback, empty, arity, count, rank/dim). Gates: coeus-ops lib 138/138, autograd lib 105/105, autograd_ops 246/246, clippy `-D warnings` + fmt clean, lock reverted. Residual (out of scope — not panics in these files): tracked dim-mismatch still panics inside `matmul`/`mul` internals (owned by those ops' fallibility); see COEUS-EINSUM-DOT-LENGTH for the silent dot truncation. Ready to close.
- Note (2026-10-10 takeover sweep): convergent live work found — PR #458 (`fix/coeus-einsum-panic`, pushed 10-08, earlier claim 09-29) implements the same scope independently (ops `invalid_einsum` Storage-wrap + rich tracked `EinsumError` with subscript context + tests; verified: empty-subscript and `n - 2` swap guard covered there too). Comparison: theirs richer tracked context; mine adds ops typed variants + dot-length + empty-subscript delegate test. Reviewed #458 with findings (dot-length gap, ops-variant suggestion) + sequencing proposal (whoever lands first, other rebases; no competing PR from this lane). This lane's hunks stay uncommitted pending that sequencing — NOT an independent close; the dot-length item below has no overlap and stays live regardless.

<a id="coeus-einsum-dot-length"></a>
## COEUS-EINSUM-DOT-LENGTH — einsum dot rejects length-mismatched vectors

- Status: todo; priority: correctness; [patch]; owner: unclaimed.
- Outcome: `coeus-ops` `"i,i->"` errors (typed `EinsumDim`) instead of silently zipping to the shorter length.
- Evidence: found during COEUS-EINSUM-PANIC (2026-10-10): the dot branch zips both slices and sums, so `[4]·[3]` returns the 3-pair sum without complaint — a wrong answer, not a panic, hence out of that item's scope. The tracked `"i,i->"` routes through `mul`, whose unbroadcastable-shape behavior needs the same decision.
- Scope: `coeus-ops/src/shape/einsum.rs` dot branch (+ tracked `"i,i->"` length check if `mul` broadcasts instead of failing).
- Acceptance: mismatched-length dot returns `EinsumDim`; matched behavior unchanged; negative test pins both layers.
- Next step: check `mul`'s unbroadcastable-shape contract, then add the length check(s).
- Note (2026-10-10, uncommitted): delivered — `mul`'s contract confirmed as panic (`elementwise_binary(...).expect("mul: incompatible shapes")`), so both layers needed their own checks: delegate `"i,i->"` gains the length check via the existing `dim` closure (`EinsumDim{operand 1}`); tracked `"i,i->"` gains rank checks (it had none — 2-D inputs would elementwise-multiply into a wrong scalar) plus the length check via a new additive `EinsumError::DimMismatch` variant. Tests: `einsum_dot_rejects_length_mismatch` + `test_einsum_dot_rejects_length_mismatch` ([4]·[3] → expected 4, actual 3). Gates: coeus-ops lib 139/139, autograd_ops 247/247, clippy `-D warnings` + fmt clean, lock reverted. Ready to close.
- Note (2026-10-10 takeover sweep): verified NO overlap with live PR #458 — its `"i,i->"` still zip-truncates (review Finding 1 posted). This item stays live and independent; layers on top of either panic-scope landing. (Caveat: current hunks assume this lane's `EinsumDim` variant; if #458's Storage-wrap lands first, re-target the check to its `invalid_einsum` + keep the tests.)

<a id="coeus-gradcheck-generic-scalar"></a>
## COEUS-GRADCHECK-GENERIC-SCALAR — Generic finite-difference gradcheck over f32/f64

- Status: todo; priority: verification; [patch]; owner: unclaimed.
- Outcome: the finite-difference gradient-check harness is one generic function/macro instantiated per scalar type (f32 and f64), each with its own analytically derived tolerance, not an f64-only harness.
- Scope: coeus-autograd's gradcheck test infrastructure.
- Acceptance: existing f64 gradcheck coverage still passes; f32 instantiation runs with an f32-appropriate tolerance derived from step size and precision.
- Next step: parameterize the harness's step size and tolerance by `T::EPSILON`, add the f32 instantiation.
- Note (2026-10-10, verified against landed+uncommitted code — no change needed, close on next filing sweep): already delivered. The harness is one generic `gradcheck<T: Float + FloatElement>` (`gradcheck/check.rs`, split from `gradcheck.rs` in the OVERSIZED-FILES work): step `ε^(1/3)` and floor `ε^(2/3)` derive from `machine_epsilon::<T>` (exact halving loop, since `Float` exposes no `EPSILON`; `eunomia::RealField::EPSILON` deliberately not used — dev-dep promotion would exclude reduced-precision types), tolerance `64·floor·|loss|` with the margin rationale documented in `precision.rs`. All 11 integration files instantiate both dtypes (2–46 f32 sites each). `cargo test -p coeus-autograd --test autograd_ops gradcheck`: 148/148 green.

<a id="coeus-gradcheck-missing-ops"></a>
## COEUS-GRADCHECK-MISSING-OPS — Fill finite-difference gradient-check gaps

- Status: todo; priority: verification; [minor]; owner: unclaimed.
- Outcome: every differentiable op has a finite-difference gradient-check test (depends on COEUS-GRADCHECK-GENERIC-SCALAR landing first).
- Scope: add/sub/div/remainder/maximum/minimum/neg, all pooling ops, conv1d/conv3d/conv_transpose1d/2d/3d, fold/unfold, cross_entropy, ctc, dropout (mask-fixed), embedding, sparse_matmul/sparse_matmul_coo, transpose_2d, index_put, rotate_half, linear_interpolation.
- Needs: COEUS-GRADCHECK-GENERIC-SCALAR.
- Acceptance: each listed op has a passing FD gradcheck against its analytical Jacobian within the derived tolerance.
- Next step: work the list in dependency-free batches (e.g. arithmetic ops first, since they share the harness shape).
- Note (delivered uncommitted): coverage audit found every scope op already checked except `cross_entropy` and `embedding`; this batch closes both (`cross_entropy_case` via the `coeus_nn` forward wrapper in `gradcheck/losses.rs`, `embedding_case` with a repeated/omitted-row fixture in `gradcheck/new_ops.rs`, each f64+f32). `embedding_with_padding_idx` excluded with reason: forward ignores `padding_idx` while backward zeroes that row, a projected-gradient semantics FD cannot reproduce. Gates: autograd_ops 249/249, clippy+fmt clean. Scope fully covered; ready to close on commit.

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
- Blocker: hephaestus is adding device dot/cross; bind to it once it lands (re-open trigger: hephaestus device dot/cross merges). Trigger status 2026-10-10: `cross` half landed (`CrossProductOps` in core + CUDA/ROCm/WGPU implementors, host-fold for Metal); `dot` half landed 2026-10-10 — `DotOps` in core + host/WGPU/CUDA implementors with ROCm/Metal mirrors, folded into the 46-param aggregate (ADR 0084, `HEPH-DEVICE-DOT-SEAM` done). Binding scope (`reduction/linalg.rs`) is peer-active; implementation waits only for a quiet lane now.
- Acceptance: no host round-trip for accelerator dot/cross; differential test against the CPU reference.

<a id="coeus-python-elementwise-dedup"></a>
## COEUS-PYTHON-ELEMENTWISE-DEDUP — Consolidate duplicated Python elementwise bindings

- Status: done 2026-10-10; priority: tightening; [patch]; owner: unclaimed.
- Outcome: one generic entry point for elementwise op bindings instead of 28 duplicated wrappers.
- Scope: `coeus-python/src/ops/elementwise.rs` vs `pytensor.rs`.
- Acceptance: one shared dispatch path; binding-level pytest suite still passes.
- Delivered 2026-10-10: new `tensor/dispatch.rs` owns `apply_unary`/`apply_binary` (generic over the `Var` kernel, GIL released, `from_var` wrap); all ~40 `ops/elementwise.rs` free functions are one-liners over it and `unary_dispatch` delegates to it. Side-effect fix: the `scalar_*` free functions previously bypassed `from_var` (raw `PyTensor { inner }`) while the method path honored grad mode — now uniform. Gates: cargo clippy/fmt clean; Rust 6+78 green; pytest pytorch 390 passed + 10 skipped, jax 190 + 3 skipped, mlx 70 skipped (no MLX on Windows), integrity 1 passed. Two transformer-construction tests (`decoder_stack`, `seq2seq`) stack-overflow in the debug wheel — reproduced on the unmodified base wheel, pre-existing and unrelated (transformer path never calls elementwise). Cargo.lock dirt in-tree is peer/environmental (patch-source resolution), left untouched.

<a id="coeus-oversized-files"></a>
## COEUS-OVERSIZED-FILES — Split files past the 500-line target

- Status: todo; priority: tightening; [patch]; owner: unclaimed.
- Outcome: the files currently over 500 lines split into leaf modules by operation family.
- Scope: repo-wide scan (`coeus-python/src/tensor/pyimpl/pytensor.rs` first).
- Inventory 2026-10-10 (the "14 files" count is stale; 5 remain): `pytensor.rs` 933, `coeus-core/src/dtype/traits.rs` 638, `coeus-ops/src/backend_ops/traits/conv.rs` 626 (peer-dirty zone), `coeus-autograd/src/ops/activation/math.rs` 555, `coeus-autograd/src/gradcheck.rs` 554.
- `pytensor.rs` slice probed 2026-10-10 and reverted: PyO3 0.23 permits exactly one `#[pymethods]` block per class — 11 family leaves fail with E0119 conflicting `PyMethods` impls. The only compliant shapes are (a) one registry file with per-method forwarders into logic leaves (~40 shims, against the no-forwarding rule), or (b) a PyO3 upgrade if a newer version supports multiple blocks (unverified). The file is a method registry of one-liners over shared dispatch; recommend exempting pyclass registries from the 500-line target or verifying the upgrade path before retrying.
- `dtype/traits.rs` slice delivered 2026-10-10: split into `dtype/traits/` (`scalar`, `float_ops`, `float`, `int`, `cpu_dispatch`, `op_tags` + `mod` carrying the sealed marker and re-exports; largest leaf 232 lines). All `traits::` paths preserved — zero caller changes; gates: coeus-core lib 13/13, dtype doctests 2/2, clippy/fmt clean, tensor/ops/autograd check green.
- `math.rs` + `gradcheck.rs` slices delivered 2026-10-10: `math.rs` → `activation/math/` (7 op leaves + re-export `mod`; macro call sites keep call-site resolution via `super::super::`); `gradcheck.rs` → `gradcheck/` (`precision`, `error`, `config`, `check` + derivation docs on `mod`). All public paths preserved — zero caller changes; gates: autograd lib 105/105, gradcheck doctest 1/1, clippy/fmt clean. Remaining: `conv.rs` (peer-dirty zone — wait for a quiet lane) and the blocked `pytensor.rs` registry.
- Acceptance: each split file's module retains domain cohesion; no file regresses over 500 lines.
- Next step: split `pytensor.rs` first since COEUS-PYTHON-ELEMENTWISE-DEDUP touches it anyway.

<a id="coeus-cuda-safety-comment"></a>
## COEUS-CUDA-SAFETY-COMMENT — Correct misleading SAFETY comment on CudaBackend::parallel_for

- Status: todo; priority: correctness; [patch]; owner: unclaimed.
- Outcome: the `# Safety`/`// SAFETY:` comment on `CudaBackend::parallel_for` accurately states the invariants it relies on.
- Scope: coeus-cuda backend `parallel_for`.
- Acceptance: comment reviewed against the actual unsafe preconditions; miri/sanitizer coverage unaffected.
- Next step: re-derive the actual safety obligation from the call site and rewrite the comment.
- Note (verified uncommitted, no code change): subject no longer exists — the backend restructure replaced `CudaBackend` with unit struct `CudaProvider` (`coeus-cuda/src/backend/provider.rs:82`), `parallel_for` survives only in coeus-core CPU backends, and the sole remaining cuda SAFETY comment covers `HephaestusStorage` Arc retention (unrelated). Nothing to correct; close as stale.

<a id="coeus-hephaestus-padops-adoption"></a>
## COEUS-HEPHAESTUS-PADOPS-ADOPTION — Adopt hephaestus's PadOps<D, T> seam

- Status: blocked; priority: architecture; [patch]; owner: unclaimed.
- Outcome: `coeus-hephaestus` consumes hephaestus-core's `PadOps<D, T>` seam once merged, instead of any local padding duplication.
- Blocker: no adoption surface — audit finds zero pad call sites and zero local padding duplication in coeus-hephaestus (re-open trigger: a GPU pad path is scoped as a feature item, or this item is closed as stale).
- Note: upstream trigger FIRED (`PadOps<D, T>` + CUDA/host/Metal/ROCm `pad_seam` impls all on hephaestus master, `domain/pad.rs:36`), but the item's premise is stale: no `*pad*` module exists in any of coeus-cuda/wgpu/rocm/metal/hephaestus, `coeus-ops::pad` is CPU-only via `coeus_leto::pad_values`, and conv/pool only thread padding config to device kernels. Adoption as written (route + remove duplication) has nothing to route or remove; a GPU pad path would be new feature scope, not this [patch]. Recommend rescope-or-close.
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
- Note (attribution slice delivered uncommitted): `alloc_budget.rs` now records seq/size/raw-IPs per counted alloc under `COEUS_ALLOC_ATTRIBUTION=1` (`backtrace::trace`, `backtrace 0.3` dev-dep) and prints both sides inside the mismatch assert; +2 hermetic plumbing tests. Hard-won constraints encoded: allocator reads an atomic arm flag (OnceLock env-read inside `alloc` deadlocks at startup — observed 30m hang), dumps print raw addresses (std symbol resolution blocks without a symbol server — observed hang; `Backtrace::frames`/`ip` are nightly-only). End-to-end proven via forced mismatch (dump + FAIL, 31KB). Gates: 7/7 serial flag-off and flag-on, clippy+fmt clean; lock is a 1-line edge addition (pristine HEAD fails `--locked` here too — overlay drift is pre-existing). Remaining: run on the hosted runner until the flake recurs — the dump now identifies the extra allocation.

<a id="coeus-gradcheck-generic-instantiation"></a>
## COEUS-GRADCHECK-GENERIC-INSTANTIATION — Generic-instantiation gradcheck coverage

- Status: todo; priority: verification; [patch]; owner: unclaimed.
- Outcome: every finite-difference check in `crates/coeus-autograd/tests/autograd/gradcheck/` runs at both `f64` (sensitivity oracle) and `f32` (instantiation coverage, per `standards`: Generic Instantiation Coverage), plus FD coverage for the ops that had none.
- Delivered: `mod.rs`'s shared fixtures (`Sampler`, `tensor`, `weighting`, `weighted`) are generic over `GradcheckScalar` (`Float + leto_ops::Scalar + coeus_leto::RealScalar` plus the `CpuAddressableStorage` bound gradcheck itself needs); the module doc states the two-role f64/f32 rationale. New FD coverage, each a generic fn instantiated at both types from a single `#[test]`: `add`/`sub`/`div`/`remainder`/`maximum`/`minimum`/`neg` (`arithmetic.rs`); `max_pool1d/2d/3d`/`avg_pool1d/2d/3d` (`pooling.rs`, via `coeus-nn` `Module` layers, new dev-dependency); `conv1d`/`conv3d`/`conv_transpose1d/2d/3d`/`unfold1d/2d`/`fold1d/2d` (`conv.rs`, via `coeus-nn` `Module` layers). Converted to dual-instantiation: `normalization.rs`, `core_ops.rs`, `attention.rs`.
- Remaining: (1) convert the 4 remaining existing check files (`activation`, `losses`, `reduction`, `shape`) to dual-instantiation; (2) FD coverage still missing: `ctc`, `dropout` (fixed mask), `sparse_matmul(_coo)`, `transpose_2d`, `index_put`, `rotate_half`, `linear_interpolation` (`cross_entropy` and `embedding` already have coverage in `losses.rs`).
- Acceptance: every listed op has a generic FD check instantiated at both types; all 11 existing files are dual-instantiated; a check that fails at `f32` under gradcheck's own derived `ε^(2/3)` bound is root-caused, never given a widened tolerance.
- Next step: convert `activation.rs` next (largest remaining, ~50 tests) then `losses`/`reduction`/`shape`, each its own PR; then the remaining missing-coverage ops as dependency-ordered per-family PRs.
- Note (verified uncommitted, no code change): item text is stale — all work is already in-tree. Audit: activation 46/46, arithmetic 7/7 (8th `#[test]` hit is a doc comment), attention 7/7, conv 9/9, core_ops 6/6, losses 17/17, new_ops 9/9, normalization 2/2, pooling 6/6, reduction 19/19, shape 22/22 dual f64+f32 instantiations; all Remaining-(2) ops covered (`ctc`/`dropout`/`sparse_matmul(_coo)`/`transpose_2d`/`index_put`/`rotate_half`/`linear_interpolation` in new_ops, `cross_entropy` in losses, `embedding` in new_ops — the parenthetical claiming the last two pre-existed was wrong); zero `#[ignore]`/`should_panic`; no widened tolerances. Evidence: autograd_ops 249/249 green. Ready to close on commit.

<a id="coeus-autograd-linear-interpolation-generic-scalar"></a>
## COEUS-AUTOGRAD-LINEAR-INTERPOLATION-GENERIC-SCALAR — Generalize linear_interpolation over T: Float

- Status: todo; priority: tightening; [patch]; owner: unclaimed.
- Outcome: `linear_interpolation`/`linear_interpolation_backward` in `coeus-autograd` and `coeus-ops` take `Var<T, B>`/`Tensor<T, B>` for `T: Float` instead of being pinned to `f32` end-to-end, matching the variation-first-authorship standard every other autograd op in this crate already follows.
- Scope: `crates/coeus-ops/src/interpolation.rs::linear_interpolation`/`linear_interpolation_backward` and `crates/coeus-autograd/src/ops/interpolation.rs::linear_interpolation` hardcode `f32` in every signature (`image: &Var<f32, B>`, `grid: &Var<f32, B>`); `BoundaryPolicy::neighbours` also hardcodes `f32` coordinates. Found while adding this op's finite-difference gradcheck coverage (`COEUS-GRADCHECK-DUAL-COVERAGE`): the check could only be written at `f32`, the sole instantiation that compiles, rather than choosing `f32` as one of two.
- Acceptance: `linear_interpolation` is generic over `T: Float` (or the minimal bound the interpolation math needs); existing `f32` callers and tests are unaffected; `crates/coeus-autograd/tests/autograd/gradcheck/new_ops.rs::linear_interpolation_backward_matches_finite_differences` extends to a generic `<T: GradcheckScalar>` case function instantiated at both `f64` and `f32`, following this module's established pattern.
- Non-goal: changing `BoundaryPolicy`'s trait shape beyond making its coordinate type generic; adding new boundary policies.
- Next step: generalize `BoundaryPolicy::neighbours`'s coordinate parameter over `T: Float` first (the narrowest dependency), then the two `linear_interpolation` call chains, then extend the gradcheck.
- Note (delivered uncommitted): `neighbours`/`linear_interpolation`/`linear_interpolation_backward`/`InterpolationGradients`/`LinearInterpolationNode` generic over `T: Float + FloatElement` (minimal: `Float` for floor/finite/compare, `FloatElement` for `from_count`; index clamp widens exact integers through `to_f64`, NaN still rejected by callers); 17 turbofish sites mechanically widened, no `T = f32` default (compat bridge refused); `grid_sample_3d` untouched (separate op). Gates: ops lib 139/139, ops integration 94/94, autograd lib 105/105, autograd_ops 247/247 (dual-dtype gradcheck passes), clippy clean, fmt clean; alloc_budget parallel flake reconfirmed environmental (5/5 serial). Ready to close on commit.
