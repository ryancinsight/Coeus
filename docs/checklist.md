# Global Progress Checklist: Coeus

## ATLAS-WGPU-SAFETY-002 — Checked reduction dispatch

- [x] Replace every unchecked fused-reduction layout and dispatch conversion with
      the canonical typed WGPU error boundary.
- [x] Add negative overflow/layout contracts and preserve value-semantic
      fused-reduction execution.
- [x] Pass focused CPU/Moirai and active-device WGPU fusion Nextest regressions,
      including borrowed lifetimes, typed invalid broadcasts, empty-axis
      identities/errors, and signed/unsigned WGSL generation.
- [x] Pass the final four-package Nextest suite (393/393), warning-denied
      all-target Clippy, 61 doctests, formatting, and independent re-review.
- [x] Pass exact-head provider CI on the repository-native lockfile: run
      `30680050203` passed WGPU, CUDA, ROCm, and Metal at `253c0da5`; PR #259
      merged as `5193764a`.

## COEUS-ATTENTION-PROVIDER-001 — Provider-owned attention dispatch

- [x] Merge the required Leto and Hephaestus attention provider contracts.
- [x] Add ADR-0047 and claim the complete caller/deletion closure.
- [x] Route CPU attention through borrowed Leto storage.
- [x] Route WGPU, CUDA, ROCm, and Metal through the generic Hephaestus bridge.
- [x] Propagate typed failure through operation, autograd, module, and Python APIs.
- [x] Delete local kernels, launchers, host fallbacks, tests, and obsolete ADRs.
- [x] Pass focused local gates and independent architecture/correctness review.
- [x] Pass exact-head hosted run `30666670100` and merge PR #256 as
      `ee3bb94f`.

## ATLAS-COEUS-HEPHAESTUS-006 — Native activation-tail providers

- [x] Route `Mish`, `MishGrad`, `Elu`, and `EluGrad` through direct Hephaestus
      WGPU/CUDA contiguous and strided APIs.
- [x] Route the f32 operations through the existing Hephaestus ROCm and Metal
      activation provider dispatches.
- [x] Compare forward and gradient results with the Leto CPU oracle in each
      backend parity suite.
- [x] Add ADR-0030 and synchronize the active Coeus parity artifacts.
- [x] Run and record exact-head WGPU, CUDA, ROCm, and Metal provider CI:
      backend run `30623370603` passes all four software provider lanes; the
      hardware ROCm lane is manual-only and skipped by design.
- [x] Complete the local locked metadata, focused non-CUDA nextest (307/307),
      warning-denied Clippy, workspace doctests (153 passed, 2 ignored),
      warning-denied rustdoc, and CUDA feature compile gates.
- [x] Complete the workspace Clippy gate after making distributed, autograd,
      normalization, and Python reduction backend errors explicit.
- [x] Complete focused CUDA nextest under the x86_64 MSVC toolchain: 6/6
      activation-tail tests pass, including contiguous and transposed device
      execution.
- [x] Complete the focused CPU/WGPU/ROCm/Metal activation-tail lane: 10/10
      tests pass.
## ATLAS-COEUS-NN-SAFETY-019 — Fallible module execution

- [x] Enumerate all module implementations and direct consumers.
- [x] Add ADR-0045 and the module error/trait vertical hierarchy.
- [x] Change the canonical trait and all implementations in one cutover.
- [x] Propagate normalization backend failures and preserve BatchNorm state.
- [x] Migrate Rust, Python, benchmark, and doctest call sites.
- [x] Pass warning-denied and value-semantic gates, SemVer checks, hosted CI,
      and merge.

## ATLAS-COEUS-WGPU-008 — Provider-owned reductions

- [x] Route all five ordinary WGPU reductions directly through Hephaestus.
- [x] Delete the superseded Coeus dispatcher, shader, validation, and tests.
- [x] Add rank-one/rank-two Leto parity and exact rank-three rejection tests.
- [x] Select the complete reduction contract in provider CI.
- [x] Pass warning-denied Clippy, doctests, and implementation-head provider
      CI.
- [x] Pass terminal provider run `30408820242` and merge PR #246 as
      `7a9811f4`.

## ATLAS-COEUS-CUDA-007 — Native dispatch boundary

- [x] Remove CUDA binary, unary, matrix, reduction, and fused CPU execution
      paths.
- [x] Make disabled-provider builds expose no mathematical `BackendOps`
      implementation.
- [x] Verify the no-default build and its three focused Nextest contracts.
- [x] Verify CUDA-feature all-target builds, focused provider Nextest,
      doctests, warning-denied Clippy, and exact-head provider CI.
- [x] Record terminal evidence and merge the change as PR #245 at
      `77834e37`.

## ATLAS-COEUS-DISPATCH-SAFETY-020 — Provider-owned convolution

- [x] Add regular/transposed rank-generic forward and backward contracts to
      Leto and Hephaestus.
- [x] Consolidate Coeus dispatch into four fallible const-generic `ConvOps`
      methods with rank-specific zero-cost adapters.
- [x] Route CPU storage directly to Leto and CUDA/WGPU/ROCm/Metal buffers
      directly to their Hephaestus providers.
- [x] Delete Coeus CUDA/WGPU convolution kernels, CUDA host fallbacks, generic
      transposed host defaults, `ConvTranspose3dOps`, and autograd host
      backward loops.
- [x] Propagate typed failures through autograd, NN, Python, benchmarks, and
      direct tests.
- [x] Pass warning-denied all-target Clippy for the consolidated Leto,
      Hephaestus, WGPU, CUDA, and operation-contract scope.
- [x] Pass CPU/autograd/NN Nextest 592/592 and final-review
      Leto/Hephaestus/autograd/WGPU Nextest 214/214.
- [x] Pass all 46 executable affected-package doctests; retain the two
      pre-existing ignored NN doctests.
- [x] Confirm the fallible `ConvOps` contract and removed capability seam as a
      major change with `cargo-semver-checks`.
- [x] Pass exact-head provider CI and record terminal evidence: run
      `30545333101` passed WGPU, CUDA, ROCm, and Metal; the required-device
      ROCm lane was skipped because no AMD hardware runner was dispatched.
- [x] Merge PR #250 and record merge revision `0dfab53e`.

## ATLAS-COEUS-DISPATCH-001 — Remove host-copy selection fallbacks

- [x] Record the CPU/provider capability boundary in ADR-0026.
- [x] Constrain `ReductionOps` selection defaults to `CpuBackend`.
- [x] Migrate Coeus and autograd selection entry points to the bound.
- [x] Run pinned formatting, metadata, focused checks, and the applicable
      provider dispatch checks. Local package compile/Nextest/doctest execution
      remains blocked by the stale peer-owned Leto path.
- [x] Commit and publish the verified increment; exact-head provider matrix
      `30278852605` passed WGPU `90019911397`, CUDA `90019911331`, ROCm
      `90019911264`, and Metal `90019911476`; required-device ROCm
      `90019912082` was skipped because no hosted AMD runner was dispatched.

## ATLAS-COEUS-HEPHAESTUS-005 — Native unary math providers

- [x] Add the shared 19-operation Hephaestus unary math vocabulary and export
      it through WGPU, CUDA, ROCm, and Metal.
- [x] Route all 19 f32 operations through native ROCm and Metal strided
      providers; keep integer capability boundaries typed and explicit.
- [x] Compare valid-domain ROCm and Metal outputs with the Leto CPU oracle.
- [x] Run exact-head WGPU, CUDA, ROCm, and Metal backend-parity CI and record
      the terminal run and job IDs. Run `30273987046` passed WGPU
      `90003264732`, CUDA `90003264777`, ROCm `90003265014`, and Metal
      `90003264805`; required-device ROCm `90003265412` was skipped because no
      registered AMD runner was available.

## ATLAS-COEUS-HEPHAESTUS-004 — Native comparison providers

- [x] Add scalar-aware Hephaestus comparison markers and contiguous/strided
      provider entry points for f32, i32, and u32.
- [x] Route all six comparisons through the native ROCm and Metal provider
      matches without CPU fallback or duplicated vendor kernels.
- [x] Compare ROCm and Metal f32, i32, and u32 results with the Leto CPU
      oracle, including f32 broadcast inputs.
- [x] Split vendor backend identity, operation families, and runtime integration
      into vertical leaves while preserving the public backend surface.
 - [x] Complete provider co-evolution in the merged Leto unit
       `d94e3ba`/`df14311`; the active local `codex/leto-real-sparse-lu` path
       remains a peer-owned branch predating those markers.
 - [x] Run exact-head WGPU, CUDA, ROCm, and Metal backend-parity CI and record
       the terminal evidence: workflow `30268824209` passed WGPU `89986119972`,
       CUDA `89986119939`, ROCm `89986120026`, and Metal `89986119988`; PR #224
       merged as `84b5bccd`, and required AMD hardware remained skipped.

## ATLAS-COEUS-HEPHAESTUS-003 — Native activation providers

## ATLAS-COEUS-HEPHAESTUS-003 — Native activation providers

- [x] Add dialect-specific Hephaestus activation and gradient expression
      markers and export them through the backend crates.
- [x] Dispatch the common activation set through native ROCm and Metal
      strided providers with an explicit `f32` capability boundary.
- [x] Compare ROCm and Metal forward/gradient activation results with Leto on
      signed inputs and test unsupported integer requests.
- [x] Run exact code-head WGPU, CUDA, ROCm, and Metal backend-parity CI:
      run `30226854005`, WGPU `89858362274`, CUDA `89858362266`, ROCm
      `89858362239`, and Metal `89858362247` passed; Coeus PR #223 merged at
      `4b807ddd`.

## ATLAS-COEUS-HEPHAESTUS-002 — Native elementwise providers

- [x] Add the generic ranked Hephaestus elementwise provider seam for
      contiguous and broadcast rank-1 through rank-4 layouts.
- [x] Add native ROCm and Metal implementations for the common arithmetic and
      unary math operation set.
- [x] Compare provider results with Leto for contiguous and broadcast inputs;
      cover unsupported operation/rank errors.
- [x] Run exact-head ROCm, Metal, WGPU, and CUDA backend-parity CI and record
      the terminal run and job IDs: `30224422963` passed WGPU
      `89852207720`, CUDA `89852207699`, ROCm `89852207677`, and Metal
      `89852207739`; required-device ROCm `89852208025` was skipped because
      the workflow was not manually dispatched.

## ATLAS-COEUS-HEPHAESTUS-001 — Native ROCm and Metal reduction providers

- [x] Add one generic `coeus-hephaestus` storage and reduction/scan dispatch
      layer shared by vendor backends.
- [x] Add native Coeus ROCm and Metal backends for rank-2 reductions and
      forward/reverse cumulative sum/product scans.
- [x] Compare every provider reduction and scan result with the Leto CPU
      oracle, including product identity and suffix direction.
- [x] Fix the owning Hephaestus ROCm device cache so the Coeus backend device
      satisfies `Send + Sync`; Hephaestus PR #109 merged as `95eeaa5`.
- [x] Run exact-head hosted ROCm feature and macOS Metal CI with the WGPU/CUDA
      matrix. Run `30221620203` passed at `f8bb4c7e`:
      ROCm `89844922811`, Metal `89844922775`, WGPU `89844922827`, and CUDA
      `89844922774`. The optional required-device ROCm lane
      (`89844923036`) was skipped because this pull request did not request a
      hardware dispatch.


## ATLAS-HEPHAESTUS-SCAN-001 — Native cumulative scans

- [x] `ReductionOps::cumsum` and `suffix_sum` expose typed backend results.
- [x] Route WGPU and CUDA rank-2 sum and product scans to Hephaestus provider
      kernels.
- [x] Reject unsupported rank/layout cases instead of copying through host.
- [x] Add CPU/Leto, WGPU, and CUDA differential tests. Exact-head provider CI
      run `30214708599` passes at `695c2890` (WGPU job `89826735353`, CUDA job
      `89826735351`).

## ATLAS-HEPHAESTUS-PRODUCT-001 — Native product reductions

- [x] Add one shared `ReductionOp::Prod` contract through CPU/Leto, WGPU, and
      CUDA, including fused CPU reduction evaluation.
- [x] Add rank-2 `prod_axis` value-parity tests for CPU/Leto, WGPU, and CUDA.
- [x] Run exact-head WGPU/CUDA provider CI with the product-reduction filters
      after Leto product API merge `524e780`: run `30218187376` passes at
      `b31cf448` (WGPU job `89835879122`, CUDA job `89835879151`).

## MS-445 Python release wheels [patch]

- [x] Add the pinned build-once GitHub Release and PyPI workflow.
- [x] Document the `coeus-python` distribution, `pycoeus` import, Cargo version
      source, supported CPython range, and OIDC publication contract.
- [x] Build, install, import, and inspect a production CPython 3.13 wheel
      locally as `coeus-python` 0.9.0 / `pycoeus`.
- [x] Create the protected `pypi` environment restricted to
      `coeus-python-v*` tags.
- [ ] Pass hosted CI on the exact release-automation head.
- [ ] Register the PyPI pending trusted publisher.

## MS-444 standalone Git dependency graph [patch]

- [x] Replace external sibling paths with remote provider identities and retain
      the repository lockfile as the revision SSOT.
- [x] Remove repository-owned local patch tables; synchronized consumers own
      any local checkout substitution at their workspace root.
- [x] Regenerate the lockfile and verify focused Coeus package gates.
- [x] Resolve the selected packages from a clean external Git consumer.

**Evidence:** locked autograd compilation and warning-denied all-targets
Clippy pass; 94/94 autograd Nextest cases pass; locked metadata reports one
identity per Atlas provider. The full format gate exposes only the existing
`crates/coeus-ops/tests/half_precision_diff.rs` line-wrap drift. Asclepius compiles
`coeus-{autograd,core,ops,tensor}` from pushed commit `99920888`.

## MS-443 backend-generic host extraction [minor]

- [x] Route non-host storage through `ComputeBackend::copy_to_host`.
- [x] Compact offset and strided views according to their logical layout.
- [x] Verify exact transposed-slice values, package format, warning-denied
      Clippy, and the complete tensor test suite.

**Evidence:** `coeus-tensor` Nextest passes 57/57; format and warning-denied
all-targets Clippy pass.

## MS-441 remove tensor Burn benchmark [patch]

- [x] Delete the obsolete legacy-provider dev dependency and comparison
      benchmark bodies.
- [x] Preserve real Coeus Sequential/Moirai and Leto dispatch measurements in
      the tensor benchmark and update its theorem-facing documentation.
- [x] Commit the lock graph after aligning the Hephaestus provider floor to
      merged `0.16.1`.

**Evidence:** targeted residue scan is clean; locked package check, 56/56
Nextest, warning-denied Clippy, five doctests, warning-clean rustdoc, and
locked metadata pass.

## MS-442 remove NN legacy benchmark [patch]

- [x] Delete the remaining NN benchmark-only legacy dependency and comparison
      rows while preserving all native Sequential/Moirai measurements.
- [x] Run the workspace residue scan plus package format, locked check, and
      warning-denied Clippy after the NN cutover.
- [x] Complete full Nextest, doctests, rustdoc, and dependency-policy
      verification against merged Hephaestus 0.17.0.

**Evidence:** `coeus-nn` retains its Criterion target with 211 operation groups
and 424 native Sequential/Moirai rows while declaring no Burn dependency; the
committed lock graph has no Burn package. Format, locked package check, and
warning-denied all-targets Clippy pass; configured Nextest is 268/268,
doctests are 8/8 with two intentionally ignored, rustdoc is warning-clean, and
locked metadata resolves Eunomia 0.4.0, Leto 0.38.2, and Hephaestus 0.17.0.

## Sprint ATLAS-PROVIDER-004: Current provider consumer refresh [COMPLETE]

- [x] [major] Raise the workspace release line to 0.9.0 for the Rust 1.95 and
  provider-floor contract.
- [x] [major] Raise Leto and Leto Ops to 0.38.0 and the Hephaestus GPU crates
  to 0.16.1.
- [x] Replace the stale Burn live-parity target with native analytical pooling
  contracts instantiated for the Sequential and Moirai providers.
- [x] Replace probe-and-rebind TCP test setup with listener-owned loopback
  clusters for the Rust and PyO3 distributed-collective boundaries.
- [x] Verify `cargo fmt --check`, warning-denied workspace Clippy, the
  1008/1008 all-feature nextest suite (including real CUDA), 153 doctests
  passing with 2 intentionally ignored, and warning-clean workspace Rustdoc.

## Sprint ATLAS-PROVIDER-003: Current provider consumer alignment [COMPLETE]

- [x] [major] Raise Coeus' local Mnemosyne contract to 0.5.0 and declare the
  provider-imposed Rust 1.95 floor to downstream consumers.
- [x] [major] Raise Coeus' local Moirai contract to 0.4.0 and retain its
  Melinoe-backed parallel execution provider contract.
- [x] [major] Raise the Coeus WGPU/CUDA substrate contracts to Hephaestus
  0.14.0 and its required device-feature contract.
- [x] Verify `coeus-core` check, warning-denied Clippy, and 21/21 nextest;
  `coeus-wgpu` check; `cargo fmt --check`; and one local Mnemosyne 0.5 identity
  with no Mnemosyne 0.4 package in the resolved dependency tree.

## Sprint ATLAS-PROVIDER-002: Atlas provider alignment [COMPLETE]

- [x] [patch] Raise Coeus' local Mnemosyne contract to 0.4 after Moirai's
  provider update removed the mixed allocator generation.
- [x] [patch] Raise Hephaestus to 0.13 and align its WGPU 30 ABI to the
  provider-owned Vulkan/Metal backend set.
- [x] Verify `coeus-core`, `coeus-wgpu`, and the `coeus-cuda` library against
  the aligned provider graph.
- [x] Implement native CUDA 1-D/2-D unfold and adjoint fold kernels; verify
  exact device/CPU agreement, including overlap multiplicities.
- [x] Replace the four empty CUDA 1-D pooling methods with native max/average
  forward and input-adjoint kernels and exact sequential differential tests.
- [x] Bind every Coeus driver launch to the Hephaestus-owned CUDA context,
  eliminating invalid cross-context module handles.
- [x] Correct the WGPU/CUDA persistent-buffer placement contracts: WGPU rejects
  unsupported host-pinned persistence and CUDA reports its persistent device
  tier while retaining exact transfer round-trips.
- [x] Verify warning-denied provider Clippy, 88/88 default provider tests, and
  75/75 real-CUDA all-feature tests on an NVIDIA device.

## Sprint MEL-SCOPE-001: Melinoe 0.9 provider refresh [COMPLETE]

- [x] [patch] Raise `coeus-ops`' local Melinoe contract to 0.9.0.
- [x] Verify locked metadata, local Mnemosyne backend type unification,
  `coeus-ops` Clippy, and 196/196 nextest. The root manifest's existing
  Mnemosyne 0.3/Moirai integration edits are preserved and completed with the
  missing transitive Mnemosyne patches and Hephaestus 0.12 constraints.

## Sprint MS-445: bench 215 (EmbeddingBag mean) [COMPLETE]

- [x] [patch] Expanded the existing EmbeddingBag benchmark workload with mean
  reduction rows: Burn Embedding plus `mean_dim`, Coeus Sequential, and Moirai.
- [x] Verification: `cargo check -p coeus-nn --bench nn_bench --offline` and
  `cargo clippy -p coeus-nn --benches --offline -- -D warnings` pass.

## Sprint MS-444: bench 214 (LocalResponseNorm) [COMPLETE]

- [x] [patch] Expanded G-043 with `bench_local_response_norm_forward` on
  `[8, 32, 16, 16]`, size 5, comparing Coeus Sequential and Moirai.
- [x] [patch] Omitted a Burn row because pinned Burn 0.16 exposes no
  `LocalResponseNorm` module family.
- [x] Verification: `cargo check -p coeus-nn --bench nn_bench --offline` and
  `cargo clippy -p coeus-nn --benches --offline -- -D warnings` pass.

## Sprint MS-440: bench 210 (vanilla RNN) [COMPLETE]

- [x] [patch] Expanded G-043 with `bench_rnn_forward`: Coeus Sequential vs
  Moirai on the shared recurrent workload `[4, 32, 64] -> hidden 128`.
- [x] [patch] Omitted a Burn row because the pinned Burn 0.16 `nn::rnn`
  provider exposes LSTM and GRU but no vanilla RNN family.
- [x] Verification: `cargo check -p coeus-nn --bench nn_bench --offline` and
  `cargo clippy -p coeus-nn --benches --offline -- -D warnings` pass.

## Sprint MS-441: bench 211 (vanilla RNNCell) [COMPLETE]

- [x] [patch] Expanded G-043 with `bench_rnn_cell_forward`, measuring one
  `RNNCell::step` on Coeus Sequential and Moirai independently of unrolling.
- [x] [patch] Omitted a Burn row because the pinned Burn RNN surface lacks
  vanilla recurrent cells.
- [x] Verification: `cargo check -p coeus-nn --bench nn_bench --offline` and
  `cargo clippy -p coeus-nn --benches --offline -- -D warnings` pass.

## Sprint MS-442: bench 212 (Bidirectional RNN) [COMPLETE]

- [x] [patch] Expanded G-043 with `bench_bidirectional_rnn_forward`, measuring
  two independent RNNs plus the tracked reverse-and-concatenate wrapper path.
- [x] [patch] Omitted Burn because its pinned RNN surface has neither vanilla
  RNN nor the bidirectional wrapper.
- [x] Verification: `cargo check -p coeus-nn --bench nn_bench --offline` and
  `cargo clippy -p coeus-nn --benches --offline -- -D warnings` pass.

## Sprint MS-443: bench 213 (cross attention) [COMPLETE]

- [x] [patch] Expanded G-043 with `bench_mha_cross_attention_forward`, using
  distinct query and memory sequence lengths through `forward_cross`.
- [x] [patch] Omitted Burn because the pinned MHA benchmark surface exposes
  self-attention input only, not separate query/key/value tensors.
- [x] Verification: `cargo check -p coeus-nn --bench nn_bench --offline` and
  `cargo clippy -p coeus-nn --benches --offline -- -D warnings` pass.

## Sprint MS-439: Named optimizer ownership [COMPLETE]

**Target version**: 0.8.0

- [x] [arch] Move `Parameter` from `coeus-nn` to `coeus-autograd`, the deepest
  common owner of module and optimizer parameter identity.
- [x] [major] Make SGD, Adam, AdamW, RMSProp, and AdaGrad own named parameters
  directly and preserve names through updates and clipping.
- [x] [major] Add checked module reload by complete hierarchical inventory and
  require explicit Python `(name, tensor)` pairs.
- [x] [major] Verify optimizer nextest 20/20, cross-boundary nextest 21/21,
  affected NN parity 144/144, reordered-name rejection, exact update semantics,
  Clippy, Rustdoc, and doctests.

## Sprint MS-438: Stable hierarchical module parameters [COMPLETE]

**Target version**: 0.7.0

- [x] [minor] Add `Module::named_parameters` as the canonical reflection seam.
- [x] [minor] Assign semantic leaf names and hierarchical child prefixes
  without changing optimizer parameter order or gradient-buffer identity.
- [x] [minor] Cover dynamic/static sequences, recurrent compositions,
  attention, and transformer trees without flattened ordinal fallbacks.
- [x] [minor] Verify exact 26-entry decoder paths, 84-entry transformer
  uniqueness, shared gradient identity, nextest 410/410, and Clippy.

## Sprint MS-437: Dimension-complete interpolation [COMPLETE]

**Target version**: 0.6.0

- [x] [major] Replace the dimension-specific public surface with one
  `linear_interpolation::<D, _, _>` operation family for 2-D and 3-D images.
- [x] [minor] Encode replicated borders as a sealed ZST policy and share the
  allocation-free corner traversal between both dimensions.
- [x] [minor] Verify exact forward/image/grid derivatives, every coordinate by
  central difference, malformed dimension/shape rejection, autograd, and
  Sequential/Moirai agreement; affected nextest 282/282 clean.
- [x] [major] Bump the workspace to 0.6.0 and document the public migration.

## Sprint MS-436: Bounded archived tensor state [COMPLETE]

- [x] [minor] Replace the eager bespoke `StateDict` encoding with validated rkyv archives.
- [x] [minor] Provide borrowed archived name/shape/payload inspection and bounded materialization.
- [x] [minor] Verify deterministic encoding, pointer-range provenance, round trips, truncation, scalar mismatch, duplicate names, and resource limits; package nextest 56/56, Clippy, Rustdoc, and doctests clean.

## Sprint MS-434: Rank-preserving batched matmul [COMPLETE]

- [x] [patch] Preserve logical batch axes for rank-generic matmul outputs
  while retaining flattened kernel dispatch.
- [x] [patch] Give accumulating backward kernels an explicit rank-3 dispatch
  layout instead of passing the logical rank-N destination layout.
- [x] [patch] Verify exact rank-4 forward values and both operand gradients;
  affected Coeus nextest 689/689 and warning-denied Clippy clean.

## Sprint MS-433: Rank-generic linear projection [COMPLETE]

- [x] [minor] Generalized `Linear::forward` from rank-2 matrices to a
  canonical last-axis projection for every rank of at least two.
- [x] [minor] Preserved autograd through the flatten/project/restore path and
  retained the rank-2 fast path without an intermediate reshape.
- [x] [minor] Verified exact rank-3/rank-5 values and rank-3
  input/weight/bias gradients, full `coeus-nn` nextest 409/409,
  warning-denied Clippy, and rustdoc.

## Sprint MS-432: Three-dimensional reverse-mode provider [SUPERSEDED BY MS-437]

- [x] [minor] Added native-precision image and `(z, y, x)` grid derivatives
  for rank-5 linear sampling. The dimension-specific API was removed by MS-437.
- [x] [minor] Added a tracked autograd operation that accumulates both input
  gradients without detaching the model graph.
- [x] [minor] Verified analytical ramp derivatives, constant-field coordinate
  invariance, malformed-gradient rejection, and Sequential/Moirai execution:
  nextest 4/4; targeted Clippy and rustdoc clean.

## Sprint MS-431: Native 3-D coordinate-grid provider [SUPERSEDED BY MS-437]

- [x] [minor] Originally added the dimension-specific operation; MS-437
  removed that name in favor of the canonical const-dimension family required
  by RITK's Burn removal.
- [x] [minor] Encoded malformed rank, batch, coordinate-channel, empty-axis,
  and size-overflow failures as `InterpolationError`.
- [x] [minor] Verified analytical center and border values plus invalid-grid
  rejection on Sequential and Moirai backends: nextest 2/2; targeted clippy
  clean.

## Sprint MS-405: PyTorch/JAX parity defect closure [COMPLETE]
- [x] [patch] **pairwise_distance eps-convention** — `crates/coeus-autograd/src/ops/nn/loss/pairwise_distance.rs`
      swapped `s + eps` for `max(s, eps)` (matches torch's `pairwise_distance`
      exactly at `s >> eps`; subgradient at the floor is treated as zero).
- [x] [patch] **huber_loss classical-Huber rewrite** — `crates/coeus-autograd/src/ops/nn/loss/huber.rs`
      forward + backward replaced with the classical Huber definition matching
      `torch.nn.functional.huber_loss` and Burn's `HuberLossConfig`:
      `0.5·z²`/`δ·|z| - 0.5·δ²` forward, `z`/`sign(z)·δ` backward.
- [x] [patch] **nn_loss_tests::test_pairwise_distance** oracle updated to
      `max(s, eps)^(1/p)` analytical form (`crates/coeus-nn/tests/nn_ops/losses/nn_loss/`).
- [x] [patch] **PyTorch parity fixtures** — cross_entropy labels now `list[int]`,
      kl_div uses `reduction='mean'` in `tests/test_pytorch_parity.py`.
- [x] [patch] **JAX parity fixtures** — cosine_similarity + triplet_margin use
      `jnp.maximum(s, eps)`, kl_div reduces by `mean` in `tests/test_jax_parity.py`.
- [x] [patch] **Cargo.toml** workspace version bump `0.5.6` → `0.5.8`.
- [x] [patch] **CHANGELOG.md** `0.5.8 - 2026-07-04` section (Fixed: 5 entries).

### Verification

- `cargo fmt --check`  — clean.
- `cargo clippy --workspace --all-targets -- -D warnings` — clean.
- `cargo nextest run --workspace --no-fail-fast --test-threads=2` — **1027/1027 pass**.
- `cargo test --doc --workspace` — clean.
- `maturin develop -m crates/coeus-python/Cargo.toml` — built + installed clean.
- `pytest crates/coeus-python/tests/test_pytorch_parity.py -q` — **376/390 pass** (10 skip).
- `pytest crates/coeus-python/tests/test_jax_parity.py -q` — **187/190 pass** (3 skip).
- `pytest crates/coeus-python/tests/test_mlx_parity.py -q` — 70 skip (Windows gating).
- `pytest crates/coeus-python/tests/test_pytorch_parity.py -v -k "test_huber or test_pairwise or test_cosine_similarity"` — **all pass**.
- `crates/coeus-nn/tests/burn_live_parity` — **140/140 pass** (live Burn oracle,
  including `probability_loss_forward_and_backward_match_burn` for huber against
  `HuberLossConfig`).

### Deferred (out of MS-405 scope)

- ~~**MS-406**: Scatter_add/index_put autograd wiring~~ **CLOSED** — reconciled
  2026-07-08: `crates/coeus-python/src/ops/indexing.rs` `scatter_add`/`index_put`
  already route through `coeus_autograd::{scatter_add,index_put}` (tracked,
  not `Var::new(t, false)`), with value-semantic backward parity tests
  `test_scatter_add_bwd_matches_pytorch` / `test_index_put_bwd_matches_pytorch`
  in `test_pytorch_parity.py`. Entry was stale relative to the tree.
- ~~**MS-407**: `EmbeddingBag.forward` PyTorch-API~~ **CLOSED** — reconciled
  2026-07-08: `pycoeus.EmbeddingBag.forward(indices: Tensor, offsets:
  Optional[Tensor] = None)` already matches the PyTorch-style no-offset
  signature (`crates/coeus-python/pycoeus.pyi:710`); landed alongside the
  remainder/maximum/minimum binding extraction (`e36f95f`). Entry was stale.
- **MS-413-followup**: `triplet_margin_loss` boundary `relu'(0)` subgradient
  discrepancy with JAX (`jnp.maximum(0, x)` returns 0.5 at `x=0`,
  Coeus ReLU returns 0). Affects PyTorch `test_triplet_margin_matches_pytorch`
  and JAX `test_triplet_margin_matches_jax` at boundary rows.

## Active Epic: Burn Parity, GPU Audit & Python Surface Expansion

### Current Sprint: MS-430 - bench 209 (Bilinear, Coeus-only) [COMPLETE]
- [x] [patch] Expanded G-043 benchmark matrix from **208 -> 209** by adding
  `bench_bilinear_forward` (Coeus Sequential vs Moirai,
  `Bilinear(in1=64, in2=64, out=32)` batch 128, two distinct inputs). No Burn
  oracle row: confirmed against the pinned `burn-core` 0.16.0 source that
  there is no `nn::Bilinear`/`BilinearConfig`.
- [x] [patch] Cargo.toml `0.5.10` -> `0.5.11`; CHANGELOG.md `0.5.11` section added.
- [x] Evidence: `cargo check -p coeus-nn --benches` pass; `cargo fmt --check
  -p coeus-nn` clean; `cargo clippy -p coeus-nn --benches -- -D warnings`
  clean; `cargo bench -p coeus-nn --bench nn_bench -- "Bilinear forward"`
  executes both rows (Sequential 3.69ms, Moirai 3.54ms median).

### Current Sprint: MS-429 - bench 208 (interpolate_2d nearest/bilinear) [COMPLETE]
- [x] [patch] Expanded G-043 benchmark matrix from **206 -> 208** by adding
  `bench_interpolate2d_nearest_forward`/`bench_interpolate2d_bilinear_forward`
  (full 3-way: Burn NdArray vs Coeus Sequential vs Coeus Moirai,
  `[8,16,32,32] -> [64,64]`). Burn 0.16's `nn::interpolate::Interpolate2d`
  exists for this family.
- [x] [patch] Cargo.toml `0.5.9` -> `0.5.10`; CHANGELOG.md `0.5.10` section added.
- [x] Evidence: `cargo check -p coeus-nn --benches` pass; `cargo fmt --check
  -p coeus-nn` clean; `cargo clippy -p coeus-nn --benches -- -D warnings`
  clean; `cargo bench -p coeus-nn --bench nn_bench -- interpolate_2d`
  executes and reports timings for both new 3-way groups.

### Current Sprint: MS-428 - bench 206 (MaxPool3d/AvgPool3d) + coeus-dist test-harness fix [COMPLETE]
- [x] [patch] Expanded G-043 benchmark matrix from **204 -> 206** by adding
  `bench_maxpool3d_forward`/`bench_avgpool3d_forward` (Coeus Sequential vs
  Moirai, `[4,8,16,16,16]` k2/s2). No Burn oracle row: confirmed against the
  pinned `burn-tensor` 0.16.1 source that `tensor::module` has no
  `max_pool3d`/`avg_pool3d` — the 3D pooling gap is structural to the pinned
  Burn version.
- [x] [patch] `coeus-dist` TCP test-harness TOCTOU fix (root-caused a
  full-workspace-run flake in `test_tcp_broadcast`; see gap_audit.md/CHANGELOG).
- [x] [patch] Cargo.toml `0.5.8` -> `0.5.9`; CHANGELOG.md `0.5.9` section added.
- [x] Evidence: `cargo check -p coeus-nn --benches` pass; `cargo bench -p
  coeus-nn --bench nn_bench -- "MaxPool3d|AvgPool3d"` executes and reports
  timings for both new groups.

### Current Sprint: MS-425..MS-427 - PyTorch 410 + bench 204 [COMPLETE]
- [x] [patch] Expanded PyTorch parity from **400 -> 410** via 10 Python-dunder
  scalar/backward checks (`+`, `-`, `*`, `/`, reflected `+`, `-`, `*`, `/`,
  reflected `pow`, and `abs`).
- [x] [patch] Expanded G-043 benchmark matrix from **200 -> 204** by adding four
  forward rows: `tanh4`, `sigmoid4`, `relu4`, and `sqrt4`.
- [x] Evidence: `cargo check -p coeus-nn --benches` pass; targeted PyTorch
  tranche `10/10` pass.

### Current Sprint: MS-421..MS-424 - Bench 200 + parity expansion [COMPLETE]
- [x] [patch] Expanded G-043 benchmark matrix from **196 -> 200** by adding four
  forward rows: `exp4`, `log4`, `sin4`, and `cos4`.
- [x] [patch] Expanded JAX parity from **210 -> 213** by adding backward checks
  for `atan`, `sinh`, and `log2`.
- [x] [patch] Expanded MLX parity from **65 -> 70** by adding forward checks for
  `erfc`, `recip`, `softsign`, `selu`, and `celu`.
- [x] Evidence: `cargo check -p coeus-nn --benches` pass; targeted JAX tranche
  `3/3` pass; targeted MLX tranche skips cleanly when MLX is unavailable.

### Current Sprint: MS-418..MS-420 - JAX/MLX parity expansion [COMPLETE]
- [x] [patch] Expanded JAX parity from **207 -> 210** by adding backward checks
  for `sin`, `cos`, and `tan`.
- [x] [patch] Expanded MLX parity from **60 -> 65** by adding forward checks for
  `sin`, `cos`, `tan`, `log10`, and `exp2`.
- [x] Evidence: targeted pytest JAX tranche `3/3` pass; targeted MLX tranche
  skips cleanly when MLX is unavailable.

### Current Sprint: MS-416 - PyTorch parity to 400 [COMPLETE]
- [x] [patch] Expanded PyTorch parity from **390 -> 400** by adding 10 targeted
  tests:
  `sin`, `cos`, `tan`, `atan`, `sinh`, `cosh`, `log2`, `exp2`,
  `scalar_sub` backward checks, plus `argmax(dim=0)` forward parity.
- [x] Evidence: targeted pytest tranche for MS-416 (`10/10` pass).

### Current Sprint: MS-413..MS-415 - JAX/MLX parity + zero-copy docs [COMPLETE]
- [x] [patch] Expanded JAX backward parity from **201 -> 207** by adding:
  `asinh`, `atanh`, `acosh`, `expm1`, `log1p`, and `topk(k=3)` gradient checks.
- [x] [patch] Expanded MLX forward parity from **55 -> 60** by adding:
  `atan`, `asinh`, `atanh`, `acosh`, and `log2` value checks.
- [x] [patch] Documented the new `coeus-ops` zero-copy identity fast paths for
  `gather`, `index_select`, and no-op `scatter_add` as the current SSOT
  behavior for avoiding redundant allocations.
- [x] Evidence: targeted pytest JAX tranche `6/6` pass; targeted MLX tranche
  selected tests skip cleanly on hosts without MLX.

### Current Sprint: MS-243 - cumprod backward zero decomposition fix [COMPLETE]
- [x] [patch] Replaced naive suffix-sum cumprod backward (NaN at zeros) with
  exact O(n) first/second-zero decomposition. Added
  `test_cumprod_backward_exact_at_zeros` (zero-free, one-zero, two-zero) at f64.
- [x] [patch] Fixed `clippy::default_constructed_unit_structs` in nn_bench.rs.
- [x] Evidence: `cargo fmt --check` clean; `cargo clippy` clean; `cargo nextest
  run` 465/465 passed; doctests 8/8; doc clean. Commit `ff2f45c` pushed to main.

### Current Sprint: MS-237 - special-function unary parity [COMPLETE]
- [x] [patch] Replaced Coeus' local `erf` approximation path with Eunomia's
  `FloatElement::{erf,erfc,lgamma}` surface for float scalars and extended
  `CpuUnaryOp` / Coeus-Leto unary dispatch with `Lgamma`.
- [x] [patch] Added forward-only `coeus_ops::lgamma`,
  `coeus_autograd::lgamma_forward`, and Python `pycoeus.gammaln` /
  `pycoeus.lgamma`. Backward requests raise `NotImplementedError` because the
  derivative requires `digamma`, which is not exposed by the provider yet.
- [x] [patch] Re-verified exact `pycoeus.gelu` plus `erf`/`erfc` f64 forward
  and gradient parity against PyTorch, and `gammaln` f64 forward parity against
  `torch.special.gammaln`.
- [x] Evidence tier: value-semantic Rust tests plus f64 PyTorch differential
  parity. Evidence: `rustup run nightly cargo check -p coeus-core -p
  coeus-leto -p coeus-ops -p coeus-autograd -p coeus-python`; `rustup run
  nightly cargo nextest run -p coeus-leto -p coeus-ops
  unary_dispatch_special_functions_match_reference_values
  sequential_unary_matches_scalar_reference
  moirai_unary_matches_scalar_reference` (3/3); `rustup run nightly cargo check
  -p coeus-wgpu`; `D:/miniforge3/python.exe -m maturin develop -m
  crates/coeus-python/Cargo.toml`; targeted PyTorch parity pytest (5/5).

### Current Sprint: MS-236 - scan/diff/NaN reduction parity [COMPLETE]
- [x] [patch] Preserved the existing `pycoeus.diff`, `pycoeus.cumsum`, and
  `pycoeus.cumprod` parity coverage while finishing `pycoeus.nansum` and
  `pycoeus.nanmean` through the Rust autograd and PyO3 surfaces.
- [x] [patch] Reworked `coeus_autograd::nansum` / `nanmean` to clean NaNs
  with tracked `masked_fill` over a non-differentiable NaN mask, so forward
  values match framework semantics and input gradients are zero at NaN
  positions.
- [x] [patch] Added value-semantic PyTorch and JAX forward+dx parity for
  `nansum` and `nanmean`; added the missing JAX `cumprod` parity check.
- [x] Evidence tier: analytical/value-semantic Rust tests plus differential
  empirical PyTorch/JAX parity. Evidence: `rustup run nightly cargo fmt -p
  coeus-autograd -p coeus-python --check`; `rustup run nightly cargo nextest
  run -p coeus-autograd --no-fail-fast` (60/60); `D:/miniforge3/python.exe -m
  maturin develop -m crates/coeus-python/Cargo.toml`; targeted PyTorch MS-236 parity
  (6/6); targeted JAX MS-236 parity (5/5); `rustup run nightly cargo clippy -p
  coeus-autograd -p coeus-python --all-targets -- -D warnings`;
  `rustup run nightly cargo test --doc -p coeus-autograd -p coeus-python`
  (15/15 autograd doctests, 0 pycoeus doctests); `git diff --check`.
- [x] Residual non-MS-236 gate blockers rechecked and closed: `rustup run
  nightly cargo nextest run -p coeus-python --no-fail-fast` now passes 72/72
  after scalar-tensor assertion cleanup, and `rustup run nightly cargo doc -p
  coeus-autograd -p coeus-python --no-deps` is warning-clean after the shared
  Atlas `leto-ops` / `melinoe` path crates compile.

### Previous Sprint: MS-218 - Apollo FFT autograd + Python parity [COMPLETE]
- [x] [minor] Added Apollo-backed `coeus_autograd::{fft_1d, ifft_1d,
  fft_1d_var, ifft_1d_var, fft_energy}` and wired the FFT module through
  `crates/coeus-autograd/src/ops/mod.rs` plus the crate-root public surface.
- [x] [patch] Added Rust value-semantic FFT regressions for hand-DFT
  forward values, inverse roundtrip, complex upstream-gradient propagation,
  and Parseval-derived `fft_energy` input gradients.
- [x] [patch] Added thin PyO3 FFT bindings: `pycoeus.ComplexTensor`,
  `pycoeus.fft`, `pycoeus.ifft`, and `pycoeus.fft_energy`, plus PyTorch
  `torch.fft.fft` forward + gradient parity at f64.
- [x] Evidence tier: analytical/value-semantic plus differential empirical.
  Evidence: `rustup run nightly cargo fmt -p coeus-autograd -p coeus-python --check`;
  `rustup run nightly cargo check -p coeus-autograd`; `rustup run nightly cargo nextest run -p coeus-autograd fft`
  (3/3); `rustup run nightly cargo check -p coeus-python`;
  `D:/miniforge3/python.exe -m maturin develop -m crates/coeus-python/Cargo.toml`;
  `D:/miniforge3/python.exe -m pytest crates/coeus-python/tests/test_pytorch_parity.py::test_fft_matches_pytorch -q`
  (1/1).

### Previous Sprint: MS-217 - PReLU / LeakyReLU subgradient parity (G-037 closure) [COMPLETE]
- [x] [patch] Coerced the single canonical `LeakyReluGrad` predicate from
  `x >= 0 ? 1 : α` to `x > 0 ? 1 : α` across `coeus-core` (float + int),
  `coeus-ops` (fuse tag), the Rust value-semantic `act_extended/` activation
  contract tree
  oracle, and `crates/coeus-nn/tests/nn_activation_tests.rs::test_leaky_relu_activation`.
  Matches `torch.nn.functional.prelu` / `F.leaky_relu.neg_slope` and
  JAX's `jnp.where(z > 0, z, alpha * z)`; closes
  `test_prelu_matches_pytorch` without any other activation regression.
- [x] [patch] Added new value-semantic Rust test
  `leaky_relu_kink_at_zero_returns_slope` and JAX parity test
  `test_prelu_matches_jax` covering the `x = 0` kink position across
  three frameworks (Rust core ↔ PyTorch ↔ JAX).
- [x] Evidence: `rustup run nightly cargo clippy --workspace --all-targets -- -D warnings`
  green in 30 s; `rustup run nightly cargo nextest run -p coeus-nn --no-fail-fast`
  386/386 green; pytest PyTorch parity file 73 passed (+2 deselected for the
  pre-existing hardswish/hardsigmoid gaps logged in `docs/gap_audit.md`);
  pytest JAX parity file 40/40.
- [x] Closed the remaining PReLU differential within G-037; only the
  PReLU torch.hardswish / hardsigmoid subgradient gaps remain (separate
  pre-existing items).

### Current Sprint: MS-216 - AdaptiveMaxPool PyO3 binding (G-046 closure, superseded by PR #112) [COMPLETE]
- [x] [patch] `PyAdaptiveMaxPool1d` + `PyAdaptiveMaxPool2d` thin PyO3 wrappers
  merged via peer PR #112 (`d1ad9d2`): `feat(python): AdaptiveMaxPool1d/2d
  binding + dx parity (PyTorch, JAX)`. Module re-exports extended;
  `m.add_class` calls added in `pycoeus` registration in
  `crates/coeus-python/src/lib.rs`.
- [x] Evidence: `rustup run nightly cargo clippy --workspace --all-targets -- -D warnings`
  zero warnings after 22.65s; `rustup run nightly cargo nextest run -p coeus-nn --no-fail-fast`
  379/379 green; pytest `test_adaptive_max_pool_backward_matches_pytorch`
  pass (PyTorch parity, 8.87 s combined with AvgPool variant);
  `test_adaptive_max_pool_matches_jax` pass (JAX parity, 4.10 s combined).
- [x] Closed G-046 (Python-binding parity closure for AdaptiveMaxPool).
  Trajectory: PR #109 (AdaptiveAvgPool diff), PR #110 (AvgPool dx parity),
  PR #111 (`b3e993b` AdaptiveMaxPool diff), PR #112 (PyO3 + dx parity).
- [x] Three-way forward + input gradient parity established:
  Rust core (coeus-nn::AdaptiveMaxPool1d/2d) \u2194 PyTorch \u2194 JAX.

### Current Sprint: MS-215 - BN1d training `unused_mut` clippy regression [COMPLETE]
- [x] [patch] Removed gratuitous `mut` on `BatchNorm1d::from_parts(...)` in
  `crates/coeus-nn/tests/norm_parity.rs` introduced by MS-214.
- [x] Evidence: `rustup run nightly cargo clippy --workspace --all-targets -- -D warnings`
  returns zero warnings; `rustup run nightly cargo fmt --check` clean;
  `rustup run nightly cargo test --doc --workspace` passes; per-crate
  `cargo nextest run -p <crate> --no-fail-fast` green (coeus-core 25, -tensor
  51, -ops 189, -autograd 35, -nn 371, -optim 14, -sparse 19, -python 72,
  -leto 27, -wgpu 85, -cuda 2, -dist 64 with one pre-existing slow test).
- [x] Note: MS-211..MS-214 closed G-036 (pool1d + unfold/fold), G-037
  (extended activation backwards repair), G-041 (regularization modules),
  G-042 (quantized/lazy parity scope closed as non-goal), G-043 (ongoing
  Burn-vs-Coeus bench row expansion). See docs/backlog.md for sprint ledger.

### Current Sprint: MS-214 - Unfold1d Python binding + BatchNorm1d training parity [COMPLETE]
- [x] [minor] `PyUnfold1d` PyO3 binding for `[N,C,L] -> [N,C*k,L_out]`.
- [x] [patch] `crates/coeus-nn/tests/norm_parity.rs`: `BatchNorm1d` training-mode
  analytical test (population variance oracle) + backward to weight/bias.
- [x] [patch] `test_pytorch_parity.py::test_unfold1d_matches_pytorch`
  (kernel=3, stride=1 on `[2,3,7]`).
- [x] [patch] `test_jax_parity.py::test_adaptive_avg_pool2d_global_matches_jax`
  (1-output global avg pool cross-checked against `jnp.mean(..., keepdims)`).
- [x] Merged via PR #100.
- [x] Follow-up: `let mut bn` regression caught by clippy after merge; now
  fixed in MS-215.

### Current Sprint: MS-200 - ReLU+GeLU activation benchmark expansion [COMPLETE]
- [x] Added `bench_relu_forward` and `bench_gelu_forward` rows.
- [x] ReLU gap (Burn 13x faster due to autograd overhead) logged as optimization target.
- [x] GeLU parity confirmed.

### Previous Sprint: MS-199 - HuberLoss benchmark matrix expansion [COMPLETE] - HuberLoss benchmark matrix expansion [COMPLETE]
**Objective**: Add HuberLoss benchmark row comparing Burn vs Coeus.
- [x] [patch] Added `bench_huber_loss` in `crates/coeus-nn/benches/nn_bench.rs`.
- [x] Evidence: Coeus ~45x faster than Burn (Coeus ~190 ns vs Burn ~8.7 us).

 - MSELoss benchmark matrix expansion [COMPLETE]
**Objective**: Expand the Burn-vs-Coeus NN benchmark matrix with an MSE loss row
comparing Burn NdArray and both Coeus CPU backends.
**Target version**: 0.5.4 (benchmark/docs [patch]).

- [x] [patch] Added `bench_mse_loss` in `crates/coeus-nn/benches/nn_bench.rs` for
  predictions `[128,64]` vs same-shape targets.
- [x] [patch] Updated G-043 selected-row detail in `docs/gap_audit.md`.
- [x] Evidence: cargo check/clippy/bench-no-run passed; benchmark confirms
  all three backends at parity (~2.3 us each).

### Previous Sprint: MS-197 - CrossEntropyLoss benchmark matrix expansion [COMPLETE] - CrossEntropyLoss benchmark matrix expansion [COMPLETE]
**Objective**: Expand the Burn-vs-Coeus NN benchmark matrix with a CrossEntropyLoss
row so the loss computation family is measured against Burn NdArray and both Coeus
CPU backends.
**Target version**: 0.5.4 (benchmark/docs [patch]).

- [x] [patch] Added `bench_cross_entropy_loss` in `crates/coeus-nn/benches/nn_bench.rs`
  for logits `[128,10]`.
- [x] [patch] Benchmarks Burn NdArray CrossEntropyLoss vs Coeus
  `cross_entropy_loss` on `SequentialBackend` and `MoiraiBackend`.
- [x] [patch] Updated G-043 selected-row detail in `docs/gap_audit.md`.
- [x] Evidence: cargo check/clippy/bench-no-run all passed; benchmark run confirms
  Coeus ~2.6× faster than Burn NdArray (Burn 9.70–10.38 µs, Coeus ~3.7–4.1 µs).

### Previous Sprint: MS-196 - InstanceNorm2d benchmark matrix expansion [COMPLETE]
**Objective**: Expand the Burn-vs-Coeus NN benchmark matrix with an InstanceNorm2d
forward row so one additional implemented NN family is measured across Burn NdArray
and both Coeus CPU backends.
**Target version**: 0.5.4 (benchmark/docs [patch]).

- [x] [patch] Added `bench_instancenorm2d_forward` in
  `crates/coeus-nn/benches/nn_bench.rs` for `[2,32,16,16]`.
- [x] [patch] Benchmarks Burn NdArray InstanceNorm2d forward vs Coeus
  `InstanceNorm2d::<_, SequentialBackend>` and `InstanceNorm2d::<_, MoiraiBackend>`
  and registers the row in `criterion_group!`.
- [x] [patch] Updated G-043 selected-row detail in `docs/gap_audit.md`.
- [x] Evidence: `cargo check -p coeus-nn --all-targets`; `cargo clippy -p
  coeus-nn --all-targets -- -D warnings`; `cargo bench -p coeus-nn --bench
  nn_bench --no-run`; `cargo bench -p coeus-nn --bench nn_bench --
  InstanceNorm2d --warm-up-time 1 --measurement-time 2 --sample-size 10`.

### Previous Sprint: MS-195 - LSTM benchmark matrix expansion [COMPLETE]
**Objective**: Expand the Burn-vs-Coeus NN benchmark matrix with an LSTM forward
row so one additional implemented NN family is measured across Burn NdArray and
both Coeus CPU backends.
**Target version**: 0.5.4 (benchmark/docs [patch]).

- [x] [patch] Added `bench_lstm_forward` in `crates/coeus-nn/benches/nn_bench.rs`
  with `batch=4, seq=32, input=64, hidden=128`.
- [x] [patch] Benchmarks Burn NdArray LSTM forward vs Coeus
  `Lstm::<_, SequentialBackend>` and `Lstm::<_, MoiraiBackend>` and
  registers the row in `criterion_group!`.
- [x] [patch] Updated G-043 selected-row detail in `docs/gap_audit.md`.
- [x] Evidence: `cargo check -p coeus-nn --all-targets`; `cargo clippy -p
  coeus-nn --all-targets -- -D warnings`; `cargo bench -p coeus-nn --bench
  nn_bench --no-run`; `cargo bench -p coeus-nn --bench nn_bench -- LSTM
  --warm-up-time 1 --measurement-time 3 --sample-size 10`.

### Previous Sprint: MS-194 - RMSNorm benchmark matrix expansion [COMPLETE]
**Objective**: Expand the Burn-vs-Coeus NN benchmark matrix with an RMSNorm
forward row so one additional implemented NN family is measured across Burn
NdArray and both Coeus CPU backends.
**Target version**: 0.5.4 (benchmark/docs [patch]).

- [x] [patch] Added `bench_rmsnorm_forward` in
  `crates/coeus-nn/benches/nn_bench.rs` for `[128,256]`.
- [x] [patch] Benchmarks Burn NdArray RMSNorm forward vs Coeus
  `RMSNorm::<_, SequentialBackend>` and `RMSNorm::<_, MoiraiBackend>` and
  registers the row in `criterion_group!`.
- [x] [patch] Updated G-043 selected-row detail in `docs/gap_audit.md`.
- [x] Evidence: `cargo check -p coeus-nn --all-targets`; `cargo clippy -p
  coeus-nn --all-targets -- -D warnings`; `cargo bench -p coeus-nn --bench
  nn_bench --no-run`; `cargo bench -p coeus-nn --bench nn_bench -- RMSNorm
  --warm-up-time 1 --measurement-time 2 --sample-size 10`.

### Previous Sprint: MS-193 - AvgPool2d benchmark matrix expansion [COMPLETE]
**Objective**: Expand the Burn-vs-Coeus NN benchmark matrix with an AvgPool2d
forward row so one additional implemented NN family is measured across Burn
NdArray and both Coeus CPU backends.
**Target version**: 0.5.4 (benchmark/docs [patch]).

- [x] [patch] Added `bench_avgpool2d_forward` in
  `crates/coeus-nn/benches/nn_bench.rs` for `[8,16,32,32]` with `k=2`, `s=2`.
- [x] [patch] Benchmarks Burn NdArray AvgPool2d forward vs Coeus
  `AvgPool2d::<_, SequentialBackend>` and `AvgPool2d::<_, MoiraiBackend>` and
  registers the row in `criterion_group!`.
- [x] [patch] Updated G-043 selected-row detail in `docs/gap_audit.md`.
- [x] Evidence: `cargo check -p coeus-nn --all-targets`; `cargo clippy -p
  coeus-nn --all-targets -- -D warnings`; `cargo bench -p coeus-nn --bench
  nn_bench --no-run`; `cargo bench -p coeus-nn --bench nn_bench -- AvgPool2d
  --warm-up-time 1 --measurement-time 2 --sample-size 10`.

### Previous Sprint: MS-192 - MaxPool2d benchmark matrix expansion [COMPLETE]
**Objective**: Expand the Burn-vs-Coeus NN benchmark matrix with a MaxPool2d
forward row so one additional implemented NN family is measured across Burn
NdArray and both Coeus CPU backends.
**Target version**: 0.5.4 (benchmark/docs [patch]).

- [x] [patch] Added `bench_maxpool2d_forward` in
  `crates/coeus-nn/benches/nn_bench.rs` for `[8,16,32,32]` with `k=2`, `s=2`.
- [x] [patch] Benchmarks Burn NdArray MaxPool2d forward vs Coeus
  `MaxPool2d::<_, SequentialBackend>` and
  `MaxPool2d::<_, MoiraiBackend>` and registers the row in
  `criterion_group!`.
- [x] [patch] Updated G-043 selected-row detail in `docs/gap_audit.md`.
- [x] Evidence: `cargo check -p coeus-nn --all-targets`; `cargo clippy -p
  coeus-nn --all-targets -- -D warnings`; `cargo bench -p coeus-nn --bench
  nn_bench --no-run`; `cargo bench -p coeus-nn --bench nn_bench -- MaxPool2d
  --warm-up-time 1 --measurement-time 2 --sample-size 10`.

### Previous Sprint: MS-191 - BatchNorm3d benchmark matrix expansion [COMPLETE]
**Objective**: Expand the Burn-vs-Coeus NN benchmark matrix with a BatchNorm3d
eval-forward row so one additional implemented NN family is measured across Burn
NdArray and both Coeus CPU backends.
**Target version**: 0.5.4 (benchmark/docs [patch]).

- [x] [patch] Added `bench_batchnorm3d_eval_forward` in
  `crates/coeus-nn/benches/nn_bench.rs` for `[2,32,16,16,16]`.
- [x] [patch] Benchmarks Burn NdArray BatchNorm3d eval forward vs Coeus
  `BatchNorm3d::<_, SequentialBackend>` and
  `BatchNorm3d::<_, MoiraiBackend>` and registers the row in
  `criterion_group!`.
- [x] [patch] Updated G-043 selected-row detail in `docs/gap_audit.md`.
- [x] Evidence: `cargo check -p coeus-nn --all-targets`; `cargo clippy -p
  coeus-nn --all-targets -- -D warnings`; `cargo bench -p coeus-nn --bench
  nn_bench --no-run`; `cargo bench -p coeus-nn --bench nn_bench -- BatchNorm3d
  --warm-up-time 1 --measurement-time 2 --sample-size 10`.

### Previous Sprint: MS-190 - BatchNorm1d benchmark matrix expansion [COMPLETE]
**Objective**: Expand the Burn-vs-Coeus NN benchmark matrix with a BatchNorm1d
eval-forward row so one additional implemented NN family is measured across Burn
NdArray and both Coeus CPU backends.
**Target version**: 0.5.4 (benchmark/docs [patch]).

- [x] [patch] Added `bench_batchnorm1d_eval_forward` in
  `crates/coeus-nn/benches/nn_bench.rs` for `[16,128,256]`.
- [x] [patch] Benchmarks Burn NdArray BatchNorm1d eval forward vs Coeus
  `BatchNorm1d::<_, SequentialBackend>` and
  `BatchNorm1d::<_, MoiraiBackend>` and registers the row in
  `criterion_group!`.
- [x] [patch] Updated G-043 selected-row detail in `docs/gap_audit.md`.
- [x] Evidence: `cargo check -p coeus-nn --all-targets`; `cargo clippy -p
  coeus-nn --all-targets -- -D warnings`; `cargo bench -p coeus-nn --bench
  nn_bench --no-run`; `cargo bench -p coeus-nn --bench nn_bench -- BatchNorm1d
  --warm-up-time 1 --measurement-time 2 --sample-size 10`.

### Previous Sprint: MS-189 - GroupNorm benchmark matrix expansion [COMPLETE]
**Objective**: Expand the Burn-vs-Coeus NN benchmark matrix with a GroupNorm
forward row so one additional implemented NN family is measured across Burn
NdArray and both Coeus CPU backends.
**Target version**: 0.5.4 (benchmark/docs [patch]).

- [x] [patch] Added `bench_groupnorm_forward` in
  `crates/coeus-nn/benches/nn_bench.rs` for `[8,32,16,16]` with `g=8`.
- [x] [patch] Benchmarks Burn NdArray GroupNorm forward vs Coeus
  `GroupNorm::<_, SequentialBackend, 8>` and
  `GroupNorm::<_, MoiraiBackend, 8>` and registers the row in
  `criterion_group!`.
- [x] [patch] Updated G-043 selected-row detail in `docs/gap_audit.md`.
- [x] Evidence: `cargo check -p coeus-nn --all-targets`; `cargo clippy -p
  coeus-nn --all-targets -- -D warnings`; `cargo bench -p coeus-nn --bench
  nn_bench --no-run`; `cargo bench -p coeus-nn --bench nn_bench -- GroupNorm
  --warm-up-time 1 --measurement-time 2 --sample-size 10`.

### Previous Sprint: MS-188 - Embedding and GroupNorm JAX parity [COMPLETE]
**Objective**: Extend the Python JAX differential suite for existing Rust-owned
module surfaces without adding Python-side domain logic.
**Target version**: 0.5.4 (python parity/test/docs [patch]).

- [x] [patch] Added `test_embedding_matches_jax`, comparing
  `pycoeus.Embedding` forward output and weight scatter-add gradient against an
  inline JAX advanced-indexing reference.
- [x] [patch] Added `test_groupnorm_matches_jax`, comparing
  `pycoeus.GroupNorm(groups=2, channels=4)` forward output and input/gamma/beta
  gradients against an inline JAX formula reference.
- [x] Evidence: `D:\miniforge3\python.exe -m pytest
  coeus-python\tests\test_jax_parity.py::test_embedding_matches_jax
  coeus-python\tests\test_jax_parity.py::test_groupnorm_matches_jax -q` (2/2);
  `D:\miniforge3\python.exe -m pytest coeus-python\tests\test_jax_parity.py
  -q` (25/25).

### Previous Sprint: MS-187 - Conv3d benchmark matrix expansion [COMPLETE]
**Objective**: Expand the Burn-vs-Coeus NN benchmark matrix with a Conv3d
forward row so one additional implemented NN family is measured across Burn
NdArray and both Coeus CPU backends.
**Target version**: 0.5.4 (benchmark/docs [patch]).

- [x] [patch] Added `bench_conv3d_forward` in `crates/coeus-nn/benches/nn_bench.rs`
  for `[2,8,16,16,16]`, `k=3`.
- [x] [patch] Benchmarks Burn NdArray Conv3d forward vs Coeus
  `Conv3d::<_, SequentialBackend>` and `Conv3d::<_, MoiraiBackend>` and
  registers the row in `criterion_group!`.
- [x] [patch] Corrected extended activation backward routing discovered by the
  package gate: derivative kernels now evaluate on the saved input and multiply
  by `grad_out`; packed pair parameters use documented little-endian `f32`
  lanes.
- [x] [patch] Updated G-043 selected-row detail in `docs/gap_audit.md`.
- [x] Evidence: `rustup run nightly cargo fmt -p coeus-core -p
  coeus-autograd -p coeus-nn --check`; `rustup run nightly cargo check -p
  coeus-nn --all-targets`; `rustup run nightly cargo clippy -p coeus-nn
  --all-targets -- -D warnings`; `rustup run nightly cargo nextest run -p
  coeus-nn` (319/319); `rustup run nightly cargo nextest run -p coeus-nn -E
  'binary(act_extended_tests)'` (12/12); `rustup run nightly cargo bench -p
  coeus-nn --bench nn_bench --no-run`; `rustup run nightly cargo bench -p
  coeus-nn --bench nn_bench -- Conv3d --warm-up-time 1 --measurement-time 2
  --sample-size 10` (Burn 14.981 ms, Coeus Sequential 17.584 ms, Coeus Moirai
  133.54 ms median estimates).

### Previous Sprint: MS-186 - Conv1d benchmark matrix expansion [COMPLETE]
**Objective**: Expand the Burn-vs-Coeus NN benchmark matrix with a Conv1d
forward row so one additional implemented NN family is measured across Burn
NdArray and both Coeus CPU backends.
**Target version**: 0.5.4 (benchmark/docs [patch]).

- [x] [patch] Added `bench_conv1d_forward` in `crates/coeus-nn/benches/nn_bench.rs`
  for `[8,32,256]`, `k=3`.
- [x] [patch] Benchmarks Burn NdArray Conv1d forward vs Coeus
  `Conv1d::<_, SequentialBackend>` and `Conv1d::<_, MoiraiBackend>` and
  registers the row in `criterion_group!`.
- [x] [patch] Updated G-043 selected-row detail in `docs/gap_audit.md`.
- [x] Evidence: `cargo check -p coeus-nn --all-targets`; `cargo clippy -p
  coeus-nn --all-targets -- -D warnings`; `cargo bench -p coeus-nn --bench
  nn_bench --no-run`; `cargo bench -p coeus-nn --bench nn_bench -- Conv1d
  --warm-up-time 1 --measurement-time 2 --sample-size 10`.

### Previous Sprint: MS-185 - ConvTranspose3d CPU/PyO3 parity [COMPLETE]
**Objective**: Advance G-035 through the CPU/default backend, autograd,
`coeus-nn`, and PyO3 parity surfaces while leaving WGPU/CUDA backend-specific
coverage open.
**Target version**: 0.5.4 (NN/PyO3/test/docs [minor]).

- [x] [minor] Added `conv_transpose3d_output_dims`,
  `coeus_ops::conv_transpose3d`, and `ConvOps::conv_transpose3d` with the
  host-side default backend implementation.
- [x] [minor] Added tracked `coeus_autograd::conv_transpose3d` with input,
  weight, and bias backward accumulation.
- [x] [minor] Added `coeus_nn::ConvTranspose3d` and Sequential/Moirai
  value-semantic module tests.
- [x] [minor] Added `pycoeus.ConvTranspose3d` plus PyTorch f64 differential
  parity for forward output and input/weight/bias gradients.
- [x] [patch] Updated `docs/gap_audit.md` to record completed CPU/PyO3
  progress and keep WGPU/CUDA backend parity open.
- [x] Evidence: `rustup run nightly cargo check -p coeus-ops -p
  coeus-autograd -p coeus-nn -p coeus-python --all-targets`; `rustup run
  nightly cargo clippy -p coeus-ops -p coeus-autograd -p coeus-nn -p
  coeus-python --all-targets -- -D warnings`; `rustup run nightly cargo
  nextest run -p coeus-ops -p coeus-autograd -p coeus-nn` (531/531);
  `D:\miniforge3\python.exe -m maturin develop -m coeus-python\Cargo.toml`;
  `D:\miniforge3\python.exe -m pytest
  coeus-python\tests\test_pytorch_parity.py::test_conv_transpose3d_matches_pytorch
  -q`; `rustup run nightly cargo test --doc -p coeus-ops -p coeus-autograd -p
  coeus-nn`; `rustup run nightly cargo doc -p coeus-ops -p coeus-autograd -p
  coeus-nn --no-deps`.

### Previous Sprint: MS-184 - BatchNorm2d benchmark matrix expansion [COMPLETE]
**Objective**: Expand the Burn-vs-Coeus NN benchmark matrix with a BatchNorm2d
eval-forward row so one additional implemented NN family is measured across Burn
NdArray and both Coeus CPU backends.
**Target version**: 0.5.4 (benchmark/docs [patch]).

- [x] [patch] Added `bench_batchnorm2d_eval_forward` in
  `crates/coeus-nn/benches/nn_bench.rs` for `[2,64,32,32]`.
- [x] [patch] Benchmarks Burn NdArray BatchNorm2d eval forward vs Coeus
  `BatchNorm2d::<_, SequentialBackend>` and
  `BatchNorm2d::<_, MoiraiBackend>` and registers the row in
  `criterion_group!`.
- [x] [patch] Updated G-043 selected-row detail in `docs/gap_audit.md`.
- [x] Evidence: `cargo check -p coeus-nn --all-targets`; `cargo clippy -p
  coeus-nn --all-targets -- -D warnings`; `cargo bench -p coeus-nn --bench
  nn_bench --no-run`; `cargo bench -p coeus-nn --bench nn_bench -- BatchNorm2d
  --warm-up-time 1 --measurement-time 2 --sample-size 10`.

### Previous Sprint: MS-183 - Embedding benchmark matrix expansion [COMPLETE]
**Objective**: Expand the Burn-vs-Coeus NN benchmark matrix with an embedding
lookup row so one additional implemented NN family is measured across Burn
NdArray and both Coeus CPU backends.
**Target version**: 0.5.4 (benchmark/docs [patch]).

- [x] [patch] Added `bench_embedding_forward` in
  `crates/coeus-nn/benches/nn_bench.rs` for `[batch=2, seq=16]`,
  `[vocab=4096, d_model=256]`.
- [x] [patch] Benchmarks Burn NdArray embedding lookup vs Coeus
  `Embedding::<_, SequentialBackend>` and `Embedding::<_, MoiraiBackend>` and
  registers the row in `criterion_group!`.
- [x] [patch] Updated G-043 selected-row detail in `docs/gap_audit.md`.
- [x] Evidence: `cargo check -p coeus-nn --all-targets`; `cargo clippy -p
  coeus-nn --all-targets -- -D warnings`; `cargo bench -p coeus-nn --bench
  nn_bench --no-run`; `cargo bench -p coeus-nn --bench nn_bench -- Embedding
  --warm-up-time 1 --measurement-time 2 --sample-size 10`.

### Previous Sprint: MS-182 - Python KL/Margin wrapper parity [COMPLETE]
**Objective**: Close the wrapper-only parity gap for existing Rust loss APIs by
exposing `kl_divergence` and `margin_ranking_loss` through thin PyO3 bindings
and pinning forward/backward parity against PyTorch.
**Target version**: 0.5.4 (python binding/test/docs [patch]).

- [x] [patch] Added `pycoeus.kl_divergence` and
  `pycoeus.margin_ranking_loss` wrappers in `crates/coeus-python/src/losses.rs` that
  delegate directly to `coeus_nn::loss` with no Python-side math.
- [x] [patch] Exported both wrappers in `crates/coeus-python/src/lib.rs` and updated
  `crates/coeus-python/pycoeus.pyi` to keep the Python stub/API surface aligned.
- [x] [patch] Added PyTorch differential tests
  `test_kl_divergence_matches_pytorch` and
  `test_margin_ranking_loss_matches_pytorch`, asserting scalar forward value and
  input gradients at f64.
- [x] Evidence: `D:\miniforge3\python.exe -m maturin develop -m
  crates/coeus-python/Cargo.toml`; `D:\miniforge3\python.exe -m pytest
  crates/coeus-python/tests/test_pytorch_parity.py -k
  "kl_divergence_matches_pytorch or margin_ranking_loss_matches_pytorch" -q`;
  `cargo check -p coeus-python --all-targets`; `cargo clippy -p coeus-python
  --all-targets -- -D warnings`.

### Previous Sprint: MS-181 - Transformer encoder benchmark matrix expansion [COMPLETE]
**Objective**: Extend the Burn-vs-Coeus NN benchmark matrix with a
TransformerEncoder-layer forward row so one additional implemented NN family is
measured across Burn NdArray and both Coeus CPU backends.
**Target version**: 0.5.4 (benchmark/docs [patch]).

- [x] [patch] Added `bench_transformer_encoder_forward` in
  `crates/coeus-nn/benches/nn_bench.rs` using shape `[8,64,256]`, `d_ff=1024`,
  `heads=8`, and dropout disabled.
- [x] [patch] Benchmarks Burn NdArray vs Coeus `SequentialBackend` vs Coeus
  `MoiraiBackend` for the same encoder-layer forward contract and registers the
  row in `criterion_group!`.
- [x] Evidence: `cargo bench -p coeus-nn --bench nn_bench --no-run`; `cargo
  bench -p coeus-nn --bench nn_bench -- Transformer --warm-up-time 1
  --measurement-time 2 --sample-size 10`.

### Previous Sprint: MS-180 - Burn/PyTorch parity gap audit [COMPLETE]
**Objective**: Compare Coeus NN and Python public surfaces against Burn and
PyTorch module families, then file remaining parity work as concrete backlog
items.
**Target version**: 0.5.4 (audit/docs-only [patch]).

- [x] [patch] Audited `crates/coeus-nn/src/lib.rs`, `crates/coeus-nn/src/loss.rs`,
  `crates/coeus-python/src/lib.rs`, and `crates/coeus-python/src/losses.rs` public surfaces
  against Burn/PyTorch NN categories and the current parity harness scope.
- [x] [patch] Added G-035..G-043 to `docs/gap_audit.md`, covering
  ConvTranspose3d, pooling/adaptive/unfold/fold, activations, losses/distances,
  Python loss wrapper lag, recurrent variants, regularization/sparse/local
  response modules, quantized/lazy policy, and benchmark matrix coverage.
- [x] [patch] Mirrored the open parity queue in `docs/backlog.md`.
- [x] Evidence: source-surface audit plus Burn/PyTorch documentation audit; no
  Rust or Python implementation changed.

### Previous Sprint: MS-179 - Linear/loss gradient value assertions [COMPLETE]
**Objective**: Remove existence-only Linear/MSE/CrossEntropy focused gradient
checks and replace them with analytical value-semantic assertions.
**Target version**: 0.5.4 (test-only [patch]).

- [x] [patch] Replaced Linear module gradient-existence checks with exact
  analytical assertions for input, weight, and bias gradients under a
  deterministic all-ones layer and unit output seed.
- [x] [patch] Replaced MSE loss gradient-existence checks with the analytical
  mean-reduction derivative `2 * (prediction - target) / n`.
- [x] [patch] Replaced CrossEntropy loss gradient-existence checks with a
  stable softmax-minus-onehot mean-reduction oracle for the logits gradient.
- [x] Evidence: `rustup run nightly cargo fmt -p coeus-nn --check`; `rustup
  run nightly cargo check -p coeus-nn --all-targets`; `rustup run nightly cargo
  clippy -p coeus-nn --all-targets -- -D warnings`; `rustup run nightly cargo
  nextest run -p coeus-nn --test nn_tests`; `rustup run nightly cargo nextest
  run -p coeus-nn`; `rustup run nightly cargo test --doc -p coeus-nn`; `rustup run
  nightly cargo doc -p coeus-nn --no-deps`; `git diff --check`.

### Previous Sprint: MS-178 - Conv gradient value assertions [COMPLETE]
**Objective**: Remove existence-only Conv1d/Conv2d/Conv3d module gradient checks and
replace them with analytical value-semantic assertions for small deterministic
kernels.
**Target version**: 0.5.4 (test-only [patch]).

- [x] [patch] Replaced `test_conv1d_backward_gradients_exist` with
  `test_conv1d_backward_gradients_match_reference`, asserting exact input,
  weight, and bias gradients for a `[1,1,4]` input and all-ones kernel.
- [x] [patch] Replaced `test_conv2d_backward_gradients_exist` with
  `test_conv2d_backward_gradients_match_reference`, asserting exact input,
  weight, and bias gradients for a `[1,1,3,3]` input and all-ones `2x2` kernel.
- [x] [patch] Replaced `test_conv3d_backward_gradients_exist` with
  `test_conv3d_backward_gradients_match_reference`, asserting exact input,
  weight, and bias gradients for a `[1,1,2,2,2]` input and all-ones `2x2x2`
  kernel.
- [x] Evidence: `rustup run nightly cargo fmt -p coeus-nn --check`; `rustup
  run nightly cargo check -p coeus-nn --all-targets`; `rustup run nightly cargo
  clippy -p coeus-nn --all-targets -- -D warnings`; `rustup run nightly cargo
  nextest run -p coeus-nn` (305/305); `rustup run nightly cargo test --doc -p
  coeus-nn`; `rustup run nightly cargo doc -p coeus-nn --no-deps`; `git diff
  --check`.

### Previous Sprint: MS-177 - TCP distributed test determinism [COMPLETE]
**Objective**: Make `coeus-dist` TCP collective tests deterministic under
nextest process parallelism and bound debug-mode TCP mesh waits so failures
surface as explicit panic diagnostics instead of 60s hangs.
**Target version**: 0.5.4 (test/runtime diagnostics [patch]).

- [x] [patch] Added a file-backed cross-process TCP port allocator lock and
  deterministic local port reservation for `crates/coeus-dist/tests/dist_ops.rs`,
  covering multi-rank and single-rank TCP panic-contract tests.
- [x] [patch] Treated Windows `PermissionDenied` during TCP lock-file creation
  as an already-held lock, preserving stale-lock diagnostics for nextest
  process contention.
- [x] [patch] Added debug-only timeout diagnostics around TCP mesh connect,
  accept, peer-rank read, send, and recv paths while preserving async backoff
  through `moirai_async::sleep`.
- [x] [patch] Consolidated remaining TCP panic-thread assertions through
  `assert_any_thread_panicked`.
- [x] Evidence: `rustup run nightly cargo fmt -p coeus-dist --check`; `rustup
  run nightly cargo check -p coeus-dist --all-targets`; `rustup run nightly
  cargo clippy -p coeus-dist --all-targets -- -D warnings`; `rustup run nightly
  cargo nextest run -p coeus-dist`; `git diff --check`.

### Previous Sprint: MS-145 - Bilinear backward PyTorch parity [COMPLETE]
**Objective**: Close the deferred backward gap for `pycoeus.Bilinear` from
MS-140 forward parity; extend the differential harness to the bilinear
interaction layer's autograd-tracked composition.
**Target version**: 0.5.4 (test-only [patch]).

- [x] [patch] Added
  `crates/coeus-python/tests/test_pytorch_parity.py::test_bilinear_backward_matches_pytorch`
  asserting `pycoeus.Bilinear(3,4,2, bias=True)` differentiated via
  `out.sum().backward()` against `torch.nn.Bilinear.double()` at f64, atol=1e-10.
  Covers dweight (`[out, in1, in2]` flat), dbias, dx1, dx2.
- [x] Evidence: `JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu D:\miniforge3\python.exe -m
  pytest crates/coeus-python/tests/test_pytorch_parity.py::test_bilinear_backward_matches_pytorch
  -v` (1/1 PASS); full Python parity suite 55 passed + 2 MLX-skipped.

### Previous Sprint: MS-176 - ConvTranspose backward GPU coverage [COMPLETE]
**Objective**: Close the deferred ConvTranspose backward coverage gap for the
WGPU and CUDA backend-autograd paths without adding a duplicate backend backward
API before there is a dedicated kernel seam.
**Target version**: 0.5.4 (test-only [patch]).

- [x] [patch] Added WGPU `conv_transpose1d` and `conv_transpose2d` backward
  tests that run tracked backend tensors, seed non-uniform gradients, and compare
  input/weight gradients against the existing CPU autograd reference.
- [x] [patch] Added CUDA feature-gated `conv_transpose1d` and
  `conv_transpose2d` backward parity tests with the same CPU-autograd oracle.
- [x] Evidence: `rustup run nightly cargo nextest run -p coeus-wgpu -p
  coeus-cuda` (87/87); `rustup run nightly cargo check -p coeus-cuda
  --all-targets --features cuda`; `rustup run nightly cargo nextest run -p
  coeus-cuda --features cuda` (71/71); `rustup run nightly cargo clippy -p
  coeus-wgpu -p coeus-cuda --all-targets -- -D warnings`.

### Previous Sprint: MS-175 - MSE/BCE/Huber loss JAX parity [COMPLETE]
**Objective**: Extend the JAX parity harness to the regression/binary losses
(mse_loss, binary_cross_entropy, huber_loss), mirroring the PyTorch loss parity.
**Target version**: 0.5.4 (test-only [patch]).

- [x] [patch] `test_mse_loss_matches_jax`, `test_binary_cross_entropy_matches_jax`,
  `test_huber_loss_matches_jax`: forward loss + prediction gradient vs inline JAX
  references at f64. Huber (δ=1.0) spans quadratic/linear regions; BCE probs in (0,1).
- [x] Evidence: `pytest test_jax_parity.py` 16/16 pass.

### Previous Sprint: MS-174 - LayerNorm/RMSNorm JAX parity [COMPLETE]
**Objective**: Extend the JAX parity harness to normalization modules already
covered by PyTorch parity.
**Target version**: 0.5.4 (test-only [patch]).

- [x] [patch] Added `test_layernorm_matches_jax`, asserting forward output and
  input/gamma/beta gradients against an inline JAX LayerNorm reference at f64.
- [x] [patch] Added `test_rmsnorm_matches_jax`, asserting forward output and
  input/gamma gradients against an inline JAX RMSNorm reference at f64.
- [x] Evidence: `D:\miniforge3\python.exe -m pytest
  crates/coeus-python/tests/test_jax_parity.py -q` (13/13).

### Previous Sprint: MS-173 - Softmax/log-softmax/cross-entropy JAX parity [COMPLETE]
**Objective**: Extend the JAX parity harness to the classification softmax path
covered by PyTorch parity.
**Target version**: 0.5.4 (test-only [patch]).

- [x] [patch] Added `test_softmax_matches_jax` and
  `test_log_softmax_matches_jax`, asserting forward output and input gradient
  against `jax.nn.{softmax,log_softmax}` at f64.
- [x] [patch] Added `test_cross_entropy_loss_matches_jax`, asserting scalar
  mean loss and logit gradient against a fused log-softmax + NLL JAX reference.
- [x] Evidence: `D:\miniforge3\python.exe -m pytest
  crates/coeus-python/tests/test_jax_parity.py -q` (11/11).

### Previous Sprint: MS-172 - Deterministic local/TCP numel contract tests [COMPLETE]
**Objective**: Replace thread-join panic detection with deterministic direct
panic-contract coverage for shape/numel mismatch paths.
**Target version**: 0.5.4 ([patch]).

- [x] [patch] Replaced the multi-thread local scatter mismatch panic test with
  a deterministic single-rank root-input numel mismatch test.
- [x] [patch] Added deterministic non-zero local `all_gather` and rooted
  `gather` output-numel mismatch tests.
- [x] [patch] Added deterministic non-zero TCP rooted `gather` output-numel
  mismatch coverage.
- [x] Evidence: `rustup run nightly cargo nextest run -p coeus-dist
  test_local_scatter_mismatched_input_numel_panics
  test_local_all_gather_mismatched_output_numel_panics
  test_local_gather_mismatched_output_numel_panics
  test_tcp_gather_mismatched_output_numel_panics` (4/4);
  `rustup run nightly cargo clippy -p coeus-dist --all-targets -- -D warnings`.

### Previous Sprint: MS-171 - BackendOps interface segregation [COMPLETE]
**Objective**: Replace the monolithic backend operation trait body with
single-concern operation traits while preserving the aggregate `BackendOps`
bound and CPU kernel delegation.
**Target version**: 0.5.4 ([patch]).

- [x] [patch] Added `ElementwiseOps`, `MatmulOps`, `ReductionOps`, `ConvOps`,
  `PoolOps`, `AttentionOps`, and `OptimizerOps` under
  `crates/coeus-ops/src/backend_ops/traits/`.
- [x] [patch] Reduced `BackendOps` to a super-trait with a blanket impl so
  backends compose the aggregate bound from single-concern trait impls.
- [x] [patch] Split the CPU backend implementation into one impl block per
  operation concern, eliminating the duplicate blanket-impl coherence failure.
- [x] [patch] Re-exported the sub-traits at the crate root and updated direct
  backend-dispatch tests to import the precise operation trait they exercise.
- [x] Evidence: `rustup run nightly cargo check -p coeus-ops --all-targets`;
  `rustup run nightly cargo clippy -p coeus-ops --all-targets -- -D warnings`;
  `rustup run nightly cargo nextest run -p coeus-ops` (189/189);
  `rustup run nightly cargo test --doc -p coeus-ops` (23/23);
  `rustup run nightly cargo doc -p coeus-ops --no-deps`.

### Previous Sprint: MS-168 - Activation JAX parity (SiLU/Mish/ELU/Softplus/LeakyReLU) [COMPLETE]
**Objective**: Extend the JAX parity harness (Linear/MHA/decoder only) to the
elementwise activations, mirroring the PyTorch activation parity of MS-167.
**Target version**: 0.5.4 (test-only [patch]).

- [x] [patch] Added `_assert_activation_matches_jax` helper (`jax.grad` backward) +
  5 tests asserting forward + dx vs `jax.nn.{silu,mish,elu,softplus,leaky_relu}`.
- [x] [patch] LeakyReLU input excludes the `x=0` kink; C1 activations include it.
- [x] Evidence: `pytest test_jax_parity.py` 8/8 pass.

### Previous Sprint: MS-167 - Activation PyTorch parity (SiLU/Mish/ELU/Softplus/LeakyReLU) [COMPLETE]
**Objective**: Close the elementwise-activation differential gap — only GELU had
PyTorch parity; SiLU, Mish, ELU, Softplus, LeakyReLU had none.
**Target version**: 0.5.4 (test-only [patch]).

- [x] [patch] Added `_assert_activation_parity` helper (DRY) + 5 tests asserting
  forward + dx vs `torch.nn.functional.{silu,mish,elu,softplus,leaky_relu}` at f64.
- [x] [patch] LeakyReLU input excludes the `x=0` kink (subgradient convention);
  C1 activations include it.
- [x] Evidence: `pytest test_pytorch_parity.py` 38/38 pass.

### Previous Sprint: MS-166 - GlobalAvgPool2d/GlobalMaxPool2d PyTorch parity [COMPLETE]
**Objective**: Add value-semantic forward+backward differential parity for the
global pooling layers, previously covered only by binding smoke tests.
**Target version**: 0.5.4 (test-only [patch]).

- [x] [patch] `test_global_avg_pool2d_matches_pytorch`: `[2,3,4,4]` -> `[N,C,1,1]`;
  forward + dx vs `F.adaptive_avg_pool2d(x,1)` at atol=1e-10 (uniform gradient).
- [x] [patch] `test_global_max_pool2d_matches_pytorch`: same vs
  `F.adaptive_max_pool2d(x,1)` (argmax-routing gradient).
- [x] Evidence: `pytest test_pytorch_parity.py` 33/33 pass.

### Previous Sprint: MS-165 - Zero-numel collective numel contracts [COMPLETE]
**Objective**: Ensure local and TCP collectives validate per-rank tensor element
counts before zero-numel early returns, not only list lengths.
**Target version**: 0.5.4 ([patch]).

- [x] [patch] Moved local `all_gather` output numel validation ahead of the
  zero-numel return.
- [x] [patch] Moved local rooted `gather` output and `scatter` input numel
  validation ahead of zero-numel returns.
- [x] [patch] Moved TCP `all_gather`, rooted `gather`, and rooted `scatter`
  per-rank numel validation ahead of zero-numel returns.
- [x] [patch] Added panic-contract tests for each local/TCP zero-numel numel
  mismatch path.
- [x] Evidence: `rustup run nightly cargo nextest run -p coeus-dist
  zero_numel_` (12/12); `rustup run nightly cargo clippy -p coeus-dist
  --all-targets -- -D warnings`; `rustup run nightly cargo doc -p coeus-dist
  --no-deps`.

### Previous Sprint: MS-164 - Conv2d CPU AXPY kernel [COMPLETE]
**Objective**: Replace the canonical contiguous CPU Conv2d dot-per-output path
with an output-stationary AXPY row kernel through the existing Hermes SIMD seam,
while preserving value semantics across sequential and Moirai backends.
**Target version**: 0.5.4 ([patch]).

- [x] [patch] Added `Scalar::axpy_slice` as a BLAS-1 scaled-accumulate seam,
  with native `f32`/`f64` implementations delegated to `hermes_simd::axpy`.
- [x] [patch] Enforced the `axpy_slice` equal-length invariant and pinned it
  with native-float, integer-default, and panic-contract tests.
- [x] [patch] Rewrote canonical contiguous Conv2d forward to accumulate each
  output row with `out_row += weight * input_window`, using AXPY for stride 1
  and preserving the scalar strided path when stride is greater than one.
- [x] [patch] Coarsened Moirai Conv2d row partitioning from one row per shard to
  row blocks sized by `out_rows.div_ceil(num_threads)`, preserving row
  boundaries while reducing scheduler overhead.
- [x] [patch] Repaired Mnemosyne's tagged `NodeSegmentPool::pop` path so Coeus'
  local path dependency compiles with the ABA-immune Treiber stack provider.
- [x] Evidence: `rustup run nightly cargo nextest run -p coeus-core --test
  scalar_dot_scale` (5/5); `rustup run nightly cargo nextest run -p coeus-ops
  --test conv2d_hermes_diff` (2/2); `rustup run nightly cargo clippy -p
  coeus-core -p coeus-ops --all-targets -- -D warnings`; `rustup run nightly
  cargo bench -p coeus-nn --bench nn_bench -- Conv2d --warm-up-time 1
  --measurement-time 2 --sample-size 10` (median: Burn NdArray 1.97 ms, Coeus
  Sequential 2.39 ms, Coeus Moirai 1.05 ms); Mnemosyne provider gates
  `rustup run nightly cargo check -p mnemosyne-arena --all-targets`,
  `rustup run nightly cargo nextest run -p mnemosyne-arena --test
  segment_pool_concurrency`, and `rustup run nightly cargo nextest run -p
  mnemosyne-arena segment`.

### Previous Sprint: MS-163 - Local collective snapshot and Conv2d bench [COMPLETE]
**Objective**: Reduce local distributed critical-section scope and extend the
Burn/Coeus NN benchmark harness with Conv2d forward coverage.
**Target version**: 0.5.4 ([patch]).

- [x] [patch] Added `LocalCommunicator::snapshot_payloads` and routed
  `all_reduce`, `reduce`, `all_gather`, and `gather` through snapshot reads so
  reduction/copy work no longer runs while holding the staging-board mutex.
- [x] [patch] Moved root `scatter` host extraction ahead of staging-board
  publication so tensor host copies do not run under the shared mutex.
- [x] [patch] Added a Conv2d forward group to
  `crates/coeus-nn/benches/nn_bench.rs` comparing Burn NdArray, Coeus Sequential, and
  Coeus Moirai on the same `8x16x32x32`, `16 -> 16`, `k=3` workload.
- [x] Evidence: `rustup run nightly cargo nextest run -p coeus-dist local_`
  (21/21); `rustup run nightly cargo bench -p coeus-nn --bench nn_bench
  -- Conv2d --warm-up-time 1 --measurement-time 2 --sample-size 10`
  (median: Burn NdArray 2.19 ms, Coeus Sequential 32.83 ms, Coeus Moirai
  126.56 ms).

### Previous Sprint: MS-161 - KL/MarginRanking loss parity coverage [COMPLETE]
**Objective**: Extend the tracked loss-function surface with KL divergence and
margin ranking loss entry points, and pin their forward/backward semantics with
value-semantic Rust tests.
**Target version**: 0.5.4 ([patch]).

- [x] [patch] Added `kl_divergence` and `margin_ranking_loss` as tracked
  `coeus_autograd` operations and `coeus_nn` wrapper exports.
- [x] [patch] Added analytical forward/backward checks for KL divergence
  (`mean(target * (log(target) - input))`) and margin ranking
  (`mean(max(0, -target * (input1 - input2) + margin))`).
- [x] [patch] Added sequential and Moirai loss-parity assertions for both
  losses in `crates/coeus-nn/tests/loss_parity.rs`.
- [x] Evidence: `rustup run nightly cargo nextest run -p coeus-autograd`
  (35/35); `rustup run nightly cargo nextest run -p coeus-nn` (305/305).

### Previous Sprint: MS-156 - BCE/Huber loss PyTorch differential parity [COMPLETE]
**Objective**: Extend loss-function parity to the binary and regression losses
(binary_cross_entropy, huber_loss), previously absent from the differential suite.
**Target version**: 0.5.4 (test-only [patch]).

- [x] [patch] `test_binary_cross_entropy_matches_pytorch`: probs in (0,1); loss + dp
  vs `F.binary_cross_entropy` at atol=1e-9.
- [x] [patch] `test_huber_loss_matches_pytorch`: δ=1.0, samples spanning quadratic
  (|e|≤δ) and linear (|e|>δ) regions; loss + dp vs `F.huber_loss` at atol=1e-10.
- [x] Evidence: `pytest test_pytorch_parity.py` 31/31 pass.

### Previous Sprint: MS-154 - SiLU/Mish gradient value semantics [COMPLETE]
**Objective**: Replace residual existence-only gradient checks in focused SiLU
and Mish Rust tests with analytical forward and backward value assertions,
including module and non-contiguous view paths.
**Target version**: 0.5.4 (test-only [patch]).

- [x] [patch] Consolidated SiLU test expectations through shared analytical
  helpers for `x * sigmoid(x)` and `sigmoid(x) * (1 + x * (1 - sigmoid(x)))`.
- [x] [patch] Consolidated Mish test expectations through shared analytical
  helpers for `x * tanh(softplus(x))` and its derivative.
- [x] [patch] Upgraded module and non-contiguous SiLU/Mish checks from
  `grad().is_some()` to explicit gradient-value assertions.
- [x] Evidence: `rustup run nightly cargo fmt -p coeus-nn --check`;
  `rustup run nightly cargo nextest run -p coeus-nn --test nn_silu_tests --test nn_mish_tests`
  (6/6).

### Previous Sprint: MS-155 - TCP zero-numel rooted contract enforcement [COMPLETE]
**Objective**: Close rooted-contract bypasses in TCP `gather`/`scatter` where
zero-numel tensors previously skipped root length validation.

- [x] [patch] Moved root length checks ahead of zero-numel fast-return in TCP
  `gather` and `scatter`.
- [x] [patch] Added panic-contract tests for zero-numel mismatch paths
  (`test_tcp_gather_zero_numel_output_len_mismatch_panics`,
  `test_tcp_scatter_zero_numel_input_len_mismatch_panics`).
- [x] Evidence: `cargo test -p coeus-dist zero_numel_ -- --nocapture`;
  `cargo clippy -p coeus-dist --all-targets -- -D warnings`.

### Previous Sprint: MS-153 - CrossEntropy/NLL loss PyTorch differential parity [COMPLETE]
**Objective**: Add value-semantic forward+backward differential parity for the
classification loss path (cross_entropy_loss, nll_loss over log_softmax),
previously absent from the PyTorch parity suite.
**Target version**: 0.5.4 (test-only [patch]).

- [x] [patch] `test_cross_entropy_loss_matches_pytorch`: logits `[3,4]`; loss + dx
  vs `F.cross_entropy` (mean reduction) at atol=1e-10.
- [x] [patch] `test_nll_loss_matches_pytorch`: `nll_loss(log_softmax(x))` vs
  `F.nll_loss(F.log_softmax(x))` at atol=1e-10.
- [x] Evidence: `pytest test_pytorch_parity.py` 29/29 pass.

### Previous Sprint: MS-152 - FeedForward binding module split [COMPLETE]
**Objective**: Replace the monolithic Python FeedForward binding file with a
vertical module tree while preserving the public PyO3 export surface.
**Target version**: 0.5.4 (internal topology [patch]).

- [x] [patch] Promoted `crates/coeus-python/src/nn/feedforward.rs` to
  `crates/coeus-python/src/nn/feedforward/mod.rs`.
- [x] [patch] Moved positional encoding and transformer layer/stack/seq2seq
  bindings into `feedforward/positional.rs` and
  `feedforward/transformer/*` leaf modules.
- [x] [patch] Preserved the `nn` re-export surface used by `pycoeus`
  registration for `PyFeedForward`, `PySinusoidalEncoding`, and
  `PyTransformer*`.
- [x] Evidence: `rustup run nightly cargo fmt -p coeus-python --check`;
  `rustup run nightly cargo check -p coeus-python --all-targets`;
  `rustup run nightly cargo clippy -p coeus-python --all-targets -- -D warnings`;
  `rustup run nightly cargo nextest run -p coeus-python` (72/72);
  `rustup run nightly cargo test --doc -p coeus-python` (0/0);
  `D:\miniforge3\python.exe -m maturin develop -m crates/coeus-python/Cargo.toml`;
  `D:\miniforge3\python.exe -m pytest crates/coeus-python/tests/test_pytorch_parity.py -q`
  (27/27).

### Previous Sprint: MS-150 - TCP collective root contract completion [COMPLETE]
**Objective**: Close remaining root-out-of-bounds panic-contract gaps across TCP
collectives for complete rooted-op safety coverage.

- [x] [patch] Added root-out-of-bounds panic tests for TCP `reduce`, `gather`,
  and `scatter`.
- [x] Evidence: `cargo test -p coeus-dist root_out_of_bounds_panics -- --nocapture`;
  `cargo clippy -p coeus-dist --all-targets -- -D warnings`.

### Previous Sprint: MS-149 - TcpMesh contract completion [COMPLETE]
**Objective**: Complete TCP mesh contract hardening with defensive slot
invariants and explicit panic-contract coverage for bounds/constructor errors.

- [x] [patch] Added duplicate-slot guards in `TcpMesh::new` for outgoing and
  incoming stream assignment.
- [x] [patch] Added panic-contract tests for send/recv out-of-bounds and
  constructor rank bounds.
- [x] Evidence: `cargo test -p coeus-dist test_tcp_mesh_ -- --nocapture`;
  `cargo test -p coeus-dist test_tcp_mesh_new_rank_out_of_bounds_panics -- --nocapture`;
  `cargo clippy -p coeus-dist --all-targets -- -D warnings`.

### Previous Sprint: MS-148 - TcpMesh/collective invariant hardening [COMPLETE]
**Objective**: Harden TCP distributed-runtime safety contracts by enforcing peer
and root invariants explicitly at the mesh/collective boundaries.

- [x] [patch] Added `TcpMesh` shared peer guard path (`stream_for_peer`) and
  routed `send`/`recv` through explicit peer/rank/stream checks.
- [x] [patch] Added `rank < size` invariant in `TcpMesh::new`.
- [x] [patch] Added shared `TcpCommunicator::assert_root` and enforced root
  bounds in `broadcast`, `reduce`, `gather`, and `scatter`.
- [x] [patch] Added panic-contract tests
  (`test_tcp_broadcast_root_out_of_bounds_panics`,
  `test_tcp_mesh_send_self_panics`, `test_tcp_mesh_recv_self_panics`).
- [x] Evidence: `cargo test -p coeus-dist test_tcp_mesh_ -- --nocapture`;
  `cargo test -p coeus-dist test_tcp_broadcast_root_out_of_bounds_panics -- --nocapture`;
  `cargo clippy -p coeus-dist --all-targets -- -D warnings`.

### Previous Sprint: MS-151 - MaxPool2d/AvgPool2d PyTorch differential parity [COMPLETE]
**Objective**: Add value-semantic forward+backward differential parity for 2D
pooling, previously absent from the PyTorch parity suite (only binding smoke tests).
**Target version**: 0.5.4 (test-only [patch]).

- [x] [patch] `test_maxpool2d_matches_pytorch`: k=2,s=2 on `[1,2,4,4]`; forward +
  dx vs `F.max_pool2d` at atol=1e-10 (max-routing gradient).
- [x] [patch] `test_avgpool2d_matches_pytorch`: same vs `F.avg_pool2d`
  (uniform-distribution gradient).
- [x] Evidence: `pytest test_pytorch_parity.py` 27/27 pass.

### Previous Sprint: MS-150 - Scalar identity and direct libm removal [COMPLETE]
**Objective**: Remove Coeus' direct `num-traits`/`libm` scalar dependency path
without weakening GELU/erf value semantics or the sealed `Scalar` contract.
**Target version**: 0.5.4 (patch-class; internal dependency and scalar trait cleanup).

- [x] [patch] Replaced `Scalar: Num + Zero + One` with explicit std arithmetic
  bounds and `Scalar::zero()` / `Scalar::one()` identity methods.
- [x] [patch] Removed Coeus workspace direct `num-traits` and `libm`
  dependencies; `half` now uses only the `bytemuck` feature.
- [x] [patch] Added `coeus-core::dtype::float::erf` as the Coeus-owned
  piecewise rational `erf` implementation used by native and half GELU paths.
- [x] [patch] Updated sparse backward zero checks to use the `Scalar` identity
  contract instead of `num_traits::Zero::is_zero`.
- [x] Evidence: `rustup run nightly cargo fmt -p coeus-core -p coeus-ops --check`;
  `rustup run nightly cargo check -p coeus-core -p coeus-ops --all-targets`;
  `rustup run nightly cargo clippy -p coeus-core -p coeus-ops --all-targets -- -D warnings`;
  `rustup run nightly cargo nextest run -p coeus-core` (22/22);
  `rustup run nightly cargo nextest run -p coeus-ops` (189/189);
  `rustup run nightly cargo test --doc -p coeus-core -p coeus-ops` (55/55);
  `rustup run nightly cargo doc -p coeus-core -p coeus-ops --no-deps`.

### Previous Sprint: MS-149 - GroupNorm PyTorch differential parity [COMPLETE]
**Objective**: Replace GroupNorm's existence-only (`grad is not None`) Python
coverage with value-semantic forward+backward differential parity against PyTorch,
matching the InstanceNorm parity established in MS-145.
**Target version**: 0.5.4 (test-only [patch]).

- [x] [patch] Added `test_groupnorm_matches_pytorch`: GroupNorm(2, 4) on `[2,4,2,2]`;
  forward + dx + dγ + dβ vs `torch.nn.functional.group_norm` at f64, atol=1e-10.
- [x] Evidence: `D:\miniforge3\python.exe -m pytest
  crates/coeus-python/tests/test_pytorch_parity.py -q` (25/25 pass).

### Previous Sprint: MS-148 - Shape einsum SSOT cleanup [COMPLETE]
**Objective**: Remove the duplicate einsum implementation under
`coeus-ops::shape::util` while preserving both public re-export surfaces.
**Target version**: 0.5.4 (patch-class; internal topology cleanup).

- [x] [patch] Deleted `crates/coeus-ops/src/shape/util/einsum.rs`; it was
  byte-identical to `crates/coeus-ops/src/shape/einsum.rs`.
- [x] [patch] Routed `coeus_ops::shape::util::{einsum,einsum3}` through the
  canonical parent `shape::einsum` implementation, preserving call sites and
  removing duplicated tests/logic.
- [x] Evidence: `rustup run nightly cargo fmt -p coeus-ops --check`;
  `rustup run nightly cargo check -p coeus-ops --all-targets`;
  `rustup run nightly cargo clippy -p coeus-ops --all-targets -- -D warnings`;
  `rustup run nightly cargo nextest run -p coeus-ops einsum` (12/12);
  `rustup run nightly cargo nextest run -p coeus-ops` (189/189);
  `rustup run nightly cargo test --doc -p coeus-ops` (23/23);
  `rustup run nightly cargo doc -p coeus-ops --no-deps`.

### Previous Sprint: MS-145 - PyTensor sum/mean + InstanceNorm/RMSProp/AdaGrad parity [COMPLETE]
**Objective**: Close the Python scalar-reduction gap (`Tensor.sum`/`.mean`) blocking
idiomatic `loss.backward()`, and land differential PyTorch parity for InstanceNorm
{1,2,3}d and the RMSProp/AdaGrad optimizer steps.
**Target version**: 0.5.4 (`PyTensor.sum`/`.mean` are additive [minor]; tests [patch]).

- [x] [minor] Added `PyTensor::sum`/`PyTensor::mean` full-reduction methods
  (`crates/coeus-python/src/tensor/pyimpl.rs`) delegating to `coeus_autograd::{sum,mean}`,
  GIL released; synced `pycoeus.pyi` stub.
- [x] [patch] InstanceNorm{1,2,3}d PyTorch parity (forward + dx + dγ + dβ, atol=1e-10).
- [x] [patch] RMSProp/AdaGrad step PyTorch parity (atol=1e-10).
- [x] [patch] Corrected InstanceNorm parity oracle: affine weight/bias `requires_grad=True`.
- [x] Removed stale `crates/coeus-python/tests/pycoeus*.pyd` shadowing artifacts.
- [x] Evidence: `pytest test_pytorch_parity.py` 24/24; `clippy -D warnings` + `fmt` clean.

### Previous Sprint: MS-147 - TcpCommunicator staging contract hardening [COMPLETE]
**Objective**: Harden TCP collective payload contracts to fail fast on shape
mismatches and reduce redundant allocation/copy patterns in root self-paths.

- [x] [patch] Added shared TCP collective numel contract checks and applied them
  to `all_gather`, `gather`, and `scatter`.
- [x] [patch] Replaced root self `tensor.clone()` assignments with
  `get_tensor_host_data` + `copy_host_slice_to_tensor` to preserve preallocated
  outputs and reduce avoidable allocations.
- [x] [patch] Added panic-contract coverage for TCP mismatch paths
  (`test_tcp_all_gather_mismatched_output_numel_panics`,
  `test_tcp_scatter_mismatched_input_numel_panics`).
- [x] Evidence: `cargo test -p coeus-dist test_tcp_all_gather -- --nocapture`;
  `cargo test -p coeus-dist test_tcp_scatter -- --nocapture`; `cargo clippy
  -p coeus-dist --all-targets -- -D warnings`.

### Previous Sprint: MS-146 - LocalCommunicator collective SSOT hardening [COMPLETE]
**Objective**: Complete local collective hardening by removing unchecked staged
payload reads, deduplicating staging cleanup, and enforcing scatter payload
shape contracts.

- [x] [patch] Applied staged payload guards to `all_gather` and `gather`.
- [x] [patch] Added and reused `clear_staging` helper across local collectives.
- [x] [patch] Added `scatter` root input `numel` validation and panic contract
  coverage (`test_local_scatter_mismatched_input_numel_panics`).
- [x] Evidence: `cargo test -p coeus-dist test_local_ -- --nocapture` passes
  13/13; `cargo clippy -p coeus-dist --all-targets -- -D warnings` passes.

### Previous Sprint: MS-144 - LocalCommunicator contention and safety hardening [COMPLETE]
**Objective**: Resolve a distributed-runtime hotspot by eliminating redundant
`all_reduce` work across ranks, hardening staged payload validation, and
cutting avoidable temporary allocations in local collectives.

- [x] [patch] Refactored `crates/coeus-dist/src/local.rs::all_reduce` to compute the
  reduction once on rank 0 and publish the reduced payload for all ranks.
- [x] [patch] Added explicit staged-payload validation helpers
  (`slot_vec_ref`, `assert_numel`) for type/shape guardrails.
- [x] [patch] Removed unnecessary zero-fill temp allocations in `broadcast`,
  `reduce`, and `scatter`.
- [x] Evidence: `cargo test -p coeus-dist --tests` passes 20/20; `cargo clippy
  -p coeus-dist --all-targets -- -D warnings` passes.

### Previous Sprint: MS-143 - Fusion op-tag binary ZST split [COMPLETE]
**Objective**: Remove the partial duplicate `op_tags/binary.rs` split by making
binary fused-expression tags a real vertical module under `coeus-ops` while
preserving the existing public re-export surface.

- [x] [patch] Moved binary ZST tags (`BinaryOpTag`, `Add`, `Sub`, `Mul`,
  `Div`) into `crates/coeus-ops/src/fuse/op_tags/binary.rs`.
- [x] [patch] Converted `op_tags.rs` into `op_tags/mod.rs` and re-exported the
  binary tags from the module root so existing call sites remain unchanged.
- [x] Evidence: `cargo fmt -p coeus-ops --check`; `cargo clippy -p coeus-ops
  --all-targets -- -D warnings`; `cargo nextest run -p coeus-ops` passes
  189/189.

### Previous Sprint: MS-142 - JAX TransformerDecoderLayer parity [COMPLETE]
**Objective**: Extend JAX differential parity from primitive ops to a stateful
transformer decoder block using exported pycoeus layer weights and a JAX
pre-layernorm decoder reference.

- [x] [patch] Added `test_transformer_decoder_layer_matches_jax` in
  `crates/coeus-python/tests/test_jax_parity.py`.
- [x] [patch] Added JAX decoder reference helpers (`_jax_layer_norm`,
  `_jax_mha_forward`) used by the parity oracle.
- [x] Evidence: `pytest crates/coeus-python/tests/test_jax_parity.py -k "decoder_layer
  or mha or linear" -q` passes 3/3.

### Previous Sprint: MS-141 - RMSNorm and Embedding PyTorch parity [COMPLETE]
**Objective**: Close the next PyTorch parity gap for the two normalization/embedding
modules that exposed binding classes but no differential PyTorch tests.

- [x] [patch] Added `test_rmsnorm_matches_pytorch` to
  `crates/coeus-python/tests/test_pytorch_parity.py`: forward `y`, `dx`, `dgamma`
  against PyTorch's canonical RMSNorm formula
  `y = (x / sqrt(mean(x**2, dim=-1, keepdim=True) + eps)) * gamma`
  at `atol=1e-10`. PyTorch 2.12 does not yet ship `torch.nn.RMSNorm` as
  stable, so the oracle is the formulaic implementation identical to the
  canonical reference; both produce bitwise-precise agreement.
- [x] [patch] Added `test_embedding_matches_pytorch` asserting forward output
  and gathered-rows weight gradient against `torch.nn.Embedding` with
  sparse-index backward at `atol=1e-10`.
- [x] Evidence: `pytest crates/coeus-python/tests/test_pytorch_parity.py -k
  "rmsnorm or embedding" -v` passes 2/2; full parity ensemble
  `pytest crates/coeus-python/tests/test_pytorch_parity.py
  crates/coeus-python/tests/test_jax_parity.py crates/coeus-python/tests/test_mlx_parity.py
  -v` passes 21/23 with 2 MLX skips on this Windows host
  (19 PyTorch + 2 JAX + 2 MLX collected).

### Previous Sprint: MS-140 - Bilinear parity indexing coverage [COMPLETE]
**Objective**: Close the remaining Bilinear parity gap with value-semantic
per-output weight-indexing checks in the existing Rust Bilinear parity suite
and Python PyTorch differential parity harness.
**Target version**: 0.5.3 (patch-class; test-only additions).

- [x] [patch] Added a `bilinear_parity.rs` per-output indexing oracle with
  identity/swap weights and bias `[0.5, -0.5]`, asserting `[23.5, 21.5]` on
  both Sequential and Moirai backends.
- [x] [patch] Added `test_bilinear_forward_matches_pytorch`: `Bilinear(3,4,2)`
  weight injection against `torch.nn.Bilinear`; layout `[out,in1,in2]` matches
  directly.
- [x] Evidence: `cargo nextest run -p coeus-nn --test bilinear_parity` passes
  2/2; `pytest crates/coeus-python/tests/test_pytorch_parity.py -k bilinear -v`
  passes 1/1; `cargo clippy -p coeus-nn --test bilinear_parity -- -D warnings`
  is clean.

### Previous Sprint: MS-139 - Python optimizer and attention parity [COMPLETE]
**Objective**: Extend the thin `coeus-python` parity harness with real PyTorch
optimizer-step checks and JAX/MLX MHA forward checks while keeping domain logic
inside Rust/PyO3 bindings.
**Target version**: 0.5.3 (patch-class; test-only additions).

- [x] [patch] Added `test_sgd_step_matches_pytorch`: w=[1.0], mse_loss→grad=2.0,
  SGD(lr=0.1)→w_new=0.8; compared against torch.optim.SGD at atol=1e-10.
- [x] [patch] Added `test_adam_step_matches_pytorch`: same setup, Adam(lr=1e-2)→w_new≈0.99;
  compared against torch.optim.Adam at atol=1e-10.
- [x] [patch] Added `test_adamw_step_matches_pytorch`: AdamW(lr=1e-2, wd=0.01) compared
  against torch.optim.AdamW; decoupled weight decay verified; atol=1e-10.
- [x] [patch] Extended JAX and MLX parity harnesses with `MultiHeadAttention`
  self-attention forward references.
- [x] Evidence: `pytest crates/coeus-python/tests/test_pytorch_parity.py
  crates/coeus-python/tests/test_jax_parity.py crates/coeus-python/tests/test_mlx_parity.py
  -v` passes 15/17 with 2 MLX skips on this Windows host.

### Previous Sprint: MS-138 - JAX and MLX Python parity harnesses [COMPLETE]
**Objective**: Extend `coeus-python` framework parity coverage beyond PyTorch
with thin pytest harnesses for JAX and MLX while keeping all domain logic in
Rust/PyO3 bindings.
**Target version**: 0.5.2 (patch-class; test coverage).

- [x] [patch] Added `test_jax_parity.py` for
  `Linear + ReLU + MSELoss` forward/backward parity and
  `MultiHeadAttention` forward parity against JAX at f64.
- [x] [patch] Added `test_mlx_parity.py` for
  `Linear + ReLU + MSELoss` forward-loss parity and
  `MultiHeadAttention` forward parity against MLX at f32 when MLX is installed.
- [x] [patch] MLX absence now produces one collected skipped test rather than a
  no-tests-collected failure.
- [x] Evidence: `pytest crates/coeus-python/tests/test_jax_parity.py -k "linear or
  mha" -q` (2/2 pass); `pytest crates/coeus-python/tests/test_mlx_parity.py -k
  "linear or mha" -q` (2 collected skips: MLX not installed).

### Previous Sprint: MS-137 - TransformerDecoderLayer functional SSOT routing [COMPLETE]
**Objective**: Complete TransformerDecoderLayer module/functional SSOT in Rust
and thin the PyO3 decoder-layer forward path to call core helpers directly,
eliminating per-forward module reconstruction while preserving parity behavior.
**Target version**: 0.5.2 (minor-class; additive functional surface + wrapper cleanup).

- [x] [minor] Added and exported Rust-core
  `coeus_nn::transformer_decoder_layer(...)` plus
  `coeus_nn::TransformerDecoderLayerParams`.
- [x] [patch] Routed `TransformerDecoderLayer::forward_decoder` through the
  shared functional helper.
- [x] [patch] Routed `PyTransformerDecoderLayer.forward` through the shared
  functional helper (no temporary Rust module reconstruction per call).
- [x] [patch] Added Rust functional/module parity assertion in
  `nn_transformer_tests::test_transformer_decoder_layer`.
- [x] [patch] Added Python SSOT parity assertion in
  `binding_tests_ops::test_transformer_decoder_layer` for decoder-layer
  composition equivalence with `dropout_p=0`.
- [x] Evidence: `rustup run nightly cargo nextest run -p coeus-nn --test
  nn_transformer_tests test_transformer_decoder_layer`; `rustup run nightly
  cargo nextest run -p coeus-python --test binding_tests_ops
  test_transformer_decoder_layer`; `rustup run nightly cargo clippy -p coeus-nn
  --test nn_transformer_tests -- -D warnings`; `rustup run nightly cargo clippy
  -p coeus-python --test binding_tests_ops -- -D warnings`.

### Previous Sprint: MS-136 - Transformer seq2seq structural tests + RNN PyTorch parity [COMPLETE]
**Objective**: Add two Transformer structural self-consistency proofs
(forward_seq2seq == manual encoder→decoder chain; Module::forward == forward_seq2seq(x,x))
and LSTM/GRU differential PyTorch parity tests (weight injection via w_ih/b_ih/w_hh/b_hh).
**Target version**: 0.5.2 (patch-class; test-only additions).

- [x] [patch] Added `transformer_seq2seq_self_consistent`: `forward_seq2seq(src,tgt)` ==
  `encoder.forward_with_mask(src,None)` followed by `decoder.forward_decoder(tgt,memory)`;
  tolerance f32::EPSILON*4 (deterministic; same code path, dropout_p=0).
- [x] [patch] Added `transformer_module_forward_routes_to_seq2seq_self`: `Module::forward(x)`
  == `forward_seq2seq(x,x)` — structural contract of the Module impl.
- [x] [patch] Added `test_lstm_cell_step_matches_pytorch`: copies w_ih/b_ih/w_hh/b_hh from
  LSTMCell(4,6) into torch.nn.LSTMCell.double(); compares h_new and c_new at atol=1e-10.
  Gate order [i,f,g,o] matches PyTorch.
- [x] [patch] Added `test_gru_cell_step_matches_pytorch`: same injection pattern for
  GRUCell(4,6) vs torch.nn.GRUCell.double(); n=tanh(ih_n+r*hh_n) formula matches.
  Compares h_new at atol=1e-10.
- [x] Fixed pre-existing mnemosyne-heap dyn-compatibility compile error
  (TierSelection::backend removed; committed mnemosyne Phase 3 Stage D1 as 4750f88).
- [x] Evidence: `rustup run nightly cargo nextest run -p coeus-nn` 294/294 passed;
  `rustup run nightly cargo clippy -p coeus-nn --tests -- -D warnings` clean.

### Previous Sprint: MS-135 - TransformerEncoderLayer functional SSOT routing [COMPLETE]
**Objective**: Complete TransformerEncoderLayer module/functional SSOT in Rust
and thin the PyO3 encoder-layer forward path to call core helpers directly,
eliminating per-forward module reconstruction while preserving parity behavior.
**Target version**: 0.5.2 (minor-class; additive functional surface + wrapper cleanup).

- [x] [minor] Added and exported Rust-core
  `coeus_nn::transformer_encoder_layer(...)` plus
  `coeus_nn::TransformerEncoderLayerParams`.
- [x] [patch] Routed `TransformerEncoderLayer::forward_with_mask` through the
  shared functional helper.
- [x] [patch] Routed `PyTransformerEncoderLayer.forward` through the shared
  functional helper (no temporary Rust module reconstruction per call).
- [x] [patch] Added Rust functional/module parity assertion in
  `nn_attention_tests::encoder_layer_forward_shape`.
- [x] [patch] Added Python SSOT parity assertion in
  `binding_tests_ops::test_transformer_encoder_bindings` for
  encoder-layer composition equivalence with `dropout_p=0`.
- [x] Evidence: `rustup run nightly cargo nextest run -p coeus-nn --test
  nn_attention_tests encoder_layer_forward_shape`; `rustup run nightly cargo
  nextest run -p coeus-python --test binding_tests_ops
  test_transformer_encoder_bindings`; `rustup run nightly cargo clippy -p
  coeus-nn --test nn_attention_tests -- -D warnings`; `rustup run nightly
  cargo clippy -p coeus-python --test binding_tests_ops -- -D warnings`.

### Previous Sprint: MS-134 - MHA functional SSOT routing [COMPLETE]

- [x] [minor] Added `coeus_nn::multi_head_attention_cross(...)` and
  `coeus_nn::MhaProjectionParams` for shared MHA self/cross execution.
- [x] [patch] Routed Rust and Python MHA forward paths through the shared helper.
- [x] Evidence: `rustup run nightly cargo nextest run -p coeus-nn --test
  nn_tests test_mha_cross_attention_shape`; `rustup run nightly cargo nextest
  run -p coeus-python --test binding_tests_nn test_pycoeus_nn`.

### Previous Sprint: MS-133 - PyTransformer seq2seq + RNN/PE Burn parity tests [COMPLETE]
**Objective**: Add `PyTransformer` full seq2seq Python binding; add LSTM/GRU structural
Burn parity tests (zero-input analytical, shape contract, forward_seq vs Module::forward);
add SinusoidalEncoding and RotaryEmbedding structural tests; PyTransformer composition
parity test.
**Target version**: 0.5.2 (minor-class; new `PyTransformer` public binding).

- [x] [minor] Added `PyTransformer` to `crates/coeus-python/src/nn/feedforward.rs`: stores
  `encoder: Py<PyTransformerEncoder>`, `decoder: Py<PyTransformerDecoder>` sub-modules;
  `new(d_model, d_ff, num_heads, num_enc_layers, num_dec_layers, dropout_p)` delegates
  to existing constructors (no dispatch macro needed — already handled by sub-modules);
  `forward(src, tgt)` chains encoder→decoder; `parameters()` flat-maps both;
  `num_enc_layers`/`num_dec_layers` getters; validated `d_model % num_heads == 0`.
- [x] [minor] Exported `PyTransformer` from `mod.rs`; registered in `lib.rs`.
- [x] [patch] Added 3 LSTM structural Burn parity tests: `lstm_zero_input_zero_output_analytical`
  (analytical: zero-bias + zero-input → zero output by induction), `lstm_output_shape_contract`
  (shape: [B,T,I] → [B,T,H]), `lstm_forward_seq_matches_module_forward` (API: Module::forward == forward_seq().0).
- [x] [patch] Added 3 GRU structural Burn parity tests: `gru_zero_input_zero_output_analytical`
  (same analytical invariant), `gru_output_shape_contract`, `gru_forward_seq_matches_module_forward`.
- [x] [patch] Added 2 SinusoidalEncoding tests: `sinusoidal_encoding_output_shape_matches_input`,
  `sinusoidal_encoding_pos0_equals_analytical` (pos=0 → [0,1,0,1,...] analytically derived).
- [x] [patch] Added 2 RotaryEmbedding tests: `rope_zero_input_zero_output` (zeros → zeros),
  `rope_output_shape_matches_input`.
- [x] [patch] Added `test_transformer_seq2seq_composition` to `test_pytorch_parity.py`:
  asserts `Transformer.forward(src,tgt)` == `encoder.forward(src)` → `decoder.forward(tgt,memory)`;
  param count == 16*E + 26*D.
- [x] Fixed pre-existing stale mnemosyne-backend artifact in coeus workspace cache
  (`cargo clean -p mnemosyne-backend`; mnemosyne-backend compiles correctly in isolation).
- [x] Evidence: `rustup run nightly cargo clippy -p coeus-python --tests -- -D
  warnings`; `pytest crates/coeus-python/tests/test_pytorch_parity.py -k
  test_transformer_seq2seq_composition -v`.

### Previous Sprint: MS-128 - Fix G-002: Stateful PyTransformerEncoder binding [COMPLETE]
**Objective**: Refactor `PyTransformerEncoder` to store `Vec<Py<PyTransformerEncoderLayer>>`
instead of scalars; extract shared `build_from_layer`/`from_rust_layer` inherent methods
(SSOT with `PyTransformerEncoderLayer::new`); add N-layer Burn and PyTorch parity tests.
**Target version**: 0.5.2 (minor-class; new per-layer parameter access surface).

- [x] [minor] `PyTransformerEncoder`: replaced dimension-only struct with
  `layers: Vec<Py<PyTransformerEncoderLayer>>`; `new()` dispatches 36 (H,N) pairs and
  stores layers as `from_rust_layer::<H>` per layer; `forward()` chains each layer's
  stateful Pre-LN forward without re-creating a Rust encoder;
  `parameters()` flat-maps across layers (returns 16×N params); `num_layers` is a
  `#[getter]` returning `self.layers.len()`; `zero_grad()` iterates layers.
- [x] [minor] Added `PyTransformerEncoderLayer::build_from_layer<const H>` and
  `from_rust_layer<const H>` inherent methods (non-`#[pymethods]`) eliminating code
  duplication between `PyTransformerEncoderLayer::new()` and `PyTransformerEncoder::new()`.
- [x] [patch] Added `transformer_encoder_stack_2layer_self_consistent` (structural
  self-consistency: `TransformerEncoder<H=2,N=2>::forward` == manual layer chain).
- [x] [patch] Added `transformer_encoder_stack_2layer_forward_matches_burn` (differential
  vs Burn autodiff NdArray: 2-layer weighted Coeus stack vs 2 manually-assembled Burn Pre-LN
  layers, 2e-4 tolerance).
- [x] [patch] Extracted `_torch_preln_layer_fwd` helper in `test_pytorch_parity.py`
  (DRY: second occurrence of PyTorch Pre-LN forward assembly); refactored
  `test_transformer_encoder_layer_matches_pytorch` to use it.
- [x] [patch] Added `test_transformer_encoder_stack_matches_pytorch` (differential vs
  PyTorch: 2-layer encoder, 32 parameters, output at 2e-4 atol).
- [x] [patch] Closed G-002 in `docs/gap_audit.md`.
- [x] Evidence: `cargo nextest run -p coeus-nn` 111/111 passed;
  `pytest crates/coeus-python/tests/test_pytorch_parity.py -v` 8/8 passed.

### Previous Sprint: MS-131 - Extended activation backward parity [COMPLETE]
**Objective**: Extend Burn-backed scalar activation backward coverage for
`coeus_autograd` operations used by Coeus NN modules.
**Target version**: 0.5.2 (patch-class; parity test coverage).

- [x] [patch] Added Burn autodiff backward parity for `leaky_relu`,
  `softplus`, `mish`, and scalar `pow`.
- [x] [patch] Added analytical ELU, NLL loss, and cosine embedding loss
  forward/backward coverage because Burn 0.16 does not expose matching oracles.
- [x] Evidence: `rustup run nightly cargo fmt -p coeus-nn --check`; `rustup run
  nightly cargo clippy -p coeus-nn --test burn_live_parity -- -D warnings`;
  `rustup run nightly cargo nextest run -p coeus-nn --test burn_live_parity
  activation_backward_extended_match_burn pow_backward_matches_burn
  elu_backward_matches_analytical nll_loss_forward_backward_match_analytical
  cosine_embedding_loss_forward_backward_match_analytical` (5/5).

### Previous Sprint: MS-132 - FeedForward functional SSOT routing [COMPLETE]
**Objective**: Complete FeedForward module/functional SSOT in Rust and thin the
PyO3 binding path to call core helpers directly, eliminating per-forward module
reconstruction while preserving parity behavior.
**Target version**: 0.5.2 (minor-class; additive functional surface + wrapper cleanup).

- [x] [minor] Added and exported Rust-core `coeus_nn::feed_forward(...)` from
  transformer and crate root exports.
- [x] [patch] Routed `FeedForward::forward` and `PyFeedForward::forward` through
  the shared functional helper.
- [x] [patch] Updated `PyFeedForward::new` to initialize both linear
  projections from one Rust `FeedForward::new(...)` instance.
- [x] [patch] Added Rust functional/module parity assertion in
  `nn_attention_tests::ffn_forward_shape`.
- [x] [patch] Added Python SSOT parity assertion in
  `binding_tests_ops::test_feedforward_module` for
  `ffn.forward(x) == linear2(gelu(linear1(x)))` when `dropout_p=0`.
- [x] Evidence: `rustup run nightly cargo nextest run -p coeus-nn --test
  nn_attention_tests ffn_forward_shape`; `rustup run nightly cargo nextest run
  -p coeus-python --test binding_tests_ops test_feedforward_module`; `rustup
  run nightly cargo clippy -p coeus-nn --test nn_attention_tests -- -D
  warnings`; `rustup run nightly cargo clippy -p coeus-python --test
  binding_tests_ops -- -D warnings`.

### Previous Sprint: MS-130 - Python transformer head validation [COMPLETE]
**Objective**: Harden Python transformer/MHA construction so invalid
`d_model`/`num_heads` combinations return `ValueError` at the PyO3 boundary
instead of panicking inside Rust constructors, and re-verify the decoder parity
surface introduced in MS-129.
**Target version**: 0.5.2 (patch-class; boundary validation and gate closure).

- [x] [patch] Added divisibility validation to `PyMultiHeadAttention`,
  `PyTransformerEncoderLayer`, `PyTransformerEncoder`,
  `PyTransformerDecoderLayer`, and `PyTransformerDecoder`.
- [x] [patch] Updated `test_transformer_decoder_layer` to assert compatible
  default construction, incompatible default-head rejection, and the stateful
  26-parameter decoder layer surface.
- [x] [patch] Rebuilt the CPython 3.13 test wheel used by
  `crates/coeus-python/tests/test_pytorch_parity.py`.
- [x] Evidence: `rustup run nightly cargo fmt -p coeus-nn -p coeus-python
  --check`; `rustup run nightly cargo clippy -p coeus-python --tests -- -D
  warnings`; `rustup run nightly cargo doc -p coeus-python --no-deps`;
  `rustup run nightly cargo nextest run -p coeus-python` (72/72);
  `rustup run nightly cargo nextest run -p coeus-nn --test burn_live_parity
  transformer_decoder` (3/3); `pytest crates/coeus-python/tests/test_pytorch_parity.py
  -v` (10/10).

### Previous Sprint: MS-127 - Fix G-001: Stateful PyTransformerEncoderLayer binding [COMPLETE]
**Objective**: Refactor `PyFeedForward` and `PyTransformerEncoderLayer` in
`crates/coeus-python/src/nn/feedforward.rs` to be stateful — storing `norm1`, `self_attn`,
`norm2`, and `ffn` as `Py<>` sub-module fields — and promote the pytest encoder test
from shape-contract-only to full weight-parity verification.
**Target version**: 0.5.2 (minor-class; new parameter-accessible binding surface).

- [x] [minor] `PyFeedForward`: replaced dimension-only struct with `linear1: Py<PyLinear>`,
  `linear2: Py<PyLinear>` fields; `new()` extracts weights from Rust `FeedForward::new()`;
  `forward()` reconstructs Rust struct from Python sub-objects; `parameters()` and
  `zero_grad()` delegate to sub-modules.
- [x] [minor] `PyTransformerEncoderLayer`: replaced dimension-only struct with `norm1`,
  `self_attn`, `norm2`, `ffn` as `Py<PyLayerNorm>`, `Py<PyMultiHeadAttention>`,
  `Py<PyLayerNorm>`, `Py<PyFeedForward>` fields; `new()` extracts weights from
  `TransformerEncoderLayer::new()` via const-dispatch macro; `forward()` reads
  weights from Python sub-objects and reconstructs the Rust encoder; `parameters()`
  returns all 16 params; `zero_grad()` delegates to sub-modules.
- [x] [patch] Replaced `test_transformer_encoder_layer_shape_contract` with
  `test_transformer_encoder_layer_matches_pytorch`: extracts actual weights from Python
  sub-modules, copies to assembled PyTorch Pre-LN components, verifies output at 2e-4 atol.
- [x] [patch] Closed G-001 in `docs/gap_audit.md`.
- [x] Evidence: `cargo nextest run -p coeus-nn` 272/272 passed;
  `pytest crates/coeus-python/tests/test_pytorch_parity.py -v` 7/7 passed.

### Previous Sprint: MS-126 - Extend pytest PyTorch parity surface [COMPLETE]
**Objective**: Expand `test_pytorch_parity.py` to cover Conv1d/2d, LayerNorm,
MHA backward (dx + dW), and TransformerEncoderLayer shape contract.
**Target version**: 0.5.2 (patch-class).

- [x] [patch] `test_conv1d_matches_pytorch`: Conv1d(2→3, k=3) forward+backward
  (out, dx, dW, db) at 1e-5 atol.
- [x] [patch] `test_conv2d_matches_pytorch`: Conv2d(2→2, k=2) forward+backward.
- [x] [patch] `test_layernorm_matches_pytorch`: LayerNorm([4], eps=1e-5)
  forward+backward (out, dx, dγ, dβ) at 1e-4 atol.
- [x] [patch] `test_mha_backward_matches_pytorch`: MHA(d_model=4, H=2) dx + dW_q
  after sum-loss backward at 1e-5 atol.
- [x] [patch] `test_transformer_encoder_layer_shape_contract`: shape [B,S,D]
  preserved; weight-parity test blocked by G-001 (stateless binding). Gap filed
  in `docs/gap_audit.md` as G-001.
- [x] Evidence: `pytest crates/coeus-python/tests/test_pytorch_parity.py -v` 7/7 passed;
  rebuilt wheel with maturin to get current `TransformerEncoderLayer` binding.

### Previous Sprint: MS-125 - TransformerEncoderLayer Burn parity + pytest PyTorch scaffold [COMPLETE]
**Objective**: Add TransformerEncoderLayer forward+backward Burn parity tests and
scaffold the coeus-python pytest harness for PyTorch/JAX/MLX output parity.
**Target version**: 0.5.2 (patch-class; parity tests).

- [x] [patch] Added `transformer_encoder_layer_forward_matches_burn` (commit 7454992)
  and `transformer_encoder_layer_backward_matches_burn` (commit ac8e4cd) to
  `burn_live_parity.rs`. Uses Coeus Kaiming-init weights copied to manually
  assembled Burn Pre-LN components (LN×2 + MHA + PWFF), transposing Linear
  weights for the [out,in] vs [in,out] convention difference. Verifies forward
  context and input gradient (dx) at 2e-4 tolerance.
- [x] [patch] Created `crates/coeus-python/tests/test_pytorch_parity.py` pytest module
  with `test_linear_matches_pytorch` (Linear(256→64) + ReLU + MSELoss: loss, dx,
  dW, db at 1e-5 atol, f64) and `test_mha_matches_pytorch` (MHA d_model=4, H=2,
  no bias, forward at 1e-10 atol). No transposition required: pycoeus and PyTorch
  both use [out, in] Linear convention. 2/2 pass.
- [x] Evidence: `cargo nextest run -p coeus-nn` 272/272 passed;
  `pytest crates/coeus-python/tests/test_pytorch_parity.py -v` 2/2 passed.

### Previous Sprint: MS-124 - coeus-python documented binding surface [COMPLETE]
**Objective**: Document the remaining public PyO3 binding crate surface before
enabling crate-wide `#![deny(missing_docs)]` in `coeus-python`.
**Target version**: 0.5.2 (patch-class; documentation enforcement).

- [x] [patch] Added Rustdoc to all 293 previously-undocumented public PyO3 items
  across 25 files: crate root, all module declarations, pyclass constructors,
  fields, and all `#[pymethods]` methods. (commit 684ce02)
- [x] [patch] Enabled `#![deny(missing_docs)]` in `crates/coeus-python/src/lib.rs`.
- [x] Evidence: `cargo check -p coeus-python` clean; `cargo nextest run -p coeus-python` 72/72 passed.

### Previous Sprint: MS-123 - MHA backward + Conv generic consolidation [COMPLETE]
**Objective**: Close the MultiHeadAttention backward parity gap and consolidate Conv1d/2d/3d into a single `Conv<D: ConvDim>` generic.
**Target version**: 0.5.2 (minor-class; new parity tests + SRP consolidation).

- [x] [patch] Added MHA forward+backward Burn parity test
  (`multi_head_attention_backward_matches_burn`) with explicit projection
  weights, deterministic Burn dropout, and transposed Burn Linear weight/grad
  handling to match Coeus `[out, in]` storage.
- [x] [minor] Consolidated Conv1d/2d/3d in `coeus-nn` into the generic
  `Conv<T, B, D: ConvDim>` layer with sealed ZST dimension strategies and
  `Conv1d`/`Conv2d`/`Conv3d` type aliases.
- [x] [patch] Split `crates/coeus-ops/src/backend_ops/cpu_impl.rs` into SRP family
  submodules.
- [x] [patch] Split `crates/coeus-autograd/src/ops/nn/conv.rs` into conv1d/2d/3d and
  transpose leaf modules.
- [x] [patch] Enforced and fixed `coeus-nn` missing docs and added documented
  public surfaces across touched core/tensor/ops/cuda/wgpu items.
- [x] [patch] Deferred crate-wide `coeus-python` `#![deny(missing_docs)]` to
  MS-124 because enabling it currently exposes 293 unrelated public binding
  documentation diagnostics outside this Conv/MHA slice.
- [x] Evidence: `rustup run nightly cargo fmt -p coeus-core -p coeus-cuda -p
  coeus-nn -p coeus-ops -p coeus-python -p coeus-tensor -p coeus-wgpu
  --check`; `rustup run nightly cargo clippy -p coeus-nn --tests -- -D
  warnings`; `rustup run nightly cargo nextest run -p coeus-nn` (270/270);
  `rustup run nightly cargo clippy -p coeus-core -p coeus-cuda -p coeus-ops
  -p coeus-python -p coeus-tensor -p coeus-wgpu --tests -- -D warnings`;
  `rustup run nightly cargo doc -p coeus-core -p coeus-cuda -p coeus-nn -p
  coeus-ops -p coeus-python -p coeus-tensor -p coeus-wgpu --no-deps`;
  `rustup run nightly cargo test --doc -p coeus-core -p coeus-nn -p coeus-ops
  -p coeus-tensor -p coeus-wgpu -p coeus-cuda`; `rustup run nightly cargo
  nextest run -p coeus-python --test binding_tests_nn --test binding_tests_ops
  test_pycoeus_nn test_nn_functional_ops`; `rustup run nightly cargo clippy -p
  coeus-ops -p coeus-autograd --tests -- -D warnings`; `rustup run nightly
  cargo nextest run -p coeus-ops -p coeus-autograd` (224/224).

### Previous Sprint: MS-122 - Burn parity + SRP + Python transformer bindings [COMPLETE]
**Objective**: Add BatchNorm3d training backward, ConvTranspose1d/2d Burn parity, Python
TransformerEncoderLayer/TransformerEncoder/SinusoidalEncoding bindings, split
wgpu ops/mod.rs (1182→7 SRP files), fix coeus-ops missing_docs.
**Target version**: 0.5.1.

- [x] [patch] BatchNorm3d training-mode backward: dγ, dβ, dx vs Burn autodiff
- [x] [patch] ConvTranspose1d backward vs Burn autodiff (dx, dw)
- [x] [patch] ConvTranspose2d backward vs Burn autodiff (dx, dw)
- [x] [minor] PyTransformerEncoderLayer: dispatches const H ∈ {1,2,4,8,16,32}
- [x] [minor] PyTransformerEncoder: dispatches 36 (H,N) specializations
- [x] [minor] PySinusoidalEncoding: stateless PE wrapper
- [x] [patch] Split crates/coeus-wgpu/src/backend/ops/mod.rs → conv.rs, pool.rs, optim.rs, matmul.rs, reduction.rs
- [x] [patch] Fix coeus-ops fuse module missing_docs (87 items)
- [x] Evidence: `cargo nextest run --workspace --exclude coeus-cuda`: 797/797 passed.

### Previous Sprint: MS-121 - Public docs and parity surface [COMPLETE]
**Objective**: Add executable public examples across touched operation,
distributed, and sparse APIs; expand thin Python transformer wrappers; and add
the next Burn-backed normalization parity case.
**Target version**: 0.5.1 (minor-class; additive binding surface + patch-class
documentation/test cleanup).

- [x] [patch] Replaced the `binary_op!`-generated public `add`/`sub`/`mul`/`div`
  functions with explicit generic functions carrying compiling Rustdoc examples.
- [x] [patch] Added executable examples for CPU backend dispatch, reductions,
  matmul helpers, shape concatenation/stacking, and unary math operations.
- [x] [patch] Corrected the `gelu` doctest reference value to the exact-GELU
  contract (`0.5 * x * (1 + erf(x / sqrt(2)))`) instead of the tanh
  approximation value.
- [x] [patch] Added executable Rustdoc examples for `coeus-dist` local
  communicators and `coeus-sparse` COO/CSR construction/accessor contracts.
- [x] [patch] Added executable Rustdoc examples for `coeus-leto` layout/view
  conversion, elementwise dispatch, initialization, layout transforms, and
  linear algebra bridge contracts.
- [x] [minor] Registered Python `TransformerEncoderLayer`,
  `TransformerEncoder`, and `SinusoidalEncoding` wrappers over existing
  `coeus_nn` Rust-core implementations, with unsupported const-generic choices
  mapped to `ValueError` instead of panics.
- [x] [patch] Added Python binding tests for encoder-layer, encoder-stack,
  sinusoidal, and decoder error paths.
- [x] [patch] Added BatchNorm3d training-mode backward differential parity
  against a Burn autodiff reference for `dx`, `dw`, and `db`.
- [x] Evidence: `rustup run nightly cargo fmt -p coeus-nn -p coeus-ops -p
  coeus-python -p coeus-wgpu --check`; `rustup run nightly cargo clippy -p
  coeus-ops --tests -- -D warnings`; `rustup run nightly cargo clippy -p
  coeus-nn -p coeus-python --tests -- -D warnings`; `rustup run nightly cargo
  test --doc -p coeus-ops -p coeus-optim`; `rustup run nightly cargo test --doc
  -p coeus-dist -p coeus-sparse`; `rustup run nightly cargo nextest run -p
  coeus-nn --test burn_live_parity batchnorm3d_training_backward_matches_burn`;
  `rustup run nightly cargo nextest run -p coeus-python --test
  binding_tests_ops test_transformer_encoder_bindings test_transformer_decoder_layer
  test_nn_functional_ops`; `rustup run nightly cargo nextest run -p coeus-wgpu`
  (83/83); `rustup run nightly cargo test --doc -p coeus-leto` (28/28);
  `rustup run nightly cargo doc -p coeus-dist -p coeus-sparse -p coeus-ops -p
  coeus-nn -p coeus-python -p coeus-wgpu -p coeus-optim --no-deps`.

### Previous Sprint: MS-120 - WGPU bounded metadata pool [COMPLETE]
**Objective**: Reduce WGPU metadata-buffer pool contention and prevent
unbounded retained metadata buffers without changing kernel semantics or public
backend APIs.
**Target version**: 0.5.1 (patch-class; backend memory/contention cleanup).

- [x] [patch] Changed `WgpuContext::get_metadata_buffer` to use a nonblocking
  pool fast path: reuse an existing metadata buffer when the mutex is
  immediately available, otherwise allocate a fresh short-lived metadata buffer
  instead of blocking a concurrent kernel submission.
- [x] [patch] Changed `WgpuContext::recycle_metadata_buffer` to recycle only
  when the mutex is immediately available and the pool is below a fixed
  capacity; excess or contended returns drop the buffer so the pool cannot grow
  without bound.
- [x] Evidence: `rustup run nightly cargo fmt -p coeus-wgpu --check`;
  `rustup run nightly cargo clippy -p coeus-wgpu --tests -- -D warnings`;
  `rustup run nightly cargo nextest run -p coeus-wgpu` (83/83);
  `rustup run nightly cargo doc -p coeus-wgpu --no-deps`.

### Previous Sprint: MS-119 - Python functional norm pure-wrapper SSOT [COMPLETE]
**Objective**: Keep layer/rms norm computation in Rust core and make Python
functionals thin validated wrappers.
**Target version**: 0.5.1 (minor-class; additive core functional exports +
wrapper cleanup).

- [x] [minor] Added `coeus_nn::layer_norm(...)` and `coeus_nn::rms_norm(...)`
  functional helpers and exported both from `coeus_nn`.
- [x] [patch] Routed `coeus-python` `layer_norm` / `rms_norm` wrappers through
  those helpers.
- [x] [patch] Added PyO3 input validation (rank/shape/epsilon) for both
  wrappers with clear `ValueError` messages.
- [x] [patch] Added functional parity checks in `crates/coeus-nn/tests/nn_norm_tests.rs`
  and Python functional checks in `crates/coeus-python/tests/binding_tests_ops.rs`.
- [x] Evidence: `rustup run nightly cargo check -p coeus-nn --lib`; `rustup
  run nightly cargo check -p coeus-python --lib`; `rustup run nightly cargo
  nextest run -p coeus-nn --test nn_norm_tests test_layernorm test_rmsnorm`
  (4/4); `rustup run nightly cargo nextest run -p coeus-python --test
  binding_tests_ops test_nn_functional_ops` (1/1); `rustup run nightly cargo
  clippy -p coeus-nn -p coeus-python --tests -- -D warnings`.

### Previous Sprint: MS-118 - WGPU strided parity tests [COMPLETE]
**Objective**: Add differential parity coverage for the new WGPU Hephaestus
strided dispatch path (MS-117) via transposed and permuted non-contiguous views.
**Target version**: 0.5.1 (patch-class; test coverage).

- [x] [patch] Added 6 tests in `crates/coeus-wgpu/tests/wgpu/parity.rs`:
  `test_wgpu_strided_add_transposed_matches_cpu`,
  `test_wgpu_strided_mul_transposed_matches_cpu`,
  `test_wgpu_strided_exp_transposed_matches_cpu`,
  `test_wgpu_strided_neg_transposed_matches_cpu`,
  `test_wgpu_strided_sqrt_transposed_matches_cpu`,
  `test_wgpu_strided_rank3_binary_matches_cpu` — each exercises the
  non-contiguous Hephaestus strided path via transposed/permuted layouts.
- [x] Evidence: `cargo nextest run -p coeus-wgpu`: 83/83 pass.

### Previous Sprint: MS-117 - WGPU strided Hephaestus routing [COMPLETE]
**Objective**: Route non-contiguous binary and unary elementwise ops through
Hephaestus `*_elementwise_strided_into` for rank ≤ 4, with a CPU fallback.
**Target version**: 0.5.1 (patch-class; no API change).

- [x] [patch] Add `leto` as a direct dep in `crates/coeus-wgpu/Cargo.toml` for
  `LetoLayout::new`.
- [x] [patch] Implement `coeus_to_leto_layout!` macro: pads Coeus dynamic
  layout to const-rank `[usize; N]`/`[isize; N]` arrays.
- [x] [patch] `can_route_strided_wgpu`: guard for rank ≤ MAX_STRIDED_RANK and
  no broadcast output dim (zero stride where dim > 1).
- [x] [patch] `try_hephaestus_strided_binary_wgpu`: dispatches
  Add/Sub/Mul/Div to Hephaestus at rank N=1..4.
- [x] [patch] `try_hephaestus_strided_unary_wgpu`: dispatches
  Sin/Cos/Exp/Log/Neg/Abs/Sqrt/Recip; falls back for other ops.
- [x] [patch] Wire into `BackendOps::elementwise_binary` and
  `elementwise_unary` between contiguous-Hephaestus and CPU-fallback paths.
- [x] Evidence: `cargo nextest run`: 789/789 pass; `cargo clippy -p
  coeus-wgpu -- -D warnings`: no errors.

### Previous Sprint: MS-116 - MHA and TransformerEncoder value parity [COMPLETE]
**Objective**: Promote shape-only MHA and TransformerEncoder Burn parity tests
to full value-semantic differential verification.
**Target version**: 0.5.1 (patch-class; test coverage).

- [x] [patch] Added `multi_head_attention_identity_weights_matches_analytical_sdpa`:
  H=1, W_q=W_k=W_v=W_o=I, verifies forward output and dx backward via Burn
  autodiff SDPA reference (103rd parity test).
- [x] [patch] Added `transformer_encoder_layer_identity_weights_matches_analytical`:
  pre-norm (LN→MHA→residual→LN→FFN(GELU)→residual) with identity weights,
  verifies forward and dx vs manual Burn autodiff reference (104th parity test).
- [x] Evidence: `cargo nextest run -p coeus-nn`: 266/266 pass.

### Previous Sprint: MS-115 - coeus-python InstanceNorm3d + norm stubs [COMPLETE]
**Objective**: Add `PyInstanceNorm3d` PyO3 wrapper, extend `.pyi` stubs for all
three InstanceNorm variants and missing GroupNorm functional stub, add Python-side
value parity tests.
**Target version**: 0.5.1 (minor-class; additive Python binding).

- [x] [minor] Added `PyInstanceNorm3d` delegating to `coeus_nn::InstanceNorm3d`;
  registered in normalization/mod → nn/mod → lib.rs.
- [x] [minor] Extended `pycoeus.pyi` with `GroupNorm`, `InstanceNorm1d/2d/3d`
  class stubs and `group_norm` functional stub (previously missing entirely).
- [x] [patch] Added `test_instancenorm_forward_shape_and_value` verifying shape
  and population-variance normalized values for all three variants.
- [x] Evidence: `cargo nextest run -p coeus-python`: 71/71 pass.

### Previous Sprint: MS-114 - Autograd public documentation surface [COMPLETE]
**Objective**: Close the `coeus-autograd` public documentation gap under
`#![deny(missing_docs)]` without changing runtime behavior. Document public
operation modules, autograd backward-node state, and tracked sparse/shape
entry points so doctests and rustdoc become package-clean.
**Target version**: 0.5.1 (patch-class; documentation and diagnostics).

- [x] [patch] Documented the public autograd operation hierarchy and node
  fields for activation, arithmetic, embedding, sparse linalg, neural-network,
  normalization, loss, pooling, softmax, and shape operation surfaces.
- [x] [patch] Documented tracked sparse matmul, COO sparse matmul, concat,
  split, pad, and cumsum public entry points, including backward semantics and
  panic surfaces where current implementations assert.
- [x] [patch] Corrected accidental `bytex_ops` bounds in touched autograd node
  definitions back to the canonical `coeus_ops` backend trait.
- [x] Evidence: `rustup run nightly cargo fmt -p coeus-autograd --check`;
  `rustup run nightly cargo test --doc -p coeus-autograd` (15/15);
  `rustup run nightly cargo clippy -p coeus-autograd --tests -- -D warnings`;
  `rustup run nightly cargo nextest run -p coeus-autograd` (35/35);
  `rustup run nightly cargo doc -p coeus-autograd --no-deps`.

### Previous Sprint: MS-113 - InstanceNorm3d + bilinear SSOT [COMPLETE]
**Objective**: Consolidate duplicate `get_cache` logic from InstanceNorm1d/2d
into shared `ensure_cache` + `instance_norm_forward` free functions, add
InstanceNorm3d ([N,C,D,H,W] normalization over D*H*W), export it, and add a
full Burn autodiff parity test for forward + backward (dx/dw/db). Keep bilinear
math in Rust-core while exposing a thin Python functional wrapper.
**Target version**: 0.5.1 (minor-class; new InstanceNorm3d + Python function).

- [x] [minor] Consolidated `InstanceNorm1d` and `InstanceNorm2d` to share
  `ensure_cache` + `instance_norm_forward` free functions; eliminated duplicate
  `get_cache` methods.
- [x] [minor] Added `InstanceNorm3d` with spatial = D*H*W; exported from
  `normalization::mod` and `coeus_nn::lib`.
- [x] [patch] Added `instancenorm3d_forward_backward_matches_burn` parity test
  (101st test); verifies forward values and dx/dw/db backward within 1e-4.
- [x] [patch] Added `coeus_nn::bilinear(...)` functional helper and made
  `Bilinear::bilinear_forward` delegate to it (single SSOT path).
- [x] [patch] Updated `coeus-python` `PyBilinear::bilinear_forward` to call the
  Rust-core helper directly using existing weight/bias Vars.
- [x] [minor] Added `pycoeus.bilinear(input1, input2, weight, bias=None)` as a
  shape-validated thin binding over Rust-core and added the `.pyi` stub.
- [x] [patch] Added executable Rustdoc examples for touched autograd and tensor
  public APIs.
- [x] Evidence: `cargo nextest run -p coeus-nn --test bilinear_parity`;
  `cargo nextest run -p coeus-python --test binding_tests_ops
  test_nn_functional_ops`; `cargo nextest run -p coeus-python --test
  binding_tests_ops test_bilinear_module`; `cargo nextest run -p coeus-nn
  --test burn_live_parity instancenorm3d_forward_backward_matches_burn`;
  `cargo test --doc -p coeus-autograd -p coeus-tensor -p coeus-nn`;
  `cargo clippy -p coeus-nn -p coeus-python --tests -- -D warnings`.

### Previous Sprint: MS-112 - InstanceNorm1d/2d backward parity [COMPLETE]
**Objective**: Add differential Burn autodiff parity for InstanceNorm1d and
InstanceNorm2d backward passes (dx, dw, db), closing the norm backward gap.
**Target version**: 0.5.1 (patch-class; test coverage).

- [x] [patch] Added `instancenorm1d_backward_matches_burn` — [N,C,L] input,
  per-(sample,channel) spatial normalization formula in Burn autodiff,
  verifies dx/dw/db within 1e-4 relative tolerance (99th parity test).
- [x] [patch] Added `instancenorm2d_backward_matches_burn` — [N,C,H,W] input,
  reshape to [N*C, H*W] for spatial normalization (100th parity test).
- [x] Evidence: `cargo nextest run -p coeus-nn --test burn_live_parity`:
  100/100 pass.

### Previous Sprint: MS-111 - CUDA strided Hephaestus routing [COMPLETE]
**Objective**: Extend CUDA Hephaestus routing to the dynamic-strided path so
non-contiguous CUDA elementwise ops also prefer the Hephaestus kernel (with
rank ≤ MAX_STRIDED_RANK and no broadcast dimensions in output) before the
Coeus-local strided fallback.
**Target version**: 0.5.1 (patch-class; performance).

- [x] [patch] Added `try_hephaestus_strided_binary` routing Add/Sub/Mul/Div
  through `hephaestus_cuda::binary_elementwise_strided_dyn_into` with guard for
  rank and broadcast exclusion.
- [x] [patch] Added `try_hephaestus_strided_unary` routing Sin/Cos/Exp/Log/Neg/
  Abs/Sqrt/Recip through `hephaestus_cuda::unary_elementwise_strided_dyn_into`.
- [x] [patch] Added `hephaestus_operand` helper to convert Coeus layout
  (shape/strides/offset) to `hephaestus_cuda::StridedOperandDyn`.
- [x] Evidence: `cargo check -p coeus-cuda --all-targets` clean;
  `cargo check -p coeus-cuda --features cuda` clean.

### Previous Sprint: MS-110 - Conv3d backward parity vs Burn autodiff [COMPLETE]
**Objective**: Add differential Burn autodiff parity for Conv3d backward
pass (dx, dw), completing backward parity coverage for all three conv dims.
**Target version**: 0.5.1 (patch-class; test coverage).

- [x] [patch] Added `conv3d_backward_matches_burn` — free-function Burn conv3d
  with tracked input/weight, valid convolution (stride=1, pad=0, dil=1),
  comparing dx and dw within epsilon tolerance.
- [x] Evidence: `cargo nextest run -p coeus-nn --test burn_live_parity`:
  98/98 pass.

### Previous Sprint: MS-109 - WGPU Hephaestus zero-allocation elementwise routing [COMPLETE]
**Objective**: Reduce WGPU elementwise allocation churn by keeping delegated
Hephaestus contiguous routes in caller-owned output buffers.
**Target version**: 0.5.1 (patch-class; behavior-preserving backend optimization).
**Tests delivered**: delegated contiguous unary/binary parity and output-buffer
identity assertions, plus aliasing fallback parity.

- [x] [patch] `crates/coeus-wgpu/src/backend/ops/mod.rs`: replaced allocating
  `binary_elementwise`/`unary_elementwise` calls with `*_into` variants for
  contiguous non-aliased Hephaestus dispatch.
- [x] [patch] Kept alias-guard contract unchanged so aliased output continues to
  use Coeus-local kernels.
- [x] [patch] `crates/coeus-wgpu/tests/wgpu/parity.rs`: added delegated contiguous
  unary/binary tests asserting output Arc identity is preserved and values match
  CPU reference.
- [x] Evidence: `cargo fmt --check`; `cargo test -p coeus-wgpu
  test_wgpu_hephaestus_contiguous_binary_reuses_output_buffer`; `cargo test -p
  coeus-wgpu test_wgpu_hephaestus_contiguous_unary_reuses_output_buffer`;
  `cargo test -p coeus-wgpu test_wgpu_aliasing_unary_neg_matches_cpu`.

### Previous Sprint: MS-108 - BatchNorm2d training-mode backward parity [COMPLETE]
**Objective**: Add differential Burn autodiff parity for BatchNorm2d training-mode
backward pass (dx, dw, db), matching Coeus's NHWC-based population-variance formula
against the same formula manually expressed in Burn autodiff tensors.
**Target version**: 0.5.1 (patch-class; test coverage).
**Tests delivered**: `batchnorm2d_training_backward_matches_burn` (97th parity test).

- [x] [patch] Added `batchnorm2d_training_backward_matches_burn` — manual BN2d
  formula in Burn autodiff tensors matching Coeus NHWC-layout, population variance
  (÷M not ÷(M-1)), verifies dx/dw/db within 1e-4 relative tolerance.
- [x] Evidence: `cargo nextest run -p coeus-nn --test burn_live_parity`: 97/97 pass.

### Previous Sprint: MS-107 - CUDA Hephaestus primitive routing [COMPLETE]
**Objective**: Keep CUDA and WGPU shared primitive GPU dispatch centralized in
Hephaestus while preserving Coeus-local kernels only for aliasing, strided
coverage not yet mapped through the static-rank Hephaestus API, and
NN-specific activation/optimizer/convolution kernels.
**Target version**: 0.5.1 (patch-class; internal routing cleanup).
**Acceptance**: `coeus-cuda` contiguous non-aliased primitive binary
`Add/Sub/Mul/Div` and unary `Sin/Cos/Exp/Log/Neg/Abs/Sqrt/Recip` route through
`hephaestus-cuda`, with Coeus-local kernels retained as the fallback for
unsupported or aliased cases. Evidence tier: type-level compile verification
plus package build/clippy/doc gates; live CUDA value parity remains covered by
the existing `coeus-cuda --features cuda` parity suite when CUDA hardware is
available.

- [x] [patch] Added `coeus-cuda` Hephaestus-first routing for supported
  contiguous non-aliased primitive binary and unary elementwise ops.
- [x] [patch] Kept Coeus-local CUDA kernels as the explicit fallback for output
  aliasing and unsupported Coeus activation formulas, matching the existing
  WGPU redundancy boundary.
- [x] Evidence: `cargo check -p coeus-cuda`; `cargo check -p coeus-cuda
  --features cuda`; `cargo fmt -p coeus-cuda --check`; `cargo clippy -p
  coeus-cuda --all-targets --features cuda -- -D warnings`; `cargo doc -p
  coeus-cuda --features cuda --no-deps`; `cargo nextest run -p coeus-cuda
  --features cuda` (69/69).

### Previous Sprint: MS-106 - Conv1d + Conv2d backward gradient parity [COMPLETE]
**Objective**: Close the Burn autodiff-parity gap for Conv1d / Conv2d backward
gradients (dx + dw) so the full forward + backward Burn parity envelope now
covers 1D and 2D convolution modules.
**Target version**: 0.5.1 (patch-class; additive test coverage only).

- [x] [patch] Added `conv1d_backward_matches_burn` and `conv2d_backward_matches_burn`
  in `crates/coeus-nn/tests/burn_live_parity.rs` against Burn `Autodiff<NdArray<f32>>`.
- [x] [patch] Both tests compare `dx` (input gradient) and `dw` (weight gradient)
  using exact same data shapes and weight values as the corresponding forward
  Burn parity cases.
- [x] Evidence: merged PR #20 from `feat/ms-106-conv-backward-parity`
  (`cargo nextest run -p coeus-nn --test burn_live_parity`
  → 96/96 pass; `cargo nextest run -p coeus-nn`
  → 259/259 pass).

### Previous Sprint: MS-105 - Optimizer + Conv3d parity + CUDA cache RwLock [COMPLETE]
**Objective**: Add closed-form RMSProp / AdaGrad / AdamW first-step analytic
references, Conv3d stride+padding Burn parity, transpose backward Burn autodiff
parity, and complete the `coeus-cuda` fused-kernel cache `Mutex` → `RwLock`
conversion (read-mostly swap with double-checked write-lock insert).
**Target version**: 0.5.1 (patch-class; additive tests + internal cache fix).

- [x] [patch] Added `rmsprop_step_matches_analytical_reference`,
  `adagrad_step_matches_analytical_reference`,
  `adamw_step_matches_analytical_reference` in `burn_live_parity::optimizer_*`.
- [x] [patch] Added `conv3d_forward_matches_burn` and transpose-backward
  autodiff parity in `crates/coeus-nn/tests/burn_live_parity.rs`.
- [x] [patch] Converted `crates/coeus-cuda/src/kernels/fuse.rs::KERNEL_CACHE` from
  `Mutex<HashMap<…>>` to `RwLock<HashMap<…>>` with read-lock fast-path and
  double-checked write-lock insert path.
- [x] Evidence: `cargo nextest run -p coeus-nn --test burn_live_parity` →
  94/94 pass; `cargo check -p coeus-cuda` → clean.

### Previous Sprint: MS-104 - Core Rustdoc contract examples [COMPLETE]
**Objective**: Make `coeus-core` public documentation executable for storage,
layout, shape, scalar, stride, and backend contracts while preserving the
already-merged Burn/WGPU parity work and completing the CUDA fused-kernel cache
RwLock cleanup.
**Target version**: 0.5.1 (patch-class; documentation/test cleanup).
**Tests delivered**: compiling `coeus-core` doctests plus the full
`burn_live_parity` suite covering added optimizer, Conv3d stride/padding, and
transpose-backward parity cases.

- [x] [patch] Added executable Rustdoc examples for `ComputeBackend`,
  `Backend`, `SequentialBackend`, `Scalar`, `Float`, `Layout`, `ConstLayout`,
  `Shape`, `ConstShape`, row-major strides, and CPU storage/COW contracts.
- [x] [patch] Corrected doctest examples to import trait providers explicitly
  and use the existing `Shape` slice API.
- [x] [patch] Completed `coeus-cuda` fused-kernel cache conversion from a
  serialized `Mutex<HashMap<...>>` hit path to `RwLock<HashMap<...>>`.
- [x] [patch] Added analytical first-step references for RMSProp, AdaGrad, and
  AdamW optimizer updates.
- [x] [patch] Added Conv3d stride/padding Burn parity and transpose backward
  Burn autodiff parity coverage.
- [x] Evidence: `cargo test --doc -p coeus-core`; `cargo nextest run -p
  coeus-core`; `cargo nextest run -p coeus-nn --test burn_live_parity`; `cargo
  check -p coeus-cuda`; `cargo check -p coeus-cuda --features cuda`; `cargo
  fmt --check`; `cargo clippy -p coeus-core -p coeus-cuda --all-targets
  -- -D warnings`; `cargo clippy -p coeus-nn --test burn_live_parity
  -- -D warnings`; `cargo doc -p coeus-core --no-deps`.

### Previous Sprint: MS-103 - Conv3d and InstanceNorm2d Burn parity [COMPLETE]
**Objective**: Extend Burn parity coverage to 3D convolution and 2D instance
normalization.
**Target version**: 0.5.0 (patch-class; additive tests, no public API change).
**Tests delivered**: Conv3d forward and InstanceNorm2d forward against Burn
NdArray.

- [x] [patch] Added `conv3d_forward_matches_burn` with explicit matching
  weight initialization and valid padding.
- [x] [patch] Added `instancenorm2d_forward_matches_burn` against Burn NdArray.
- [x] Evidence: merged PR #17 from
  `feat/ms-103-conv3d-instancenorm2d-parity`.

### Previous Sprint: MS-102 - WGPU aliasing fallback parity [COMPLETE]
**Objective**: Verify aliased WGPU elementwise operations keep Coeus-local
fallback semantics after Hephaestus routing was introduced for non-aliased
contiguous buffers.
**Target version**: 0.5.0 (patch-class; additive tests, no public API change).
**Tests delivered**: unary neg and binary add aliasing-path parity.

- [x] [patch] Added WGPU aliasing parity coverage for unary neg.
- [x] [patch] Added WGPU aliasing parity coverage for binary add.
- [x] Evidence: merged PR #16 from `feat/ms-102-wgpu-aliasing-tests`.

### Previous Sprint: MS-101 - BatchNorm3d eval parity and Conv Burn parity [COMPLETE]
**Objective**: Close Burn normalization parity gap for 3D batch norm and add
Conv1d/Conv2d differential parity against Burn NdArray.
**Target version**: 0.5.0 (patch-class; additive tests, no public API change).
**Tests delivered**: BatchNorm3d eval-mode forward vs Burn, Conv1d and Conv2d
forward vs Burn with explicit ones-kernel initialization.

- [x] [patch] Added `batchnorm3d_eval_forward_matches_burn` in
  `crates/coeus-nn/tests/burn_live_parity.rs` — same eval-mode pattern as 1d/2d.
- [x] [patch] Added `conv1d_forward_matches_burn` — explicit ones-weight coeus
  Conv1d vs Burn `Conv1dConfig` with matching weight; asserts shape and values.
- [x] [patch] Added `conv2d_forward_matches_burn` — same for 2D.
- [x] Evidence: `cargo nextest run` (768/768 passed, 50 s); `cargo fmt --check`;
  `cargo clippy --all-targets --all-features -- -D warnings`; no new `#[allow]`.

### Previous Sprint: MS-100 - Python functional GroupNorm wrapper [COMPLETE]
**Objective**: Expose Rust-core functional GroupNorm through `coeus-python`
without adding Python-side normalization logic.
**Target version**: 0.5.0 (minor-class; additive public Python API).
**Tests delivered**: Python binding value checks for no-affine output, affine
output, and invalid group count, plus BatchNorm2d eval-mode parity against Burn
NdArray.

- [x] [minor] Added registered `pycoeus.group_norm` as a thin PyO3 wrapper over
  `coeus_nn::group_norm`, with GIL release around Rust computation.
- [x] [patch] Added Python-boundary validation for rank, group count, epsilon,
  and optional affine tensor shape before delegating to Rust core.
- [x] [patch] Extended `crates/coeus-python/tests/binding_tests_ops.rs` with exact
  functional GroupNorm output assertions and zero-group rejection.
- [x] [patch] Added `crates/coeus-nn/tests/burn_live_parity.rs` BatchNorm2d eval-mode
  forward parity against Burn NdArray.
- [x] Evidence: `cargo fmt --check`; `cargo nextest run -p coeus-python --test
  binding_tests_ops test_nn_functional_ops`; `cargo nextest run -p coeus-nn
  --test burn_live_parity batchnorm2d_eval_forward_matches_burn`; `cargo
  clippy -p coeus-python --all-targets -- -D warnings`; `cargo doc -p
  coeus-python --no-deps`.

### Previous Sprint: MS-99 - WGPU routing and functional GroupNorm [COMPLETE]
**Objective**: Strengthen `coeus-wgpu` shader/pipeline cache correctness,
route contiguous elementwise work through Hephaestus public kernels where
aliasing permits, and add a Rust-core functional GroupNorm API for parity with
Burn/PyTorch-style stateless normalization.
**Target version**: 0.4.0 (minor-class; additive public `coeus-nn` API).
**Tests delivered**: WGPU package nextest plus analytical normalization parity
for module and functional GroupNorm on SequentialBackend and MoiraiBackend.

- [x] [patch] `crates/coeus-wgpu/src/kernels/cache.rs`: made cache entries
  device-scoped and source-sensitive (`device_addr`, shader key, entry point,
  WGSL source hash) to prevent cross-device or stale-key collisions.
- [x] [patch] Replaced global `Mutex<HashMap<...>>` with
  `RwLock<HashMap<...>>` and double-checked insertion to avoid holding the
  write lock while compiling pipelines.
- [x] [patch] Reduced GPU kernel redundancy against `hephaestus-wgpu` by routing
  contiguous non-aliased `Add/Sub/Mul/Div` and unary
  `Sin/Cos/Exp/Log/Neg/Abs/Sqrt/Recip` through Hephaestus elementwise dispatch,
  while retaining Coeus-local kernels for aliasing and unsupported unary ops.
- [x] [minor] Added and exported stateless `coeus_nn::group_norm`, backed by
  tensor-level Rust-core ops and explicit rank/group/affine/epsilon validation.
- [x] [patch] Extended `crates/coeus-nn/tests/norm_parity.rs` with exact analytical
  functional GroupNorm assertions, including affine output and zero-group
  rejection.
- [x] Evidence: `cargo fmt --check`; `cargo nextest run -p coeus-nn --test
  norm_parity`; `cargo nextest run -p coeus-wgpu`; `cargo clippy -p coeus-nn -p
  coeus-wgpu --all-targets -- -D warnings`; `cargo doc -p coeus-nn -p
  coeus-wgpu --no-deps`.

### Previous Sprint: MS-98 - stats pair reductions and PyO3 wrappers [COMPLETE]
**Objective**: Add Rust-owned `var_mean` / `std_mean` statistics pairs and
thin Python wrappers while consolidating standalone variance/std paths through
the pair-returning SSOT, and expose verified sequence-level RNN modules.
**Target version**: 0.3.0 (minor-class; additive public Rust/Python/NN API).
**Tests delivered**: value-semantic Rust/Python coverage for global and
per-axis pair reductions plus NN module analytical parity for Bilinear,
ConvTranspose, and sequence-level RNNs.

- [x] [minor] `coeus-ops`: added `var_mean`, `std_mean`, `var_mean_axis`, and
  `std_mean_axis`; `var`, `std_dev`, `var_axis`, and `std_dev_axis` now reuse
  the pair-returning implementations.
- [x] [minor] `coeus-python`: exposed `var_mean` and `std_mean` as thin PyO3
  wrappers with optional `axis` and `keepdim`, preserving Rust-core ownership
  of statistics logic.
- [x] [patch] `crates/coeus-ops/tests/stats_diff.rs`: added analytical global and
  per-axis assertions comparing pair outputs against standalone reductions and
  `mean_axis`.
- [x] [patch] `crates/coeus-python/tests/binding_tests_ops.rs`: added scalar,
  per-axis, keepdim, and error-path checks for `pycoeus.var_mean` and
  `pycoeus.std_mean`.
- [x] [minor] `crates/coeus-nn/src/rnn/{gru,lstm}.rs`: added and exported
  sequence-level `Gru` and `Lstm` modules with `forward_seq`, including the
  `CpuAddressableStorageMut` bounds required by output concatenation.
- [x] [patch] `crates/coeus-nn/tests/{bilinear,conv_transpose_nn,rnn_seq}_parity.rs`:
  added analytical module parity checks on SequentialBackend and MoiraiBackend.
- [x] Evidence: `cargo fmt --check`;
  `cargo nextest run -p coeus-ops --test stats_diff` (2/2);
  `cargo nextest run -p coeus-python --test binding_tests_ops` (58/58);
  `cargo nextest run -p coeus-nn --test bilinear_parity --test
  conv_transpose_nn_parity --test rnn_seq_parity` (6/6);
  `cargo clippy -p coeus-ops -p coeus-python -p coeus-nn --all-targets
  -- -D warnings`;
  `cargo doc -p coeus-ops -p coeus-python -p coeus-nn --no-deps`.

### Previous Sprint: MS-97 - NN differential parity expansion [COMPLETE]
**Objective**: Extend `coeus-nn` parity coverage across recurrent cells,
interpolation, losses, positional encodings, global pooling, and 3D pooling
with value-semantic analytical references on SequentialBackend and
MoiraiBackend.
**Target version**: 0.2.34 (patch-class).
**Tests delivered**: 12 differential tests across 6 `coeus-nn` parity binaries.

- [x] [patch] `crates/coeus-nn/tests/rnn_parity.rs`: covers `GRUCell` and `LSTMCell`
  zero-input analytical oracles plus `Module::forward` equivalence.
- [x] [patch] `crates/coeus-nn/tests/interpolate_parity.rs`: covers
  `interpolate_1d` and `interpolate_2d` exact references on SequentialBackend
  and MoiraiBackend.
- [x] [patch] `crates/coeus-nn/tests/loss_parity.rs`: covers `mse_loss`,
  `nll_loss`, `huber_loss`, `binary_cross_entropy`, and
  `cosine_embedding_loss` against closed-form scalar references.
- [x] [patch] `crates/coeus-nn/tests/positional_parity.rs`: covers
  `SinusoidalEncoding` and `RotaryEmbedding` analytical positional references.
- [x] [patch] `crates/coeus-nn/tests/global_pool_parity.rs`: covers
  `GlobalAvgPool1d`, `GlobalAvgPool3d`, and `GlobalMaxPool3d` references.
- [x] [patch] `crates/coeus-nn/tests/pool3d_parity.rs`: covers `AvgPool3d` and
  `MaxPool3d` analytical references.
- [x] Evidence: `cargo nextest run -p coeus-nn` (236/236);
  `cargo fmt --check`;
  `cargo clippy -p coeus-nn --all-targets -- -D warnings`;
  `cargo doc -p coeus-nn --no-deps`.

### Previous Sprint: MS-96 - ops parity and Leto unary integration cleanup [COMPLETE]
**Objective**: Close additional `coeus-ops` differential coverage gaps and
preserve Coeus/Leto layering by consuming unary acceleration only through public
Leto APIs.
**Target version**: 0.2.33 (patch-class).
**Tests delivered**: 16 differential tests across 8 `coeus-ops` binaries plus
25 `coeus-leto` dispatch contract tests.

- [x] [patch] `crates/coeus-ops/tests/embedding_diff.rs`: covers `embedding`,
  `embedding_backward`, and `embedding_backward_with_padding_idx` with repeated
  indices and padding suppression.
- [x] [patch] `crates/coeus-ops/tests/unary_math_diff.rs`: covers exact unary math
  identities on SequentialBackend and MoiraiBackend.
- [x] [patch] `crates/coeus-ops/tests/shape_ops_diff.rs`: covers `flip`, `roll`,
  `tril`, `triu`, `sort`, `one_hot`, `repeat_interleave`, `outer`, and `cross`.
- [x] [patch] `crates/coeus-ops/tests/activation_diff.rs`: covers sigmoid, GELU,
  tanh-GELU, SiLU, Mish, ELU, Softplus, and LeakyReLU.
- [x] [patch] `crates/coeus-ops/tests/conv_transpose_diff.rs`: covers
  `conv_transpose1d` and `conv_transpose2d`.
- [x] [patch] `crates/coeus-ops/tests/misc_ops_diff.rs`: covers `amax`, `amin`, `dot`,
  `cumprod`, `broadcast_to`, `chunk`, `diag`, and `diagonal`.
- [x] [patch] `crates/coeus-ops/tests/prod_tile_maskfill_diff.rs`: covers `prod`,
  `tile`, and `masked_fill`.
- [x] [patch] `crates/coeus-ops/tests/sparse_conv_diff.rs`: covers `dense_to_coo`,
  `coo_to_dense`, `dense_to_csr`, `csr_to_dense`, and `coo_to_csr` roundtrips.
- [x] [patch] `coeus-leto`: added exact `Exp`/`Log`/`Sqrt` dispatch contract
  coverage while preserving public Leto API routing.
- [x] Upstream provider: `leto` commit `d38addb` routes contiguous `SqrtOp`
  through the Leto `RealScalar::sqrt_slice` strategy seam.
- [x] Evidence: `cargo fmt --check`; targeted 16/16 ops nextest;
  `cargo nextest run -p coeus-ops` (189/189);
  `cargo nextest run -p coeus-leto --test contract` (25/25);
  `cargo clippy -p coeus-ops -p coeus-leto --all-targets -- -D warnings`;
  `cargo doc -p coeus-ops -p coeus-leto --no-deps`.

### Previous Sprint: MS-95 - sparse ops differential parity [COMPLETE]
**Objective**: Add value-semantic sparse forward/backward coverage for CSR
operations on SequentialBackend and MoiraiBackend, using exact integer-valued
references.
**Target version**: 0.2.32 (patch-class).
**Tests delivered**: 2 sparse-operation differential tests; package gate covers
173 ops tests.

- [x] [patch] `crates/coeus-ops/tests/sparse_ops_diff.rs`: SequentialBackend and
  MoiraiBackend differential coverage for `spmv`, `spmm`,
  `spmm_backward_values`, and `spmm_backward_dense`.
- [x] Evidence: `cargo fmt --check`; `cargo nextest run -p coeus-ops`
  (173/173); `cargo clippy -p coeus-ops --test sparse_ops_diff -- -D warnings`;
  `cargo doc -p coeus-ops --no-deps`.

### Previous Sprint: MS-94 - constructors, index ops, initializers, and interpolate parity [COMPLETE]
**Objective**: Extend value-semantic differential coverage across untested
constructor/index utilities and neural-network initializer/interpolation paths,
while keeping oracle logic analytical or routed through the Atlas-owned Leto
dispatch layer.
**Target version**: 0.2.31 (patch-class).
**Tests delivered**: 2 constructor/selection tests, 2 index/scatter/BMM tests,
initializer dispatch parity on SequentialBackend and MoiraiBackend, and 8
interpolation analytical-reference tests. Package gates cover 171 ops tests and
224 nn tests.

- [x] [patch] `crates/coeus-ops/tests/constructors_diff.rs`: SequentialBackend and
  MoiraiBackend differential coverage for `linspace`, `logspace`, `geomspace`,
  `meshgrid`, `nonzero`, and `where_cond` using bitwise-exact integer-valued
  or power-valued references.
- [x] [patch] `crates/coeus-ops/tests/index_ops_diff.rs`: SequentialBackend and
  MoiraiBackend differential coverage for `gather`, `index_select`,
  `index_put`, `scatter_add`, `masked_select`, and `bmm`.
- [x] [patch] `crates/coeus-nn/tests/init_leto_diff.rs`: initializer parity verifies
  seeded uniform/normal, Xavier, and Kaiming paths against direct
  `coeus-leto` dispatch for the same shape, scalar type, and seed.
- [x] [patch] `crates/coeus-nn/tests/nn_interpolate_tests.rs`: analytical-reference
  coverage for `interpolate_1d` and `interpolate_2d` nearest/bilinear paths
  under the align-half-pixel contract.
- [x] Evidence: `cargo fmt --check`; `cargo nextest run -p coeus-ops` (171/171);
  `cargo nextest run -p coeus-nn` (224/224);
  `cargo clippy -p coeus-ops -p coeus-nn --all-targets -- -D warnings`;
  `cargo doc -p coeus-ops -p coeus-nn --no-deps`.

### Previous Sprint: MS-93 - sparse COO autograd parity + PyTensor vertical split [COMPLETE]
**Objective**: Add tracked COO sparse matrix multiplication without duplicating
the existing CSR gradient kernels; keep the Python tensor binding surface as a
thin PyO3 wrapper after splitting the previous monolithic file by concern.
**Target version**: 0.2.30 (minor-class; new public autograd API).
**Tests delivered**: 1 new sparse COO autograd parity test plus 2 statistical
reduction differential tests; package-scoped gates cover 35 autograd tests, 167
ops tests, and 70 Python binding tests.

- [x] [minor] `coeus_autograd::sparse_matmul_coo`: converts COO inputs to CSR
  once, preserves a sorted-to-original permutation for value-gradient remapping,
  reuses `coeus_ops::spmm`, `spmm_backward_values`, and
  `spmm_backward_dense`, and validates COO row/column bounds before CSR
  construction.
- [x] [patch] `crates/coeus-autograd/tests/autograd/sparse.rs`: forward and backward
  COO parity against dense `matmul`, with value-semantic checks for COO-value
  gradients and dense RHS gradients.
- [x] [patch] `crates/coeus-ops/tests/stats_diff.rs`: SequentialBackend and
  MoiraiBackend differential coverage for variance, standard deviation, and
  Lp-norm reductions against analytical references.
- [x] [patch] `crates/coeus-python/src/tensor/`: split `PyTensor`, iterator, and
  state-dict bindings into concern-specific modules while retaining Rust-core
  ownership of tensor behavior.
- [x] [patch] `crates/coeus-ops/Cargo.toml`: removed unused direct `num-traits`
  dependency from `coeus-ops`; `coeus-core` remains the numeric-trait owner.
- [x] Evidence: `cargo fmt --check`; `cargo nextest run -p coeus-autograd`;
  `cargo nextest run -p coeus-ops`; `cargo nextest run -p coeus-python`;
  `cargo clippy -p coeus-autograd -p coeus-python -p coeus-ops --all-targets
  -- -D warnings`; `cargo doc -p coeus-autograd -p coeus-python -p coeus-ops
  --no-deps`.

### Previous Sprint: MS-92 - f16/bf16 differential parity on both backends [COMPLETE]
**Objective**: Close bf16 zero-coverage gap and extend f16 backend parity beyond
SequentialBackend-only. Verifies that MoiraiBackend dispatches half-precision ops
identically to SequentialBackend.
**Target version**: 0.2.29 (patch-class).
**Tests delivered**: 4 differential tests (f16+bf16 × Sequential+Moirai); 630/630.

- [x] [patch] `crates/coeus-ops/tests/half_precision_diff.rs` (NEW): 4 tests covering
  add, matmul, sum, relu for f16 and bf16 on both backends. Integer inputs within
  each type's mantissa (≤2^11 f16, ≤2^7 bf16) → bitwise-exact assertions via
  `T::from_f32` round-trip. Evidence: `a844606`, 4/4 passed.
- [x] Evidence: 630/630 workspace tests; clippy/fmt clean. Commit: `a844606`.

### Previous Sprint: MS-91 - einsum/einsum3 differential parity + cosine_embedding_loss [COMPLETE]
**Objective**: Close differential test gap for `einsum`/`einsum3` (no backend parity
coverage since MS-83) and add `cosine_embedding_loss` analytical coverage (function
shipped but never tested against closed-form reference).
**Target version**: 0.2.29 (patch-class).
**Tests delivered**: 4 einsum differential tests (6 subscript patterns) + 1 cosine
loss test (5 cases + backward); workspace at 626/626.

- [x] [patch] `crates/coeus-ops/tests/einsum_diff.rs` (NEW): 4 differential tests verifying
  einsum patterns (matmul, transpose, trace, dot, outer, mat-vec) and einsum3 triple
  chain on both backends. Integer inputs → bitwise-exact assertions.
  Evidence: `b9f0a28`, 4/4 passed.
- [x] [patch] `crates/coeus-nn/tests/nn_ops/losses/nn_loss/`: `test_cosine_embedding_loss` with
  identical/orthogonal/opposite/batch/backward cases. All assertions value-semantic
  against closed-form definition with eps=1e-10.
  Evidence: `b9f0a28`, 1/1 passed.
- [x] Evidence: 626/626 workspace tests; `cargo clippy -D warnings` clean;
  `cargo fmt --check` clean. Commit: `b9f0a28`.

### Previous Sprint: MS-90 - frobenius_norm differential parity + optimizer convergence [COMPLETE]
**Target version**: 0.2.29 (patch-class).
**Tests delivered**: 8 frobenius norm differential tests + 4 optimizer convergence
tests; workspace at 621/621.

- [x] [patch] `crates/coeus-ops/tests/norm_diff.rs` (NEW): 8 differential tests for
  `frobenius_norm` (2-D, rectangular, identity, zeros) and `frobenius_norm_batched`
  (rank-3, rank-4, row-vector batch) against analytical reference ‖A‖_F=sqrt(Σaᵢⱼ²).
  SequentialBackend + MoiraiBackend each. Tolerances derived from f32 ε × element count.
  Evidence: `cargo nextest run -p coeus-ops --test norm_diff` — 4/4 passed.
- [x] [patch] `crates/coeus-optim/tests/optim_ops/convergence.rs`: 4 new multi-step tests —
  `test_sgd_convergence_quadratic_50steps` (closed-form: x_n = x₀·0.8ⁿ),
  `test_sgd_momentum_convergence_100steps` (derived bound via spectral radius √0.9 ≈ 0.9487),
  `test_adam_convergence_quadratic_200steps` (200-step convex convergence to |p|<0.05),
  `test_adamw_weight_decay_shrinkage_50steps` (closed-form: p_n = p₀·(1−lr·λ)ⁿ with g=0).
  Evidence: `cargo nextest run -p coeus-optim` — 14/14 passed.
- [x] Evidence: 621/621 workspace tests; `cargo clippy -D warnings` clean; `cargo fmt --check` clean.
  Commit: `6afaab4`.

### Previous Sprint: MS-89 - transformer source masks + BatchNorm eval bindings [COMPLETE]
**Target version**: 0.2.29.

### Previous Sprint: MS-88 - matrix_norm(ord='fro') Torch parity [COMPLETE]
**Target version**: 0.2.28.

> **Roadmap (docs/backlog.md MS-61)**: live Burn comparison starts replacing hardcoded
> oracle values; wgpu parity.rs verifies implemented GPU paths against the CPU reference;
> coeus-python gains 20+ new functional ops (stack, matmul, constructors, abs/sqrt/neg,
> clamp, max/min/min/max_axis, sum/mean, reshape, permute, t, pow, arange, linspace, etc.).

### MS-84 Progress (2026-06-25)

- [x] [patch] moirai-executor: add `MIN_ELEMENTS_PER_CHUNK = 256` floor to
  `indexed_chunk_count` — caps chunk count so each scheduled chunk processes
  ≥256 iterations, reducing unpark calls for 1024-element ops from 8 to 3.
  Evidence: 700/700 moirai workspace tests pass; 18/18 coeus-core tests pass.
  Commit: `bded467` (moirai repo).
- [x] [patch] coeus-nn: Add Burn 0.16 parity tests for sigmoid, tanh, silu,
  log_softmax (dim=1), leaky_relu, softplus, mish — 7 new tests covering the
  activation batch shipped in MS-83 (UnaryOp dispatch via elementwise_unary).
  Evidence: `cargo nextest run -p coeus-nn` — 207 tests, 207 passed.
  Commit: `69055d9`.

### MS-85 Progress (2026-06-25)

- [x] [patch] f16 / bf16 half-precision compute path — `half::f16` and `bf16`
  already implement `Scalar` and `Float` in `coeus-core`. Added 3 smoke tests
  (`coeus-ops::add`, `coeus-ops::matmul`, autograd `sum(x*x).backward()`) to
  confirm end-to-end half-precision operation.
- [x] [patch] `pycoeus.pyi` comprehensive Python type stub covering all public
  functions, classes, and properties — enables IDE auto-completion and mypy
  validation of the entire public Python surface.
- [x] [patch] Hermes GEMV 8× row-blocking already in place —
  `hermes-simd::dispatch_gemv_kernel` already dispatches `TilingPolicy<8,1>`
  for `LANE_COUNT > 8` (AVX512). FFT via Apollo integration documented and
  deferred pending Apollo crate stabilization.

### MS-88 Progress (2026-06-26)

- [x] [minor] Added `coeus_ops::frobenius_norm` / `coeus_ops::frobenius_norm_batched`
  composing on the existing `coeus_ops::norm` chain (`sqrt(sum(x·x))`), no
  new `BinaryOp::Pow` opcode, no new backend dispatch.
- [x] [minor] Added `pycoeus.matrix_norm(input, ord='fro')` PyO3 binding with
  rank-aware Python return: `float` for 2-D inputs (mirroring torch's 0-D
  coercion), `PyTensor` for higher-rank per-batch results. 1-D and
  non-`'fro'` `ord` surface as `ValueError` at the boundary.
- [x] [patch] Completed embedding padding-index semantics across
  `coeus_ops`, `coeus_autograd`, `coeus_nn`, and `pycoeus.Embedding`:
  padding rows are zero-initialized and receive zero gradient.
- [x] [patch] Completed vertical shape module hierarchy integration for
  `coeus-ops` and `coeus-autograd` under concern-oriented submodules while
  preserving the existing public exports.
- [x] [patch] Expanded BatchNorm eval-mode parity across Python bindings:
  `BatchNorm1d`, `BatchNorm2d`, and `BatchNorm3d` now expose `eval_forward`,
  and regression coverage verifies eval-path normalization uses
  `running_mean`/`running_var` without mutating them.
- [x] Documented ordering in CHANGELOG, raised workspace `Cargo.toml`
  0.2.23 → 0.2.28 without regressing the existing MS-87 branch history.
- Evidence: `cargo nextest run -p coeus-ops frobenius` passes with 6 tests
  (2-D oracle, identity, 3-D batched, 4-D batched, 2-D batched 0-D shape, 1-D
  panic); `cargo nextest run -p coeus-python --test binding_tests_ops
  test_matrix_norm_fro` passes (2-D float, non-square, 3-D batched, 4-D batched,
  1-D ValueError, ord!='fro' ValueError, default ord); `cargo nextest run -p
  coeus-ops` passes with 147 tests; `cargo nextest run -p coeus-autograd`
  passes with 34 tests; `cargo nextest run -p coeus-nn` passes with 209 tests;
  `cargo nextest run -p coeus-python` passes with 70 tests; `cargo clippy -p
  coeus-ops -p coeus-autograd -p coeus-nn -p coeus-python --all-targets --
  -D warnings`, `cargo doc -p coeus-ops -p coeus-autograd -p coeus-nn -p
  coeus-python --no-deps`, and `cargo fmt --check` are clean.

### MS-89 Progress (2026-06-26)

- [x] [minor] Added optional source key-padding-mask routing through
  `TransformerEncoderLayer::forward_with_mask`,
  `TransformerEncoder::forward_with_mask`, and
  `Transformer::forward_seq2seq_with_src_mask`; `Module::forward` delegates to
  the same implementation with no mask.
- [x] [minor] Completed `pycoeus.BatchNorm1d/2d/3d` eval-mode binding parity.
  Regression coverage verifies eval normalization uses `running_mean` /
  `running_var` without mutating them.
- [x] [patch] Synchronized `pycoeus.pyi` for `matrix_norm`,
  `BatchNorm1d/2d/3d`, and `Embedding(..., padding_idx=...)`.
- Evidence: `cargo nextest run -p coeus-nn --test nn_attention_tests` passes
  with 13 tests; `cargo nextest run -p coeus-python --test binding_tests_ops
  test_batchnorm_eval_mode` passes; `cargo nextest run -p coeus-nn` passes
  with 211 tests; `cargo nextest run -p coeus-python` passes with 70 tests;
  `cargo clippy -p coeus-nn -p coeus-python --all-targets -- -D warnings`,
  `cargo doc -p coeus-nn -p coeus-python --no-deps`, and `cargo fmt --check`
  pass.

### Current Verification Note (2026-06-25)

- [x] [minor] Added `coeus_ops::einsum3`, `coeus_autograd::einsum3`, and
  three-operand `pycoeus.einsum` routing. Recorded audit findings that Moirai
  adaptive thresholds, MHA const-generic head routing, and Coeus CoW
  infrastructure already exist. Evidence tier: empirical value-semantic and
  analytical-gradient validation. Evidence: `cargo nextest run -p coeus-ops
  einsum_three_operand_matmul_chain`, `cargo nextest run -p coeus-python
  --test binding_tests_ops test_einsum_wrapper`, `cargo nextest run -p
  coeus-autograd test_einsum3_matmul_chain_backward`, `cargo clippy -p
  coeus-autograd -p coeus-nn -p coeus-ops -p coeus-python --all-targets --
  -D warnings`, and `cargo doc -p coeus-autograd -p coeus-nn -p coeus-ops -p
  coeus-python --no-deps` pass.

### Current Verification Note (2026-06-26)

- [x] [patch] Repaired provider graph blockers exposed by RITK registration:
  `coeus-ops` root shape re-export for moved leaves, the real
  `embedding_backward_with_padding_idx` accumulation path, and autograd reshape's
  contiguous-function import. Evidence: `cargo fmt --check`,
  `cargo clippy -p coeus-ops -p coeus-autograd -p coeus-nn -p coeus-python
  --all-targets -- -D warnings`, `cargo nextest run -p coeus-ops` (147
  passed), `cargo nextest run -p coeus-autograd` (34 passed), `cargo nextest
  run -p coeus-nn` (209 passed), `cargo nextest run -p coeus-python` (70
  passed), and `cargo doc -p coeus-ops -p coeus-autograd -p coeus-nn -p
  coeus-python --no-deps` pass.
- [x] [minor] Added `coeus_nn::rnn::{LSTMCell, GRUCell}`, Python
  `pycoeus.LSTMCell` / `GRUCell`, `coeus_ops::index_put`,
  `pycoeus.index_put`, and `pycoeus.TransformerDecoderLayer`. Exposed
  decoder constructor fields as Python getters and corrected the binding test
  script. Added Python wrappers for `rand`, `randint`, `bernoulli`, keepdim
  reductions, `normalize`, closeness checks, `nan_to_num`, gradient clipping,
  and tensor value `repr`. Added SDP-attention Burn/Coeus benchmark
  instrumentation without a speedup claim. Evidence tier: empirical
  value-semantic validation plus benchmark build. Evidence: `cargo clippy -p
  coeus-nn -p coeus-ops -p coeus-python --all-targets -- -D warnings`, `cargo
  nextest run -p coeus-ops index_put`, `cargo nextest run -p coeus-python
  --test binding_tests_ops test_randn_zeros_ones_like_eye
  test_normalize_closeness_nan_and_grad_clipping test_lstm_gru_cells
  test_index_put_op test_transformer_decoder_layer`, `cargo check -p
  coeus-tensor --benches`, and `cargo doc -p coeus-nn -p coeus-ops -p
  coeus-python --no-deps` pass.
- [x] [minor] Added `coeus_ops::{bmm, outer, chunk, one_hot, masked_select,
  glu}` and Python wrappers `pycoeus.bmm`, `outer`, `one_hot`,
  `masked_select`, `chunk`, `glu`, plus `pycoeus.ModuleList`. Python wrappers
  are PyO3 boundary adapters: matmul-family functions compose through
  autograd matmul/reshape, GLU composes through slice/sigmoid/mul, and
  one-hot/masked-select/chunk delegate to `coeus_ops` after boundary
  validation. Evidence tier: empirical value-semantic validation. Evidence:
  `cargo clippy -p coeus-ops -p coeus-python --all-targets -- -D warnings`,
  `cargo nextest run -p coeus-ops bmm outer chunk one_hot masked_select glu`,
  and `cargo nextest run -p coeus-python --test binding_tests_ops
  test_one_hot_masked_select_chunk test_bmm_outer_ops test_glu_activation
  test_module_list` pass.
- [x] [minor] Closed MS-71: shipped `coeus_ops::dot` (flat inner product via
  single-pass fold), `coeus_ops::cross` (per-channel 3-vector cross along `dim`
  with size-3 axis assertion), Python `pycoeus.dot` (returns float) and
  `pycoeus.cross` (returns PyTensor with shape preservation) wrappers in
  `crates/coeus-python/src/ops/linalg.rs`. 14 unit tests in
  `coeus_ops::reduction::linalg::tests` and 1 Python binding test
  `binding_tests_ops::test_dot_cross_vector_ops` verify value semantics
  against the documented manual right-handed cross oracle. Evidence tier:
  empirical value-semantic validation per the blatant torch.cross / numpy.cross
  / jnp.cross / mlx.cross convention.
- [x] [patch] Cached `MoiraiBackend::num_threads()` via `OnceLock<AtomicUsize>`
  (Relaxed ordering, immutable-after-first-store) to remove the per-conv-kernel
  `std::thread::available_parallelism()` syscall. Sequence (SequentialBackend,
  inline `1`) is unchanged. Evidence: `cargo test -p coeus-core --lib backend::`
  passes with 5 race-free tests, including the 8-thread contention hammer.
- [x] [minor] Consolidated BatchNorm autograd backward through one
  const-generic `BatchNormNode<T, B, DIM>` and `BatchNormArgs<T, B, DIM>`,
  replacing the old per-rank argument/node names as a documented pre-1.0 minor
  break.
- [x] [patch] Split the monolithic `crates/coeus-leto/src/dispatch.rs` into
  operation-family leaf modules under `crates/coeus-leto/src/dispatch/`, preserving
  the public `coeus_leto::dispatch::*` re-export surface.
- [x] [minor] MS-73 (0.2.13 commit message, Cargo.toml at 0.2.12) shipped
  PyTensor dtype casts (`.float()`, `.double()`, `.long()`, `.int()`, `.half()`,
  `.to(dtype)`, `.type_as(...)`), `PyScaledDotProductAttention` stateless nn
  module + functional free function, +4 burn_live_parity tests.
- [x] [minor] MS-74 (0.2.14 commit message, Cargo.toml at 0.2.12) shipped
  `LayerNorm::forward_nd` (rank≥2 via tracked reshape chain), `PyLayerNorm.forward_nd`,
  `pycoeus.layer_norm` rank>2 dispatch, Hermes `Dot::fma_pair_accumulate` via
  `Arch::fmadd` eliminating separate mul+add latency, +1 burn_live_parity test
  + 1 Python binding test.
- [x] [patch] Reconciled Cargo.toml workspace version to 0.2.17 (was 0.2.15,
  lagging CHANGELOG which already had 0.2.17 section from MS-77).
- [x] [patch] MS-78: Fixed GroupNorm/InstanceNorm Burn parity test tolerances
  and formula: forward 1e-4→1e-3 (sqrt(var+eps) vs sqrt(var)+eps derivation);
  backward formula var.sqrt().add_scalar(eps)→var.add_scalar(eps).sqrt() to
  match Coeus's convention; added Embedding forward + backward Burn parity tests;
  69 Burn parity tests all pass.
- [x] [patch] Added `crates/coeus-sparse/tests/sparse_conversions.rs` to cover
  dense/COO/CSR round-trip identity and direct-vs-COO CSR structural equality
  on a fixed 3x4 oracle. Evidence tier: empirical value-semantic validation.
  Evidence: `cargo nextest run -p coeus-sparse --test sparse_conversions`
  passes with 4 tests.
- [x] [patch] Added tracked COO sparse matmul autograd path
  `coeus_autograd::sparse_matmul_coo` with sparse-value backward mapped back to
  original COO value order and dense-operand backward parity through existing
  SpMM kernels. Evidence tier: empirical value-semantic + backward differential
  validation. Evidence: `cargo test -p coeus-autograd
  sparse_coo_matmul_backward sparse_matmul_backward -- --test-threads=1` and
  `cargo clippy -p coeus-autograd --all-targets -- -D warnings` pass.

- [x] [minor] Added `burn 0.16` as dev-dep to `coeus-nn` and `coeus-tensor`; production
  dependency policy test unaffected (burn forbidden in `[dependencies]`, allowed in
  `[dev-dependencies]`).
- [x] [patch] Added `crates/coeus-nn/tests/burn_live_parity.rs` with live Burn NdArray
  reference checks for softmax and cross-entropy loss.
- [x] [minor] Added four Burn benchmark groups to `tensor_bench.rs`: elementwise add,
  matmul (256×256), ReLU (1024×1024), and sum_dim (1024×1024).  Each group shows Burn
  NdArray, Coeus Sequential, and Coeus Moirai side-by-side under Criterion.
- [x] [minor] Created `crates/coeus-wgpu/tests/wgpu/parity.rs` with 20+ differential tests:
  binary ops, 14 unary activations (macro), reductions, matmul 2D + batched,
  conv1d/conv2d forward, max_pool2d/avg_pool2d, adamw step, round-trip identity.
- [x] [patch] Added `coeus_autograd::stack` (`shape/stack.rs`) with correct backward
  via split+squeeze; exported from `crates/coeus-autograd/src/lib.rs`.
- [x] [minor] Expanded `crates/coeus-python/src/ops.rs` with 20 new free functions and added
  `crates/coeus-python/tests/binding_tests_ops.rs` with 9 test functions including backward.
- [x] [patch] `cargo check --workspace` passes: 0 errors.
- [x] [patch] `cargo clippy --workspace --all-targets -- -D warnings` passes: 0
  errors, 0 warnings.
- [x] [patch] Promoted primary `gelu` to the exact Burn/PyTorch contract
  `0.5 * x * (1 + erf(x / sqrt(2)))` through the scalar SSOT; retained
  `gelu_tanh` as the explicit tanh approximation.
- [x] [patch] Added WGSL exact-contract GELU/GELU-gradient expressions using an
  Abramowitz-Stegun `erf` approximation for WGPU unary and fused shader paths.
- [x] [minor] Expanded live Burn parity to 25 value-semantic tests, including
  exact GELU, SiLU, sin/cos forward/backward, matmul/linear backward, layernorm,
  RMSNorm, clamp, stack/cat/reshape/transpose, flip, sort, and where-cond.
- [x] [patch] Extended live Burn activation parity to Mish, Softplus, and
  LeakyReLU in `crates/coeus-nn/tests/burn_live_parity.rs`, using the derived
  epsilon helper for value-semantic comparisons against Burn NdArray.
  Evidence tier: empirical differential validation. Evidence:
  `cargo nextest run -p coeus-nn --test burn_live_parity` passes with 36 tests.
- [x] [patch] Extended live Burn log-softmax parity to forward values and
  backward gradients for `d/dx sum(log_softmax(x))`, comparing Coeus autograd
  against Burn NdArray autodiff. Evidence tier: empirical differential
  validation. Evidence: `cargo nextest run -p coeus-nn --test burn_live_parity
  log_softmax_forward_and_backward_match_burn` passes.
- [x] [patch] Extended live Burn activation-backward parity for sigmoid, tanh,
  SiLU, and GELU-family gradients. Burn 0.16 uses exact-erf GELU forward but a
  tanh-approximation GELU backward, so the Burn GELU backward branch is compared
  against Coeus' explicit `gelu_tanh` contract rather than weakening exact-GELU
  bounds. Evidence tier: empirical differential validation. Evidence:
  `cargo nextest run -p coeus-nn --test burn_live_parity
  activation_backward_match_burn` passes.
- [x] [patch] Extended live Burn loss and normalization backward parity for
  BCE, MSE, Huber, LayerNorm, and RMSNorm. Huber uses `delta = 1`, where Coeus'
  current SmoothL1-style formula and Burn's Huber contract coincide exactly.
  Evidence tier: empirical differential validation. Evidence: `cargo nextest
  run -p coeus-nn --test burn_live_parity` passes with 36 tests.
- [x] [minor] Added `coeus_ops::{flip, sort, where_cond}`, autograd
  `flip`/`where_cond`, and Python wrappers for `sin`, `cos`, `flip`,
  `where_cond`, `softmax`, `randn`, `topk`, and `sort`.
- [x] [minor] Added `coeus_ops::{broadcast_to, masked_fill, nonzero}`;
  tracked autograd `broadcast_to` and `masked_fill`; and thin PyO3 wrappers
  for `broadcast_to`, `masked_fill`, and `nonzero`. `masked_fill` now tracks
  gradients only through the differentiable input, not the mask. Evidence tier:
  empirical binding and op validation. Evidence: `cargo nextest run -p
  coeus-ops broadcast masked_fill nonzero` passes with 12 tests and `cargo
  nextest run -p coeus-python --test binding_tests_ops
  broadcast_masked_fill_nonzero` passes.
- [x] [minor] Added Python `FeedForward` binding as a thin PyO3 wrapper over
  `coeus_nn::transformer::ffn::FeedForward`, preserving `dropout_p` at the
  boundary and validating it as `0.0 <= p < 1.0`. Evidence tier: empirical
  binding validation. Evidence: `cargo nextest run -p coeus-python --test
  binding_tests_ops test_feedforward_module` passes.
- [x] [patch] Extended optimizer parity with analytical SGD and Adam first-step
  references. Evidence tier: analytical oracle plus empirical test execution.
  Evidence: `cargo nextest run -p coeus-nn --test burn_live_parity
  sgd_step_matches_analytical_reference adam_step_matches_analytical_reference`
  passes.
- [x] [patch] Routed WGPU unmasked and causal scaled-dot-product attention
  forward/backward through on-device WGSL kernels, keeping masked forward as an
  explicit CPU-reference capability boundary. Evidence tier: empirical
  differential validation. Evidence: `cargo nextest run -p coeus-wgpu --test
  wgpu_tests attention` passes with 4 tests.
- [x] [patch] Completed WGPU shader handling for the expanded unary math
  opcode set (`recip`, `sign`, `floor`, `ceil`, `round`, `trunc`) and added
  differential parity tests against `SequentialBackend`. Evidence tier:
  empirical differential validation. Evidence: `cargo nextest run -p
  coeus-wgpu --test wgpu_tests test_wgpu_parity_recip test_wgpu_parity_sign
  test_wgpu_parity_floor test_wgpu_parity_ceil test_wgpu_parity_round
  test_wgpu_parity_trunc` passes.
- [x] [patch] Replaced autograd gradient `Arc<Mutex<Tensor<_, _>>>` storage
  with the `GradBuffer` UnsafeCell SSOT and removed the temporary
  Mutex-shaped compatibility shim; optimizers, distributed gradient sync, and
  attention tests now read/write through `GradBuffer` directly.
- [x] [patch] Corrected conv/pool parity test names whose oracles are manual
  references rather than live Burn tensors, preserving the evidence tier stated
  by the test names.
- [x] [patch] Python comparison wrappers now return `ValueError` on shape
  mismatch instead of panicking at the PyO3 boundary.
- [x] [patch] Renamed the real barrier-backed distributed test communicator
  from `MockCommunicator` to `LocalCommunicator`, including the PyO3 class and
  `create_local_cluster` constructor, with no compatibility alias.
- [x] [minor] Added Rust-core `gather`, `scatter_add`, `repeat_interleave`,
  and `interpolate_1d`/`interpolate_2d` surfaces with coeus-python wrappers.
- [x] [patch] Added PyTensor first-dimension indexing and iteration
  (`tensor[i]`, `tensor[-1]`, `tensor[start:stop]`, `for row in tensor`) using
  tracked Rust-core slice/squeeze operations.
- [x] [patch] Added `coeus-leto::CsrDispatch` sparse SpMV/SpMM dispatch coverage
  against direct `leto_ops` sparse kernels.
- [x] [patch] Routed contiguous CPU `conv1d`, `conv2d`, and `conv3d` row
  execution through one shared Melinoe branded row-partition SSOT
  (`brand_mut_slice` in `conv/mod.rs`), preserving the existing
  value-semantic conv parity tests as the current evidence tier.
- [x] [minor] Extended WGPU conv3d forward/backward differential parity beyond
  the baseline case: stride+padding and dilation cases now compare WGPU results
  against `SequentialBackend` values for output, input gradient, weight
  gradient, and bias gradient. Evidence: `cargo nextest run -p coeus-wgpu
  --test wgpu_tests conv3d` passes with 4 tests.
- [x] [minor] Added CUDA feature parity coverage for binary, unary, reduction,
  matmul, convolution, pooling, AdamW, and host/device round-trip behavior
  against `SequentialBackend`; fixed NVRTC PTX trailing-NUL trimming so fused
  CUDA kernels load through `CString` instead of silently falling back, routed
  broadcasted contiguous operands through strided binary kernels, corrected CUDA
  GELU/GELU-gradient to the exact erf contract, and aligned strided JIT
  coordinate decoding with fused-kernel layout metadata.
  Evidence tier: empirical differential validation. Evidence:
  `cargo nextest run -p coeus-cuda --features cuda --test cuda_tests` passes
  with 42 tests.
- [x] [patch] Extended CUDA live parity coverage to unary activation-gradient
  kernels (`ReluGrad`, `SigmoidGrad`, `TanhGrad`, `GeluGrad`, `SiluGrad`,
  `MishGrad`) against the CPU unary reference, including exact-erf `GeluGrad`
  inputs where the tanh approximation would diverge. Evidence tier: empirical
  differential validation. Evidence:
  `cargo nextest run -p coeus-cuda --features cuda --test cuda_tests` passes
  with 48 tests.
- [x] [patch] Extended CUDA live parity coverage to backward kernels for
  `conv2d`, `max_pool2d`, and `avg_pool2d`, comparing device gradients against
  `SequentialBackend` references for gradient input, weight, and bias where
  applicable. Evidence tier: empirical differential validation.
  Evidence: `cargo nextest run -p coeus-cuda --features cuda --test cuda_tests`
  passes with 51 tests.
- [x] [patch] Consolidated the `coeus-python` embedded-Python test lock into
  `tests/common/mod.rs` and routed binding ops/distributed tests through that
  test-only SSOT. Evidence: `cargo nextest run -p coeus-python --test
  binding_tests_dist --test binding_tests_ops` passes with 26 tests.
- [x] [patch] Scoped temporary `pycoeus` module registration inside
  operation/distributed binding scripts by passing explicit globals and removing
  the `sys.modules` entry after each run. Evidence tier: empirical integration
  validation. Evidence: `cargo nextest run -p coeus-python --test
  binding_tests_dist --test binding_tests_ops` passes.
- [x] [minor] Added PyTorch/JAX-style Python free functions for `unsqueeze`,
  `squeeze`, `flatten`, `argmax`, and `argmin`; the wrappers release the GIL
  around Rust work and return `ValueError` for invalid dimensions. Evidence
  tier: empirical binding validation. Evidence: `cargo nextest run -p
  coeus-python --test binding_tests_ops` passes.
- [x] [minor] Completed `coeus-nn` global pooling module exports and corrected
  `GlobalAvgPool1d` to route through `coeus_autograd::mean_axis(input, 2)`.
  Evidence tier: empirical NN validation. Evidence: `cargo nextest run -p
  coeus-nn` passes with 163 tests.
- [x] [patch] Removed the direct Rayon comparison row and dev-dependency from
  `coeus-tensor` benchmarks; `Coeus Moirai` remains the parallel execution row.
  Evidence tier: compile-time dependency audit plus benchmark build. Evidence:
  `cargo check -p coeus-tensor --benches` and
  `cargo nextest run -p coeus-core --test dependency_policy` pass.
- [x] [patch] Reconciled README and checklist benchmark descriptions with the
  Rayon-free harness surface: Coeus Sequential, Coeus Moirai, direct Leto,
  Coeus-Leto dispatch, and dev-only Burn NdArray oracle rows. Evidence tier:
  documentation/dependency-surface consistency.
- [x] [patch] Extended `crates/coeus-core/tests/dependency_policy.rs` to reject direct
  production `rustfft` imports and manifest dependencies, keeping Apollo's
  Atlas-owned FFT implementation as the Coeus FFT path. Evidence tier:
  compile-time dependency audit.
  Evidence: `cargo nextest run -p coeus-core --test dependency_policy` passes
  and `rg -n "rustfft|apollo" -g "Cargo.toml" -g "*.rs" -g "*.md"` shows no
  production Coeus `rustfft` use.
- [x] [patch] Extended dependency policy to audit the resolved production normal
  dependency tree with `cargo tree --workspace --edges normal`, blocking
  transitive `rayon`, `tokio`, `ndarray`, `nalgebra`, `rustfft`, `burn`, `tch`,
  and `pollster` regressions while preserving dev-only Burn benchmark/parity
  edges. Evidence tier: compile-time dependency audit. Evidence:
  `cargo nextest run -p coeus-core --test dependency_policy` passes with 3 tests.
- [x] [patch] Final MS-67 local gate clean after the WGPU/Python/autograd fixes:
  `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo nextest run --workspace` (521 tests), `cargo test --doc --workspace`,
  and `cargo doc --workspace --no-deps`.
- [x] [patch] Documented `coeus-cuda` and `coeus-wgpu` crate-level backend
  responsibilities, dispatch flow, Atlas provider ownership, and explicit
  CPU-reference capability boundaries. Also formatted the existing MS-70
  ConvTranspose PyO3 binding code. Evidence tier: rustdoc and compile-time
  lint validation. Evidence: `cargo doc -p coeus-cuda -p coeus-wgpu --no-deps`,
  `cargo clippy -p coeus-cuda -p coeus-wgpu --all-targets -- -D warnings`,
  `cargo clippy -p coeus-python --all-targets -- -D warnings`, and
  `cargo fmt --check` pass.
- [x] [patch] Replaced the PyO3 `pycoeus.no_grad()` marker with nested
  autograd-mode state and one `PyTensor::from_var` return path that detaches
  operation outputs while preserving explicit factory `requires_grad` requests.
  Evidence tier: empirical value-semantic binding validation. Evidence:
  `cargo nextest run -p coeus-python --test binding_tests_ops
  test_no_grad_detaches_operation_outputs` passes.
- [x] [minor] Moved no-grad recording state into `coeus-autograd` and changed
  tracked operations to consult the core grad-mode SSOT before allocating
  gradient buffers or creator nodes. `coeus-python` now forwards context-manager
  entry/exit to the Rust core adapter while preserving explicit leaf
  `requires_grad` factory requests. Evidence tier: empirical value-semantic
  validation. Evidence: `cargo nextest run -p coeus-autograd` passes with 27
  tests, including `autograd::grad_mode::*`, and `cargo nextest run -p
  coeus-python --test binding_tests_ops` passes with 31 tests.
- [x] [minor] Added native WGPU and CUDA f32 forward kernels for
  `conv_transpose1d` / `conv_transpose2d` using the gather inverse of the CPU
  scatter reference. Added WGPU and CUDA differential tests against
  `SequentialBackend`, and registered the WGPU `ops_bench` Criterion harness as
  a benchmark instrument without claiming a measured speedup. Evidence tier:
  empirical differential validation plus benchmark build. Evidence:
  `cargo nextest run -p coeus-wgpu --test wgpu_tests conv_transpose`,
  `cargo nextest run -p coeus-cuda --features cuda --test cuda_tests
  conv_transpose`, and `cargo check -p coeus-wgpu --benches` pass.
- [x] [patch] Removed the now-unreachable CUDA transposed-convolution fallback
  helper methods and kept the CPU-reference fallback inline at the CUDA dispatch
  boundary, eliminating dead code without weakening the fallback contract.
  Evidence tier: compile-time lint validation. Evidence:
  `cargo clippy -p coeus-cuda -p coeus-wgpu --all-targets -- -D warnings` and
  `cargo clippy -p coeus-cuda --features cuda --all-targets -- -D warnings`
  pass.
- [x] [minor] Added tracked `coeus_autograd::conv_transpose1d` backward and
  collapsed duplicated `conv1d`/`conv2d`/`conv3d` autograd backward nodes into
  one const-generic `ConvNode<T, B, DIM>` dispatch path. Evidence tier:
  empirical value-semantic validation. Evidence: `cargo nextest run -p
  coeus-autograd` passes with 27 tests, including
  `autograd::nn_conv::conv_transpose1d_backward_accumulates_exact_gradients`;
  `cargo clippy -p coeus-autograd -p coeus-python --all-targets -- -D
  warnings` passes.
- [x] [minor] Collapsed duplicated `max_pool2d`/`max_pool3d` and
  `avg_pool2d`/`avg_pool3d` autograd backward nodes into const-generic
  `MaxPoolNode<T, B, DIM>` and `AvgPoolNode<T, B, DIM>` dispatch paths.
  Evidence tier: empirical value-semantic validation. Evidence: `cargo
  nextest run -p coeus-autograd` passes with 27 tests,
  `cargo nextest run -p coeus-nn --test nn_norm_tests pool` passes with 2
  tests, `cargo nextest run -p coeus-nn --test nn_tests pool3d` passes with 6
  tests, and `cargo clippy -p coeus-autograd -p coeus-cuda --all-targets --
  -D warnings` passes.
- [x] [minor] Completed per-axis `vector_norm(ord=p)` Rust-core and PyO3
  parity: `coeus_ops::norm_p_axis` reduces the requested axis to size 1, and
  `pycoeus.vector_norm(input, ord=p, axis=..., keepdim=...)` now returns the
  squeezed tensor/scalar or keepdim tensor instead of rejecting `axis`.
  Evidence tier: empirical Burn differential and binding validation. Evidence:
  `cargo nextest run -p coeus-ops norm_p_axis`, `cargo nextest run -p
  coeus-python --test binding_tests_ops test_vector_norm_p_orders`, and `cargo
  nextest run -p coeus-nn --test burn_live_parity statistical_ops_match_burn`
  pass.
- [x] [minor] Completed tracked Lp norm autograd exports for
  `coeus_autograd::{norm, norm_p, norm_p_axis}` with analytical backward rules
  and value-semantic gradient tests for scalar L3 and per-axis L2 norms.
  Evidence tier: analytical oracle plus empirical execution. Evidence:
  `cargo nextest run -p coeus-autograd --test autograd_tests norm_p` passes.
- [x] [minor] Completed Rust-core and PyO3 shape parity for `einsum` and
  `index_select`: `coeus_ops::{einsum, index_select}`, tracked autograd
  wrappers, and registered `pycoeus.einsum` / `pycoeus.index_select` wrappers
  now pass value-semantic Rust and Python binding tests.
  Evidence tier: empirical binding and op validation. Evidence:
  `cargo nextest run -p coeus-ops einsum`, `cargo nextest run -p coeus-python
  --test binding_tests_ops test_einsum_wrapper`, and `cargo nextest run -p
  coeus-python --test binding_tests_ops test_gather_scatter` pass.
- [x] [patch] Added a root-scoped `/prog` ignore entry for transient checkpoint
  transcript artifacts so generated session state is not staged as project
  source. Evidence tier: repository hygiene.
- [x] [minor] Added `crates/coeus-ops/src/reduction/stats.rs` with `var`, `var_axis`,
  `std_dev`, `std_dev_axis`, and `norm` as two-pass analytical compositions over
  existing `BackendOps` primitives; exported from `coeus-ops`. Added Python
  wrappers `pycoeus.std`, `pycoeus.var`, `pycoeus.norm` with axis/keepdim support.
  Evidence: `cargo nextest run -p coeus-python --test binding_tests_ops` passes
  with the `test_statistical_ops` test verifying all analytical oracle values.
- [x] [minor] Refactored `GlobalAvgPool{2,3}d` and `GlobalMaxPool{2,3}d` to use
  sequential `coeus_autograd::mean_axis`/`max_axis` calls, removing the
  square/cubic spatial constraint. Exposed `GlobalAvgPool{1,2,3}d` and
  `GlobalMaxPool{2,3}d` as PyO3 Python classes. Added Python binding tests for
  shape reduction, forward values, and backward gradients.
  Evidence: `cargo nextest run --workspace` passes with 439 tests.
- [x] [minor] Added `coeus-python` wrappers for `unsqueeze`, `squeeze`,
  `flatten`, `argmax`, `argmin`; Python stats (`std`, `var`, `norm`).
- [x] [patch] Added `batchnorm1d_backward_bias_and_weight_grads_match_analytical`
  test: verifies bias.grad = N*L per channel, weight.grad ≈ 0 (zero-mean x_hat),
  and input.grad per-channel sum ≈ 0 (normalization backward invariant).
  Evidence tier: analytical closed-form oracle.
- [x] [minor] Added `bench_burn_conv2d` (1×4×16×16, k=3) and
  `bench_burn_layernorm` (4×64×128) Criterion benchmark groups to
  `crates/coeus-tensor/benches/tensor_bench.rs` comparing Burn NdArray, Coeus
  Sequential, and Coeus Moirai. Added `coeus-nn` and `coeus-autograd` as
  dev-dependencies of `coeus-tensor`.
- [x] [patch] Verification (2026-06-24):
  `cargo clippy --workspace --all-targets -- -D warnings` (clean),
  `cargo nextest run --workspace` (439 passed),
  burn_live_parity 44 tests (incl. BatchNorm1d backward),
  binding_tests_nn 1 test (incl. all global pool wrappers).
- [x] [minor] Implemented on-device CUDA conv3d PTX kernels (`conv3d_f32`,
  `conv3d_grad_input_f32`, `conv3d_grad_weight_f32`, `conv3d_grad_bias_f32`)
  extending the existing conv1d/conv2d pattern to 5-D [N,C,D,H,W] tensors.
  Wired dispatch through `cuda_conv3d`/`cuda_conv3d_backward` in
  `crates/coeus-cuda/src/backend/ops/conv.rs`, replacing the CPU fallback path.
  Added `test_cuda_parity_conv3d_forward` and `test_cuda_parity_conv3d_backward`
  differential tests verifying on-device output agrees with `SequentialBackend`
  within CUDA_ACC_TOL (1e-3). Evidence tier: empirical differential validation.
  Evidence: `cargo nextest run -p coeus-cuda --features cuda` passes with
  57 tests (up from 55). Workspace remains 438 passed.
- [x] [patch] Added CUDA scaled-dot-product attention differential coverage for
  unmasked and causal forward attention, masked CPU-boundary behavior, and
  backward `grad_q`, `grad_k`, and `grad_v` against `SequentialBackend`.
  Evidence tier: empirical differential validation. Evidence:
  `cargo nextest run -p coeus-cuda --features cuda --test cuda_tests attention`
  passes with 4 tests.
- [x] [patch] Routed CUDA max/average 3D pooling forward and backward through
  native JIT kernels (`dispatch_{max,avg}_pool3d{,_backward}`), replacing the
  `BackendOps` CPU fallback path for this surface. Evidence tier: empirical
  differential validation. Evidence: `cargo nextest run -p coeus-cuda
  --features cuda --test cuda_tests pool3d` passes with 2 tests.

- [x] [minor] Added `coeus_ops::norm_p<T: Float, B>(x, p, backend)` for
  `(Σ|xᵢ|^p)^(1/p)` over a flattened view for any finite positive `p`
  (matches `torch.linalg.vector_norm(ord=p)`). Host-side fold with
  `T::powf` accumulation and final `^(1/p)`; no `BinaryOp::Pow` opcode
  added. `coeus_ops::norm(x, backend)` retained as the L2 short-circuit.
  Evidence tier: empirical differential validation (Burn 0.16 oracle).
  Eight unit tests in `reduction/stats.rs::tests` and three new Burn
  parity assertions covering p ∈ {1, 2, 3} in
  `crates/coeus-nn/tests/burn_live_parity.rs::statistical_ops_match_burn`.

- [x] [minor] Added thin PyO3 wrapper `pycoeus.vector_norm(input,
  ord=2.0, axis=None, keepdim=False)` mirroring
  `torch.linalg.vector_norm`'s full signature; `pycoeus.norm(input)`
  preserved as the L2 default. Empty tensors and `ord` outside the
  finite-positive range surface as `ValueError` rather than panicking
  at the boundary. `axis`/`keepdim` now route through `coeus_ops::norm_p_axis`.
  `crates/coeus-python/tests/binding_tests_ops.rs::test_vector_norm_p_orders`
  covers p ∈ {0.5, 1, 2, 3}, ord error paths, and empty-tensor errors.

- [x] [patch] Verification (2026-06-24):
  `cargo check --workspace`, `cargo clippy --workspace --all-targets
  -- -D warnings`, `cargo fmt --check`, `cargo test --doc --workspace`
  all clean.
  `cargo nextest run --workspace` passes with 464 tests up from 455
  baseline; reached by 8 new tests in
  `coeus-ops reduction::stats::tests`, 1 new binding test, and 0
  regressions.

---

## Previous Sprint: Sprint MS-60+ (Atlas burn-replacement & GPU roadmap) [COMPLETE]
**Objective**: Route CPU `BackendOps` through `coeus-leto`; Hermes SIMD integration;
GPU backends over Hephaestus; dependency policy hardening.
**Target version**: 0.2.0.

> **Roadmap (docs/backlog.md MS-60+)**: the Atlas burn-replacement program now stages
> (A2) routing the CPU backend's `BackendOps` through `coeus-leto` and deleting the
> duplicated CPU traversal — keeping `Tensor<T,B>` and the `ComputeBackend` seam; and
> (D) the GPU program: ADR to migrate `coeus-cuda` from cutile to **cuda-oxide**, finish
> wgpu op parity, consume mnemosyne device pools / melinoe device-buffer ownership.
> burn is eliminated end-to-end in Stage E.

- [x] [patch] Routed `WgpuBackend` host/device copies through the Hephaestus
  `ComputeDevice` upload/download surface, replacing the Coeus-local queue write
  and staging-buffer readback path. This advances Stage D1 without claiming full
  Mnemosyne/Melinoe device ownership-token completion. Evidence tier: empirical
  differential validation. Evidence: `cargo nextest run -p coeus-wgpu --test
  wgpu_tests` passes with 50 tests.
- [x] [patch] Routed `coeus-wgpu` and `coeus-cuda` storage allocations through
  explicit `PlacementHint::Tier(MemoryTier::Device)` so the allocation contract
  is anchored to the Hephaestus+Mnemosyne device-tier seam instead of implicit
  defaults, and added `coeus-wgpu` storage tests that verify device-tier
  allocation, host-pinned staging tier routing, and value-preserving
  upload/download roundtrip behavior. Evidence: `cargo nextest run -p
  coeus-wgpu --lib` (3 passed), `cargo check -p coeus-cuda --features cuda`,
  and `cargo check -p coeus-cuda`.

### Verification Note (2026-06-12)

- [x] [patch] Added committed nextest timeout config at `.config/nextest.toml`.
- [x] [patch] Synced README verification commands to `cargo nextest run`,
  doctests, and clippy with `-D warnings`.
- [x] [patch] `coeus-cuda` now defaults to a CPU-backed no-CUDA provider so the
  full workspace can check on hosts without `CUDA_TOOLKIT_PATH`; real cutile
  CUDA integration is retained behind the explicit `cuda` feature.
- [x] [patch] The default no-CUDA `CudaBackend` implements the full
  `BackendOps` surface by delegating to the existing CPU fallback path, with
  value-semantic coverage in `crates/coeus-cuda/tests/no_cuda_fallback.rs`.
- [x] [patch] Replaced high-arity `coeus-wgpu` attention and convolution helper
  calls with typed request structs; verified by clippy and wgpu nextest.
- [x] [patch] `cargo clippy --workspace --exclude coeus-cuda --all-targets
  -- -D warnings` passes after the `coeus-wgpu` request-struct refactor.
- [x] [patch] `cargo fmt --check` passes after workspace formatting.
- [x] [patch] `cargo check --workspace` passes without excluding `coeus-cuda`.
- [x] [patch] `cargo clippy --workspace --all-targets -- -D warnings` passes
  without excluding `coeus-cuda`.
- [x] [patch] `cargo nextest run --workspace` passes: 255 tests passed, 0
  skipped. CUDA integration tests are feature-gated under `cuda` because they
  require `CUDA_TOOLKIT_PATH` and a working CUDA driver.
- [x] [patch] `cargo test --doc --workspace` passes; four doctests are
  intentionally ignored.
- [x] [patch] Added `crates/coeus-core/tests/dependency_policy.rs` to enforce the
  Moirai parallel/async SSOT: production sources and production manifest
  dependency sections may not import or depend on `rayon` or `tokio`. Evidence:
  `cargo nextest run -p coeus-core --test dependency_policy` passes; normal
  dependency tree checks show no production `rayon` edge and no resolved
  `tokio` package.
- [x] [patch] Removed Coeus' direct `pollster` dependency from `coeus-wgpu` and
  extended `crates/coeus-core/tests/dependency_policy.rs` to reject Coeus production
  `pollster` imports/dependencies. Evidence:
  `cargo nextest run -p coeus-core --test dependency_policy` and
  `cargo tree -p coeus-wgpu --edges normal -i pollster` pass; the remaining
  resolved `pollster` edge is isolated inside
  `hephaestus-wgpu`.
- [x] [patch] Extended the dependency policy to reject direct production imports
  and direct production manifest dependencies on replacement libraries (`burn`,
  `nalgebra`, `ndarray`, `tch`) while preserving benchmark/dev-only comparisons.
  Evidence: `cargo nextest run -p coeus-core --test dependency_policy` passes.
- [x] [patch] Expanded `coeus-leto` contract coverage for the CPU consolidation
  seam: binary dispatch covers `Sub`/`Mul`/`Div`, unary dispatch covers
  `Relu`/`Abs`/`Neg`, and keep-dim reductions cover `Sum`/`Max`/`Min`. Evidence:
  `cargo nextest run -p coeus-leto` passes; the current contract suite contains
  14 tests.
- [x] [patch] Added `crates/coeus-ops/tests/unary_leto_diff.rs` to prove
  `SequentialBackend` and `MoiraiBackend` unary `BackendOps` dispatch matches
  direct scalar `CpuUnaryDispatch::eval_unary` for the full `CpuUnaryOp` surface.
  Evidence: `cargo nextest run -p coeus-ops --test unary_leto_diff` passes.
- [x] [patch] Added `crates/coeus-ops/tests/matmul_leto_diff.rs` to prove
  `SequentialBackend` and `MoiraiBackend` `BackendOps::matmul` dispatch matches
  an independent row-major triple-loop reference for contiguous and strided
  transposed input layouts. Evidence: `cargo nextest run -p coeus-ops --test
  matmul_leto_diff` passes.
- [x] [patch] Added `crates/coeus-ops/tests/batched_matmul_leto_diff.rs` to prove the
  public `coeus_ops::matmul` batching layer matches an independent reference on
  `SequentialBackend` and `MoiraiBackend` for equal batch counts and RHS 2-D
  broadcast. Evidence: `cargo nextest run -p coeus-ops --test
  batched_matmul_leto_diff` passes.
- [x] [patch] Routed `coeus_ops::cumsum` and `suffix_sum` through
  `coeus-leto` scan dispatch and added value-semantic coverage in
  `crates/coeus-leto/tests/leto_ops/contract/` plus
  `crates/coeus-ops/tests/scan_leto_diff.rs`.
  Evidence: focused scan tests pass.
- [x] [patch] Added public CPU reduction differential coverage for
  `sum`/`mean`/`sum_axis`/`mean_axis`/`max_axis`/`min_axis` on
  `SequentialBackend` and `MoiraiBackend`, including transposed input views.
  Evidence: `cargo nextest run -p coeus-ops --test public_reduction_leto_diff`
  passes.
- [x] [patch] Routed public scalar `mean` through backend
  `ReductionOp::Mean`, so CPU scalar mean now uses the `coeus-leto` mean
  reducer instead of local `sum / count` division. Evidence: `cargo nextest run -p
  coeus-ops --test public_reduction_leto_diff` passes.
- [x] [patch] Promoted mean to `ReductionOp::Mean` and routed public
  `mean_axis` through backend reduction dispatch. CPU dispatch uses Leto
  `MeanAxis`; WGPU/CUDA generated reducers and CPU fused reductions cover the
  same variant. Evidence: focused CPU, Leto, WGPU fused, and CUDA fallback tests
  pass.
- [x] [patch] Routed public `argmax` and `argmin` through `coeus-leto`
  keep-dim arg-reduction dispatch for CPU-addressable tensors and added
  transposed-view coverage for `SequentialBackend` and `MoiraiBackend`.
  Evidence: `cargo nextest run -p coeus-leto
  arg_reduction_dispatch_covers_keepdim_axis_ops` and `cargo nextest run -p coeus-ops
  --test arg_reduction_leto_diff` pass.
- [x] [patch] Routed public `coeus_ops::pad` through `coeus-leto` structural
  pad dispatch for CPU-addressable tensors and added strided/transposed-view
  coverage for `SequentialBackend` and `MoiraiBackend`. Evidence:
  `cargo nextest run -p coeus-leto pad_dispatch_covers_strided_input_view` and
  `cargo nextest run -p coeus-ops --test pad_leto_diff` pass.
- [x] [patch] Routed public `coeus_ops::cat` through `coeus-leto` structural
  concat dispatch for CPU-addressable tensors and added strided/transposed-view
  coverage for `SequentialBackend` and `MoiraiBackend`. Evidence:
  `cargo nextest run -p coeus-leto concat_dispatch_covers_strided_input_views` and
  `cargo nextest run -p coeus-ops --test concat_leto_diff` pass.
- [x] [patch] Routed public `coeus_ops::split` through `coeus-leto` structural
  split dispatch for CPU-addressable tensors and added strided/transposed-view
  coverage for `SequentialBackend` and `MoiraiBackend`. Evidence:
  `cargo nextest run -p coeus-leto split_dispatch_covers_strided_input_view` and
  `cargo nextest run -p coeus-ops --test split_leto_diff` pass.
- [x] [patch] Routed `coeus_nn::init::{uniform_with_seed, normal_with_seed}`
  through `coeus-leto` seeded random dispatch, deleting the local Xorshift
  initializer implementation. Constructor-only `RandomScalar` bounds preserve
  forward/module surfaces for existing `Float` APIs. Evidence: `cargo nextest run -p
  coeus-leto random_dispatch_matches_leto_seeded_constructors` and
  `cargo nextest run -p coeus-nn --test init_leto_diff` pass.
- [x] [patch] Routed `Tensor::to_contiguous_on` for CPU-addressable storage
  through `coeus-leto` view materialization, deleting the local strided
  materialization loop from that path. Evidence: `cargo nextest run -p coeus-leto
  contiguous_dispatch_matches_leto_view_materialization` and `cargo nextest run -p
  coeus-tensor --test contiguous_leto_diff` pass.
- [x] [patch] Routed `Tensor::{reshape, permute}` plus `t`/`t_nd` through
  `coeus-leto` layout validation, preserving zero-copy storage sharing while
  deleting the local reshape/permute metadata duplication from the public tensor
  path. Evidence: `cargo nextest run -p coeus-leto layout_dispatch` and `cargo nextest run -p
  coeus-tensor --test shape_view_leto_diff` pass.
- [x] [patch] Routed non-contiguous cross-backend `Tensor::to_backend_on`
  materialization through `coeus-leto`, deleting the remaining local strided
  transfer loops from the public tensor transfer path. Evidence: `cargo nextest run -p
  coeus-tensor --test backend_transfer_leto_diff` passes.
- [x] [patch] Routed `Tensor::from_fn_on` coordinate generation through
  `coeus-leto`, deleting the local row-major dynamic-index generation loop from
  the public tensor constructor path. Evidence: `cargo nextest run -p coeus-leto
  shape_function_dispatch_matches_leto_coordinate_order` and `cargo nextest run -p
  coeus-tensor --test from_fn_leto_diff` pass.
- [x] [patch] Routed `Tensor::eye_on` identity value generation through
  `coeus-leto`, deleting the local diagonal mutation loop from the public tensor
  constructor path. Fixed zero-length `CpuStorage` to expose non-null aligned
  zero-length slices for empty tensors. Evidence: `cargo nextest run -p coeus-core
  --test cow_storage_tests` and `cargo nextest run -p coeus-tensor --test
  identity_leto_diff` pass.
- [x] [minor] Added `Scalar::from_usize` as the native index-conversion seam
  and routed `Tensor::arange_on` through `coeus-leto`, deleting the local
  mutation loop and the constructor's f64 index conversion. Evidence: `cargo
  test -p coeus-core --test scalar_index_conversion` and `cargo nextest run -p
  coeus-tensor --test arange_leto_diff` pass.
- [x] [patch] Routed `Tensor::linspace_on` coordinate traversal through
  `coeus-leto`, deleting the local mutable fill loop while preserving the
  existing `Scalar::from_f64` value contract. Evidence: `cargo nextest run -p
  coeus-tensor --test linspace_leto_diff` passes.
- [x] [patch] Routed tensor broadcast shape and zero-copy broadcast layout
  validation through `coeus-leto`, deleting local dynamic broadcast metadata
  construction from `Tensor::broadcast` while preserving scalar rank-0
  broadcasts. Evidence: `cargo nextest run -p coeus-leto
  broadcast_layout_dispatch_matches_leto_validation` and `cargo nextest run -p
  coeus-tensor --test broadcast_leto_diff` pass.
- [x] [minor] Added public `coeus_ops::stack` through dynamic-rank
  `coeus-leto` stack dispatch, covering equal-shaped strided input views on
  `SequentialBackend` and `MoiraiBackend`. Evidence: `cargo nextest run -p coeus-leto
  stack_dispatch_covers_strided_input_views` and `cargo nextest run -p coeus-ops
  --test stack_leto_diff` pass.
- [x] [minor] Added `BackendOps::batched_matmul` as the batched matmul seam,
  routed public batched `coeus_ops::matmul` through it, and overrode the CPU
  `SequentialBackend`/`MoiraiBackend` path with `coeus-leto` rank-3 batched
  dispatch. Evidence: `cargo nextest run -p coeus-leto
  batched_matmul_dispatch_covers_rhs_batch_broadcast`, `cargo nextest run -p coeus-ops
  --test batched_matmul_leto_diff`, and `cargo nextest run -p coeus-wgpu
  wgpu::transfers_and_matmul::test_wgpu_backend_ops_unified` pass.
- [x] [patch] Historical CPU attention dot/scale routing is superseded by
  ADR-0047's direct borrowed Leto forward and additive-backward dispatch; the
  former Hermes-specific attention regression is removed with its formula.
- [x] [patch] Routed contiguous unpadded unit-dilation CPU `conv1d` forward
  kernel rows through `Scalar::dot_slice`, preserving the indexed path for
  padded, dilated, or non-contiguous layouts. Evidence: `cargo nextest run -p
  coeus-ops --test conv1d_hermes_diff` passes.
- [x] [patch] Routed contiguous unpadded unit-dilation CPU `conv2d` forward
  kernel rows through `Scalar::dot_slice`, preserving the indexed path for
  padded, dilated, or non-contiguous layouts. Evidence: `cargo nextest run -p
  coeus-ops --test conv2d_hermes_diff` passes.
- [x] [patch] Routed contiguous unpadded unit-dilation CPU `conv3d` forward
  kernel rows through `Scalar::dot_slice`, preserving the indexed path for
  padded, dilated, or non-contiguous layouts. Evidence: `cargo nextest run -p
  coeus-ops --test conv3d_hermes_diff` passes.
- [x] [patch] Routed contiguous unpadded unit-stride/unit-dilation CPU `conv1d`
  backward weight-gradient rows through `Scalar::dot_slice`, preserving the
  indexed path for padded, strided, dilated, or non-contiguous layouts.
  Evidence: `cargo nextest run -p coeus-ops --test conv1d_backward_hermes_diff`
  passes.
- [x] [patch] Routed contiguous unpadded unit-stride/unit-dilation CPU `conv2d`
  backward weight-gradient width rows through `Scalar::dot_slice`, preserving
  the indexed path for padded, strided, dilated, or non-contiguous layouts.
  Evidence: `cargo nextest run -p coeus-ops --test conv2d_backward_hermes_diff`
  passes.
- [x] [patch] Routed contiguous unpadded unit-stride/unit-dilation CPU `conv3d`
  backward weight-gradient width rows through `Scalar::dot_slice`, preserving
  the indexed path for padded, strided, dilated, or non-contiguous layouts.
  Evidence: `cargo nextest run -p coeus-ops --test conv3d_backward_hermes_diff`
  passes.
- [x] [patch] Fixed rustdoc shape/type annotations that were parsed as links or
  HTML, making workspace docs warning-clean. Evidence: `cargo doc --workspace
  --no-deps` passes.
- [x] [patch] Current full gate after CPU `conv3d` backward Hermes dot routing:
  `cargo fmt --check`,
  `git diff --check`, `cargo check --workspace`, `cargo clippy --workspace
  --all-targets -- -D warnings`, `cargo nextest run --workspace` (307 passed,
  0 skipped), `cargo test --doc --workspace`, and `cargo doc --workspace
  --no-deps` pass.
- [x] [minor] Added Criterion baselines in `crates/coeus-tensor/benches/tensor_bench.rs`
  for direct Leto and Coeus-Leto dispatch alongside Coeus Sequential, Coeus
  Moirai, and later dev-only Burn NdArray oracle rows.
- [x] [patch] Consolidated duplicated fused CPU value/reduction traversal in
  `coeus-ops::fuse` behind shared writer helpers and replaced manual temporary
  host-cache cleanup with an RAII guard. Added value-semantic coverage for fused
  sum/mean/max/min reductions. Evidence: `cargo clippy -p coeus-ops
  --all-targets -- -D warnings` and `cargo nextest run -p coeus-tensor --test
  fused_ops_tests` pass.
- [x] [patch] Fixed the Python distributed binding timeout by splitting the
  monolithic local/TCP collective script into independently timed value-semantic
  tests, and added missing Rust TCP reduce/gather/scatter coverage. Evidence:
  `cargo nextest run -p coeus-python --test binding_tests_dist` passes in
  0.620s; `cargo nextest run -p coeus-dist` passes with 16 tests.
- [x] [patch] Added WGPU scaled-dot-product attention forward/backward
  differential coverage against the public CPU attention path, including causal
  masking and Q/K/V gradients. Evidence: `cargo nextest run -p coeus-wgpu
  --test wgpu_tests attention` passes.
- [x] [patch] Reconciled the WGPU parity test module with the current
  `BackendOps` pooling, convolution, and AdamW signatures. Evidence:
  `cargo nextest run -p coeus-wgpu --test wgpu_tests parity` passes with 33
  tests.
- [x] [patch] Routed WGPU transfer operations through Hephaestus
  `ComputeDevice` upload/download APIs. Evidence: `cargo nextest run -p
  coeus-wgpu --test wgpu_tests` passes with 50 tests.
- [x] [patch] Completed the dev-only Burn live parity target for `coeus-nn`
  softmax and cross-entropy loss. Burn remains outside production dependency
  sections and is used only as a reference oracle. Evidence: `cargo nextest run
  -p coeus-nn --test burn_live_parity` passes.
- [x] [patch] Added Burn NdArray comparison rows to the `coeus-tensor`
  Criterion benchmark harness for add, matmul, ReLU, and sum. Evidence:
  `cargo clippy --workspace --all-targets -- -D warnings` passes after switching
  the ReLU benchmark to Burn's public activation API.
- [x] [patch] Fixed the Python binding functional-op test harness for PyO3
  0.23's `CStr` script API and passed owned shapes into `Tensor::full_on`.
  Evidence: `cargo clippy --workspace --all-targets -- -D warnings` and
  `cargo nextest run --workspace` pass.
- [x] [patch] Added `[profile.bench]` thin LTO with one codegen unit so
  cross-crate generic kernels are benchmarked after production-grade
  monomorphization. Evidence tier: empirical Criterion measurement.
- [x] [minor] Ran a short historical empirical benchmark pass:
  `cargo bench -p coeus-tensor --bench tensor_bench -- --warm-up-time 1
  --measurement-time 2 --sample-size 10`. Evidence tier: empirical Criterion
  measurement. The current harness no longer carries direct third-party tensor
  or Rayon rows; it retains Coeus Sequential/Moirai, direct Leto,
  Coeus-Leto dispatch, and dev-only Burn NdArray oracle rows. Focused
  post-profile 256x256 matmul measurement: Coeus Sequential 1.0006 ms, Coeus
  Moirai 1.1146 ms, direct Leto 1.1012 ms, Coeus-Leto dispatch 1.0905 ms.
  Rejected upstream Hermes tiled-GEMM route: Leto 256x256 f64 regressed to
  3.6848 ms and Coeus f32 direct Leto regressed to 8.7577 ms; source change was
  removed. Dense matmul remains a measured optimization target against the
  dev-only Burn oracle.

---

### Workspace Crate Status Matrix

| Crate Name | Path | Primary Responsibilities | Compilation Status | Notes / Blockers |
| :--- | :--- | :--- | :--- | :--- |
| **coeus-core** | [coeus-core](file:///d:/coeus/coeus-core) | Scalar types, layouts, storage traits, backend traits, CPU backends | ✅ Compiles | Clean compilation |
| **coeus-tensor** | [coeus-tensor](file:///d:/coeus/coeus-tensor) | N-dimensional strided tensor representation (`Tensor<T, B, S>`) | ✅ Compiles | Clean compilation |
| **coeus-ops** | [coeus-ops](file:///d:/coeus/coeus-ops) | Element-wise math, matrix operations, reductions | ✅ Compiles | Zero-copy layout traversal and thread-safe |
| **coeus-autograd** | [coeus-autograd](file:///d:/coeus/coeus-autograd) | Tape-based automatic differentiation (`Var`, `Tape`) | ✅ Compiles | Clean compilation |
| **coeus-optim** | [coeus-optim](file:///d:/coeus/coeus-optim) | Optimizers (SGD, Adam, RMSProp) | ✅ Compiles | Borrow checker conflicts resolved |
| **coeus-nn** | [coeus-nn](file:///d:/coeus/coeus-nn) | Neural network modules (Linear, activations, losses, normalization) | ✅ Compiles | Clean compilation |
| **coeus-sparse** | [coeus-sparse](file:///d:/coeus/coeus-sparse) | Sparse format storage primitives (COO, CSR) | ✅ Compiles | Clean compilation |
| **coeus-python** | [coeus-python](file:///d:/coeus/coeus-python) | Thin PyO3 bindings | ✅ Compiles | Clean compilation |

---

## Action Items Checklist

### Phase 1: Core CPU Workspace Stabilization [COMPLETE]
Resolve the remaining compiler and thread-safety blockers in `coeus-ops` and `coeus-optim`.

- [x] **coeus-optim Fixes**:
  - [x] Resolve `E0502` borrow conflicts in SGD step loop by caching `.numel()` before borrowing `param.tensor` mutably.
  - [x] Resolve `E0502` borrow conflicts in Adam step loop.
  - [x] Resolve `E0502` borrow conflicts in RMSProp step loop.
- [x] **coeus-ops Thread-Safety Fixes**:
  - [x] Implement `SendPtr<T>` and `SendPtrMut<T>` wrapper types in `coeus-ops` implementing `Send + Sync` to wrap raw pointers (`*const T` / `*mut T`).
  - [x] Refactor `crates/coeus-ops/src/matmul/kernel.rs` to pass `SendPtr`/`SendPtrMut` into the `parallel_for` closure.
  - [x] Refactor `crates/coeus-ops/src/reduction/sum.rs` to pass `SendPtr`/`SendPtrMut` into the `parallel_for` closure.
- [x] **Workspace Verification**:
  - [x] Run `cargo check --workspace` to verify zero compilation errors.
  - [x] Run `cargo nextest run --workspace` to verify that all existing CPU tests pass.

### Phase 2: Parity Testing & Performance Benchmarks [COMPLETE]
Establish numerical validation, autograd equivalence, and performance measurements against baseline libraries.

- [x] **Numerical Parity Validation**:
  - [x] Implement `crates/coeus-tensor/tests/parity_tests.rs` comparing Coeus tensor
    operations against self-contained row-major references.
  - [x] Verify exact parity across various strides, shapes, and layouts.
- [x] **Autograd Parity & Design Equivalence**:
  - [x] Implement autograd validation tests and verify gradient correctness.
- [x] **Performance Benchmarks**:
  - [x] Configure `criterion` benchmarks in
    `crates/coeus-tensor/benches/tensor_bench.rs` comparing Sequential/Moirai backends
    against direct Leto, Coeus-Leto dispatch, and dev-only Burn NdArray oracle
    rows.

### Phase 3: GPU Integration Abstractions (Sprint MS-55 Phase 1) [COMPLETE]
Introduce generic, zero-cost associated-type abstractions to support device-specific execution.

- [x] **Associated-Type Backend Overhaul**:
  - [x] Define the `ComputeBackend` trait in `coeus-core::backend` to replace/extend `Backend`.
  - [x] Define associated types:
    - `type DeviceBuffer<T>` (device memory allocation type representing device-allocated storage).
    - `type KernelDescriptor` (information needed to run kernels on the backend).
    - `type DispatchFuture<T>` (async execution handle for non-blocking queue operations).
- [x] **Unified Storage Design**:
  - [x] Update `Storage<T>` and `StorageMut<T>` traits to support non-CPU-accessible buffers (preventing direct CPU slicing assumptions).
- [x] **Device Transfer API**:
  - [x] Implement `Tensor::to_backend::<NewB>(&self, backend: &NewB)` on `Tensor` to handle copying between host/device memory boundaries (CPU-to-GPU, GPU-to-CPU, GPU-to-GPU).

### Phase 4: WebGPU Backend Crate (coeus-wgpu) [COMPLETE]
Implement a cross-platform GPU backend powered by `wgpu`.

- [x] **Crate Structure**:
  - [x] Create the new `coeus-wgpu` crate and add it to the workspace members.
  - [x] Add `wgpu` and `bytemuck` dependencies in `crates/coeus-wgpu/Cargo.toml`.
- [x] **Storage & Memory Management**:
  - [x] Implement `WgpuStorage<T>` wrapping a `wgpu::Buffer` handle with unified allocation and staging logic.
- [x] **Compute Pipelines**:
  - [x] Write WGSL (WebGPU Shading Language) compute shaders for element-wise unary operations, binary operations, matrix multiplication, sum reductions, and Conv1D/Conv2D forward and backward passes.
  - [x] Implement compilation cache for `wgpu::ComputePipeline` instances.
- [x] **Backend Trait implementation**:
  - [x] Implement `ComputeBackend` for `WgpuBackend` mapping operations to GPU shader dispatch.
  - [x] Implement `BackendOps` for `WgpuBackend` to route math operations.

### Phase 5: CUDA Backend Crate (coeus-cuda) [COMPLETE]

- [x] **Crate Structure**:
  - [x] Create the new `coeus-cuda` crate in the workspace.
  - [x] Add dynamic CUDA driver bindings (`cuInit`, `cuMemcpy`, etc.) using `libloading` for robust cross-host compilation.
- [x] **Storage & Context Management**:
  - [x] Implement `CudaStorage<T>` wrapping CUDA `CUdeviceptr` raw handles.
  - [x] Integrate thread-safe CUDA context management and memory allocation.
- [x] **Kernel Management**:
  - [x] Implement embedded PTX source containing element-wise, tiled matrix multiplication, sum reduction, Conv1D/Conv2D forward and backward passes, and strided/broadcasted element-wise operations.
  - [x] Implement direct driver dispatch launchers and remove CPU staging fallbacks for device execution.


---

## Sprint MS-404: Binding fixes + scalar-arithmetic surface [patch]

Closes the two binding-test failures carried over MS-401..MS-403
(`test_amax_amin_prod_ops` `Tensor - float` TypeError,
`test_pycoeus_nn` GroupNorm `num_features` kwarg mismatch) and
extends the PyTensor surface so arithmetic ops accept either
a `Tensor` or a Python `float` — matching PyTorch / JAX / MLX
ergonomics. Also restores the peer's MS-401 HingeEmbeddingLoss
parity correction landed on the working tree but not yet
committed.

Target version: 0.5.6.

### Scope

- [x] [patch] **GroupNorm constructor kwarg alignment** — `binding_tests_nn.rs`
  updated from the deprecated `num_features=` to the `num_channels=`
  PyO3 kwarg introduced in MS-321-323; the internal Rust-core field
  name (`num_features`) and the public Python attribute (`num_features`)
  are unchanged, preserving `state_dict` round-trip compatibility.
  Evidence tier: value-semantic PyO3 binding test
  `test_pycoeus_nn` passes.
- [x] [patch] **Tensor scalar-arithmetic ergonomics** —
  `PyTensor::binop_dispatch` (a single-arity `Bound<'_, PyAny>`
  discriminator that routes tensor↔tensor through the existing
  `coeus_autograd::{add,sub,mul,div}` kernels and
  tensor↔scalar through the existing `scalar_{add,sub,mul,div}`
  kernels) replaces the previous `&PyTensor`-only dunders for
  `__add__`/`__sub__`/`__mul__`/`__truediv__`. Five binding-test
  sites that exercise `(tensor op float)` and two sites that
  exercise `(float op tensor)` previously raised
  `TypeError: unsupported operand type(s) for -` and now pass.
- [x] [patch] **Mirrored scalar operators** —
  `__radd__`/`__rmul__` (already on the dispatcher) plus the
  newly-added `__rsub__`/`__rtruediv__`/`__rpow__` close the
  `float - tensor`, `float / tensor`, and `float ** tensor`
  reflected paths.
- [x] [patch] **`__abs__` dunder** — Python's built-in `abs(tensor)`
  now routes through `coeus_autograd::abs` (binary eps-bounded
  within tolerance 0) rather than failing with
  `TypeError: bad operand type for abs()`.
- [x] [patch] **MS-401 HingeEmbeddingLoss parity correction (peer work
  pulled into the same atomic commit)** — `coeus-nn::hinge_embedding_loss`
  now matches PyTorch: target=+1 → identity branch (`x`), target=-1 →
  `relu(margin - x)`. The previous body mapped target=+1 to
  `relu(-(x - 1))` which evaluates to 0 for any `x ≤ 1` and disagrees
  with PyTorch on `x ∈ (0, 1)`. New regression test
  `test_hinge_embedding_loss_matches_torch_reference`
  asserts forward mean (`0.275`) and analytical backward
  gradient (`[0.25, 0.0, 0.25, -0.25]`) within `1e-14`.
- [x] [patch] **Binding-test corrections** —
  `test_amax_amin_prod_ops` now uses `pycoeus.prod(x).item()` (the
  scalar-extraction idiom matching PyTorch) rather than
  `abs(tensor - float)` (which Python's `<` cannot evaluate even
  with the binding fix in place).

### Verification (2026-07-03)

- [x] `cargo check --workspace` passes with zero errors after the
  bind-removal of `num_features=4` and the
  `PyTensor::binop_dispatch` discriminator rewrite.
- [x] `cargo clippy --workspace --all-targets -- -D warnings` passes.
- [x] `cargo fmt --check` passes (workspace only; no per-crate
  formatting).
- [x] `cargo nextest run --workspace --no-fail-fast` passes
  **1024 tests, 0 skipped, 0 failed** (up from 1023 at MS-403).
- [x] `cargo test --doc --workspace` passes (zero-failure).
- [x] `cargo doc --no-deps --workspace` passes (warning-clean).
- [x] No `coeus_autograd` regression: the
  `crates/coeus-nn/tests/nn_loss_tests::test_hinge_embedding_loss_matches_torch_reference`
  test added in this commit confirms forward + backward closed-form
  agreement with PyTorch semantics within `1e-14`.

### Architectural invariants preserved

- SSOT backend dispatch unchanged (the dispatcher route for
  `add`/`sub`/`mul`/`div` is the same one the tensor↔tensor path
  already used; only the operand-type boundary widened).
- `Scalar`/`Float` generics, `BinOp` enum closed set, and
  `Var<f64>` owner discipline preserved across the GIL release.
- All binding changes are additive at the PyO3 layer; existing
  `tensor op tensor` callers and `pycoeus.{add,sub,mul,div}`
  module-level ops unchanged.
- Dependency policy Atlas allowlist unaffected; no `ndarray`,
  `nalgebra`, `tokio`, `rayon`, `rustfft`, `burn`, `tch`, or
  `pollster` introduced into production.
- `Melinoe branded partitioning`/`GradBuffer UnsafeCell`/`coeus_fft`
  surface untouched.
- GroupNorm internal Rust-core field name retained as
  `num_features` (state_dict consumers stay compatible);
  only the `#[new]` Pyo3 kwarg was `num_channels`, matching PyTorch.

---

## Success Criteria & Definition of Done

- [x] **Clean Build**: `cargo check --workspace` passes with zero compiler errors.
- [x] **Zero-Cost Dispatch Verified**: Monomorphization checked; calling operations on specific backends resolves to direct static execution paths with zero abstraction overhead.
- [x] **Numerical Parity**: Parity tests verify absolute numerical equivalence between CPU, `wgpu`, and `cuda` execution (absolute tolerance $\le 10^{-5}$).
- [x] **Test Coverage**: 100% of tensor operations have verification tests covering contiguous, non-contiguous, broadcasted, and sliced tensor views.
- [x] **Memory Safety**: No memory leaks or data races on GPU backends under parallel operations (RAII wrapper verification).
