# ADR 0085: Coeus next-major release scope

## Status

Proposed (2026-11-16). Drafted by the `M-tensor-stack` closure sweep: the
milestone's nine exits passed at the fetched default commit `7db0beba`, so the
1.0.0 scope recorded in [ADR 0083](0083-release-scope.md) is delivered. This
record proposes the scope that follows it and takes effect when the user
accepts it; the user's veto redirects the proposal.

- Revised 2026-11-24: the `Scalar` supertrait drop leaves the 2.0.0 scope and is delivered in the 0.x line as a breaking pre-1.0 change, because `Scalar` is sealed (no outside implementor to migrate) and Coeus#695 ships it under Unreleased in the changelog; evidence CHANGELOG.md:5, Cargo.toml:24 (0.15.0) and ADR 0068.

## Context

Coeus is the Atlas differentiation substrate: strided tensors, kernels,
reverse-mode autograd, neural-network layers, optimizers, vendor backends and
Python bindings over provider kernels (README, "Crate Structure"). The
workspace holds 16 member packages at 0.12.0.

The `M-tensor-stack` milestone closed with its nine exits green at
`7db0beba`: the Python comparison operators and their example, the matmul,
classifier and nonlinear-fit examples, and the `HephaestusProvider`,
`Communicator` and Apollo FFT seams. The comparison-dunder feature item
landed in PR 645, and the oversized-file standing target holds at seven files
after PR 651.

What remains open is not missing capability but unfinished convergence on
delivered areas:

- `ATLAS-COEUS-BACKEND-045`: `HephaestusBackend<P>` still lacks
  `PoolOps`/`UnfoldFoldOps`, so it does not satisfy `BackendOps<f32>` for the
  Metal and ROCm backends; the pool and unfold/fold kernels stay forked per
  vendor crate until the upstream seam lands.
- `COEUS-ACCELERATOR-PADOPS-ADOPTION`: the same upstream seam, reached
  through the pad-ops contract.
- The remaining items are verification, architecture and tightening work with
  no user-visible capability gap.

No source asks for a new model family, a new transport, sparse kernels or a
Python API beyond the comparison operators.

## Decision

The next major is **2.0.0**, and it carries one milestone,
`M-provider-surface`, whose outcome is a single vendor-neutral provider
surface: the pool and unfold/fold operations reach the Metal and ROCm
backends through the `HephaestusProvider` seam instead of per-vendor forks.
That seam change is a breaking change to public surface, which is what the
major version admits.

The `coeus_core::Scalar` marker change (Eunomia's `Pod` as the only layout
marker) is not 2.0.0 work. It is delivered in the 0.x line as a breaking
change, which Cargo treats as incompatible across a 0.x minor bump, and ships
with the next breaking 0.x version beside the Unreleased Hephaestus 0.32
change (CHANGELOG.md:5-7). `Scalar` is sealed, so no outside crate implements
it; the break reaches only generic code that relied on `T: Scalar` implying
`bytemuck::Pod` ([ADR 0068](0068-coeus-eunomia-device-layout.md)).

The scope of every shipped crate follows. Classification is **delivered** (on
the default branch, witness given), **in scope** (required before 2.0 and
served by a milestone exit check) or **non-goal** (not part of 2.0; additive
items can land in a 2.x minor).

### Foundation and CPU stack

| Package | Area | Status | Witness |
| --- | --- | --- | --- |
| `coeus-core` | `Scalar`/`Float`, strided `Layout`, `Storage`, `ComputeBackend`, sequential and Moirai CPU backends | delivered | `src/{dtype,layout,storage,backend}` |
| `coeus-core` | `Scalar` supertrait set reduced to Eunomia's marker | delivered | `src/dtype/traits/scalar.rs`; [ADR 0068](0068-coeus-eunomia-device-layout.md) |
| `coeus-tensor` | `Tensor<T, B>` with copy-on-write, views, broadcasting, rkyv checkpointing | delivered | `src/{tensor,views,broadcast,checkpoint}.rs` |
| `coeus-ops` | elementwise, matmul, reductions, convolution, pooling, embedding, interpolation, attention, fused expressions | delivered | `src/{unary,binary,matmul,reduction,fuse}`; exit `book_matmul` |
| `coeus-autograd` | tape-based reverse mode over `Var<T, B>`, gradcheck harness | delivered | `src/{var,backward,node,gradcheck}.rs` |
| `coeus-nn` | linear, convolution, normalization, pooling, attention, transformer, RNN, activations, losses | delivered | `src/{linear,conv,normalization,attention,transformer,rnn}`; exit `mlp_classifier` |
| `coeus-optim` | SGD, Adam, AdamW, RMSProp, AdaGrad, clipping, schedulers | delivered | `src/{sgd,adam,adamw,rmsprop,adagrad,clip,scheduler}.rs` |
| `coeus-optim` | Levenberg-Marquardt and leading-axis batched least squares | delivered | `src/least_squares/`; exits on `least_squares::tests` |
| `coeus-sparse` | `CooTensor`, `CsrTensor` data structures | delivered | `src/{coo,csr}.rs` |
| `coeus-sparse` | sparse kernels owned by the crate | non-goal | the README scopes it to data structures; products live in `coeus-ops` |
| `coeus-cpu` | const-rank dispatch onto Leto array kernels | delivered | `src/{dispatch,convert}.rs` |

### Accelerator and integration

| Package | Area | Status | Witness |
| --- | --- | --- | --- |
| `coeus-accelerator` | vendor-neutral device storage, transfer, layout validation and dispatch over `HephaestusProvider` | delivered | `src/{storage,backend,layout,seam}.rs`; exit 7 |
| `coeus-accelerator` | `PoolOps`/`UnfoldFoldOps` on the Hephaestus provider | in scope | `M-provider-surface`; items `ATLAS-COEUS-BACKEND-045`, `COEUS-ACCELERATOR-PADOPS-ADOPTION` |
| `coeus-accelerator` | staggered gradient/divergence dispatch | delivered | `src/staggered/` |
| `coeus-wgpu` | WebGPU backend over provider operations | delivered | `src/{backend,storage,fusion}.rs` |
| `coeus-cuda` | CUDA backend over provider operations | delivered | `src/{backend,storage,fusion}.rs` |
| `coeus-rocm`, `coeus-metal` | provider wiring | delivered | `src/backend/` |
| `coeus-cuda`, `coeus-metal` | `StaggeredPairOps` binding verified on real devices | non-goal | hardware evidence cannot be a runnable exit on the development host; item `COEUS-STAGGERED-VENDOR-WIRING` stays open |
| `coeus-rocm` | staggered kernels | non-goal | blocked on upstream Hephaestus kernels |
| `coeus-dist` | `Communicator`, thread-local and TCP collectives, gradient reduction | delivered | `src/{communicator,local,tcp,gradients}`; exit 8 |
| `coeus-dist` | transports beyond thread-local and TCP | non-goal | no source asks for one |
| `coeus-fft` | tensor and autograd FFT wrappers over Apollo | delivered | `src/{fft,autograd}.rs`; exit 9 |
| `coeus-fft` | FFT math owned by Coeus | non-goal | README invariant 5: Apollo owns it |

### Bindings and release

| Package | Area | Status | Witness |
| --- | --- | --- | --- |
| `coeus-python` | `pycoeus` tensor, ops, autograd, nn, optim and dist bindings | delivered | `src/{tensor,ops,nn,optim.rs,dist}`; tests `tests/binding_ops/` |
| `coeus-python` | comparison operators on `PyTensor` | delivered | exits 1 and 2 of the closed milestone; PR 645 |
| `coeus-python` | API beyond comparison operators | non-goal | additive |
| all | crates.io publication and PyPI upload | non-goal | release authority; `MS-445` and `COEUS-REGISTRY-PACKAGE-1` stay as blocked items |
| all | new model families | non-goal | no source asks for one |

## Veto points

Each follows the recommended option; the user's veto redirects it.

1. The next major is 2.0.0 with one milestone, `M-provider-surface`, rather
   than a 1.1 minor: the provider-seam change is breaking, and a milestone
   must carry a capability a user can exercise end to end. The `Scalar` marker
   change is outside this choice: it is already delivered as a breaking 0.x
   change, and until its release the user can still hold that release or
   have the change reverted.
2. The milestone waits for the upstream Hephaestus pool and unfold/fold seam.
   The alternative, forking the kernels into the Metal and ROCm backends
   again, is rejected: it keeps the duplication the item exists to remove.
3. `coeus-sparse` kernels, `coeus-dist` transports, ROCm staggered kernels and
   device-hosted CUDA/Metal staggered evidence are non-goals.

## Alternatives

- Ship the residual convergence as 1.x patches: rejected; the provider seam
  changes public surface and needs the major version.
- Declare 2.0.0 with no milestone because no capability is missing: rejected;
  a version without an exercised capability gives the ladder nothing to
  check.
- Hold the `Scalar` marker change for 2.0.0: rejected; `Scalar` is sealed, so
  no implementor migration exists to batch with the provider seam, and the
  change reaches only generic code that relied on the implied
  `bytemuck::Pod` bound.

## Consequences

- No ladder is drafted until this proposal is answered; until then the
  repository takes only non-feature items and the open items carry no
  milestone.
- The ladder and its item milestones are drafted when this proposal is
  accepted.
- This record is rewritten in place when scope changes.
