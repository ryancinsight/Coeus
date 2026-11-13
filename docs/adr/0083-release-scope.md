# ADR 0083: Coeus 1.0 release scope

## Status

Accepted (2026-11-13). Rewrites the 2026-11-12 three-milestone draft in place:
that draft named quality states and a release act as milestones and omitted
`coeus-sparse`, `coeus-dist` and `coeus-fft`.

Revised 2026-11-13: adds the three seam exits, the `oversized_files` standing
target, because the ladder must prove
every seam the scope names and state a conformance target.

## Context

Coeus is the Atlas differentiation substrate: strided tensors, kernels,
reverse-mode autograd, neural-network layers, optimizers, vendor backends and
Python bindings over provider kernels (README, "Crate Structure"). The workspace
holds 16 member packages at 0.12.0. The only open capability gap is Python
comparison operators; every other open item is verification, architecture or
tightening work on delivered areas. The ladder at the top of `backlog.md`
is checked against this record.

Classification: **delivered** (on the default branch, witness given), **in
scope** (required before 1.0 and served by a ladder exit check), **non-goal**
(not part of 1.0; additive items can land in a 1.x minor).

## Decision

Coeus 1.0 is the delivered surface below plus one in-scope capability: the
`PyTensor` comparison operators `<`, `<=`, `>`, `>=`, `==`, `!=`. One milestone,
`M-tensor-stack`, carries 1.0.0 because no other capability is missing. Its
nine exits pin the new capability (exits 1 and 2), four delivered end-to-end
paths (matmul, classifier, nonlinear fit, batched fit) and the three delivered
integration seams (`HephaestusProvider`, `Communicator`, Apollo FFT). Exits 1
and 2 need an interpreter-capable host: CI's native tests exclude
`coeus-python` because its `extension-module` build leaves libpython unlinked
(`.github/workflows/ci.yml:178-182`), so they run on a host whose build links
libpython, such as the Windows development host.

### Foundation and CPU stack

| Package | Area | Status | Witness |
| --- | --- | --- | --- |
| `coeus-core` | `Scalar`/`Float`, strided `Layout`, `Storage`, `ComputeBackend`, sequential and Moirai CPU backends | delivered | `src/{dtype,layout,storage,backend}` |
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
| `coeus-accelerator` | staggered gradient/divergence dispatch | delivered | `src/staggered/` |
| `coeus-wgpu` | WebGPU backend over provider operations | delivered | `src/{backend,storage,fusion}.rs` |
| `coeus-cuda` | CUDA backend over provider operations | delivered | `src/{backend,storage,fusion}.rs` |
| `coeus-rocm`, `coeus-metal` | provider wiring | delivered | `src/backend/` |
| `coeus-cuda`, `coeus-metal` | `StaggeredPairOps` binding verified on real devices | non-goal | hardware evidence cannot be a runnable exit on the development host; item COEUS-STAGGERED-VENDOR-WIRING stays open |
| `coeus-rocm` | staggered kernels | non-goal | blocked on upstream Hephaestus kernels |
| `coeus-dist` | `Communicator`, thread-local and TCP collectives, gradient reduction | delivered | `src/{communicator,local,tcp,gradients}`; exit 8 |
| `coeus-dist` | transports beyond thread-local and TCP | non-goal | no source asks for one |
| `coeus-fft` | tensor and autograd FFT wrappers over Apollo | delivered | `src/{fft,autograd}.rs`; exit 9 |
| `coeus-fft` | FFT math owned by Coeus | non-goal | README invariant 5: Apollo owns it |

### Bindings and release

| Package | Area | Status | Witness |
| --- | --- | --- | --- |
| `coeus-python` | `pycoeus` tensor, ops, autograd, nn, optim and dist bindings | delivered | `src/{tensor,ops,nn,optim.rs,dist}`; tests `tests/binding_ops/` |
| `coeus-python` | comparison operators on `PyTensor` | in scope | exits 1 and 2; only the `eq`/`lt`/... methods exist today |
| `coeus-python` | API beyond comparison operators | non-goal | additive |
| all | crates.io publication and PyPI upload | non-goal | release authority; MS-445 and COEUS-REGISTRY-PACKAGE-1 stay as blocked items |
| all | new model families | non-goal | no source asks for one |

## Veto points

Each follows the recommended option; the user's veto redirects it.

1. One milestone at 1.0.0 rather than a pre-1.0 verification milestone: a
   quality-state milestone is forbidden, and no second capability is missing.
2. MS-445 and COEUS-REGISTRY-PACKAGE-1 are release acts, not user-named in the
   meta board: kept as `blocked` verification items, not exits.
3. The conformance standing target is `oversized_files` at or below 7, half the
   current 14 (`scripts/conformance-baseline.json` at meta
   `e0f893530eb2efe3de5918471bb2d18ad8cb2279`), the largest Coeus row.
4. `coeus-sparse` kernels, `coeus-dist` transports, ROCm staggered kernels and
   device-hosted CUDA/Metal staggered evidence are non-goals.

## Alternatives

- Keep the three-rung draft: rejected; its exits ran whole packages and its
  rungs were quality states and a release.
- Make the wheel and registry items exits: rejected; publication needs release
  authority and has no cargo-runnable check.

## Consequences

- Items carry `milestone: M-tensor-stack`; feature items name the exit they serve.
- The milestone closes when its exits pass at a fetched default commit; the
  1.0.0 release stays gated on release authority.
- This record is rewritten in place when scope changes.
