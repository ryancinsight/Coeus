# ADR-0049: Route random initialization through Leto and Hephaestus

- Status: Accepted
- Date: 2026-08-02
- Scope: Coeus CPU, WGPU, CUDA, ROCm, Metal, neural-network initialization,
  and Python bindings
- Change class: `[major] [arch]`
- Board item: `COEUS-RANDOM-INIT-PROVIDER-001`

## Context

The public neural-network random initializers always call `coeus-leto`, collect
the generated values in a host `Vec`, and rebuild the tensor with
`Tensor::from_slice_on`. Backend selection therefore does not select the
execution provider: WGPU, CUDA, ROCm, and Metal all execute through the CPU
Leto path and then upload. The same functions convert provider failures into
input-dependent panics, and Xavier/Kaiming fan arithmetic accepts zero and can
overflow.

The locked Hephaestus revision already provides one device-neutral
`RandomInitOps` contract with WGPU, CUDA, ROCm, and Metal implementations. Its
seeded contract delegates generation to Leto and owns the provider transfer,
which preserves identical deterministic values across backends without making
Coeus own vendor dispatch or a second random algorithm.

## Decision

Add an interface-segregated Coeus random-initialization capability. Sequential
and Moirai implementations call the dynamic-rank `coeus-leto` adapter.
Accelerator implementations project their backend into one generic
`coeus-hephaestus` bridge, which dispatches once by rank and monomorphizes the
locked Hephaestus operation for the selected provider.

The operation returns newly initialized backend storage. `coeus-nn` replaces
the tensor storage in place without constructing a second tensor or performing
a Coeus-owned host copy. The provider remains responsible for allocation and
transfer behavior. Coeus supports the existing random-initializer rank domain
of one through six on every backend.

CPU storage initializes its raw allocation before any readable `T` slice can
exist. Leto then writes into that single final allocation. COW detachment copies
only from initialized source storage, preserving the same validity boundary.

All random initializer functions return typed backend results. Xavier and
Kaiming validate positive fan values and checked fan arithmetic before
dispatch. Every in-repository caller migrates in the same change; Python maps
invalid domains to `ValueError` and provider failures to `RuntimeError`. No
compatibility wrapper or infallible parallel API remains.

## Alternatives rejected

- Retain the Leto host-vector path for accelerators: rejected because backend
  identity would continue to disagree with provider ownership.
- Add random kernels or vendor branches in Coeus: rejected because Hephaestus
  already owns the accelerator role and its per-provider implementations.
- Fall back from an accelerator failure to Leto: rejected because it hides a
  provider fault and changes placement silently.
- Keep infallible APIs and panic on failure: rejected because rank, fan, device,
  allocation, and transfer failures are reachable from public input or runtime
  state.

## Verification

Generic CPU tests compare Sequential and Moirai results directly with Leto for
uniform, normal, Xavier, and Kaiming distributions. Accelerator tests compare
seeded provider values with the same Leto oracle and assert that invalid ranks
return typed errors before device acquisition. Neural-network tests assert
failure atomicity after validation and provider errors. Python tests retain
exact constant behavior and analytically bounded distribution evidence while
adding typed failure propagation. Warning-denied Clippy, Nextest, doctests,
semver classification, and exact-head CUDA/WGPU/ROCm/Metal CI gate the
migration.

These checks establish dispatch selection and value semantics. They do not
establish a runtime, allocation-count, bandwidth, or resident-memory
improvement; those claims require controlled measurements.

## Revisit trigger

Revisit when Hephaestus provides device-native deterministic generation. The
Coeus seam remains unchanged because transfer and generation policy stay owned
by the selected provider.
