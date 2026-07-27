# ADR-0027: Make unsupported ConvTranspose3d dispatch statically unavailable

- Status: accepted
- Date: 2026-07-27
- Scope: `coeus-ops` and `coeus-autograd` transposed-convolution dispatch plus
  the Coeus NN wrapper

## Context

`ConvOps::conv_transpose3d` had a generic default that copied accelerator
storage to host memory, ran the CPU scatter kernel, and copied the result back.
WGPU and CUDA provide native 1-D and 2-D transposed-convolution dispatch, but
neither provides a native 3-D operation. The generic `BackendOps` surface made
the missing accelerator capability appear available and permitted silent host
execution.

## Decision

Move 3-D transposed convolution to the dedicated `ConvTranspose3dOps`
capability seam. The default implementation is provided only for
`CpuBackend`; the public Coeus operation dispatches through the capability
seam, while the current autograd forward/backward node and NN module remain
CPU-only because their gradient implementation is scalar host code. CPU
backends retain the canonical scatter kernel and gradient loops. Native
WGPU/CUDA 1-D and 2-D dispatch remains unchanged. Accelerator 3-D support
becomes available only when its owning provider implements the complete native
operation family; no host fallback, runtime backend-name branch, or
compatibility adapter is added.

## Consequences

This is a breaking generic capability boundary: callers of the CPU-only NN and
autograd 3-D paths state `CpuBackend`, while the public operation requires the
new capability trait. Accelerator callers fail at compile time until a
provider-owned kernel exists. The reduction and transposed-convolution
boundaries now use the same static dispatch rule. Native 3-D provider work
remains a separate Hephaestus/provider item and can implement the capability
seam without inheriting the CPU default.

## Verification

CPU value-semantic differential tests retain the existing scatter oracle. The
backend-parity matrix verifies that WGPU, CUDA, ROCm, and Metal compilation and
focused contracts remain warning-clean; the unavailable accelerator 3-D path
is enforced by the `CpuBackend` bound rather than by a runtime test branch.
