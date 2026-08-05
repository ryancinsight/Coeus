# ADR 0055: Provider-owned binary cross-entropy with logits

- Status: Accepted
- Date: 2026-08-05
- Scope: `coeus-autograd::bce_with_logits`
- Tracking: `docs/backlog.md#coeus-bce-logits-provider-001`

## Context

`bce_with_logits` accepted a generic `BackendOps` backend but required
CPU-addressable storage in practice. Forward copied logits and targets to host
memory, evaluated the stable scalar formula in a `Vec<T>`, and uploaded a
single-element result. Backward retained host vectors for the saved derivative
and logits, copied the output gradient to the host, then uploaded gradient
tensors. This violated the backend contract: a CUDA, WGPU, ROCm, or Metal
variable could enter the API only by crossing the host boundary.

The existing Coeus operation surface already provides provider-dispatched
sigmoid, ReLU, absolute value, exponentiation, `log1p`, elementwise arithmetic,
and axis mean. Those operations route CPU execution through Leto and accelerator
execution through the selected Hephaestus backend.

## Decision

Implement the stable expression

`mean(max(z, 0) - z * y + log1p(exp(-abs(z))))`

with the existing provider operations. Flatten only the provider result for
the mean reduction, keeping the one-element output on the selected backend.
Save `sigmoid(z) - y` and the mean scale as backend-native tensors in the
backward node. Compute both input gradients with provider arithmetic and add
them directly to the backend gradient buffers.

WGPU and CUDA mark the BCE unary primitives (`Relu`, `Log1p`, and `Sigmoid`)
as provider-owned in both contiguous and strided dispatch tables. If the
Hephaestus call cannot satisfy its layout contract, the backend returns a typed
error; it does not fall through to a Coeus-local kernel. ROCm and Metal already
route these primitives through their Hephaestus elementwise bridges.

The public function signature remains unchanged. No host-staging fallback,
CPU-addressable bound, compatibility wrapper, or new provider capability is
introduced. Other host-staged loss families remain separate migration items.

## Consequences

- CPU and accelerator execution share one generic implementation and dispatch
  through their existing provider seams.
- Backward state no longer retains per-element host vectors.
- The implementation has more intermediate provider tensors than the old host
  loop; this is required to preserve backend ownership and is not a runtime or
  allocation improvement claim without controlled measurement.
- The stable formula and existing mean reduction contract remain unchanged.

## Verification

- Existing CPU tests compare forward and both gradients with independent
  analytical references.
- The source audit must find no `copy_to_host`, `CpuAddressableStorage`, or
  host `Vec<T>` state in this operation.
- Locked all-target checks and warning-denied lints verify generic backend
  compilation; provider CI verifies WGPU, CUDA, ROCm, and Metal integration.
