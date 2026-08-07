# ADR-0061: Provider-owned Huber loss

- Status: Accepted
- Date: 2026-08-06
- Board item: `COEUS-HUBER-PROVIDER-001`

## Context

`coeus_autograd::huber_loss` accepted a generic backend but copied the full
prediction and target tensors to host storage, evaluated the classical Huber
formula in a Rust loop, uploaded a one-element result, and retained the full
difference vector as host state for backward. Accelerator execution therefore
crossed the host boundary and bypassed the selected Leto or Hephaestus
operation path.

The existing provider operation surface already supplies elementwise
subtraction, absolute value, multiplication, sign, conditional selection, and
mean reduction. Those operations are sufficient to express both Huber regions
without a new provider kernel.

## Decision

Compose Huber on the selected backend:

1. Compute `diff = pred - target` with provider binary subtraction.
2. Select the quadratic value `0.5 * diff²` or the linear value
   `delta * abs(diff) - 0.5 * delta²` with provider conditional selection.
3. Reduce the provider result to the one-element mean tensor.
4. Save `diff` as a provider-native tensor in the backward node.
5. Reuse the same provider conditional boundary for the derivative, multiply
   by the final scalar seed and mean scale, and accumulate directly into the
   provider gradient buffer.

`where_cond` now uses its fallible elementwise primitives directly, so a
provider dispatch failure propagates as the backend error rather than being
converted to a panic by an infallible convenience wrapper. CPU implementations
continue through Leto; accelerator implementations continue through their
selected Hephaestus-backed Coeus backend.

## Consequences

- Huber forward and backward no longer require CPU-addressable storage or
  input-sized host vectors.
- The public `HuberLossNode::diffs` field changes from `Vec<T>` to
  `Tensor<T, B>`, which is a major public contract change. In-repository
  callers are migrated in this increment; no compatibility adapter is added.
- The composition allocates intermediate provider tensors. This preserves
  backend ownership but is not a runtime or memory improvement claim without
  controlled measurements.
- The classical Huber mathematical contract and existing fallible validation
  remain unchanged.

## Verification

- Existing analytical CPU coverage remains the oracle for both Huber regions
  and the mean gradient.
- The focused regression also checks the boundary value and that forward and
  backward preserve the input storage values.
- Source residue inspection must find no `copy_to_host`, `Cow`, or host
  `Vec<T>` in the Huber implementation or node state.
- Locked workspace check and warning-denied workspace Clippy pass. The full
  `coeus-autograd`/`coeus-ops` native collection passes 313/313; focused Huber
  coverage passes 3/3; the provider constructor regression passes 2/2; and
  the touched package doctests previously pass 39/39.
- `cargo semver-checks` against `HEAD^` completes for patch, minor, and major
  release modes with no required update. The tool does not report the public
  generic field type change; the source-level contract remains classified as
  major because downstream struct field construction changes from `Vec<T>` to
  `Tensor<T, B>`.
- Exact-head CPU/WGPU/CUDA/ROCm/Metal provider contracts remain merge gates. The
  decomposed workspace doctest run is blocked by
  Windows Defender error 225 in unchanged doctest binaries: two `coeus-leto`
  doctests and one `coeus-ops` `exp` doctest were rejected as potentially
  unwanted software. No Defender exclusion was added.
