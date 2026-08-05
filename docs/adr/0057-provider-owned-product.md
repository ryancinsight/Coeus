# ADR 0057: Provider-owned global product

- Status: Accepted
- Date: 2026-08-05
- Board item: `COEUS-AUTOGRAD-PROD-PROVIDER-001`

## Context

`coeus_ops::prod` currently copies every input element to host memory and
folds there. The tracked autograd product node repeats that transfer and
constructs its gradient in a host vector. This violates the selected-backend
ownership boundary even though Leto and Hephaestus already provide product-axis
reductions.

## Decision

Implement one provider-resident `prod_tensor` composition by reducing each
logical input axis through the backend's `ReductionOps` implementation. This
preserves strided views without a host-side linearization. The existing
scalar-returning `prod` API reads only the final one-element result, preserving
its value contract while removing input-sized host staging. CPU dispatch
therefore reaches Leto `ProductAxis`; WGPU, CUDA, ROCm, and Metal reach their
selected Hephaestus product provider.

`ProdNode` stores the provider-resident input and one-element product tensor.
Its backward computes the exact derivative with provider equality, reduction,
fill, multiplication, and division operations. It reads only scalar product
and zero-count values to select among the zero-free, one-zero, and multi-zero
analytic cases. No input-sized host vector or CPU-addressable storage bound is
retained.

## Rejected alternatives

- Host prefix/suffix vectors: correct for zeros but violates provider ownership
  and scales host memory with input size.
- `product / input` for every element: produces invalid `0 / 0` at a single
  zero and loses the exact derivative there.
- A new provider kernel: duplicates existing axis-reduction ownership without
  a capability gap.

## Verification

The slice must prove value and gradient parity for zero-free, one-zero, and
multi-zero inputs, non-unit seeds, strided views, and copy-on-write inputs.
Host-residue scans must find no input-sized transfer or host payload in the
product operation family. Backend-specific runtime coverage is hosted where
the device is available; local CUDA is used when its provider graph is clean,
and ROCm remains CI-only.

Local evidence: format check, warning-denied Clippy for `coeus-ops` and
`coeus-autograd`, full Nextest (208/208 and 103/103), focused differential
coverage, doctests (23/23 and 16/16), and package checks pass. SemVer reports
the same three pre-existing failures from the published 0.9.0 baseline and no
product-specific failure. The WGPU package check is still
blocked by the peer Hephaestus checkout missing the cross-entropy exports
required by Coeus; locked verification is blocked by the shared local overlay
patch graph. No runtime or memory delta is claimed without controlled
measurements.
