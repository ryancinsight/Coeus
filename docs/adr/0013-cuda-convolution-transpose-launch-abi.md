# ADR 0013: CUDA transposed-convolution launch ABI

- Status: Accepted
- Date: 2026-07-23
- Change class: [arch]
- Driver: ATLAS-CUDA-SAFETY-013

## Context

The CUDA transposed-convolution launchers narrowed all spatial, channel,
kernel, stride, padding, dilation, and output-work dimensions directly to
`u32`. Their input, weight, bias, and output element products were also
unchecked before native compilation and dispatch. A large or malformed
buffer contract could therefore wrap the ABI or let the kernel address past
the supplied storage.

## Decision

Add operation-owned checked launch records for 1-D and 2-D transposed
convolution. Validate positive non-padding dimensions, checked input/weight/
output products, optional bias capacity, representable `u32` ABI values, and
the shared 1-D grid contract before compiling the NVRTC kernel. The backend
selects the native path only for rank-correct, contiguous, offset-zero layouts
with matching batch and channel contracts. Device-side gather arithmetic uses
64-bit unsigned intermediates for dilation products and subtraction, avoiding
32-bit wrap before bounds checks. Preserve the native gather kernels, f32 ABI,
and device-buffer ownership.

The output dimensions remain supplied by the existing operation boundary,
which owns the public output-shape formula and output-padding contract. The
launch boundary verifies the resulting output work and storage capacity; it
does not duplicate that public formula or introduce a second shape SSOT.

## Verification

Feature-enabled package check and warning-denied Clippy pass. Pure product
tests cover representable and overflowing work sizes. Default package
Nextest, feature rustdoc, and the CUDA-feature linker result are recorded in
the checklist. CUDA-feature Nextest remains blocked when the Windows GNU
linker cannot resolve `-lcuda` from `/usr/local/cuda-11.3/lib64/`.

## Revisit trigger

Any transposed-convolution dimensionality, output-padding formula, layout
view, or bias representation change must update the launch record and its
independent shape/product tests before native dispatch changes.
