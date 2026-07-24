# ADR 0011 — CUDA attention launch ABI

- Status: Accepted
- Date: 2026-07-23
- Change class: [arch]
- Driver: ATLAS-CUDA-SAFETY-011

## Context

The NVRTC attention launchers narrowed sequence, feature, work-count, mask,
and head dimensions directly to the CUDA `u32` ABI. Backward attention also
multiplied the transient score-buffer size without checked arithmetic. The
kernel assumes contiguous rank-three tensors and can divide by sequence or
head dimensions, so invalid layouts or zero dimensions must not reach native
dispatch.

## Decision

Use one attention-owned checked dimension record for forward and backward
launches. It validates positive dimensions, checked element counts, the CUDA
`u32` representation, mask-rank/head relationships, and every device-buffer
length before kernel compilation or transient allocation. Route the shared
1-D launcher through the canonical checked grid helper. The operation
boundary dispatches only contiguous offset-zero rank-three tensors with
compatible shapes and contiguous rank-one/rank-two key-padding masks; other
layouts remain on the verified CPU capability path.

The native NVRTC kernels, f32 ABI, and device-buffer ownership remain
unchanged. This is a launch-safety and dispatch-contract change; it makes no
performance claim.

## Verification

Pure attention-boundary tests cover valid rank-two masks, zero dimensions,
mask-rank inconsistency, non-divisible mask heads, and product overflow.
Feature-enabled package check and warning-denied Clippy pass. Default package
Nextest passes 3/3 with zero skipped in 0.171 seconds; default doctests pass
4/4 in 14.21 seconds. CUDA-feature Nextest is an external environment gate:
the Windows GNU linker cannot resolve `-lcuda` from
`/usr/local/cuda-11.3/lib64/`, so no feature test executes.

## Revisit trigger

Any new attention batching, mask rank, non-contiguous view, or backend kernel
must extend the checked contract and its value-semantic differential tests
before native dispatch is enabled.
