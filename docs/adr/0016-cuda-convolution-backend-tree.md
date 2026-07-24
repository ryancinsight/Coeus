# ADR 0016 — CUDA convolution backend tree

- Status: Accepted
- Date: 2026-07-23
- Change class: [arch]
- Driver: ATLAS-CUDA-TREE-001

## Context

`coeus-cuda/src/backend/ops/conv.rs` combined 1-D, 2-D, and 3-D forward and
backward dispatch with transposed-convolution dispatch in one 614-line file.
The shared file crossed operation families and exceeded the repository's
500-line leaf target, increasing review and ownership coupling.

## Decision

Make `backend/ops/conv/` the canonical convolution dispatch home. Keep the
manifest and shared layout helpers in `mod.rs`; place forward, backward, and
transposed-convolution implementations in separate leaf modules. Preserve the
existing `CudaBackend` methods and `ops.rs` module boundary, so callers and
provider ownership do not change. No forwarding or compatibility module is
retained.

## Verification

The four convolution leaves are 36, 186, 236, and 181 lines. Feature package
check, warning-denied Clippy, feature rustdoc, and default package Nextest pass
3/3 with zero skipped in 0.054 seconds. CUDA-feature Nextest cannot execute
because the Windows GNU linker cannot resolve `-lcuda` from
`/usr/local/cuda-11.3/lib64/`.

## Revisit trigger

Any new convolution dimensionality or backend execution regime must land under
the canonical operation-family leaf, split again when a leaf crosses 500 lines
or contains multiple bounded contexts.
