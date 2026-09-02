# ADR 0008: CUDA pool2d launch ABI

- Status: Accepted
- Date: 2026-07-23
- Change class: [arch]
- Driver: ATLAS-CUDA-SAFETY-008

## Context

The 2-D average and max pooling leaves independently narrowed pooling
parameters, output/input work counts, and launch dimensions. Their generated
kernels index fixed-rank layout fields and divide by stride, but dispatch did
not validate those contracts. The 1-D dispatcher carried parallel validation
logic.

## Decision

Create `pool/validation.rs` as the pooling-boundary SSOT and migrate 1-D
dispatch to it. Apply the same checked parameter, work-count/grid,
rank/nonempty-layout, batch/channel-prefix, and backward-shape validation to
2-D average and max forward/backward dispatch. Reuse the shared CUDA block
size through the pooling seam. Preserve native kernels, device pointers, and
allocation behavior.

## Verification

Feature-enabled package check and warning-denied Clippy pass. Default package
Nextest passes 3/3 with zero skipped. A source audit finds no input-dependent
pool2d narrowing, unchecked shape product, or local grid/block derivation.

## Revisit trigger

Apply the same pooling seam to 3-D dispatch only after its five-dimensional
layout and parameter contracts are verified explicitly.
