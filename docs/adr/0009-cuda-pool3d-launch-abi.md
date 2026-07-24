# ADR 0009 — CUDA pool3d launch ABI

- Status: Accepted
- Date: 2026-07-23
- Change class: [arch]
- Driver: ATLAS-CUDA-SAFETY-009

## Context

The 3-D average and max pooling leaves retained the same unchecked parameter,
work-count, grid, and block-size conversions removed from 1-D and 2-D
dispatch. Their generated kernels index fixed-rank five-dimensional layouts.

## Decision

Apply the pool-owned validation SSOT to 3-D average and max forward/backward
dispatch with rank-five nonempty layouts, checked parameters/work/grid, shared
block size, batch/channel prefix matching, and max-backward input/state shape
matching. Preserve the native kernels and device-buffer ownership.

## Verification

Feature-enabled package check and warning-denied Clippy pass. Default package
Nextest passes 3/3 with zero skipped. A source audit finds no input-dependent
pool3d narrowing, unchecked shape product, or local grid/block derivation.

## Revisit trigger

Keep the pool validation seam as the only pooling launch boundary; new pooling
families must add their rank and shape contract before dispatch.
