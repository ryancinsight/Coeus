# ADR 0004 — Shared CUDA launch validation SSOT

- Status: Accepted
- Date: 2026-07-23
- Change class: [arch]
- Driver: ATLAS-CUDA-SAFETY-004

## Context

Convolution had the first checked CUDA launch validator, but its helper module
was nested under `launch_conv`. Reduction needed the same `u32` narrowing,
checked element count, layout-fit, and grid-size rules. Keeping a second
family-local copy would allow the CUDA ABI contract to drift.

## Decision

Move the validation helpers to `kernels::validation`, the deepest common
module for CUDA launch families. Convolution imports that shared module, and
reduction uses it for axis bounds, layout representation, output cardinality,
parameter narrowing, and grid sizing. Fused reduction rejects missing or
over-rank expression shapes and serializes its layout vector through
`bytemuck::cast_slice`.

The operation boundary continues to return `false` for invalid launch input or
unavailable dispatch. No host fallback or compatibility validator is added.

## Verification

Feature-enabled package check and warning-denied Clippy pass. Default package
Nextest passes 3/3 with zero skipped. Shared helper tests cover multiplication
overflow and grid-size overflow. A source audit finds no input-dependent
`as u32`, unchecked output product, expression-shape indexing, or panic in
`reduce.rs`.

## Revisit trigger

Every new CUDA launch family imports `kernels::validation`; a family-local
copy is a structural defect. Extend the shared helper only when the CUDA ABI
contract itself gains a new reusable rule.
