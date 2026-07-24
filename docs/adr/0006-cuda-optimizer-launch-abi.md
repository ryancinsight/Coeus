# ADR 0006 — CUDA optimizer launch ABI

- Status: Accepted
- Date: 2026-07-23
- Change class: [arch]
- Driver: ATLAS-CUDA-SAFETY-006

## Context

The five CUDA optimizer leaves derived parameter element counts with
`Layout::numel`, narrowed counts and grids with unchecked casts, and repeated
the block-size literal. Strided optimizer kernels also assumed that gradient
and state layouts had the parameter shape and rank.

## Decision

Validate the parameter element count, CUDA count, launch grid, layout ABI, and
same-shape contract once at each optimizer boundary through
`kernels::validation`. Reuse the shared block-size constant in every
contiguous and strided launch. Reject an Adam or AdamW step count that cannot
represent the kernel's `i32` exponent contract. Preserve the existing native
CUDA kernels and layout views; validation adds no hot-loop allocation or copy.

## Verification

Feature-enabled package check and warning-denied Clippy pass. Default package
Nextest passes 3/3 with zero skipped. Shared validation tests cover shape
mismatch in addition to count, grid, product-overflow, zero-work, and
zero-stride output cases. A source audit finds no input-dependent narrowing in
the optimizer leaves.

## Revisit trigger

Any new optimizer launch family must use the shared validation seam and must
prove its layout/indexing contract before dispatch.
