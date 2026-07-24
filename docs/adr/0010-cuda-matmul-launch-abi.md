# ADR 0010 — CUDA matmul launch ABI

- Status: Accepted
- Date: 2026-07-23
- Change class: [arch]
- Driver: ATLAS-CUDA-SAFETY-010

## Context

The tiled CUDA matmul launcher indexed the first two dimensions without a
rank contract and narrowed both 16-wide grid axes directly. Incompatible
matrix shapes could reach the kernel with invalid indexing assumptions.

## Decision

Validate representable layout metadata, exact rank-two nonempty shapes,
`A.cols == B.rows`, and output shape compatibility before compiling or
dispatching the native 16×16 tiled kernel. Compute both grid axes through the
shared checked arbitrary-block grid helper. Keep the f32 PTX kernel, layout
views, and device-buffer ownership unchanged.

## Verification

Feature-enabled package check and warning-denied Clippy pass. Default package
Nextest passes 3/3 with zero skipped. Shared validation tests cover custom
block widths, zero work, and zero block sizes. A source audit finds no
input-dependent matmul grid narrowing or unchecked rank indexing.

## Revisit trigger

Any additional matmul layout or batching mode must define its shape contract
before reusing the tiled launcher.
