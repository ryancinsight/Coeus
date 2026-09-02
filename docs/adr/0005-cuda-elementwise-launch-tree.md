# ADR 0005: CUDA elementwise launch tree

- Status: Accepted
- Date: 2026-07-23
- Change class: [arch]
- Driver: ATLAS-CUDA-SAFETY-005

## Context

`kernels/launch_ops.rs` contained contiguous binary, contiguous unary, strided
binary, and strided unary CUDA launch families in one 530-line file. Each path
also narrowed element counts and grid dimensions directly and used raw layout
serialization for strided dispatch.

## Decision

Keep `launch_ops.rs` as a manifest and place contiguous and strided operation
families in separate leaves. All four launchers use the shared
`kernels::validation` boundary for `u32` counts and grid sizes. Strided paths
reject layouts whose rank exceeds the output rank before the generated CUDA
kernel can underflow a broadcast-rank calculation. Layout descriptors transfer
through `bytemuck::cast_slice` without an intermediate or raw slice cast.
Strided paths also reject zero-stride output layouts because the generated
kernel divides by each output stride and multiple writers would alias.

The public helper names remain unchanged; only the physical module home
changes. No dynamic dispatch, host copy, or compatibility wrapper is added.

## Verification

The manifest and leaf tree format cleanly. Feature-enabled package check and
warning-denied Clippy pass; default package Nextest passes 3/3 with zero
skipped. Shared validation tests cover grid overflow, zero work, shape-product
overflow, and zero-stride output layouts. A source audit finds no
input-dependent `as u32`, raw layout slice, unchecked elementwise grid, or
family-local kernel helper.

## Revisit trigger

Add a new elementwise policy as a leaf under `launch_ops/` only when it is a
distinct current operation family. Shared launch validation remains owned by
`kernels::validation`.
