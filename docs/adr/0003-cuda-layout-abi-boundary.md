# ADR 0003: Checked CUDA layout ABI boundary

- Status: Accepted
- Date: 2026-07-23
- Change class: [major] [arch]
- Driver: ATLAS-CUDA-SAFETY-003

## Context

`GpuLayoutInfo` is the single host-to-device representation for CUDA tensor
layouts. Its conversion previously used truncating `usize` to `u32` casts and
an input-dependent rank assertion. Forward convolution callers also computed
output element counts with an unchecked product before reaching the launch
validator. The descriptor and conversion helper were public even though they
are CUDA implementation details.

## Decision

Make the descriptor and serializer crate-private. Implement
`TryFrom<&Layout>` as the only conversion path and reject rank mismatch, rank
above eight, and any offset, shape, stride, or rank value outside the CUDA
`u32` ABI. Serialize the `Pod` descriptor through `bytemuck::cast_slice`, so
the transfer remains allocation-free and does not require a raw slice cast.
Every CUDA layout consumer returns its existing `false` dispatch result when
conversion fails. Forward 1D, 2D, and 3D convolution compute output element
counts with one checked multiplication seam before dispatch.

The former public `GpuLayoutInfo::from_layout` and
`create_layout_buffer` surfaces are removed rather than retained as wrappers;
the crate-private seam is the single source of truth.

## Verification

Feature-enabled package check and warning-denied Clippy pass. Boundary tests
cover representable values, unsupported rank, shape/stride rank mismatch, and
an offset above `u32::MAX`. The default package Nextest suite passes 3/3 in
0.053 seconds. CUDA-feature Nextest cannot link in the current Windows GNU
environment because `-lcuda` is absent from `/usr/local/cuda-11.3/lib64/`.
`cargo semver-checks -p coeus-cuda --baseline-rev HEAD` reports the two
intentional removed public items and classifies the change as major.

## Revisit trigger

Add a new descriptor field only with a matching CUDA kernel ABI change and a
layout conformance test. Do not add another host-to-device layout converter.
