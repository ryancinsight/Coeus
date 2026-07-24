# ADR 0012 — CUDA unfold/fold launch ABI

- Status: Accepted
- Date: 2026-07-23
- Change class: [arch]
- Driver: ATLAS-CUDA-SAFETY-012

## Context

The CUDA unfold/fold launcher kept 1-D and 2-D kernels, source text,
parameter narrowing, output-size arithmetic, and dispatch in one 467-line
module. Parameter conversion panicked on values outside the CUDA `u32` ABI;
output element products and sliding-window formulas were unchecked. Invalid
rank or shape relationships could therefore reach kernels that divide by
output widths, derive channel indices from kernel areas, and address storage
through a layout descriptor.

## Decision

Partition the operation into a manifest, shared dispatch/source leaves,
validation, and 1-D/2-D operation leaves. Load the native CUDA source from a
co-located asset without runtime allocation. The validation leaf is the
single owner of checked sliding-window formulas, rank/layout/storage bounds,
positive kernel/stride/dilation parameters, output element counts, and CUDA
`u32` conversion. It also checks the maximum physical layout offset fits the
device index ABI and rejects zero-stride output aliasing.

Each dimensional dispatcher validates its exact input/output shape relation
before compiling or launching the native kernel. The shared 1-D launch path
supplies the checked grid helper; no local narrowing or unchecked product
remains. Native kernels and device-buffer ownership remain unchanged.

## Verification

Feature-enabled package check, warning-denied Clippy, and rustdoc pass.
Default package Nextest passes 3/3 with zero skipped in 0.193 seconds.
Validation tests cover the sliding-window formula and invalid/overflowing
parameters; the feature build compiles the source/validation test modules.
CUDA-feature Nextest reaches the Windows GNU linker but cannot resolve
`-lcuda` from `/usr/local/cuda-11.3/lib64/`, so no feature test executes.

## Revisit trigger

Any new unfold/fold dimensionality, layout view, kernel parameter, or source
asset must extend the validation leaf and its shape/differential tests before
native dispatch is enabled.
