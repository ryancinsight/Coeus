# ADR 0014: CUDA fused-dispatch launch ABI

- Status: Accepted
- Date: 2026-07-23
- Change class: [arch]
- Driver: ATLAS-CUDA-SAFETY-014

## Context

Fused CUDA dispatch narrowed `Layout::numel()` directly to `u32`, derived a
local grid with unchecked arithmetic, and launched against layout descriptors
without proving that logical offsets fit the backing device allocations. The
generated indexing also assumed compatible broadcast ranks and nonzero output
strides. These assumptions could produce invalid device reads or writes.

## Decision

Validate the checked output count and canonical grid before source generation.
Require a contiguous offset-zero output with sufficient storage, broadcast
compatibility for every input, representable physical input offsets, and
storage capacities covering every reachable layout offset. Reject null input
pointers before dereference and use fallible host-vector reservation. Reuse a
shared physical-storage bound helper across fused and unfold/fold dispatch.
Serialize the already initialized `GpuLayoutInfo` vector as a zero-copy POD
view with an explicit safety proof, and use the canonical CUDA block constant.

## Verification

Feature package check and warning-denied Clippy pass. Default package Nextest
passes 3/3 with zero skipped in 0.055 seconds; feature rustdoc passes in 3.09
seconds. CUDA-feature Nextest cannot execute because the Windows GNU linker
cannot resolve `-lcuda` from `/usr/local/cuda-11.3/lib64/`.

## Revisit trigger

Any change to fused expression ownership, broadcast semantics, layout views,
device storage allocation, or generated indexing must update the validation
contract and its independent boundary tests before dynamic dispatch changes.
