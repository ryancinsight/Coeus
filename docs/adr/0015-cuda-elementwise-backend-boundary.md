# ADR 0015: CUDA elementwise backend boundary

- Status: Accepted
- Date: 2026-07-23
- Change class: [patch]
- Driver: ATLAS-CUDA-SAFETY-015

## Context

CUDA unary and binary backend dispatch computed output work with an unchecked
iterator product before reaching the kernel validation seam. The Hephaestus
contiguous and strided adapters also converted provider errors to panics,
preventing the existing explicit CPU capability path from recovering.

## Decision

Compute output work through the shared checked `kernels::checked_numel`
contract. If the product is not representable, use the existing CPU backend
path. Treat Hephaestus provider errors as a failed acceleration attempt and
continue through the same explicit fallback path; successful results still
replace the destination storage without copying.

## Verification

Feature package check and warning-denied Clippy pass. Default package Nextest
passes 3/3 with zero skipped in 0.114 seconds; feature rustdoc passes in 3.55
seconds. CUDA-feature Nextest cannot execute because the Windows GNU linker
cannot resolve `-lcuda` from `/usr/local/cuda-11.3/lib64/`.

## Revisit trigger

Any change to elementwise storage ownership, provider dispatch errors, or
output-shape construction must preserve checked work-count derivation and the
non-panicking fallback contract.
