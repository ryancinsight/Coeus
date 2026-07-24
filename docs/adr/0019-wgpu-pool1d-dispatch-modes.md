# ADR-0019: Type-state WGPU pool1d dispatch modes

- Status: accepted
- Date: 2026-07-23
- Scope: `crates/coeus-wgpu/src/kernels/pool/pool1d/`

## Context

The pool1d forward dispatcher accepted a `PoolKind` containing both forward
and backward variants, then rejected the backward variants with
`unreachable!`. The call sites were already statically separated, so the
enum encoded more states than the dispatcher could handle.

## Decision

Introduce a forward-only mode type at the forward dispatcher boundary and
convert it to the existing shader-source mode only when generating WGSL.
The shader body remains single-sourced. The backward dispatchers retain the
existing backward modes.

## Consequences

The invalid forward/backward call-state is rejected by the type checker, the
forward dispatcher no longer carries an input-dependent `unreachable!`, and
the generated shader source and public launch functions remain unchanged.
The separate WGPU layout metadata conversion still narrows `usize` values to
the WGSL `u32` ABI at 23 sites. Correctly surfacing those failures requires a
typed error through the currently infallible backend-operation traits; that
larger API migration is not folded into this patch.

## Verification

Run format and diff checks, scan the pool1d subtree for unreachable or
placeholder paths, and run the package gate when the preserved peer manifest
resolves. No runtime performance claim follows from this type-state change.
