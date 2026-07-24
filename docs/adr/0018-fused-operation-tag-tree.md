# ADR 0018 — Fused operation-tag tree

- Status: Accepted
- Date: 2026-07-23
- Change class: [arch]
- Driver: ATLAS-CUDA-TREE-003

## Context

`coeus-ops/src/fuse/op_tags/mod.rs` combined the public operation-tag manifest,
the `UnaryOpTag` trait, and 30 unary tag implementations in one 625-line
module. Binary tags, runtime-parameterized leaky-ReLU tags, and WGSL helpers
already had separate canonical leaves.

## Decision

Make `fuse/op_tags/` a manifest and move the unary seam under
`fuse/op_tags/unary/`. Keep the trait in `unary/mod.rs`; place tags in
elementary, transcendental, and activation leaves. Preserve the existing
public tag names and monomorphized `UnaryOpTag<T>` dispatch through curated
re-exports. Do not add adapters or duplicate tag bodies.

## Verification

The operation-tag manifest is 9 lines; unary leaves are 27, 125, 180, and 294
lines. Format and diff checks pass. Package compile and test gates are blocked
by unrelated dirty provider dependency state; no compiled or test result is
claimed for this structural increment.

## Revisit trigger

New tag families land under their canonical unary leaf or a new bounded leaf;
split again when a leaf crosses 500 lines or combines distinct operation
families.
