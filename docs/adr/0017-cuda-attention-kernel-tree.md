# ADR 0017 — CUDA attention kernel tree

- Status: Accepted
- Date: 2026-07-23
- Change class: [arch]
- Driver: ATLAS-CUDA-TREE-002

## Context

`crates/coeus-cuda/src/kernels/attention.rs` combined validation metadata, three
embedded NVRTC sources, forward dispatch, backward dispatch, and boundary tests
in one 567-line file. The mixed concerns crossed ownership boundaries and
exceeded the repository's 500-line leaf target.

## Decision

Make `kernels/attention/` the canonical attention-kernel home. Keep the module
manifest and curated re-exports in `mod.rs`; place dimension validation,
embedded device sources, forward dispatch, backward dispatch, and tests in
separate leaves. Preserve the existing public launch functions and explicit
CPU capability boundary. Remove the flat module rather than retaining an alias
or forwarding adapter.

## Verification

The six attention leaves are 12, 81, 92, 101, 135, and 149 lines. Format and
diff checks pass. Package compile and test gates are blocked by unrelated dirty
`Cargo.toml` state: the manifest requests `mnemosyne ^0.6.0` while locked
Moirai requires `mnemosyne ^0.5.0`. No compiled or test result is claimed for
this slice.

## Revisit trigger

Any new attention execution regime or mask/layout family must land under the
canonical operation-family tree, with another leaf split when a leaf crosses
500 lines or contains multiple bounded contexts.
