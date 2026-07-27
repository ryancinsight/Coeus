# ADR-0025: Add native comparison providers

## Status

Accepted; implementation and hosted backend-parity verification are complete.

## Decision

Route Coeus `BinaryOp::{Eq, Ne, Lt, Gt, Le, Ge}` through native Hephaestus
strided kernels in the ROCm and Metal providers. The provider operation keeps
the input scalar type and returns zero or one, matching the Coeus CPU and Leto
contracts. f32, i32, and u32 dispatch use Hephaestus's scalar-aware
`TypedBinaryExpr` expressions so WGSL, CUDA C++, and HIP C++ literals remain
well-typed.

WGPU and CUDA retain their existing direct comparison kernels in the common
Coeus operation path; their capability and Leto contracts remain in the same
CI matrix. No host fallback or vendor-local comparison kernel is added.

Each vendor backend keeps `backend.rs` as a module manifest. Provider identity,
reduction/scan dispatch, elementwise dispatch, and the public backend runtime
implementation live in separate `backend/` leaves, preserving the public
re-exports while keeping each operation family in one canonical home.

## Alternatives rejected

- Keep ROCm and Metal comparisons unsupported: rejected because the Coeus
  operation enum and Leto CPU oracle already define these operations.
- Add comparison source strings inside Coeus providers: rejected because
  Hephaestus owns device expression and strided traversal semantics.
- Route ROCm or Metal comparisons through CPU/WGPU: rejected because it would
  hide the provider capability gap and violate native backend integration.

## Verification

The Hephaestus core tests pin scalar-correct comparison expressions. Coeus
ROCm and Metal tests compare all six f32, i32, and u32 operations with
`coeus_leto::elementwise_binary_into`; the f32 cases include broadcasted
inputs. Package-scoped rustfmt and locked offline checks passed before the
final provider co-evolution. Exact-head workflow `30268824209` passed WGPU,
CUDA, ROCm, and Metal, and PR #224 merged as `84b5bccd`. The required-device
ROCm lane was skipped because the workflow was not manually dispatched. The
active local Leto path still predates `d94e3ba`/`df14311`, so a local offline
Coeus gate cannot compile there; adding local comparison adapters would
violate upstream ownership.
