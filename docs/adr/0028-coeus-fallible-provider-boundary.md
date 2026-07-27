# ADR-0028: Make Coeus provider acquisition and transfers fallible

- Status: accepted
- Date: 2026-07-27
- Scope: `coeus-core` backend/storage contracts and the Hephaestus-backed
  Coeus provider implementations

## Context

`ComputeBackend` already exposes an associated error for fallible operation
dispatch, but its allocation, fill, host-to-device, and device-to-host methods
return no error. `HephaestusBackend`, `HephaestusStorage`, and the ROCm/Metal
provider leaves therefore convert device acquisition, allocation, transfers,
and copy-on-write uniqueness failures into `expect` panics. The failure is at
the provider boundary: a missing accelerator or rejected transfer must reach
the caller as a typed backend error, not be hidden by a CPU path or a default
buffer.

The workspace also declared first-party providers as sibling path dependencies
while the Atlas development overlay patched the same Git sources to `repos/*`.
That created two local package identities in a worktree, with the collision
surfacing before Coeus compilation. The member manifest must remain standalone;
the stack overlay is the only local-resolution mechanism.

## Decision

Make the deepest common backend and storage mutation boundaries fallible. The
allocation, fill, host-to-device, device-to-host, and storage uniqueness
operations return the backend's associated error; provider initialization uses
a cached `Result` so the first typed acquisition failure is retained and
returned on later calls. `HephaestusBackendError` preserves the original
Hephaestus error category and adds only the Coeus operation context.

First-party provider dependencies remain Git-plus-version declarations in the
member manifest. The committed lockfile is regenerated from that standalone
graph; local Atlas development may apply the generated stack overlay, but no
member-local `[patch]` table or path dependency is retained.

Migrate implementors and callers in dependency order: core traits and CPU
implementors, Hephaestus storage/provider seams, accelerator backend
implementors, tensor construction/materialization, then operation and binding
callers. The generic backend remains monomorphized. No `dyn` boundary,
compatibility wrapper, silent CPU fallback, default buffer, or swallowed
`Result` is introduced.

## Rejected alternatives

- Retain infallible methods and replace `expect` with a default or empty
  buffer: masks device failure and violates value-semantic correctness.
- Add a parallel `try_*` API while retaining the infallible path: leaves two
  sources of truth and permits callers to keep the panic path.
- Select CPU execution when a selected provider fails: changes backend
  semantics and hides an unavailable or faulty device.

## Failure modes and verification

The migration must preserve distinct acquisition, allocation, length, transfer,
and COW uniqueness errors. Tests cover unavailable providers, allocation and
length rejection, transfer failure, and shared-storage uniqueness failure using
value-semantic error matching. Production panic scans cover every migrated
provider and caller. The exact delivered revision runs warning-denied package
checks, sanctioned Nextest tests, doctests, and the WGPU/CUDA/ROCm/Metal
provider matrix; unavailable hardware is reported as a skipped hardware lane,
not as a passing device test.
