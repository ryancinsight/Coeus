# ADR 0080: Concern-owned names for the CPU and accelerator adaptation crates

Status: Proposed  \
Date: 2026-10-10  \
Change class: [arch] [major]  \
Board item: [COEUS-SIBLING-NAMED-CRATES](../../backlog.md#coeus-sibling-named-crates)

## Context

`coeus-leto` and `coeus-hephaestus` name stack members: a stack
member's name must never appear in another member's crate, module,
feature, or item names. Both crates are published (0.10.0 on
crates.io, workspace-versioned) and both are adaptation boundaries,
not thin facades:

- `coeus-leto` (19 files: `convert` + 17 `dispatch` modules) owns the
  CPU execution adaptation — converting Coeus dynamic-rank layouts
  and views onto Leto array kernels and dispatching every operation
  family (elementwise, linalg, convolution, pooling, reductions,
  rotary, sparse, stateful updates, structural ops, unfold/fold) to
  them. Nothing in the crate is Leto itself; Leto is one dependency
  in its manifest.
- `coeus-hephaestus` (57 files: per-operation
  `dispatch`/`implementation`/`provider` triplets plus storage,
  error, layout, and window modules) owns the accelerator execution
  adaptation — binding Coeus operations to Hephaestus seams
  generically over vendor providers, per ADR 0071's consumer-side
  concern list (shape/expression adaptation, borrowed provider
  views, provider-backed storage, error mapping, marker selection).

In-repo rename blast radius, measured 2026-10-10: ~332 `coeus-leto`
references (code + manifests + tests + benches) and ~61
`coeus-hephaestus` references, plus 89 documentation mentions. Three
out-of-repo manifests depend on `coeus-leto`
(`leoneuro-segmentation`, `ritk-model`, `ritk-tensor-ops`); nothing
outside Coeus depends on `coeus-hephaestus`.

## Decision

Name each crate for the concern it owns, so the manifest — not the
crate name — records which provider implements it:

- `coeus-leto` becomes `coeus-cpu`: the CPU backend implementation,
  aligned with the existing `CpuBackend`/`CpuStorage` vocabulary.
- `coeus-hephaestus` becomes `coeus-accelerator`: the generic
  accelerator backend implementation, aligned with ADR 0071's
  "provider-owned accelerator backends" language.

The rename executes as one mechanical transform over every call
site in a single commit — crate directories, manifest names and
dependency entries, `use` paths, feature references, documentation,
and the three out-of-repo manifests (coordinated as a consumer
update, since they live in other repositories). No re-export bridge
or compatibility facade remains: a `pub use old as new` alias would
reintroduce the prohibited name into the new crate's surface.

Published-name migration: the new crates publish fresh under the
workspace version in step with the next release. The old names are
not yanked — yanking breaks pinned downstream builds to no benefit.
Their final published versions carry a README and CHANGELOG note
pointing at the new names; there is no code-level deprecation
because Rust has no crate-level deprecation marker and a
`compile_error!` shell would be hostile to downstream builds. The
three out-of-repo dependents migrate their manifests to the new
names in the same release window.

## Alternatives rejected

- Keep the names: rejected because it preserves the naming-prohibition
  violation the board item files.
- Rename to provider-parametric names (e.g. `coeus-leto-backend`):
  rejected because it reintroduces member names under a suffix.
- Merge each crate into its parent (`coeus-ops`, `coeus-tensor`):
  rejected because it destroys the adaptation boundaries ADR 0071
  and the Leto rank-boundary decision deliberately established;
  provider-swap diffs would scatter across consumer crates.
- `coeus-host` / `coeus-device` instead of `coeus-cpu` /
  `coeus-accelerator`: rejected because host/device describe memory
  placement in this stack (host staging, device-resident storage),
  not execution backends; the chosen names match the backend
  vocabulary already in use.

## Consequences

The rename is a breaking change for every direct dependent and
requires the release SemVer review the board item already
prescribes. After the transform, a workspace-wide search for the
old crate names must return only historical records (this ADR,
CHANGELOG entries, and the old crates' final notes).

## Verification

The executing change proves completeness by search: zero live
`coeus-leto`/`coeus-hephaestus` references outside historical
records, the full workspace gate green (including the three
out-of-repo consumers' builds), and the new crates publishing under
their new names.
