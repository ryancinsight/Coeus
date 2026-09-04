# ADR 0071: Provider-owned accelerator backends

Status: Accepted  \
Date: 2026-09-04  \
Change class: [arch] [major]  \
Board item: `COEUS-HEPHAESTUS-CUDA-FUSION-001`

## Context

Coeus had consumer-side WGPU elementwise kernels and a CUDA driver/NVRTC
facade alongside the generic Hephaestus backend. These paths duplicated
device acquisition, source generation, layout metadata, compilation caching,
memory operations, and launch responsibilities. The duplicate ownership made
backend behavior and precision mapping able to diverge between Coeus and
Hephaestus.

Atlas uses Hephaestus as Coeus's GPU backend. Eunomia owns the reduced-
precision representation contract; Coeus must not introduce a direct `half`
precision dependency.

## Decision

Coeus routes WGPU and CUDA operations through the generic Hephaestus provider
seams. Coeus retains only the consumer concerns required at that boundary:

- tensor shape and expression adaptation;
- conversion of Coeus layouts to borrowed provider views;
- provider-backed storage handles and public Coeus error mapping;
- operation marker selection where the Coeus expression contract requires it.

Hephaestus is the single owner of device acquisition, GPU source generation,
layout metadata ABI, pipeline or compilation caching, transfers, fills, bind
groups, command submission, and kernel launch. Coeus's consumer WGPU kernel
tree, CUDA driver/NVRTC facade, checked-in PTX artifact, and direct CUDA memory
operation calls are removed. No compatibility facade remains for the removed
consumer driver API.

## Alternatives rejected

- Keep local kernels beside provider dispatch: rejected because it preserves
  two backend owners and two cache/source contracts.
- Copy provider internals into Coeus: rejected because copied internals drift
  and violate upstream capability ownership.
- Route missing provider operations through CPU execution: rejected because it
  hides a provider capability gap and breaks device-resident execution.
- Add `half` beside Eunomia: rejected because it forks the Atlas precision
  representation contract.

## Invariants

- GPU data remains in provider-owned storage across operation dispatch.
- Coeus adapters use borrowed layout views and do not copy tensor storage to
  host memory for provider dispatch.
- Provider errors retain their typed source context at the Coeus boundary.
- Backend and scalar variation is represented by generic provider seams, not
  duplicated operation bodies or vendor-named public APIs.
- Coeus contains no GPU source-generation, layout-ABI, cache, driver, or
  launch implementation for WGPU or CUDA.
- Reduced-precision mapping continues through Eunomia and the provider scalar
  contracts; no direct `half` dependency is introduced.

## Verification plan and evidence

The delivery revision must run the locked Coeus WGPU, CUDA, and bridge native
suites, strict Clippy, doctests, formatting, lockfile freshness, and source
audits. The source audit must find no Coeus WGPU kernel tree, CUDA driver
facade, direct CUDA memory calls, direct WGPU dependency, or direct `half`
dependency. Hephaestus provider contract tests remain the upstream behavioral
oracle; optimized accelerator results are compared with the existing Coeus
CPU paths under their documented numerical bounds.

The implementation is delivered in Coeus PR #368 against Hephaestus PR #274,
locked to provider revision `f8811d1`.
