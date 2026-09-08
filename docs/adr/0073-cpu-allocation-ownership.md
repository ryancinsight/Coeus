# ADR 0073: Private CPU allocation ownership

Status: Accepted  \
Date: 2026-09-08  \
Change class: [major]  \
Board item: [COEUS-CPU-STORAGE-OWNERSHIP](../backlog.md#coeus-cpu-storage-ownership)

## Context

At `3cf2d670`, `CpuStorage::into_raw` returns `RawBlock`. Although its module
is private, callers infer the returned public type and can write its public
pointer and allocation layout. Its destructor passes both values to Mnemosyne
without further validation. Replacing the pointer with another live storage
allocation's pointer causes a premature free using only safe Rust.

The compile-fail ownership example unexpectedly compiles on that revision.
Rustdoc reports failure without executing the example. A source search across
Atlas repositories and worktrees finds no Coeus caller of `into_raw`.

## Decision

Remove the unused raw ownership export. Keep the allocation block, its pointer
and its layout private. Construction records the allocator's exact pointer and
layout; only destruction deallocates it. `CpuStorage` retains typed initialized
access, reference-counted cloning and copy-on-write mutation.

Keeping an opaque exported block would add a public ownership concept with no
present consumer. An unsafe raw ownership transfer also has no present caller
or interoperability requirement. Neither is retained as a replacement API.

## Migration

`CpuStorage::into_raw` is removed. Keep the `CpuStorage` owner alive and use its
borrowed typed access for memory access. Clone storage to share its lifetime;
mutation detaches shared allocations. No in-stack call site needs migration.
Any future ownership-transfer requirement must specify its allocator, element
initialization, lifetime and deallocation obligations before adding an API.

## Verification and limits

The compile-fail doctest prevents the original safe ownership escape. Existing
initialized-storage and copy-on-write tests, plus clone/drop/mutate coverage,
check values and lifetime independence under the committed Nextest budget.
SemVer checks classify the removed public method; an independent safety review
checks the allocation's construction, access and destruction closure.

Compilation prevents this public escape; native tests do not prove the
allocator's unsafe implementation. This change does not claim recoverable
allocation failure or close the separate fallible-storage migration.

Five storage tests pass under Miri run `42c1cabd-0b5c-4bd9-a656-a4b629bba898`.
Mnemosyne's Windows backing implementation substitutes provenance-tracked
`System` allocations under `cfg(miri)`. This exercises Coeus's pointer access
and reference-counted lifetimes, not Windows virtual-memory calls. The native
storage baseline passes four tests at `3cf2d670`; the candidate adds the
clone/drop/mutate case. Logs reside in `test_output/storage/`.
