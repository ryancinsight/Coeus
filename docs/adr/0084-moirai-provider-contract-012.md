# ADR 0084: Advance the Moirai provider contract to 0.12.0

Status: Accepted  \
Date: 2026-11-15  \
Change class: [arch] [major]  \
Board item: [COEUS-PROVIDER-RESOLUTION-2026-09-07](../../backlog.md#coeus-provider-resolution-2026-09-07)

## Context

Moirai 0.12.0 removes the required `StateWord::get_mut` method. The
final state of a result cell is now classified through the existing
atomic load, and the concurrent publication and consumption orderings
are unchanged. Coeus implements no `StateWord` and calls no removed
method, so no Coeus source changes with the move.

The requirement does move the public contract of one Coeus item.
`coeus-core` re-exports the provider's backend type
(`pub use moirai::MoiraiBackend;`), so the re-exported type's identity
changes with the provider line. A consumer that names
`coeus_core::backend::MoiraiBackend` and also depends on Moirai 0.11
directly resolves two distinct types for the same name, and the
mixture stops compiling. This is the RFC 1977 case: a public dependency
version bump is a breaking change of the bumping crate.

The requirement also gates the rest of the stack. The providers' current
default heads require Moirai 0.12 (`apollo-fft` 0.29.5 and
`coeus-core` 0.12.0 name `moirai-runtime ^0.12.0`), so every consumer
that resolves those providers with a `^0.11.0` requirement cannot
re-resolve onto the current provider heads at all. Coeus is the
provider-side member of that chain: RITK, CFDrs, and Helios consume
`coeus-core` and `apollo-fft`, so their cohort advances follow this one.

Two alternatives were considered:

1. Stay on Moirai 0.11 and keep the current lock. The stack then keeps
   one provider revision per repository, but the provider heads that
   the consumers resolve are the ones that already require 0.12, so the
   consumers' requirement lag and the root overlay's pin drift both
   remain open.
2. Require `>=0.11.0, <0.13.0` and let the lock choose. The range admits
   both provider lines, but the lock selects the newest compatible
   revision anyway, so the re-exported type's identity moves exactly as
   it does under the selected option. The wider range only hides the
   move from the requirement.

## Decision

Move the workspace Moirai requirements to 0.12.0 (`moirai` and
`moirai-async`) and regenerate the lock outside the Atlas dependency
overlay, so every `moirai-*` row names one revision per provider
repository and the `source = "git+..."` rows survive.

No Coeus source changes with the move. The removed provider method is
unused in this repository, and the backend's parallel dispatch keeps its
existing exactly-once obligation.

## Consequences

- The `coeus_core::backend::MoiraiBackend` re-export names the Moirai
  0.12 type. A consumer that mixes Coeus with its own Moirai 0.11
  dependency moves to Moirai 0.12 in the same change; no compatibility
  re-export or forwarding type is added, because two provider identities
  under one name would keep the split the provider owns.
- The downstream consumers that resolve `coeus-core` or `apollo-fft`
  advance their own Moirai requirements in the same sweep, in dependency
  order, so one provider revision reaches the whole stack.
- The public contract change is a [major] change class; the release
  tooling assigns the version at release.
