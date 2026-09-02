# ADR 0002: Named optimizer ownership

- Status: Accepted
- Date: 2026-07-11
- Change class: [arch]
- Driver: RITK ADR 0004 / MIG-526

## Context

`coeus-nn::Module` exposed stable hierarchical parameters, but every optimizer
immediately flattened that inventory into `Vec<Var>`. Names therefore vanished
before update and persistence, recreating positional visitor coupling at the
RITK boundary. The `Parameter` carrier lived in `coeus-nn`, so `coeus-optim`
could not depend on it without a dependency cycle.

## Decision

`Parameter` moves to `coeus-autograd`, the deepest crate that owns `Var` and is
already shared by NN modules and optimizers. Every optimizer stores
`Vec<Parameter>` and updates `parameter.var` in place. Names survive stepping,
gradient clearing, and clipping. `Module::load_named_parameters` compares the
full expected count and ordered paths before loading values. The PyO3 boundary
accepts explicit `(name, tensor)` pairs; it does not synthesize ordinal names.

The prior `coeus-nn::Parameter` path and unnamed optimizer constructors are
deleted. A re-export or `From<Var>` fallback is rejected because either would
allow callers to bypass the stable-name invariant.

## Verification

All five optimizers retain exact names while preserving their analytical and
convergence behavior. NN integration proves updated values reload only when
the path inventory matches, and rejects reordered names with a typed error.
PyO3 optimizer execution verifies the explicit named boundary end to end.

## Revisit trigger

Revisit only if parameter groups require additional stable metadata. Extend
the canonical carrier rather than introducing optimizer-local wrappers.
