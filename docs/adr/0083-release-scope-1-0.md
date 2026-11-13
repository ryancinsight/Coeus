# ADR 0083: Coeus release scope for 1.0

## Status

Accepted (2026-11-12)

## Context

Coeus (`coeus-*`, workspace crates 0.12.0) is the Atlas stack's
differentiation substrate: tensors, ops, autograd, nn, optimizers, vendor
backends, and Python bindings over provider kernels (Eunomia, Leto,
Hephaestus, Mnemosyne, Moirai). Twenty-four open items remain (typed-error
propagation, provider-norm hosted evidence, gradcheck coverage,
backend-unification residuals, registry/Python release mechanics) and no
milestone ladder exists to 1.0.

## Decision

Adopt a three-milestone ladder to 1.0 recorded in `backlog/milestones.md`:

- **M-core-correctness** (0.13.0): typed errors, allocation budget, and the
  core tensor/backend surface verify on the provider cohort.
- **M-autograd-verify** (0.14.0): autograd, gradcheck, and nn verification
  evidence complete; optimizer and tightening work lands.
- **M-ecosystem** (1.0.0): stable release with registry publication and
  Python wheels (authority-gated).

In scope: tensor/ops surface, autograd/nn/optim verification, vendor
backend bindings, Python bindings. Out of scope: new model families or
Python API additions beyond comparison dunders; any `publish = true` or
PyPI upload without release authority (veto point: M-ecosystem never
forces publication).

## Alternatives

- Plan 1.0 as a single release: rejected because provider-norm evidence
  and gradcheck coverage remain open.
- Keep roadmap implicit in backlog items: rejected because stack policy
  requires a runnable capability-oriented milestone ladder.

## Consequences

- Open items carry `milestone:` assignments and name the exit check they serve.
- Milestones close only when exit checks pass on fetched default commits; 1.0
  release/publication remains authority-gated.
- This ADR is rewritten in place as scope evidence changes.
