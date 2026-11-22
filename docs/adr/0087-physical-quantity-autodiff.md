# ADR 0087: Physical quantities in Coeus automatic differentiation

- Status: Accepted
- Date: 2026-11-22
- Class: [minor] [arch]

## Context

Coeus reverse mode computes derivatives of the numerical coordinates
supplied to its graph. Aequitas stores physical values in canonical SI
and distinguishes quantities that share SI exponents but carry different
meanings. Coeus resolves Aequitas 0.5 through its existing provider cohort.

Helios's planning system unwraps typed doses at the numerical tape.
Asclepius checks and normalizes dose values in its differentiable response
and accepts raw Coeus variables. Both own their physical domain validation.

## Decision

Expose an opt-in Rust API under `coeus_autograd::physical`. Use Aequitas
quantity and unit traits at the boundary, compile-time dimensions for
supported graph operations, and existing Coeus operators for forward and
reverse evaluation. Give each call one fresh graph, one homogeneous
differentiated field and one one-element objective. Reuse Aequitas's
dimension quotient for the canonical gradient. Retain the original input
and output types so coordinate extraction can select the correct scales.
Aequitas already supplies the required laws.

See the [co-located design](../../crates/coeus-autograd/src/physical/DESIGN.md)
for the proof, accepted operations, failure behavior and verification.

## Alternatives

- Tensor-wide runtime dimensions would require propagation across views,
  broadcasting, mutation, serialization and backend operations. Dynamic
  or heterogeneous units would need a separate field-level contract.
- Typing only the final gradient leaves graph operations unchecked and
  permits re-labelling an accumulator shared by different objectives.
- Quantity-valued Coeus scalar elements cannot express an operator whose
  output type changes with the physical dimension.

## Compatibility

The additive API targets the 0.15.0 Rust CPU cohort under
[ADR 0086](0086-registry-provider-cohort.md).
Existing Coeus callers and provider APIs keep their current behavior.
Verify the published API baseline and provider identity using the
[release checklist](../../checklist.md).
Domain-specific migrations and Python bindings need their own acceptance.
