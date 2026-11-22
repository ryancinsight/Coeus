# ADR 0088: Extend external reference qualification for the Atlas stable stack

- Status: Proposed
- Date: 2026-11-22
- Class: [arch]

## Context

[ADR 0025](0025-external-framework-parity-without-burn.md) retains PyTorch, JAX and MLX correctness suites and compares performance between Coeus backends. [ADR 0083](0083-release-scope.md) defines the local tensor milestone. Atlas DEC-002 requires isolated external correctness and performance oracles for the frozen NumPy, JAX, PyTorch and TensorFlow surfaces. Local milestone closure and the proposed next-major [ADR 0085](0085-release-scope.md) do not settle that evidence contract.

## Proposed decision

Retain the existing comparators and native analytical tests. For each Coeus-owned mapped API, bind the pinned framework identity and version, operation, dtype, shape, layout, device, gradient order, domain and rejection behavior to a reproducible reference program. Run external frameworks only in reference or benchmark environments. Production execution stays on the existing native provider graph.

Begin with a tensor operation whose source and provider mapping the Atlas owner has reviewed. Capture inputs and reference outputs, then execute the production Coeus path on the named backend. Compare exact discrete results, error classes and analytically bounded floating-point outputs. For differentiable operations, add an independent derivative or adjoint check. For performance, match shapes, precision, target, synchronization and data movement and report uncertainty. Record the exact candidate, reference environment and raw measurements.

## Alternatives and approval

Using existing local test counts as coverage omits supported axes and TensorFlow. Adding a framework runtime as a production fallback changes the implementation being qualified. Keep unavailable targets and unknown publicness open in the Atlas register. Review this extension with the parity and Coeus owners before modifying required CI or claiming release credit. Existing local and next-major scope decisions retain their own approval path.
