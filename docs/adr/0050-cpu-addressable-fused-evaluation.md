# ADR-0050: Restrict CPU fusion to CPU-addressable backends

- Status: Accepted
- Date: 2026-08-03
- Scope: Coeus fused expression traits and CPU evaluators
- Change class: `[major] [arch]`
- Board item: `COEUS-FUSED-CPU-BOUNDARY-001`

## Context

The public CPU fused evaluators accept every `Backend`. Device-resident inputs
are downloaded into a thread-local, type-erased cache, and device outputs are
computed in a host `Vec` before upload. Backend selection therefore does not
select the execution provider, and the CPU evaluator silently crosses device
placement boundaries.

CPU evaluation methods also live on the device-neutral expression trait used
to emit accelerator shader expressions. That combines two execution roles and
forces device expression leaves to carry host fallback behavior.

## Decision

Keep `ExprNode` as the device-neutral expression-shape and shader contract.
Move host scalar evaluation into a `CpuExprNode` subtrait implemented only for
expressions whose backend implements `CpuBackend`. CPU evaluators require both
traits and borrow input and output storage through the CPU backend contract.

Delete the thread-local cache, device downloads, host output staging, runtime
type erasure, and upload branches. Accelerator fusion remains owned by each
selected provider and cannot call the CPU evaluator because its expression
leaves do not implement `CpuExprNode`.

## Alternatives rejected

- Retain runtime storage probing: rejected because it preserves silent device
  fallback and makes placement a runtime accident.
- Add a Hephaestus adapter to the CPU evaluator: rejected because accelerator
  providers already own their fused execution paths.
- Duplicate expression trees by backend: rejected because operation structure
  remains one generic `ExprNode` hierarchy with an interface-segregated CPU
  evaluation capability.

## Verification

Compile-time bounds reject accelerator expressions at the CPU evaluator.
Generic Sequential and Moirai contracts cover values, broadcasting,
high-rank coordinates, empty axes, and reductions. Residue scans must find no
cache, type erasure, device transfer, or host staging in CPU fusion. Warning-
denied Clippy, focused Nextest, doctests, semver classification, and exact-head
provider CI gate delivery.

These checks establish ownership and value semantics. They do not establish a
runtime or memory improvement without controlled measurements.

## Revisit trigger

Revisit only if a new CPU backend cannot expose addressable storage through the
canonical `CpuBackend` contract.
