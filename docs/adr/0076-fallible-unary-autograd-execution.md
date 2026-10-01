# ADR 0076: Fallible unary autograd execution

Status: Accepted  \
Date: 2026-09-27  \
Change class: [major] [arch]  \
Delivery: [PR #461](https://github.com/ryancinsight/Coeus/pull/461)
Revision 2026-09-29: implemented the signature and caller migration with fallible storage.

## Context

`coeus-ops`'s elementwise unary functions (`sin`, `cos`, `exp`, `log`, `neg`,
`abs`, `sqrt`, `recip`, `sign`, `floor`, `ceil`, `round`, `trunc`, and the
twelve identified in the original probe — `erf`,
`erfc`, `tan`, `asin`, `acos`, `atan`, `log2`, `log10`, `exp2`, `atanh`,
`asinh`, `acosh`) all discard the backend's `Result` via
`elementwise_unary(...).expect("<op>")`, turning any device/allocator failure
into an unconditional panic. The original backlog audit proposed widening
these signatures to `Result<Tensor<T, B>, B::Error>`.

An attempt to widen the second twelve (unpushed, since reverted) proved that
none of `coeus-ops`'s unary functions is actually a leaf a caller invokes
directly outside `coeus-ops`'s own tests. Every one of them is wired into
`coeus-autograd`'s activation machinery as the `$fwd` of a
`unary_autograd!(Op, "name", fwd, back)` invocation (36 in
`crates/coeus-autograd/src/ops/activation/{gelu,math,relu,sigmoid,silu,tanh_act,trig}.rs`)
or a hand-written `impl UnaryAutogradOp` (`ext.rs`, `math.rs`, `relu.rs`,
`linalg.rs`; ~44 implementors total). The macro's generated
`UnaryAutogradOp::forward`/`backward`
(`ops/activation/mod.rs:184`/`189`) call `coeus_ops::$fwd`/derivative
closures and return `Tensor<T, B>` directly — the trait itself declares both
methods infallible (`ops/activation/mod.rs:14`, `:20`). `unary_op`
(`mod.rs:78`) calls `Op::forward` eagerly to build the `Var`'s tensor, and the
public wrapper each macro invocation emits (e.g. `coeus_autograd::sin`)
returns `Var<T, B>`, not `Result<Var<T, B>, _>`. Widening a `coeus-ops`
function's return type without widening this layer only moves the `.expect()`
into the macro's generated `forward` — no panic removed, confirmed by trying
it: `cargo check -p coeus-autograd --tests` fails at exactly the 12 call sites
converted, with `E0308: expected Tensor<T,B>, found Result<Tensor<T,B>, _>`.

The lower graph-execution layer is already fallible: `BackwardNode::backward`
(`node.rs:84`) returns `Result<(), B::Error>`, and `Var::backward`
(`var.rs:243`) propagates it. `UnaryNode`'s `BackwardNode` impl
(`mod.rs:59`-`72`) already calls `coeus_ops::add_assign(..)?` inside a
`Result`-returning method — only `Op::backward`'s own return (the derivative
tensor before it is added into the gradient buffer) is the infallible link in
that chain. So the gap is exactly two trait methods and the wrapper functions
built on them, not the graph engine itself.

`COEUS-FALLIBLE-TENSOR-STORAGE` made `Tensor` construction and copy-on-write
fallible in the same delivery. `UnaryAutogradOp::forward` constructs a new
tensor on every call, so the unified cutover changed each affected signature
once.

## Decision

1. `COEUS-FALLIBLE-TENSOR-STORAGE` and the unary caller closure land together.
   `UnaryAutogradOp::forward`/`backward` return
   `Result<Tensor<T, B>, B::Error>` in the same migration that makes `Tensor`
   construction fallible, so each affected signature changes exactly once.
2. `unary_autograd!`'s generated `forward`/`backward`
   (`ops/activation/mod.rs:184`-`196`) propagate with `?`; `unary_backward`
   (`mod.rs:109`) and every `$back:expr` closure gain the same return type.
   `UnaryNode::backward` (`mod.rs:59`) already returns `Result`, so its own
   change is `let mask = Op::backward(..)?;` — no cascade beyond this file.
3. `unary_op` (`mod.rs:78`) returns `Result<Var<T, B>, B::Error>`; the eager
   `Op::forward` call becomes `Op::forward(&a.tensor, &backend)?`.
4. Every macro-emitted and hand-written public wrapper (`coeus_autograd::sin`,
   `::relu`, `::gelu`, …, ~44 in total) returns `Result<Var<T, B>, B::Error>`.
   Composite ops built from these (e.g. `mish` from `softplus`+`tanh`, losses
   built from `log`/`exp`) propagate with `?`, cascading through
   `coeus-autograd`'s own composite activation and loss modules.
5. `coeus-nn` and `coeus-python` callers of these wrappers (confirmed direct
   callers: `coeus-python/src/ops/elementwise.rs`, `pytensor.rs`;
   `coeus-nn/src/{loss.rs,activation/basic.rs}`) propagate with `?`; Python
   bindings map the backend error to the existing exception-mapping path
   `COEUS-FALLIBLE-UNARY-EXECUTION`'s acceptance criterion already names.
6. `coeus-ops`'s own unary functions widen to `Result` as part of step 1's
   same-cycle signature change, not before it — a `coeus-ops`-only widening
   with no consumer able to receive the `Result` is exactly the mistake this
   ADR documents.

Non-differentiable direct-`Tensor` callers with no `Var`/autograd involvement
(if any survive an updated caller search once storage fallibility lands)
propagate with `?` directly against the widened `coeus-ops` signature; they do
not need this ADR's `UnaryAutogradOp` migration.

## Rejected alternatives

- **Widen `coeus-ops` only, `.expect()` at the macro boundary.** Falsified by
  an unpushed, since-reverted attempt: the panic relocates one file up and the
  workspace still panics on the identical failure; `COEUS-EXPECT-SWALLOWED-
  RESULTS`'s stated goal (no input-dependent panics) is not met.
- **Land unary fallibility before tensor storage fallibility.** Every
  `UnaryAutogradOp::forward` implementor constructs its output tensor; a
  storage-fallible follow-up would touch the same ~44 signatures a second
  time. Rejected on the same one-signature-change-per-callsite basis
  `COEUS-FALLIBLE-TENSOR-STORAGE`'s own board entry already applies to
  unary/index-reduction.
- **A parallel fallible API (`try_sin` beside `sin`) to avoid breaking
  callers.** Rejected by `integrity`: compatibility soup — two entry points
  for one operation is exactly the split-source-of-truth `ADR 0045` (fallible
  module forward) also rejected for the same reason.

## Migration

PR #461 migrates `COEUS-FALLIBLE-TENSOR-STORAGE` and these unary APIs in one
caller closure. The twelve functions from the original probe (`erf`, `erfc`,
`tan`, `asin`, `acos`, `atan`,
`log2`, `log10`, `exp2`, `atanh`, `asinh`, `acosh`) and the remaining thirteen
(`sin`, `cos`, `exp`, `log`, `neg`, `abs`, `sqrt`, `recip`, `sign`, `floor`,
`ceil`, `round`, `trunc`) convert together, since both sets now share one
`Result`-returning path with no infallible fallback to choose between them.
Each `coeus_autograd::{op}` call site propagates with `?` or maps the typed
backend error at its public boundary, including `coeus-nn` and `coeus-python`.

## Verification and limits

Positive: a forced backend allocation failure (test double or an
over-budget device allocation) on a unary op used inside a tracked `Var`
graph returns a typed error through `Var::backward`'s existing `Result`
channel instead of panicking; a gradcheck-style value-semantic test confirms
the same op's gradient is unaffected when the backend succeeds. Negative: the
existing gradcheck suite (`crates/coeus-autograd/tests/autograd/gradcheck/`)
continues to pass with `?` substituted for the removed panics — a widened
signature must not change any already-verified gradient value. PR #461
verifies the value and gradient paths through the migrated caller closure.
The forced-allocation-failure fixture and device-matrix evidence remain
tracked by `COEUS-FALLIBLE-UNARY-EXECUTION` in `backlog.md`.
