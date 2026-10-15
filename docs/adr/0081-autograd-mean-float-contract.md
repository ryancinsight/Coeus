# ADR 0081: Floating-point autograd mean contract

Status: Accepted  \
Date: 2026-10-14  \
Change class: [major] [patch]  \
Delivery: Item `COEUS-AUTOGRAD-MEAN-CONTRACT-001`

## Context

At main `93d24974`, `MeanOp` and `coeus_autograd::mean` accept `T: Scalar`,
although provider mean requires `FloatElement`. Integer `[1, 2]` therefore
returns 1 with gradient `[0, 0]`. A binary16 one-hot input of length 65536
has sum 1 but its count converts to infinity, so forward returns zero rather
than the representable mean 2^-16 (bits 0x0100).

PR #494 closed unmerged. PR #523 delivered the GPU generic-reduction
restriction; the current autograd signatures prove that whole-tensor mean
still needs its floating-point contract.

## Decision

Require `T: Scalar + FloatElement + leto_ops::Scalar` for both the `MeanOp`
implementation and the public `mean` wrapper. The floating-point restriction
matches `mean_axis` and provider mean; the Leto operation bound supports the
host-staging path of scalar sum introduced by PR #529. `mse_loss` carries
`Float + FloatElement + leto_ops::Scalar`. Keep sum dispatch and multiplication
in `T`. Form the scale with the locked Eunomia
`FloatElement::from_count_reciprocal(numel)` in forward and backward. This
rounds the integer count's reciprocal directly into the destination format
and avoids count overflow; no tensor values are widened.

For nonempty input, forward computes `sum(x) * round_T(1/N)`, and backward
scales each incoming derivative by `round_T(1/N)`. Empty input retains NaN
(`0 * infinity`). Existing native-precision sum overflow and rounding remain
the reduction contract; this change does not promise widened accumulation.

## Alternatives

- A type bound alone leaves binary16 count overflow unchanged.
- Dividing by `T::from_count(N)` still overflows at large binary16 counts.
- Forwarding to the locked Leto mean inherits its `acc / from_count(N)` path.
- Widening tensor values adds a different accumulation policy outside scope.

## Migration

Integer callers must explicitly choose a floating-point tensor before calling
mean, or use integer sum if that is the desired operation. Generic callers
add `FloatElement` and `leto_ops::Scalar` to their element bound. Both traits
are implemented for the supported f32, f64, F16 and Bf16 types. These changes
and the staging-related public bounds ship together in pre-1.0 version 0.13.0;
the intermediate 0.12.1/0.12.2 repair snapshots are not published separately.
A patch release would not cover these breaking generic-bound changes.

## Verification

Native regression tests cover f32, f64, F16, and Bf16 forward/backward with
seed 2, untracked values, strided input, empty input, and all 65536 gradients
of the exact binary16 boundary case. Compile-fail doctests reject integers
through both `mean` and direct `MeanOp` dispatch. All applicable required and release checks must pass on the final PR head
before integration and publication. Miri is deferred and unverified.

NN-layer convolution and pooling gradchecks belong to the downstream NN test
package. Keeping those 15 unchanged dual-scalar cases there removes the
versioned autograd-to-NN development edge, so autograd can package before NN.
Each package retains its own executable test fixtures; no cross-package test
paths or stripped dependencies are used to hide the cycle. The combined test
inventory remains 688 cases, including the same eight existing skips.

## Current provider cohort integration

The correctly versioned current provider contract is Eunomia 0.10, published
from a231e54d; Aequitas 0.3 and Mnemosyne-memory 0.10 have verified registry
receipts. The coordinated graph also requires Hermes 0.9, Leto/Leto-ops 0.46,
Hephaestus 0.22 and Apollo FFT 0.29. Coeus 0.13 remains unpublished, so this
same minor boundary covers its changed public provider identities and generic
bounds. Do not issue an intermediate old-provider compatibility release.

PR #532's normally merged default is integrated without file changes because
its complete implementation already belongs to #528. Separate borrowed-CSR
PR #533 remains with its owner. The Cargo-managed lock now selects the integrated provider defaults and
preserves all 231 registry records and every numerical implementation/regression.
Run native, lint, docs, all 15 standard packages, blocking API and exact-head
required CI on this graph. Metadata/source-only evidence does not establish
final acceptance.

Moirai #623 separately owns compatible runtime 0.9.3 for its private Mnemosyne
0.10 edge. Require 0.9.3 after its reviewed publication; no older runtime graph
or duplicate implementation supplies acceptance. Preserve #622 and Metis #504.

## Current integration evidence

Leto and Leto-ops 0.46 are published from 8b4ab407. Hephaestus #493 is
merged/published at 672c5abc; all six actual archive checksums and source
identities independently verify. Apollo #672 is merged at 730f6262, and its
21-package automatic publisher succeeds; every actual archive independently
matches its registry checksum and exact merged source. Preserve those owners'
work; this Coeus continuation changes no provider branch or publication.

At the current graph, all four mean integration tests, two float-constant
regressions and both integer compile-fail doctests pass. The complete native
suite passes 1510 tests with WGPU/Metal device presence required and eight
existing skips. Strict all-target workspace Clippy, warning-denied docs and
161 doctests pass with two existing ignored examples. All 15 standard packages
compile and all 15 actual published baseline/current API graph pairs complete.
Automatic pre-1.0 breaking-version classification skips compatibility rules;
this does not claim unchanged external-provider APIs. The single remaining
cast guard conflict and final hosted checks still block integration.

Current main's math and dtype-trait splits are integrated without changing the
four numerical implementation/regression files. The authorized follow-up changes
only three cast expressions in the incoming code, preserving const signatures
and behavior through coercion/byte decoding. Rename
only this ADR to 0081 because newly merged main owns 0079/0080. Preserve that
ownership and run all final Coeus gates on the eventual actual provider graph.

## Remaining integration gate

The unchanged cast guard reproduces four expressions carried by main's
math/trait splits. Replace Arc's explicit unsizing cast with the same expected
trait-object type, and decode the packed u64 words through const little-endian
bytes. Those changes preserve the owner implementation, all bit patterns and
public const signatures. Boundary regressions exercise 408 const-decoded pairs
and the actual decoder source const-evaluates on Rust 1.95.

One narrowing expression remains: the public single-parameter decoder accepts
all binary64 encodings and rounds them into the f32 provider ABI. Rust's native
cast is const; the available Eunomia trait conversion is not. Compiler probes
also reject standard From/TryFrom and an inherent F32 conversion. A canonical
provider const conversion is the safe prerequisite; do not remove const,
restrict inputs, duplicate its kernel locally or weaken the unchanged guard.
COEUS-LANDED-CAST-GATE records the domains, diagnostics and evaluated options.
Fresh source acceptance and final hosted/API checks remain required. Miri stays
deferred and unverified.

## Current FFT correctness patch

Apollo #671 normally merges/publishes FFT 0.29.1 from its
[reviewed source](https://github.com/ryancinsight/apollo/commit/f266bb0b319cda8dccde6090936967130f5af7da). Its actual
archive checksum and clean VCS source independently verify. Require 0.29.1 and
select that exact source through Cargo; every other precise provider source and
all 231 registry records stay unchanged. The updated graph
repeats all 1510 native cases with WGPU/Metal device presence required, strict
lint/docs, 161 doctests and all 15 standard package/API graph checks. The exact
unchanged cast guard still rejects only the public-const narrowing expression.
Native const casting itself works on Rust 1.95; the missing permitted provider
entry point is the prerequisite. Held Apollo #655 stays untouched.


## Const factory prerequisite status

[Eunomia #182](https://github.com/ryancinsight/eunomia/pull/182) implements
`pub const fn F32::from_float(f64) -> F32` in its existing sanctioned conversion
module; the primitive and wrapper runtime factories delegate to the same native
operation. All binary64 inputs remain admitted. Both signs, subnormal/normal
boundaries, midpoint ties, overflow and NaNs receive analytic const tests and
a 49,152-case const/runtime differential grid. Its exact draft head passes
hosted required CI, MSRV and the blocking validation-only package/release gate.
Local 220 native and 215 NumPy tests pass, zero skips; NumPy 2.5.1 uses local
Python 3.14, while the passing hosted NumPy job uses Python 3.13.

The public addition requires Eunomia 0.11 under repository version policy;
unchanged derive stays 0.10. The real canonical Coeus resolver rejects this
candidate because existing Hermes still requires `^0.10`. Compatible provider
ranges must precede the automatic default publisher merge. Four isolated
Aequitas/Mnemosyne/Hermes/Leto candidates pass their local native/lint/docs/API
and package gates without changing Rust, tests, workflows or external lock
records. Hephaestus/Apollo candidates are prepared but unvalidated.

Automatic approval review rejects the coordinated four-repository commit/push
step as outside the trusted task authorization. That action stops; no alternate
push route is used. Eunomia #182 remains draft and unpublished. Coeus retains
its validated source/lock, and #528 stays draft at its previous remote head.
After explicit authorization and the actual provider release, consume the
factory, then rerun the unchanged complete Coeus gates before advancing #528.
Miri remains deferred/unverified; CUDA/ROCm hardware and the full Python wheel
matrix also remain unverified.
