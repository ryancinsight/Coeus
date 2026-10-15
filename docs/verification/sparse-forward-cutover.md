# Borrowed CSR tensor forward cutover

Target: Coeus 0.13.0; accepted ADR 0081 and Leto ADR 0036. Base:
`f0d01021fdad36ddb1303d811cd51a4ceedb6979`. Provider and binding sources:
[review graph](sparse-provider-bindings.md). Original preservation draft
[#531](https://github.com/ryancinsight/Coeus/pull/531), head
`4b7d360efdd829538ea68905795c02b7bd4f4252`, remains intact.

## Mathematical and safety contracts

For each output (r,j), start at `T::ZERO` and apply separate native-precision
`s += values[p] * rhs[columns[p],j]` in increasing physical p. Duplicates,
unsorted columns and explicit zero coefficients retain their original order.
No sorting, coalescing, FMA substitution, widening or new scalar bound occurs.
Leto owns the single row traversal; a local one-column accumulator writes once,
and wider rows accumulate in their initialized output slice.

Coeus checks the original tensor metadata/storage and dense layout in the
original order. One checked signed provider construction proves lossless
index decoding, lengths, endpoints, monotonicity and column bounds. Monotone
representable offsets with endpoints 0/nnz imply every offset lies in 0..=nnz.
On rejection only, the original index validator produces the exact historical
column-first diagnostic, including width-dependent conversion categories.
Unexpected provider errors propagate through the shared contextual mapper.
No CSR buffer is copied or reinterpreted; contiguous logical offsets retain
their slices and noncontiguous operands retain typed rejection.

Fallible output allocation and mutable access keep their original precedence.
CPU storage initializes T before forming a typed mutable slice. The provider
checks output/RHS products and exact lengths, then gives row r exclusive
`[r*n,(r+1)*n)` output. Width zero forms no element reference or input read;
empty rows/shared dimensions still initialize all existing output elements.
The backend dispatch remains exactly the selected `parallel_for` range.
Accepted completion-on-return/unwind obligations keep all borrowed storage
live until every admitted callback completes and prevent callback retention.

## Acceptance evidence

The original forward implementation first passes the all-scalar/wide-output
characterization. Two new multiple-fault witnesses also pass on byte-identical
main before validation changes (Nextest `88d87fb9`, 2/2). Their first metadata
fixture is corrected to a constructible rank mismatch: CsrTensor's constructor
already rejects unequal component counts before the operation can run.

All 151 real CPU component integration cases pass after the single-checked
adapter (`35c9aa18`, 0.136 s, zero skipped). They include all sixteen scalar
families, both real CPU backends and every slot of a 4097x257 output with
stored duplicates, row-dependent values and empty rows. The exact dyadic
closed form bounds all intermediate values; no empirical tolerance defines
those assertions. Existing bridge coverage enumerates all 689 small canonical
patterns and compares independent computed values.

The complete review graph resolves and accepts `--offline --locked`, with ten
canonical provider identities and no excess source. A broader compile finds
and corrects the moved CsrTensor test import and an empty-shape assertion whose
element type becomes ambiguous through a current transitive PartialEq impl.
The assertion remains exact; its expected slice is explicitly usize.

Provider `a2778334` passes all 1200 native core/ops tests, zero skipped, strict
all-target Clippy in both feature modes, and all 18 deterministic sparse Miri
contracts under strict provenance/isolation in both borrow models. Provider
native properties use 256 generated stored-order cases with shrinking.
The full applicable Coeus native suite passes 1272 tests (`42193f0a`,
5.302 s), with eight existing ignored cases. Strict all-workspace/all-target
Clippy and warning-denied workspace Rustdoc pass. Doctests pass 159 cases with
two existing ignored examples. The merged current-main candidate repeats 1272 native passes
(`94738d0e`, 7.152 s), strict Clippy, 159 doctests and warning-denied Rustdoc.

The merged candidate passes all thirteen selected Coeus memory cases in both
strict-provenance models, with isolation and arithmetic-error modeling enabled:

| Group | Stacked run / elapsed | Tree run / elapsed |
| --- | --- | --- |
| Ten forward/layout/rejection/allocation cases | `b3ce81fb`, 74.414 s | `4d053bae`, 68.011 s |
| All sixteen scalar families | `a12924c0`, 27.363 s | `68f57c39`, 35.663 s |
| Two real backend unwind/capture cases | `6fc0f450`, 24.597 s | `6a581b5d`, 31.938 s |

The first Moirai interpreter run passes every callback/capture assertion but
fails process exit because idle global workers remain live. The isolated Miri
process now calls the real shared runtime's shutdown only after all those
assertions. Both models then pass without disabling leak checks or isolation.
Hosted memory run [38790814858](https://github.com/ryancinsight/Coeus/actions/runs/38790814858)
passes all four jobs: all thirteen cases on macOS and twelve on Linux, in each
borrow model. Linux's real Moirai topology discovery calls `opendir`, which
Miri isolation rejects before callbacks run. Its unchanged 4097-row case stays
in both macOS jobs; Linux additionally covers every other selected case.
No assertion, workload, native limit, isolation or leak check is relaxed.
The installer explicitly selects nightly because the pinned release-specific
Rust action otherwise installs stable even when given a nightly input.

## Final paired performance against current main

`scripts/sparse_forward_compare.py` extracts all five original sparse/pointer
files byte-for-byte from the main revision into a real reference crate. Both
paths use the same actual current CPU storage, backend and numeric graph.
The instrument retains all fixtures, exact oracles, result consumption,
calibration of at least 20 ms per side and 21 alternating-order sample pairs.
The initial exported-main control at `f9ebef34` preserves all 615 production
Rust files and every external locked package. The five reference files remain
byte-identical through main `33f6f1f0` and `f0d01021`; incoming changes partition
unrelated traits/autograd modules and update the CUDA crossover/provider graph. Cross-run absolute times are not paired.

The first dynamic-width/twice-validated adapter regresses SpMV by 2–4 times.
Native disassembly shows repeated destination loads/stores for partial sums.
One local accumulator removes those writes without cloning the CSR traversal;
reusing checked validation removes the second successful index scan.

The final run uses the current complete review lock and no concurrent owned
heavy verification. It compares candidate/main time. Quartiles crossing 1
show uncertainty around parity; they do not establish a universal
non-regression bound. The host runs other work. All outputs pass exact value
checks before and after timing. No workload, scalar, tolerance or assertion
changes in response to the measurements.

| Shape / RHS | Sequential ratio median [Q1,Q3] | Moirai ratio median [Q1,Q3] |
| --- | --- | --- |
| 1024x1024 / 1 | 0.929 [0.883,0.976] | 1.028 [0.974,1.044] |
| 16384x16384 / 1 | 1.028 [0.956,1.049] | 1.022 [0.945,1.081] |
| 1024x1024 / 16 | 0.094 [0.093,0.095] | 0.092 [0.090,0.093] |
| 4096x4096 / 64 | 0.076 [0.075,0.077] | 0.182 [0.177,0.188] |
| 4097x4096 / 257 | 0.207 [0.205,0.210] | 0.291 [0.284,0.298] |

The large SpMV medians are 160.493/163.655 us sequential and 122.505/119.766 us
Moirai (main/candidate). The paired-ratio median need not equal the ratio of
those separate medians. Backend execution retains its observed speedup and
its call path. SpMM improves with separate rounding; the legacy SIMD entry
point's exact `-2^-104` versus +0 counterexample is not used as a replacement.

Candidate source fingerprint (sorted JSON of every CPU source/manifest and the
instrument): `ece0fe1ad50e727327f0eaa8d0a3adaaaebfb9a20dd5b20eec51c28d4d41c6c5`. The instrument checks that fingerprint before and
after measurement; both implementations' full computed outputs pass.


## Exact source compatibility and downstream availability

Explicit prebuilt-Rustdoc comparisons use actual committed base/head locks;
Coeus core, Leto bridge and operations each pass 196 API checks with 57 tool
checks skipped, comparing `33f6f1f0` with `d9ee1c90` under an explicit minor
compatibility requirement. The later CPU source/API remains unchanged.
The old hosted informational job fails dependency resolution, yet stays green:
it discards both revision locks, rejecting candidate Apollo ^0.29.2 and base
Eunomia ^0.9 before comparison. That green job is not API acceptance evidence.
CI now uses Atlas's reviewed locked-source comparison for thirteen portable
packages; the fourteenth, ROCm, executes with its version guard in the existing
Linux SDK lane. Build/schema/comparison errors remain failures. The separate
fresh-consumer gate still checks actual default/registry availability.

Fresh consumers outside Atlas start without a lock or path/patch overrides.
Actual review-source Leto 0.47 (`a2778334`), Hephaestus 0.22.2 (`e867be0d`) and
Apollo FFT 0.29.2 (`f8fa292c`) resolve 7/7/9 canonical providers, respectively,
with zero excess identities. A real fresh Leto binary aliases the original
signed arrays and computes `[98,0]` and `[140,166,200,0,0,0]`, including an empty
row, then rejects a negative column. These are executed outputs, not only
metadata success. Fresh Coeus ops/autograd 0.13 at `4c2a29c5` fails before
compilation because default Apollo still exposes only 0.29.1. No default-source
or registry availability follows from the passing complete review lock.

Current-main integration preserves S2/S3 and the CUDA crossover update #539.
Its 231 registry lock records match `f0d01021` exactly, including all four newly
updated package records. Concurrent Hephaestus publication consumes 0.22.1;
compatibility #497 therefore preserves both current CUDA fixes and targets
0.22.2. The current Coeus lock selects that exact review source while retaining
one identity for each of ten providers. Current-head hosted checks are separate
from the earlier recorded runs and must be inspected before integration.

## Reproduction and limits

Run Cargo from outside Atlas with absolute manifest/target/config paths so
its development overlay cannot substitute sibling checkouts. Native CI uses
`nextest --workspace --exclude coeus-python --exclude coeus-wgpu` and the
committed 60-second per-test termination budget. Strict Clippy checks all
workspace targets; docs deny warnings. Installed supported Python 3.9 selects
the unchanged PyO3 0.23 ABI instead of the host's unsupported 3.14 default.
Miri retains isolation, strict provenance and default arithmetic-error modeling;
its separate bounded interpreter profile leaves native limits unchanged.
The unused workspace keyword field is removed; all sixteen packages retain
identical actual keyword metadata, and nightly no longer emits that warning.

The 4097x257 execution runs natively; interpreter cases target the proven
address/lifetime, scalar, rejection and empty-output boundaries. The complete Coeus ARM32 attempt stops before tests in current Moirai
`ChaseLevInner<u8, DeferredReclaim>` alignment >=64. The second borrow-model
command therefore does not run. Provider-only ARM32 executes eight signed/product/partition cases in each
strict/isolation borrow model, with default parallel runtime features disabled:
Stacked `6cafdb7b` (123.766 s), Tree `ee7e731c` (230.257 s). Those tests exercise
the actual row kernel, all sixteen scalar families, target-width 2^32 rejection,
stored order, separate rounding, empty output and failure atomicity. Real Clang
cross-compiles the existing C helper. This establishes provider address/scalar
portability; it does not establish full-consumer ARM32 support.
i686 still fails the upstream binary64 alignment assertion before tests; no
i686 execution result follows.
Unseeded default-source and registry availability require the independent
provider reviews to integrate first. No release or main merge is part of
this independently reviewable recovery.
