# Borrowed CSR tensor forward cutover

Target: Coeus 0.13.0; accepted ADR 0081 and Leto ADR 0036. Base:
`f9ebef3462a8c1704d0a4c7daf1770aeb1c04527`. Provider and binding sources:
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
two existing ignored examples. These results precede the new main refactors;
the merged candidate repeats its affected gates.

Stacked Borrows passes ten forward/layout/rejection/allocation cases
(`27fcbfe3`, 78.738 s) and the sixteen-family scalar witness (`dbb5d730`,
27.190 s). The first real-Moirai interpreter case passes every callback and
capture assertion but fails process exit because global workers remain live.
Its isolated Miri process now calls the actual shared runtime's shutdown only
after those assertions. No callback check, workload or leak detection changes.
The corrected lifecycle and Tree Borrows gates remain in progress.

## Performance method and checkpoint

`scripts/sparse_forward_compare.py` extracts all five original sparse/pointer
files byte-for-byte from the main revision into a real reference crate. Both
paths use the same actual current CPU storage, backend and numeric graph.
The instrument retains all fixtures, exact oracles, result consumption,
calibration of at least 20 ms per side and 21 alternating-order sample pairs.
A separate exported-main control preserves all 615 production Rust files and
every external locked package. Cross-run absolute times are not paired.

The first dynamic-width/twice-validated adapter regresses SpMV by 2–4 times.
Native disassembly shows repeated destination loads/stores for partial sums.
One local accumulator removes those writes without cloning the CSR traversal;
reusing checked validation removes the second successful index scan.

The resulting checkpoint compares candidate/main time. Quartiles crossing 1
show uncertainty around parity; they do not establish a universal
non-regression bound. The host runs other work. All outputs pass exact value
checks before and after timing. No workload, scalar, tolerance or assertion
changes in response to the measurements.

| Shape / RHS | Sequential ratio median [Q1,Q3] | Moirai ratio median [Q1,Q3] |
| --- | --- | --- |
| 1024x1024 / 1 | 0.974 [0.934,1.019] | 1.013 [0.976,1.033] |
| 16384x16384 / 1 | 0.986 [0.958,1.032] | 1.016 [0.975,1.097] |
| 1024x1024 / 16 | 0.094 [0.094,0.095] | 0.094 [0.093,0.094] |
| 4096x4096 / 64 | 0.076 [0.075,0.077] | 0.185 [0.178,0.195] |
| 4097x4096 / 257 | 0.211 [0.209,0.212] | 0.297 [0.288,0.324] |

The large SpMV medians are 183.202/180.201 us sequential and 125.688/125.238 us
Moirai (main/candidate). This retains the observed backend speedup as well as
the backend call itself. SpMM improves while keeping separate rounding; the
legacy SIMD entry point's exact `-2^-104` versus +0 counterexample remains
observable and is not used as a compatible replacement.

## Reproduction and limits

Run Cargo from outside Atlas with absolute manifest/target/config paths so
its development overlay cannot substitute sibling checkouts. Native CI uses
`nextest --workspace --exclude coeus-python --exclude coeus-wgpu` and the
committed 60-second per-test termination budget. Strict Clippy checks all
workspace targets; docs deny warnings. Installed supported Python 3.9 selects
the unchanged PyO3 0.23 ABI instead of the host's unsupported 3.14 default.
Miri retains isolation, strict provenance and default arithmetic-error modeling;
its separate bounded interpreter profile leaves native limits unchanged.

The 4097x257 execution runs natively; interpreter cases target the proven
address/lifetime, scalar, rejection and empty-output boundaries. Provider
32-bit ARM executes checked representability; i686 still fails the upstream
binary64 alignment assertion before tests. No i686 execution result follows.
Unseeded default-source and registry availability require the independent
provider reviews to integrate first. No release or main merge is part of
this independently reviewable recovery.
