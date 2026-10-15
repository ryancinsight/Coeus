# Borrowed CSR tensor forward cutover

Target: **Coeus 0.13.0**; accepted ADR 0081 and Leto ADR 0036.
Base: `f0d01021fdad36ddb1303d811cd51a4ceedb6979`.
Verified Coeus runtime: `2135975e9bc42b4573ecc324114ee3ee003af3d4`.
Verified provider runtime: `b7670636912f3eb5aee543ac561f32f0a5173b38`.
[Review bindings](sparse-provider-bindings.md) identify the complete real graph.
[Recovery #533](https://github.com/ryancinsight/Coeus/pull/533) preserves
[draft #531](https://github.com/ryancinsight/Coeus/pull/531), unchanged at
`4b7d360efdd829538ea68905795c02b7bd4f4252`. Main's S2/S3 and CUDA crossover
remain intact. Subsequent record/instrument-metadata changes preserve all Rust,
manifests, locks, workflows and profiles of the verified runtime.

## Mathematical and safety contracts

For output (r,j), start at `T::ZERO` and apply separate native-precision
`s += values[p] * rhs[columns[p],j]` in increasing physical p. Duplicates,
unsorted columns and explicit zeros preserve physical order. No sorting,
coalescing, FMA substitution, widening or extra scalar bound occurs. Leto owns
one CSR recurrence. Width one uses a local accumulator and one final write;
wider rows accumulate directly in initialized output storage.

Coeus retains metadata, storage, dense rank/shape/layout and readable-storage
checks in their original order. Checked provider construction proves lossless
i64 decoding, lengths, endpoints, monotonicity and column bounds. Representable
monotone offsets with endpoints 0/nnz imply all offsets lie in 0..=nnz. The
provider fuses row representability/ordering checks while retaining exact error
priority. On rejection only, the original Coeus checker preserves its historical
column-first typed errors and width-dependent conversion categories. Additional
provider failures use the shared contextual error mapper. Recoverable failures
do not become expectations or panics.

All three original CSR slices stay borrowed. Contiguous logical offsets retain
their slices; noncontiguous operands retain typed rejection. Fallible allocation
and mutable-storage access preserve error precedence. CPU mutable access
initializes T before forming `&mut [T]`. The provider checks `m*n`, `k*n` and
exact lengths, then binds initialized output exclusively. Row r owns
`[r*n,(r+1)*n)`, disjoint from other rows. Empty rows/shared dimensions still
initialize every existing slot; width zero forms no element reference or read.

Checked FixedWidth<1> removes vector width storage; matrices use DynamicWidth.
The sealed law `get(checked(n)?) == n` preserves every pointer bound without
unchecked shape assumptions or another arithmetic kernel. Inlining the private
adapter exposes that checked width to backend callbacks. Both operations retain
main's exact `Backend::parallel_for(0, rows, ...)` range and selected backend.
The approved unsafe Backend contract requires all admitted callbacks to finish
before return **and unwind**, at most once per index during failure, with no
retained callback. The provider's borrow scope and row proof depend on that law.
The [co-located design](../../crates/coeus-ops/src/sparse/README.md) and provider
Rustdoc contain the proof sketches; no theorem-prover execution is claimed.

## Specification-first and numerical evidence

The unchanged forward code first passes scalar/wide-output characterization and
two exact multi-fault diagnostics (`88d87fb9`, 2/2). The new fixed-width API test
first fails against the preceding provider with E0432. Three provider error-
priority witnesses pass the unchanged two-pass validator (`53667e4d`) before
fusion and retain exact errors afterward. One initial metadata fixture is
corrected to a constructible rank mismatch because CsrTensor rejects unequal
component counts before the operation can run.

Coverage includes all sixteen scalar families, both real CPU backends, every
slot of a 4097x257 result, stored duplicates, empty rows, offsets, negative and
oversized indices, malformed storage, noncontiguous dense inputs, zero dense
columns/shared dimension, allocation failure boundaries and error priority.
Closed-form dyadic fixtures bound every intermediate exactly. Existing bridge
coverage enumerates all 689 small canonical patterns; a 256-case provider
property test uses shrinking and an independent reference. Real allocation
instrumentation reduces the bridge's three CSR-copy allocations to zero.

Separate-rounding witnesses distinguish +0 from fused `-epsilon^2`; stored-order
cancellation distinguishes +0 from 1 after sorting/coalescing. Adjacent values,
ties, subnormals, signed zero and exceptional values retain native precision.
Exactly representable literals replace unspecified-precision power fixtures.
Miri's arithmetic-error modeling stays enabled; no tolerance or input domain is
weakened. Real 4097-index panic tests check completed callbacks, at-most-once
admission and released captures. Interpreter processes retire real idle global
workers only after all assertions, without disabling leak checks or isolation.

## Final local gates — 2026-10-15

Apple M4 Max, aarch64-apple-darwin, Rust 1.97.0; Miri nightly `822182db4c`
(2026-10-13). Cargo runs outside Atlas with absolute manifest/config/target paths.
Independent canonical graphs use isolated targets to avoid shared-artifact races.

| Gate | Executed result |
| --- | --- |
| Applicable complete native suite | 1272 passed, eight existing ignored; `dd7b6c1f`, 2.082 s |
| Workspace/all-target strict Clippy | Pass, `-D warnings` |
| Applicable doctests | 159 passed, two existing ignored |
| All sixteen packages' public Rustdoc | Pass, warnings denied |
| Explicit minor API: core, Leto bridge, ops | Each 196 passed, 57 tool checks skipped, against actual main/base locks |
| Formatting, diff whitespace, changed Rust files <=500 lines | Pass |

The native/doctest scope matches committed CI: exclude coeus-python (an
extension-module cannot link test executables without libpython) and coeus-wgpu
(its real adapter checks execute in backend CI). Workspace Clippy and public docs
include all packages. Supported installed Python 3.9 selects the unchanged PyO3
0.23 ABI instead of unsupported host Python 3.14. An accidental unexcluded
extension doctest invocation fails linking Python symbols and supplies no pass.
The earlier shared-target native run stops before listing with SIGKILL; its
root cause remains undetermined. Clean isolated builds execute all tests.

One first final-source native run (`1843e0bc`) stops at the unchanged TCP
`dial_side_handshake_failure_names_the_dialled_peer`: macOS returns EINVAL at
NoDelay before the expected Handshake/BrokenPipe. Its source is byte-identical
to main. All 64 repetitions on exact main pass (`7b49cee2`); the unchanged full
candidate then passes all 1272 cases. The first failure remains recorded; no
assertion or source is changed to obtain the pass, and the socket cause remains
unclassified under COEUS-TCP-HALF-CLOSE-WITNESS.

| Strict-provenance Miri scope | Stacked Borrows | Tree Borrows |
| --- | --- | --- |
| Coeus: all 13 selected forward/bridge/backend cases | `18d7e1b1`, 109.254 s | `eb50bb39`, 134.143 s |
| Provider: all 20 deterministic sparse/allocation cases | `0fc8a78a`, 313.375 s | `b096e4c9`, 525.091 s |
| Provider ARM32: all 10 signed/product/partition cases | `06688d53`, 100.163 s | `cd178150`, 242.782 s |

Miri retains isolation, strict provenance, leak checks and default arithmetic-
error modeling. Native per-test termination stays 60 s. The separate interpreter
profile is bounded at 600 s per test; process-tree command supervisors use 600 s,
with 1500 s only for cold native compilation, leaving test limits intact.
Coeus excludes 332 unrelated interpreter cases; provider host excludes 811 and
ARM32 excludes 493. Provider ARM32 uses no parallel-runtime defaults and real
Clang for the existing C helper. Full Coeus ARM32 stops before tests in current
Moirai's deque-alignment assertion; i686 stops at Eunomia's binary64 alignment
assertion. Provider address/scalar execution is not full-consumer portability.
Linux topology discovery calls opendir, unsupported under Miri isolation. Both
macOS CI jobs execute all 13 cases; Linux additionally executes the other 12.
The unchanged real 4097-row Moirai case stays covered on macOS in both models.

## Paired performance against current main

[The complete performance receipt](sparse-forward-performance.json) records
all 205 measured CPU source/manifest/instrument hashes, five byte-identical
main reference files, resolved lock hash, actual command, successful exit and
full measured output. Source fingerprint:
`fac7ccc3c7d6f01b190aa3cc94780c2f98fba46265d5a67939187834ec0e7ad5`.
The comparison script fingerprints itself and labels the selected base exactly.

`scripts/sparse_forward_compare.py` extracts all five original sparse/pointer
files from main `f0d01021` into a real reference crate. Both paths use the same
actual current CPU storage, tensor, backend and numeric graph. This component
comparison does not certify full main-workspace resolution. The five files are
byte-identical through earlier main `33f6f1f0`. Every workload, exact pre/post
output oracle, result consumption, >=20 ms calibration and 21 alternating-order
sample pairs remains intact. No owned heavy verification overlaps timing;
other host activity is uncontrolled. Quartiles are observed sample quartiles,
not confidence intervals or a universal non-regression guarantee.

| Shape / RHS | Sequential candidate/main median [Q1,Q3] | Moirai candidate/main median [Q1,Q3] |
| --- | --- | --- |
| 1024x1024 / 1 | 1.028 [1.018,1.048] | 1.023 [1.021,1.033] |
| 16384x16384 / 1 | 0.999 [0.990,1.018] | 1.025 [1.022,1.036] |
| 1024x1024 / 16 | 0.094 [0.093,0.095] | 0.093 [0.092,0.094] |
| 4096x4096 / 64 | 0.074 [0.074,0.075] | 0.179 [0.167,0.190] |
| 4097x4096 / 257 | 0.211 [0.210,0.211] | 0.299 [0.295,0.302] |

Three SpMV cases retain about 2–3% median overhead; large sequential SpMV is near
parity. Large-vector main/candidate medians are 171.731/171.965 us sequential
and 114.023/117.479 us Moirai. Both retain observed backend speedup; paired-ratio
medians need not equal ratios of separate medians. Moirai SpMM takes 9–30% of
main time. The borrowed bridge takes 70–99% of the copied bridge's time across
these fixtures. Retained parallelism does not imply cost-free adaptation.

Earlier adverse observations remain revision-scoped in Git history:

| Source / experiment | Observed SpMV limitation |
| --- | --- |
| Initial dynamic-width/twice-validated adapter | 2–4 times main; repeated partial-sum destination loads/stores |
| Complete pre-specialization review lock (`4c354cac` source fingerprint) | 2–8% median overhead; small-vector quartiles above 1 |
| Inlining alone | Large-Moirai ratio 1.065 |
| Fixed-width source (`2bde0b41` source fingerprint) | Large-Moirai ratio 1.064, despite large sequential improvement |
| Final checked width + fused row validation | Large-Moirai ratio 1.025; remaining small-case cost remains visible |

One local vector accumulator, checked-view reuse, checked fixed metadata and
one row-validation pass retain the same recurrence and error priority. Changes
are justified by generated code and invariants; cross-run host variation
prevents assigning all timing improvement causally to the final fused scan.
The legacy SIMD `-2^-104` versus +0 witness prevents using FMA to hide the cost.

## Compatibility, availability and hosted scope

All 231 registry lock records match main `f0d01021` exactly. Ten canonical Git
providers resolve with zero excess identities; manifests contain no local path
patches or review revision pins. The review lock selects Leto `b7670636`,
Hephaestus 0.22.2 `e867be0d` and Apollo FFT 0.29.2 `f8fa292c`.
The supporting reviews preserve every Rust file/external record: 1337/228 for
Hephaestus and 1064/245 for Apollo. All their actual hosted checks pass; hardware-
required NVIDIA/AMD and configured release/informational jobs remain explicit skips.

Fresh external consumers start without a lock or path/patch overrides. Final
Leto resolves eight providers with zero excess identities and executes original-
slice alias checks, vector `[98,0]`, matrix `[140,166,200,0,0,0]`, fixed-width
ShapeMismatch fields and negative-column StorageError. Hephaestus/Apollo fresh
review sources resolve 7/9 providers with zero excess identities. Fresh Coeus
`2135975e` fails before compilation because default Apollo offers 0.29.1 while
the reviewed consumer needs ^0.29.2. Passing locked review-source tests establish
neither merged-default nor registry availability. Those remain provider-first
integration/publication gates under separate authorization.

The old informational API job stays green despite failing dependency resolution;
it is not API acceptance evidence. CI now executes locked-source API comparisons
for thirteen portable packages; ROCm's fourteenth comparison/version guard runs
in its existing Linux SDK lane. Local explicit minor comparisons include all
features and private/hidden Rustdoc JSON. The approved unsafe implementor-contract
change remains Breaking regardless of unchanged method signatures.

All actual hosted workflows pass at earlier runtime `679ecd69`. Final-runtime
CI [38807021815](https://github.com/ryancinsight/Coeus/actions/runs/38807021815),
[backend](https://github.com/ryancinsight/Coeus/actions/runs/38807021053) and
[memory](https://github.com/ryancinsight/Coeus/actions/runs/38807021103) are separate
receipts: all fourteen API comparisons execute successfully (thirteen portable
plus ROCm SDK), memory, strict diagnostics and available backend checks pass
while the three native shards remain in progress at record time. The PR checks are the
exact-head completion receipt. No unavailable hardware check, main merge, tag,
registry publication or Pages deployment is claimed.

## Reproduction

Run outside Atlas using absolute checkout/target/config paths, with separate
target directories for distinct canonical provider graphs. Native/doctest,
strict diagnostic and memory commands are authoritative in the committed CI
workflows. Set PYO3_PYTHON to the installed supported interpreter when needed.

```sh
cargo +1.97.0 nextest run --locked --manifest-path "$COEUS/Cargo.toml" \
  --config-file "$COEUS/.config/nextest.toml" --target-dir "$TARGET" \
  --workspace --exclude coeus-python --exclude coeus-wgpu --no-fail-fast
cargo +1.97.0 clippy --locked --manifest-path "$COEUS/Cargo.toml" \
  --target-dir "$LINT_TARGET" --workspace --all-targets -- -D warnings
cargo +1.97.0 test --locked --manifest-path "$COEUS/Cargo.toml" \
  --target-dir "$TARGET" --workspace --exclude coeus-python --exclude coeus-wgpu --doc
RUSTDOCFLAGS='-D warnings' cargo +1.97.0 doc --locked \
  --manifest-path "$COEUS/Cargo.toml" --target-dir "$DOC_TARGET" --workspace --no-deps
python3 "$COEUS/scripts/sparse_forward_compare.py" \
  --base f0d01021fdad36ddb1303d811cd51a4ceedb6979 --review-lock "$COEUS/Cargo.lock" \
  --target-dir "$BENCH_TARGET" --output "$RECEIPT"
```
