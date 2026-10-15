# Sparse view recovery from PR #531

## Scope and revisions

This patch recovers the allocation-free `coeus-leto` bridge. It does not claim
the tensor forward cutover is complete. Preservation draft
[Coeus #531](https://github.com/ryancinsight/Coeus/pull/531), head
`4b7d360efdd829538ea68905795c02b7bd4f4252`, remains intact.

The original audit base is Coeus main `cd8e984e0d9c4f3621bd89f7830646de845a22e5`.
Its canonical lock selects Leto `30a02d338f2d291dbb5725c366d3571dd68171b8`
(0.45.2), Hermes `f0138e4e9a521196d8b7c06420663581077488ff`
(0.8.1), and Moirai `ddde64ef33a2fcd73773e22466fd60c52239f205`
(0.8.2). Leto main `248fe4f63c674f6fca1f89ac19c1f817e67bc417`
has the same relevant borrowed-CSR, arithmetic and serial scheduling contracts.
The rescue cites ADR 0035; no Coeus ADR with that number appears in the
preserved tree or current main. The decisions below use current executable
contracts.

The branch subsequently integrates main
`dd27214f88a5c23e0e9abc58e014e58b902d67fb` without rewriting the rescue history.
The final section records repeated checks and performance on that provider
graph; earlier measurements remain historical evidence for the original base.

## Compatible increment

`CsrDispatch` already carries borrowed `usize` indices and Leto's canonical
CSR contract. Previously each call copied values, columns and row pointers
into `CsrMatrix::from_parts`. It now calls `CsrView::from_slices` and the
borrowed sparse kernel. Both constructors share `validate_csr_parts`; both
kernel entry points execute the same implementation. Therefore:

- Accepted/rejected CSR structures and typed errors are identical.
- Values, columns and offsets retain their original storage; no cast or copy
  occurs on any pointer width.
- Per-output operation order, scalar/SIMD selection, signed-zero and NaN bits
  remain those of the existing Leto entry point.
- Dense view construction and output-size checks precede output mutation.
- SpMV writes one sum per row, including empty rows. SpMM initializes the
  entire destination before accumulating, including empty shared dimensions.

The descriptor and public signatures remain unchanged. This is a patch-class
change with target 0.12.1. It adds no production unsafe code, dependency, CSR
validator or mathematical kernel. Test tracing uses the already locked
`tracing-subscriber` 0.3.23 package as an explicit development dependency.

## Why the tensor forward replacement is not sound yet

| Contract | Current Coeus tensor path | Current Leto borrowed path | Required resolution |
| --- | --- | --- | --- |
| Errors | `Result<Tensor<T, B>, B::Error>`; validation, allocation and mutable borrowing propagate errors | `Result<(), LetoError>` | Preserve existing Coeus validation and typed backend failures through the complete caller chain. |
| Index representation | `i64`, checked nonnegative and `usize`-representable | `&[usize]` | Prefer signed/generic provider indexing. An equal-width safe Pod cast after validation can borrow on 64-bit; 32-bit needs checked conversion and necessarily distinct storage with the present interface. Never reinterpret a 64-bit slice as same-length 32-bit indices. |
| Row order | Unsorted columns and duplicates accept; entries accumulate in stored order | Columns must be strictly increasing in each row | Provider view validation must support stored-order CSR without sorting or coalescing. |
| Dense layouts | Noncontiguous operands reject with `BackendError::Storage`; contiguous operands may have nonzero offsets | General strided views may materialize | Keep current rejection and slice offsets unless an additive layout feature is specified separately. |
| Arithmetic | Separate native-precision multiplication and addition | SpMM uses SIMD `axpy`, which may fuse multiply/add | Provider arithmetic policy must explicitly preserve separate rounding. |
| Scheduling | `B::parallel_for` partitions rows | Sparse view kernels run serial row loops | Expose provider row partitioning without replicating sparse mathematics in Coeus. |
| Scalar bounds | `T: coeus_core::Scalar` | `T: leto_ops::Scalar` | Audit every existing scalar implementation and caller before adding a public bound. |
| Initialization | Unsafe `alloc_on` requires complete initialization; allocation and mutable access are fallible | Caller supplies initialized mutable output | Prove every successful result slot is initialized, including all empty dimension combinations; propagate preparation failures. |

Canonicalizing is observably incompatible: in binary64, the stored sequence
`2^53, 1, -2^53` with columns `[1, 0, 1]` sums to 0; coalescing column 1 first
changes the result to 1. The regression keeps the original order and duplicates.
Sorting can also change rounding: for columns `[1, 1, 0]`, sorting the same
values by column puts `-2^53` first, giving `(-2^53 + 2^53) + 1 = 1`.

Fusing is also observably incompatible. With `u = 2^-52`,
`(1+u)(1-u) = 1-2^-104`. Separate binary64 rounding produces 1 before adding
−1, hence +0. Fused multiply/add instead produces −2^-104. The release
instrument confirms Leto returns `−4.930380657631324e-32` on this host.
This is an exact counterexample, not a tolerance choice.

The historical replacement would remove backend row execution as well. A
sequential provider is not performance-equivalent merely because its real
arithmetic formula agrees. The next implementation belongs to
[COEUS-SPARSE-VIEW-FORWARD](../../backlog.md#coeus-sparse-view-forward).

## Reproducible verification

Run Cargo from a directory outside Atlas so its development overlay cannot
substitute stale local provider checkouts. Pass the absolute manifest and the
shared target directory. Native Nextest uses the committed profile: 30 seconds
marks a slow test and a second period terminates it at 60 seconds.

```sh
cargo +1.97.0 nextest run --offline --locked --manifest-path "$COEUS/Cargo.toml" \
  --target-dir "$ATLAS/target" --config-file "$COEUS/.config/nextest.toml" \
  -p coeus-leto -p coeus-ops
cargo +1.97.0 clippy --offline --locked --manifest-path "$COEUS/Cargo.toml" \
  --target-dir "$ATLAS/target" -p coeus-leto -p coeus-ops --all-targets -- -D warnings
cargo +1.97.0 fmt --manifest-path "$COEUS/Cargo.toml" --all -- --check
```

The allocator regression fails on unchanged main with three allocations from
the CSR copies, after checking computed products. New contract tests enumerate
all 689 sparsity patterns over shapes 0..=3 by 0..=3, comparing an independent
dense sum and the owned Leto route. Additional cases verify both native float
precisions, NaN payload bits, cancellation, typed negative/oversized indices,
contiguous offsets, layout rejection, empty rows, zero columns and complete
output initialization.

## Performance method

`crates/coeus-ops/benches/sparse_recovery.rs` measures 21 alternating-order
pairs, calibrating each side to at least 20 ms per sample. It reports medians
and paired ratio quartiles. Every output is checked against a separate dyadic
dense reference before measurement, and bridge outputs are checked again
afterward. Results are consumed during timing.

The owned-bridge reference reproduces main's three copies and owned Leto call.
An exported main tree receives only the identical instrument and its test
dependency declaration; its production sources match the recorded base byte
for byte. Its bridge-control ratios near 1 verify the comparison instrument.
The candidate then compares that owned reference with the recovered bridge.
Forward comparisons include validation and output allocation on both sides;
the proposed provider side starts with native `usize` indices, an optimistic
case that excludes the cost of signed-index conversion.

Host: Apple M4 Max, 16 logical CPUs, `aarch64-apple-darwin`, Rust 1.97.0,
workspace release settings (optimization 3, thin LTO, one codegen unit).
The host also runs unrelated work; paired samples reduce order bias but do
not establish a quiet-host throughput guarantee. No workload or tolerance is
changed to obtain a pass. Source equality proves this patch preserves tensor
forward scheduling independently of noisy timing measurements.

```sh
python "$ATLAS/scripts/atlas-lane.py" export coeus \
  cd8e984e0d9c4f3621bd89f7830646de845a22e5 "$BASELINE"
cargo +1.97.0 bench --offline --locked --manifest-path "$COEUS/Cargo.toml" \
  --target-dir "$ATLAS/target" -p coeus-ops --bench sparse_recovery
```


## Measured results (2026-10-15)

The current-main bridge control has paired ratios 0.986, 1.006, 0.993 and
1.001 in the four cases below. The recovered bridge measures:

| Case | Owned main reference (µs) | Borrowed bridge (µs) | Paired ratio, median [Q1, Q3] |
| --- | ---: | ---: | ---: |
| 1024x1024/rhs1 | 9.639 | 6.776 | 0.704 [0.697, 0.710] |
| 16384x16384/rhs1 | 271.549 | 242.164 | 0.890 [0.882, 0.895] |
| 1024x1024/rhs16 | 28.350 | 25.981 | 0.919 [0.910, 0.924] |
| 4096x4096/rhs64 | 481.853 | 469.066 | 0.972 [0.968, 0.977] |

The optimistic direct-Leto replacement compared with the existing Moirai
forward kernel measures:

| Case | Current Coeus (µs) | Direct borrowed Leto (µs) | Paired ratio, median [Q1, Q3] |
| --- | ---: | ---: | ---: |
| 1024x1024/rhs1 | 5.314 | 6.971 | 1.310 [1.296, 1.320] |
| 16384x16384/rhs1 | 115.267 | 241.647 | 2.069 [1.970, 2.118] |
| 1024x1024/rhs16 | 309.358 | 28.176 | 0.091 [0.090, 0.091] |
| 4096x4096/rhs64 | 786.928 | 481.068 | 0.612 [0.608, 0.614] |

The 16,384-row vector case is approximately 2.07 times slower despite the
provider comparison excluding signed-index conversion. SpMM benefits from
SIMD on these exactly representable fixtures, but the separate rounding
counterexample prevents treating that speed as a compatible implementation.

## Gate results and limits

| Gate | Result | Scope |
| --- | --- | --- |
| Native Nextest | 293 pass, 0 skipped | Complete `coeus-leto` and `coeus-ops` suites; committed 60-second test bound. |
| Public doctests | 38 pass | `coeus-leto`, canonical lock, Rustdoc warnings denied. |
| Strict Clippy | Pass | Both packages, all targets, `-D warnings`, canonical locked Git graph. |
| Formatting and whitespace | Pass | Workspace formatting and `git diff --check`. |
| Allocation regression | Red on main: 3 allocations; green after recovery: 0 | Real System allocator counting; vector and matrix values checked. |
| Miri | 3 selected tests pass | `aarch64-apple-darwin`, strict provenance and isolation enabled: allocation/output initialization, malformed CSR/dense-shape rejection, empty shared dimensions and output columns. |
| API compatibility | 223 checks pass, 30 skipped by the tool | `cargo-semver-checks` compares baseline/candidate Rustdoc JSON generated from their locked graphs; patch classification. |
| Release performance | 21 pairs per comparison | Exact exported-main control and recovered bridge; full values verified. |
| Fresh Git consumer | Fails on unchanged main | No existing lock, no path/patch overrides: `eunomia = ^0.9.0` cannot resolve against its default Git branch, now 0.10.0. |
| Normal SemVer source build | Cannot resolve | Same existing fresh-resolution failure. The API-only comparison does not prove a publishable fresh dependency graph. |
| 32-bit i686 Miri | Compile failure before Coeus | Locked Eunomia asserts `align_of::<F64>() == 8`; i686 provides 4. No 32-bit memory-safety result. |
| 32-bit ARMv7 Miri | Compile failure before Coeus | Locked Moirai asserts Chase–Lev deque alignment >=64. No 32-bit memory-safety result. |

Miri uses the installed nightly (`miri 0.1.0`, compiler commit `822182db4c`,
2026-10-13). Its initial test run stopped at the logger's wall-clock query;
the logger now omits wall-clock timestamps, preserving hierarchical span
events with isolation enabled. Nightly Cargo also reports the pre-existing
unused `workspace.package.keywords` field, and Miri reports that it ignores
optimization levels. Neither diagnostic is suppressed.

The API-only comparison uses Rust 1.97 Rustdoc JSON, enabling that diagnostic
format for these commands only. Generate and save the baseline JSON before
the candidate overwrites the shared output:

```sh
RUSTC_BOOTSTRAP=1 cargo +1.97.0 rustdoc --offline --locked \
  --manifest-path "$BASELINE/Cargo.toml" --target-dir "$ATLAS/target" \
  -p coeus-leto --lib -- -Z unstable-options --output-format json
cp "$ATLAS/target/doc/coeus_leto.json" "$BASE_JSON"
RUSTC_BOOTSTRAP=1 cargo +1.97.0 rustdoc --offline --locked \
  --manifest-path "$COEUS/Cargo.toml" --target-dir "$ATLAS/target" \
  -p coeus-leto --lib -- -Z unstable-options --output-format json
cargo +1.97.0 semver-checks check-release --baseline-rustdoc "$BASE_JSON" \
  --current-rustdoc "$ATLAS/target/doc/coeus_leto.json" --release-type patch
```

Reproduce the selected memory checks from outside Atlas:

```sh
MIRIFLAGS=-Zmiri-strict-provenance cargo +nightly miri test --offline --locked \
  --manifest-path "$COEUS/Cargo.toml" --target-dir "$ATLAS/target" \
  -p coeus-leto --test sparse_allocations
MIRIFLAGS=-Zmiri-strict-provenance cargo +nightly miri test --offline --locked \
  --manifest-path "$COEUS/Cargo.toml" --target-dir "$ATLAS/target" \
  -p coeus-leto --test leto_ops sparse_contracts::malformed_csr_and_dense_shapes_reject_before_writes
MIRIFLAGS=-Zmiri-strict-provenance cargo +nightly miri test --offline --locked \
  --manifest-path "$COEUS/Cargo.toml" --target-dir "$ATLAS/target" \
  -p coeus-leto --test leto_ops sparse_contracts::zero_dense_columns_and_zero_shared_dimension_are_fully_initialized
```

All external locked package versions and source revisions remain identical
to main. No registry publication, release tag or complete tensor-forward
migration is claimed. Remaining work is provider contract support, dependency
graph co-evolution and rerunning cross-architecture gates once providers compile.

## Provider specification follow-up (2026-10-15)

Leto ADR 0036 proposes checked signed/stored-order borrowing, an explicit
separate-rounding strategy and a validated row dispatch scope. It requires
architectural sign-off before implementation. The generic Coeus adapter also
needs an explicit `Backend::parallel_for` unwind-completion obligation and a
reviewed implementor-contract change; the present Safety text states its
guarantees on normal return. The forward cutover is therefore [arch] work.
The concrete provider proposal is [Leto draft #362](https://github.com/ryancinsight/leto/pull/362).

Provider Miri characterization exposed an inexact numerical fixture:
`powi(53)` is not a guaranteed-exact way to construct `2^53` under
[Rust's documented precision contract](https://doc.rust-lang.org/std/primitive.f64.html#method.powi).
The forward tests now use the exact binary64 literal `9_007_199_254_740_992.0`
and the exact native product `-(EPSILON * EPSILON)` for `-2^-104`.
No input domain, assertion or tolerance changes. Production Rust and external
dependency locks remain identical to the recovered bridge commit
`35e452686785cb409dd648c0565b3f9600e5de8d`.

After this fixture correction, all **293 native tests pass** again and all
**five tensor-forward contract cases pass strict-provenance aarch64 Miri in
both borrow models** (Stacked run `a0ab5695-46f3-4176-9309-ec966d080333`, Tree
run `db9028ea-8606-4b5d-abf5-2ad9bd1dd907`; 94 unrelated tests excluded in each).
Isolation remains enabled. The pre-existing nightly Cargo warning about
`workspace.package.keywords` remains visible. This adds memory evidence for
the existing Coeus forward contracts, not for an unimplemented provider API.

```sh
MIRIFLAGS=-Zmiri-strict-provenance cargo +nightly miri nextest run --offline --locked \
  --manifest-path "$COEUS/Cargo.toml" --target-dir "$ATLAS/target" \
  --config-file "$COEUS/.config/nextest.toml" \
  -p coeus-ops --test ops -E 'test(sparse::forward_contracts::)'
# Repeat with MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-tree-borrows'.
```

## Current-main integration and repeated performance

Main advances during the provider specification to
`dd27214f88a5c23e0e9abc58e014e58b902d67fb` (Moirai 0.9 migration). Merge
`fdd719e48f77f9380454c9d9bcf490b7f3d0a88e` integrates it on the independent
recovery branch. Relevant tensor-forward, scalar, backend and CPU storage
source contracts remain unchanged. The updated canonical lock selects:

- Leto 0.45.2 at `248fe4f63c674f6fca1f89ac19c1f817e67bc417`.
- Eunomia 0.9.1 at `0f4898f86856980b1825805e27fa80f587caf821`.
- Hermes 0.8.2 at `b4d73c94ebce3699d0162ea9196ec6f45fe28af0`.
- Moirai 0.9.2 at `2b189f0a8aaf4a45aab1f2085c2ec748e483b8b3`.

Every external package entry equals current main's lock. A new main export
receives only the identical benchmark and its development dependency; all
**615 production Rust source files** compare by Git blob hash with main.
The candidate's tensor-forward source remains byte-identical to main.

After integration and the final width-dependent regression assertion,
**293 native tests pass** with zero skipped (Nextest run
`b0a44630-9b6e-434f-b1b4-f7ad645ed5bd`), and strict all-target Clippy passes
for both affected packages. All **38 warning-denied bridge doctests pass**
again. These results exercise the updated graph; they do not rely on the
earlier checks against Moirai 0.8.

All **eight selected sparse memory cases pass in each borrow model** with
strict provenance and isolation: Stacked run
`d19abf18-318c-4469-9857-5730b065ff30` (12.434 seconds), Tree run
`2e7d7af9-5960-4ecf-aea8-1eadf9f66525` (18.091 seconds); 285 unrelated tests
are excluded. This combines the allocator/full-write bridge case, malformed
CSR/dense output failure atomicity, empty-output bridge coverage and all five
tensor-forward cases. Neither run receives a Nextest handle marker.

The signed-index test now also checks a positive `i64::MAX` row offset.
When representable, it exceeds `nnz` and returns `Storage`; on a narrower
target, conversion fails with the existing `InvalidNumericInput` category
and its "negative CSR row offset" message. An unrepresentable column returns
`IndexOutOfRange` with `index = usize::MAX`. These error distinctions remain
part of the integration contract. The 32-bit branch is not executed here;
the earlier upstream 32-bit limitations remain explicitly unverified on the
new graph, rather than silently receiving a green result.

The locked API-only patch comparison repeats against current main: **223
checks pass, 30 tool checks skip**. It generates baseline/candidate
`coeus-leto` rustdoc JSON with Rust 1.97.0 and compares those files with
`cargo semver-checks --release-type patch`. Both use the same current external
lock. This establishes the reviewed bridge API comparison, not fresh-source
resolution or registry availability. Formatting and diff checks also pass.

The same 21-pair instrument, unchanged workloads and full dyadic value
assertions repeat on the same Apple M4 Max host. The main bridge control's
paired median ratios are **1.002, 0.998, 1.002, 1.000**, in table order.
The candidate compares its borrowed bridge with the same owned reference:

| Shape / RHS width | Owned reference median (us) | Borrowed median (us) | Paired ratio median [quartiles] |
| --- | ---: | ---: | --- |
| 1024 x 1024 / 1 | 9.446 | 6.808 | 0.718 [0.712, 0.724] |
| 16384 x 16384 / 1 | 271.004 | 241.658 | 0.889 [0.884, 0.908] |
| 1024 x 1024 / 16 | 28.604 | 26.140 | 0.916 [0.907, 0.921] |
| 4096 x 4096 / 64 | 564.638 | 538.221 | 0.968 [0.939, 0.984] |

The optimistic forward comparison still starts with native `usize` provider
indices and includes validation and output allocation. It measures the
prospective direct serial provider against current Coeus Moirai execution:

| Shape / RHS width | Coeus median (us) | Direct provider median (us) | Paired ratio median [quartiles] |
| --- | ---: | ---: | --- |
| 1024 x 1024 / 1 | 6.050 | 7.041 | 1.168 [1.144, 1.195] |
| 16384 x 16384 / 1 | 119.825 | 240.652 | 2.008 [1.964, 2.088] |
| 1024 x 1024 / 16 | 354.820 | 32.125 | 0.090 [0.089, 0.091] |
| 4096 x 4096 / 64 | 1042.539 | 526.525 | 0.494 [0.475, 0.525] |

The host runs unrelated work, and the control measurement overlaps other
verification. Paired samples and the owned control bound interpretation; no
quiet-host or universal throughput claim follows. SpMM's fused-rounding
counterexample remains `-2^-104` versus Coeus +0. Faster SIMD fixture times
cannot waive that observable contract. The unchanged forward production
source independently proves this bridge patch removes no backend scheduling.
