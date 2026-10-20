# Current Hephaestus 0.25 source and graph renewal

[patch] The unpublished Coeus 0.13 cohort requires all five Hephaestus core and
backend packages at ^0.25.0. Current provider main is
36965bf2d775f09f4e5a76daae57d2a64b145e1c (PR #506); all five public archives
match its source. The prior ^0.24 interval excludes that default, so preserving
an old lock alone does not establish fresh compatibility.

## Preserved design and source

Build on the independently accepted [0.24 reconciliation](hephaestus-024-reconciliation.md)
tracked by [Coeus PR #611](https://github.com/ryancinsight/Coeus/pull/611).
The external review packet binds its exact private head, tree and receipt.
Repository citations use public PRs or repository documents; keep private
candidate identities in revision-scoped external evidence, not landed-history citations.
Actual #613 and #614 merge histories remain ancestors. Keep the owner's exact
CUDA real/stub and WGPU FusionScalar<L> bounds, every existing numerical/NN
and Python-linkage repair, private 0.13 versions, strict Moirai floors and
Leto/LetoOps >=0.46.3,<0.48. No product Rust implementation is changed.

The provider adds CUDA/ROCm embedding implementations, pipeline keys and Metal
forwarding. Its 205 core Rust files, FusionScalar and EmbeddingOps are unchanged;
Coeus does not directly bind EmbeddingOps. This supports a narrow source-range
renewal but does not prove compatibility, runtime correctness or performance.
No provider-owned embedding code is duplicated. Athena's separate 0.25 renewal,
#500 takeovers 2-3 and provider hardware review remain with their owners.

## Canonical graph and verification

Resolve the normal current Git graph and verify ordinary registry-normalized
packages with no path/branch/rev workaround. Preserve published repaired native
0.46.3 at 5051cd9577650a5712aa6a7effcd8955c9d8bcc7. Do not copy #614's defective
0.46.2 lock or the provider workspace's older native lock.

Run material native/backend, scalar/layout/seed, strict lint/docs/fmt, Rust 1.95,
conformance, normal package and API checks at the resulting exact head. Existing
0.24 tests and unchanged core code do not accept the new graph. The six proven
historical 0.10 registry baselines qualify for the existing pinned release
policy's explicit notice/skip handling; they are not mandatory repair blockers.
Retain the exact evidence and gate every buildable sibling through that policy.
Additional locked owner-source diagnostics are supplemental only and cannot
replace required registry comparisons or current ROCm API compilation on Linux.

## Publication and platform boundaries

Use the already authorized existing Cargo login or existing publishing workflow
without copying credentials, changing settings or creating persistent access.
Prepare a complete dependency-ordered cohort dry run and a partial-failure
recovery plan before any merge can trigger a core-only shipment. A dry run
verifies packaging/order, not upload authority. No push, dispatch, environment
switch, merge or publication occurs during this local preparation.

CUDA compilation and test collection do not prove device execution; all three
storage unit tests and the full suite require a real supported NVIDIA device.
The manual required-device jobs run full unfiltered CUDA and ROCm suites.
ROCm current feature/API extraction requires Linux; physical ROCm contracts require AMD
hardware. Identify the supported paths without changing coverage or limits.

Common PyO3 0.23-to-0.30, future Leto 0.47 stored-order policy, complete wheel/
platform/cohort and absolute/Harmonia performance acceptance stay separate.
Coeus Miri remains DEFERRED AND UNVERIFIED. The prior denied push stays paused.
