# Coeus 0.15.0 release checklist

Target: Rust 0.15.0 ([minor], [ADR 0087](docs/adr/0087-physical-quantity-autodiff.md)). The released CPU cohort comes from `0f69d4a812e08a06494678c5382171bdf287548d`.

- [x] [minor] The [workspace lock check](https://github.com/ryancinsight/Coeus/actions/runs/45060488487), [all six native shards across both feature modes](https://github.com/ryancinsight/Coeus/actions/runs/45060488325), [backend parity](https://github.com/ryancinsight/Coeus/actions/runs/45060487888) and [release SemVer check against 0.14.0](https://github.com/ryancinsight/Coeus/actions/runs/45060488822/job/137327163352) passed on the source revision. [PR #686](https://github.com/ryancinsight/Coeus/pull/686) had already passed the protected checks in both feature modes on `fee15594`. The merge preserved that tree.
- [x] [minor] The [read-only planner](https://github.com/ryancinsight/Coeus/actions/runs/45060488822/job/137325065274) selected the eight packages below. The [initial upload](https://github.com/ryancinsight/Coeus/actions/runs/45060488822/job/137328612002) published Core and received HTTP 403 for CPU. The existing Cargo credential published CPU from the exact release revision. The [second attempt](https://github.com/ryancinsight/Coeus/actions/runs/45060488822/job/137331707512) completed publication.
- [x] [patch] For each package below, the downloaded archive matched its unyanked crates.io index checksum. Its `.cargo_vcs_info.json` identifies the source revision above and the corresponding `crates/<package>` directory. All eight [GitHub release tags](https://github.com/ryancinsight/Coeus/releases) resolve to the source revision.
- [x] [minor] A clean Rust 1.95 consumer pinned all eight registry packages to 0.15.0. Three minimum/maximum gradient tests passed with physical units disabled. With the feature enabled, the same three tests and all eight [physical-unit tests](crates/coeus-autograd/tests/physical_units.rs) passed. The 197 registry dependencies included one identity each for Aequitas, Eunomia, Leto, Leto Ops, Apollo FFT, Moirai Core and Mnemosyne Memory. Every cohort package matched its recorded checksum.
- [x] [patch] Two independent Rust 1.95 consumers resolved all eight archive checksums directly from crates.io. Both computed 0.002 J for work and a gradient of 0.001 J/mm. With physical units enabled, the canonical work gradient was 1 N, the affine Fahrenheit slope was 5/9 K/°F, and the pressure ramp slope was 4,000 Pa/s.
- [x] [patch] In a separate Core-only Rust 1.95 consumer, all six offsets of a 2×3 layout matched `3r + c`. Splitting a column preserved element count and physical indices, and an out-of-range split was rejected.

| Published 0.15.0 archive | SHA-256 |
| --- | --- |
| [coeus-core](https://crates.io/crates/coeus-core/0.15.0) | `4a4a6b60854d1e175acae149ee6b050767784ea73718d1a387e6be9e16ea6c8f` |
| [coeus-cpu](https://crates.io/crates/coeus-cpu/0.15.0) | `2874af5659638693c67db3f637d392b0fb13b98b8e7530dafcced30c36f82b9d` |
| [coeus-tensor](https://crates.io/crates/coeus-tensor/0.15.0) | `3aa082dc4b663c38ed8f00d3817cbe8526e11c39bba96d60b50398837ede14e2` |
| [coeus-sparse](https://crates.io/crates/coeus-sparse/0.15.0) | `960bcfa98d117c18753515f539c03d677c105e9569ed8774d2cf1d74d82bca38` |
| [coeus-ops](https://crates.io/crates/coeus-ops/0.15.0) | `c15dffe83781389ed602c38a977c2f4c695705a5c9bb65a4a11a5d91cd19a54b` |
| [coeus-autograd](https://crates.io/crates/coeus-autograd/0.15.0) | `c0846877604bc81b0b26279fc48eeb8ea3aae201388db6e88e50eab059c1a23e` |
| [coeus-nn](https://crates.io/crates/coeus-nn/0.15.0) | `fa954647ed6d853cb907944dc196c4d53e35bfb7eae97046a6d701002edb8962` |
| [coeus-optim](https://crates.io/crates/coeus-optim/0.15.0) | `6e43f696c8bb8f6dcc169a2e605638df8ad0242de23cd17a94c56e778ad077d8` |
