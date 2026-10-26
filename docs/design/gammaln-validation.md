# Gamma-log validation ledger

Baseline source: Coeus 57f7c4596884949e15815659544096b62c570d85.
A CPython 3.13 extension built with Maturin from the unchanged source rejects
both tracked aliases with `NotImplementedError: gammaln backward requires
digamma support`. The two new weighted-gradient acceptance cases fail at
invocation for that exact reason. Untracked inputs still return the existing
libm forward values. The raw Cargo cdylib attempt failed macOS Python linking;
the supported Maturin build succeeded without modifying linkage source.

Scalar provider baseline: Eunomia 23c6047 has no digamma method; its new
acceptance tests fail to compile on the missing API. The additive implementation
passes the complete 217-case native suite, including six digamma tests with
analytic anchors, recurrence/reflection, exact-binary high-precision fixtures,
real domain boundaries and exhaustive reduced-storage encoding coverage.

Framework witnesses: PyTorch 2.14.1, SciPy 1.18.1, JAX 0.12.0.1 (x64 enabled).
Their special-value classifications agree except JAX maps both signed zeros
to negative infinity; PyTorch and SciPy preserve the -1/x signed limit. Finite
values near negative poles can also differ, as the exact counterexample below
records. The provider follows the signed Laurent limit and exact-binary
high-precision references.

Leto provider acceptance passes 1,204 native cases, strict all-feature Clippy
and docs on canonical Git dependencies. The four initial Coeus native gamma
cases pass, including scalar/empty forward and backward, slices, transposes,
broadcasts, non-unit seeds and repeated/shared accumulation. Final canonical
Coeus diagnostics, native and Python acceptance, independent binary discrepancy
reproduction and hosted checks remain. Hardware remains partial: no
Hephaestus digamma kernel or accelerator gradient validation exists in this
slice. Registry/source publication remains held; only docs deployment is
approved. Local overlay checks do not prove a fresh canonical consumer resolves
unpublished provider versions; canonical lock receipts must be validated after
provider candidate delivery, before any downstream integration claim.

Exact-binary near-pole discrepancy: at nextafter(-1, 0), mpmath (100 decimal
digits) and JAX return -9007199254740992 for digamma, matched by Coeus.
SciPy 1.18.1 and PyTorch 2.14.1 return -5545090608933970. The acceptance
keeps strict high-precision coverage for that input and records the exact
framework counterexample separately; no larger epsilon masks it. The other
adjacent side (nextafter(-1, -infinity)) agrees across those references.

Independent review also exposed the existing CPU rank-zero unary/binary/assign
adapter gap. A scalar tracked input reproduced the rank rejection. Shared
provider adapters now admit rank zero, and native and both-alias Python tests
cover scalar and empty tensors. Final rerun remains required after this repair.

Final candidate dependency identities: Eunomia ebb1923 (0.11.0), Leto 93e2a8c
(0.47.0), and the unchanged Apollo PR #696 head b62b9c7e. The committed lock
uses canonical Git sources, with no path patch. Locked offline fetch succeeds
with locally available candidate commit objects; remote fresh-consumer delivery
remains pending branch publication. Current Moirai is 4bb7d2d (0.11.0),
Hephaestus 13dad9cd, Hermes f9b6322e and Mnemosyne b40870ea.
