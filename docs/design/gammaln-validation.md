# Gamma-log validation ledger

Baseline source: Coeus 57f7c4596884949e15815659544096b62c570d85.
A CPython 3.13 extension built with Maturin from the unchanged source rejects
both tracked aliases with `NotImplementedError: gammaln backward requires
digamma support`. The two new weighted-gradient acceptance cases fail at
invocation for that exact reason. Untracked inputs still return the existing
libm forward values. The raw Cargo cdylib attempt failed macOS Python linking;
the supported Maturin build succeeds without linkage source changes.

Scalar provider baseline: Eunomia 23c6047 has no digamma method; its new
acceptance tests fail to compile on the missing API. Eunomia ebb1923 (0.11.0)
passes all 228 native cases, six feature configurations, strict all-feature
Clippy, doctests and warning-denied Rustdoc. Analytic anchors, recurrence and
reflection, exact-binary high-precision fixtures, real domain boundaries and
exhaustive reduced-storage encoding checks cover the scalar contract. The
API comparison against 23c6047 reports 196 passing checks, 57 inapplicable
checks and no required SemVer update under the declared minor change.

Leto 93e2a8c (0.47.0) passes 1,204 native cases, strict all-feature Clippy,
33 doctests and warning-denied Rustdoc on canonical Git dependencies. One
pre-existing doctest remains ignored. Independent source review clears both
provider candidates and their precision and layout boundaries.

Coeus local CPU acceptance passes 254 autograd cases, 42 CPU adapter cases
and 20 selected Python gamma-log cases against the default-feature CPython
3.13 wheel. Strict all-feature Clippy and warning-denied CPU Rustdoc pass.
Coverage includes both aliases, non-unit seeds, repeated/shared accumulation,
transposes, offset slices, broadcasts, scalar/empty forward and backward,
native f32/f64 and reduced-storage boundaries. Every finite high-precision
fixture keeps its derived absolute-plus-relative provider/VJP bound.

Independent consumer review exposes inherited rank-zero exclusions in unary,
binary, assignment and reshape adapters. A tracked scalar first reproduces
the rank rejection. Shared adapters now admit zero ranks through Leto.
The first full Python run then fails both weighted scalar alias cases at
reshape during broadcast-gradient reduction. The shared reshape repair adds
zero source and target ranks, an offset-preserving round trip and a weighted
scalar native regression. All 20 selected Python cases pass after this repair.
Independent source review includes forward, multiply, accumulation and reshape.

Framework witnesses use PyTorch 2.14.1, SciPy 1.18.1 and JAX 0.12.0.1
(x64 enabled), all on CPU float64. Their special-value classifications agree
except JAX maps both signed zeros to negative infinity; PyTorch and SciPy
preserve the -1/x signed limit. The provider follows that Laurent limit.

The observed exact-binary counterexample is nextafter(-1, 0), bits
0xbfefffffffffffff. Mpmath at 100 digits and JAX round digamma to
-9007199254740992, matched by Coeus; SciPy and PyTorch return
-5545090608933970. The other adjacent side, nextafter(-1, -infinity), agrees.
Acceptance retains the input and strict high-precision bound, records the
framework output separately and never expands epsilon to emulate it. A
separate source-independent binary witness must establish exact input bits,
dtype/device/version pins, 100/150/200-digit convergence, a Laurent-series
cross-check and full output/error metrics before calling this a verified
upstream numerical discrepancy. Its receipt belongs to the work-package
evidence with the tested extension hash and exact source identity.

The committed lock uses Eunomia ebb1923, Leto 93e2a8c and the unchanged
Apollo PR #696 head b62b9c7e through canonical Git sources, with no path patch.
Locked offline fetch succeeds with locally available candidate commit objects;
this does not establish remote fresh-consumer delivery. Current Moirai is
4bb7d2d (0.11.0), Hephaestus 13dad9cd, Hermes f9b6322e and Mnemosyne
b40870ea. Current Apollo main cannot fresh-resolve Moirai 0.11, so its existing
owned prerequisite must be delivered through the normal review path.

Hosted exact-head checks, remaining API/release checks, remote provider delivery
and fresh-consumer resolution remain gates. Hardware is partial: no Hephaestus
digamma kernel or accelerator gradient validation exists in this slice. Source
publication remains held; only docs deployment is approved. Main version-bump
workflows can auto-publish, so these local results do not authorize integration,
registry publication or a full-parity 1.0 claim.
