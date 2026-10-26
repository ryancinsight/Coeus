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
provider candidates and their precision and layout boundaries. Those executed
local commands bind to the listed source heads. Eunomia's final reviewed head
`009187de0bfce7dd126324f4bd7199c81bb5c02a` merges as
`5ae27c75c153304f4f9fb3a0cbaf6320a4c59098` with an identical tree. Published
eunomia/eunomia-derive 0.11.0 archives carry that VCS identity; all 103
non-generated payloads match Git. Fresh registry-only consumer metadata resolves
both 0.11.0 packages without Git/path/patch overrides; this is resolution,
not a claim of a new local consumer compilation.

Coeus local CPU acceptance passes 254 autograd cases, 42 CPU adapter cases
and 20 selected Python gamma-log cases against the default-feature CPython
3.13 wheel. Strict all-feature Clippy and warning-denied CPU Rustdoc pass.
Coverage includes both aliases, non-unit seeds, repeated/shared accumulation,
transposes, offset slices, broadcasts, scalar/empty forward and backward,
native f32/f64 and reduced-storage boundaries. Every finite high-precision
fixture keeps its derived absolute-plus-relative provider/VJP bound. These
executed local results and the independent extension witness bind to Coeus
`068a5ff0ba74db9fde4d8bbe7c90f493222a0480`; final updated-head hosted checks
remain separate gates.

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
separate source-independent binary witness verifies those exact bits, CPU
float64, the pinned versions, 100/150/200-digit convergence and an independent
Laurent-series remainder bound. The reference begins
`-9007199254740991.5772156649015325669598421270470114999265336545109873`;
SciPy/PyTorch's relative error is
`0.38437127323287764598911598737341846658`. Coeus/JAX return the correctly
rounded unit result. The complete output/error receipt binds to extension
SHA-256 `5bf8536c4e32c915d51c2235c980570cd8ec33ca5417efbff0d3a25d825949d9`
and Coeus `068a5ff0`. Weighted products retain their propagated budgets;
they are not all claimed correctly rounded.

The updated lock uses published Eunomia merge `5ae27c75`, reviewed remote
Leto `bd0696cd` and the unchanged
Apollo PR #696 head b62b9c7e through canonical Git sources, with no path patch.
Locked offline fetch succeeds with locally available candidate commit objects;
this does not establish remote fresh-consumer delivery. Current Moirai is
4bb7d2d (0.11.0), Hephaestus 13dad9cd, Hermes f9b6322e and Mnemosyne
b40870ea. Current Apollo main cannot fresh-resolve Moirai 0.11, so its existing
owned prerequisite must be delivered through the normal review path. The
provider lock refresh changes only the two Eunomia and two Leto Git references;
it does not change runtime versions or relax the dependency policy. Independent
review clears Leto `bd0696cd`, including this source-only provider reference
change and its read-only package plan. That plan verifies `leto`, but explicitly
waits on absent `moirai-runtime ^0.11.0` for `leto-ops`.

At Coeus `068a5ff0`, hosted Tests shard 3 passes 423 cases and fails
`policy::dependency_policy::resolved_normal_dependency_tree_excludes_non_ssot_runtime_or_replacement_crates`.
The normal tree contains `tokio 1.54.1 -> quinn 0.11.12 -> moirai-quic 0.1.0
-> moirai-runtime 0.11.0`. The native owner holds this repair; shards 1/2
are cancelled by fail-fast and do not count as passed. Apollo #696 remains at
`b62b9c7e` with version-bump and license-policy failures. No owner takeover,
runtime downgrade, whitelist exemption or license-policy change is introduced.

Locked strict-minor API comparisons pass for seven selected Leto/Coeus
packages (196 checks each, 57 inapplicable); ordinary fresh unmodified API
resolution remains a distinct release gate. Updated-head PR CI adds the same
canonical read-only package plan and runs the unchanged 20-case Python gamma-log
selection with pinned CPU frameworks and a newly built default-feature wheel.
These new jobs must actually finish; their presence alone is not acceptance.

Hosted exact-head checks, remaining release gates, provider delivery and
fresh-consumer resolution remain required. Existing authorization covers
qualifying integration and publication after those gates. Documentation deploys
are authorized; app/service deploys are not. Hardware is partial: no Hephaestus
digamma kernel or accelerator gradient validation exists in this slice, and no
full-parity 1.0 claim follows from CPU progress.
