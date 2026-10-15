# Coeus Gap Audit

Risks nobody is actively fixing. A risk with a planned fix is a backlog item
instead (see `backlog.md`); a closed risk's residue lives in its ADR or PR,
not here.

## 2026-10-15 — Known-width adaptation cost [patch]

The pre-specialization source passes every actual hosted gate. Inlining alone
brings small-vector ratios near parity but leaves a 1.065 large-Moirai ratio.
Checked fixed-width metadata now removes the runtime width representation for
vectors through the same provider kernel. Provider native/strict/Stacked checks
and consumer strict Clippy pass; complete current-source verification and the
unchanged paired workloads remain the next action.

## 2026-10-15 — Borrowed CSR source contracts close

The accepted provider seam and Coeus adapter remove copied CSR reconstruction
and duplicate forward arithmetic while preserving signed-index checks,
stored order, separate rounding, fallible storage, full initialization,
exact typed diagnostics and backend dispatch. ADRs 0036/0081 record ownership
and the approved backend unwind obligation. [The current verification record](docs/verification/sparse-forward-cutover.md)
separates each exact-source result from the historical bridge audit.

The final current-main graph passes 1272 native cases, strict Clippy, 159
doctests, warning-denied documentation and three explicit minor API comparisons.
Provider native/properties, both strict memory models, allocation guards and
real threshold-crossing callbacks cover the new boundary. Complete Coeus
current-head hosted receipts remain a pre-integration check in the backlog.

Residuals have a defined next action in
[COEUS-SPARSE-VIEW-FORWARD](backlog.md#coeus-sparse-view-forward):

- Final-lock SpMV medians cost 2–8% more; Moirai SpMM takes 9–32% of main time.
  Review the narrow cost with all timing runs visible; do not infer universal
  performance preservation from retained parallel dispatch.
- Fresh Leto/Hephaestus/Apollo review consumers resolve one source per provider;
  fresh Coeus still rejects unmerged Apollo ^0.29.2. Default/registry evidence
  requires provider-first authorized integration, not another local lock seed.
- Provider-only ARM32 executes eight cases in each strict model. Full Coeus
  ARM32 stops before tests at Moirai's deque-alignment assertion; i686 stops at
  Eunomia's binary64 alignment assertion. Neither is a full-consumer pass.
- Linux's real Moirai topology `opendir` is unsupported under Miri isolation.
  The full unchanged case executes in both macOS jobs; Linux additionally runs
  all twelve other cases. Native limits, assertions, isolation and leak checks
  remain intact.

## CUDA context creation failure remains unclassified

Consumer run `87532469-ce04-4b9a-8225-f53e7ea000a2` fails in `cuCtxCreate_v2`
with status 999 after releasing a successful temporary device. The isolated
reproduction passes, so no fix is confirmed; driver health inspection found
no matching Windows driver event.
Re-open trigger: a captured driver fault or a reproducible failing lifecycle.

## Slop pattern: per-element coordinate buffer in flat-index decode loops

A kernel walking a flat output index and decoding it into coordinates through
a `vec![0usize; ndim]` allocated *inside* the loop is waste: the buffer never
outlives the iteration that fills it. Fix: fuse the decode into the
accumulation instead of allocating per element.
Check (no lint covers this yet): `rg --multiline 'for \w+ in 0\.\.[^\n]*\{[^}]{0,200}?vec!\[0usize;'`.
Re-open trigger: the pattern recurs outside the five sites already fixed under the closed COEUS-OPS-INDEX-DECODE-ALLOC-001 — promote to a Clippy lint if so.

## Slop pattern: stale local `*.pyd` shadows the installed extension

pytest prepends the test directory to `sys.path`, so a leftover
`crates/coeus-python/tests/pycoeus*.pyd` build artifact silently overrides the
freshly `maturin develop`-installed module, pinning an out-of-date binary and
producing spurious `AttributeError`s for newly-added bindings.
Mitigation: keep built extensions out of `tests/`; the canonical module is the
site-packages install.
Re-open trigger: the artifact reappears (add a pre-test cleanup or a `.gitignore`/CI check if it recurs).

## G-049 residual: `gammaln` backward blocked on upstream `digamma`

`d/dx lgamma(x) = digamma(x)`, and Eunomia (pinned `0.8.0`) does not expose
`digamma` yet — verified absent from the workspace. Python raises
`NotImplementedError` for grad-tracked `gammaln` inputs rather than a fake or
zero gradient, which is correct pending the upstream capability.
Owning item: none in this repo (upstream Eunomia capability gap).
Re-open trigger: Eunomia exposes `digamma`, then wire the CPU/Leto backward path.
