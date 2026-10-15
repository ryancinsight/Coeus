# Coeus Gap Audit

Risks nobody is actively fixing. A risk with a planned fix is a backlog item
instead (see `backlog.md`); a closed risk's residue lives in its ADR or PR,
not here.

## 2026-10-15 — Borrowed CSR source contracts close

The accepted provider seam removes CSR copies and duplicated forward math,
retaining signed portable addressing, typed diagnostics, stored order, separate
rounding, fallible initialized output and backend row dispatch. Checked fixed
widths and fused row checks pass the complete final local acceptance scope:
1272 consumer/1202 provider native cases, strict diagnostics, docs/API, both
memory models and ten provider ARM32 cases per model. Current evidence lives
in [the forward record](docs/verification/sparse-forward-cutover.md).

Residuals and concrete actions remain in
[COEUS-SPARSE-VIEW-FORWARD](backlog.md#coeus-sparse-view-forward): final tested
SpMV cost about 2–3% in three cases; provider-first default/registry availability;
full-consumer ARM32/i686 upstream alignment limits; Linux Moirai topology under
Miri isolation; final hosted receipts before integration. The unclassified
macOS socket setup observation has its own
[reproduction item](backlog.md#coeus-tcp-half-close-witness). No safety rule,
workload, oracle or native timeout is weakened to close this source increment.

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
