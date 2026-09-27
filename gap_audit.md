# Coeus Gap Audit

Risks nobody is actively fixing. A risk with a planned fix is a backlog item
instead (see `backlog.md`); a closed risk's residue lives in its ADR or PR,
not here.

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
