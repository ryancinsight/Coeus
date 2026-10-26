## Real gamma-log gradient provider slice — October 26, 2026 [minor]

Current source confirms the missing scalar/API chain. The compiled Coeus
baseline rejects both tracked Python aliases at invocation. The additive
provider-first CPU implementation is under validation. CPU results cannot
establish device parity. No scalar formula or device-to-host fallback exists
in the Coeus consumer. Eunomia 0.11.0 is published with verified source receipts.
Fresh canonical resolution, final-head review/CI and provider release gates
remain required under the existing conditional publication authorization.

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

## G-049: provider-owned gamma-log backward [minor]

At base `57f7c459`, the actual lock uses Eunomia 0.10 (`f263446f`), Leto
0.46.4 (`6778a0cc`) and Hephaestus 0.27 (`6f4dd9b9`). The old 0.8 wording
was stale. A compiled base wheel rejects both tracked aliases with the missing
digamma error; non-gradient forward works. The new provider chain adds real
Eunomia digamma, Leto DigammaOp, optional CPU dispatch and the derivative
`grad_out * digamma(input)`. Local acceptance passes 254 autograd cases,
42 CPU adapter cases and 20 Python alias cases, including weighted scalar and
empty backward. Strict Clippy/docs pass and independent source review clears
the implementation. Locked strict-minor API checks pass for the selected
consumer packages; final-head hosted/release gates remain open. The current
lock advances only Eunomia to published merge `5ae27c75` and Leto to reviewed
remote `bd0696cd`. Native tests expose the existing forbidden Tokio dependency
through Quinn/Moirai QUIC. The native owner holds that prerequisite; do not
relax the dependency policy. Leto's `leto-ops` packaging also waits for Moirai
0.11 to reach the registry.

Owner: task-4, `feat/gammaln-autograd`. Design and remaining acceptance:
[contract](docs/design/gammaln-autograd.md) and
[validation](docs/design/gammaln-validation.md). Current Apollo's separate
Moirai 0.11 admission prerequisite is owned by existing PR #696, whose version
bump and license checks remain red. No license policy change is part of this
slice. Existing publication authorization applies after final gates. Device parity
remains partial until Hephaestus implements and validates digamma; no device
fallback or complete-1.0 claim follows from CPU progress.
