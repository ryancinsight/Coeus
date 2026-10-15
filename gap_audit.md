# Coeus Gap Audit

Risks nobody is actively fixing. A risk with a planned fix is a backlog item
instead (see `backlog.md`); a closed risk's residue lives in its ADR or PR,
not here.

Sparse recovery audit (2026-10-15): the signed-index, stored-order, rounding
and scheduling gaps have an explicit provider-first action in
[COEUS-SPARSE-VIEW-FORWARD](backlog.md#coeus-sparse-view-forward).
The [verification record](docs/verification/sparse-recovery.md) distinguishes
the compatible bridge recovery from the unresolved tensor forward cutover.
Accepted Leto ADR 0036 implements that provider seam; ADR 0081 records the
approved backend completion-on-unwind contract. Exact numerical fixtures use
representable literals/native products rather than unspecified-precision
power calls; five tensor-forward contract cases now also pass Miri.

## 2026-10-15 — Approved provider and backend execution boundary

- ADR 0036 is approved by Ryan Clanton in the task conversation; consumer ADR
  0081 records the unchanged-signature, tightened unwind obligation for unsafe
  backend implementations. The workspace target becomes 0.13.0.
- Real Sequential/Moirai panic regressions pass on the existing graph (Moirai
  crosses 4096 rows). Audit finds only inline CUDA delegations beyond those
  two CPU implementations; the existing test counter also executes inline.
- Leto main advances to the Eunomia 0.10 cohort during implementation. Follow
  the active provider dependency owners before the consumer graph is refreshed.
- Next: verified provider binding, full integration and performance gates.

## 2026-10-15 — Recovered forward path and coherent review graph

- The provider owns checked signed decoding, stored-order arithmetic and row
  writes. Coeus reuses the checked descriptor and retains fallible tensor
  storage/allocation, original rejection diagnostics and backend scheduling.
- 151 real CPU integration cases pass. Single-column local accumulation and
  removal of duplicate validation resolve the measured multi-fold SpMV loss;
  paired medians now lie near parity and SpMM improves on all measured widths.
- Hephaestus #497 and Apollo #674 add compatible 0.46/0.47 intervals. Cargo
  regenerates/validates the complete review lock with ten canonical provider
  identities and no excess source, path patch or manifest revision pin.
- Remaining: exact candidate full native, strict/API/memory/hosted evidence
  and provider-first default/registry availability. See the current
  [forward record](docs/verification/sparse-forward-cutover.md).

## 2026-10-15: fresh Git resolution exceeds current provider version bounds

An unlocked external consumer of main `cd8e984e` fails before compilation:
Coeus requests Eunomia `^0.9.0`, while its default Git branch provides 0.10.0.
The canonical lock still builds, and this recovery retains every external
locked source/version. Impact: fresh consumers and the normal source-based
SemVer build cannot resolve; no release-readiness claim follows from locked
tests or the API-only comparison. Owner: provider dependency co-evolution.
Re-open trigger: version bounds/source identities advance together, followed
by a fresh consumer without local path/patch overrides.

## 2026-10-15: 32-bit memory checks stop in locked provider assertions

i686 Miri fails Eunomia's `align_of::<F64>() == 8` assertion (target alignment
is 4); ARMv7 Miri fails Moirai's Chase–Lev deque alignment >=64 assertion.
Both fail before Coeus compilation. Impact: neither run provides 32-bit
memory-safety evidence; the borrowed `usize` descriptor adds no index cast.
Owner: respective providers. Re-open trigger: a compatible provider graph
compiles on those targets and the selected Miri checks execute.

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
