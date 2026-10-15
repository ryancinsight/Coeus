# Coeus sparse recovery checklist

Active target: **0.13.0** [arch]. Owner: Codex. Current integration base:
`f0d01021fdad36ddb1303d811cd51a4ceedb6979`.

## Borrowed CSR forward recovery — Closure / review

- [x] [patch] Audit #531, current main, actual provider APIs and source ownership.
- [x] [arch] Obtain approval for ADR 0036 and downstream ADR 0081 on 2026-10-15.
- [x] [arch] Implement checked signed/stored-order provider borrowing, arithmetic
      policy and synchronous row scope; preserve one authoritative recurrence.
- [x] [arch] Audit every Backend implementation and execute real threshold-crossing
      return/unwind completion, unique admission and capture-release regressions.
- [x] [arch] Remove both tensor forward loops; preserve public Result signatures,
      scalar bounds, exact typed errors, fallible storage and complete initialization.
- [x] [patch] Eliminate all three bridge CSR-copy allocations and compare real values.
- [x] [patch] Cover all sixteen scalar families, 4097x257 full output, stored order,
      separate rounding, negative/oversized indices, offsets, noncontiguous rejection,
      empty dimensions, failure atomicity and exact multi-fault priority.
- [x] [minor] Deliver independent compatible Leto #362, Hephaestus #497 and Apollo #674
      review sources with one provider identity each and intact current-main changes.
- [x] [patch] Verify checked fixed widths, fuse row checks with preserved error
      precedence, retain exact backend row dispatch and rerun complete source gates.
- [x] [patch] Pass 1272 native tests (eight existing ignored), strict workspace Clippy,
      159 doctests (two existing ignored), warning-denied docs, three explicit minor
      API comparisons, both 13-case memory models and both provider ARM32 selections.
- [x] [patch] Retain all workloads/oracles and run 21-pair current-main timing;
      record complete measured provenance, including the remaining SpMV cost.
- [x] [patch] Execute fresh final-provider products; independently reproduce the
      current unseeded Coeus availability failure before compilation.
- [x] [patch] Synchronize versions, changelog, design, evidence and boards; keep #531 intact.
- [ ] [patch] Inspect final hosted checks before any separately authorized integration.
- [ ] [minor] Integrate/publish compatible providers in order, refresh defaults and
      verify fresh registry consumers only within separate release authorization.

The independently reviewable runtime is `2135975e`, with Leto `b7670636`.
The final timing repeat observes about 2–3% median SpMV overhead in three cases,
near-parity large sequential SpMV and Moirai SpMM ratios 0.093–0.299. Parallel
row dispatch remains intact. No universal non-regression guarantee follows.
The initial macOS TCP setup failure remains classified as reproduction-open;
the complete unchanged rerun passes. Architecture, Linux-Miri and availability
limits are explicit in [the current record](docs/verification/sparse-forward-cutover.md).

The earlier unsigned bridge audit and its 293-test/38-doctest evidence remain
historical in [the recovery record](docs/verification/sparse-recovery.md); they
are superseded by the complete 0.13.0 acceptance scope above. No merge or release
is part of the delivered implementation.
