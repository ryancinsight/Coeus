# Coeus sparse recovery checklist

Active target: **0.13.0**. Owner: Codex. Current integration base:
`f0d01021fdad36ddb1303d811cd51a4ceedb6979` (original audit: `cd8e984e`).

## Borrowed dispatch recovery [patch]

- [x] [patch] Audit PR #531, current main and locked/current Leto contracts.
- [x] [patch] Reproduce three CSR-copy allocations on the unchanged bridge.
- [x] [patch] Replace owned reconstruction with the validated borrowed view.
- [x] [patch] Keep public descriptors, signatures, errors and arithmetic unchanged.
- [x] [patch] Specify ordering, rounding, signed indices, layouts and empty outputs.
- [x] [patch] Run native, formatting, strict Clippy and selected memory checks;
      record the unavailable 32-bit checks and fresh-consumer failure.
- [x] [patch] Compare performance with exported current main.
- [x] [patch] Synchronize evidence and prepare the independent reviewable change.

Local evidence: 293 native tests and 38 doctests pass; strict Clippy and
formatting pass; eight selected Miri cases pass in both borrow models; the locked API comparison
passes 223 checks (30 tool skips). Fresh Git resolution and attempted 32-bit
memory checks remain limited by existing provider failures. The bridge is a
complete compatible increment; the tensor forward migration remains open.

## General forward cutover [arch] — Closure / review

- [x] [arch] Ryan Clanton approves Leto ADR 0036 and its downstream backend
      safety clarification on 2026-10-15; ADR 0081 records the consumer slice.
- [x] [arch] Audit real backend implementations and execute threshold-crossing
      panic-completion regressions before using lifetime-erased provider tasks.
- [x] [arch] Bind the verified provider capability and remove the two tensor
      forward loops while preserving existing typed validation/allocation errors.
- [x] [patch] Verify all scalar families, wide output and multiple-fault
      precedence; measure the checked adapter against byte-identical main code.
- [x] [patch] Resolve the actual provider review graph with one source each;
      deliver compatible Hephaestus #497 and Apollo #674 independently.
- [x] [arch] Complete scalar/layout/initialization regressions, both memory
      models, complete native/strict/doc gates and current-main performance.
- [x] [patch] Record exact current-main native/strict/doc/API and paired timing
      evidence, fresh review-source consumers and successful hosted memory scopes.
- [ ] [patch] Before separately authorized integration, inspect all final hosted
      receipts and the measured narrow-product cost; refresh defaults/registry
      consumers in provider order.

The source increment is independently reviewable against main `f0d01021`.
Final local acceptance: 1272 native passes (`b08e612b`, eight existing ignored),
strict all-target/workspace Clippy, 159 doctests (two existing ignored),
warning-denied docs, and three explicit minor API comparisons (196 checks each,
57 tool skips each). The final timing repeat records 2–8% median SpMV overhead
and Moirai SpMM ratios 0.09–0.32. Hosted/availability limits remain explicit in
the linked record and PR checks. The 0.12.1 bridge evidence below remains historical; it does not
certify the 0.13.0 forward cutover. Preservation #531 stays intact.

Current-main merge `fdd719e4` revalidates 293 native cases, 38 doctests, eight
Miri cases in both models, strict Clippy and the 223-check API comparison
with Moirai 0.9.2 and Leto `248fe4f6`. Repeated paired bridge ratios are
0.718, 0.889, 0.916 and 0.968; the unintegrated serial-provider large SpMV
still costs 2.008 times current Moirai execution. Detailed revision and
measurement evidence remains in the sparse recovery record.


## Known-width callback specialization [patch] — Closure optimization

- [x] Preserve one provider recurrence and select checked FixedWidth<1> for vectors.
- [x] Keep dynamic matrix widths, scalar bounds, validation and backend dispatch.
- [x] Bind provider `efa8ba2a` and pass all-workspace/all-target strict Clippy.
- [ ] Repeat the complete current-head native/doc/API/memory/ARM32/performance gates.
