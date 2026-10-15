# Coeus sparse recovery checklist

Active target: **0.13.0**. Owner: Codex. Current integration base:
`f9ebef3462a8c1704d0a4c7daf1770aeb1c04527` (original audit: `cd8e984e`).

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

## General forward cutover [arch] — Execution

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
- [ ] [arch] Complete scalar/layout/initialization regressions, both memory
      models, exact-main performance, API compatibility and consumer resolution.

The provider implementation and backend-contract commit are independently
reviewable. The 0.12.1 bridge evidence below remains historical; it does not
certify the 0.13.0 forward cutover. Preservation #531 stays intact.

Current-main merge `fdd719e4` revalidates 293 native cases, 38 doctests, eight
Miri cases in both models, strict Clippy and the 223-check API comparison
with Moirai 0.9.2 and Leto `248fe4f6`. Repeated paired bridge ratios are
0.718, 0.889, 0.916 and 0.968; the unintegrated serial-provider large SpMV
still costs 2.008 times current Moirai execution. Detailed revision and
measurement evidence remains in the sparse recovery record.
