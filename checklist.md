# Coeus sparse recovery checklist

Active target: **0.12.1**. Owner: Codex. Current integration base:
`dd27214f88a5c23e0e9abc58e014e58b902d67fb` (original audit: `cd8e984e`).

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

## General forward cutover [arch]

- [ ] [arch] Obtain ADR sign-off and provider support for the contracts in
      [COEUS-SPARSE-VIEW-FORWARD](backlog.md#coeus-sparse-view-forward).
      The 0.12.1 bridge recovery does not close this item.

Provider specification: Leto ADR 0036 is proposed. No forward implementation
checklist item opens before its decision and the downstream backend unwind
contract review. Exact power-of-two fixture corrections retain the 0.12.1
patch scope; all 293 native tests and five forward-contract Miri cases pass
under both borrow models. Strict package diagnostics and formatting pass.

Current-main merge `fdd719e4` revalidates 293 native cases, 38 doctests, eight
Miri cases in both models, strict Clippy and the 223-check API comparison
with Moirai 0.9.2 and Leto `248fe4f6`. Repeated paired bridge ratios are
0.718, 0.889, 0.916 and 0.968; the unintegrated serial-provider large SpMV
still costs 2.008 times current Moirai execution. Detailed revision and
measurement evidence remains in the sparse recovery record.
