# Coeus sparse recovery checklist

Active target: **0.12.1**. Owner: Codex. Base: `cd8e984e0d9c4f3621bd89f7830646de845a22e5`.

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
formatting pass; three selected Miri cases pass; the locked API comparison
passes 223 checks (30 tool skips). Fresh Git resolution and attempted 32-bit
memory checks remain limited by existing provider failures. The bridge is a
complete compatible increment; the tensor forward migration remains open.

## General forward cutover [minor]

- [ ] [minor] Obtain provider support for the contracts in
      [COEUS-SPARSE-VIEW-FORWARD](backlog.md#coeus-sparse-view-forward).
      The 0.12.1 bridge recovery does not close this item.
