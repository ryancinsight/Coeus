# Coeus 0.15.0 release checklist

Target: Rust 0.15.0 ([minor], [ADR 0087](docs/adr/0087-physical-quantity-autodiff.md)).

- [ ] [minor] Verify the exact candidate with the workspace lockfile, native
  checks in both feature modes, and the locked-registry SemVer gate against
  the published 0.14.0 cohort.
- [ ] [minor] Confirm the read-only planner builds the eight pinned crates:
  `coeus-core`, `coeus-cpu`, `coeus-tensor`, `coeus-sparse`, `coeus-ops`,
  `coeus-autograd`, `coeus-nn`, and `coeus-optim`.
- [ ] [minor] Resolve the candidate archives in fresh consumers at the Rust
  1.95 minimum with the feature enabled and disabled. Check numerical
  gradients and ensure each shared provider resolves to one package identity.
- [ ] [minor] Report qualification before requesting authorization to publish.
  Merging the version pin to main enables the automatic crates.io workflow.
- [ ] [minor] After authorized publication, verify registry checksums and repeat
  fresh-consumer verification directly against crates.io.
