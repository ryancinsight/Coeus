# Coeus 0.15.0 release checklist

Target: Rust 0.15.0 ([minor], [ADR 0087](docs/adr/0087-physical-quantity-autodiff.md)).

- [ ] [minor] Verify the exact candidate with the workspace lockfile, native
  checks in both feature modes, and the locked-registry SemVer gate against
  the published 0.14.0 cohort.
- [x] [minor] The [read-only planner](https://github.com/ryancinsight/Coeus/actions/runs/45060488822/job/137325065274) selected the eight pinned crates at `0f69d4a8`:
  `coeus-core`, `coeus-cpu`, `coeus-tensor`, `coeus-sparse`, `coeus-ops`,
  `coeus-autograd`, `coeus-nn`, and `coeus-optim`.
- [ ] [minor] Resolve the candidate archives in fresh consumers at the Rust
  1.95 minimum with the feature enabled and disabled. Check numerical
  gradients and ensure each shared provider resolves to one package identity.
- [ ] [minor] Verify the release gate and the outcome of the [automatic publication workflow](https://github.com/ryancinsight/Coeus/actions/runs/45060488822) triggered by the merge of the reviewed version pin.
- [ ] [minor] After authorized publication, verify registry checksums and repeat
  fresh-consumer verification directly against crates.io.
