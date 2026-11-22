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
- [x] [minor] The [locked-registry release SemVer gate](https://github.com/ryancinsight/Coeus/actions/runs/45060488822/job/137327163352) passed at `0f69d4a8`. The [publish job](https://github.com/ryancinsight/Coeus/actions/runs/45060488822/job/137328612002) uploaded Core and received HTTP 403 for CPU.
- [x] [patch] Verified the coeus-core 0.15.0 archive against the sparse-index SHA-256 `4a4a6b60854d1e175acae149ee6b050767784ea73718d1a387e6be9e16ea6c8f` and its VCS identity `0f69d4a812e08a06494678c5382171bdf287548d`.
- [ ] [patch] Release owner: verify the trusted publisher binding for coeus-cpu and the six other pending packages against the actual release workflow. On an authorized retry, use the planner to select only missing versions.
- [ ] [minor] After authorized publication, verify registry checksums and repeat
  fresh-consumer verification directly against crates.io.
