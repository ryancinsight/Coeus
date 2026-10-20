# Coeus compatibility preparation checklist

Target: 0.14.0 (private proposal; no release acceptance).

## Authorized private preparation

- [x] [major] Classify public Hephaestus device/buffer/provider identity migration in Proposed ADR 0085 and the Breaking changelog.
- [x] [major] Prepare all 16 workspace versions and all internal path requirements atomically with Hephaestus ^0.27.1 and Apollo FFT ^0.30.0.
- [x] [patch] Preserve #611 and #585 owners, all Rust/tests/profiles/workflows, numerical assertions and existing time budgets.
- [x] [major] Regenerate a Cargo-written lock with exact Git identities; locked default/all-feature metadata accepts the private graph without a path overlay.
- [ ] [major] Complete current CPU, strict/docs/doctest, MSRV and API/package evidence recorded in the external receipt.

## Adoption and release prerequisites (not opened implementation items)

- [ ] [major] Obtain ADR 0085 sign-off and independent final exact-head review before adoption or public updates.
- [ ] [major] Verify intended upstream Git defaults, registry artifacts and one provider/Apollo type-bearing graph, then regenerate ordinary locks/fresh consumers and standard packages.
- [ ] [major] Complete all applicable hosted backend, physical-device, performance, complete Python/wheel/cohort and release gates without changing skips, assertions, features, baselines or budgets.
- [ ] [patch] Retain numerical-tail, six portable countermodel, deferred Coeus Miri and service holds; do not retry the service-blocked initial Apollo LaneBuffer task.
- [ ] [major] Coordinate upstream-first availability and automatic publish-pending side effects under separate public lifecycle authorization.
