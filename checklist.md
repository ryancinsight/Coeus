# Coeus checklist

Target version: 0.13.0

Sprint phase: Execution

## Real digamma / gamma-log gradients [minor]

- [x] Inspect ownership and isolate current source; preserve other lanes.
- [x] Reproduce missing capability and define provider/domain contracts.
- [x] Implement provider-owned CPU operation and additive consumer capability.
- [x] Complete local CPU diagnostics, 254 autograd and 42 CPU adapter cases.
- [x] Pass 20 Python alias cases, including weighted scalar/empty backward.
- [x] Independently review scalar providers and consumer rank-zero adapters.
- [x] Verify Eunomia 0.11.0 publication from merge `5ae27c75` and update the lock.
- [x] Resolve the reviewed remote Leto candidate `bd0696cd` without path patches.
- [x] Compare seven selected Leto/Coeus packages under the locked strict-minor API contract.
- [ ] Complete canonical remote dependency delivery, API/release gates and hosted CI.
- [ ] Resolve the runtime-owned dependency-policy failure without a whitelist exemption.
- [ ] Verify the complete release cohort after Moirai 0.11 and owned Apollo #696 prerequisites.
- [ ] Renew independent source review and Python CPU checks on the updated exact head.
- [ ] Implement and validate Hephaestus device capability in a separate slice.
- [ ] Integrate and publish provider-first after final checks, source receipts and fresh consumers.

The existing user authorization covers qualifying repaired-package publication.
The remaining holds are verification and provider prerequisites; app/service
deployment is not authorized. A green package-plan job with omitted packages
does not satisfy the complete-cohort gate.
