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
- [ ] Complete canonical remote dependency delivery, API/release gates and hosted CI.
- [ ] Implement and validate Hephaestus device capability in a separate slice.
- [ ] Obtain source-publication approval before any auto-release merge.
