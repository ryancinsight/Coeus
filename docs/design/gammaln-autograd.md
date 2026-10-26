# Provider-owned gamma-log gradients

This additive CPU slice uses Eunomia's real digamma scalar contract and Leto's
DigammaOp traversal. Coeus only allocates tensors, dispatches the operation,
records the node and multiplies upstream gradients by digamma(saved input).

Use an optional DigammaOps capability, like ScalarPowerOps, rather than adding
a variant to the public exhaustive CpuUnaryOp enum or adding a required method
to FloatOps/BackendOps. Existing backend implementations remain compatible.
Only CPU implements the new capability in this slice. Device backends cannot
instantiate the tracked gamma-log API until their provider implements it;
there is no device-to-host fallback or numerical formula in this consumer.

Add lgamma(&Var)->Var and retain the existing lgamma_forward(&Var)->Tensor
as the existing public forward-only API. Both Python aliases use the tracked
operation, which also supports non-gradient and no-grad calls. This is an
additive API, with no replacement of canonical traits or removed public items.

For real non-poles d/dx ln|Gamma(x)|=psi(x), including negative nonintegers.
Backward is grad_out * psi(input), with ordinary IEEE multiplication at poles,
signed zero, infinity and NaN. The scalar domain and precision contract is in
the Eunomia provider design and tests. Python holds f64 values throughout.
Tests cover both aliases, non-unit seeds, repeated/shared accumulation,
transposes/slices/broadcasts, direct real CPU execution and framework VJPs.

Acceptance and remaining gates: scalar analytic/identity/high-precision tests,
Leto layouts and output-domain tests, Coeus native and Python gradients,
strict diagnostics, independent source review and exact-head hosted CI. No
accelerator gradient parity or release completion follows from CPU progress.
Only docs deployments are authorized; source publication and auto-publish
side effects from merged version bumps remain held until explicitly approved.

Current-graph prerequisite: Apollo PR #696 (`b62b9c7e`) owns the existing
Moirai-0.11 manifest admission. The CPU candidate uses a separate unchanged
reference checkout of that PR for validation; it does not modify or merge the
Apollo service hold. Current Apollo main cannot resolve fresh Moirai 0.11.
Canonical downstream integration remains blocked until that owned prerequisite
is delivered or its exact candidate is admitted through the normal review path.
