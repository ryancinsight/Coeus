# Provider-owned gamma-log gradients

This additive CPU slice uses Eunomia's real digamma scalar contract and Leto's
DigammaOp traversal. Coeus only allocates tensors, dispatches the operation,
records the node and multiplies upstream gradients by digamma(saved input).
Rank-zero support includes the existing reshape adapter used by scalar
broadcast-gradient reduction, so weighted scalar backward completes.

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
Existing user authorization covers qualifying source integration and repaired
package publication after final gates. Documentation deployments are authorized;
app/service deployments are not. Main version bumps may invoke the existing
publisher, so exact-head review, checks, release-cohort verification and
fresh-consumer/source receipts must precede integration.

Eunomia 0.11.0 is published from merge `5ae27c75`; both archive VCS receipts
match that source. The lock uses that merge and reviewed remote Leto
`bd0696cd`, whose numerical source and manifests are unchanged from the
validated provider. Leto's package plan verifies `leto`, while `leto-ops`
waits for the required Moirai 0.11 registry version. Coeus's native policy test
also rejects Tokio via Quinn/Moirai QUIC; its native owner holds that repair.
These prerequisites do not authorize downgrades or policy exemptions.

Current-graph prerequisite: Apollo PR #696 (`b62b9c7e`) owns the existing
Moirai-0.11 manifest admission. The CPU candidate uses a separate unchanged
reference checkout of that PR for validation; it does not modify or merge the
Apollo service hold. Current Apollo main cannot resolve fresh Moirai 0.11.
Canonical downstream integration remains blocked until that owned prerequisite
is delivered or its exact candidate is admitted through the normal review path.
Its current version-bump and license checks fail; this slice changes neither
Apollo nor the license policy. Python binding acceptance and the canonical
read-only package plan now run in PR CI without publication permissions.
