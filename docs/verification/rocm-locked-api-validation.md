# Exact locked ROCm consumer API validation

[patch] This private validation-only proposal extends the existing registered
`Backend parity` workflow with a default-false `locked_rocm_api_only` input.
It does not adopt a consumer/provider source or authorize a public update.
The product preparation, its independent review and its receipts remain intact.

## Selected contract

Compare only `coeus-rocm` from the accepted Coeus 0.13 source at `5d5979bc`
to the exact 0.14 source named in `scripts/validation/rocm-api-input.json`.
The input policy binds both complete heads, trees, Cargo.lock checksums,
provider/Apollo commits and versions. Its SHA256 is a required dispatch input.
The candidate keeps Hephaestus ^0.27.1 and Apollo FFT ^0.30.0. No registry
baseline, default-head substitution, source regeneration, path patch, failure
exemption or provider package comparison is permitted.

Both consumer sides compile every declared stable feature: `default` and
`rocm`. The script checks the feature inventory and uses `--all-features`.
These consumer features do not add a provider decomposition API comparison;
the completed provider six-package Linux evidence is retained separately.

The snapshot/locked-Rustdoc/prebuilt-JSON mechanism follows Atlas's existing
`atlas_semver_locked.py` at `7a5a1036`. The narrow wrapper additionally binds
this reviewed pair, rejects source/policy drift, retains both actual JSON
files and raw command logs, and forces a patch structural inventory even
though the proposed pre-1.0 version migration is intentionally incompatible.
That structural inventory does not establish nominal old-provider type
compatibility, runtime numerical accuracy or release acceptance.

## Execution boundary

The API-only job uses hosted Ubuntu 24.04 and the existing ROCm 7.2.4 SDK
image family. It captures actual base/tool image identities and runs as the
workspace owner's UID. It mounts no GPU devices and executes no tests,
`rocminfo`, acquisition, publication or provider API campaign. It grants only
`contents: read`; it has no OIDC, environment or publishing job.

All eight original job bodies, normal PR/push triggers, permissions, feature
selections, assertions and limits remain unchanged. Default-false mode keeps
existing manual behavior. API-only mode excludes all six runtime jobs and
both physical/self-hosted jobs even when either hardware input is true.
This is an explicit isolated validation mode, not a new release verdict or
a waiver of any ordinary gate.

The current release semver job cannot establish this distinct comparison:
its pinned shared workflow chooses published/tag baselines, lacks this ROCm
SDK/source-pair input, and has its existing registry-rot policy. Its release
workflow also contains OIDC publishing and main/scheduled publish-pending.
Those existing policies remain unchanged and still apply at their authorized
lifecycle step.

## Required later external action

First obtain independent exact-head review of this validation-only proposal.
The owner then separately authorizes making the reviewed validation branch
and its exact private Apollo source available to the hosted runner. The
branch descends from the selected product head; no default-branch merge is
needed to reuse the already registered `backend-parity.yml` workflow.

After those source/access prerequisites and explicit dispatch authorization,
run that existing workflow on the reviewed validation ref with
`locked_rocm_api_only=true`, `api_candidate_sha` equal to the input policy's
candidate head, and `api_policy_sha256` equal to the reviewed file digest.
Keep the two hardware inputs false; the mode guards also reject their
eligibility if true. Ordinary exact Git fetches must succeed. A missing private
Apollo object fails closed; neither a newer default nor an older provider
supplies a fallback. No dispatch, SDK acquisition or public push occurs in
this preparation pass.

For an already authorized Linux SDK host with the exact Git objects/source
cache available, the same script runs directly with an isolated Cargo home
and the exact 0.51.0 checker, avoiding any workflow-registration change.
Current-source metadata is not ordinary registry or release availability.

## Local preparation evidence and remaining holds

The external receipt records YAML/shell/Python validation, preservation of
the eight original job bodies, 12 unchanged normal-mode cases, all four
hardware-input combinations selecting only the API job, and actual-source
positive/negative input-policy controls. Preflight-only reports always keep
`completed=false`; they cannot count as Linux API comparisons. Real Linux
full-feature JSON generation and the comparison remain unexecuted here.

All owner/ADR, numerical-tail/portable-bound, physical hardware, performance,
broader/future/FFI/Python/wheel, deferred Coeus Miri and service holds remain.
The service-blocked initial Apollo LaneBuffer task and inherited process
inventory denial are untouched. No gate, credential, setting, public branch,
PR, workflow run, device, merge or publication is changed by this preparation.

## Unselected FFI-source policy alternative

This separate private validation proposal binds the unselected Coeus FFI-source
alternative `c26c2f68afa1c47c2dc482a9e81bbc478e2f5803`, tree
`53f70ecd5edfe5749a7ec06ff6aadd872ff11e47`, and Apollo
`124b67ba7f2d7e8380b9ff76fe5a970f1e1ca68a`. The accepted 0.13 baseline,
provider a735, floors, checker and stable features stay unchanged. Only the
input policy and this source-bound note differ from the primary validation
proposal; its workflow, wrapper and Dockerfile bytes remain identical.
Neither source adoption nor Linux execution is implied, and the primary
`18ff1af854d5cde450316c65300492d57b7f9035` proposal and receipt remain intact.

## Canonical board repair identity renewal

The candidate policy now names the product above after its complete preparation
checklist moves into the anchored canonical backlog item. The earlier b7311c7a
product and 16f49724 validation source and their receipts stay intact. Rust,
manifests and Cargo.lock remain identical. The lock selects Apollo
`124b67ba7f2d7e8380b9ff76fe5a970f1e1ca68a`; the separate Apollo documentation
repair `57d2a123debcd2e68cddef9b2ad7e2a7c08f6537` is its direct descendant
with unchanged code and lock, preparing ordinary source availability without
changing this consumer graph. The API candidate input is the exact product
head in the policy, while the workflow ref is this separate validation
successor. These documentation repairs require independent exact-head review
before use and establish no native, Linux, hardware or release result.
