# ADR-0052: Route cross-entropy through Leto and Hephaestus

- Status: Proposed
- Date: 2026-08-04
- Scope: mean cross-entropy forward/backward, saved probabilities, CPU Leto
  dispatch, and WGPU/CUDA/ROCm/Metal Hephaestus dispatch
- Change class: `[minor] [arch]`, subject to SemVer verification
- Board item: `COEUS-CROSS-ENTROPY-PROVIDER-001`

## Context

Coeus currently downloads complete logits, computes stable log-sum-exp and
softmax locally, uploads the scalar loss, and stores probabilities in `Vec<T>`.
Backward downloads the upstream scalar, computes every logits gradient on the
host, and uploads the result. This silently changes accelerator execution to
CPU execution and duplicates provider mathematics.

Leto and Hephaestus currently expose no cross-entropy role. The provider APIs
therefore land before the Coeus consumer cutover.

## Decision

Define one mean cross-entropy contract with logits `[N, C]`, one validated class
index per row, scalar loss output, and provider-resident softmax probabilities
for backward. Leto owns the generic CPU implementation over borrowed slices.
Hephaestus owns one accelerator orchestration contract parameterized by its
device API; WGPU, CUDA, ROCm, and Metal provide only device implementations.

Coeus adds a narrow backend capability that selects Leto or Hephaestus once per
operation. The tracked node stores a tensor, not a host vector, and backward
writes provider-resident logits gradients. Validation and provider failures are
typed and propagate through NN and Python callers. No CPU, local-kernel, or
compatibility fallback remains.

## Alternatives rejected

- Compose Coeus elementwise/reduction operations: rejected because target gather
  and the shared stable softmax contract would remain consumer-owned.
- Retain host probabilities for backward: rejected because it preserves the
  full-payload transfer and host-memory lifetime defect.
- Add only CUDA/WGPU kernels in Coeus: rejected because it duplicates the
  provider dimension and leaves ROCm/Metal incomplete.

## Verification

One generic conformance suite covers every supported scalar/provider. CPU and
accelerator differentials compare loss and logits gradients with the stable
analytical formula and external PyTorch fixtures, using derived floating-point
bounds. Negative contracts cover rank, target count, empty classes, and label
range. Residue scans reject host transfers and host probability storage in the
migrated closure. Warning-denied Clippy, Nextest, doctests, SemVer checks,
independent review, and exact-head provider CI gate delivery.

## Revisit trigger

Revisit when reduction modes, label smoothing, class weights, or sparse target
representations become current requirements; extend the provider contract
rather than adding parallel loss APIs.
