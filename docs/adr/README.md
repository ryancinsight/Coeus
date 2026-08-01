# Architecture decision records

Statuses use the canonical lifecycle `Proposed`, `Accepted`, or `Rejected`.
Duplicate historical numbers remain visible until a decision-scoped
renumbering migration can update every durable reference.

| Record | Status |
|---|---|
| [0001 Dimension-complete interpolation](0001-dimension-complete-interpolation.md) | Accepted |
| [0002 Named optimizer ownership](0002-named-optimizer-ownership.md) | Accepted |
| [0003 CUDA layout ABI boundary](0003-cuda-layout-abi-boundary.md) | Accepted |
| [0004 CUDA launch validation SSOT](0004-cuda-launch-validation-ssot.md) | Accepted |
| [0005 CUDA elementwise launch tree](0005-cuda-elementwise-launch-tree.md) | Accepted |
| [0007 CUDA pool1d launch ABI](0007-cuda-pool1d-launch-abi.md) | Accepted |
| [0008 CUDA pool2d launch ABI](0008-cuda-pool2d-launch-abi.md) | Accepted |
| [0009 CUDA pool3d launch ABI](0009-cuda-pool3d-launch-abi.md) | Accepted |
| [0010 CUDA matmul launch ABI](0010-cuda-matmul-launch-abi.md) | Accepted |
| [0012 CUDA unfold/fold launch ABI](0012-cuda-unfold-fold-launch-abi.md) | Accepted |
| [0013 CUDA transposed-convolution launch ABI](0013-cuda-convolution-transpose-launch-abi.md) | Accepted |
| [0014 CUDA fused dispatch ABI](0014-cuda-fused-dispatch-abi.md) | Accepted |
| [0015 CUDA elementwise backend boundary](0015-cuda-elementwise-backend-boundary.md) | Accepted |
| [0016 CUDA convolution backend tree](0016-cuda-convolution-backend-tree.md) | Accepted |
| [0018 Fused operation tag tree](0018-fused-operation-tag-tree.md) | Accepted |
| [0019 WGPU pool1d dispatch modes](0019-wgpu-pool1d-dispatch-modes.md) | Accepted |
| [0020 WGPU fallible dispatch boundary](0020-wgpu-fallible-dispatch-boundary.md) | Accepted |
| [0021 Coeus operations activation tree](0021-coeus-ops-activation-tree.md) | Accepted |
| [0021 Crates workspace layout](0021-crates-workspace-layout.md) | Accepted |
| [0021 Native cumulative scan dispatch](0021-native-cumulative-scan-dispatch.md) | Accepted |
| [0022 Coeus Hephaestus provider layer](0022-coeus-hephaestus-provider-layer.md) | Accepted |
| [0023 Hephaestus elementwise providers](0023-coeus-hephaestus-elementwise-providers.md) | Accepted |
| [0024 Hephaestus activation providers](0024-coeus-hephaestus-activation-providers.md) | Accepted |
| [0025 Hephaestus comparison providers](0025-coeus-hephaestus-comparison-providers.md) | Accepted |
| [0025 External framework parity without Burn](0025-external-framework-parity-without-burn.md) | Accepted |
| [0026 Backend dispatch boundaries](0026-coeus-backend-dispatch-boundaries.md) | Accepted |
| [0026 Hephaestus unary math providers](0026-coeus-hephaestus-unary-math-providers.md) | Accepted |
| [0027 Hephaestus error-function providers](0027-coeus-hephaestus-error-function-providers.md) | Accepted |
| [0028 Hephaestus exact GELU providers](0028-coeus-hephaestus-gelu-providers.md) | Accepted |
| [0029 Hephaestus lgamma provider](0029-coeus-hephaestus-lgamma-provider.md) | Accepted |
| [0030 Hephaestus activation-tail providers](0030-coeus-hephaestus-activation-tail-providers.md) | Accepted |
| [0036 Device-local COW copy](0036-device-local-cow-copy.md) | Accepted |
| [0037 Accelerator initialization contracts](0037-uninitialized-cow-consumer.md) | Accepted |
| [0038 Hephaestus activation tail](0038-coeus-hephaestus-activation-tail.md) | Accepted |
| [0039 CUDA fused dispatch errors](0039-cuda-fused-dispatch-errors.md) | Accepted |
| [0040 CUDA math dispatch hierarchy](0040-cuda-math-dispatch-hierarchy.md) | Accepted |
| [0041 CUDA pooling index contract](0041-cuda-pooling-index-contract.md) | Accepted |
| [0042 Fallible autograd backward](0042-fallible-autograd-backward.md) | Accepted |
| [0043 CUDA native dispatch boundary](0043-cuda-native-dispatch-boundary.md) | Accepted |
| [0044 WGPU reduction provider ownership](0044-wgpu-reduction-provider-ownership.md) | Accepted |
| [0045 Fallible module forward](0045-fallible-module-forward.md) | Accepted |
| [0046 Provider-owned convolution dispatch](0046-provider-owned-convolution-dispatch.md) | Accepted |
| [0047 Provider-owned attention dispatch](0047-provider-owned-attention-dispatch.md) | Accepted |
| [0048 Provider-owned stateful update dispatch](0048-provider-owned-stateful-update-dispatch.md) | Accepted |
