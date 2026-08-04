//! Interface-segregated sub-traits for [`BackendOps`].
//!
//! This module exports ten single-concern capability traits. `BackendOps`
//! composes the six capabilities shared by every backend; attention, optimizer,
//! random initialization, and half-vector rotation remain optional capabilities
//! so unrelated kernels do not acquire unsupported bounds. Backends implement
//! each trait independently;
//! the blanket impl in
//! [`crate::backend_ops::trait_def`] provides `BackendOps` automatically.  This satisfies
//! the interface-segregation principle: call sites can bound on only the
//! sub-trait they need (e.g. `B: ElementwiseOps<T>`) instead of pulling in
//! all 36 methods.
//!
//! Sub-traits and their concerns:
//! - [`ElementwiseOps`] — binary and unary element-wise ops
//! - [`MatmulOps`] — matmul, batched matmul, accumulate variants
//! - [`ReductionOps`] — reduce, argmax/argmin, topk, cumulative sum/product scans
//! - [`ConvOps`] — regular convolution and 1D/2D transposed convolution
//! - [`PoolOps`] — max/avg pool 1D/2D/3D forward+backward
//! - [`AttentionOps`] — scaled dot-product attention forward+backward
//! - [`OptimizerOps`] — fused SGD/Adam/RMSProp/AdamW/AdaGrad steps
//! - [`RandomInitOps`] — seeded provider-native parameter initialization
//! - [`RotateHalfOps`] — rotary half-vector permutation
//! - [`UnfoldFoldOps`] — sliding-window unfold and adjoint fold (1D/2D)
//!
//! [`BackendOps`]: super::BackendOps

pub mod attention;
pub mod conv;
pub mod cross_entropy;
pub mod elementwise;
pub mod matmul;
pub mod optimizer;
pub mod pool;
pub mod random_init;
pub mod reduction;
/// Half-vector rotation capability.
pub mod rotate_half;
pub mod unfold_fold;

pub use attention::{AttentionOps, AttentionScalar};
pub use conv::{ConvOps, ConvolutionBackward, ConvolutionForward};
pub use cross_entropy::CrossEntropyOps;
pub use elementwise::ElementwiseOps;
pub use matmul::MatmulOps;
pub use optimizer::{OptimizerOps, OptimizerStateRef, OptimizerStepRule, OptimizerStepValidation};
pub use pool::PoolOps;
pub use random_init::RandomInitOps;
pub use reduction::ReductionOps;
pub use rotate_half::RotateHalfOps;
pub use unfold_fold::UnfoldFoldOps;
