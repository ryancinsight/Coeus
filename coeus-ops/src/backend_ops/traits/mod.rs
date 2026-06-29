//! Interface-segregated sub-traits for [`BackendOps`].
//!
//! `BackendOps` is a super-trait composed of eight single-concern sub-traits.
//! Backends implement each sub-trait independently; the blanket impl in
//! [`crate::backend_ops::trait_def`] provides `BackendOps` automatically.  This satisfies
//! the interface-segregation principle: call sites can bound on only the
//! sub-trait they need (e.g. `B: ElementwiseOps<T>`) instead of pulling in
//! all 36 methods.
//!
//! Sub-traits and their concerns:
//! - [`ElementwiseOps`] — binary and unary element-wise ops
//! - [`MatmulOps`] — matmul, batched matmul, accumulate variants
//! - [`ReductionOps`] — reduce, argmax/argmin, topk, cumsum, suffix_sum
//! - [`ConvOps`] — 1D/2D/3D conv forward+backward, transposed conv defaults
//! - [`PoolOps`] — max/avg pool 1D/2D/3D forward+backward
//! - [`AttentionOps`] — scaled dot-product attention forward+backward
//! - [`OptimizerOps`] — fused SGD/Adam/RMSProp/AdamW/AdaGrad steps
//! - [`UnfoldFoldOps`] — sliding-window unfold and adjoint fold (1D/2D)
//!
//! [`BackendOps`]: super::BackendOps

pub mod attention;
pub mod conv;
pub mod elementwise;
pub mod matmul;
pub mod optimizer;
pub mod pool;
pub mod reduction;
pub mod unfold_fold;

pub use attention::AttentionOps;
pub use conv::ConvOps;
pub use elementwise::ElementwiseOps;
pub use matmul::MatmulOps;
pub use optimizer::OptimizerOps;
pub use pool::PoolOps;
pub use reduction::ReductionOps;
pub use unfold_fold::UnfoldFoldOps;
