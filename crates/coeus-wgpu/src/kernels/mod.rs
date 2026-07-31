pub mod binary;
pub mod cache;
pub mod fuse;
pub mod layout;
pub mod matmul;
pub mod optim;
pub mod pool;
pub mod reduce;
pub mod unary;
pub mod unfold_fold;

pub use binary::{dispatch_binary, dispatch_contiguous_binary};
pub use fuse::dispatch_fused;
pub use matmul::dispatch_matmul;
pub use optim::{
    dispatch_adagrad_step, dispatch_adam_step, dispatch_adamw_step, dispatch_rmsprop_step,
    dispatch_sgd_step,
};
pub use pool::{
    dispatch_avg_pool1d, dispatch_avg_pool1d_backward, dispatch_avg_pool2d,
    dispatch_avg_pool2d_backward, dispatch_avg_pool3d, dispatch_avg_pool3d_backward,
    dispatch_max_pool1d, dispatch_max_pool1d_backward, dispatch_max_pool2d,
    dispatch_max_pool2d_backward, dispatch_max_pool3d, dispatch_max_pool3d_backward,
};
pub use reduce::dispatch_fused_reduce;
pub use unary::{dispatch_contiguous_unary, dispatch_unary};
pub use unfold_fold::{dispatch_fold1d, dispatch_fold2d, dispatch_unfold1d, dispatch_unfold2d};
