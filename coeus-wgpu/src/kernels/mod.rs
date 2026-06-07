pub mod cache;
pub mod layout;
pub mod binary;
pub mod unary;
pub mod matmul;
pub mod reduce;
pub mod conv;
pub mod pool;
pub mod optim;
pub mod fuse;

pub use binary::{dispatch_contiguous_binary, dispatch_binary};
pub use unary::{dispatch_unary, dispatch_contiguous_unary};
pub use matmul::dispatch_matmul;
pub use reduce::{dispatch_reduce, dispatch_fused_reduce};
pub use conv::{
    dispatch_conv1d, dispatch_conv2d, dispatch_conv3d,
    dispatch_conv1d_backward, dispatch_conv2d_backward, dispatch_conv3d_backward,
};
pub use pool::{
    dispatch_max_pool2d, dispatch_max_pool2d_backward,
    dispatch_avg_pool2d, dispatch_avg_pool2d_backward,
    dispatch_max_pool3d, dispatch_max_pool3d_backward,
    dispatch_avg_pool3d, dispatch_avg_pool3d_backward,
};
pub use optim::{dispatch_sgd_step, dispatch_adam_step, dispatch_adamw_step, dispatch_rmsprop_step, dispatch_adagrad_step};
pub use fuse::dispatch_fused;

