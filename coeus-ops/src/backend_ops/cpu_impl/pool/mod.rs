mod avg;
mod max;
mod pool1d;

pub(crate) use avg::{avg_pool2d, avg_pool2d_backward, avg_pool3d, avg_pool3d_backward};
pub(crate) use max::{max_pool2d, max_pool2d_backward, max_pool3d, max_pool3d_backward};
pub(crate) use pool1d::{
    avg_pool1d, avg_pool1d_backward, max_pool1d, max_pool1d_backward,
};
