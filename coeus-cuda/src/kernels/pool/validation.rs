use crate::kernels::validation::{
    checked_numel, cuda_u32, launch_grid_size, layouts_fit_cuda, CUDA_BLOCK_SIZE,
};
use coeus_core::Layout;

pub(crate) fn pool_layouts_are_valid(layouts: &[&Layout], rank: usize) -> bool {
    layouts_fit_cuda(layouts)
        && layouts.iter().all(|layout| {
            layout.ndim() == rank && layout.shape().iter().all(|&dimension| dimension != 0)
        })
}

pub(crate) fn pool_prefix_matches(lhs: &Layout, rhs: &Layout) -> bool {
    lhs.shape().get(..2) == rhs.shape().get(..2)
}

pub(crate) fn pool_shapes_match(lhs: &Layout, rhs: &Layout) -> bool {
    lhs.shape() == rhs.shape()
}

pub(crate) fn checked_pool_parameters(
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Option<[u32; 4]> {
    if kernel_size == 0 || stride == 0 || dilation == 0 {
        return None;
    }
    Some([
        cuda_u32(kernel_size)?,
        cuda_u32(stride)?,
        cuda_u32(padding)?,
        cuda_u32(dilation)?,
    ])
}

pub(crate) fn checked_pool_work(layout: &Layout) -> Option<(u32, u32)> {
    let numel = checked_numel(layout)?;
    Some((cuda_u32(numel)?, launch_grid_size(numel)?))
}

pub(crate) const POOL_BLOCK_SIZE: u32 = CUDA_BLOCK_SIZE;
