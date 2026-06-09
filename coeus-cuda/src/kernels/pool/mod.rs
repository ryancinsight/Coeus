pub mod avg;
pub mod max;

pub use avg::{dispatch_avg_pool2d, dispatch_avg_pool2d_backward};
pub use max::{dispatch_max_pool2d, dispatch_max_pool2d_backward};

pub(crate) const POOL_COMMON_SRC: &str = r#"
struct GpuLayoutInfo {
    unsigned int offset;
    unsigned int ndim;
    unsigned int shape[8];
    unsigned int strides[8];
};

__device__ unsigned int get_physical_index(const GpuLayoutInfo& layout, unsigned int n, unsigned int c, unsigned int h, unsigned int w) {
    unsigned int idx = layout.offset;
    if (layout.ndim > 0) idx += n * layout.strides[0];
    if (layout.ndim > 1) idx += c * layout.strides[1];
    if (layout.ndim > 2) idx += h * layout.strides[2];
    if (layout.ndim > 3) idx += w * layout.strides[3];
    return idx;
}
"#;
