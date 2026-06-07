#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLayoutInfo {
    pub offset: u32,
    pub ndim: u32,
    pub shape: [u32; 8],
    pub strides: [u32; 8],
}

impl GpuLayoutInfo {
    pub fn from_layout(layout: &coeus_core::Layout) -> Self {
        let mut shape = [0u32; 8];
        let mut strides = [0u32; 8];
        let ndim = layout.ndim();
        assert!(ndim <= 8, "WebGPU backend supports up to 8 dimensions");
        for i in 0..ndim {
            shape[i] = layout.shape()[i] as u32;
            strides[i] = layout.strides()[i] as u32;
        }
        Self {
            offset: layout.offset() as u32,
            ndim: ndim as u32,
            shape,
            strides,
        }
    }
}
