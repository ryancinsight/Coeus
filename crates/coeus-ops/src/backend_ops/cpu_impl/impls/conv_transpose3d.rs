use super::super::CpuBackend;
use crate::backend_ops::defaults;
use crate::backend_ops::traits::ConvTranspose3dOps;
use coeus_core::{CpuAddressableStorageMut, Layout, Scalar};

#[allow(clippy::too_many_arguments)]
impl<T: Scalar + leto_ops::Scalar, B: CpuBackend> ConvTranspose3dOps<T> for B
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    #[inline]
    fn conv_transpose3d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        bias: Option<&Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) where
        T: coeus_core::Float,
    {
        defaults::conv_transpose::conv_transpose3d(
            self,
            input,
            input_layout,
            weight,
            weight_layout,
            bias,
            stride,
            padding,
            output_padding,
            dilation,
            output,
            output_layout,
        );
    }
}
