use super::{dispatch, provider::UnfoldFoldProvider};
use crate::HephaestusBackend;
use coeus_core::{Layout, Scalar};
use coeus_ops::UnfoldFoldOps;

impl<P, T> UnfoldFoldOps<T> for HephaestusBackend<P>
where
    P: UnfoldFoldProvider<T>,
    T: Scalar + leto_ops::Scalar,
{
    fn unfold1d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        dispatch::unfold::<HephaestusBackend<P>, T, 3, 1>(
            "unfold1d",
            (input, input_layout),
            dispatch::parameters::<HephaestusBackend<P>, T, 1>(
                "unfold1d",
                [kernel_size],
                [stride],
                [padding],
                [dilation],
            )?,
            (output, output_layout),
        )
    }

    fn fold1d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        output_size: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        dispatch::fold::<HephaestusBackend<P>, T, 3, 1>(
            "fold1d",
            (input, input_layout),
            [output_size],
            dispatch::parameters::<HephaestusBackend<P>, T, 1>(
                "fold1d",
                [kernel_size],
                [stride],
                [padding],
                [dilation],
            )?,
            (output, output_layout),
        )
    }

    fn unfold2d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        dispatch::unfold::<HephaestusBackend<P>, T, 4, 2>(
            "unfold2d",
            (input, input_layout),
            dispatch::parameters::<HephaestusBackend<P>, T, 2>(
                "unfold2d",
                [kernel_h, kernel_w],
                [stride_h, stride_w],
                [padding_h, padding_w],
                [dilation_h, dilation_w],
            )?,
            (output, output_layout),
        )
    }

    fn fold2d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        output_h: usize,
        output_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        dispatch::fold::<HephaestusBackend<P>, T, 4, 2>(
            "fold2d",
            (input, input_layout),
            [output_h, output_w],
            dispatch::parameters::<HephaestusBackend<P>, T, 2>(
                "fold2d",
                [kernel_h, kernel_w],
                [stride_h, stride_w],
                [padding_h, padding_w],
                [dilation_h, dilation_w],
            )?,
            (output, output_layout),
        )
    }
}
