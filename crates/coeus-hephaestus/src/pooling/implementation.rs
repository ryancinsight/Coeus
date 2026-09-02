use super::{dispatch, provider::PoolingProvider};
use crate::HephaestusBackend;
use coeus_core::{Layout, Scalar};
use coeus_ops::PoolOps;
use hephaestus_core::PoolingMode;

impl<P, T> PoolOps<T> for HephaestusBackend<P>
where
    P: PoolingProvider<T>,
    T: Scalar + leto_ops::Scalar,
{
    fn max_pool1d(
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
        dispatch::forward::<HephaestusBackend<P>, T, 3, 1>(
            "max_pool1d",
            (input, input_layout),
            dispatch::parameters::<HephaestusBackend<P>, T, 1>(
                "max_pool1d",
                [kernel_size],
                [stride],
                [padding],
                [dilation],
            )?,
            PoolingMode::Maximum,
            (output, output_layout),
        )
    }

    fn max_pool1d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut Self::DeviceBuffer<T>,
        grad_input_layout: &Layout,
    ) -> Result<(), Self::Error> {
        dispatch::backward::<HephaestusBackend<P>, T, 3, 1>(
            "max_pool1d_backward",
            (grad_out, grad_out_layout),
            Some((input, input_layout)),
            dispatch::parameters::<HephaestusBackend<P>, T, 1>(
                "max_pool1d_backward",
                [kernel_size],
                [stride],
                [padding],
                [dilation],
            )?,
            PoolingMode::Maximum,
            (grad_input, grad_input_layout),
        )
    }

    fn avg_pool1d(
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
        dispatch::forward::<HephaestusBackend<P>, T, 3, 1>(
            "avg_pool1d",
            (input, input_layout),
            dispatch::parameters::<HephaestusBackend<P>, T, 1>(
                "avg_pool1d",
                [kernel_size],
                [stride],
                [padding],
                [dilation],
            )?,
            PoolingMode::Average,
            (output, output_layout),
        )
    }

    fn avg_pool1d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut Self::DeviceBuffer<T>,
        grad_input_layout: &Layout,
    ) -> Result<(), Self::Error> {
        dispatch::backward::<HephaestusBackend<P>, T, 3, 1>(
            "avg_pool1d_backward",
            (grad_out, grad_out_layout),
            None,
            dispatch::parameters::<HephaestusBackend<P>, T, 1>(
                "avg_pool1d_backward",
                [kernel_size],
                [stride],
                [padding],
                [dilation],
            )?,
            PoolingMode::Average,
            (grad_input, grad_input_layout),
        )
    }

    fn max_pool2d(
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
        dispatch::forward::<HephaestusBackend<P>, T, 4, 2>(
            "max_pool2d",
            (input, input_layout),
            dispatch::parameters::<HephaestusBackend<P>, T, 2>(
                "max_pool2d",
                [kernel_size; 2],
                [stride; 2],
                [padding; 2],
                [dilation; 2],
            )?,
            PoolingMode::Maximum,
            (output, output_layout),
        )
    }

    fn max_pool2d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut Self::DeviceBuffer<T>,
        grad_input_layout: &Layout,
    ) -> Result<(), Self::Error> {
        dispatch::backward::<HephaestusBackend<P>, T, 4, 2>(
            "max_pool2d_backward",
            (grad_out, grad_out_layout),
            Some((input, input_layout)),
            dispatch::parameters::<HephaestusBackend<P>, T, 2>(
                "max_pool2d_backward",
                [kernel_size; 2],
                [stride; 2],
                [padding; 2],
                [dilation; 2],
            )?,
            PoolingMode::Maximum,
            (grad_input, grad_input_layout),
        )
    }

    fn avg_pool2d(
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
        dispatch::forward::<HephaestusBackend<P>, T, 4, 2>(
            "avg_pool2d",
            (input, input_layout),
            dispatch::parameters::<HephaestusBackend<P>, T, 2>(
                "avg_pool2d",
                [kernel_size; 2],
                [stride; 2],
                [padding; 2],
                [dilation; 2],
            )?,
            PoolingMode::Average,
            (output, output_layout),
        )
    }

    fn avg_pool2d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut Self::DeviceBuffer<T>,
        grad_input_layout: &Layout,
    ) -> Result<(), Self::Error> {
        dispatch::backward::<HephaestusBackend<P>, T, 4, 2>(
            "avg_pool2d_backward",
            (grad_out, grad_out_layout),
            None,
            dispatch::parameters::<HephaestusBackend<P>, T, 2>(
                "avg_pool2d_backward",
                [kernel_size; 2],
                [stride; 2],
                [padding; 2],
                [dilation; 2],
            )?,
            PoolingMode::Average,
            (grad_input, grad_input_layout),
        )
    }

    fn max_pool3d(
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
        dispatch::forward::<HephaestusBackend<P>, T, 5, 3>(
            "max_pool3d",
            (input, input_layout),
            dispatch::parameters::<HephaestusBackend<P>, T, 3>(
                "max_pool3d",
                [kernel_size; 3],
                [stride; 3],
                [padding; 3],
                [dilation; 3],
            )?,
            PoolingMode::Maximum,
            (output, output_layout),
        )
    }

    fn max_pool3d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut Self::DeviceBuffer<T>,
        grad_input_layout: &Layout,
    ) -> Result<(), Self::Error> {
        dispatch::backward::<HephaestusBackend<P>, T, 5, 3>(
            "max_pool3d_backward",
            (grad_out, grad_out_layout),
            Some((input, input_layout)),
            dispatch::parameters::<HephaestusBackend<P>, T, 3>(
                "max_pool3d_backward",
                [kernel_size; 3],
                [stride; 3],
                [padding; 3],
                [dilation; 3],
            )?,
            PoolingMode::Maximum,
            (grad_input, grad_input_layout),
        )
    }

    fn avg_pool3d(
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
        dispatch::forward::<HephaestusBackend<P>, T, 5, 3>(
            "avg_pool3d",
            (input, input_layout),
            dispatch::parameters::<HephaestusBackend<P>, T, 3>(
                "avg_pool3d",
                [kernel_size; 3],
                [stride; 3],
                [padding; 3],
                [dilation; 3],
            )?,
            PoolingMode::Average,
            (output, output_layout),
        )
    }

    fn avg_pool3d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut Self::DeviceBuffer<T>,
        grad_input_layout: &Layout,
    ) -> Result<(), Self::Error> {
        dispatch::backward::<HephaestusBackend<P>, T, 5, 3>(
            "avg_pool3d_backward",
            (grad_out, grad_out_layout),
            None,
            dispatch::parameters::<HephaestusBackend<P>, T, 3>(
                "avg_pool3d_backward",
                [kernel_size; 3],
                [stride; 3],
                [padding; 3],
                [dilation; 3],
            )?,
            PoolingMode::Average,
            (grad_input, grad_input_layout),
        )
    }
}
