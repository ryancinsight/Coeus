//! Depthwise three-dimensional convolution.

use super::{Conv3d, ConvParams};
use crate::Module;
use coeus_autograd::{add, cat, reshape, slice, Var};
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Float, MoiraiBackend};
use coeus_tensor::Tensor;

/// Channel-independent 3-D convolution with one learned kernel per channel.
#[derive(Clone)]
pub struct DepthwiseConv3d<
    T: Float,
    B: coeus_ops::BackendOps<T> + Default = MoiraiBackend,
> {
    /// Kernels with shape `[channels, 1, kernel, kernel, kernel]`.
    pub weight: Var<T, B>,
    /// Optional channel bias with shape `[channels]`.
    pub bias: Option<Var<T, B>>,
    /// Number of input and output channels.
    pub channels: usize,
    /// Isotropic kernel side length.
    pub kernel_size: usize,
    /// Isotropic stride.
    pub stride: usize,
    /// Symmetric zero padding.
    pub padding: usize,
    /// Isotropic kernel dilation.
    pub dilation: usize,
}

impl<T, B> DepthwiseConv3d<T, B>
where
    T: Float,
    B: coeus_ops::BackendOps<T> + Default,
{
    /// Construct a depthwise convolution with unit stride and dilation.
    #[must_use]
    pub fn new(channels: usize, kernel_size: usize, padding: usize, bias: bool) -> Self {
        Self::with_params(channels, kernel_size, 1, padding, 1, bias)
    }

    /// Construct a depthwise convolution with explicit spatial parameters.
    #[must_use]
    pub fn with_params(
        channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        bias: bool,
    ) -> Self {
        assert!(channels > 0, "DepthwiseConv3d: channels must be positive");
        assert!(stride > 0, "DepthwiseConv3d: stride must be positive");
        assert!(dilation > 0, "DepthwiseConv3d: dilation must be positive");
        let backend = B::default();
        Self {
            weight: Var::new(
                Tensor::ones_on([channels, 1, kernel_size, kernel_size, kernel_size], &backend),
                true,
            ),
            bias: bias.then(|| Var::new(Tensor::zeros_on([channels], &backend), true)),
            channels,
            kernel_size,
            stride,
            padding,
            dilation,
        }
    }
}

impl<T, B> Module<T, B> for DepthwiseConv3d<T, B>
where
    T: Float,
    B: coeus_ops::BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    fn parameters(&self) -> Vec<Var<T, B>> {
        let mut parameters = vec![self.weight.clone()];
        parameters.extend(self.bias.iter().cloned());
        parameters
    }

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        let shape = input.tensor.shape();
        assert_eq!(shape.len(), 5, "DepthwiseConv3d: input must have rank 5");
        assert_eq!(shape[1], self.channels, "DepthwiseConv3d: channel mismatch");
        let params = ConvParams::new(1, 1, self.kernel_size, self.stride, self.padding, self.dilation);
        let outputs: Vec<_> = (0..self.channels)
            .map(|channel| {
                let channel_input = slice(
                    input,
                    &[
                        (0, shape[0]),
                        (channel, channel + 1),
                        (0, shape[2]),
                        (0, shape[3]),
                        (0, shape[4]),
                    ],
                );
                let channel_weight = slice(
                    &self.weight,
                    &[
                        (channel, channel + 1),
                        (0, 1),
                        (0, self.kernel_size),
                        (0, self.kernel_size),
                        (0, self.kernel_size),
                    ],
                );
                let output = Conv3d::from_vars(channel_weight, None, params).forward(&channel_input);
                self.bias.as_ref().map_or(output.clone(), |bias| {
                    let channel_bias = reshape(
                        &slice(bias, &[(channel, channel + 1)]),
                        [1, 1, 1, 1, 1],
                    );
                    add(&output, &channel_bias)
                })
            })
            .collect();
        cat(&outputs.iter().collect::<Vec<_>>(), 1)
    }

    fn load_parameters(&mut self, parameters: &[Var<T, B>]) {
        self.weight = parameters[0].clone();
        if self.bias.is_some() {
            self.bias = Some(parameters[1].clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::MoiraiBackend;

    #[test]
    fn applies_independent_channel_kernels_and_gradients() {
        let backend = MoiraiBackend::new();
        let mut convolution = DepthwiseConv3d::<f32>::new(2, 1, 0, true);
        convolution.weight = Var::new(
            Tensor::from_slice_on([2, 1, 1, 1, 1], &[2.0, 3.0], &backend),
            true,
        );
        convolution.bias = Some(Var::new(
            Tensor::from_slice_on([2], &[1.0, -1.0], &backend),
            true,
        ));
        let input = Var::new(
            Tensor::from_slice_on([1, 2, 1, 1, 2], &[4.0, 5.0, 6.0, 7.0], &backend),
            true,
        );

        let output = convolution.forward(&input);

        assert_eq!(output.tensor.shape(), &[1, 2, 1, 1, 2]);
        assert_eq!(output.tensor.as_slice(), &[9.0, 11.0, 17.0, 20.0]);
        output.backward();
        assert_eq!(input.grad().expect("input gradient").as_slice(), &[2.0, 2.0, 3.0, 3.0]);
        assert!(convolution.weight.grad().is_some());
        assert!(convolution.bias.as_ref().expect("bias enabled").grad().is_some());
    }
}
