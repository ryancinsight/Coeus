use backend::CpuBackend;
use dtype::{traits::FloatExt, DataType};
use storage::DenseStorage;
use tensor::Tensor;

use crate::core::error::Result;
use crate::core::module::Module;
use crate::core::parameter::Parameter;

/// 3D Adaptive Max Pooling layer.
///
/// Applies a 3D adaptive max pooling over an input signal composed of several input planes.
///
/// # Shape
/// - Input: `(N, C, D_in, H_in, W_in)`
/// - Output: `(N, C, D_out, H_out, W_out)` where `(D_out, H_out, W_out)` is specified by `output_size`
#[derive(Debug, Clone)]
pub struct AdaptiveMaxPool3d {
    /// Output size (depth, height, width)
    pub output_size: (usize, usize, usize),
}

impl AdaptiveMaxPool3d {
    /// Create a new AdaptiveMaxPool3d layer.
    pub fn new(output_size: (usize, usize, usize)) -> Self {
        assert!(
            output_size.0 > 0 && output_size.1 > 0 && output_size.2 > 0,
            "output_size must be > 0"
        );
        Self { output_size }
    }
}

impl<T: DataType + FloatExt + PartialOrd> Module<CpuBackend<T>, DenseStorage<T>, T>
    for AdaptiveMaxPool3d
{
    fn forward(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        let input_shape = input.shape().dims();
        assert_eq!(
            input_shape.len(),
            5,
            "Input must be 5D [N, C, D_in, H_in, W_in]"
        );

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let input_d = input_shape[2];
        let input_h = input_shape[3];
        let input_w = input_shape[4];

        let (output_d, output_h, output_w) = self.output_size;
        let mut output_data =
            Vec::with_capacity(batch_size * channels * output_d * output_h * output_w);
        let input_data = input.as_slice();

        for n in 0..batch_size {
            for c in 0..channels {
                for od in 0..output_d {
                    let d_start = (od * input_d) / output_d;
                    let d_end = ((od + 1) * input_d + output_d - 1) / output_d;

                    for oh in 0..output_h {
                        let h_start = (oh * input_h) / output_h;
                        let h_end = ((oh + 1) * input_h + output_h - 1) / output_h;

                        for ow in 0..output_w {
                            let w_start = (ow * input_w) / output_w;
                            let w_end = ((ow + 1) * input_w + output_w - 1) / output_w;

                            let mut max_val = T::from(f64::NEG_INFINITY).unwrap();

                            for id in d_start..d_end {
                                for ih in h_start..h_end {
                                    for iw in w_start..w_end {
                                        let idx = (((n * channels + c) * input_d + id) * input_h
                                            + ih)
                                            * input_w
                                            + iw;
                                        let val = input_data[idx];
                                        if val > max_val {
                                            max_val = val;
                                        }
                                    }
                                }
                            }
                            output_data.push(max_val);
                        }
                    }
                }
            }
        }

        Tensor::from_vec(
            output_data,
            &[batch_size, channels, output_d, output_h, output_w],
        )
        .map_err(Into::into)
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend<T>, DenseStorage<T>, T>> {
        Vec::new()
    }

    fn zero_grad(&mut self) {}

    fn train(&mut self, _mode: bool) {}

    fn name(&self) -> &str {
        "AdaptiveMaxPool3d"
    }

    fn clone_box(&self) -> Box<dyn Module<CpuBackend<T>, DenseStorage<T>, T>> {
        Box::new(self.clone())
    }
}
