use backend::CpuBackend;
use dtype::{traits::FloatExt, DataType};
use storage::DenseStorage;
use tensor::Tensor;

use crate::core::error::Result;
use crate::core::module::Module;
use crate::core::parameter::Parameter;

/// 1D Adaptive Max Pooling layer.
///
/// Applies a 1D adaptive max pooling over an input signal composed of several input planes.
///
/// # Shape
/// - Input: `(N, C, L_in)`
/// - Output: `(N, C, L_out)` where `L_out` is specified by `output_size`
#[derive(Debug, Clone)]
pub struct AdaptiveMaxPool1d {
    /// Output size
    pub output_size: usize,
}

impl AdaptiveMaxPool1d {
    /// Create a new AdaptiveMaxPool1d layer.
    pub fn new(output_size: usize) -> Self {
        assert!(output_size > 0, "output_size must be > 0");
        Self { output_size }
    }
}

impl<T: DataType + FloatExt + PartialOrd> Module<CpuBackend<T>, DenseStorage<T>, T>
    for AdaptiveMaxPool1d
{
    fn forward(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        let input_shape = input.shape().dims();
        assert_eq!(input_shape.len(), 3, "Input must be 3D [N, C, L_in]");

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let input_l = input_shape[2];

        let output_l = self.output_size;
        let mut output_data = Vec::with_capacity(batch_size * channels * output_l);
        let input_data = input.as_slice();

        for n in 0..batch_size {
            for c in 0..channels {
                for ol in 0..output_l {
                    let start = (ol * input_l) / output_l;
                    let end = ((ol + 1) * input_l + output_l - 1) / output_l;

                    let mut max_val = T::from(f64::NEG_INFINITY).unwrap();
                    for il in start..end {
                        let idx = (n * channels + c) * input_l + il;
                        let val = input_data[idx];
                        if val > max_val {
                            max_val = val;
                        }
                    }
                    output_data.push(max_val);
                }
            }
        }

        Tensor::from_vec(output_data, &[batch_size, channels, output_l]).map_err(Into::into)
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend<T>, DenseStorage<T>, T>> {
        Vec::new()
    }

    fn zero_grad(&mut self) {}

    fn train(&mut self, _mode: bool) {}

    fn name(&self) -> &str {
        "AdaptiveMaxPool1d"
    }

    fn clone_box(&self) -> Box<dyn Module<CpuBackend<T>, DenseStorage<T>, T>> {
        Box::new(self.clone())
    }
}
