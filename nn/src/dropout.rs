//! Dropout layer for regularization.

use backend::CpuBackend;
use dtype::{traits::FloatExt, DataType};
use storage::DenseStorage;
use tensor::Tensor;

use crate::error::Result;
use crate::module::Module;
use crate::parameter::Parameter;

/// Dropout layer for regularization.
///
/// During training, randomly zeros elements of the input tensor with probability `p`
/// using samples from a Bernoulli distribution. The outputs are scaled by `1/(1-p)`
/// during training to maintain the expected value.
///
/// During evaluation, the layer simply returns the input unchanged.
///
/// # Examples
/// ```rust
/// use nn::{Dropout, Module};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// // Create dropout layer with p=0.5
/// let mut dropout = Dropout::new(0.5);
///
/// // Set to training mode
/// <Dropout as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::train(&mut dropout, true);
///
/// // Input: [2, 3]
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0),
///          Float32::new(4.0), Float32::new(5.0), Float32::new(6.0)],
///     &[2, 3]
/// ).unwrap();
///
/// // Output: Some elements zeroed, others scaled by 2.0
/// let output = <Dropout as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(&dropout, &input).unwrap();
/// assert_eq!(output.shape().dims(), &[2, 3]);
///
/// // Set to evaluation mode
/// <Dropout as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::train(&mut dropout, false);
/// let output_eval = <Dropout as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(&dropout, &input).unwrap();
/// // In eval mode, output == input
/// ```
#[derive(Debug, Clone)]
pub struct Dropout {
    /// Dropout probability (0.0 to 1.0)
    pub p: f64,
    /// Training mode flag
    pub training: bool,
}

impl Dropout {
    /// Create a new Dropout layer.
    ///
    /// # Arguments
    /// * `p` - Dropout probability (0.0 to 1.0)
    ///
    /// # Panics
    /// Panics if `p` is not in [0.0, 1.0]
    pub fn new(p: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "Dropout probability must be in [0.0, 1.0], got {}",
            p
        );

        Self {
            p,
            training: true, // Default to training mode
        }
    }
}

impl Default for Dropout {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl<T: DataType + FloatExt> Module<CpuBackend<T>, DenseStorage<T>, T> for Dropout {
    fn forward(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        if !self.training || self.p == 0.0 {
            // Evaluation mode or p=0: pass through unchanged
            return Ok(input.clone());
        }

        if self.p == 1.0 {
            // p=1: zero out everything
            let zeros = vec![T::zero(); input.as_slice().len()];
            return Tensor::from_vec(zeros, input.shape().dims()).map_err(Into::into);
        }

        // Training mode: apply inverted dropout
        // mask ~ Bernoulli(1 - p)
        // output = input * mask / (1 - p)

        let scale = T::from(1.0 / (1.0 - self.p)).unwrap();
        let keep_prob_f64 = 1.0 - self.p;

        let output_data: Vec<T> = input
            .as_slice()
            .iter()
            .map(|&x| {
                let random_val = rand::random::<f64>();
                if random_val < keep_prob_f64 {
                    x * scale // Keep and scale
                } else {
                    T::zero() // Drop
                }
            })
            .collect();

        Tensor::from_vec(output_data, input.shape().dims()).map_err(Into::into)
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend<T>, DenseStorage<T>, T>> {
        vec![] // Dropout has no learnable parameters
    }

    fn zero_grad(&mut self) {
        // No-op: Dropout has no parameters
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn name(&self) -> &str {
        "Dropout"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use dtype::float::Float32;

    #[test]
    fn test_dropout_eval_mode() {
        let mut dropout = Dropout::new(0.5);
        <Dropout as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::train(
            &mut dropout,
            false,
        ); // Evaluation mode

        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(1.0),
                Float32::new(2.0),
                Float32::new(3.0),
                Float32::new(4.0),
            ],
            &[4],
        )
        .unwrap();

        let output =
            <Dropout as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
                &dropout, &input,
            )
            .unwrap();
        let input_data: Vec<f32> = input.as_slice().iter().map(|x: &Float32| x.get()).collect();
        let output_data: Vec<f32> = output
            .as_slice()
            .iter()
            .map(|x: &Float32| x.get())
            .collect();

        // In eval mode, output should equal input
        for (i, o) in input_data.iter().zip(output_data.iter()) {
            assert_relative_eq!(i, o);
        }
    }

    #[test]
    fn test_dropout_training_mode() {
        let mut dropout = Dropout::new(0.5);
        <Dropout as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::train(
            &mut dropout,
            true,
        ); // Training mode

        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0); 100], // 100 ones
            &[100],
        )
        .unwrap();

        let output =
            <Dropout as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
                &dropout, &input,
            )
            .unwrap();
        let output_data: Vec<f32> = output
            .as_slice()
            .iter()
            .map(|x: &Float32| x.get())
            .collect();

        // Count zeros (dropped) and non-zeros (kept)
        let zeros = output_data.iter().filter(|&&x| x == 0.0).count();
        let non_zeros = output_data.iter().filter(|&&x| x != 0.0).count();

        // With p=0.5, expect roughly 50% zeros and 50% non-zeros
        // Allow 20% tolerance for randomness
        assert!(
            (30..=70).contains(&zeros),
            "Expected ~50 zeros, got {}",
            zeros
        );
        assert!(
            (30..=70).contains(&non_zeros),
            "Expected ~50 non-zeros, got {}",
            non_zeros
        );

        // Non-zero values should be scaled by 1/(1-p) = 2.0
        for &val in output_data.iter().filter(|&&x| x != 0.0) {
            assert_relative_eq!(val, 2.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_dropout_p_zero() {
        let mut dropout = Dropout::new(0.0);
        <Dropout as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::train(
            &mut dropout,
            true,
        );

        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        )
        .unwrap();

        let output =
            <Dropout as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
                &dropout, &input,
            )
            .unwrap();
        let input_data: Vec<f32> = input.as_slice().iter().map(|x: &Float32| x.get()).collect();
        let output_data: Vec<f32> = output
            .as_slice()
            .iter()
            .map(|x: &Float32| x.get())
            .collect();

        // With p=0, no dropout should occur
        for (i, o) in input_data.iter().zip(output_data.iter()) {
            assert_relative_eq!(i, o);
        }
    }

    #[test]
    fn test_dropout_p_one() {
        let mut dropout = Dropout::new(1.0);
        <Dropout as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::train(
            &mut dropout,
            true,
        );

        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        )
        .unwrap();

        let output =
            <Dropout as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
                &dropout, &input,
            )
            .unwrap();
        let output_data: Vec<f32> = output
            .as_slice()
            .iter()
            .map(|x: &Float32| x.get())
            .collect();

        // With p=1, all values should be zero
        for &val in &output_data {
            assert_relative_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_dropout_no_parameters() {
        let dropout = Dropout::new(0.5);
        let params =
            <Dropout as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::parameters(
                &dropout,
            );
        assert_eq!(params.len(), 0);
    }

    #[test]
    #[should_panic(expected = "Dropout probability must be in [0.0, 1.0], got 1.5")]
    fn test_dropout_invalid_probability() {
        let _ = Dropout::new(1.5); // Invalid probability
    }
}

/// Dropout2d layer for spatial regularization in CNNs.
///
/// Randomly zeros entire channels of the input tensor with probability `p`.
/// This is particularly effective for convolutional layers where adjacent pixels
/// are highly correlated. By dropping entire feature maps, the network is forced
/// to learn more robust features.
///
/// During training, randomly zeros entire channels with probability `p`.
/// The outputs are scaled by `1/(1-p)` during training to maintain the expected value.
///
/// During evaluation, the layer simply returns the input unchanged.
///
/// # Shape
/// - Input: `(N, C, H, W)` where N is batch size, C is channels, H is height, W is width
/// - Output: Same shape as input
///
/// # Examples
/// ```rust
/// use nn::{Dropout2d, Module};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// // Create dropout2d layer with p=0.5
/// let mut dropout = Dropout2d::new(0.5);
///
/// // Set to training mode
/// <Dropout2d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::train(&mut dropout, true);
///
/// // Input: [2, 64, 32, 32]
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[2, 64, 32, 32]).unwrap();
///
/// // Output: Some channels zeroed, others scaled by 2.0
/// let output = <Dropout2d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(&dropout, &input).unwrap();
/// assert_eq!(output.shape().dims(), &[2, 64, 32, 32]);
/// ```
///
/// # References
/// - Tompson et al. (2015): "Efficient Object Localization Using Convolutional Networks"
/// - Srivastava et al. (2014): "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
#[derive(Debug, Clone)]
pub struct Dropout2d {
    /// Dropout probability (0.0 to 1.0)
    pub p: f64,
    /// Training mode flag
    pub training: bool,
}

impl Dropout2d {
    /// Create a new Dropout2d layer.
    ///
    /// # Arguments
    /// * `p` - Dropout probability (0.0 to 1.0)
    ///
    /// # Panics
    /// Panics if `p` is not in [0.0, 1.0]
    pub fn new(p: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "Dropout probability must be in [0.0, 1.0], got {}",
            p
        );

        Self { p, training: true }
    }
}

impl<T: DataType + FloatExt> Module<CpuBackend<T>, DenseStorage<T>, T> for Dropout2d {
    fn forward(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        let input_shape = input.shape().dims();
        assert_eq!(input_shape.len(), 4, "Input must be 4D [N, C, H, W]");

        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }

        if self.p == 1.0 {
            let zeros = vec![T::zero(); input.as_slice().len()];
            return Tensor::from_vec(zeros, input_shape).map_err(Into::into);
        }

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let height = input_shape[2];
        let width = input_shape[3];
        let spatial_size = height * width;

        let scale = T::from(1.0 / (1.0 - self.p)).unwrap();
        let keep_prob_f64 = 1.0 - self.p;

        let input_data = input.as_slice();
        let mut output_data = Vec::with_capacity(input_data.len());

        // For each batch and channel, decide whether to drop the entire channel
        for n in 0..batch_size {
            for c in 0..channels {
                let random_val = rand::random::<f64>();
                let keep_channel = random_val < keep_prob_f64;

                for _spatial in 0..spatial_size {
                    let idx = ((n * channels + c) * height * width) + _spatial;
                    if keep_channel {
                        output_data.push(input_data[idx] * scale);
                    } else {
                        output_data.push(T::zero());
                    }
                }
            }
        }

        Tensor::from_vec(output_data, input_shape).map_err(Into::into)
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend<T>, DenseStorage<T>, T>> {
        vec![]
    }

    fn zero_grad(&mut self) {
        // No-op
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn name(&self) -> &str {
        "Dropout2d"
    }
}

/// Dropout3d layer for spatial regularization in 3D CNNs.
///
/// Randomly zeros entire channels of the input tensor with probability `p`.
/// This is the 3D extension of Dropout2d, designed for volumetric data and video processing.
///
/// During training, randomly zeros entire channels with probability `p`.
/// The outputs are scaled by `1/(1-p)` during training to maintain the expected value.
///
/// During evaluation, the layer simply returns the input unchanged.
///
/// # Shape
/// - Input: `(N, C, D, H, W)` where N is batch size, C is channels, D is depth, H is height, W is width
/// - Output: Same shape as input
///
/// # Examples
/// ```rust
/// use nn::{Dropout3d, Module};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// // Create dropout3d layer with p=0.5
/// let mut dropout = Dropout3d::new(0.5);
///
/// // Set to training mode
/// <Dropout3d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::train(&mut dropout, true);
///
/// // Input: [1, 64, 16, 32, 32] (video/volumetric data)
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 64, 16, 32, 32]).unwrap();
///
/// // Output: Some channels zeroed, others scaled by 2.0
/// let output = <Dropout3d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(&dropout, &input).unwrap();
/// assert_eq!(output.shape().dims(), &[1, 64, 16, 32, 32]);
/// ```
///
/// # References
/// - Tompson et al. (2015): "Efficient Object Localization Using Convolutional Networks"
/// - Tran et al. (2015): "Learning Spatiotemporal Features with 3D Convolutional Networks" (C3D)
#[derive(Debug, Clone)]
pub struct Dropout3d {
    /// Dropout probability (0.0 to 1.0)
    pub p: f64,
    /// Training mode flag
    pub training: bool,
}

impl Dropout3d {
    /// Create a new Dropout3d layer.
    ///
    /// # Arguments
    /// * `p` - Dropout probability (0.0 to 1.0)
    ///
    /// # Panics
    /// Panics if `p` is not in [0.0, 1.0]
    pub fn new(p: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "Dropout probability must be in [0.0, 1.0], got {}",
            p
        );

        Self { p, training: true }
    }
}

impl<T: DataType + FloatExt> Module<CpuBackend<T>, DenseStorage<T>, T> for Dropout3d {
    fn forward(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        let input_shape = input.shape().dims();
        assert_eq!(input_shape.len(), 5, "Input must be 5D [N, C, D, H, W]");

        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }

        if self.p == 1.0 {
            let zeros = vec![T::zero(); input.as_slice().len()];
            return Tensor::from_vec(zeros, input_shape).map_err(Into::into);
        }

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let depth = input_shape[2];
        let height = input_shape[3];
        let width = input_shape[4];
        let spatial_size = depth * height * width;

        let scale = T::from(1.0 / (1.0 - self.p)).unwrap();
        let keep_prob_f64 = 1.0 - self.p;

        let input_data = input.as_slice();
        let mut output_data = Vec::with_capacity(input_data.len());

        // For each batch and channel, decide whether to drop the entire channel
        for n in 0..batch_size {
            for c in 0..channels {
                let random_val = rand::random::<f64>();
                let keep_channel = random_val < keep_prob_f64;

                for _spatial in 0..spatial_size {
                    let idx = ((n * channels + c) * depth * height * width) + _spatial;
                    if keep_channel {
                        output_data.push(input_data[idx] * scale);
                    } else {
                        output_data.push(T::zero());
                    }
                }
            }
        }

        Tensor::from_vec(output_data, input_shape).map_err(Into::into)
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend<T>, DenseStorage<T>, T>> {
        vec![]
    }

    fn zero_grad(&mut self) {
        // No-op
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn name(&self) -> &str {
        "Dropout3d"
    }
}

#[cfg(test)]
mod tests_dropout2d {
    use super::*;
    use dtype::float::Float32;

    #[test]
    fn test_dropout2d_eval_mode() {
        let mut dropout = Dropout2d::new(0.5);
        <Dropout2d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::train(
            &mut dropout,
            false,
        );

        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[2, 3, 4, 4])
                .unwrap();
        let output =
            <Dropout2d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
                &dropout, &input,
            )
            .unwrap();

        // In eval mode, output should equal input
        let input_data: Vec<f32> = input.as_slice().iter().map(|x: &Float32| x.get()).collect();
        let output_data: Vec<f32> = output
            .as_slice()
            .iter()
            .map(|x: &Float32| x.get())
            .collect();

        assert_eq!(input_data, output_data);
    }

    #[test]
    fn test_dropout2d_training_mode() {
        let mut dropout = Dropout2d::new(0.5);
        <Dropout2d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::train(
            &mut dropout,
            true,
        );

        // Input: [1, 10, 8, 8] (10 channels)
        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 10, 8, 8])
                .unwrap();
        let output =
            <Dropout2d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
                &dropout, &input,
            )
            .unwrap();

        assert_eq!(output.shape().dims(), &[1, 10, 8, 8]);

        // Check that entire channels are either all zero or all scaled
        let output_data: Vec<f32> = output
            .as_slice()
            .iter()
            .map(|x: &Float32| x.get())
            .collect();

        let channel_size = 8 * 8;
        for c in 0..10 {
            let channel_start = c * channel_size;
            let channel_end = (c + 1) * channel_size;
            let channel_data = &output_data[channel_start..channel_end];

            // Check if all values in channel are the same (either all 0 or all 2.0)
            let first_val = channel_data[0];
            assert!(channel_data.iter().all(|&x| (x - first_val).abs() < 1e-5));
            assert!(first_val == 0.0 || (first_val - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dropout2d_p_zero() {
        let mut dropout = Dropout2d::new(0.0);
        <Dropout2d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::train(
            &mut dropout,
            true,
        );

        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 3, 4, 4])
                .unwrap();
        let output =
            <Dropout2d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
                &dropout, &input,
            )
            .unwrap();

        let input_data: Vec<f32> = input.as_slice().iter().map(|x: &Float32| x.get()).collect();
        let output_data: Vec<f32> = output
            .as_slice()
            .iter()
            .map(|x: &Float32| x.get())
            .collect();

        assert_eq!(input_data, output_data);
    }

    #[test]
    fn test_dropout2d_p_one() {
        let mut dropout = Dropout2d::new(1.0);
        <Dropout2d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::train(
            &mut dropout,
            true,
        );

        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 3, 4, 4])
                .unwrap();
        let output =
            <Dropout2d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
                &dropout, &input,
            )
            .unwrap();

        let output_data: Vec<f32> = output
            .as_slice()
            .iter()
            .map(|x: &Float32| x.get())
            .collect();
        assert!(output_data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_dropout2d_cnn_regularization() {
        let mut dropout = Dropout2d::new(0.5);
        <Dropout2d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::train(
            &mut dropout,
            true,
        );

        // ResNet-style: [4, 256, 14, 14]
        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[4, 256, 14, 14])
                .unwrap();
        let output =
            <Dropout2d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
                &dropout, &input,
            )
            .unwrap();

        assert_eq!(output.shape().dims(), &[4, 256, 14, 14]);
    }

    #[test]
    #[should_panic(expected = "Input must be 4D [N, C, H, W]")]
    fn test_dropout2d_invalid_input_shape() {
        let mut dropout = Dropout2d::new(0.5);
        <Dropout2d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::train(
            &mut dropout,
            true,
        );

        // Invalid 3D input
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[2, 3, 4])
            .unwrap();
        let _ = <Dropout2d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
            &dropout, &input,
        );
    }
}

#[cfg(test)]
mod tests_dropout3d {
    use super::*;
    use dtype::float::Float32;

    #[test]
    fn test_dropout3d_eval_mode() {
        let mut dropout = Dropout3d::new(0.5);
        <Dropout3d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::train(
            &mut dropout,
            false,
        );

        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 3, 4, 4, 4])
                .unwrap();
        let output =
            <Dropout3d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
                &dropout, &input,
            )
            .unwrap();

        let input_data: Vec<f32> = input.as_slice().iter().map(|x: &Float32| x.get()).collect();
        let output_data: Vec<f32> = output
            .as_slice()
            .iter()
            .map(|x: &Float32| x.get())
            .collect();

        assert_eq!(input_data, output_data);
    }

    #[test]
    fn test_dropout3d_training_mode() {
        let mut dropout = Dropout3d::new(0.5);
        <Dropout3d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::train(
            &mut dropout,
            true,
        );

        // Input: [1, 8, 4, 4, 4] (8 channels)
        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 8, 4, 4, 4])
                .unwrap();
        let output =
            <Dropout3d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
                &dropout, &input,
            )
            .unwrap();

        assert_eq!(output.shape().dims(), &[1, 8, 4, 4, 4]);

        // Check that entire channels are either all zero or all scaled
        let output_data: Vec<f32> = output
            .as_slice()
            .iter()
            .map(|x: &Float32| x.get())
            .collect();

        let channel_size = 4 * 4 * 4;
        for c in 0..8 {
            let channel_start = c * channel_size;
            let channel_end = (c + 1) * channel_size;
            let channel_data = &output_data[channel_start..channel_end];

            // Check if all values in channel are the same (either all 0 or all 2.0)
            let first_val = channel_data[0];
            assert!(channel_data.iter().all(|&x| (x - first_val).abs() < 1e-5));
            assert!(first_val == 0.0 || (first_val - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dropout3d_p_zero() {
        let mut dropout = Dropout3d::new(0.0);
        <Dropout3d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::train(
            &mut dropout,
            true,
        );

        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 3, 4, 4, 4])
                .unwrap();
        let output =
            <Dropout3d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
                &dropout, &input,
            )
            .unwrap();

        let input_data: Vec<f32> = input.as_slice().iter().map(|x: &Float32| x.get()).collect();
        let output_data: Vec<f32> = output
            .as_slice()
            .iter()
            .map(|x: &Float32| x.get())
            .collect();

        assert_eq!(input_data, output_data);
    }

    #[test]
    fn test_dropout3d_p_one() {
        let mut dropout = Dropout3d::new(1.0);
        <Dropout3d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::train(
            &mut dropout,
            true,
        );

        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 3, 4, 4, 4])
                .unwrap();
        let output =
            <Dropout3d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
                &dropout, &input,
            )
            .unwrap();

        let output_data: Vec<f32> = output
            .as_slice()
            .iter()
            .map(|x: &Float32| x.get())
            .collect();
        assert!(output_data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_dropout3d_video_classification() {
        let mut dropout = Dropout3d::new(0.5);
        <Dropout3d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::train(
            &mut dropout,
            true,
        );

        // C3D-style: [2, 64, 16, 32, 32]
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[
            2, 64, 16, 32, 32,
        ])
        .unwrap();
        let output =
            <Dropout3d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
                &dropout, &input,
            )
            .unwrap();

        assert_eq!(output.shape().dims(), &[2, 64, 16, 32, 32]);
    }

    #[test]
    #[should_panic(expected = "Input must be 5D [N, C, D, H, W]")]
    fn test_dropout3d_invalid_input_shape() {
        let mut dropout = Dropout3d::new(0.5);
        <Dropout3d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::train(
            &mut dropout,
            true,
        );

        // Invalid 4D input
        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[2, 3, 4, 4])
                .unwrap();
        let _ = <Dropout3d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
            &dropout, &input,
        );
    }
}

#[cfg(test)]
mod dropout_forward_var_tests {
    // Empty test module placeholder
}
