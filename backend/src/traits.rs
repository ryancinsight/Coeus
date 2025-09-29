//! Backend traits for generic dtype operations.

use crate::{Dtype, Result, Device};
use crate::BackendData;
use num_traits::Float;

/// Core backend trait for device-agnostic tensor operations.
/// All backends (CPU, GPU, etc.) must implement this trait.
pub trait Backend<T: Dtype>: Clone + Send + Sync + std::default::Default + std::fmt::Debug {
    fn device(&self) -> Device;

    fn create_tensor_data(&self, data: Vec<T>, shape: Vec<usize>) -> Result<BackendData<T>>;

    fn zeros(&self, shape: Vec<usize>) -> Result<BackendData<T>>;

    fn ones(&self, shape: Vec<usize>) -> Result<BackendData<T>>;

    fn add(&self, a: &BackendData<T>, b: &BackendData<T>) -> Result<BackendData<T>>
    where T: std::ops::Add<Output = T> + Clone;

    fn sub(&self, a: &BackendData<T>, b: &BackendData<T>) -> Result<BackendData<T>>
    where T: std::ops::Sub<Output = T> + Clone;

    fn mul(&self, a: &BackendData<T>, b: &BackendData<T>) -> Result<BackendData<T>>
    where T: std::ops::Mul<Output = T> + Clone;

    fn div(&self, a: &BackendData<T>, b: &BackendData<T>) -> Result<BackendData<T>>
    where T: std::ops::Div<Output = T> + Clone;

    fn matmul(&self, a: &BackendData<T>, b: &BackendData<T>) -> Result<BackendData<T>>
    where T: Float + Clone;

    fn transpose(&self, tensor: &BackendData<T>, dim0: usize, dim1: usize) -> Result<BackendData<T>>;

    fn exp(&self, input: &BackendData<T>) -> Result<BackendData<T>>
    where T: Float + Clone;

    fn log(&self, input: &BackendData<T>) -> Result<BackendData<T>>
    where T: Float + Clone;

    fn sin(&self, input: &BackendData<T>) -> Result<BackendData<T>>
    where T: Float + Clone;

    fn cos(&self, input: &BackendData<T>) -> Result<BackendData<T>>
    where T: Float + Clone;

    fn sqrt(&self, input: &BackendData<T>) -> Result<BackendData<T>>
    where T: Float + Clone;

    fn pow(&self, input: &BackendData<T>, exponent: &BackendData<T>) -> Result<BackendData<T>>
    where T: Float + Clone;

    fn relu(&self, input: &BackendData<T>) -> Result<BackendData<T>>
    where T: std::cmp::PartialOrd + Clone;

    fn sigmoid(&self, input: &BackendData<T>) -> Result<BackendData<T>>
    where T: Float + Clone;

    fn tanh(&self, input: &BackendData<T>) -> Result<BackendData<T>>
    where T: Float + Clone;

    fn softmax(&self, input: &BackendData<T>, dim: usize) -> Result<BackendData<T>>
    where T: Float + Clone;

    fn sum_dim(&self, input: &BackendData<T>, dim: usize) -> Result<BackendData<T>>
    where T: std::ops::Add<Output = T> + Clone;

    fn mean_dim(&self, input: &BackendData<T>, dim: usize) -> Result<BackendData<T>>
    where T: Float + Clone;

    fn max_dim(&self, input: &BackendData<T>, dim: usize) -> Result<BackendData<T>>
    where T: std::cmp::PartialOrd + Clone;

    fn min_dim(&self, input: &BackendData<T>, dim: usize) -> Result<BackendData<T>>
    where T: std::cmp::PartialOrd + Clone;

    fn argmax(&self, input: &BackendData<T>, dim: usize) -> Result<BackendData<T>>
    where T: std::cmp::PartialOrd + Clone;

    fn argmin(&self, input: &BackendData<T>, dim: usize) -> Result<BackendData<T>>
    where T: std::cmp::PartialOrd + Clone;

    fn gather<U: Dtype + Float + Clone>(&self, dim: usize, input: &BackendData<U>, indices: &BackendData<i32>) -> Result<BackendData<U>>;

    fn take<U: Dtype + Float + Clone>(&self, input: &BackendData<U>, indices: &BackendData<i64>) -> Result<BackendData<U>>;

    fn add_scalar(&self, input: &BackendData<T>, scalar: T) -> Result<BackendData<T>>
    where T: std::ops::Add<Output = T> + Clone;

    fn mul_scalar(&self, input: &BackendData<T>, scalar: T) -> Result<BackendData<T>>
    where T: std::ops::Mul<Output = T> + Clone;

    fn sub_scalar(&self, input: &BackendData<T>, scalar: T) -> Result<BackendData<T>>
    where T: std::ops::Sub<Output = T> + Clone;

    fn div_scalar(&self, input: &BackendData<T>, scalar: T) -> Result<BackendData<T>>
    where T: std::ops::Div<Output = T> + Clone;

    fn full(&self, shape: Vec<usize>, value: T) -> Result<BackendData<T>>
    where T: Clone;

    fn from_vec(&self, data: Vec<T>, shape: Vec<usize>) -> Result<BackendData<T>>;

    fn reduce_mean(&self, tensor: &BackendData<T>, dim: usize) -> Result<BackendData<T>>
    where T: Float + Clone;

    fn reduce_var(&self, tensor: &BackendData<T>, dim: usize, mean: Option<&BackendData<T>>) -> Result<BackendData<T>>
    where T: Float + Clone;

    fn unsqueeze(&self, tensor: &BackendData<T>, dim: usize) -> Result<BackendData<T>>;

    fn expand(&self, tensor: &BackendData<T>, shape: Vec<usize>) -> Result<BackendData<T>>;

    fn bitwise_and(&self, a: &BackendData<T>, b: &BackendData<T>) -> Result<BackendData<T>>
    where T: std::ops::BitAnd<Output = T> + Copy;

    fn bitwise_or(&self, a: &BackendData<T>, b: &BackendData<T>) -> Result<BackendData<T>>
    where T: std::ops::BitOr<Output = T> + Copy;

    fn bitwise_xor(&self, a: &BackendData<T>, b: &BackendData<T>) -> Result<BackendData<T>>
    where T: std::ops::BitXor<Output = T> + Copy;

    fn bitwise_not(&self, a: &BackendData<T>) -> Result<BackendData<T>>
    where T: std::ops::Not<Output = T> + Copy;

    fn cast_to_i32(&self, input: &BackendData<T>) -> Result<BackendData<i32>>;

    fn cast_from_i32(&self, input: &BackendData<i32>) -> Result<BackendData<T>>;

    /// 1D convolution operation
    #[allow(clippy::too_many_arguments)]
    fn conv1d(
        &self,
        input: &BackendData<T>,
        weight: &BackendData<T>,
        bias: Option<&BackendData<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<BackendData<T>>
    where
        T: Float + Clone;

    /// Pad tensor with specified padding and value
    fn pad(&self, input: &BackendData<T>, padding: Vec<usize>, value: T) -> Result<BackendData<T>>
    where
        T: Clone;

    /// Compute gradient with respect to weight for 1D convolution
    fn conv1d_grad_weight(
        &self,
        input: &BackendData<T>,
        grad_output: &BackendData<T>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<BackendData<T>>
    where
        T: Float + Clone;

    // Additional methods for distributed and advanced operations
    fn allreduce(&self, input: &BackendData<T>, world_size: usize) -> Result<BackendData<T>>;

    fn upsample(&self, input: &BackendData<T>, scale: f32) -> Result<BackendData<T>>;

    fn l2_norm(&self, input: &BackendData<T>) -> Result<f32>;

    fn layernorm_backward(
        &self,
        grad_out: &BackendData<T>,
        input: &BackendData<T>,
        mean: f32,
        var: f32,
        gamma: Option<&BackendData<T>>,
        eps: f32,
    ) -> Result<BackendData<T>>;

    fn attention_backward(
        &self,
        grad_out: &BackendData<T>,
        query: &BackendData<T>,
        key: &BackendData<T>,
        value: &BackendData<T>,
    ) -> Result<(BackendData<T>, BackendData<T>, BackendData<T>)>;

    fn gelu(&self, input: &BackendData<T>) -> Result<BackendData<T>>
    where T: Float + Clone;

    fn attention(&self, query: &BackendData<T>, key: &BackendData<T>, value: &BackendData<T>) -> Result<BackendData<T>>;

    fn layernorm(&self, input: &BackendData<T>, mean: Option<f32>, var: Option<f32>, gamma: Option<f32>, beta: Option<f32>, eps: f32) -> Result<BackendData<T>>
    where T: Float + Clone;

    fn fused_batchnorm(&self, input: &BackendData<T>, mean: f32, var: f32, gamma: f32, beta: f32, eps: f32) -> Result<BackendData<T>>;

    fn adam_step(&self, m: &BackendData<T>, v: &BackendData<T>, grad: &BackendData<T>, lr: f32, beta1: f32, beta2: f32, eps: f32, t: f32) -> Result<BackendData<T>>;

    fn fused_adam(&self, m: &BackendData<T>, v: &BackendData<T>, grad: &BackendData<T>, lr: f32, beta1: f32, beta2: f32, eps: f32, t: f32) -> Result<BackendData<T>>;
    fn rmsprop(&self, v: &BackendData<T>, grad: &BackendData<T>, lr: f32, eps: f32) -> Result<BackendData<T>>
    where T: Float + Clone;

    fn pooling(&self, input: &BackendData<T>, kernel: Vec<usize>, stride: Vec<usize>, pool_type: &str) -> Result<BackendData<T>>;

    fn dropout(&self, input: &BackendData<T>, p: f32) -> Result<BackendData<T>>
    where T: Float + Clone;
    fn quantized_infer(&self, input: &BackendData<T>) -> Result<BackendData<T>>;
}

// Minimal impl - use existing CpuBackend, no new stubs to avoid regression