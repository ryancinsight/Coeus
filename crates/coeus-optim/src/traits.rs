// ── Optimizer trait ──

use coeus_core::{MoiraiBackend, Scalar};

/// Trait for parameter optimizers.
///
/// # Examples
///
/// ```
/// use coeus_autograd::{Parameter, Var};
/// use coeus_optim::{Optimizer, SGD};
/// use coeus_tensor::Tensor;
///
/// let x: Var<f32> = Var::new(
///     Tensor::from_slice(vec![1], &[1.0f32]).expect("construct tensor"),
///     true,
/// ).expect("construct variable");
/// x.set_grad(Tensor::from_slice(vec![1], &[-2.0f32]).expect("construct tensor"));
///
/// let mut opt: SGD<f32> = SGD::new(vec![Parameter::new(x.clone(), "x")], 0.1f32, 0.0f32)
///     .expect("construct SGD optimizer");
/// // `step`, `zero_grad`, and `set_lr` come from the `Optimizer` trait.
/// opt.step().expect("run optimizer step");
/// // x' = x - lr * grad = 1.0 - 0.1 * (-2.0) = 1.2
/// assert!((opt.params[0].tensor.as_slice()[0] - 1.2).abs() < 1e-5);
///
/// opt.zero_grad().expect("clear gradients");
/// assert_eq!(opt.params[0].grad().unwrap().as_slice(), &[0.0f32]);
///
/// opt.set_lr(0.5f32);
/// assert_eq!(opt.lr, 0.5f32);
/// ```
pub trait Optimizer<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Perform one optimization step using accumulated gradients.
    fn step(&mut self) -> Result<(), B::Error>;

    /// Zero all parameter gradients.
    fn zero_grad(&mut self) -> Result<(), B::Error>;

    /// Update the learning rate for all parameter groups.
    fn set_lr(&mut self, lr: T);

    /// Clip gradient L2 norms across all parameters to `max_norm`.
    ///
    /// Returns the pre-clip total L2 norm.
    ///
    /// # Errors
    /// Returns the backend storage error if a gradient cannot become uniquely
    /// mutable.
    fn clip_grad_norm(&mut self, max_norm: T) -> Result<T, B::Error>
    where
        B::DeviceBuffer<T>:
            coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>;
}
