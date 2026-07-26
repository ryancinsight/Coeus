use crate::traits::Optimizer;
use coeus_autograd::Parameter;
use coeus_core::{Float, MoiraiBackend};
use coeus_tensor::Tensor;

/// RMSProp optimizer.
///
/// # Examples
///
/// ```
/// use coeus_autograd::{Parameter, Var};
/// use coeus_optim::{Optimizer, RMSProp};
/// use coeus_tensor::Tensor;
///
/// let x: Var<f32> = Var::new(Tensor::from_slice(vec![2], &[2.0f32, 3.0]), true);
/// x.set_grad(Tensor::from_slice(vec![2], &[1.0f32, -2.0]));
///
/// let mut opt = RMSProp::new(vec![Parameter::new(x.clone(), "x")], 0.1f32, 0.99f32, 1e-8f32);
/// opt.step();
/// // v = (1-alpha) * grad^2 = 0.01 * [1.0, 4.0] = [0.01, 0.04]
/// // update = lr * grad / (sqrt(v) + eps) = 0.1 * [1.0, -2.0] / [0.1, 0.2] = [1.0, -1.0]
/// // p' = [2.0, 3.0] - [1.0, -1.0] = [1.0, 4.0]
/// let updated = opt.params[0].var.tensor.as_slice();
/// assert!((updated[0] - 1.0).abs() < 1e-4);
/// assert!((updated[1] - 4.0).abs() < 1e-4);
/// ```
pub struct RMSProp<T: Float, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// List of tracked parameters.
    pub params: Vec<Parameter<T, B>>,
    /// Learning rate.
    pub lr: T,
    /// Decay rate.
    pub alpha: T,
    /// Small constant for numerical stability.
    pub eps: T,
    /// Mean square gradients (v) matching the parameters shape.
    pub v: Vec<Tensor<T, B>>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> RMSProp<T, B> {
    /// Create a new RMSProp optimizer.
    pub fn new(params: Vec<Parameter<T, B>>, lr: T, alpha: T, eps: T) -> Self {
        let backend = B::default();
        let mut v = Vec::with_capacity(params.len());
        for param in &params {
            v.push(Tensor::zeros_on(param.var.tensor.shape(), &backend));
        }

        Self {
            params,
            lr,
            alpha,
            eps,
            v,
        }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Optimizer<T, B> for RMSProp<T, B> {
    fn step(&mut self) {
        let backend = B::default();

        for (i, param) in self.params.iter_mut().enumerate() {
            if let Some(ref g) = param.var.grad {
                let grad_tensor = g.read();
                let v_tensor = &mut self.v[i];

                let (param_storage, param_layout) = param.var.tensor.storage_mut_and_layout();
                let (v_storage, v_layout) = v_tensor.storage_mut_and_layout();

                backend.rmsprop_step(
                    param_storage,
                    param_layout,
                    grad_tensor.storage(),
                    grad_tensor.layout(),
                    v_storage,
                    v_layout,
                    self.lr,
                    self.alpha,
                    self.eps,
                );
            }
        }
    }

    fn zero_grad(&mut self) {
        for p in &self.params {
            p.var.zero_grad();
        }
    }

    fn set_lr(&mut self, lr: T) {
        self.lr = lr;
    }

    fn clip_grad_norm(&mut self, max_norm: T) -> T
    where
        B::DeviceBuffer<T>:
            coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
    {
        crate::clip::clip_grad_norm_iter(
            self.params.iter().map(|parameter| &parameter.var),
            max_norm,
        )
    }
}
