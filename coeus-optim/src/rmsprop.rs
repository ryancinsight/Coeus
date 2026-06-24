use crate::traits::Optimizer;
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend};
use coeus_tensor::Tensor;

/// RMSProp optimizer.
pub struct RMSProp<T: Float, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// List of tracked parameters.
    pub params: Vec<Var<T, B>>,
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
    pub fn new(params: Vec<Var<T, B>>, lr: T, alpha: T, eps: T) -> Self {
        let backend = B::default();
        let mut v = Vec::with_capacity(params.len());
        for param in &params {
            v.push(Tensor::zeros_on(param.tensor.shape(), &backend));
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
            if let Some(ref g) = param.grad {
                let grad_tensor = g.read();
                let v_tensor = &mut self.v[i];

                let (param_storage, param_layout) = param.tensor.storage_mut_and_layout();
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
            p.zero_grad();
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
        crate::clip::clip_grad_norm(&self.params, max_norm)
    }
}
