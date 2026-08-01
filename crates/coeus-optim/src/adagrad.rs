use crate::traits::Optimizer;
use coeus_autograd::Parameter;
use coeus_core::{Float, MoiraiBackend};
use coeus_tensor::Tensor;

/// AdaGrad optimizer.
///
/// # Examples
///
/// ```
/// use coeus_autograd::{Parameter, Var};
/// use coeus_optim::{AdaGrad, Optimizer};
/// use coeus_tensor::Tensor;
///
/// let x: Var<f32> = Var::new(Tensor::from_slice(vec![2], &[2.0f32, 3.0]), true);
/// x.set_grad(Tensor::from_slice(vec![2], &[1.0f32, -2.0]));
///
/// let mut opt = AdaGrad::new(vec![Parameter::new(x.clone(), "x")], 0.1f32, 1e-6f32);
/// opt.step().unwrap();
/// // history = grad^2 = [1.0, 4.0]; denom = sqrt(history) + eps ≈ [1.0, 2.0]
/// // update = lr * grad / denom ≈ 0.1 * [1.0, -1.0] = [0.1, -0.1]
/// // p' = [2.0, 3.0] - [0.1, -0.1] = [1.9, 3.1]
/// let updated = opt.params[0].var.tensor.as_slice();
/// assert!((updated[0] - 1.9).abs() < 1e-4);
/// assert!((updated[1] - 3.1).abs() < 1e-4);
/// ```
pub struct AdaGrad<T: Float, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Parameters to optimize.
    pub params: Vec<Parameter<T, B>>,
    /// Learning rate.
    pub lr: T,
    /// Small term to avoid division by zero.
    pub eps: T,
    /// Sum of squares of historical gradients.
    history: Vec<Tensor<T, B>>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> AdaGrad<T, B> {
    /// Create AdaGrad optimizer.
    pub fn new(params: Vec<Parameter<T, B>>, lr: T, eps: T) -> Self {
        let backend = B::default();
        let history = params
            .iter()
            .map(|p| Tensor::zeros_on(p.var.tensor.shape(), &backend))
            .collect();
        Self {
            params,
            lr,
            eps,
            history,
        }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + coeus_ops::OptimizerOps<T> + Default> Optimizer<T, B>
    for AdaGrad<T, B>
{
    fn step(&mut self) -> Result<(), B::Error> {
        let backend = B::default();

        for (i, param) in self.params.iter_mut().enumerate() {
            if let Some(ref g) = param.var.grad {
                let grad_tensor = g.read();
                let history_tensor = &mut self.history[i];

                let (param_storage, param_layout) = param.var.tensor.storage_mut_and_layout();
                let (history_storage, history_layout) = history_tensor.storage_mut_and_layout();

                backend.adagrad_step(
                    param_storage,
                    param_layout,
                    grad_tensor.storage(),
                    grad_tensor.layout(),
                    history_storage,
                    history_layout,
                    self.lr,
                    self.eps,
                )?;
            }
        }
        Ok(())
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
