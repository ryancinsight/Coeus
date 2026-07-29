use crate::module::{Module, ModuleError};
use coeus_autograd::Var;
use coeus_core::Float;

/// Functional PReLU activation with a learnable per-channel (or
/// shared-scalar) weight — see [`coeus_autograd::prelu`] for the composition
/// and gradient derivation.
#[inline]
pub fn prelu<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    weight: &Var<T, B>,
) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    coeus_autograd::prelu(input, weight)
}

/// PReLU activation module with a learnable weight (PyTorch/Burn semantics:
/// `num_parameters = 1` for one shared slope, or the channel count for a
/// per-channel slope broadcasting against dim 1 of the input).
#[derive(Clone)]
pub struct PReLU<T: Float, B: coeus_ops::BackendOps<T> + Default = coeus_core::MoiraiBackend> {
    /// Learnable slope(s); shape `[1]` (shared) or `[num_parameters]`
    /// (per-channel).
    pub weight: Var<T, B>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> PReLU<T, B> {
    /// Create a PReLU module with `num_parameters` learnable slopes (`1` for
    /// a shared scalar, or the channel count for per-channel slopes), each
    /// initialized to `init` (PyTorch/Burn default: `0.25`).
    pub fn new(num_parameters: usize, init: f64) -> Self {
        let backend = B::default();
        let weight = Var::new(
            coeus_tensor::Tensor::full_on([num_parameters], T::from_f64(init), &backend),
            true,
        );
        Self { weight }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Default for PReLU<T, B> {
    fn default() -> Self {
        Self::new(1, 0.25)
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for PReLU<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![self.weight.clone()]
    }

    #[inline]
    fn load_parameters(&mut self, params: &[Var<T, B>]) {
        self.weight = params[0].clone();
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        Ok(prelu(input, &self.weight))
    }
}
