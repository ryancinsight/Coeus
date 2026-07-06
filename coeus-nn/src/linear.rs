use crate::module::Module;
use coeus_autograd::Var;
use coeus_core::{MoiraiBackend, Scalar};
use coeus_tensor::Tensor;

/// Fully-connected linear layer.
#[derive(Clone)]
pub struct Linear<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Weight matrix: `[out_features × in_features]`.
    pub weight: Var<T, B>,
    /// Bias vector: `[out_features]`.
    pub bias: Option<Var<T, B>>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Linear<T, B> {
    /// Create a Linear layer with given input/output features.
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let backend = B::default();
        let w_shape = [out_features, in_features];
        let w_tensor = Tensor::ones_on(w_shape, &backend);
        let weight = Var::new(w_tensor, true);

        let bias_var = if bias {
            let b_shape = [out_features];
            let b_tensor = Tensor::zeros_on(b_shape, &backend);
            Some(Var::new(b_tensor, true))
        } else {
            None
        };

        Self {
            weight,
            bias: bias_var,
        }
    }
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Linear<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            params.push(b.clone());
        }
        params
    }

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        let w_t = coeus_autograd::transpose_2d(&self.weight);
        let out = coeus_autograd::matmul(input, &w_t);
        if let Some(ref bias) = self.bias {
            coeus_autograd::add(&out, bias)
        } else {
            out
        }
    }

    fn load_parameters(&mut self, params: &[Var<T, B>]) {
        self.weight = params[0].clone();
        if self.bias.is_some() {
            self.bias = Some(params[1].clone());
        }
    }
}
