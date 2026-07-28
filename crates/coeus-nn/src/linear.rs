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
    pub fn new(
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Self, B::Error> {
        let backend = B::default();
        let w_shape = [out_features, in_features];
        let w_tensor = Tensor::ones_on(w_shape, &backend);
        let weight = Var::new(w_tensor?, true)?;

        let bias_var = if bias {
            let b_shape = [out_features];
            let b_tensor = Tensor::zeros_on(b_shape, &backend)?;
            Some(Var::new(b_tensor, true)?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias: bias_var,
        })
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

    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
        let input_shape = input.tensor.shape();
        assert!(
            input_shape.len() >= 2,
            "Linear input must have rank >= 2, got shape {input_shape:?}"
        );
        let in_features = self.weight.tensor.shape()[1];
        assert_eq!(
            input_shape[input_shape.len() - 1],
            in_features,
            "Linear input last dimension must equal in_features"
        );

        let rows = input_shape[..input_shape.len() - 1]
            .iter()
            .copied()
            .product::<usize>();
        let flattened = if input_shape.len() == 2 {
            input.clone()
        } else {
            coeus_autograd::reshape(input, [rows, in_features])?
        };
        let w_t = coeus_autograd::transpose_2d(&self.weight)?;
        let projected = coeus_autograd::matmul(&flattened, &w_t)?;
        let projected = if let Some(ref bias) = self.bias {
            coeus_autograd::add(&projected, bias)?
        } else {
            projected
        };

        if input_shape.len() == 2 {
            Ok(projected)
        } else {
            let mut output_shape = input_shape.to_vec();
            *output_shape
                .last_mut()
                .expect("invariant: rank was validated as at least two") =
                self.weight.tensor.shape()[0];
            coeus_autograd::reshape(&projected, output_shape)
        }
    }

    fn load_parameters(&mut self, params: &[Var<T, B>]) {
        self.weight = params[0].clone();
        if self.bias.is_some() {
            self.bias = Some(params[1].clone());
        }
    }
}
