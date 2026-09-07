use crate::init::InitializationError;
use crate::module::{Module, ModuleError};
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend, Scalar};
use coeus_ops::RandomInitOps;
use coeus_tensor::Tensor;

/// Fully-connected linear layer.
#[derive(Clone)]
pub struct Linear<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Weight matrix: `[out_features × in_features]`.
    pub weight: Var<T, B>,
    /// Bias vector: `[out_features]`.
    pub bias: Option<Var<T, B>>,
}

/// Construction needs more of `T` and `B` than holding a layer does -- the
/// weights are drawn, so the scalar must be sampleable and the backend must be
/// able to sample it. The bounds sit here rather than on the type so
/// `Linear<T, B>` stays nameable for any scalar.
impl<T, B> Linear<T, B>
where
    T: Scalar + Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + RandomInitOps<T> + Default,
{
    /// Create a Linear layer with given input/output features.
    ///
    /// Weights are drawn Kaiming-uniform over `in_features` from a fixed seed,
    /// so a layer built the same way twice is the same layer; use
    /// [`Linear::with_seed`] to choose the draw. Biases are zero.
    ///
    /// # Errors
    ///
    /// Returns [`InitializationError`] when `in_features` is zero, or when the
    /// backend's draw fails.
    pub fn new(
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Self, InitializationError<B::Error>> {
        Self::build(in_features, out_features, bias, None)
    }

    /// Create a Linear layer whose weights are drawn from `seed`.
    ///
    /// # Errors
    ///
    /// As [`Linear::new`].
    pub fn with_seed(
        in_features: usize,
        out_features: usize,
        bias: bool,
        seed: u64,
    ) -> Result<Self, InitializationError<B::Error>> {
        Self::build(in_features, out_features, bias, Some(seed))
    }

    fn build(
        in_features: usize,
        out_features: usize,
        bias: bool,
        seed: Option<u64>,
    ) -> Result<Self, InitializationError<B::Error>> {
        let backend = B::default();
        let w_tensor = Tensor::zeros_on([out_features, in_features], &backend);
        let mut weight = Var::new(w_tensor, true);

        // Every weight was 1.0 here. Each unit in the layer then computed the
        // same value from the same input, took the same gradient and applied
        // the same update -- identical at step zero and identical forever, so a
        // layer of any width had the capacity of one unit. See ADR 0067.
        match seed {
            Some(seed) => crate::init::kaiming_uniform_with_seed(&mut weight, in_features, seed)?,
            None => crate::init::kaiming_uniform(&mut weight, in_features)?,
        }

        let bias_var = bias.then(|| Var::new(Tensor::zeros_on([out_features], &backend), true));

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

    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        let input_shape = input.tensor.shape();
        if input_shape.len() < 2 {
            return Err(ModuleError::InvalidRank {
                module: "Linear",
                expected: "at least 2",
                actual: input_shape.len(),
            });
        }
        let in_features = self.weight.tensor.shape()[1];
        let actual_features = input_shape[input_shape.len() - 1];
        if actual_features != in_features {
            return Err(ModuleError::ShapeMismatch {
                module: "Linear",
                parameter: "input trailing dimension",
                expected: vec![in_features],
                actual: vec![actual_features],
            });
        }

        let rows = input_shape[..input_shape.len() - 1]
            .iter()
            .copied()
            .product::<usize>();
        let flattened = if input_shape.len() == 2 {
            input.clone()
        } else {
            coeus_autograd::reshape(input, [rows, in_features])
        };
        let w_t = coeus_autograd::transpose_2d(&self.weight);
        let projected = coeus_autograd::matmul(&flattened, &w_t);
        let projected = if let Some(ref bias) = self.bias {
            coeus_autograd::add(&projected, bias)
        } else {
            projected
        };

        let output = if input_shape.len() == 2 {
            projected
        } else {
            let mut output_shape = input_shape.to_vec();
            *output_shape
                .last_mut()
                .expect("invariant: rank was validated as at least two") =
                self.weight.tensor.shape()[0];
            coeus_autograd::reshape(&projected, output_shape)
        };
        Ok(output)
    }

    fn load_parameters(&mut self, params: &[Var<T, B>]) {
        self.weight = params[0].clone();
        if self.bias.is_some() {
            self.bias = Some(params[1].clone());
        }
    }
}
