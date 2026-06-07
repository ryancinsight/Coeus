use coeus_core::Float;
use coeus_autograd::Var;
use crate::module::Module;

/// Dropout layer.
///
/// During training, randomly zeroes some of the elements of the input tensor
/// with probability `p` using samples from a Bernoulli distribution.
/// The outputs are scaled by `1 / (1 - p)` during training, and left unchanged during evaluation.
#[derive(Clone)]
pub struct Dropout {
    /// Dropout probability.
    pub p: f64,
    /// Training mode flag.
    pub is_training: bool,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Dropout {
    /// Create a new Dropout layer.
    pub fn new(p: f64) -> Self {
        assert!(p >= 0.0 && p < 1.0, "Dropout probability must be in [0.0, 1.0)");
        Self {
            p,
            is_training: true,
            seed: 42,
        }
    }

    /// Set training mode inherently without generic parameter ambiguity.
    pub fn set_training(&mut self, mode: bool) {
        self.is_training = mode;
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Dropout {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn train(&mut self, mode: bool) {
        self.set_training(mode);
    }

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        coeus_autograd::dropout(input, self.p, self.is_training, self.seed)
    }
}
