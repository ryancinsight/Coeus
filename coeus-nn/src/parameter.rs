use coeus_autograd::Var;
use coeus_core::{MoiraiBackend, Scalar};

/// A learnable parameter wrapping an autograd variable.
#[derive(Clone)]
pub struct Parameter<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// The underlying differentiable variable.
    pub var: Var<T, B>,
    /// Human-readable name (for state_dict, logging).
    pub name: String,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Parameter<T, B> {
    /// Create a new parameter with given name.
    #[inline]
    pub fn new(var: Var<T, B>, name: impl Into<String>) -> Self {
        Self {
            var,
            name: name.into(),
        }
    }

    /// Prepend a hierarchical module path to this parameter name.
    #[must_use]
    pub fn with_prefix(mut self, prefix: &str) -> Self {
        self.name = format!("{prefix}.{}", self.name);
        self
    }
}
