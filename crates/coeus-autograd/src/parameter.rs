//! Named trainable parameter carrier shared by modules and optimizers.

use crate::Var;
use coeus_core::{ComputeBackend, MoiraiBackend, Scalar};
use std::ops::{Deref, DerefMut};

/// A trainable variable with its stable hierarchical state path.
#[derive(Clone)]
pub struct Parameter<T: Scalar, B: ComputeBackend + Default = MoiraiBackend> {
    /// The differentiable variable updated by an optimizer.
    pub var: Var<T, B>,
    /// Stable hierarchical name used by state archives and diagnostics.
    pub name: String,
}

impl<T: Scalar, B: ComputeBackend + Default> Deref for Parameter<T, B> {
    type Target = Var<T, B>;

    fn deref(&self) -> &Self::Target {
        &self.var
    }
}

impl<T: Scalar, B: ComputeBackend + Default> DerefMut for Parameter<T, B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.var
    }
}

impl<T: Scalar, B: ComputeBackend + Default> Parameter<T, B> {
    /// Create a named trainable parameter.
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
