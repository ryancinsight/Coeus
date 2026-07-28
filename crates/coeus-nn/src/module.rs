use coeus_autograd::Var;
use coeus_core::{MoiraiBackend, Scalar};

use coeus_autograd::Parameter;

/// Contract failures when loading optimizer-owned named parameters.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ParameterLoadError {
    /// The incoming inventory has a different parameter count.
    #[error("named parameter count mismatch: expected {expected}, got {actual}")]
    Count {
        /// Module inventory length.
        expected: usize,
        /// Incoming inventory length.
        actual: usize,
    },
    /// A hierarchical name differs at a stable inventory position.
    #[error("named parameter mismatch at index {index}: expected {expected}, got {actual}")]
    Name {
        /// Inventory position.
        index: usize,
        /// Module-owned path.
        expected: String,
        /// Incoming path.
        actual: String,
    },
}

pub(crate) fn prefixed_parameters<T, B, M>(prefix: &str, module: &M) -> Vec<Parameter<T, B>>
where
    T: Scalar,
    B: coeus_ops::BackendOps<T> + Default,
    M: Module<T, B> + ?Sized,
{
    module
        .named_parameters()
        .into_iter()
        .map(|parameter| parameter.with_prefix(prefix))
        .collect()
}

/// Trait for neural network modules.
///
/// Each module owns `Parameter`s, can contain sub-modules, and can support train/eval modes.
pub trait Module<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Collect all trainable parameters (including from sub-modules).
    fn parameters(&self) -> Vec<Var<T, B>>;

    /// Collect trainable parameters with stable hierarchical names.
    ///
    /// Leaf modules with the canonical zero-, one-, or two-parameter layout
    /// inherit `[]`, `[weight]`, or `[weight, bias]`. Modules with a wider
    /// layout or child modules must override this method so names express the
    /// owned field hierarchy instead of depending on flattened ordinals.
    fn named_parameters(&self) -> Vec<Parameter<T, B>> {
        let parameters = self.parameters();
        let names: &[&str] = match parameters.len() {
            0 => &[],
            1 => &["weight"],
            2 => &["weight", "bias"],
            count => panic!(
                "invariant: module with {count} parameters must define stable hierarchical names"
            ),
        };
        parameters
            .into_iter()
            .zip(names)
            .map(|(parameter, name)| Parameter::new(parameter, *name))
            .collect()
    }

    /// Forward pass.
    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, B::Error>;

    /// Write optimizer-updated parameter values back into this module's own
    /// fields, consuming `params` in the same order `parameters()` enumerates
    /// them (i.e. `params.len() == parameters().len()`).
    ///
    /// This exists because `Var::clone()` shares gradient state but not the
    /// value storage: `coeus_optim::{SGD, Adam}::step` mutates its own owned
    /// `Vec<Var<T, B>>` in place (via copy-on-write on first mutation), so a
    /// module that read `parameters()` to build an optimizer must call this
    /// after each `step()` to see the updated values, unless the module's own
    /// fields *are* the optimizer's `Var`s (as in a flat training loop that
    /// never separately clones them into a named struct).
    ///
    /// Default is a no-op, which is correct for parameterless modules
    /// (activations, dropout, pooling, reshape/view layers, ...). Any module
    /// whose `parameters()` returns a non-empty `Vec` must override this.
    #[inline]
    fn load_parameters(&mut self, _params: &[Var<T, B>]) {}

    /// Load optimizer-updated named parameters after validating the complete
    /// stable path inventory.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterLoadError`] when count or path ordering differs from
    /// this module's canonical named inventory.
    fn load_named_parameters(
        &mut self,
        parameters: &[Parameter<T, B>],
    ) -> Result<(), ParameterLoadError> {
        let expected = self.named_parameters();
        if expected.len() != parameters.len() {
            return Err(ParameterLoadError::Count {
                expected: expected.len(),
                actual: parameters.len(),
            });
        }
        for (index, (expected, actual)) in expected.iter().zip(parameters).enumerate() {
            if expected.name != actual.name {
                return Err(ParameterLoadError::Name {
                    index,
                    expected: expected.name.clone(),
                    actual: actual.name.clone(),
                });
            }
        }
        let variables = parameters
            .iter()
            .map(|parameter| parameter.var.clone())
            .collect::<Vec<_>>();
        self.load_parameters(&variables);
        Ok(())
    }

    /// Zero all parameter gradients.
    #[inline]
    fn zero_grad(&self) -> Result<(), B::Error> {
        for p in self.parameters() {
            p.zero_grad()?;
        }
        Ok(())
    }

    /// Set the training mode of the module and its sub-modules.
    #[inline]
    fn train(&mut self, _mode: bool) {}

    /// Set the module and its sub-modules to evaluation mode.
    #[inline]
    fn eval(&mut self) {
        self.train(false);
    }
}
