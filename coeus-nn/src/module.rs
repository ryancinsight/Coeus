use coeus_autograd::Var;
use coeus_core::{MoiraiBackend, Scalar};

/// Trait for neural network modules.
///
/// Each module owns `Parameter`s, can contain sub-modules, and can support train/eval modes.
pub trait Module<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Collect all trainable parameters (including from sub-modules).
    fn parameters(&self) -> Vec<Var<T, B>>;

    /// Forward pass.
    fn forward(&self, input: &Var<T, B>) -> Var<T, B>;

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

    /// Zero all parameter gradients.
    #[inline]
    fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
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
