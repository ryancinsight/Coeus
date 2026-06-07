use coeus_core::{Scalar, MoiraiBackend};
use coeus_autograd::Var;

/// Trait for neural network modules.
///
/// Each module owns `Parameter`s, can contain sub-modules, and can support train/eval modes.
pub trait Module<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Collect all trainable parameters (including from sub-modules).
    fn parameters(&self) -> Vec<Var<T, B>>;

    /// Forward pass.
    fn forward(&self, input: &Var<T, B>) -> Var<T, B>;

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
