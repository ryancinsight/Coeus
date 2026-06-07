// ── Sequential module combinator ──
//
// Sequential<T, B> chains a runtime-dynamic list of boxed modules.
// `dyn Module<T, B>` is justified here by condition (1) in the architecture rules:
// the concrete layer types are genuinely unknown at compile time and type erasure
// is the domain requirement for a user-constructable heterogeneous module stack.

use coeus_core::{Scalar, MoiraiBackend};
use coeus_autograd::Var;
use crate::module::Module;

/// A sequential container that chains modules.
///
/// `forward` passes the output of each module as the input to the next.
///
/// # Example
/// ```rust,ignore
/// let mut seq = Sequential::<f64>::new();
/// seq.add(Linear::new(64, 128, true));
/// seq.add(ReLU);
/// seq.add(Linear::new(128, 10, true));
/// let output = seq.forward(&input);
/// ```
pub struct Sequential<
    T: Scalar,
    B: coeus_ops::BackendOps<T> + Default = MoiraiBackend,
> {
    /// Type-erased module list. dyn dispatch is justified: types are unknown at compile time.
    layers: Vec<Box<dyn Module<T, B>>>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Sequential<T, B> {
    /// Create an empty Sequential.
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Append a module to the end of the sequence.
    ///
    /// Returns `&mut Self` to enable method chaining.
    pub fn add<M: Module<T, B> + 'static>(&mut self, module: M) -> &mut Self {
        self.layers.push(Box::new(module));
        self
    }

    /// Return the number of modules in this sequential.
    #[inline]
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Return true if this sequential has no modules.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Default for Sequential<T, B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Sequential<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        self.layers.iter().flat_map(|m| m.parameters()).collect()
    }

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        self.layers.iter().fold(input.clone(), |x, m| m.forward(&x))
    }

    fn train(&mut self, mode: bool) {
        for m in &mut self.layers {
            m.train(mode);
        }
    }
}

/// A compile-time static sequential container that chains modules.
///
/// Chaining is achieved via nested `StaticSeq<H, T>` structs created using
/// the `ModuleExt::append` builder method. Evaluation is fully monomorphized
/// at compile-time by the Rust compiler, eliminating all dynamic dispatch.
///
/// # Example
/// ```rust,ignore
/// let model = Linear::new(64, 128)
///     .append(ReLU)
///     .append(Linear::new(128, 10));
/// let output = model.forward(&input);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct StaticSeq<H, T>(pub H, pub T);

impl<
    ScalarType: Scalar,
    B: coeus_ops::BackendOps<ScalarType> + Default,
    H: Module<ScalarType, B>,
    T: Module<ScalarType, B>,
> Module<ScalarType, B> for StaticSeq<H, T> {
    #[inline]
    fn parameters(&self) -> Vec<Var<ScalarType, B>> {
        let mut params = self.0.parameters();
        params.extend(self.1.parameters());
        params
    }

    #[inline]
    fn forward(&self, input: &Var<ScalarType, B>) -> Var<ScalarType, B> {
        let out = self.0.forward(input);
        self.1.forward(&out)
    }

    #[inline]
    fn train(&mut self, mode: bool) {
        self.0.train(mode);
        self.1.train(mode);
    }
}

/// Extension trait for all `Module`s, enabling fluent construction of `StaticSeq` chains.
pub trait ModuleExt<T: Scalar, B: coeus_ops::BackendOps<T> + Default>: Module<T, B> {
    /// Append another module to the end of the sequence.
    fn append<M: Module<T, B>>(self, next: M) -> StaticSeq<Self, M>
    where
        Self: Sized;
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default, M: Module<T, B>> ModuleExt<T, B> for M {
    #[inline]
    fn append<Next: Module<T, B>>(self, next: Next) -> StaticSeq<Self, Next> {
        StaticSeq(self, next)
    }
}
