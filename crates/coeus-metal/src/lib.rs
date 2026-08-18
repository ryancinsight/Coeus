//! Coeus Metal backend integration through the Hephaestus Metal provider.
//!
//! The crate exposes the generic `HephaestusBackend<MetalProvider>` for
//! elementwise, scalar-power, matmul, convolution, axis-reduction, scan,
//! random, rotate-half, stateful-update, and cross-entropy dispatch through the
//! shared Coeus-Hephaestus bridge, with no host fallback for unsupported
//! layouts.
#![deny(missing_docs)]

mod backend;

pub use backend::{HephaestusBackend, MetalProvider};

#[cfg(test)]
mod tests {
    use super::{HephaestusBackend, MetalProvider};

    /// The provider seam, not a per-vendor kernel, is what supplies matmul to
    /// this backend: the bound resolves only through
    /// `coeus_hephaestus::MatmulProvider`.
    ///
    /// `BackendOps<f32>` is not yet assertable here because `PoolOps` and
    /// `UnfoldFoldOps` have no `hephaestus-core` device seam (ADR-0066).
    #[test]
    fn backend_satisfies_matmul_through_the_provider_seam() {
        fn require<B: coeus_ops::MatmulOps<f32>>() {}
        require::<HephaestusBackend<MetalProvider>>();
    }
}
