//! Compile-time contract for the ROCm sinusoidal module capability boundary.

use coeus_nn::{Module, SinusoidalEncoding};
use coeus_ops::{ElementwiseOps, ReductionOps};

fn assert_module<T, B, M>()
where
    T: coeus_core::Scalar,
    B: coeus_core::ComputeBackend + Default,
    M: Module<T, B>,
{
}

fn assert_sinusoidal_capabilities<B>()
where
    B: ElementwiseOps<f32> + ReductionOps<f32> + Default,
{
    assert_module::<f32, B, SinusoidalEncoding<f32, B>>();
}

#[test]
fn rocm_backend_satisfies_sinusoidal_capability_boundary() {
    assert_sinusoidal_capabilities::<coeus_rocm::RocmBackend>();
}
