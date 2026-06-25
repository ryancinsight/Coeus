use coeus_core::{Layout, Scalar, Shape, Strides};

use crate::backend_ops::ops::BinaryOp;
use crate::backend_ops::trait_def::BackendOps;

fn shape3(shape: &[usize], name: &str) -> [usize; 3] {
    assert_eq!(
        shape.len(),
        3,
        "batched_matmul: {name} shape must have rank 3"
    );
    [shape[0], shape[1], shape[2]]
}

/// Default implementation of `BackendOps::matmul_accumulate`.
pub fn matmul_accumulate<T: Scalar>(
    backend: &impl BackendOps<T>,
    a: &impl BackendOps<T, DeviceBuffer<T> = <impl BackendOps<T> as coeus_core::ComputeBackend>::DeviceBuffer<T>>,
    // we can't do this -- need a different approach
) {
    unimplemented!()
}
