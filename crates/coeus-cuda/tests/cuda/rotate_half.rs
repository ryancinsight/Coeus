use coeus_autograd::{rotate_half, sum, Var};
use coeus_core::{ComputeBackend, Layout, SequentialBackend};
use coeus_cuda::CudaBackend;
use coeus_ops::{BinaryOp, ElementwiseOps};
use coeus_tensor::Tensor;

#[test]
fn rotate_half_dispatches_with_cuda_parity() {
    if !crate::availability::device_available() {
        return;
    }
    let cpu = SequentialBackend::new();
    let cuda = CudaBackend::new();
    let input = Tensor::from_slice_on([2, 4], &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &cpu)
        .expect("invariant: test backend operation succeeds")
        .to_backend_on(&cpu, &cuda)
        .expect("invariant: test backend transfer succeeds");
    let input = Var::new(input, true).expect("invariant: test backend operation succeeds");
    let output = rotate_half(&input).expect("CUDA rotate-half dispatch");
    sum(&output)
        .expect("invariant: test backend operation succeeds")
        .backward()
        .expect("CUDA rotate-half backward");
    assert_eq!(
        output
            .tensor
            .to_backend_on(&cuda, &cpu)
            .expect("invariant: test backend transfer succeeds")
            .as_slice(),
        &[-3.0, -4.0, 1.0, 2.0, -7.0, -8.0, 5.0, 6.0]
    );
    assert_eq!(
        input
            .grad()
            .expect("tracked CUDA input gradient")
            .to_backend_on(&cuda, &cpu)
            .expect("invariant: test backend transfer succeeds")
            .as_slice(),
        &[1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0]
    );
}

#[test]
fn unary_assignment_detaches_shared_cuda_view() {
    if !crate::availability::device_available() {
        return;
    }

    let cpu = SequentialBackend::new();
    let cuda = CudaBackend::new();
    let base = Tensor::from_slice_on([2, 3], &[-3.0_f32, -1.0, 0.0, 2.0, 4.0, -5.0], &cpu)
        .expect("invariant: test backend operation succeeds")
        .to_backend_on(&cpu, &cuda)
        .expect("invariant: test backend transfer succeeds");
    let mut assigned = base.slice(&[(0, 2), (1, 3)]);
    let shared = assigned.clone();

    coeus_ops::neg_assign(&mut assigned, &cuda).expect("CUDA neg assignment");

    assert_eq!(
        assigned
            .to_backend_on(&cuda, &cpu)
            .expect("invariant: test backend transfer succeeds")
            .as_slice(),
        &[1.0, 0.0, -4.0, 5.0]
    );
    assert_eq!(
        shared
            .to_vec_on(&cuda)
            .expect("invariant: test backend transfer succeeds"),
        [-1.0, 0.0, 4.0, -5.0],
        "shared CUDA source must remain unchanged"
    );
}

#[test]
fn partial_update_preserves_cuda_parent_and_shared_source() {
    if !crate::availability::device_available() {
        return;
    }

    let backend = CudaBackend::new();
    let parent_layout = Layout::new([2, 3].into());
    let destination_layout = parent_layout.slice(&[(0, 2), (1, 3)]);
    let rhs_layout = Layout::new([2, 2].into());
    // SAFETY: the following `copy_to_device` initializes every element before any read.
    let mut destination =
        unsafe { backend.allocate::<f32>(6) }.expect("invariant: test backend operation succeeds");
    // SAFETY: the following `copy_to_device` initializes every element before any read.
    let mut rhs =
        unsafe { backend.allocate::<f32>(4) }.expect("invariant: test backend operation succeeds");
    backend
        .copy_to_device(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &mut destination)
        .expect("invariant: test backend operation succeeds");
    backend
        .copy_to_device(&[10.0, 20.0, 30.0, 40.0], &mut rhs)
        .expect("invariant: test backend operation succeeds");
    let shared = destination.clone();

    backend
        .elementwise_binary_update(
            BinaryOp::Add,
            &mut destination,
            &destination_layout,
            &rhs,
            &rhs_layout,
        )
        .expect("CUDA partial update");

    let mut actual = [0.0; 6];
    backend
        .copy_to_host(&destination, &mut actual)
        .expect("invariant: test backend operation succeeds");
    assert_eq!(actual, [1.0, 12.0, 23.0, 4.0, 35.0, 46.0]);
    let mut shared_values = [0.0; 6];
    backend
        .copy_to_host(&shared, &mut shared_values)
        .expect("invariant: test backend operation succeeds");
    assert_eq!(shared_values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}
