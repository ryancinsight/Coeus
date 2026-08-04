//! Value and adjoint verification for provider-selected half-vector rotation.

use coeus_autograd::{rotate_half, sum, Var};
use coeus_core::{MoiraiBackend, SequentialBackend};
use coeus_ops::RotateHalfOps;
use coeus_tensor::Tensor;

fn check_values<B>(backend: &B)
where
    B: RotateHalfOps<f64> + Default,
    B::DeviceBuffer<f64>:
        coeus_core::CpuAddressableStorage<f64> + coeus_core::CpuAddressableStorageMut<f64>,
{
    let input = Tensor::from_slice_on([2, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], backend);
    let output = coeus_ops::rotate_half(&input, backend).expect("valid even final extent");
    assert_eq!(
        output.as_slice(),
        &[-3.0, -4.0, 1.0, 2.0, -7.0, -8.0, 5.0, 6.0]
    );
}

#[test]
fn sequential_and_moirai_match_the_rotation_definition() {
    check_values(&SequentialBackend);
    check_values(&MoiraiBackend);
}

#[test]
fn odd_final_extent_is_rejected() {
    let backend = SequentialBackend;
    let input = Tensor::from_slice_on([1, 3], &[1.0_f64, 2.0, 3.0], &backend);
    let error = match coeus_ops::rotate_half(&input, &backend) {
        Ok(_) => panic!("odd extent must fail"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("even final extent"),
        "unexpected error: {error}"
    );
}

fn check_backward<B>(backend: B)
where
    B: coeus_ops::BackendOps<f64> + RotateHalfOps<f64> + Default + 'static,
    B::DeviceBuffer<f64>:
        coeus_core::CpuAddressableStorage<f64> + coeus_core::CpuAddressableStorageMut<f64>,
{
    let input = Var::new(
        Tensor::from_slice_on([1, 4], &[1.0_f64, 2.0, 3.0, 4.0], &backend),
        true,
    );
    let output = rotate_half(&input).expect("valid rotation");
    sum(&output).backward().expect("rotation backward");
    assert_eq!(
        input.grad().expect("tracked input gradient").as_slice(),
        &[1.0, 1.0, -1.0, -1.0]
    );
}

#[test]
fn sequential_and_moirai_backward_apply_the_exact_transpose() {
    check_backward(SequentialBackend);
    check_backward(MoiraiBackend);
}
