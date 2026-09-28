use coeus_autograd::{linear_interpolation, sum, Var};
use coeus_core::{
    Backend, ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Float, MoiraiBackend,
    Scalar,
};
use coeus_ops::Replicate;
use coeus_tensor::Tensor;
use eunomia::{Bf16, F16};

fn verify_three_dimensional<T: Float>()
where
    MoiraiBackend: Backend + coeus_ops::BackendOps<T> + Default,
    <MoiraiBackend as ComputeBackend>::DeviceBuffer<T>:
        CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let backend = MoiraiBackend;
    let image_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0].map(T::from_f64);
    let grid_values = [0.5, 0.5, 0.5].map(T::from_f64);
    let image = Var::new(
        Tensor::from_slice_on([1, 1, 2, 2, 2], &image_values, &backend),
        true,
    );
    let grid = Var::new(
        Tensor::from_slice_on([1, 3, 1, 1, 1], &grid_values, &backend),
        true,
    );
    let sampled = linear_interpolation::<3, MoiraiBackend, _, T>(&image, &grid, Replicate)
        .expect("valid three-dimensional contract");
    assert_eq!(sampled.tensor.as_slice(), &[T::from_f64(3.5)]);
    sum(&sampled)
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    assert_eq!(
        image.grad().expect("tracked image gradient").as_slice(),
        &[T::from_f64(0.125); 8]
    );
    assert_eq!(
        grid.grad().expect("tracked grid gradient").as_slice(),
        &[T::from_f64(4.0), T::from_f64(2.0), T::from_f64(1.0),]
    );
}

fn verify_two_dimensional<T: Float>()
where
    MoiraiBackend: Backend + coeus_ops::BackendOps<T> + Default,
    <MoiraiBackend as ComputeBackend>::DeviceBuffer<T>:
        CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let backend = MoiraiBackend;
    let image_values = [0.0, 1.0, 2.0, 3.0].map(T::from_f64);
    let grid_values = [0.25, 0.75].map(T::from_f64);
    let image = Var::new(
        Tensor::from_slice_on([1, 1, 2, 2], &image_values, &backend),
        true,
    );
    let grid = Var::new(
        Tensor::from_slice_on([1, 2, 1, 1], &grid_values, &backend),
        true,
    );
    let sampled = linear_interpolation::<2, MoiraiBackend, _, T>(&image, &grid, Replicate)
        .expect("valid two-dimensional contract");
    assert_eq!(sampled.tensor.as_slice(), &[T::from_f64(1.25)]);
    sum(&sampled)
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    assert_eq!(
        image.grad().expect("tracked image gradient").as_slice(),
        &[
            T::from_f64(0.1875),
            T::from_f64(0.5625),
            T::from_f64(0.0625),
            T::from_f64(0.1875),
        ]
    );
    assert_eq!(
        grid.grad().expect("tracked grid gradient").as_slice(),
        &[T::from_f64(2.0), T::from_f64(1.0)]
    );
}

fn verify_constant_image<T: Float>()
where
    MoiraiBackend: Backend + coeus_ops::BackendOps<T> + Default,
    <MoiraiBackend as ComputeBackend>::DeviceBuffer<T>:
        CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let backend = MoiraiBackend;
    let image_values = [T::from_f64(7.0); 4];
    let grid_values = [0.25, 0.75].map(T::from_f64);
    let image = Var::new(
        Tensor::from_slice_on([1, 1, 2, 2], &image_values, &backend),
        false,
    );
    let grid = Var::new(
        Tensor::from_slice_on([1, 2, 1, 1], &grid_values, &backend),
        true,
    );
    let sampled = linear_interpolation::<2, MoiraiBackend, _, T>(&image, &grid, Replicate)
        .expect("valid two-dimensional contract");
    assert_eq!(sampled.tensor.as_slice(), &[T::from_f64(7.0)]);
    sum(&sampled)
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    assert_eq!(
        grid.grad().expect("tracked grid gradient").as_slice(),
        &[T::zero(), T::zero()]
    );
}

#[test]
fn three_dimensional_backward_matches_analytical_derivatives() {
    verify_three_dimensional::<f32>();
    verify_three_dimensional::<f64>();
    verify_three_dimensional::<F16>();
    verify_three_dimensional::<Bf16>();
}

#[test]
fn two_dimensional_backward_matches_analytical_derivatives() {
    verify_two_dimensional::<f32>();
    verify_two_dimensional::<f64>();
    verify_two_dimensional::<F16>();
    verify_two_dimensional::<Bf16>();
}

#[test]
fn constant_image_has_zero_coordinate_gradient() {
    verify_constant_image::<f32>();
    verify_constant_image::<f64>();
    verify_constant_image::<F16>();
    verify_constant_image::<Bf16>();
}
