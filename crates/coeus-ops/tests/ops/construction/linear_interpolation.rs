use coeus_core::{
    Backend, ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Float, MoiraiBackend,
    Scalar, SequentialBackend,
};
use coeus_ops::{
    linear_interpolation, linear_interpolation_backward, InterpolationError, Replicate,
};
use coeus_tensor::Tensor;
use eunomia::{Bf16, F16};

fn verify_three_dimensions<B, T>()
where
    B: Backend + Default,
    T: Float,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let image_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0].map(<T as Scalar>::from_f64);
    let grid_values = [0.5, 0.5, 0.5].map(<T as Scalar>::from_f64);
    let image = Tensor::from_slice_on([1, 1, 2, 2, 2], &image_values, &backend);
    let grid = Tensor::from_slice_on([1, 3, 1, 1, 1], &grid_values, &backend);
    let output = linear_interpolation::<3, B, _, T>(&image, &grid, Replicate)
        .expect("valid three-dimensional contract");
    assert_eq!(output.shape(), &[1, 1, 1, 1, 1]);
    assert_eq!(output.as_slice(), &[T::from_f64(3.5)]);

    let upstream_values = [T::one()];
    let upstream = Tensor::from_slice_on([1, 1, 1, 1, 1], &upstream_values, &backend);
    let gradients =
        linear_interpolation_backward::<3, B, _, T>(&image, &grid, &upstream, Replicate)
            .expect("valid three-dimensional backward contract");
    assert_eq!(gradients.image.as_slice(), &[T::from_f64(0.125); 8]);
    assert_eq!(
        gradients.grid.as_slice(),
        &[T::from_f64(4.0), T::from_f64(2.0), T::from_f64(1.0),]
    );

    let border_values = [-1.0, 2.5, 2.5].map(<T as Scalar>::from_f64);
    let border_grid = Tensor::from_slice_on([1, 3, 1, 1, 1], &border_values, &backend);
    let border = linear_interpolation::<3, B, _, T>(&image, &border_grid, Replicate)
        .expect("replicated border");
    assert_eq!(border.as_slice(), &[T::from_f64(3.0)]);
}

fn verify_two_dimensions<B, T>()
where
    B: Backend + Default,
    T: Float,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let image_values = [0.0, 1.0, 2.0, 3.0].map(<T as Scalar>::from_f64);
    let grid_values = [0.25, 0.75].map(<T as Scalar>::from_f64);
    let image = Tensor::from_slice_on([1, 1, 2, 2], &image_values, &backend);
    let grid = Tensor::from_slice_on([1, 2, 1, 1], &grid_values, &backend);
    let output = linear_interpolation::<2, B, _, T>(&image, &grid, Replicate)
        .expect("valid two-dimensional contract");
    assert_eq!(output.shape(), &[1, 1, 1, 1]);
    assert_eq!(output.as_slice(), &[T::from_f64(1.25)]);

    let upstream_values = [T::one()];
    let upstream = Tensor::from_slice_on([1, 1, 1, 1], &upstream_values, &backend);
    let gradients =
        linear_interpolation_backward::<2, B, _, T>(&image, &grid, &upstream, Replicate)
            .expect("valid two-dimensional backward contract");
    assert_eq!(
        gradients.image.as_slice(),
        &[
            T::from_f64(0.1875),
            T::from_f64(0.5625),
            T::from_f64(0.0625),
            T::from_f64(0.1875),
        ]
    );
    assert_eq!(
        gradients.grid.as_slice(),
        &[T::from_f64(2.0), T::from_f64(1.0)]
    );
}

fn verify_errors<B, T>()
where
    B: Backend + Default,
    T: Float,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let image_values = [T::zero(); 4];
    let image = Tensor::from_slice_on([1, 1, 2, 2], &image_values, &backend);
    let malformed_values = [T::zero(); 3];
    let malformed = Tensor::from_slice_on([1, 3, 1, 1], &malformed_values, &backend);
    match linear_interpolation::<2, B, _, T>(&image, &malformed, Replicate) {
        Err(error) => assert_eq!(
            error,
            InterpolationError::GridChannels {
                expected: 2,
                actual: 3,
            }
        ),
        Ok(_) => panic!("three-channel grid must violate the two-dimensional contract"),
    }

    let grid_values = [T::zero(); 2];
    let grid = Tensor::from_slice_on([1, 2, 1, 1], &grid_values, &backend);
    let malformed_gradient_values = [T::one()];
    let malformed_gradient = Tensor::from_slice_on([1, 1], &malformed_gradient_values, &backend);
    match linear_interpolation_backward::<2, B, _, T>(&image, &grid, &malformed_gradient, Replicate)
    {
        Err(InterpolationError::GradientShape { expected, actual }) => {
            assert_eq!(expected, vec![1, 1, 1, 1]);
            assert_eq!(actual, vec![1, 1]);
        }
        Err(error) => panic!("unexpected backward contract error: {error}"),
        Ok(_) => panic!("malformed upstream gradient must be rejected"),
    }

    let non_finite_values = [T::NAN, T::zero()];
    let non_finite = Tensor::from_slice_on([1, 2, 1, 1], &non_finite_values, &backend);
    match linear_interpolation::<2, B, _, T>(&image, &non_finite, Replicate) {
        Err(error) => assert_eq!(
            error,
            InterpolationError::NonFiniteCoordinate { axis: 0, point: 0 }
        ),
        Ok(_) => panic!("non-finite coordinates must be rejected"),
    }
}

fn verify_backend<B, T>()
where
    B: Backend + Default,
    T: Float,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    verify_two_dimensions::<B, T>();
    verify_three_dimensions::<B, T>();
    verify_errors::<B, T>();
}

fn verify_scalar_types<B>()
where
    B: Backend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
    B::DeviceBuffer<F16>: CpuAddressableStorage<F16> + CpuAddressableStorageMut<F16>,
    B::DeviceBuffer<Bf16>: CpuAddressableStorage<Bf16> + CpuAddressableStorageMut<Bf16>,
{
    verify_backend::<B, f32>();
    verify_backend::<B, f64>();
    verify_backend::<B, F16>();
    verify_backend::<B, Bf16>();
}

#[test]
fn sequential_backend_matches_analytical_values() {
    verify_scalar_types::<SequentialBackend>();
}

#[test]
fn moirai_backend_matches_analytical_values() {
    verify_scalar_types::<MoiraiBackend>();
}

fn verify_two_dimensional_coordinate_gradients<T: Float>()
where
    <SequentialBackend as ComputeBackend>::DeviceBuffer<T>:
        CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let backend = SequentialBackend;
    // The fixture is at least 1/4 from a voxel boundary. A dyadic step of 1/8
    // stays in one bilinear cell and keeps every fixture operation exact.
    let step = T::from_f64(0.125);
    let image_values = [0.0, 1.0, 2.0, 3.0].map(T::from_f64);
    let coordinates = [0.25, 0.75].map(T::from_f64);
    let upstream_values = [T::one()];
    let image = Tensor::from_slice_on([1, 1, 2, 2], &image_values, &backend);
    let grid = Tensor::from_slice_on([1, 2, 1, 1], &coordinates, &backend);
    let upstream = Tensor::from_slice_on([1, 1, 1, 1], &upstream_values, &backend);
    let analytical =
        linear_interpolation_backward::<2, _, _, T>(&image, &grid, &upstream, Replicate)
            .expect("valid two-dimensional backward contract");

    for axis in 0..2 {
        let mut lower = coordinates;
        let mut upper = coordinates;
        lower[axis] -= step;
        upper[axis] += step;
        let lower_grid = Tensor::from_slice_on([1, 2, 1, 1], &lower, &backend);
        let upper_grid = Tensor::from_slice_on([1, 2, 1, 1], &upper, &backend);
        let lower_value = linear_interpolation::<2, _, _, T>(&image, &lower_grid, Replicate)
            .expect("lower perturbation")
            .as_slice()[0];
        let upper_value = linear_interpolation::<2, _, _, T>(&image, &upper_grid, Replicate)
            .expect("upper perturbation")
            .as_slice()[0];
        let numerical = (upper_value - lower_value) / (step + step);
        assert_eq!(analytical.grid.as_slice()[axis], numerical);
    }
}

fn verify_three_dimensional_coordinate_gradients<T: Float>()
where
    SequentialBackend::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let backend = SequentialBackend;
    // The fixture is at least 1/4 from a voxel boundary. A dyadic step of 1/8
    // stays in one trilinear cell and keeps every fixture operation exact.
    let step = T::from_f64(0.125);
    let image_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0].map(T::from_f64);
    let coordinates = [0.25, 0.5, 0.75].map(T::from_f64);
    let upstream_values = [T::one()];
    let image = Tensor::from_slice_on([1, 1, 2, 2, 2], &image_values, &backend);
    let grid = Tensor::from_slice_on([1, 3, 1, 1, 1], &coordinates, &backend);
    let upstream = Tensor::from_slice_on([1, 1, 1, 1, 1], &upstream_values, &backend);
    let analytical =
        linear_interpolation_backward::<3, _, _, T>(&image, &grid, &upstream, Replicate)
            .expect("valid three-dimensional backward contract");

    for axis in 0..3 {
        let mut lower = coordinates;
        let mut upper = coordinates;
        lower[axis] -= step;
        upper[axis] += step;
        let lower_grid = Tensor::from_slice_on([1, 3, 1, 1, 1], &lower, &backend);
        let upper_grid = Tensor::from_slice_on([1, 3, 1, 1, 1], &upper, &backend);
        let lower_value = linear_interpolation::<3, _, _, T>(&image, &lower_grid, Replicate)
            .expect("lower perturbation")
            .as_slice()[0];
        let upper_value = linear_interpolation::<3, _, _, T>(&image, &upper_grid, Replicate)
            .expect("upper perturbation")
            .as_slice()[0];
        let numerical = (upper_value - lower_value) / (step + step);
        assert_eq!(analytical.grid.as_slice()[axis], numerical);
    }
}

#[test]
fn coordinate_gradients_match_central_differences_in_each_dimension() {
    verify_two_dimensional_coordinate_gradients::<f32>();
    verify_three_dimensional_coordinate_gradients::<f32>();
    verify_two_dimensional_coordinate_gradients::<f64>();
    verify_three_dimensional_coordinate_gradients::<f64>();
}

#[test]
fn bfloat16_indices_respect_extent_and_keep_adjacent_voxels() {
    let backend = SequentialBackend;
    let mut image_values = [Bf16::zero(); 260];
    image_values[256] = Bf16::from_f64(5.0);
    image_values[257] = Bf16::from_f64(9.0);
    image_values[259] = Bf16::from_f64(13.0);
    let image = Tensor::from_slice_on([1, 1, 260, 1], &image_values, &backend);
    let upstream_values = [Bf16::one()];
    let upstream = Tensor::from_slice_on([1, 1, 1, 1], &upstream_values, &backend);

    let adjacent_values = [Bf16::from_f64(256.0), Bf16::zero()];
    let adjacent_grid = Tensor::from_slice_on([1, 2, 1, 1], &adjacent_values, &backend);
    let adjacent = linear_interpolation::<2, _, _, Bf16>(&image, &adjacent_grid, Replicate)
        .expect("valid BF16 coordinate at the adjacent-index boundary");
    assert_eq!(adjacent.as_slice(), &[Bf16::from_f64(5.0)]);
    let adjacent_gradients = linear_interpolation_backward::<2, _, _, Bf16>(
        &image,
        &adjacent_grid,
        &upstream,
        Replicate,
    )
    .expect("valid BF16 backward at the adjacent-index boundary");
    assert_eq!(
        adjacent_gradients.grid.as_slice(),
        &[Bf16::from_f64(4.0), Bf16::zero()]
    );
    assert_eq!(adjacent_gradients.image.as_slice()[256], Bf16::one());
    assert_eq!(adjacent_gradients.image.as_slice()[257], Bf16::zero());

    let border_values = [Bf16::from_f64(260.0), Bf16::zero()];
    let border_grid = Tensor::from_slice_on([1, 2, 1, 1], &border_values, &backend);
    let border = linear_interpolation::<2, _, _, Bf16>(&image, &border_grid, Replicate)
        .expect("finite coordinate beyond the integer extent is replicated");
    assert_eq!(border.as_slice(), &[Bf16::from_f64(13.0)]);
    let border_gradients =
        linear_interpolation_backward::<2, _, _, Bf16>(&image, &border_grid, &upstream, Replicate)
            .expect("valid BF16 backward at the replicated border");
    assert_eq!(
        border_gradients.grid.as_slice(),
        &[Bf16::zero(), Bf16::zero()]
    );
    assert_eq!(border_gradients.image.as_slice()[259], Bf16::one());
}
