use coeus_core::{MoiraiBackend, SequentialBackend};
use coeus_ops::{
    linear_interpolation, linear_interpolation_backward, InterpolationError, Replicate,
};
use coeus_tensor::Tensor;

fn verify_three_dimensions<B>()
where
    B: coeus_core::Backend + Default,
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    let backend = B::default();
    let image = Tensor::from_slice_on([1, 1, 2, 2, 2], &[0., 1., 2., 3., 4., 5., 6., 7.], &backend).expect("construct tensor");
    let grid = Tensor::from_slice_on([1, 3, 1, 1, 1], &[0.5, 0.5, 0.5], &backend).expect("construct tensor");
    let output = linear_interpolation::<3, _, _>(&image, &grid, Replicate)
        .expect("valid three-dimensional contract");
    assert_eq!(output.shape(), &[1, 1, 1, 1, 1]);
    assert_eq!(output.as_slice(), &[3.5]);

    let upstream = Tensor::from_slice_on([1, 1, 1, 1, 1], &[1.0], &backend).expect("construct tensor");
    let gradients = linear_interpolation_backward::<3, _, _>(&image, &grid, &upstream, Replicate)
        .expect("valid three-dimensional backward contract");
    assert_eq!(gradients.image.as_slice(), &[0.125; 8]);
    assert_eq!(gradients.grid.as_slice(), &[4.0, 2.0, 1.0]);

    let border_grid = Tensor::from_slice_on([1, 3, 1, 1, 1], &[-1.0, 2.5, 2.5], &backend).expect("construct tensor");
    let border = linear_interpolation::<3, _, _>(&image, &border_grid, Replicate)
        .expect("replicated border");
    assert_eq!(border.as_slice(), &[3.0]);
}

fn verify_two_dimensions<B>()
where
    B: coeus_core::Backend + Default,
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    let backend = B::default();
    let image = Tensor::from_slice_on([1, 1, 2, 2], &[0., 1., 2., 3.], &backend).expect("construct tensor");
    let grid = Tensor::from_slice_on([1, 2, 1, 1], &[0.25, 0.75], &backend).expect("construct tensor");
    let output = linear_interpolation::<2, _, _>(&image, &grid, Replicate)
        .expect("valid two-dimensional contract");
    assert_eq!(output.shape(), &[1, 1, 1, 1]);
    assert_eq!(output.as_slice(), &[1.25]);

    let upstream = Tensor::from_slice_on([1, 1, 1, 1], &[1.0], &backend).expect("construct tensor");
    let gradients = linear_interpolation_backward::<2, _, _>(&image, &grid, &upstream, Replicate)
        .expect("valid two-dimensional backward contract");
    assert_eq!(
        gradients.image.as_slice(),
        &[0.1875, 0.5625, 0.0625, 0.1875]
    );
    assert_eq!(gradients.grid.as_slice(), &[2.0, 1.0]);
}

fn verify_errors<B>()
where
    B: coeus_core::Backend + Default,
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    let backend = B::default();
    let image = Tensor::from_slice_on([1, 1, 2, 2], &[0.; 4], &backend).expect("construct tensor");
    let malformed = Tensor::from_slice_on([1, 3, 1, 1], &[0.; 3], &backend).expect("construct tensor");
    match linear_interpolation::<2, _, _>(&image, &malformed, Replicate) {
        Err(error) => assert_eq!(
            error,
            InterpolationError::GridChannels {
                expected: 2,
                actual: 3,
            }
        ),
        Ok(_) => panic!("three-channel grid must violate the two-dimensional contract"),
    }
    let grid = Tensor::from_slice_on([1, 2, 1, 1], &[0.; 2], &backend).expect("construct tensor");
    let malformed_gradient = Tensor::from_slice_on([1, 1], &[1.0], &backend).expect("construct tensor");
    match linear_interpolation_backward::<2, _, _>(&image, &grid, &malformed_gradient, Replicate) {
        Err(InterpolationError::GradientShape { expected, actual }) => {
            assert_eq!(expected, vec![1, 1, 1, 1]);
            assert_eq!(actual, vec![1, 1]);
        }
        Err(error) => panic!("unexpected backward contract error: {error}"),
        Ok(_) => panic!("malformed upstream gradient must be rejected"),
    }
    let non_finite = Tensor::from_slice_on([1, 2, 1, 1], &[f32::NAN, 0.0], &backend).expect("construct tensor");
    match linear_interpolation::<2, _, _>(&image, &non_finite, Replicate) {
        Err(error) => assert_eq!(
            error,
            InterpolationError::NonFiniteCoordinate { axis: 0, point: 0 }
        ),
        Ok(_) => panic!("non-finite coordinates must be rejected"),
    }
}

fn verify_backend<B>()
where
    B: coeus_core::Backend + Default,
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    verify_two_dimensions::<B>();
    verify_three_dimensions::<B>();
    verify_errors::<B>();
}

#[test]
fn sequential_backend_matches_analytical_values() {
    verify_backend::<SequentialBackend>();
}

#[test]
fn moirai_backend_matches_analytical_values() {
    verify_backend::<MoiraiBackend>();
}

#[test]
fn coordinate_gradients_match_central_differences_in_each_dimension() {
    let backend = SequentialBackend;
    let step = 1.0e-3_f32;
    // A multilinear polynomial has no central-difference truncation error in
    // one coordinate. The bound covers the four rounded evaluations per axis.
    let bound = 16.0 * f32::EPSILON / step;

    let image = Tensor::from_slice_on([1, 1, 2, 2], &[0., 1., 2., 3.], &backend).expect("construct tensor");
    let coordinates = [0.25, 0.75];
    let grid = Tensor::from_slice_on([1, 2, 1, 1], &coordinates, &backend).expect("construct tensor");
    let upstream = Tensor::from_slice_on([1, 1, 1, 1], &[1.0], &backend).expect("construct tensor");
    let analytical = linear_interpolation_backward::<2, _, _>(&image, &grid, &upstream, Replicate)
        .expect("valid two-dimensional backward contract");
    for axis in 0..2 {
        let mut lower = coordinates;
        let mut upper = coordinates;
        lower[axis] -= step;
        upper[axis] += step;
        let lower_grid = Tensor::from_slice_on([1, 2, 1, 1], &lower, &backend).expect("construct tensor");
        let upper_grid = Tensor::from_slice_on([1, 2, 1, 1], &upper, &backend).expect("construct tensor");
        let lower_value = linear_interpolation::<2, _, _>(&image, &lower_grid, Replicate)
            .expect("lower perturbation")
            .as_slice()[0];
        let upper_value = linear_interpolation::<2, _, _>(&image, &upper_grid, Replicate)
            .expect("upper perturbation")
            .as_slice()[0];
        let numerical = (upper_value - lower_value) / (2.0 * step);
        assert!((analytical.grid.as_slice()[axis] - numerical).abs() <= bound);
    }

    let image = Tensor::from_slice_on([1, 1, 2, 2, 2], &[0., 1., 2., 3., 4., 5., 6., 7.], &backend).expect("construct tensor");
    let coordinates = [0.25, 0.5, 0.75];
    let grid = Tensor::from_slice_on([1, 3, 1, 1, 1], &coordinates, &backend).expect("construct tensor");
    let upstream = Tensor::from_slice_on([1, 1, 1, 1, 1], &[1.0], &backend).expect("construct tensor");
    let analytical = linear_interpolation_backward::<3, _, _>(&image, &grid, &upstream, Replicate)
        .expect("valid three-dimensional backward contract");
    for axis in 0..3 {
        let mut lower = coordinates;
        let mut upper = coordinates;
        lower[axis] -= step;
        upper[axis] += step;
        let lower_grid = Tensor::from_slice_on([1, 3, 1, 1, 1], &lower, &backend).expect("construct tensor");
        let upper_grid = Tensor::from_slice_on([1, 3, 1, 1, 1], &upper, &backend).expect("construct tensor");
        let lower_value = linear_interpolation::<3, _, _>(&image, &lower_grid, Replicate)
            .expect("lower perturbation")
            .as_slice()[0];
        let upper_value = linear_interpolation::<3, _, _>(&image, &upper_grid, Replicate)
            .expect("upper perturbation")
            .as_slice()[0];
        let numerical = (upper_value - lower_value) / (2.0 * step);
        assert!((analytical.grid.as_slice()[axis] - numerical).abs() <= bound);
    }
}
