use coeus_core::{MoiraiBackend, SequentialBackend};
use coeus_ops::{trilinear_interpolation, trilinear_interpolation_backward, InterpolationError};
use coeus_tensor::Tensor;

fn verify_backend<B>()
where
    B: coeus_core::Backend + Default,
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    let backend = B::default();
    let image = Tensor::from_slice_on([1, 1, 2, 2, 2], &[0., 1., 2., 3., 4., 5., 6., 7.], &backend);
    let grid = Tensor::from_slice_on([1, 3, 1, 1, 1], &[0.5, 0.5, 0.5], &backend);
    let output = trilinear_interpolation(&image, &grid).expect("valid trilinear contract");
    assert_eq!(output.shape(), &[1, 1, 1, 1, 1]);
    assert_eq!(output.as_slice(), &[3.5]);

    let upstream = Tensor::from_slice_on([1, 1, 1, 1, 1], &[1.0], &backend);
    let gradients = trilinear_interpolation_backward(&image, &grid, &upstream)
        .expect("valid trilinear backward contract");
    assert_eq!(gradients.image.as_slice(), &[0.125; 8]);
    assert_eq!(gradients.grid.as_slice(), &[4.0, 2.0, 1.0]);

    let border_grid = Tensor::from_slice_on([1, 3, 1, 1, 1], &[-1.0, 2.5, 2.5], &backend);
    let border = trilinear_interpolation(&image, &border_grid).expect("border replication");
    assert_eq!(border.as_slice(), &[3.0]);

    let malformed = Tensor::from_slice_on([1, 2, 1, 1, 1], &[0.0, 0.0], &backend);
    match trilinear_interpolation(&image, &malformed) {
        Err(error) => assert_eq!(error, InterpolationError::GridChannels(2)),
        Ok(_) => panic!("two-channel grid must violate the coordinate contract"),
    }

    let malformed_gradient = Tensor::from_slice_on([1, 1], &[1.0], &backend);
    match trilinear_interpolation_backward(&image, &grid, &malformed_gradient) {
        Err(InterpolationError::GradientShape { expected, actual }) => {
            assert_eq!(expected, vec![1, 1, 1, 1, 1]);
            assert_eq!(actual, vec![1, 1]);
        }
        Err(error) => panic!("unexpected backward contract error: {error}"),
        Ok(_) => panic!("malformed upstream gradient must be rejected"),
    }
}

#[test]
fn sequential_backend_matches_analytical_values() {
    verify_backend::<SequentialBackend>();
}

#[test]
fn moirai_backend_matches_analytical_values() {
    verify_backend::<MoiraiBackend>();
}
