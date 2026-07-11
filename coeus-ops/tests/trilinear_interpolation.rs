use coeus_core::{MoiraiBackend, SequentialBackend};
use coeus_ops::{trilinear_interpolation, InterpolationError};
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

    let border_grid = Tensor::from_slice_on([1, 3, 1, 1, 1], &[-1.0, 2.5, 2.5], &backend);
    let border = trilinear_interpolation(&image, &border_grid).expect("border replication");
    assert_eq!(border.as_slice(), &[3.0]);

    let malformed = Tensor::from_slice_on([1, 2, 1, 1, 1], &[0.0, 0.0], &backend);
    match trilinear_interpolation(&image, &malformed) {
        Err(error) => assert_eq!(error, InterpolationError::GridChannels(2)),
        Ok(_) => panic!("two-channel grid must violate the coordinate contract"),
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
