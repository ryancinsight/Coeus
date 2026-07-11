use coeus_autograd::{sum, trilinear_interpolation, Var};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

#[test]
fn backward_matches_analytical_image_and_grid_derivatives() {
    let backend = MoiraiBackend;
    let image = Var::new(
        Tensor::from_slice_on(
            [1, 1, 2, 2, 2],
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            &backend,
        ),
        true,
    );
    let grid = Var::new(
        Tensor::from_slice_on([1, 3, 1, 1, 1], &[0.5, 0.5, 0.5], &backend),
        true,
    );

    let sampled = trilinear_interpolation(&image, &grid).expect("valid rank-5 contract");
    assert_eq!(sampled.tensor.as_slice(), &[3.5]);
    sum(&sampled).backward();

    assert_eq!(
        image.grad().expect("tracked image gradient").as_slice(),
        &[0.125; 8]
    );
    assert_eq!(
        grid.grad().expect("tracked grid gradient").as_slice(),
        &[4.0, 2.0, 1.0]
    );
}

#[test]
fn constant_image_has_zero_coordinate_gradient() {
    let backend = MoiraiBackend;
    let image = Var::new(
        Tensor::from_slice_on([1, 1, 2, 2, 2], &[7.0; 8], &backend),
        false,
    );
    let grid = Var::new(
        Tensor::from_slice_on([1, 3, 1, 1, 1], &[0.25, 0.75, 0.5], &backend),
        true,
    );
    let sampled = trilinear_interpolation(&image, &grid).expect("valid rank-5 contract");
    sum(&sampled).backward();
    assert_eq!(
        grid.grad().expect("tracked grid gradient").as_slice(),
        &[0.0, 0.0, 0.0]
    );
}
