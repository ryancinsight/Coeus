use coeus_autograd::{linear_interpolation, sum, Var};
use coeus_core::MoiraiBackend;
use coeus_ops::Replicate;
use coeus_tensor::Tensor;

#[test]
fn three_dimensional_backward_matches_analytical_derivatives() {
    let backend = MoiraiBackend;
    let image = Var::new(
        Tensor::from_slice_on(
            [1, 1, 2, 2, 2],
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            &backend,
        ).expect("valid tensor construction"),
        true,
    ).expect("valid variable construction");
    let grid = Var::new(
        Tensor::from_slice_on([1, 3, 1, 1, 1], &[0.5, 0.5, 0.5], &backend).expect("valid tensor construction"),
        true,
    ).expect("valid variable construction");
    let sampled = linear_interpolation::<3, _, _>(&image, &grid, Replicate)
        .expect("valid three-dimensional contract");
    assert_eq!(sampled.tensor.as_slice(), &[3.5]);
    sum(&sampled)
        .expect("valid autograd operation")
        .backward()
        .expect("valid backward propagation");
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
fn two_dimensional_backward_matches_analytical_derivatives() {
    let backend = MoiraiBackend;
    let image = Var::new(
        Tensor::from_slice_on([1, 1, 2, 2], &[0.0, 1.0, 2.0, 3.0], &backend).expect("valid tensor construction"),
        true,
    ).expect("valid variable construction");
    let grid = Var::new(
        Tensor::from_slice_on([1, 2, 1, 1], &[0.25, 0.75], &backend).expect("valid tensor construction"),
        true,
    ).expect("valid variable construction");
    let sampled = linear_interpolation::<2, _, _>(&image, &grid, Replicate)
        .expect("valid two-dimensional contract");
    assert_eq!(sampled.tensor.as_slice(), &[1.25]);
    sum(&sampled)
        .expect("valid autograd operation")
        .backward()
        .expect("valid backward propagation");
    assert_eq!(
        image.grad().expect("tracked image gradient").as_slice(),
        &[0.1875, 0.5625, 0.0625, 0.1875]
    );
    assert_eq!(
        grid.grad().expect("tracked grid gradient").as_slice(),
        &[2.0, 1.0]
    );
}

#[test]
fn constant_image_has_zero_coordinate_gradient() {
    let backend = MoiraiBackend;
    let image = Var::new(
        Tensor::from_slice_on([1, 1, 2, 2], &[7.0; 4], &backend).expect("valid tensor construction"),
        false,
    ).expect("valid variable construction");
    let grid = Var::new(
        Tensor::from_slice_on([1, 2, 1, 1], &[0.25, 0.75], &backend).expect("valid tensor construction"),
        true,
    ).expect("valid variable construction");
    let sampled = linear_interpolation::<2, _, _>(&image, &grid, Replicate)
        .expect("valid two-dimensional contract");
    sum(&sampled)
        .expect("valid autograd operation")
        .backward()
        .expect("valid backward propagation");
    assert_eq!(
        grid.grad().expect("tracked grid gradient").as_slice(),
        &[0.0, 0.0]
    );
}
