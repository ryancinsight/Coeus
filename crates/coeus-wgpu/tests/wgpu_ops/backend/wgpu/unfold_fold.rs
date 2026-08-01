use coeus_core::SequentialBackend;
use coeus_tensor::Tensor;
use coeus_wgpu::WgpuBackend;

#[test]
fn wgpu_unfold_fold_1d_matches_sequential() {
    let sequential = SequentialBackend::new();
    let wgpu = WgpuBackend::new();
    let input = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 4], &[1.0, 2.0, 3.0, 4.0]);
    let input_wgpu = input.to_backend_on(&sequential, &wgpu);

    let expected_unfold = coeus_ops::unfold1d(&input, 3, 1, 1, 1, &sequential)
        .expect("sequential unfold geometry is valid");
    let actual_unfold =
        coeus_ops::unfold1d(&input_wgpu, 3, 1, 1, 1, &wgpu).expect("WGPU unfold geometry is valid");
    let actual_unfold_cpu = actual_unfold.to_backend_on(&wgpu, &sequential);
    assert_eq!(actual_unfold_cpu.as_slice(), expected_unfold.as_slice());

    let expected_fold = coeus_ops::fold1d(&expected_unfold, 4, 3, 1, 1, 1, &sequential)
        .expect("sequential fold geometry is valid");
    let actual_fold = coeus_ops::fold1d(&actual_unfold, 4, 3, 1, 1, 1, &wgpu)
        .expect("WGPU fold geometry is valid");
    let actual_fold_cpu = actual_fold.to_backend_on(&wgpu, &sequential);
    assert_eq!(actual_fold_cpu.as_slice(), expected_fold.as_slice());
}

#[test]
fn wgpu_unfold_fold_2d_matches_sequential() {
    let sequential = SequentialBackend::new();
    let wgpu = WgpuBackend::new();
    let input = Tensor::<f32, SequentialBackend>::from_slice(
        vec![1, 1, 3, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    );
    let input_wgpu = input.to_backend_on(&sequential, &wgpu);

    let expected_unfold = coeus_ops::unfold2d(&input, 2, 2, 1, 1, 1, 1, 1, 1, &sequential)
        .expect("sequential unfold geometry is valid");
    let actual_unfold = coeus_ops::unfold2d(&input_wgpu, 2, 2, 1, 1, 1, 1, 1, 1, &wgpu)
        .expect("WGPU unfold geometry is valid");
    let actual_unfold_cpu = actual_unfold.to_backend_on(&wgpu, &sequential);
    assert_eq!(actual_unfold_cpu.as_slice(), expected_unfold.as_slice());

    let expected_fold =
        coeus_ops::fold2d(&expected_unfold, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, &sequential)
            .expect("sequential fold geometry is valid");
    let actual_fold = coeus_ops::fold2d(&actual_unfold, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, &wgpu)
        .expect("WGPU fold geometry is valid");
    let actual_fold_cpu = actual_fold.to_backend_on(&wgpu, &sequential);
    assert_eq!(actual_fold_cpu.as_slice(), expected_fold.as_slice());
}
