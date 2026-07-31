use coeus_core::{ComputeBackend, Layout, SequentialBackend};
use coeus_tensor::Tensor;
use coeus_wgpu::WgpuBackend;

const INPUT: [f32; 7] = [1.0, 5.0, 2.0, 7.0, 3.0, 10.0, 4.0];
const KERNEL_SIZE: usize = 3;
const STRIDE: usize = 2;
const PADDING: usize = 1;
const DILATION: usize = 2;

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let scale = actual.abs().max(expected.abs()).max(1.0);
        let bound = 8.0 * f32::EPSILON * scale;
        assert!(
            (actual - expected).abs() <= bound,
            "mismatch at {index}: {actual} vs {expected}, bound {bound}"
        );
    }
}

#[test]
fn wgpu_pool1d_max_matches_sequential() {
    let sequential = SequentialBackend::new();
    let wgpu = WgpuBackend::new();
    let input = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 7], &INPUT);
    let input_wgpu = input.to_backend_on(&sequential, &wgpu);
    let output_layout = Layout::new(vec![1, 1, 3].into());

    let mut expected_storage = sequential.allocate::<f32>(3);
    coeus_ops::PoolOps::max_pool1d(
        &sequential,
        input.storage(),
        input.layout(),
        KERNEL_SIZE,
        STRIDE,
        PADDING,
        DILATION,
        &mut expected_storage,
        &output_layout,
    )
    .expect("sequential max_pool1d dispatch");
    let expected =
        Tensor::<f32, SequentialBackend>::from_raw_parts(expected_storage, output_layout.clone());

    let mut actual_storage = wgpu.allocate::<f32>(3);
    coeus_ops::PoolOps::max_pool1d(
        &wgpu,
        input_wgpu.storage(),
        input_wgpu.layout(),
        KERNEL_SIZE,
        STRIDE,
        PADDING,
        DILATION,
        &mut actual_storage,
        &output_layout,
    )
    .expect("WGPU max_pool1d dispatch");
    let actual = Tensor::<f32, WgpuBackend>::from_raw_parts(actual_storage, output_layout.clone())
        .to_backend_on(&wgpu, &sequential);
    assert_close(actual.as_slice(), expected.as_slice());

    let grad_out = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 3], &[1.0, 2.0, 3.0]);
    let grad_out_wgpu = grad_out.to_backend_on(&sequential, &wgpu);
    let input_layout = input.layout().clone();
    let mut expected_grad_storage = sequential.allocate::<f32>(7);
    sequential.fill(&mut expected_grad_storage, 0.0);
    coeus_ops::PoolOps::max_pool1d_backward(
        &sequential,
        grad_out.storage(),
        grad_out.layout(),
        input.storage(),
        &input_layout,
        KERNEL_SIZE,
        STRIDE,
        PADDING,
        DILATION,
        &mut expected_grad_storage,
        &input_layout,
    )
    .expect("sequential max_pool1d backward dispatch");
    let expected_grad = Tensor::<f32, SequentialBackend>::from_raw_parts(
        expected_grad_storage,
        input_layout.clone(),
    );

    let mut actual_grad_storage = wgpu.allocate::<f32>(7);
    wgpu.fill(&mut actual_grad_storage, 0.0);
    coeus_ops::PoolOps::max_pool1d_backward(
        &wgpu,
        grad_out_wgpu.storage(),
        grad_out_wgpu.layout(),
        input_wgpu.storage(),
        input_wgpu.layout(),
        KERNEL_SIZE,
        STRIDE,
        PADDING,
        DILATION,
        &mut actual_grad_storage,
        &input_layout,
    )
    .expect("WGPU max_pool1d backward dispatch");
    let actual_grad = Tensor::<f32, WgpuBackend>::from_raw_parts(actual_grad_storage, input_layout)
        .to_backend_on(&wgpu, &sequential);
    assert_close(actual_grad.as_slice(), expected_grad.as_slice());
}

#[test]
fn wgpu_pool1d_avg_matches_sequential() {
    let sequential = SequentialBackend::new();
    let wgpu = WgpuBackend::new();
    let input = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 7], &INPUT);
    let input_wgpu = input.to_backend_on(&sequential, &wgpu);
    let output_layout = Layout::new(vec![1, 1, 3].into());

    let mut expected_storage = sequential.allocate::<f32>(3);
    coeus_ops::PoolOps::avg_pool1d(
        &sequential,
        input.storage(),
        input.layout(),
        KERNEL_SIZE,
        STRIDE,
        PADDING,
        DILATION,
        &mut expected_storage,
        &output_layout,
    )
    .expect("sequential avg_pool1d dispatch");
    let expected =
        Tensor::<f32, SequentialBackend>::from_raw_parts(expected_storage, output_layout.clone());

    let mut actual_storage = wgpu.allocate::<f32>(3);
    coeus_ops::PoolOps::avg_pool1d(
        &wgpu,
        input_wgpu.storage(),
        input_wgpu.layout(),
        KERNEL_SIZE,
        STRIDE,
        PADDING,
        DILATION,
        &mut actual_storage,
        &output_layout,
    )
    .expect("WGPU avg_pool1d dispatch");
    let actual = Tensor::<f32, WgpuBackend>::from_raw_parts(actual_storage, output_layout)
        .to_backend_on(&wgpu, &sequential);
    assert_close(actual.as_slice(), expected.as_slice());

    let grad_out = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 3], &[2.0, 3.0, 2.0]);
    let grad_out_wgpu = grad_out.to_backend_on(&sequential, &wgpu);
    let input_layout = input.layout().clone();
    let mut expected_grad_storage = sequential.allocate::<f32>(7);
    sequential.fill(&mut expected_grad_storage, 0.0);
    coeus_ops::PoolOps::avg_pool1d_backward(
        &sequential,
        grad_out.storage(),
        grad_out.layout(),
        KERNEL_SIZE,
        STRIDE,
        PADDING,
        DILATION,
        &mut expected_grad_storage,
        &input_layout,
    )
    .expect("sequential avg_pool1d backward dispatch");
    let expected_grad = Tensor::<f32, SequentialBackend>::from_raw_parts(
        expected_grad_storage,
        input_layout.clone(),
    );

    let mut actual_grad_storage = wgpu.allocate::<f32>(7);
    wgpu.fill(&mut actual_grad_storage, 0.0);
    coeus_ops::PoolOps::avg_pool1d_backward(
        &wgpu,
        grad_out_wgpu.storage(),
        grad_out_wgpu.layout(),
        KERNEL_SIZE,
        STRIDE,
        PADDING,
        DILATION,
        &mut actual_grad_storage,
        &input_layout,
    )
    .expect("WGPU avg_pool1d backward dispatch");
    let actual_grad = Tensor::<f32, WgpuBackend>::from_raw_parts(actual_grad_storage, input_layout)
        .to_backend_on(&wgpu, &sequential);
    assert_close(actual_grad.as_slice(), expected_grad.as_slice());
}
