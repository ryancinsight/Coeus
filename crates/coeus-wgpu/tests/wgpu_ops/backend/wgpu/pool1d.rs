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
    let input = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 7], &INPUT)
        .expect("construct tensor");
    let input_wgpu = input.to_backend_on(&sequential, &wgpu).expect("transfer tensor");
    let output_layout = Layout::new(vec![1, 1, 3].into());

    let mut expected_storage = sequential.allocate::<f32>(3).expect("allocate tensor storage");
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
    .expect("execute CPU max pool");
    let expected =
        Tensor::<f32, SequentialBackend>::from_raw_parts(expected_storage, output_layout.clone());

    let mut actual_storage = wgpu.allocate::<f32>(3).expect("allocate tensor storage");
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
    .expect("execute WGPU max pool");
    let actual = Tensor::<f32, WgpuBackend>::from_raw_parts(actual_storage, output_layout.clone())
        .to_backend_on(&wgpu, &sequential)
        .expect("transfer tensor");
    assert_close(actual.as_slice(), expected.as_slice());

    let grad_out = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 3], &[1.0, 2.0, 3.0])
        .expect("construct tensor");
    let grad_out_wgpu = grad_out.to_backend_on(&sequential, &wgpu).expect("transfer tensor");
    let input_layout = input.layout().clone();
    let mut expected_grad_storage = sequential.allocate::<f32>(7).expect("allocate tensor storage");
    sequential
        .fill(&mut expected_grad_storage, 0.0)
        .expect("fill gradient storage");
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
    .expect("execute CPU max-pool backward");
    let expected_grad = Tensor::<f32, SequentialBackend>::from_raw_parts(
        expected_grad_storage,
        input_layout.clone(),
    );

    let mut actual_grad_storage = wgpu.allocate::<f32>(7).expect("allocate tensor storage");
    wgpu
        .fill(&mut actual_grad_storage, 0.0)
        .expect("fill gradient storage");
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
    .expect("execute WGPU max-pool backward");
    let actual_grad = Tensor::<f32, WgpuBackend>::from_raw_parts(actual_grad_storage, input_layout)
        .to_backend_on(&wgpu, &sequential)
        .expect("transfer tensor");
    assert_close(actual_grad.as_slice(), expected_grad.as_slice());
}

#[test]
fn wgpu_pool1d_avg_matches_sequential() {
    let sequential = SequentialBackend::new();
    let wgpu = WgpuBackend::new();
    let input = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 7], &INPUT)
        .expect("construct tensor");
    let input_wgpu = input.to_backend_on(&sequential, &wgpu).expect("transfer tensor");
    let output_layout = Layout::new(vec![1, 1, 3].into());

    let mut expected_storage = sequential.allocate::<f32>(3).expect("allocate tensor storage");
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
    .expect("execute CPU average pool");
    let expected =
        Tensor::<f32, SequentialBackend>::from_raw_parts(expected_storage, output_layout.clone());

    let mut actual_storage = wgpu.allocate::<f32>(3).expect("allocate tensor storage");
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
    .expect("execute WGPU average pool");
    let actual = Tensor::<f32, WgpuBackend>::from_raw_parts(actual_storage, output_layout)
        .to_backend_on(&wgpu, &sequential)
        .expect("transfer tensor");
    assert_close(actual.as_slice(), expected.as_slice());

    let grad_out = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 3], &[2.0, 3.0, 2.0])
        .expect("construct tensor");
    let grad_out_wgpu = grad_out.to_backend_on(&sequential, &wgpu).expect("transfer tensor");
    let input_layout = input.layout().clone();
    let mut expected_grad_storage = sequential.allocate::<f32>(7).expect("allocate tensor storage");
    sequential
        .fill(&mut expected_grad_storage, 0.0)
        .expect("fill gradient storage");
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
    .expect("execute CPU average-pool backward");
    let expected_grad = Tensor::<f32, SequentialBackend>::from_raw_parts(
        expected_grad_storage,
        input_layout.clone(),
    );

    let mut actual_grad_storage = wgpu.allocate::<f32>(7).expect("allocate tensor storage");
    wgpu
        .fill(&mut actual_grad_storage, 0.0)
        .expect("fill gradient storage");
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
    .expect("execute WGPU average-pool backward");
    let actual_grad = Tensor::<f32, WgpuBackend>::from_raw_parts(actual_grad_storage, input_layout)
        .to_backend_on(&wgpu, &sequential)
        .expect("transfer tensor");
    assert_close(actual_grad.as_slice(), expected_grad.as_slice());
}
