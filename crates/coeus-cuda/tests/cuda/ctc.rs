//! The CTC seam reaches either backend from one call site.
//!
//! `coeus_ops::CtcOps` is implemented for the CPU backend over Leto and for
//! this one over Hephaestus. These tests call the same trait methods on
//! both and compare, which is the claim the seam exists to support: a
//! consumer binds the trait, not a device.

use coeus_core::{ComputeBackend, Layout, SequentialBackend};
use coeus_cuda::CudaBackend;
use coeus_ops::{CtcBatch, CtcOps};
use coeus_tensor::Tensor;

const SHAPE: [usize; 3] = [6, 3, 5];
const INPUT_LENGTHS: [usize; 3] = [6, 5, 0];
const TARGET_LENGTHS: [usize; 3] = [2, 1, 0];
const TARGETS: [usize; 3] = [2, 2, 1];
const BLANK: usize = 0;
const UPSTREAM: f32 = 1.5;

fn layout_for(shape: &[usize]) -> Layout {
    Layout::new(shape.to_vec().into())
}

/// Deterministic nonpositive lanes, so the provider's input contract holds
/// on both backends without normalization.
fn log_probs(shape: [usize; 3]) -> Vec<f32> {
    let mut values = Vec::with_capacity(shape.iter().product());
    for t in 0..shape[0] {
        for b in 0..shape[1] {
            for c in 0..shape[2] {
                let hash = (t * 13 + b * 7 + c * 3 + 11) % 97;
                values.push(-(hash as f32 * 0.05 + 0.01));
            }
        }
    }
    values
}

fn batch<'a>(
    targets: &'a [usize],
    input_lengths: &'a [usize],
    target_lengths: &'a [usize],
) -> CtcBatch<'a> {
    CtcBatch {
        targets,
        input_lengths,
        target_lengths,
        blank: BLANK,
    }
}

/// The kernels keep the provider's operation order, so the claim is a
/// 64-ulp bound over libm rounding rather than bitwise equality.
fn assert_close(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    let scale = expected.iter().fold(1.0_f32, |acc, v| acc.max(v.abs()));
    let bound = 64.0 * f32::EPSILON * scale;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= bound,
            "{what}: lane {index}: {actual} vs {expected}, bound {bound:e}"
        );
    }
}

fn through_both() -> ((f32, Vec<f32>), (f32, Vec<f32>)) {
    let sequential = SequentialBackend;
    let cuda = CudaBackend::new();
    let host = log_probs(SHAPE);
    let grid_layout = layout_for(&SHAPE);
    let scalar_layout = layout_for(&[1]);
    let cells: usize = SHAPE.iter().product();

    let input = Tensor::<f32, SequentialBackend>::from_slice(SHAPE.to_vec(), &host);
    let input_cuda = input.to_backend_on(&sequential, &cuda);

    let mut cpu_loss_storage = sequential.allocate_zeroed::<f32>(1);
    let cpu_state = CtcOps::<f32>::ctc_forward(
        &sequential,
        input.storage(),
        input.layout(),
        batch(&TARGETS, &INPUT_LENGTHS, &TARGET_LENGTHS),
        &mut cpu_loss_storage,
        &scalar_layout,
    )
    .expect("sequential ctc forward");
    let mut cpu_grad_storage = sequential.allocate_zeroed::<f32>(cells);
    let upstream = Tensor::<f32, SequentialBackend>::from_slice(vec![1], &[UPSTREAM]);
    CtcOps::<f32>::ctc_backward_accumulate(
        &sequential,
        &cpu_state,
        upstream.storage(),
        upstream.layout(),
        &mut cpu_grad_storage,
        &grid_layout,
    )
    .expect("sequential ctc backward");
    let expected_loss =
        Tensor::<f32, SequentialBackend>::from_raw_parts(cpu_loss_storage, scalar_layout.clone())
            .as_slice()[0];
    let expected_grad =
        Tensor::<f32, SequentialBackend>::from_raw_parts(cpu_grad_storage, grid_layout.clone())
            .as_slice()
            .to_vec();

    let mut device_loss_storage = cuda.allocate_zeroed::<f32>(1);
    let device_state = CtcOps::<f32>::ctc_forward(
        &cuda,
        input_cuda.storage(),
        input_cuda.layout(),
        batch(&TARGETS, &INPUT_LENGTHS, &TARGET_LENGTHS),
        &mut device_loss_storage,
        &scalar_layout,
    )
    .expect("cuda ctc forward");
    let mut device_grad_storage = cuda.allocate_zeroed::<f32>(cells);
    let upstream_cuda = upstream.to_backend_on(&sequential, &cuda);
    CtcOps::<f32>::ctc_backward_accumulate(
        &cuda,
        &device_state,
        upstream_cuda.storage(),
        upstream_cuda.layout(),
        &mut device_grad_storage,
        &grid_layout,
    )
    .expect("cuda ctc backward");
    let actual_loss =
        Tensor::<f32, CudaBackend>::from_raw_parts(device_loss_storage, scalar_layout)
            .to_backend_on(&cuda, &sequential)
            .as_slice()[0];
    let actual_grad = Tensor::<f32, CudaBackend>::from_raw_parts(device_grad_storage, grid_layout)
        .to_backend_on(&cuda, &sequential)
        .as_slice()
        .to_vec();

    ((actual_loss, actual_grad), (expected_loss, expected_grad))
}

#[test]
fn cuda_ctc_matches_sequential_forward_and_backward() {
    if !crate::availability::device_available() {
        return;
    }
    let ((actual_loss, actual_grad), (expected_loss, expected_grad)) = through_both();
    assert_close(&[actual_loss], &[expected_loss], "ctc loss");
    assert_close(&actual_grad, &expected_grad, "ctc gradient");
}

/// An impossible alignment — a nonempty target with zero valid frames —
/// writes positive infinity on both backends and rejects the backward
/// pass on both, before the gradient changes.
#[test]
fn cuda_ctc_rejects_an_impossible_alignment_like_sequential() {
    if !crate::availability::device_available() {
        return;
    }
    let sequential = SequentialBackend;
    let cuda = CudaBackend::new();
    let shape = [4_usize, 2, 3];
    let host = log_probs(shape);
    let grid_layout = layout_for(&shape);
    let scalar_layout = layout_for(&[1]);
    let input = Tensor::<f32, SequentialBackend>::from_slice(shape.to_vec(), &host);
    let input_cuda = input.to_backend_on(&sequential, &cuda);
    let targets = [1_usize, 2];
    let input_lengths = [4_usize, 0];
    let target_lengths = [1_usize, 1];

    let mut cpu_loss = sequential.allocate_zeroed::<f32>(1);
    let cpu_state = CtcOps::<f32>::ctc_forward(
        &sequential,
        input.storage(),
        input.layout(),
        batch(&targets, &input_lengths, &target_lengths),
        &mut cpu_loss,
        &scalar_layout,
    )
    .expect("sequential ctc forward reports the impossible loss");
    let mut device_loss = cuda.allocate_zeroed::<f32>(1);
    let device_state = CtcOps::<f32>::ctc_forward(
        &cuda,
        input_cuda.storage(),
        input_cuda.layout(),
        batch(&targets, &input_lengths, &target_lengths),
        &mut device_loss,
        &scalar_layout,
    )
    .expect("cuda ctc forward reports the impossible loss");
    let cpu_loss_value =
        Tensor::<f32, SequentialBackend>::from_raw_parts(cpu_loss, scalar_layout.clone())
            .as_slice()[0];
    let device_loss_value =
        Tensor::<f32, CudaBackend>::from_raw_parts(device_loss, scalar_layout.clone())
            .to_backend_on(&cuda, &sequential)
            .as_slice()[0];
    assert_eq!(cpu_loss_value, f32::INFINITY);
    assert_eq!(device_loss_value, f32::INFINITY);

    let upstream = Tensor::<f32, SequentialBackend>::from_slice(vec![1], &[UPSTREAM]);
    let upstream_cuda = upstream.to_backend_on(&sequential, &cuda);
    let cells: usize = shape.iter().product();
    let mut cpu_grad = sequential.allocate_zeroed::<f32>(cells);
    let mut device_grad = cuda.allocate_zeroed::<f32>(cells);
    assert!(
        CtcOps::<f32>::ctc_backward_accumulate(
            &sequential,
            &cpu_state,
            upstream.storage(),
            upstream.layout(),
            &mut cpu_grad,
            &grid_layout,
        )
        .is_err(),
        "sequential ctc backward rejects the impossible alignment"
    );
    assert!(
        CtcOps::<f32>::ctc_backward_accumulate(
            &cuda,
            &device_state,
            upstream_cuda.storage(),
            upstream_cuda.layout(),
            &mut device_grad,
            &grid_layout,
        )
        .is_err(),
        "cuda ctc backward rejects the impossible alignment"
    );
}

/// CUDA dispatches with value and gradient parity through the tracked
/// autograd op: backward runs the posterior sweep on the device, so this
/// is the claim differentiation tracking exists to support.
#[test]
fn cuda_ctc_tracking_matches_sequential_value_and_gradient() {
    if !crate::availability::device_available() {
        return;
    }
    use coeus_autograd::{ctc_loss, Var};
    let sequential = SequentialBackend;
    let cuda = CudaBackend::new();
    let host = log_probs(SHAPE);
    let input = Tensor::<f32, SequentialBackend>::from_slice(SHAPE.to_vec(), &host);
    let cpu_input = Var::new(input.clone(), true);
    let cuda_input = Var::new(input.to_backend_on(&sequential, &cuda), true);
    let cpu_loss = ctc_loss(&cpu_input, &TARGETS, &INPUT_LENGTHS, &TARGET_LENGTHS, BLANK)
        .expect("CPU ctc forward must succeed");
    let cuda_loss = ctc_loss(
        &cuda_input,
        &TARGETS,
        &INPUT_LENGTHS,
        &TARGET_LENGTHS,
        BLANK,
    )
    .expect("CUDA ctc forward must succeed");
    let cuda_value = cuda_loss.tensor.to_backend_on(&cuda, &sequential);
    assert_close(
        cuda_value.as_slice(),
        cpu_loss.tensor.as_slice(),
        "tracked ctc loss",
    );

    cpu_loss.backward().expect("CPU ctc backward must succeed");
    cuda_loss
        .backward()
        .expect("CUDA ctc backward must succeed");
    let cpu_grad = cpu_input.grad().expect("CPU input tracks a gradient");
    let cuda_grad = cuda_input.grad().expect("CUDA input tracks a gradient");
    let cuda_grad_cpu = cuda_grad.to_backend_on(&cuda, &sequential);
    assert_close(
        cuda_grad_cpu.as_slice(),
        cpu_grad.as_slice(),
        "tracked ctc gradient",
    );
}
