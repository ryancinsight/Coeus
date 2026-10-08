use super::provider::CtcBackend;
use crate::layout::check_contiguous_exact;
use coeus_core::{Layout, StorageMut};
use coeus_ops::CtcBatch;
use core::borrow::Borrow;
use hephaestus_core::{
    ComputeDevice, CtcOps, CtcProblem, CtcStateBuffers, DeviceBuffer, HephaestusError,
};

fn check_grid<B>(
    operation: &'static str,
    operand: &'static str,
    layout: &Layout,
    storage_len: usize,
    shape: [usize; 3],
) -> Result<(), B::Error>
where
    B: CtcBackend,
{
    check_contiguous_exact::<3, _>(operation, "ctc", operand, layout, |reason| {
        B::ctc_configuration_error(operation, reason)
    })?;
    if layout.shape() != shape {
        return Err(B::ctc_configuration_error(
            operation,
            format!(
                "ctc {operand} shape {:?} must equal {:?}",
                layout.shape(),
                shape
            ),
        ));
    }
    let cells = shape.iter().product();
    if storage_len != cells {
        return Err(B::ctc_configuration_error(
            operation,
            format!("ctc {operand} storage holds {storage_len} lanes for {cells} cells"),
        ));
    }
    Ok(())
}

fn check_scalar<B>(
    operation: &'static str,
    operand: &'static str,
    layout: &Layout,
    storage_len: usize,
) -> Result<(), B::Error>
where
    B: CtcBackend,
{
    check_contiguous_exact::<1, _>(operation, "ctc", operand, layout, |reason| {
        B::ctc_configuration_error(operation, reason)
    })?;
    if layout.shape() != [1] {
        return Err(B::ctc_configuration_error(
            operation,
            format!("ctc {operand} shape {:?} must equal [1]", layout.shape()),
        ));
    }
    if storage_len != 1 {
        return Err(B::ctc_configuration_error(
            operation,
            format!("ctc {operand} storage holds {storage_len} lanes for 1 cell"),
        ));
    }
    Ok(())
}

fn map_problem_error<B>(operation: &'static str, error: HephaestusError) -> B::Error
where
    B: CtcBackend,
{
    match error {
        HephaestusError::InvalidConfiguration { message } => {
            B::ctc_configuration_error(operation, message)
        }
        error => B::ctc_dispatch_error(operation, error),
    }
}

/// Run the provider CTC forward, write the loss scalar, and retain state.
///
/// # Errors
///
/// Returns the backend's configuration error for a malformed problem or
/// layout, or its dispatch error for a provider failure. Provider
/// failures return before the loss destination changes.
pub fn ctc_forward<B>(
    log_probs: (&B::DeviceBuffer<f32>, &Layout),
    batch: CtcBatch,
    loss: (&mut B::DeviceBuffer<f32>, &Layout),
) -> Result<CtcStateBuffers<B::Device>, B::Error>
where
    B: CtcBackend,
    B::Kernel: Borrow<<B::Operations as CtcOps<B::Device>>::Ctc>,
{
    const OPERATION: &str = "ctc_forward";
    let (log_probs_storage, log_probs_layout) = log_probs;
    let (loss_storage, loss_layout) = loss;
    check_contiguous_exact::<3, _>(
        OPERATION,
        "ctc",
        "log-probability",
        log_probs_layout,
        |reason| B::ctc_configuration_error(OPERATION, reason),
    )?;
    let shape = log_probs_layout.shape();
    let (frames, batch_size, classes) = (shape[0], shape[1], shape[2]);
    let cells = frames
        .checked_mul(batch_size)
        .and_then(|cells| cells.checked_mul(classes))
        .ok_or_else(|| {
            B::ctc_configuration_error(OPERATION, "ctc grid shape overflows usize".to_string())
        })?;
    let lanes = B::ctc_buffer(log_probs_storage).len();
    if lanes != cells {
        return Err(B::ctc_configuration_error(
            OPERATION,
            format!("ctc log-probability storage holds {lanes} lanes for {cells} cells"),
        ));
    }
    check_scalar::<B>(
        OPERATION,
        "loss",
        loss_layout,
        B::ctc_buffer(loss_storage).len(),
    )?;
    let problem = CtcProblem::new(
        frames,
        batch_size,
        classes,
        batch.blank,
        batch.input_lengths,
        batch.target_lengths,
        batch.targets,
    )
    .map_err(|error| map_problem_error::<B>(OPERATION, error))?;
    let device = B::ctc_device();
    let state = CtcStateBuffers::allocate(device, problem)
        .map_err(|error| B::ctc_dispatch_error(OPERATION, error))?;
    let kernel = B::ctc_kernel()?;
    let value = B::Operations::default()
        .ctc_forward_into(
            device,
            kernel.borrow(),
            B::ctc_buffer(log_probs_storage),
            &state,
        )
        .map_err(|error| B::ctc_dispatch_error(OPERATION, error))?;
    loss_storage.make_unique();
    device
        .write_buffer(B::ctc_buffer(loss_storage), &[value])
        .map_err(|error| B::ctc_dispatch_error(OPERATION, error))?;
    Ok(state)
}

/// Accumulate the seeded CTC derivative into `grad` from retained `state`.
///
/// Reads the seed and per-sample reachability from the device before
/// touching the gradient, so an impossible alignment errors exactly as
/// the CPU implementation does.
///
/// # Errors
///
/// Returns the backend's configuration error for a malformed layout, its
/// gradient error for an impossible alignment or non-finite seed, or its
/// dispatch error for a provider failure. All rejections return before
/// the gradient destination changes.
pub fn ctc_backward<B>(
    state: &CtcStateBuffers<B::Device>,
    upstream: (&B::DeviceBuffer<f32>, &Layout),
    grad: (&mut B::DeviceBuffer<f32>, &Layout),
) -> Result<(), B::Error>
where
    B: CtcBackend,
    B::Kernel: Borrow<<B::Operations as CtcOps<B::Device>>::Ctc>,
{
    const OPERATION: &str = "ctc_backward";
    let (upstream_storage, upstream_layout) = upstream;
    let (grad_storage, grad_layout) = grad;
    let problem = &state.problem;
    check_scalar::<B>(
        OPERATION,
        "upstream",
        upstream_layout,
        B::ctc_buffer(upstream_storage).len(),
    )?;
    check_grid::<B>(
        OPERATION,
        "gradient",
        grad_layout,
        B::ctc_buffer(grad_storage).len(),
        [problem.frames(), problem.batch(), problem.classes()],
    )?;
    let device = B::ctc_device();
    let mut seed = [0.0_f32; 1];
    device
        .download(B::ctc_buffer(upstream_storage), &mut seed)
        .map_err(|error| B::ctc_dispatch_error(OPERATION, error))?;
    if !seed[0].is_finite() {
        return Err(B::ctc_arithmetic_error(OPERATION, 0));
    }
    let mut likelihood = vec![0.0_f32; problem.batch() * 2];
    device
        .download(&state.likelihood, &mut likelihood)
        .map_err(|error| B::ctc_dispatch_error(OPERATION, error))?;
    for sample in 0..problem.batch() {
        if likelihood[sample * 2] == f32::NEG_INFINITY {
            return Err(B::ctc_impossible_error(OPERATION, sample));
        }
    }
    grad_storage.make_unique();
    let kernel = B::ctc_kernel()?;
    B::Operations::default()
        .ctc_backward_into(
            device,
            kernel.borrow(),
            seed[0],
            B::ctc_buffer(grad_storage),
            state,
        )
        .map_err(|error| B::ctc_dispatch_error(OPERATION, error))
}
