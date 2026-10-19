//! Forced backend-allocation failure fixture for unary autograd execution.
//!
//! Item `COEUS-FALLIBLE-UNARY-EXECUTION`: the `Result`-returning
//! `UnaryAutogradOp::forward`/`backward` and `unary_op` migration landed under
//! `COEUS-FALLIBLE-STORAGE-CORE`. These tests prove a backend allocation
//! failure surfaces as the exact typed error through both the forward
//! construction path and the `Var::backward` graph channel, and that the
//! migrated signatures preserve exact values and gradients when the backend
//! succeeds.

use std::collections::TryReserveError;
use std::sync::Arc;

use coeus_autograd::ops::activation::{unary_op, UnaryAutogradOp, UnaryNode};
use coeus_autograd::{GradBuffer, Var};
use coeus_core::{BackendError, MoiraiBackend};
use coeus_tensor::Tensor;

/// A real `TryReserveError`, produced by a genuinely impossible reservation.
/// No allocation is attempted and no memory is touched.
fn reserve_error() -> TryReserveError {
    Vec::<u8>::new()
        .try_reserve(usize::MAX)
        .expect_err("reserving usize::MAX bytes must fail")
}

fn allocation_error(operation: &'static str) -> BackendError {
    BackendError::Allocation {
        operation,
        source: reserve_error(),
    }
}

fn assert_allocation_error(error: BackendError, operation: &'static str) {
    assert_eq!(error, allocation_error(operation));
}

/// Test-double op whose forward pass fails exactly like an over-budget
/// backend allocation.
struct FailingForwardOp;

impl UnaryAutogradOp<f32, MoiraiBackend> for FailingForwardOp {
    const OP_NAME: &'static str = "failing_forward_fixture";

    fn forward(
        _x: &Tensor<f32, MoiraiBackend>,
        _backend: &MoiraiBackend,
    ) -> Result<Tensor<f32, MoiraiBackend>, BackendError> {
        Err(allocation_error(Self::OP_NAME))
    }

    fn backward(
        _grad_out: &Tensor<f32, MoiraiBackend>,
        _x: &Tensor<f32, MoiraiBackend>,
        _y: &Tensor<f32, MoiraiBackend>,
        _backend: &MoiraiBackend,
    ) -> Result<Tensor<f32, MoiraiBackend>, BackendError> {
        Err(allocation_error(Self::OP_NAME))
    }
}

/// Test-double op whose forward succeeds (identity) but whose derivative
/// computation fails exactly like an over-budget backend allocation.
struct FailingBackwardOp;

impl UnaryAutogradOp<f32, MoiraiBackend> for FailingBackwardOp {
    const OP_NAME: &'static str = "failing_backward_fixture";

    fn forward(
        x: &Tensor<f32, MoiraiBackend>,
        _backend: &MoiraiBackend,
    ) -> Result<Tensor<f32, MoiraiBackend>, BackendError> {
        Ok(x.clone())
    }

    fn backward(
        _grad_out: &Tensor<f32, MoiraiBackend>,
        _x: &Tensor<f32, MoiraiBackend>,
        _y: &Tensor<f32, MoiraiBackend>,
        _backend: &MoiraiBackend,
    ) -> Result<Tensor<f32, MoiraiBackend>, BackendError> {
        Err(allocation_error(Self::OP_NAME))
    }
}

#[test]
fn unary_forward_failure_returns_the_exact_typed_error() {
    let input: Var<f32, MoiraiBackend> = Var::new(
        Tensor::from_slice([2], &[1.0f32, 2.0])
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");

    let error = match unary_op::<f32, MoiraiBackend, FailingForwardOp>(&input) {
        Err(error) => error,
        Ok(_) => panic!("a failing forward pass must surface its error"),
    };
    assert_allocation_error(error, "failing_forward_fixture");
}

#[test]
fn unary_backward_failure_returns_the_exact_typed_error_through_the_graph() {
    let input: Var<f32, MoiraiBackend> = Var::new(
        Tensor::from_slice([2], &[1.0f32, 2.0])
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let out_tensor = input.tensor.clone();
    let output_grad = Arc::new(GradBuffer::new(
        Tensor::zeros([2]).expect("invariant: test backend operation succeeds"),
    ));
    let node = Arc::new(UnaryNode {
        output_grad: Arc::clone(&output_grad),
        inputs: [input].into_iter().collect(),
        a_tensor: out_tensor.clone(),
        out_tensor: out_tensor.clone(),
        _phantom: std::marker::PhantomData::<FailingBackwardOp>,
    });
    let output = Var::with_creator(out_tensor, Some(output_grad), node);

    let error = output
        .backward()
        .expect_err("a failing derivative must surface through Var::backward");
    assert_allocation_error(error, "failing_backward_fixture");
}

#[test]
fn migrated_unary_signatures_preserve_exact_values_and_gradients() {
    let input: Var<f32, MoiraiBackend> = Var::new(
        Tensor::from_slice([2], &[1.0f32, -2.0])
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");

    let output = coeus_autograd::neg(&input).expect("neg forward succeeds");
    assert_eq!(output.tensor.as_slice(), &[-1.0, 2.0]);

    let loss = coeus_autograd::sum(&output).expect("sum succeeds");
    loss.backward().expect("backward succeeds");
    let grad = input.grad.expect("tracked input holds a gradient");
    assert_eq!(grad.read().as_slice(), &[-1.0, -1.0]);
}
