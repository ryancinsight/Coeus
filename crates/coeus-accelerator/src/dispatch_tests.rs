//! Unavailable-device dispatch coverage (ADR 0079).
//!
//! Representative-pattern proof that every migrated dispatch path reports
//! acquisition failure instead of panicking: one `P::try_device` provider
//! path (elementwise binary), one `B::try_device` seam path (matmul), one
//! provider-associated path (`sgd_step`), plus the seam accessor itself.
//! The stub operations below always fail, so each test asserting
//! `DeviceUnavailable` also proves its kernel was never reached, and the
//! output-untouched assertions prove the failure published nothing.

use crate::backend::HephaestusBackend;
use crate::storage::tests::fixtures::{
    assert_device_unavailable, empty_buffer, expect_backend_failure, TestDevice,
    TestInitialization, UnavailableProvider,
};
use crate::storage::HephaestusStorage;
use crate::{
    ArithmeticUnaryOperations, ElementwiseProvider, HephaestusSeamBackend, MatmulProvider,
    StatefulUpdateProvider,
};
use coeus_core::{BinaryOp, Layout, Shape};
use hephaestus_core::{
    BinaryExpr, ComputeDevice, DenseProductOps, DeviceBuffer, ElementwiseOps, HephaestusError,
    Host, StatefulUpdateOps, StatefulUpdateRule, StridedView, UnaryExpr, UpdateWidth,
};
use themis::MemoryTier;

type UnavailableBackend = HephaestusBackend<UnavailableProvider>;
type UnavailableStorage = HephaestusStorage<UnavailableProvider, f32>;

/// Operation marker whose kernels always fail: acquisition fails before any
/// of these bodies can run, and the tests below prove it by asserting the
/// acquisition error instead.
#[derive(Debug, Default)]
pub(crate) struct UnavailableOps;

/// Uninhabited prepared-dispatch state: preparation always fails.
pub(crate) enum NeverPrepared {}

fn stub_unreachable() -> HephaestusError {
    HephaestusError::DispatchFailed {
        message: "unreachable: device acquisition fails before kernel dispatch".to_owned(),
    }
}

impl DenseProductOps<TestDevice, f32> for UnavailableOps {
    fn matmul_into(
        &self,
        _device: &TestDevice,
        _lhs: StridedView<'_, <TestDevice as ComputeDevice>::Buffer<f32>, 2>,
        _rhs: StridedView<'_, <TestDevice as ComputeDevice>::Buffer<f32>, 2>,
        _output: StridedView<'_, <TestDevice as ComputeDevice>::Buffer<f32>, 2>,
    ) -> hephaestus_core::Result<()> {
        Err(stub_unreachable())
    }

    fn batched_matmul_into(
        &self,
        _device: &TestDevice,
        _lhs: StridedView<'_, <TestDevice as ComputeDevice>::Buffer<f32>, 3>,
        _rhs: StridedView<'_, <TestDevice as ComputeDevice>::Buffer<f32>, 3>,
        _output: StridedView<'_, <TestDevice as ComputeDevice>::Buffer<f32>, 3>,
    ) -> hephaestus_core::Result<()> {
        Err(stub_unreachable())
    }

    fn kron_into(
        &self,
        _device: &TestDevice,
        _lhs: StridedView<'_, <TestDevice as ComputeDevice>::Buffer<f32>, 2>,
        _rhs: StridedView<'_, <TestDevice as ComputeDevice>::Buffer<f32>, 2>,
        _output: StridedView<'_, <TestDevice as ComputeDevice>::Buffer<f32>, 2>,
    ) -> hephaestus_core::Result<()> {
        Err(stub_unreachable())
    }
}

impl ElementwiseOps<TestDevice, f32> for UnavailableOps {
    type Dialect = Host;
    type PreparedUnary<'op, const N: usize> = NeverPrepared;
    type PreparedBinary<'op, const N: usize> = NeverPrepared;
    type PreparedScalar<'op, const N: usize> = NeverPrepared;
    type PreparedTypedBinary<'op, const N: usize> = NeverPrepared;

    fn prepare_unary_into<'op, Op, const N: usize>(
        &self,
        _device: &TestDevice,
        _input: StridedView<'op, <TestDevice as ComputeDevice>::Buffer<f32>, N>,
        _output: StridedView<'op, <TestDevice as ComputeDevice>::Buffer<f32>, N>,
    ) -> hephaestus_core::Result<Self::PreparedUnary<'op, N>>
    where
        Op: UnaryExpr<Self::Dialect>,
    {
        Err(stub_unreachable())
    }

    fn dispatch_unary<const N: usize>(
        &self,
        _device: &TestDevice,
        _prepared: &Self::PreparedUnary<'_, N>,
    ) -> hephaestus_core::Result<()> {
        Err(stub_unreachable())
    }

    fn prepare_binary_into<'op, Op, const N: usize>(
        &self,
        _device: &TestDevice,
        _lhs: StridedView<'op, <TestDevice as ComputeDevice>::Buffer<f32>, N>,
        _rhs: StridedView<'op, <TestDevice as ComputeDevice>::Buffer<f32>, N>,
        _output: StridedView<'op, <TestDevice as ComputeDevice>::Buffer<f32>, N>,
    ) -> hephaestus_core::Result<Self::PreparedBinary<'op, N>>
    where
        Op: BinaryExpr<Self::Dialect>,
    {
        Err(stub_unreachable())
    }

    fn dispatch_binary<const N: usize>(
        &self,
        _device: &TestDevice,
        _prepared: &Self::PreparedBinary<'_, N>,
    ) -> hephaestus_core::Result<()> {
        Err(stub_unreachable())
    }

    fn prepare_scalar_into<'op, Op, const N: usize>(
        &self,
        _device: &TestDevice,
        _input: StridedView<'op, <TestDevice as ComputeDevice>::Buffer<f32>, N>,
        _scalar: f32,
        _output: StridedView<'op, <TestDevice as ComputeDevice>::Buffer<f32>, N>,
    ) -> hephaestus_core::Result<Self::PreparedScalar<'op, N>>
    where
        Op: BinaryExpr<Self::Dialect>,
    {
        Err(stub_unreachable())
    }

    fn dispatch_scalar<const N: usize>(
        &self,
        _device: &TestDevice,
        _prepared: &Self::PreparedScalar<'_, N>,
    ) -> hephaestus_core::Result<()> {
        Err(stub_unreachable())
    }

    fn prepare_typed_binary_into<'op, Op, const N: usize>(
        &self,
        _device: &TestDevice,
        _lhs: StridedView<'op, <TestDevice as ComputeDevice>::Buffer<f32>, N>,
        _rhs: StridedView<'op, <TestDevice as ComputeDevice>::Buffer<f32>, N>,
        _output: StridedView<'op, <TestDevice as ComputeDevice>::Buffer<f32>, N>,
    ) -> hephaestus_core::Result<Self::PreparedTypedBinary<'op, N>>
    where
        Op: hephaestus_core::TypedBinaryExpr<Self::Dialect, f32>,
        f32: hephaestus_core::DialectScalar<Self::Dialect>,
    {
        Err(stub_unreachable())
    }

    fn dispatch_typed_binary<const N: usize>(
        &self,
        _device: &TestDevice,
        _prepared: &Self::PreparedTypedBinary<'_, N>,
    ) -> hephaestus_core::Result<()> {
        Err(stub_unreachable())
    }
}

impl StatefulUpdateOps<TestDevice> for UnavailableOps {
    type Dialect = Host;

    fn stateful_update<Rule, T: UpdateWidth, const N: usize>(
        &self,
        _device: &TestDevice,
        _operands: hephaestus_core::StatefulUpdateOperands<
            '_,
            <TestDevice as ComputeDevice>::Buffer<T>,
            N,
        >,
        _parameters: <Rule as StatefulUpdateRule<Self::Dialect>>::Parameters<T>,
    ) -> hephaestus_core::Result<()>
    where
        Rule: StatefulUpdateRule<Self::Dialect>,
    {
        Err(stub_unreachable())
    }
}

// SAFETY: the stub operations never execute (acquisition fails first); the
// trait contract they would uphold is therefore vacuously satisfied, and the
// tests below fail if any stub body ever runs.
unsafe impl ElementwiseProvider<f32> for UnavailableProvider {
    type Operations = UnavailableOps;
    type UnaryOperations = ArithmeticUnaryOperations;
}

// SAFETY: see `ElementwiseProvider` above.
unsafe impl MatmulProvider<f32> for UnavailableProvider {
    type Operations = UnavailableOps;
}

impl StatefulUpdateProvider for UnavailableProvider {
    type Operations = UnavailableOps;
}

fn storage(values: &[f32]) -> UnavailableStorage {
    let buffer = empty_buffer::<f32>(values.len(), MemoryTier::Device, TestInitialization::Zeroed)
        .expect("invariant: test buffer allocation succeeds");
    TestDevice
        .write_buffer(&buffer, values)
        .expect("invariant: test data write succeeds");
    // SAFETY: `write_buffer` initializes every element.
    unsafe { HephaestusStorage::from_buffer(buffer) }
}

fn download(storage: &UnavailableStorage) -> Vec<f32> {
    let mut values = vec![0.0; storage.buffer().len()];
    TestDevice
        .download(storage.buffer(), &mut values)
        .expect("invariant: test device can inspect storage");
    values
}

fn layout(shape: &[usize]) -> Layout {
    Layout::new(Shape::from(shape.to_vec()))
}

#[test]
fn seam_try_device_reports_unavailable_provider() {
    let error = <UnavailableBackend as HephaestusSeamBackend>::try_device()
        .expect_err("an unavailable provider must fail acquisition");
    assert!(
        matches!(error, HephaestusError::DeviceUnavailable { .. }),
        "expected DeviceUnavailable, got {error:?}"
    );
}

#[test]
fn matmul_dispatch_reports_acquisition_failure_without_touching_output() {
    let a = storage(&[1.0, 2.0, 3.0, 4.0]);
    let b = storage(&[5.0, 6.0, 7.0, 8.0]);
    let mut c = storage(&[0.0; 4]);
    let shape = layout(&[2, 2]);
    let before = download(&c);

    let error = expect_backend_failure(
        crate::matmul::<UnavailableBackend, f32>(&a, &shape, &b, &shape, &mut c, &shape),
        "matmul dispatch must report the unavailable device",
    );
    assert_device_unavailable(error, "matmul");
    assert_eq!(download(&c), before);
}

#[test]
fn elementwise_binary_dispatch_reports_acquisition_failure_without_touching_output() {
    use coeus_ops::ElementwiseOps;

    let backend = UnavailableBackend::new();
    let lhs = storage(&[1.0, 2.0, 3.0, 4.0]);
    let rhs = storage(&[5.0, 6.0, 7.0, 8.0]);
    let mut output = storage(&[0.0; 4]);
    let shape = layout(&[4]);
    let before = download(&output);

    let error = expect_backend_failure(
        backend.elementwise_binary(
            BinaryOp::Add,
            &lhs,
            &shape,
            &rhs,
            &shape,
            &mut output,
            &shape,
        ),
        "elementwise binary dispatch must report the unavailable device",
    );
    assert_device_unavailable(error, "elementwise_binary");
    assert_eq!(download(&output), before);
}

#[test]
fn sgd_step_reports_acquisition_failure_without_touching_state() {
    use coeus_ops::OptimizerOps;

    let backend = UnavailableBackend::new();
    let mut parameter = storage(&[1.0, 2.0]);
    let gradient = storage(&[0.5, 0.25]);
    let mut state = storage(&[0.0, 0.0]);
    let shape = layout(&[2]);
    let parameter_before = download(&parameter);
    let state_before = download(&state);

    let error = expect_backend_failure(
        backend.sgd_step(
            &mut parameter,
            &shape,
            &gradient,
            &shape,
            &mut state,
            &shape,
            0.1,
            0.9,
        ),
        "SGD step must report the unavailable device",
    );
    assert_device_unavailable(error, "SGD step");
    assert_eq!(download(&parameter), parameter_before);
    assert_eq!(download(&state), state_before);
}
