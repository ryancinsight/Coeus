use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Mutex,
};

use bytemuck::Pod;
use coeus_core::Layout;
use coeus_hephaestus::{
    AttentionProvider, HephaestusBackend, HephaestusBackendError, HephaestusProvider,
    HephaestusStorage,
};
use coeus_ops::AttentionOps as CoeusAttentionOps;
use hephaestus_core::{
    AttentionBackwardOperands, AttentionCausality, AttentionForwardOperands,
    AttentionOps as HephaestusAttentionOps, ComputeDevice, DeviceBuffer, HephaestusError,
};
use themis::{MemoryTier, PlacementHint};

struct TestBuffer<T> {
    values: Mutex<Vec<T>>,
    tier: MemoryTier,
}

fn tier(hint: PlacementHint) -> MemoryTier {
    match hint {
        PlacementHint::Tier(tier) => tier,
        PlacementHint::Current
        | PlacementHint::Numa(_)
        | PlacementHint::Domain(_)
        | PlacementHint::Any => MemoryTier::Dram,
    }
}

impl<T> TestBuffer<T> {
    fn new(values: Vec<T>, tier: MemoryTier) -> Self {
        Self {
            values: Mutex::new(values),
            tier,
        }
    }
}

impl<T> DeviceBuffer<T> for TestBuffer<T> {
    fn len(&self) -> usize {
        self.values.lock().expect("test buffer lock").len()
    }

    fn tier(&self) -> MemoryTier {
        self.tier
    }
}

#[derive(Clone, Copy, Default)]
struct TestDevice;

fn length_mismatch<T>(host_len: usize, buffer: &TestBuffer<T>) -> HephaestusError {
    HephaestusError::LengthMismatch {
        host_len,
        device_len: buffer.len(),
    }
}

impl ComputeDevice for TestDevice {
    type Buffer<T: Pod> = TestBuffer<T>;

    fn backend_name(&self) -> &'static str {
        "attention-bridge-test"
    }

    fn topology(&self) -> Option<&themis::GpuTopology> {
        None
    }

    fn alloc_zeroed_with_hint<T: Pod>(
        &self,
        len: usize,
        hint: PlacementHint,
    ) -> hephaestus_core::Result<Self::Buffer<T>> {
        Ok(TestBuffer::new(vec![T::zeroed(); len], tier(hint)))
    }

    fn alloc_uninitialized_with_hint<T: Pod>(
        &self,
        len: usize,
        hint: PlacementHint,
    ) -> hephaestus_core::Result<Self::Buffer<T>> {
        self.alloc_zeroed_with_hint(len, hint)
    }

    fn upload_with_hint<T: Pod>(
        &self,
        host: &[T],
        hint: PlacementHint,
    ) -> hephaestus_core::Result<Self::Buffer<T>> {
        Ok(TestBuffer::new(host.to_vec(), tier(hint)))
    }

    fn download<T: Pod>(
        &self,
        buffer: &Self::Buffer<T>,
        out: &mut [T],
    ) -> hephaestus_core::Result<()> {
        let values = buffer.values.lock().expect("test buffer lock");
        if values.len() != out.len() {
            return Err(length_mismatch(out.len(), buffer));
        }
        out.copy_from_slice(&values);
        Ok(())
    }

    fn write_buffer<T: Pod>(
        &self,
        buffer: &Self::Buffer<T>,
        host: &[T],
    ) -> hephaestus_core::Result<()> {
        let mut values = buffer.values.lock().expect("test buffer lock");
        if values.len() != host.len() {
            return Err(length_mismatch(host.len(), buffer));
        }
        values.copy_from_slice(host);
        Ok(())
    }

    fn write_sub_buffer<T: Pod>(
        &self,
        buffer: &Self::Buffer<T>,
        offset: usize,
        host: &[T],
    ) -> hephaestus_core::Result<()> {
        let mut values = buffer.values.lock().expect("test buffer lock");
        let device_len = values.len();
        let end =
            offset
                .checked_add(host.len())
                .ok_or_else(|| HephaestusError::TransferFailed {
                    message: "test sub-buffer range overflow".into(),
                })?;
        let destination = values
            .get_mut(offset..end)
            .ok_or(HephaestusError::LengthMismatch {
                host_len: end,
                device_len,
            })?;
        destination.copy_from_slice(host);
        Ok(())
    }

    fn copy_buffer<T: Pod>(
        &self,
        src: &Self::Buffer<T>,
        dst: &Self::Buffer<T>,
    ) -> hephaestus_core::Result<()> {
        let source = src.values.lock().expect("test buffer lock").clone();
        self.write_buffer(dst, &source)
    }

    fn synchronize(&self) -> hephaestus_core::Result<()> {
        Ok(())
    }
}

#[derive(Clone, Copy, Default)]
struct TestProvider;

static DEVICE: TestDevice = TestDevice;

// SAFETY: test buffers own synchronized host memory and remain valid for the
// lifetime of every retained handle; dispatch is synchronous.
unsafe impl HephaestusProvider for TestProvider {
    type Device = TestDevice;

    const NAME: &'static str = "attention-bridge-test";

    fn device() -> &'static Self::Device {
        &DEVICE
    }
}

#[derive(Clone, Copy, Default)]
struct TestAttentionOps;

static FORWARD_BATCHES: AtomicUsize = AtomicUsize::new(0);
static FORWARD_MASK_GROUP: AtomicUsize = AtomicUsize::new(0);
static FORWARD_CAUSAL: AtomicBool = AtomicBool::new(false);
static BACKWARD_GRADIENTS: AtomicUsize = AtomicUsize::new(0);
static FAIL_FORWARD: AtomicBool = AtomicBool::new(false);

impl HephaestusAttentionOps<TestDevice, f32> for TestAttentionOps {
    type PreparedForward<'a> = ();
    type PreparedBackward<'a> = ();

    fn prepare_attention_forward<'a>(
        &self,
        _device: &'a TestDevice,
        operands: AttentionForwardOperands<'a, TestBuffer<f32>, f32>,
    ) -> hephaestus_core::Result<Self::PreparedForward<'a>> {
        FORWARD_BATCHES.store(operands.query.layout.shape[0], Ordering::SeqCst);
        FORWARD_CAUSAL.store(
            operands.mask.causality() == AttentionCausality::Causal,
            Ordering::SeqCst,
        );
        FORWARD_MASK_GROUP.store(
            operands
                .mask
                .grouped_keep()
                .map_or(0, |mask| mask.heads_per_batch().get()),
            Ordering::SeqCst,
        );
        Ok(())
    }

    fn dispatch_attention_forward(
        &self,
        _device: &TestDevice,
        _prepared: &Self::PreparedForward<'_>,
    ) -> hephaestus_core::Result<()> {
        if FAIL_FORWARD.swap(false, Ordering::SeqCst) {
            return Err(HephaestusError::DispatchFailed {
                message: "injected provider failure".into(),
            });
        }
        Ok(())
    }

    fn prepare_attention_backward<'a>(
        &self,
        _device: &'a TestDevice,
        operands: AttentionBackwardOperands<'a, TestBuffer<f32>, f32>,
    ) -> hephaestus_core::Result<Self::PreparedBackward<'a>> {
        let selected = usize::from(operands.gradients.query.is_some())
            | (usize::from(operands.gradients.key.is_some()) << 1)
            | (usize::from(operands.gradients.value.is_some()) << 2);
        BACKWARD_GRADIENTS.store(selected, Ordering::SeqCst);
        Ok(())
    }

    fn dispatch_attention_backward(
        &self,
        _device: &TestDevice,
        _prepared: &Self::PreparedBackward<'_>,
    ) -> hephaestus_core::Result<()> {
        Ok(())
    }
}

impl AttentionProvider<f32> for TestProvider {
    type Operations = TestAttentionOps;
}

fn storage(len: usize) -> HephaestusStorage<TestProvider, f32> {
    HephaestusStorage::new(len)
}

#[test]
fn attention_bridge_binds_provider_operands_and_maps_errors() {
    let backend = HephaestusBackend::<TestProvider>::new();
    let tensor_layout = Layout::new([2, 2, 2].into());
    let mask_layout = Layout::new([1, 2].into());
    let query = storage(8);
    let key = storage(8);
    let value = storage(8);
    let mask = storage(2);
    let mut output = storage(8);
    let mut weights = storage(8);

    backend
        .sdp_attention(
            &query,
            &tensor_layout,
            &key,
            &tensor_layout,
            &value,
            &tensor_layout,
            Some(&mask),
            Some(&mask_layout),
            true,
            0.5,
            &mut output,
            &tensor_layout,
            &mut weights,
            &tensor_layout,
        )
        .expect("forward provider dispatch");
    assert_eq!(FORWARD_BATCHES.load(Ordering::SeqCst), 2);
    assert!(FORWARD_CAUSAL.load(Ordering::SeqCst));
    assert_eq!(FORWARD_MASK_GROUP.load(Ordering::SeqCst), 2);

    let grad_output = storage(8);
    let mut grad_query = storage(8);
    let mut grad_value = storage(8);
    backend
        .sdp_attention_backward(
            &grad_output,
            &tensor_layout,
            &query,
            &tensor_layout,
            &key,
            &tensor_layout,
            &value,
            &tensor_layout,
            &weights,
            &tensor_layout,
            0.5,
            Some((&mut grad_query, &tensor_layout)),
            None,
            Some((&mut grad_value, &tensor_layout)),
        )
        .expect("backward provider dispatch");
    assert_eq!(BACKWARD_GRADIENTS.load(Ordering::SeqCst), 0b101);

    FAIL_FORWARD.store(true, Ordering::SeqCst);
    let error = backend
        .sdp_attention(
            &query,
            &tensor_layout,
            &key,
            &tensor_layout,
            &value,
            &tensor_layout,
            None,
            None,
            false,
            1.0,
            &mut output,
            &tensor_layout,
            &mut weights,
            &tensor_layout,
        )
        .expect_err("provider failure must remain typed");
    match error {
        HephaestusBackendError::Device { operation, source } => {
            assert_eq!(operation, "attention forward");
            assert_eq!(
                source.to_string(),
                "kernel dispatch failed: injected provider failure"
            );
        }
        other => panic!("expected typed provider failure, got {other}"),
    }
}
