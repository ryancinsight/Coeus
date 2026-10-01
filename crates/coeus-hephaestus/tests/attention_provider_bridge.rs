use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Mutex,
};

use coeus_core::{Layout, Storage};
use coeus_hephaestus::{
    AttentionProvider, HephaestusBackend, HephaestusBackendError, HephaestusProvider,
    HephaestusStorage,
};
use coeus_ops::AttentionOps as CoeusAttentionOps;
use hephaestus_core::{
    plan_attention_backward, plan_attention_forward, AttentionBackwardOperands, AttentionCausality,
    AttentionForwardOperands, AttentionOps as HephaestusAttentionOps, ComputeDevice, DeviceBuffer,
    HephaestusError,
};
use leto::{ArrayView, ArrayViewMut};
use leto_ops::{
    scaled_dot_product_attention_backward_accumulate, scaled_dot_product_attention_into,
    AttentionGradients, AttentionMask, GroupedKeepMask,
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
    type Buffer<T: eunomia::Pod> = TestBuffer<T>;

    fn backend_name(&self) -> &'static str {
        "attention-bridge-test"
    }

    fn topology(&self) -> Option<&themis::GpuTopology> {
        None
    }

    fn alloc_zeroed_with_hint<T: eunomia::Pod>(
        &self,
        len: usize,
        hint: PlacementHint,
    ) -> hephaestus_core::Result<Self::Buffer<T>> {
        Ok(TestBuffer::new(vec![T::zeroed(); len], tier(hint)))
    }

    fn alloc_uninitialized_with_hint<T: eunomia::Pod>(
        &self,
        len: usize,
        hint: PlacementHint,
    ) -> hephaestus_core::Result<Self::Buffer<T>> {
        self.alloc_zeroed_with_hint(len, hint)
    }

    fn upload_with_hint<T: eunomia::Pod>(
        &self,
        host: &[T],
        hint: PlacementHint,
    ) -> hephaestus_core::Result<Self::Buffer<T>> {
        Ok(TestBuffer::new(host.to_vec(), tier(hint)))
    }

    fn download<T: eunomia::Pod>(
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

    fn write_buffer<T: eunomia::Pod>(
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

    fn write_sub_buffer<T: eunomia::Pod>(
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

    fn copy_buffer<T: eunomia::Pod>(
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
    type Error = HephaestusBackendError;

    const NAME: &'static str = "attention-bridge-test";

    fn device() -> &'static Self::Device {
        &DEVICE
    }

    fn try_device() -> hephaestus_core::Result<&'static Self::Device> {
        Ok(Self::device())
    }
}

#[derive(Clone, Copy, Default)]
struct TestAttentionOps;

struct TestPreparedForward<'a> {
    operands: AttentionForwardOperands<'a, TestBuffer<f32>, f32>,
}

struct TestPreparedBackward<'a> {
    operands: AttentionBackwardOperands<'a, TestBuffer<f32>, f32>,
}

static FORWARD_BATCHES: AtomicUsize = AtomicUsize::new(0);
static FORWARD_MASK_GROUP: AtomicUsize = AtomicUsize::new(0);
static FORWARD_CAUSAL: AtomicBool = AtomicBool::new(false);
static BACKWARD_GRADIENTS: AtomicUsize = AtomicUsize::new(0);
static FAIL_FORWARD: AtomicBool = AtomicBool::new(false);

impl HephaestusAttentionOps<TestDevice, f32> for TestAttentionOps {
    type PreparedForward<'a> = TestPreparedForward<'a>;
    type PreparedBackward<'a> = TestPreparedBackward<'a>;

    fn prepare_attention_forward<'a>(
        &self,
        _device: &'a TestDevice,
        operands: AttentionForwardOperands<'a, TestBuffer<f32>, f32>,
    ) -> hephaestus_core::Result<Self::PreparedForward<'a>> {
        plan_attention_forward(&operands, false)?;
        FORWARD_BATCHES.store(operands.query.layout.shape()[0], Ordering::SeqCst);
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
        Ok(TestPreparedForward { operands })
    }

    fn dispatch_attention_forward(
        &self,
        _device: &TestDevice,
        prepared: &Self::PreparedForward<'_>,
    ) -> hephaestus_core::Result<()> {
        if FAIL_FORWARD.swap(false, Ordering::SeqCst) {
            return Err(HephaestusError::DispatchFailed {
                message: "injected provider failure".into(),
            });
        }
        let operands = &prepared.operands;
        let query_values = operands
            .query
            .buffer
            .values
            .lock()
            .expect("test query buffer lock");
        let key_values = operands
            .key
            .buffer
            .values
            .lock()
            .expect("test key buffer lock");
        let value_values = operands
            .value
            .buffer
            .values
            .lock()
            .expect("test value buffer lock");
        let query =
            ArrayView::try_new(*operands.query.layout, &query_values).map_err(attention_error)?;
        let key = ArrayView::try_new(*operands.key.layout, &key_values).map_err(attention_error)?;
        let value =
            ArrayView::try_new(*operands.value.layout, &value_values).map_err(attention_error)?;
        let keep = operands.mask.grouped_keep();
        let keep_values = keep.map(|keep| {
            keep.view()
                .buffer
                .values
                .lock()
                .expect("test attention mask lock")
        });
        let keep_view = keep_values
            .as_ref()
            .zip(keep)
            .map(|(values, keep)| {
                ArrayView::try_new(*keep.view().layout, values.as_slice()).map_err(attention_error)
            })
            .transpose()?;
        let grouped = keep_view.map(|view| {
            (
                view,
                keep.expect("invariant: keep view matches keep descriptor")
                    .heads_per_batch(),
            )
        });
        let mask = grouped.map_or(
            if operands.mask.causality() == AttentionCausality::Causal {
                AttentionMask::Causal
            } else {
                AttentionMask::Unmasked
            },
            |(view, heads_per_batch)| {
                let grouped = GroupedKeepMask::new(view, heads_per_batch);
                if operands.mask.causality() == AttentionCausality::Causal {
                    AttentionMask::CausalGroupedKeep(grouped)
                } else {
                    AttentionMask::GroupedKeep(grouped)
                }
            },
        );
        let mut output_values = operands
            .output
            .buffer
            .values
            .lock()
            .expect("test attention output lock");
        let mut weight_values = operands
            .weights
            .buffer
            .values
            .lock()
            .expect("test attention weights lock");
        let mut output = ArrayViewMut::try_new(*operands.output.layout, &mut output_values)
            .map_err(attention_error)?;
        let mut weights = ArrayViewMut::try_new(*operands.weights.layout, &mut weight_values)
            .map_err(attention_error)?;
        scaled_dot_product_attention_into(
            &query,
            &key,
            &value,
            mask,
            operands.scale,
            &mut output,
            &mut weights,
        )
        .map_err(attention_error)
    }

    fn prepare_attention_backward<'a>(
        &self,
        _device: &'a TestDevice,
        operands: AttentionBackwardOperands<'a, TestBuffer<f32>, f32>,
    ) -> hephaestus_core::Result<Self::PreparedBackward<'a>> {
        plan_attention_backward(&operands, false)?;
        let selected = usize::from(operands.gradients.query.is_some())
            | (usize::from(operands.gradients.key.is_some()) << 1)
            | (usize::from(operands.gradients.value.is_some()) << 2);
        BACKWARD_GRADIENTS.store(selected, Ordering::SeqCst);
        Ok(TestPreparedBackward { operands })
    }

    fn dispatch_attention_backward(
        &self,
        _device: &TestDevice,
        prepared: &Self::PreparedBackward<'_>,
    ) -> hephaestus_core::Result<()> {
        let operands = &prepared.operands;
        let grad_output_values = operands
            .grad_output
            .buffer
            .values
            .lock()
            .expect("test attention output gradient lock");
        let query_values = operands
            .query
            .buffer
            .values
            .lock()
            .expect("test query buffer lock");
        let key_values = operands
            .key
            .buffer
            .values
            .lock()
            .expect("test key buffer lock");
        let value_values = operands
            .value
            .buffer
            .values
            .lock()
            .expect("test value buffer lock");
        let weight_values = operands
            .weights
            .buffer
            .values
            .lock()
            .expect("test attention weights lock");
        let grad_output = ArrayView::try_new(*operands.grad_output.layout, &grad_output_values)
            .map_err(attention_error)?;
        let query =
            ArrayView::try_new(*operands.query.layout, &query_values).map_err(attention_error)?;
        let key = ArrayView::try_new(*operands.key.layout, &key_values).map_err(attention_error)?;
        let value =
            ArrayView::try_new(*operands.value.layout, &value_values).map_err(attention_error)?;
        let weights = ArrayView::try_new(*operands.weights.layout, &weight_values)
            .map_err(attention_error)?;
        let mut query_gradient_values = operands.gradients.query.map(|gradient| {
            gradient
                .buffer
                .values
                .lock()
                .expect("test query gradient lock")
        });
        let mut key_gradient_values = operands.gradients.key.map(|gradient| {
            gradient
                .buffer
                .values
                .lock()
                .expect("test key gradient lock")
        });
        let mut value_gradient_values = operands.gradients.value.map(|gradient| {
            gradient
                .buffer
                .values
                .lock()
                .expect("test value gradient lock")
        });
        let mut query_gradient = match (query_gradient_values.as_mut(), operands.gradients.query) {
            (Some(values), Some(target)) => Some(
                ArrayViewMut::try_new(*target.layout, values.as_mut_slice())
                    .map_err(attention_error)?,
            ),
            (None, None) => None,
            _ => {
                return Err(HephaestusError::InvalidConfiguration {
                    message: "query gradient descriptor and storage disagree".into(),
                });
            }
        };
        let mut key_gradient = match (key_gradient_values.as_mut(), operands.gradients.key) {
            (Some(values), Some(target)) => Some(
                ArrayViewMut::try_new(*target.layout, values.as_mut_slice())
                    .map_err(attention_error)?,
            ),
            (None, None) => None,
            _ => {
                return Err(HephaestusError::InvalidConfiguration {
                    message: "key gradient descriptor and storage disagree".into(),
                });
            }
        };
        let mut value_gradient = match (value_gradient_values.as_mut(), operands.gradients.value) {
            (Some(values), Some(target)) => Some(
                ArrayViewMut::try_new(*target.layout, values.as_mut_slice())
                    .map_err(attention_error)?,
            ),
            (None, None) => None,
            _ => {
                return Err(HephaestusError::InvalidConfiguration {
                    message: "value gradient descriptor and storage disagree".into(),
                });
            }
        };
        scaled_dot_product_attention_backward_accumulate(
            &grad_output,
            &query,
            &key,
            &value,
            &weights,
            operands.scale,
            AttentionGradients::new(
                query_gradient.take(),
                key_gradient.take(),
                value_gradient.take(),
            ),
        )
        .map_err(attention_error)
    }
}

fn attention_error(error: impl std::fmt::Display) -> HephaestusError {
    HephaestusError::DispatchFailed {
        message: error.to_string(),
    }
}

// SAFETY: `TestAttentionOps` writes both forward output buffers on success and
// preserves initialized gradient values during its additive backward path.
unsafe impl AttentionProvider<f32> for TestProvider {
    type Operations = TestAttentionOps;
}

fn storage(len: usize) -> HephaestusStorage<TestProvider, f32> {
    HephaestusStorage::new(len).expect("invariant: test device allocation succeeds")
}

fn write_storage(storage: &HephaestusStorage<TestProvider, f32>, values: &[f32]) {
    DEVICE
        .write_buffer(storage.buffer(), values)
        .expect("invariant: test storage and input lengths match");
}

fn read_storage(storage: &HephaestusStorage<TestProvider, f32>) -> Vec<f32> {
    let mut values = vec![0.0; storage.len()];
    DEVICE
        .download(storage.buffer(), &mut values)
        .expect("invariant: test storage and output lengths match");
    values
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
    write_storage(&query, &[0.0; 8]);
    write_storage(&key, &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
    write_storage(&value, &[10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0]);
    write_storage(&mask, &[1.0, 1.0]);

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
    assert_eq!(
        read_storage(&output),
        [10.0, 11.0, 15.0, 16.0, 30.0, 31.0, 35.0, 36.0]
    );
    assert_eq!(
        read_storage(&weights),
        [1.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.5, 0.5]
    );

    let grad_output = storage(8);
    let mut grad_query = storage(8);
    let mut grad_value = storage(8);
    write_storage(&grad_output, &[1.0; 8]);
    write_storage(&grad_query, &[1.0; 8]);
    write_storage(&grad_value, &[1.0; 8]);
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
    assert_eq!(
        read_storage(&grad_query),
        [1.0, 1.0, -1.5, 3.5, 1.0, 1.0, -1.5, 3.5]
    );
    assert_eq!(
        read_storage(&grad_value),
        [2.5, 2.5, 1.5, 1.5, 2.5, 2.5, 1.5, 1.5]
    );

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
