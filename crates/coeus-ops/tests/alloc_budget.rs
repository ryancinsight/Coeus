//! Allocation-count budgets for the flat-index decode kernels.
//!
//! These kernels previously allocated a coordinate buffer inside their loop, so
//! the number of allocations grew with the size of the work: once per output
//! element for `gather`, `index_select`, and `repeat_interleave`, and once per
//! slice for `scatter_add` and `topk`.
//!
//! The property that fix establishes is not "faster" but "allocation count is
//! independent of workload size". That is what these tests assert, by running
//! each kernel at two sizes and requiring an identical count. Wall-clock timing
//! is the indirect proxy for this property and is noisy under host load;
//! counting allocations measures it directly and deterministically.
//!
//! ## Why this is its own test target
//!
//! It installs a `#[global_allocator]` that only counts allocations made on
//! the thread running a measurement window, so neither libtest's worker
//! threads nor a concurrently running test can perturb the counter. Keeping
//! the allocator out of the shared `ops` harness binary avoids imposing it
//! on unrelated tests.
//!
//! A regression here is a real defect: it means per-element or per-slice
//! allocation has returned to a kernel that had it removed.
//!
//! `scatter_add` allocates its result through the backend. Mnemosyne can make a
//! size-dependent number of system-allocation calls while satisfying that one
//! storage request, so counting those nested calls confuses provider policy
//! with kernel allocation growth. Its test suppresses system calls made inside
//! `ComputeBackend::allocate`, counts the request itself separately, and keeps
//! counting kernel-owned vectors. A per-element coordinate buffer therefore
//! still fails while allocator size-class behavior cannot perturb the oracle.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use coeus_core::{
    BackendError, ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, CpuStorage,
    Scalar, SequentialBackend, Storage,
};
use coeus_tensor::Tensor;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static STORAGE_REQUESTS: AtomicUsize = AtomicUsize::new(0);
// The allocation counter only advances on the measuring thread (see
// `COUNTING_DEPTH`), but `STORAGE_REQUESTS` is process-global and every test
// body reads deltas of it, so the oracle still serializes whole test bodies
// on this lock.
static ORACLE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Hold for a whole oracle test body (warmup plus measurement) so its
/// `STORAGE_REQUESTS` deltas never observe another test's storage requests.
fn oracle_guard() -> std::sync::MutexGuard<'static, ()> {
    ORACLE_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}

thread_local! {
    static STORAGE_ALLOCATION_DEPTH: Cell<usize> = const { Cell::new(0) };
    // Only the thread running a measurement window counts its allocations.
    // The harness runs tests on worker threads whose first-touch allocator
    // and channel activity otherwise lands inside another test's window and
    // makes the oracle flaky (#513). Kernels under test here are
    // single-threaded, so gating on the measuring thread excludes only
    // harness noise, never kernel allocations.
    static COUNTING_DEPTH: Cell<usize> = const { Cell::new(0) };
}

struct StorageAllocationScope;

impl StorageAllocationScope {
    fn enter() -> Self {
        STORAGE_ALLOCATION_DEPTH.with(|depth| depth.set(depth.get() + 1));
        Self
    }

    fn active() -> bool {
        STORAGE_ALLOCATION_DEPTH.with(|depth| depth.get() != 0)
    }
}

impl Drop for StorageAllocationScope {
    fn drop(&mut self) {
        STORAGE_ALLOCATION_DEPTH.with(|depth| depth.set(depth.get() - 1));
    }
}

struct CountingScope;

impl CountingScope {
    fn enter() -> Self {
        COUNTING_DEPTH.with(|depth| depth.set(depth.get() + 1));
        Self
    }

    fn active() -> bool {
        COUNTING_DEPTH.with(|depth| depth.get() != 0)
    }
}

impl Drop for CountingScope {
    fn drop(&mut self) {
        COUNTING_DEPTH.with(|depth| depth.set(depth.get() - 1));
    }
}

/// Forwards to the system allocator, counting allocation calls.
///
/// `realloc` and `alloc_zeroed` are intentionally left to the `GlobalAlloc`
/// default implementations, which route through `alloc`, so growth of a
/// collection is counted rather than hidden behind the system's optimised
/// paths. That makes the count slightly pessimistic and never optimistic,
/// which is the safe direction for a budget assertion.
struct CountingAllocator;

// SAFETY: every method forwards its arguments unchanged to `System`, which is a
// correct `GlobalAlloc`. The counter is a relaxed atomic add gated on two
// thread-local depths (plain `Cell` reads, no allocation) with no bearing on
// the returned pointers, so the allocator's contract is exactly `System`'s.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if CountingScope::active() && !StorageAllocationScope::active() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
}

#[global_allocator]
static ALLOC: CountingAllocator = CountingAllocator;

#[derive(Clone, Copy, Default)]
struct CountingBackend;

impl CountingBackend {
    fn allocate_storage<T: Scalar>(len: usize) -> CpuStorage<T> {
        STORAGE_REQUESTS.fetch_add(1, Ordering::Relaxed);
        let _scope = StorageAllocationScope::enter();
        CpuStorage::new(len).expect("invariant: test CPU storage allocation succeeds")
    }
}

/// SAFETY: the contract is that every `parallel_for` closure call is invoked
/// inline before returning, so no work escapes the call. This loop is the same
/// sequential iteration `SequentialBackend` discharges it with, and `f` is
/// `Send + Sync + 'static` either way.
unsafe impl coeus_core::backend::Backend for CountingBackend {
    #[inline]
    fn parallel_for<F>(&self, start: usize, end: usize, f: F)
    where
        F: Fn(usize) + Send + Sync + 'static,
    {
        for i in start..end {
            f(i);
        }
    }
}

/// # Safety-critical: the oracle, not the kernel
///
/// `CpuBackend` is the whole of it (one `i64` slice accessor), so the double
/// satisfies the kernel bounds and the storage scope can apply to every kernel
/// in this file rather than only to those whose bounds happen not to need it.
impl coeus_ops::backend_ops::CpuBackend for CountingBackend {
    #[inline]
    fn as_mut_slice_i64<'a>(
        &self,
        buf: &'a mut Self::DeviceBuffer<i64>,
    ) -> Result<&'a mut [i64], BackendError> {
        use coeus_core::CpuAddressableStorageMut;
        buf.as_mut_slice()
    }
}

// SAFETY: `allocate` returns initialized CPU storage, and every mutating
// operation preserves initialized storage on success.
unsafe impl ComputeBackend for CountingBackend {
    type Error = BackendError;
    type DeviceBuffer<T: Scalar> = CpuStorage<T>;
    type KernelDescriptor = ();
    type DispatchFuture<T: Scalar> = std::future::Ready<T>;

    fn name(&self) -> &'static str {
        "counting-sequential"
    }

    fn num_threads(&self) -> usize {
        1
    }

    unsafe fn allocate<T: Scalar>(&self, len: usize) -> Result<Self::DeviceBuffer<T>, Self::Error> {
        Ok(Self::allocate_storage(len))
    }

    fn allocate_zeroed<T: Scalar>(&self, len: usize) -> Result<Self::DeviceBuffer<T>, Self::Error> {
        Ok(Self::allocate_storage(len))
    }

    fn fill<T: Scalar>(&self, dst: &mut Self::DeviceBuffer<T>, val: T) -> Result<(), Self::Error> {
        dst.as_mut_slice()?.fill(val);
        Ok(())
    }

    fn copy_to_device<T: Scalar>(
        &self,
        src: &[T],
        dst: &mut Self::DeviceBuffer<T>,
    ) -> Result<(), Self::Error> {
        if src.len() != dst.len() {
            return Err(BackendError::BufferLengthMismatch {
                operation: "copy_to_device",
                source_len: src.len(),
                destination_len: dst.len(),
            });
        }
        dst.as_mut_slice()?.copy_from_slice(src);
        Ok(())
    }

    fn copy_to_host<T: Scalar>(
        &self,
        src: &Self::DeviceBuffer<T>,
        dst: &mut [T],
    ) -> Result<(), Self::Error> {
        dst.copy_from_slice(src.as_slice());
        Ok(())
    }
}

/// Allocation count observed while running `body`.
fn allocations_during<R>(body: impl FnOnce() -> R) -> usize {
    let _scope = CountingScope::enter();
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    let result = body();
    let after = ALLOCATIONS.load(Ordering::Relaxed);
    drop(result);
    after - before
}

fn tensor(shape: &[usize], values: &[f64]) -> Tensor<f64, SequentialBackend> {
    Tensor::from_slice_on(shape.to_vec(), values, &SequentialBackend::new())
        .expect("invariant: test backend operation succeeds")
}

fn ramp(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 + 1.0).collect()
}

fn indices(n: usize, extent: usize) -> Vec<f64> {
    (0..n).map(|i| (i % extent) as f64).collect()
}

/// Assert a kernel's **kernel-owned** allocation count does not grow with
/// workload size, and that its result-storage requests do not either.
///
/// Each closure is run once before measuring so that any one-time lazy
/// initialisation inside the op is not attributed to the smaller workload.
/// The large shape then repeats until its count stabilizes (at most three
/// rounds): first-large-call caches otherwise leak one-time allocations
/// into the measured window on some hosts (#513: 9 on the first large
/// call, 5 on every later call across 20 probe rounds).
///
/// # Why storage requests are a separate oracle
///
/// A result is allocated through the backend, and one storage request may be
/// satisfied with a *size-dependent* number of system allocations (see the
/// module docs). Counting those as kernel allocation growth measures provider
/// policy instead of the kernel, so `StorageAllocationScope` suppresses system
/// calls made inside `ComputeBackend::allocate` and each request is counted on
/// its own. A per-element or per-slice coordinate buffer is still counted --
/// it is a kernel-owned `Vec` -- so the property this exists to protect is
/// unchanged; only the confound is removed.
///
/// Without this split the oracle is platform-dependent: a 1 KiB result and a
/// 64 KiB result can differ by four system allocations on one host and by none
/// on another, which is a property of the allocator, not of `gather`.
fn assert_size_independent(
    kernel: &str,
    small: impl Fn() -> Box<dyn std::any::Any>,
    large: impl Fn() -> Box<dyn std::any::Any>,
    expected_storage_requests: usize,
) {
    let _oracle_guard = oracle_guard();
    drop(small());
    drop(large());
    // Stabilization rounds: repeat the large shape until its count stops
    // changing, so first-call caches settle before the measured window.
    let mut previous = allocations_during(&large);
    for _ in 0..2 {
        let current = allocations_during(&large);
        if current == previous {
            break;
        }
        previous = current;
    }

    STORAGE_REQUESTS.store(0, Ordering::Relaxed);
    let small_allocs = allocations_during(&small);
    let small_storage_requests = STORAGE_REQUESTS.swap(0, Ordering::Relaxed);
    let large_allocs = allocations_during(&large);
    let large_storage_requests = STORAGE_REQUESTS.swap(0, Ordering::Relaxed);

    assert_eq!(
        small_allocs, large_allocs,
        "{kernel}: kernel-owned allocation count must not scale with workload \
         size (small={small_allocs}, large={large_allocs}). A difference means \
         a per-element or per-slice allocation has returned to this kernel."
    );
    assert_eq!(
        (small_storage_requests, large_storage_requests),
        (expected_storage_requests, expected_storage_requests),
        "{kernel}: result-storage requests must not scale with workload size \
         (small={small_storage_requests}, large={large_storage_requests}), and \
         must be exactly {expected_storage_requests} per call."
    );
}

#[test]
fn gather_allocation_count_is_independent_of_output_size() {
    let backend = CountingBackend;
    let build = |s: [usize; 3]| {
        let input = Tensor::from_slice_on(s.to_vec(), &ramp(s.iter().product()), &backend);
        let idx_shape = [s[0], s[1] / 2, s[2]];
        let idx_numel: usize = idx_shape.iter().product();
        let index = Tensor::from_slice_on(idx_shape.to_vec(), &indices(idx_numel, s[1]), &backend);
        (input, index)
    };
    let (si, sx) = build([4, 8, 4]);
    let (li, lx) = build([16, 32, 16]);

    assert_size_independent(
        "gather",
        || {
            Box::new(
                coeus_ops::gather(&si, 1, &sx, &backend)
                    .expect("invariant: test operation succeeds"),
            )
        },
        || {
            Box::new(
                coeus_ops::gather(&li, 1, &lx, &backend)
                    .expect("invariant: test operation succeeds"),
            )
        },
    );
}

#[test]
fn index_select_allocation_count_is_independent_of_output_size() {
    let backend = CountingBackend;
    let build = |s: [usize; 3]| {
        let input = Tensor::from_slice_on(s.to_vec(), &ramp(s.iter().product()), &backend);
        let take = s[1] / 2;
        let index = Tensor::from_slice_on(vec![take], &indices(take, s[1]), &backend);
        (input, index)
    };
    let (si, sx) = build([4, 8, 4]);
    let (li, lx) = build([16, 32, 16]);

    assert_size_independent(
        "index_select",
        || {
            Box::new(
                coeus_ops::index_select(&si, 1, &sx, &backend)
                    .expect("invariant: test operation succeeds"),
            )
        },
        || {
            Box::new(
                coeus_ops::index_select(&li, 1, &lx, &backend)
                    .expect("invariant: test operation succeeds"),
            )
        },
    );
}

#[test]
fn repeat_interleave_allocation_count_is_independent_of_output_size() {
    let backend = CountingBackend;
    let small = Tensor::from_slice_on(vec![4, 8, 4], &ramp(4 * 8 * 4), &backend);
    let large = Tensor::from_slice_on(vec![16, 32, 16], &ramp(16 * 32 * 16), &backend);

    assert_size_independent(
        "repeat_interleave",
        || {
            Box::new(
                coeus_ops::repeat_interleave(&small, 2, 1, &backend)
                    .expect("invariant: test operation succeeds"),
            )
        },
        || {
            Box::new(
                coeus_ops::repeat_interleave(&large, 2, 1, &backend)
                    .expect("invariant: test operation succeeds"),
            )
        },
    );
}

#[test]
fn scatter_add_allocation_count_is_independent_of_index_size() {
    let _oracle_guard = oracle_guard();
    let backend = CountingBackend;
    let build = |s: [usize; 3]| {
        let input = Tensor::from_slice_on(s.to_vec(), &ramp(s.iter().product()), &backend)
            .expect("invariant: test backend operation succeeds");
        let src_shape = [s[0], s[1] / 2, s[2]];
        let src_numel: usize = src_shape.iter().product();
        let src = Tensor::from_slice_on(src_shape.to_vec(), &ramp(src_numel), &backend)
            .expect("invariant: test backend operation succeeds");
        let index = Tensor::from_slice_on(src_shape.to_vec(), &indices(src_numel, s[1]), &backend)
            .expect("invariant: test backend operation succeeds");
        (input, index, src)
    };
    let (si, sx, ss) = build([4, 8, 4]);
    let (li, lx, ls) = build([16, 32, 16]);

    drop(
        coeus_ops::scatter_add(&si, 1, &sx, &ss, &backend)
            .expect("invariant: test operation succeeds"),
    );
    drop(
        coeus_ops::scatter_add(&li, 1, &lx, &ls, &backend)
            .expect("invariant: test operation succeeds"),
    );

    STORAGE_REQUESTS.store(0, Ordering::Relaxed);
    let small_allocs = allocations_during(|| {
        Box::new(
            coeus_ops::scatter_add(&si, 1, &sx, &ss, &backend)
                .expect("invariant: test operation succeeds"),
        )
    });
    let small_storage_requests = STORAGE_REQUESTS.swap(0, Ordering::Relaxed);
    let large_allocs = allocations_during(|| {
        Box::new(
            coeus_ops::scatter_add(&li, 1, &lx, &ls, &backend)
                .expect("invariant: test operation succeeds"),
        )
    });
    let large_storage_requests = STORAGE_REQUESTS.swap(0, Ordering::Relaxed);

    assert_eq!(
        small_allocs, large_allocs,
        "scatter_add: kernel-owned allocation count must not scale with workload size \
         (small={small_allocs}, large={large_allocs})"
    );
    assert_eq!(
        (small_storage_requests, large_storage_requests),
        (1, 1),
        "scatter_add must make exactly one result-storage request per call",
    );
}

#[test]
fn topk_allocation_count_is_independent_of_slice_count() {
    // `k` and the reduced extent are held constant so only the number of outer
    // slices varies; the old per-slice buffers scaled with exactly that.
    //
    // Unlike the three kernels above this one is measured against the raw
    // backend: `topk` takes no backend argument and uses `B::default()`, and
    // its `B: CpuBackend` bound is not satisfied by `CountingBackend`. Routing
    // its result storage through the counting scope would mean implementing
    // `CpuBackend` for the double, which is more machinery than the property
    // needs -- so the count here is kernel-owned allocations only, and the
    // storage confound this file guards against does not apply until it does.
    let _oracle_guard = oracle_guard();
    let small = tensor(&[4, 8, 4], &ramp(4 * 8 * 4));
    let large = tensor(&[16, 8, 16], &ramp(16 * 8 * 16));

    assert_size_independent(
        "topk",
        || {
            Box::new(
                coeus_ops::topk(&small, 2, 1, true).expect("invariant: test operation succeeds"),
            )
        },
        || {
            Box::new(
                coeus_ops::topk(&large, 2, 1, true).expect("invariant: test operation succeeds"),
            )
        },
    );
}
