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
//! It installs a `#[global_allocator]`, which is process-wide. Nextest executes
//! each test in a separate process, so the counter cannot be perturbed by a
//! concurrently running test, but keeping the allocator out of the shared
//! `ops` harness binary avoids imposing it on unrelated tests.
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
//!
//! ## Allocation-site attribution
//!
//! A count mismatch reports only that the counts differ, not *which*
//! allocation is extra — and the `scatter_add` budget flakes on hosted CI
//! without ever reproducing locally, so the extra allocation has never been
//! observed. Setting `COEUS_ALLOC_ATTRIBUTION=1` makes the allocator record
//! a sequence number, size, and raw instruction pointers per counted
//! allocation (`backtrace::trace`, which captures without resolving), and a
//! mismatch then prints both sides' sites before failing; resolve the
//! addresses offline with `addr2line` against the test binary (resolving at
//! print time can block where no symbol server is reachable).
//! Recording's own allocations are guard-skipped rather than counted, and
//! the flag's absence leaves the counter-only path untouched.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use coeus_core::{
    BackendError, ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, CpuStorage,
    Scalar, SequentialBackend,
};
use coeus_tensor::Tensor;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static STORAGE_REQUESTS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static STORAGE_ALLOCATION_DEPTH: Cell<usize> = const { Cell::new(0) };
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

/// Forwards to the system allocator, counting allocation calls.
///
/// `realloc` and `alloc_zeroed` are intentionally left to the `GlobalAlloc`
/// default implementations, which route through `alloc`, so growth of a
/// collection is counted rather than hidden behind the system's optimised
/// paths. That makes the count slightly pessimistic and never optimistic,
/// which is the safe direction for a budget assertion.
struct CountingAllocator;

// SAFETY: every method forwards its arguments unchanged to `System`, which is a
// correct `GlobalAlloc`. The counter is a relaxed atomic add with no bearing on
// the returned pointers, so the allocator's contract is exactly `System`'s.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !ATTRIBUTION_GUARD.with(|guard| guard.get()) && !StorageAllocationScope::active() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            if ATTRIBUTION_ARMED.load(Ordering::Relaxed) {
                record_attribution_into(&ATTRIBUTION_LOG, &ATTRIBUTION_SEQ, layout);
            }
        }
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
}

#[global_allocator]
static ALLOC: CountingAllocator = CountingAllocator;

/// One counted allocation with its capture site.
///
/// Recorded only when [`attribution_enabled`] holds; otherwise the allocator
/// only bumps the counter and the log stays empty.
struct Attribution {
    seq: usize,
    size: usize,
    /// Raw instruction pointers, innermost first. Captured with the
    /// resolution-free `backtrace::trace` — never `std::backtrace` formatting,
    /// whose symbol resolution blocks where no symbol server is reachable.
    ips: Vec<usize>,
}

static ATTRIBUTION_SEQ: AtomicUsize = AtomicUsize::new(0);
static ATTRIBUTION_LOG: Mutex<Vec<Attribution>> = Mutex::new(Vec::new());

thread_local! {
    /// Set while the allocator records an entry, so the recording's own
    /// allocations (backtrace frames, log growth) are neither counted nor
    /// recorded instead of recursing.
    static ATTRIBUTION_GUARD: Cell<bool> = const { Cell::new(false) };
}

/// Whether allocation-site attribution is recorded.
///
/// Read once from `COEUS_ALLOC_ATTRIBUTION`; when unset, recording never
/// runs and every measurement behaves exactly as the counter-only path.
///
/// Call only from ordinary code, never from the allocator: the `OnceLock`
/// initialiser reads the environment, which allocates, and re-entering the
/// lock from inside `alloc` deadlocks the process at startup. The allocator
/// reads [`ATTRIBUTION_ARMED`] instead, which [`allocations_during`] sets
/// from safe context before each measurement.
fn attribution_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("COEUS_ALLOC_ATTRIBUTION").is_some())
}

/// Attribution switch the allocator may read.
///
/// A plain atomic: reading it allocates nothing and blocks on nothing, so
/// it is safe inside `alloc`, unlike the `OnceLock` behind
/// [`attribution_enabled`].
static ATTRIBUTION_ARMED: AtomicBool = AtomicBool::new(false);

fn lock_log(log: &Mutex<Vec<Attribution>>) -> std::sync::MutexGuard<'_, Vec<Attribution>> {
    log.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Record `layout` into `log`, returning the entry's sequence number.
///
/// Takes the log and sequence counter as parameters so the plumbing test
/// below runs against local state instead of racing the global log with
/// concurrently running tests.
fn record_attribution_into(
    log: &Mutex<Vec<Attribution>>,
    seq: &AtomicUsize,
    layout: Layout,
) -> usize {
    if ATTRIBUTION_GUARD.with(|guard| guard.replace(true)) {
        return usize::MAX;
    }
    let seq = seq.fetch_add(1, Ordering::Relaxed);
    // Raw-IP capture needs no environment and resolves nothing: attribution
    // must work on a hosted runner whose environment and symbol access the
    // test does not control. The capture's own allocations land while the
    // guard above is set, so they are skipped, not recorded.
    let mut ips = Vec::new();
    backtrace::trace(|frame| {
        ips.push(frame.ip() as usize);
        true
    });
    let entry = Attribution {
        seq,
        size: layout.size(),
        ips,
    };
    lock_log(log).push(entry);
    ATTRIBUTION_GUARD.with(|guard| guard.set(false));
    seq
}

fn drain_attribution_into(log: &Mutex<Vec<Attribution>>) -> Vec<Attribution> {
    std::mem::take(&mut *lock_log(log))
}

/// Render `entries` for a count-mismatch failure.
///
/// Prints raw instruction addresses, never resolved symbols: resolving
/// frames touches the platform symbol provider, which blocks without
/// symbol-server access (observed as a hang formatting deep kernel stacks
/// on a sandboxed runner). Resolve the addresses offline with
/// `addr2line`/`llvm-symbolizer` against the test binary instead.
fn format_attribution(kernel: &str, side: &str, entries: &[Attribution]) -> String {
    if entries.is_empty() {
        return format!(
            "{kernel} {side}: attribution log empty \
             (rerun with COEUS_ALLOC_ATTRIBUTION=1 to capture sites)\n"
        );
    }
    use std::fmt::Write as _;

    let mut out = String::new();
    for entry in entries {
        writeln!(
            out,
            "{kernel} {side}: alloc #{} ({} bytes) at:",
            entry.seq, entry.size
        )
        .expect("writing to a String is infallible");
        for ip in &entry.ips {
            writeln!(out, "    0x{ip:x}").expect("writing to a String is infallible");
        }
    }
    out
}

#[derive(Clone, Copy, Default)]
struct CountingBackend;

impl CountingBackend {
    fn allocate_storage<T: Scalar>(len: usize) -> CpuStorage<T> {
        STORAGE_REQUESTS.fetch_add(1, Ordering::Relaxed);
        let _scope = StorageAllocationScope::enter();
        CpuStorage::new(len)
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
    fn as_mut_slice_i64<'a>(&self, buf: &'a mut Self::DeviceBuffer<i64>) -> &'a mut [i64] {
        use coeus_core::CpuAddressableStorageMut;
        buf.as_mut_slice()
    }
}

impl ComputeBackend for CountingBackend {
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

    fn allocate<T: Scalar>(&self, len: usize) -> Self::DeviceBuffer<T> {
        Self::allocate_storage(len)
    }

    fn allocate_zeroed<T: Scalar>(&self, len: usize) -> Self::DeviceBuffer<T> {
        Self::allocate_storage(len)
    }

    fn fill<T: Scalar>(&self, dst: &mut Self::DeviceBuffer<T>, val: T) {
        dst.as_mut_slice().fill(val);
    }

    fn copy_to_device<T: Scalar>(&self, src: &[T], dst: &mut Self::DeviceBuffer<T>) {
        dst.as_mut_slice().copy_from_slice(src);
    }

    fn copy_to_host<T: Scalar>(&self, src: &Self::DeviceBuffer<T>, dst: &mut [T]) {
        dst.copy_from_slice(src.as_slice());
    }
}

/// Allocation count observed while running `body`, plus the entries recorded
/// during it (empty unless `COEUS_ALLOC_ATTRIBUTION` is set).
fn allocations_during<R>(body: impl FnOnce() -> R) -> (usize, Vec<Attribution>) {
    // Settle the enabled flag and drop warmup entries outside the window, so
    // neither can land inside one side's measurement but not the other's.
    // Arming reads the `OnceLock` here, in ordinary code: the allocator
    // itself must never touch it (see `attribution_enabled`).
    ATTRIBUTION_ARMED.store(attribution_enabled(), Ordering::Relaxed);
    drop(drain_attribution_into(&ATTRIBUTION_LOG));
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    let result = body();
    let after = ALLOCATIONS.load(Ordering::Relaxed);
    drop(result);
    (after - before, drain_attribution_into(&ATTRIBUTION_LOG))
}

fn tensor(shape: &[usize], values: &[f64]) -> Tensor<f64, SequentialBackend> {
    Tensor::from_slice_on(shape.to_vec(), values, &SequentialBackend::new())
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
    drop(small());
    drop(large());

    STORAGE_REQUESTS.store(0, Ordering::Relaxed);
    let (small_allocs, small_sites) = allocations_during(&small);
    let small_storage_requests = STORAGE_REQUESTS.swap(0, Ordering::Relaxed);
    let (large_allocs, large_sites) = allocations_during(&large);
    let large_storage_requests = STORAGE_REQUESTS.swap(0, Ordering::Relaxed);

    assert_eq!(
        small_allocs,
        large_allocs,
        "{kernel}: kernel-owned allocation count must not scale with workload \
         size (small={small_allocs}, large={large_allocs}). A difference means \
         a per-element or per-slice allocation has returned to this kernel.\n{}\n{}",
        format_attribution(kernel, "small", &small_sites),
        format_attribution(kernel, "large", &large_sites),
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
        || Box::new(coeus_ops::gather(&si, 1, &sx, &backend)),
        || Box::new(coeus_ops::gather(&li, 1, &lx, &backend)),
        1,
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
        || Box::new(coeus_ops::index_select(&si, 1, &sx, &backend)),
        || Box::new(coeus_ops::index_select(&li, 1, &lx, &backend)),
        1,
    );
}

#[test]
fn repeat_interleave_allocation_count_is_independent_of_output_size() {
    let backend = CountingBackend;
    let small = Tensor::from_slice_on(vec![4, 8, 4], &ramp(4 * 8 * 4), &backend);
    let large = Tensor::from_slice_on(vec![16, 32, 16], &ramp(16 * 32 * 16), &backend);

    assert_size_independent(
        "repeat_interleave",
        || Box::new(coeus_ops::repeat_interleave(&small, 2, 1, &backend)),
        || Box::new(coeus_ops::repeat_interleave(&large, 2, 1, &backend)),
        1,
    );
}

#[test]
fn scatter_add_allocation_count_is_independent_of_index_size() {
    let backend = CountingBackend;
    let build = |s: [usize; 3]| {
        let input = Tensor::from_slice_on(s.to_vec(), &ramp(s.iter().product()), &backend);
        let src_shape = [s[0], s[1] / 2, s[2]];
        let src_numel: usize = src_shape.iter().product();
        let src = Tensor::from_slice_on(src_shape.to_vec(), &ramp(src_numel), &backend);
        let index = Tensor::from_slice_on(src_shape.to_vec(), &indices(src_numel, s[1]), &backend);
        (input, index, src)
    };
    let (si, sx, ss) = build([4, 8, 4]);
    let (li, lx, ls) = build([16, 32, 16]);

    drop(coeus_ops::scatter_add(&si, 1, &sx, &ss, &backend));
    drop(coeus_ops::scatter_add(&li, 1, &lx, &ls, &backend));

    STORAGE_REQUESTS.store(0, Ordering::Relaxed);
    let (small_allocs, small_sites) =
        allocations_during(|| Box::new(coeus_ops::scatter_add(&si, 1, &sx, &ss, &backend)));
    let small_storage_requests = STORAGE_REQUESTS.swap(0, Ordering::Relaxed);
    let (large_allocs, large_sites) =
        allocations_during(|| Box::new(coeus_ops::scatter_add(&li, 1, &lx, &ls, &backend)));
    let large_storage_requests = STORAGE_REQUESTS.swap(0, Ordering::Relaxed);

    assert_eq!(
        small_allocs,
        large_allocs,
        "scatter_add: kernel-owned allocation count must not scale with workload size \
         (small={small_allocs}, large={large_allocs})\n{}\n{}",
        format_attribution("scatter_add", "small", &small_sites),
        format_attribution("scatter_add", "large", &large_sites),
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
    let small = tensor(&[4, 8, 4], &ramp(4 * 8 * 4));
    let large = tensor(&[16, 8, 16], &ramp(16 * 8 * 16));

    let count = |t: &Tensor<f64, SequentialBackend>| {
        allocations_during(|| Box::new(coeus_ops::topk(t, 2, 1, true)))
    };
    let (small_allocs, small_sites) = count(&small);
    let (large_allocs, large_sites) = count(&large);

    assert_eq!(
        small_allocs,
        large_allocs,
        "topk: allocation count must not scale with workload size \
         (small={small_allocs}, large={large_allocs}). A difference means a \
         per-element or per-slice allocation has returned to this kernel.\n{}\n{}",
        format_attribution("topk", "small", &small_sites),
        format_attribution("topk", "large", &large_sites),
    );
}

#[test]
fn attribution_log_roundtrips_size_and_captured_backtrace() {
    // Local log and counter: hermetic under parallel test threads, unlike
    // the globals the allocator itself records into.
    let log = Mutex::new(Vec::new());
    let seq = AtomicUsize::new(0);
    let layout = Layout::from_size_align(64, 8).expect("valid test layout");
    let recorded = record_attribution_into(&log, &seq, layout);
    assert_eq!(recorded, 0, "first sequence number starts at zero");
    let entries = drain_attribution_into(&log);
    assert_eq!(
        entries.len(),
        1,
        "one recorded allocation drains exactly once"
    );
    assert_eq!(entries[0].seq, 0);
    assert_eq!(entries[0].size, 64);
    assert!(
        !entries[0].ips.is_empty(),
        "capture records at least the capture frame"
    );
    assert!(
        drain_attribution_into(&log).is_empty(),
        "drain leaves the log empty"
    );
}

#[test]
fn attribution_formatter_names_kernel_side_and_size() {
    let log = Mutex::new(Vec::new());
    let seq = AtomicUsize::new(0);
    let layout = Layout::from_size_align(33, 8).expect("valid test layout");
    record_attribution_into(&log, &seq, layout);
    let entries = drain_attribution_into(&log);

    let rendered = format_attribution("scatter_add", "large", &entries);
    assert!(
        rendered.contains("scatter_add large: alloc #0 (33 bytes) at:"),
        "rendered entry must name kernel, side, seq, and size, got:\n{rendered}"
    );
    assert!(
        rendered.contains("0x"),
        "rendered entry must include raw frame addresses, got:\n{rendered}"
    );
    let empty = format_attribution("scatter_add", "small", &[]);
    assert!(
        empty.contains("COEUS_ALLOC_ATTRIBUTION=1"),
        "empty log must point at the capture flag, got:\n{empty}"
    );
}
