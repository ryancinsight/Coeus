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

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use coeus_core::SequentialBackend;
use coeus_tensor::Tensor;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

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
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
}

#[global_allocator]
static ALLOC: CountingAllocator = CountingAllocator;

/// Allocation count observed while running `body`.
fn allocations_during<R>(body: impl FnOnce() -> R) -> usize {
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    let result = body();
    let after = ALLOCATIONS.load(Ordering::Relaxed);
    drop(result);
    after - before
}

fn tensor(shape: &[usize], values: &[f64]) -> Tensor<f64, SequentialBackend> {
    Tensor::from_slice_on(shape.to_vec(), values, &SequentialBackend::default())
}

fn ramp(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 + 1.0).collect()
}

fn indices(n: usize, extent: usize) -> Vec<f64> {
    (0..n).map(|i| (i % extent) as f64).collect()
}

/// Assert a kernel's allocation count does not grow with workload size.
///
/// Each closure is run once before measuring so that any one-time lazy
/// initialisation inside the op is not attributed to the smaller workload.
fn assert_size_independent(
    kernel: &str,
    small: impl Fn() -> Box<dyn std::any::Any>,
    large: impl Fn() -> Box<dyn std::any::Any>,
) {
    drop(small());
    drop(large());

    let small_allocs = allocations_during(&small);
    let large_allocs = allocations_during(&large);

    assert_eq!(
        small_allocs, large_allocs,
        "{kernel}: allocation count must not scale with workload size \
         (small={small_allocs}, large={large_allocs}). A difference means a \
         per-element or per-slice allocation has returned to this kernel."
    );
}

#[test]
fn gather_allocation_count_is_independent_of_output_size() {
    let backend = SequentialBackend::default();
    let build = |s: [usize; 3]| {
        let input = tensor(&s, &ramp(s.iter().product()));
        let idx_shape = [s[0], s[1] / 2, s[2]];
        let idx_numel: usize = idx_shape.iter().product();
        let index = tensor(&idx_shape, &indices(idx_numel, s[1]));
        (input, index)
    };
    let (si, sx) = build([4, 8, 4]);
    let (li, lx) = build([16, 32, 16]);

    assert_size_independent(
        "gather",
        || Box::new(coeus_ops::gather(&si, 1, &sx, &backend)),
        || Box::new(coeus_ops::gather(&li, 1, &lx, &backend)),
    );
}

#[test]
fn index_select_allocation_count_is_independent_of_output_size() {
    let backend = SequentialBackend::default();
    let build = |s: [usize; 3]| {
        let input = tensor(&s, &ramp(s.iter().product()));
        let take = s[1] / 2;
        let index = tensor(&[take], &indices(take, s[1]));
        (input, index)
    };
    let (si, sx) = build([4, 8, 4]);
    let (li, lx) = build([16, 32, 16]);

    assert_size_independent(
        "index_select",
        || Box::new(coeus_ops::index_select(&si, 1, &sx, &backend)),
        || Box::new(coeus_ops::index_select(&li, 1, &lx, &backend)),
    );
}

#[test]
fn repeat_interleave_allocation_count_is_independent_of_output_size() {
    let backend = SequentialBackend::default();
    let small = tensor(&[4, 8, 4], &ramp(4 * 8 * 4));
    let large = tensor(&[16, 32, 16], &ramp(16 * 32 * 16));

    assert_size_independent(
        "repeat_interleave",
        || Box::new(coeus_ops::repeat_interleave(&small, 2, 1, &backend)),
        || Box::new(coeus_ops::repeat_interleave(&large, 2, 1, &backend)),
    );
}

#[test]
fn scatter_add_allocation_count_is_independent_of_index_size() {
    let backend = SequentialBackend::default();
    let build = |s: [usize; 3]| {
        let input = tensor(&s, &ramp(s.iter().product()));
        let src_shape = [s[0], s[1] / 2, s[2]];
        let src_numel: usize = src_shape.iter().product();
        let src = tensor(&src_shape, &ramp(src_numel));
        let index = tensor(&src_shape, &indices(src_numel, s[1]));
        (input, index, src)
    };
    let (si, sx, ss) = build([4, 8, 4]);
    let (li, lx, ls) = build([16, 32, 16]);

    assert_size_independent(
        "scatter_add",
        || Box::new(coeus_ops::scatter_add(&si, 1, &sx, &ss, &backend)),
        || Box::new(coeus_ops::scatter_add(&li, 1, &lx, &ls, &backend)),
    );
}

#[test]
fn topk_allocation_count_is_independent_of_slice_count() {
    // `k` and the reduced extent are held constant so only the number of outer
    // slices varies; the old per-slice buffers scaled with exactly that.
    let small = tensor(&[4, 8, 4], &ramp(4 * 8 * 4));
    let large = tensor(&[16, 8, 16], &ramp(16 * 8 * 16));

    assert_size_independent(
        "topk",
        || Box::new(coeus_ops::topk(&small, 2, 1, true)),
        || Box::new(coeus_ops::topk(&large, 2, 1, true)),
    );
}
