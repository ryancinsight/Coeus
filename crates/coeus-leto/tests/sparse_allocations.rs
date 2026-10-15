//! The borrowed dispatch boundary must not allocate CSR copies.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use coeus_leto::{spmm_into, spmv_into, CsrDispatch};
use tracing_subscriber::fmt::format::FmtSpan;

struct CountAllocations;

thread_local! {
    static TRACKING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn record_allocation() {
    if TRACKING.try_with(Cell::get).unwrap_or(false) {
        let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
    }
}

// SAFETY: every request delegates unchanged to System; thread-local counters
// neither allocate nor access the requested allocation.
unsafe impl GlobalAlloc for CountAllocations {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        // SAFETY: the caller supplies GlobalAlloc's valid allocation layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        // SAFETY: the caller supplies GlobalAlloc's valid allocation layout.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: allocation and layout originate from System above.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record_allocation();
        // SAFETY: pointer, original layout and new size follow GlobalAlloc's contract.
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountAllocations = CountAllocations;

fn measured<R>(operation: impl FnOnce() -> R) -> (R, usize) {
    struct StopTracking;
    impl Drop for StopTracking {
        fn drop(&mut self) {
            TRACKING.with(|tracking| tracking.set(false));
        }
    }
    ALLOCATIONS.with(|count| count.set(0));
    TRACKING.with(|tracking| tracking.set(true));
    let guard = StopTracking;
    let result = operation();
    drop(guard);
    (result, ALLOCATIONS.with(Cell::get))
}

#[test]
fn dispatch_borrows_all_csr_arrays() {
    let subscriber = tracing_subscriber::fmt()
        .without_time()
        .with_test_writer()
        .with_span_events(FmtSpan::FULL)
        .finish();
    let _subscriber = tracing::subscriber::set_default(subscriber);
    let case = tracing::info_span!("borrowed_sparse_dispatch");
    let _case = case.enter();
    let (values, columns, offsets, x, mut y, mut c) = {
        let _phase = tracing::info_span!("arrange").entered();
        (
            [2.0_f64, 3.0, -1.0],
            [0, 2, 1],
            [0, 2, 2, 3],
            [1.0, 2.0, 4.0],
            [f64::NAN; 3],
            [f64::NAN; 6],
        )
    };
    let descriptor = CsrDispatch::new(&values, &columns, &offsets, 3, 3);
    let ((vector_result, vector_allocations), (matrix_result, matrix_allocations)) = {
        let _phase = tracing::info_span!("act").entered();
        (
            measured(|| spmv_into(descriptor, &x, &mut y)),
            measured(|| spmm_into(descriptor, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, &mut c)),
        )
    };
    let _phase = tracing::info_span!("assert").entered();
    vector_result.expect("valid CSR vector product");
    matrix_result.expect("valid CSR matrix product");
    assert_eq!(y, [14.0, 0.0, -2.0]);
    assert_eq!(c, [17.0, 22.0, 0.0, 0.0, -3.0, -4.0]);
    assert_eq!(
        vector_allocations, 0,
        "SpMV must borrow all three CSR arrays"
    );
    assert_eq!(
        matrix_allocations, 0,
        "SpMM must borrow all three CSR arrays"
    );
}
