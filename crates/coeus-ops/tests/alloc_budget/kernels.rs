//! Per-kernel allocation-count budgets.

use super::*;

#[test]
fn gather_allocation_count_is_independent_of_output_size() {
    let backend = CountingBackend;
    let build = |s: [usize; 3]| {
        let input = Tensor::from_slice_on(s.to_vec(), &ramp(s.iter().product()), &backend)
            .expect("invariant: test backend operation succeeds");
        let idx_shape = [s[0], s[1] / 2, s[2]];
        let idx_numel: usize = idx_shape.iter().product();
        let index = Tensor::from_slice_on(idx_shape.to_vec(), &indices(idx_numel, s[1]), &backend)
            .expect("invariant: test backend operation succeeds");
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
        1,
    );
}

#[test]
fn index_select_allocation_count_is_independent_of_output_size() {
    let backend = CountingBackend;
    let build = |s: [usize; 3]| {
        let input = Tensor::from_slice_on(s.to_vec(), &ramp(s.iter().product()), &backend)
            .expect("invariant: test backend operation succeeds");
        let take = s[1] / 2;
        let index = Tensor::from_slice_on(vec![take], &indices(take, s[1]), &backend)
            .expect("invariant: test backend operation succeeds");
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
        1,
    );
}

#[test]
fn repeat_interleave_allocation_count_is_independent_of_output_size() {
    let backend = CountingBackend;
    let small = Tensor::from_slice_on(vec![4, 8, 4], &ramp(4 * 8 * 4), &backend)
        .expect("invariant: test backend operation succeeds");
    let large = Tensor::from_slice_on(vec![16, 32, 16], &ramp(16 * 32 * 16), &backend)
        .expect("invariant: test backend operation succeeds");

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
        1,
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
    // (`assert_size_independent` takes the oracle lock itself; holding it here
    // as well would self-deadlock on the non-reentrant mutex.)
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
        0,
    );
}
