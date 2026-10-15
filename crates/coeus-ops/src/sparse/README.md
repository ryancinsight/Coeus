# Borrowed CSR forward adaptation

Approved provider decision: Leto ADR 0036. Consumer decision:
[ADR 0081](../../../../docs/adr/0081-borrowed-csr-forward-dispatch.md).

`ops/forward.rs` retains tensor validation, fallible storage/allocation and
backend selection. Leto owns checked signed borrowing, CSR address decoding,
the row recurrence and output writes. Both public functions keep
`Result<Tensor<T, B>, B::Error>` and the existing `T: coeus_core::Scalar` bound;
no slice-hook scalar constraint narrows integer or complex callers.

For row r and RHS column j, the required recurrence starts at `T::ZERO` and
applies `s += values[p] * rhs[columns[p], j]` in increasing physical p between
the validated row offsets. CSR's representation identity proves the exact
real-arithmetic product. The explicit scalar policy additionally preserves
separate native-precision multiplication and addition: sorting, coalescing,
zero dropping, FMA and wider accumulators cannot replace that recurrence.

Validation precedence stays CSR structure/storage, then dense rank/shape,
readability and contiguous-layout checks, then the original fallible allocation
and mutable access. Logical slices retain nonzero offsets. Metadata/storage
checks precede one checked provider construction over the original `i64`
arrays. Its representable monotone offsets with endpoints 0 and nnz imply the
same bounded-offset domain as the old validator. On rejection only, the old
index validator preserves exact diagnostic categories, messages and priority;
an additional provider rejection propagates through the shared error mapper.
No unchecked construction or conversion buffer skips an invariant. Negative indices,
non-representable columns/offsets and count overflow retain existing typed
errors and their width-dependent distinctions. Noncontiguous inputs retain
their current rejection instead of being materialized.

CPU mutable access initializes storage before creating `&mut [T]`. The provider
binds that exclusive slice after checking `m*n`, `k*n` and exact lengths.
For `n>0`, row r owns `[r*n,(r+1)*n)`; different rows are disjoint. Its worker
starts each row at `T::ZERO`, including empty rows and zero shared dimension.
A one-column row keeps its accumulator local and writes once; wider rows use
the initialized destination directly. Both use the same provider traversal
and arithmetic policy. Width zero forms no element reference or dense read.
No successful call exposes a partial or uninitialized output.

The unsafe dispatch scope erases lifetimes solely to fit Backend's existing
`'static` callback. ADR 0081 obligates every implementation to join all admitted
calls before both return and unwind; panic permits a subset, at most once per
row. The closure retains no task beyond that scope. The adapter selects neither
a provider runtime nor an alternative parallel threshold. Backward operations,
accelerator kernels, owned CSR storage and the original preservation draft stay
outside this forward cutover.

The verification record maps the exact branch/base pair to native, property,
borrow-model, architecture, allocation, API and paired performance evidence.
