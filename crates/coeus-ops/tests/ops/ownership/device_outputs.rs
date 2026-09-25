use coeus_core::{ComputeBackend, Float, Layout, Scalar};
use coeus_ops::{
    BinaryOp, ElementwiseOps, MatmulOps, ReductionOp, ReductionOps, ScalarPowerOps, UnaryOp,
};
use coeus_tensor::Tensor;

pub(crate) trait OutputWrite<T: Scalar, B: ComputeBackend>: core::fmt::Debug {
    /// The operation's output column count. An instance method (not an
    /// associated const) so one enum can carry several operations whose
    /// column counts differ -- see `NumericOp` below, whose `Add`/`Sum`/
    /// `Product` variants would otherwise be unable to share one
    /// `OutputWrite` impl and one `preserves_output_clones`/
    /// `rejects_invalid_output_write` instantiation per `(T, B)`.
    fn columns(&self) -> usize;
    fn input(one: T) -> [T; 4] {
        let two = one + one;
        [one, two, two + one, two + two]
    }
    fn expected(&self, one: T) -> Vec<T>;
    fn dispatch(
        &self,
        backend: &B,
        input: &Tensor<T, B>,
        input_layout: &Layout,
        rhs: &Tensor<T, B>,
        output: &mut B::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), B::Error>;
}

// `preserves_output_clones`/`rejects_invalid_output_write` are instantiated
// once per (scalar type, backend, `OutputWrite` impl) -- required
// generic-instantiation coverage, not a redundancy to collapse by itself.
// The *number of `OutputWrite` impls* is a real lever, though: `Negate`,
// `Add`, `Sum`, `Product`, and `Square` were five separate impls, each
// forcing its own instantiation set even where the same `(T, B)` pairs
// were exercised by more than one of them. `Add`, `Sum`, and `Product` need
// no per-type bound beyond `Scalar` (`Negate` additionally needs
// `Neg<Output = T>`, restricting it to signed/float types; `Square` needs
// `Float`), so they fold into one `NumericOp` enum -- the same
// runtime-dispatch pattern `Scan` below already uses -- cutting their
// combined instantiation count roughly 3x for every `(T, B)` pair actually
// shared across the three (measured 2026-09-25: see the module-level
// harness for the before/after numbers). `Negate` and `Square` keep their
// own impls: merging either with `NumericOp` would force every `NumericOp`
// caller's `T` to additionally satisfy `Neg<Output = T>` or `Float`,
// breaking the unsigned-integer calls that exercise `Add`/`Sum`/`Product`
// today.
//
// None of the arithmetic below the `for offset_view` loop depends on the
// scalar type, backend, or operation, and every `assert_eq!` invocation
// independently re-expanded its `Debug`-formatting and panic path per
// instantiation even though the comparison logic is identical across every
// operation for a given `T`. Both are hoisted into plain functions --
// `usize`/`bool`-only for the index arithmetic, `T`-only (never `B` or the
// operation type) for the assertions -- mirroring the
// `optimizer::assert_values` pattern used elsewhere in this harness.

/// The destination-view shape for a given operation's column count and
/// offset mode. Depends only on `columns` and `offset_view` (both plain
/// runtime values, not generic parameters), so this compiles once for the
/// whole harness rather than once per instantiation.
fn view_shape(columns: usize, offset_view: bool) -> [usize; 2] {
    if offset_view {
        [4, columns + 2]
    } else {
        [2, columns]
    }
}

/// The flat index into the destination/expected buffers for output
/// position `(row, column)` under the given shape and offset mode.
fn destination_flat_index(
    row: usize,
    column: usize,
    columns: usize,
    shape1: usize,
    offset_view: bool,
) -> usize {
    if offset_view {
        (row + 1) * shape1 + column + 1
    } else {
        row * columns + column
    }
}

/// Compares two value vectors for exact equality, panicking with `context`
/// on mismatch. Generic over `T` alone (never `B` or the operation type),
/// so rustc monomorphizes this once per scalar type instead of once per
/// instantiation.
fn assert_vec_eq<T: Scalar>(actual: Vec<T>, expected: &[T], context: String) {
    assert_eq!(actual, expected, "{context}");
}

pub(crate) fn preserves_output_clones<T, B, O>(backend: &B, operation: O, one: T)
where
    T: Scalar,
    B: ComputeBackend,
    O: OutputWrite<T, B>,
{
    let two = one + one;
    let three = two + one;
    let four = two + two;
    let input_values = O::input(one);
    let rhs_values = [two, one, one, two];
    let input = Tensor::from_slice_on([2, 2], &input_values, backend);
    let rhs = Tensor::from_slice_on([2, 2], &rhs_values, backend);
    let expected = operation.expected(one);
    let columns = operation.columns();

    for offset_view in [false, true] {
        let shape = view_shape(columns, offset_view);
        let original_values: Vec<_> = (0..shape[0] * shape[1])
            .map(|index| [four, three, two, one][index % 4])
            .collect();
        let original = Tensor::from_slice_on(shape, &original_values, backend);
        let mut destination = if offset_view {
            original.slice(&[(1, 3), (1, columns + 1)])
        } else {
            original.clone()
        };
        let mut expected_storage = original_values.clone();
        for row in 0..2 {
            for column in 0..columns {
                let index = destination_flat_index(row, column, columns, shape[1], offset_view);
                expected_storage[index] = expected[row * columns + column];
            }
        }
        let (output, output_layout) = destination.storage_and_layout_mut();
        operation
            .dispatch(backend, &input, input.layout(), &rhs, output, output_layout)
            .expect("valid output operation");
        assert_vec_eq(
            original.to_vec_on(backend),
            &original_values,
            format!(
                "{operation:?} changed its destination clone ({offset_view}, {})",
                core::any::type_name::<T>()
            ),
        );
        assert_vec_eq(
            destination.to_vec_on(backend),
            &expected,
            format!("{operation:?} computed incorrect logical output"),
        );
        let mut actual_storage = vec![T::zero(); expected_storage.len()];
        backend.copy_to_host(destination.storage(), &mut actual_storage);
        assert_vec_eq(
            actual_storage,
            &expected_storage,
            format!("{operation:?} changed elements outside its output view"),
        );
    }
}

pub(crate) fn rejects_invalid_output_write<T, B, O>(backend: &B, operation: O, one: T)
where
    T: Scalar,
    B: ComputeBackend,
    O: OutputWrite<T, B>,
{
    let two = one + one;
    let input = Tensor::from_slice_on([2, 2], &[one, two, two, one], backend);
    let invalid_input_layout = Layout::new([2, 3].into());
    let columns = operation.columns();
    let original_values: Vec<_> = (0..2 * columns)
        .map(|index| [two, one][index % 2])
        .collect();
    let original = Tensor::from_slice_on([2, columns], &original_values, backend);
    let mut destination = original.clone();
    let (output, output_layout) = destination.storage_and_layout_mut();
    operation
        .dispatch(
            backend,
            &input,
            &invalid_input_layout,
            &input,
            output,
            output_layout,
        )
        .expect_err("an input layout extending beyond its allocation must fail");
    assert_vec_eq(
        original.to_vec_on(backend),
        &original_values,
        format!("{operation:?} changed shared values on rejection"),
    );
    assert_vec_eq(
        destination.to_vec_on(backend),
        &original_values,
        format!("{operation:?} wrote output before rejecting an invalid input"),
    );
}

#[derive(Debug)]
pub(crate) struct Negate;

impl<T: Scalar + core::ops::Neg<Output = T>, B: ElementwiseOps<T>> OutputWrite<T, B> for Negate {
    fn columns(&self) -> usize {
        2
    }
    fn expected(&self, one: T) -> Vec<T> {
        let two = one + one;
        let three = two + one;
        let four = two + two;
        vec![-one, -two, -three, -four]
    }
    fn dispatch(
        &self,
        backend: &B,
        input: &Tensor<T, B>,
        input_layout: &Layout,
        _rhs: &Tensor<T, B>,
        output: &mut B::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), B::Error> {
        backend.elementwise_unary(
            UnaryOp::Neg,
            input.storage(),
            input_layout,
            output,
            output_layout,
        )
    }
}

/// The `Add`/`Sum` differential-output-write operations, unified into one
/// `OutputWrite` impl (see the module-level rationale above). Both need
/// only `T: Scalar` and their own backend sub-trait -- `B`'s bound below is
/// the union of the two, which every backend exercised against either
/// operation already satisfies (`SequentialBackend`, `MoiraiBackend`,
/// `WgpuBackend`, `CudaBackend`, and `HephaestusBackend<WgpuBackend>` /
/// `HephaestusBackend<CudaBackend>` all implement both `ElementwiseOps<T>`
/// and `ReductionOps<T>`).
///
/// `Product` (`MatmulOps<T>`) deliberately stays its own impl rather than
/// joining this enum: `HephaestusBackend<WgpuBackend>` and
/// `HephaestusBackend<CudaBackend>` are exercised against `Add`/`Sum` in
/// `coeus-wgpu`'s and `coeus-cuda`'s test suites (both `#[path]`-include
/// this file) but do not implement `MatmulOps<T>` -- folding `Product` in
/// would add that bound to every `NumericOp` call and break those two
/// crates' builds. This was caught by CI (`E0277`: `HephaestusBackend<
/// WgpuBackend>: MatmulOps<_>` not satisfied) after an initial merge that
/// only checked `coeus-ops`'s own two backends, which do implement all
/// three traits; the coverage gap is specific to `HephaestusBackend`'s
/// narrower trait surface in the crates that reuse this file.
#[derive(Clone, Copy, Debug)]
pub(crate) enum NumericOp {
    Add,
    Sum,
}

impl<T: Scalar, B: ElementwiseOps<T> + ReductionOps<T>> OutputWrite<T, B> for NumericOp {
    fn columns(&self) -> usize {
        match self {
            Self::Add => 2,
            Self::Sum => 1,
        }
    }
    fn expected(&self, one: T) -> Vec<T> {
        let two = one + one;
        let three = two + one;
        let four = two + two;
        match self {
            Self::Add => vec![three, three, four, four + two],
            Self::Sum => vec![three, three + four],
        }
    }
    fn dispatch(
        &self,
        backend: &B,
        input: &Tensor<T, B>,
        input_layout: &Layout,
        rhs: &Tensor<T, B>,
        output: &mut B::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), B::Error> {
        match self {
            Self::Add => backend.elementwise_binary(
                BinaryOp::Add,
                input.storage(),
                input_layout,
                rhs.storage(),
                rhs.layout(),
                output,
                output_layout,
            ),
            Self::Sum => backend.reduce(
                ReductionOp::Sum,
                input.storage(),
                input_layout,
                1,
                output,
                output_layout,
            ),
        }
    }
}

#[derive(Debug)]
pub(crate) struct Product;

impl<T: Scalar, B: MatmulOps<T>> OutputWrite<T, B> for Product {
    fn columns(&self) -> usize {
        2
    }
    fn expected(&self, one: T) -> Vec<T> {
        let two = one + one;
        let three = two + one;
        let four = two + two;
        vec![four, one + four, four + four + two, four + four + three]
    }
    fn dispatch(
        &self,
        backend: &B,
        input: &Tensor<T, B>,
        input_layout: &Layout,
        rhs: &Tensor<T, B>,
        output: &mut B::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), B::Error> {
        backend.matmul(
            input.storage(),
            input_layout,
            rhs.storage(),
            rhs.layout(),
            output,
            output_layout,
        )
    }
}

#[derive(Debug)]
pub(crate) struct Square;

impl<T: Float, B: ScalarPowerOps<T>> OutputWrite<T, B> for Square {
    fn columns(&self) -> usize {
        2
    }
    fn input(one: T) -> [T; 4] {
        let two = one + one;
        let four = two + two;
        [one, two, four, four + four]
    }
    fn expected(&self, one: T) -> Vec<T> {
        let two = one + one;
        let four = two + two;
        let sixteen = four * four;
        // Powers of two exercise exact integer exponents without rounding ambiguity.
        vec![one, four, sixteen, sixteen * four]
    }
    fn dispatch(
        &self,
        backend: &B,
        input: &Tensor<T, B>,
        input_layout: &Layout,
        _rhs: &Tensor<T, B>,
        output: &mut B::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), B::Error> {
        backend.elementwise_pow_scalar(
            input.storage(),
            input_layout,
            T::one() + T::one(),
            output,
            output_layout,
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Scan {
    CumulativeSum,
    SuffixSum,
    CumulativeProduct,
    SuffixProduct,
}

impl<T: Scalar + leto_ops::Scalar, B: ReductionOps<T>> OutputWrite<T, B> for Scan {
    fn columns(&self) -> usize {
        2
    }
    fn expected(&self, one: T) -> Vec<T> {
        let two = one + one;
        let three = two + one;
        let four = two + two;
        // Inclusive row scans of [[1,2],[3,4]]: sums 3/7, products 2/12.
        match self {
            Self::CumulativeSum => vec![one, three, three, three + four],
            Self::SuffixSum => vec![three, two, three + four, four],
            Self::CumulativeProduct => vec![one, two, three, three * four],
            Self::SuffixProduct => vec![two, two, three * four, four],
        }
    }
    fn dispatch(
        &self,
        backend: &B,
        input: &Tensor<T, B>,
        input_layout: &Layout,
        _rhs: &Tensor<T, B>,
        output: &mut B::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), B::Error> {
        match self {
            Self::CumulativeSum => {
                backend.cumsum(input.storage(), input_layout, 1, output, output_layout)
            }
            Self::SuffixSum => {
                backend.suffix_sum(input.storage(), input_layout, 1, output, output_layout)
            }
            Self::CumulativeProduct => {
                backend.cumprod(input.storage(), input_layout, 1, output, output_layout)
            }
            Self::SuffixProduct => {
                backend.suffix_prod(input.storage(), input_layout, 1, output, output_layout)
            }
        }
    }
}

pub(crate) fn scans_preserve_output_clones<T: Scalar + leto_ops::Scalar, B: ReductionOps<T>>(
    backend: &B,
    one: T,
) {
    for scan in [
        Scan::CumulativeSum,
        Scan::SuffixSum,
        Scan::CumulativeProduct,
        Scan::SuffixProduct,
    ] {
        preserves_output_clones(backend, scan, one);
        rejects_invalid_output_write(backend, scan, one);
    }
}
