use coeus_core::{ComputeBackend, Float, Layout, Scalar};
use coeus_ops::{
    BinaryOp, ElementwiseOps, MatmulOps, ReductionOp, ReductionOps, ScalarPowerOps, UnaryOp,
};
use coeus_tensor::Tensor;

pub(crate) trait OutputWrite<T: Scalar, B: ComputeBackend>: core::fmt::Debug {
    const COLUMNS: usize;
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
    let columns = O::COLUMNS;

    for offset_view in [false, true] {
        let shape = if offset_view {
            [4, columns + 2]
        } else {
            [2, columns]
        };
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
                let index = if offset_view {
                    (row + 1) * shape[1] + column + 1
                } else {
                    row * columns + column
                };
                expected_storage[index] = expected[row * columns + column];
            }
        }
        let (output, output_layout) = destination.storage_and_layout_mut();
        operation
            .dispatch(backend, &input, input.layout(), &rhs, output, output_layout)
            .expect("valid output operation");
        assert_eq!(
            original.to_vec_on(backend),
            original_values,
            "{operation:?} changed its destination clone ({offset_view}, {})",
            core::any::type_name::<T>()
        );
        assert_eq!(
            destination.to_vec_on(backend),
            expected,
            "{operation:?} computed incorrect logical output"
        );
        let mut actual_storage = vec![T::zero(); expected_storage.len()];
        backend.copy_to_host(destination.storage(), &mut actual_storage);
        assert_eq!(
            actual_storage, expected_storage,
            "{operation:?} changed elements outside its output view"
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
    let columns = O::COLUMNS;
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
    assert_eq!(
        original.to_vec_on(backend),
        original_values,
        "{operation:?} changed shared values on rejection"
    );
    assert_eq!(
        destination.to_vec_on(backend),
        original_values,
        "{operation:?} wrote output before rejecting an invalid input"
    );
}

#[derive(Debug)]
pub(crate) struct Negate;

impl<T: Scalar + core::ops::Neg<Output = T>, B: ElementwiseOps<T>> OutputWrite<T, B> for Negate {
    const COLUMNS: usize = 2;
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

#[derive(Debug)]
pub(crate) struct Add;

impl<T: Scalar, B: ElementwiseOps<T>> OutputWrite<T, B> for Add {
    const COLUMNS: usize = 2;
    fn expected(&self, one: T) -> Vec<T> {
        let two = one + one;
        let three = two + one;
        let four = two + two;
        vec![three, three, four, four + two]
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
        backend.elementwise_binary(
            BinaryOp::Add,
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
pub(crate) struct Sum;

impl<T: Scalar, B: ReductionOps<T>> OutputWrite<T, B> for Sum {
    const COLUMNS: usize = 1;
    fn expected(&self, one: T) -> Vec<T> {
        let two = one + one;
        let three = two + one;
        let four = two + two;
        vec![three, three + four]
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
        backend.reduce(
            ReductionOp::Sum,
            input.storage(),
            input_layout,
            1,
            output,
            output_layout,
        )
    }
}

#[derive(Debug)]
pub(crate) struct Product;

impl<T: Scalar, B: MatmulOps<T>> OutputWrite<T, B> for Product {
    const COLUMNS: usize = 2;
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
    const COLUMNS: usize = 2;
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
    const COLUMNS: usize = 2;
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
