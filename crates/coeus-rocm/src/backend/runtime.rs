use super::provider::RocmProvider;
use coeus_core::{ComputeBackend, Layout, Scalar};
use coeus_hephaestus::{
    ConvolutionProvider, ElementwiseProvider, HephaestusBackend, HephaestusBackendError,
    HephaestusProvider, HephaestusStorage, ReductionProvider,
};
use coeus_ops::{
    BinaryOp, ConvOps, ConvolutionBackward, ConvolutionForward, ElementwiseOps, ReductionOp,
    ReductionOps, UnaryOp,
};
use hephaestus_core::{CommandStream, KernelDevice};

/// Coeus ROCm backend with native Hephaestus storage and rank-2 reductions.
#[derive(Debug, Clone, Copy, Default)]
pub struct RocmBackend(HephaestusBackend<RocmProvider>);

impl coeus_core::backend::private::Sealed for RocmBackend {}

impl RocmBackend {
    /// Construct the ROCm backend selector.
    #[must_use]
    pub const fn new() -> Self {
        Self(HephaestusBackend::new())
    }
}

impl ComputeBackend for RocmBackend {
    type Error = HephaestusBackendError;
    type DeviceBuffer<T: Scalar> = HephaestusStorage<RocmProvider, T>;
    type KernelDescriptor = ();
    type DispatchFuture<T: Scalar> = std::future::Ready<T>;

    fn name(&self) -> &'static str {
        self.0.name()
    }

    fn num_threads(&self) -> usize {
        self.0.num_threads()
    }

    fn allocate<T: Scalar>(&self, len: usize) -> Self::DeviceBuffer<T> {
        self.0.allocate(len)
    }

    fn allocate_zeroed<T: Scalar>(&self, len: usize) -> Self::DeviceBuffer<T> {
        self.0.allocate_zeroed(len)
    }

    fn fill<T: Scalar>(&self, dst: &mut Self::DeviceBuffer<T>, val: T) {
        if val.has_zero_bit_pattern() {
            self.fill_zero(dst);
            return;
        }
        self.0.fill(dst, val)
    }

    fn fill_zero<T: Scalar>(&self, dst: &mut Self::DeviceBuffer<T>) {
        let mut stream = RocmProvider::device()
            .stream()
            .expect("ROCm zero fill stream creation failed");
        stream
            .fill_zero(dst.buffer())
            .expect("ROCm zero fill encoding failed");
        stream.submit().expect("ROCm zero fill submission failed");
    }

    fn copy_to_device<T: Scalar>(&self, src: &[T], dst: &mut Self::DeviceBuffer<T>) {
        self.0.copy_to_device(src, dst)
    }

    fn copy_to_host<T: Scalar>(&self, src: &Self::DeviceBuffer<T>, dst: &mut [T]) {
        self.0.copy_to_host(src, dst)
    }
}

impl<T> ConvOps<T> for RocmBackend
where
    T: Scalar + leto_ops::Scalar,
    RocmProvider: ConvolutionProvider<T>,
{
    fn convolution_forward<const R: usize, const D: usize>(
        &self,
        request: ConvolutionForward<'_, Self, T>,
        stride: [usize; D],
        padding: [usize; D],
        dilation: [usize; D],
    ) -> Result<(), Self::Error> {
        self.0.convolution_forward::<R, D>(
            ConvolutionForward {
                input: request.input,
                input_layout: request.input_layout,
                weight: request.weight,
                weight_layout: request.weight_layout,
                bias: request.bias,
                output: request.output,
                output_layout: request.output_layout,
            },
            stride,
            padding,
            dilation,
        )
    }

    fn convolution_backward<const R: usize, const D: usize>(
        &self,
        request: ConvolutionBackward<'_, Self, T>,
        stride: [usize; D],
        padding: [usize; D],
        dilation: [usize; D],
    ) -> Result<(), Self::Error> {
        self.0.convolution_backward::<R, D>(
            ConvolutionBackward {
                grad_output: request.grad_output,
                grad_output_layout: request.grad_output_layout,
                input: request.input,
                input_layout: request.input_layout,
                weight: request.weight,
                weight_layout: request.weight_layout,
                grad_input: request.grad_input,
                grad_input_layout: request.grad_input_layout,
                grad_weight: request.grad_weight,
                grad_weight_layout: request.grad_weight_layout,
                grad_bias: request.grad_bias,
            },
            stride,
            padding,
            dilation,
        )
    }

    fn convolution_transposed_forward<const R: usize, const D: usize>(
        &self,
        request: ConvolutionForward<'_, Self, T>,
        stride: [usize; D],
        padding: [usize; D],
        output_padding: [usize; D],
        dilation: [usize; D],
    ) -> Result<(), Self::Error>
    where
        T: coeus_core::Float,
    {
        self.0.convolution_transposed_forward::<R, D>(
            ConvolutionForward {
                input: request.input,
                input_layout: request.input_layout,
                weight: request.weight,
                weight_layout: request.weight_layout,
                bias: request.bias,
                output: request.output,
                output_layout: request.output_layout,
            },
            stride,
            padding,
            output_padding,
            dilation,
        )
    }

    fn convolution_transposed_backward<const R: usize, const D: usize>(
        &self,
        request: ConvolutionBackward<'_, Self, T>,
        stride: [usize; D],
        padding: [usize; D],
        output_padding: [usize; D],
        dilation: [usize; D],
    ) -> Result<(), Self::Error>
    where
        T: coeus_core::Float,
    {
        self.0.convolution_transposed_backward::<R, D>(
            ConvolutionBackward {
                grad_output: request.grad_output,
                grad_output_layout: request.grad_output_layout,
                input: request.input,
                input_layout: request.input_layout,
                weight: request.weight,
                weight_layout: request.weight_layout,
                grad_input: request.grad_input,
                grad_input_layout: request.grad_input_layout,
                grad_weight: request.grad_weight,
                grad_weight_layout: request.grad_weight_layout,
                grad_bias: request.grad_bias,
            },
            stride,
            padding,
            output_padding,
            dilation,
        )
    }
}

impl<T> ReductionOps<T> for RocmBackend
where
    T: Scalar + leto_ops::Scalar,
    RocmProvider: ReductionProvider<T>,
{
    fn reduce(
        &self,
        op: ReductionOp,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error> {
        self.0.reduce(op, a, a_layout, axis, c, c_layout)
    }

    fn cumsum(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error> {
        self.0.cumsum(a, a_layout, axis, c, c_layout)
    }

    fn suffix_sum(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error> {
        self.0.suffix_sum(a, a_layout, axis, c, c_layout)
    }

    fn cumprod(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error> {
        self.0.cumprod(a, a_layout, axis, c, c_layout)
    }

    fn suffix_prod(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error> {
        self.0.suffix_prod(a, a_layout, axis, c, c_layout)
    }
}

impl<T> ElementwiseOps<T> for RocmBackend
where
    T: Scalar + leto_ops::Scalar,
    RocmProvider: ElementwiseProvider<T>,
{
    fn elementwise_binary(
        &self,
        operation: BinaryOp,
        lhs: &Self::DeviceBuffer<T>,
        lhs_layout: &Layout,
        rhs: &Self::DeviceBuffer<T>,
        rhs_layout: &Layout,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        self.0.elementwise_binary(
            operation,
            lhs,
            lhs_layout,
            rhs,
            rhs_layout,
            output,
            output_layout,
        )
    }

    fn elementwise_unary(
        &self,
        operation: UnaryOp,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        self.0
            .elementwise_unary(operation, input, input_layout, output, output_layout)
    }
}
