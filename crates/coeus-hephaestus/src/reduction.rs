use crate::{
    error::HephaestusBackendError,
    layout::{ranked, ranked_axis},
    storage::HephaestusStorage,
};
use coeus_core::{ComputeBackend, Layout, Scalar};
use coeus_ops::ReductionOp;
use hephaestus_core::{
    AxisReductionOps, CombineExpr, ComputeDevice, DeviceBuffer, IdentityToken, MaxOp, MinOp,
    OpIdentity, ScanDirection, ScanOps, StridedView, SumOp,
};
use leto::Layout as LetoLayout;
use std::future::Ready;

/// Common provider identity and device acquisition seam.
///
/// # Safety
///
/// An implementation must ensure that every typed buffer exposed by its
/// [`hephaestus_core::ComputeDevice`] remains safe to retain behind an
/// `Arc` and to move between Coeus worker threads. Provider kernels must also
/// synchronize access according to their device API's contract.
pub unsafe trait HephaestusProvider: Send + Sync + Clone + Copy + Default + 'static {
    /// Concrete Hephaestus device type selected by this provider.
    type Device: ComputeDevice + Send + Sync + 'static;

    /// Stable backend name used by Coeus diagnostics.
    const NAME: &'static str;

    /// Return the lazily acquired device owned by this provider.
    fn device() -> &'static Self::Device;

    /// Try to acquire the provider device without panicking.
    ///
    /// # Errors
    ///
    /// Returns the provider's typed acquisition failure.
    fn try_device() -> hephaestus_core::Result<&'static Self::Device>;
}

/// Cumulative operation selected by a provider scan dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanOperation {
    /// Inclusive cumulative addition.
    Sum,
    /// Inclusive cumulative multiplication.
    Product,
}

/// Dispatches Coeus reduction requests through one Hephaestus axis seam.
pub trait AxisReductionDispatch<D: ComputeDevice, T: bytemuck::Pod + eunomia::Pod> {
    /// Execute one Coeus reduction operation.
    fn reduce(
        device: &D,
        operation: ReductionOp,
        input: RankedOperand<'_, D::Buffer<T>, 2>,
        axis: usize,
        output: RankedOperand<'_, D::Buffer<T>, 2>,
    ) -> hephaestus_core::Result<()>;
}

impl<D, T, R> AxisReductionDispatch<D, T> for R
where
    D: ComputeDevice,
    T: bytemuck::Pod
        + eunomia::Pod
        + hephaestus_core::DialectScalar<R::Dialect>
        + OpIdentity<SumOp>
        + OpIdentity<hephaestus_core::ProdOp>
        + OpIdentity<MinOp>
        + OpIdentity<MaxOp>
        + IdentityToken<SumOp, R::Dialect>
        + IdentityToken<hephaestus_core::ProdOp, R::Dialect>
        + IdentityToken<MinOp, R::Dialect>
        + IdentityToken<MaxOp, R::Dialect>,
    R: AxisReductionOps<D, T> + Default,
    SumOp: CombineExpr<R::Dialect>,
    hephaestus_core::ProdOp: CombineExpr<R::Dialect>,
    MinOp: CombineExpr<R::Dialect>,
    MaxOp: CombineExpr<R::Dialect>,
{
    fn reduce(
        device: &D,
        operation: ReductionOp,
        input: RankedOperand<'_, D::Buffer<T>, 2>,
        axis: usize,
        output: RankedOperand<'_, D::Buffer<T>, 2>,
    ) -> hephaestus_core::Result<()> {
        let operations = R::default();
        let input = StridedView::new(input.buffer, input.layout);
        let output = StridedView::new(output.buffer, output.layout);
        match operation {
            ReductionOp::Sum => operations.reduce_axis_into::<SumOp>(device, input, axis, output),
            ReductionOp::Prod => {
                operations.reduce_axis_into::<hephaestus_core::ProdOp>(device, input, axis, output)
            }
            ReductionOp::Mean => operations.mean_axis_into(device, input, axis, output),
            ReductionOp::Min => operations.min_axis_into(device, input, axis, output),
            ReductionOp::Max => operations.max_axis_into(device, input, axis, output),
        }
    }
}

/// Dispatches Coeus scans through one Hephaestus scan seam.
pub trait ScanDispatch<D: ComputeDevice, T: bytemuck::Pod + eunomia::Pod> {
    /// Execute one Coeus scan operation.
    fn scan(
        device: &D,
        input: RankedOperand<'_, D::Buffer<T>, 2>,
        axis: usize,
        operation: ScanOperation,
        direction: ScanDirection,
        output: RankedOperand<'_, D::Buffer<T>, 2>,
    ) -> hephaestus_core::Result<()>;
}

impl<D, T, S> ScanDispatch<D, T> for S
where
    D: ComputeDevice,
    T: bytemuck::Pod
        + eunomia::Pod
        + hephaestus_core::DialectScalar<S::Dialect>
        + OpIdentity<hephaestus_core::CumSumOp>
        + OpIdentity<hephaestus_core::CumProdOp>
        + IdentityToken<hephaestus_core::CumSumOp, S::Dialect>
        + IdentityToken<hephaestus_core::CumProdOp, S::Dialect>,
    S: ScanOps<D, T> + Default,
    hephaestus_core::CumSumOp: CombineExpr<S::Dialect>,
    hephaestus_core::CumProdOp: CombineExpr<S::Dialect>,
{
    fn scan(
        device: &D,
        input: RankedOperand<'_, D::Buffer<T>, 2>,
        axis: usize,
        operation: ScanOperation,
        direction: ScanDirection,
        output: RankedOperand<'_, D::Buffer<T>, 2>,
    ) -> hephaestus_core::Result<()> {
        let operations = S::default();
        let input = StridedView::new(input.buffer, input.layout);
        let output = StridedView::new(output.buffer, output.layout);
        match operation {
            ScanOperation::Sum => {
                let prepared = operations.prepare_scan_axis::<hephaestus_core::CumSumOp, 2>(
                    device, input, axis, direction, output,
                )?;
                operations.dispatch_scan(device, &prepared)
            }
            ScanOperation::Product => {
                let prepared = operations.prepare_scan_axis::<hephaestus_core::CumProdOp, 2>(
                    device, input, axis, direction, output,
                )?;
                operations.dispatch_scan(device, &prepared)
            }
        }
    }
}

/// A fixed-rank Hephaestus buffer paired with its logical Leto layout.
#[derive(Clone, Copy)]
pub struct RankedOperand<'a, B, const N: usize> {
    /// Typed device buffer handle.
    pub buffer: &'a B,
    /// Logical shape, strides, and offset.
    pub layout: &'a LetoLayout<N>,
}

/// Provider implementation of scalar-specific rank-2 reduction and scan
/// kernels.
pub trait ReductionProvider<T>: HephaestusProvider
where
    T: Scalar + leto_ops::Scalar,
{
    /// Provider-owned axis-reduction kernel bundle.
    type AxisOperations: AxisReductionOps<Self::Device, T>
        + AxisReductionDispatch<Self::Device, T>
        + Default;

    /// Provider-owned scan kernel bundle.
    type ScanOperations: ScanOps<Self::Device, T> + ScanDispatch<Self::Device, T> + Default;

    /// Reduce a rank-2 strided input into a keep-dimension output.
    fn reduce(
        device: &Self::Device,
        op: ReductionOp,
        input: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<T>, 2>,
        axis: usize,
        output: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<T>, 2>,
    ) -> hephaestus_core::Result<()> {
        Self::AxisOperations::reduce(device, op, input, axis, output)
    }

    /// Execute an inclusive prefix or suffix scan over a rank-2 strided input.
    fn scan(
        device: &Self::Device,
        input: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<T>, 2>,
        axis: usize,
        operation: ScanOperation,
        direction: ScanDirection,
        output: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<T>, 2>,
    ) -> hephaestus_core::Result<()> {
        Self::ScanOperations::scan(device, input, axis, operation, direction, output)
    }
}

/// Generic Coeus backend implementation over one Hephaestus provider.
#[derive(Debug)]
pub struct HephaestusBackend<P>(std::marker::PhantomData<P>);

struct ScanRequest<'a, P, T>
where
    P: HephaestusProvider,
    T: bytemuck::Pod + eunomia::Pod,
{
    input: &'a HephaestusStorage<P, T>,
    input_layout: &'a Layout,
    axis: usize,
    operation_kind: ScanOperation,
    direction: ScanDirection,
    output: &'a mut HephaestusStorage<P, T>,
    output_layout: &'a Layout,
    operation: &'static str,
}

impl<P> Copy for HephaestusBackend<P> {}

impl<P> Clone for HephaestusBackend<P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<P> Default for HephaestusBackend<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P> HephaestusBackend<P> {
    /// Construct the zero-sized generic backend selector.
    #[must_use]
    pub const fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<P> ComputeBackend for HephaestusBackend<P>
where
    P: HephaestusProvider,
{
    type Error = HephaestusBackendError;
    type DeviceBuffer<T: Scalar> = HephaestusStorage<P, T>;
    type KernelDescriptor = ();
    type DispatchFuture<T: Scalar> = Ready<T>;

    fn name(&self) -> &'static str {
        P::NAME
    }

    fn num_threads(&self) -> usize {
        1
    }

    fn allocate<T: Scalar>(&self, len: usize) -> Self::DeviceBuffer<T> {
        HephaestusStorage::uninitialized(len)
    }

    fn allocate_zeroed<T: Scalar>(&self, len: usize) -> Self::DeviceBuffer<T> {
        HephaestusStorage::new(len)
    }

    fn fill<T: Scalar>(&self, dst: &mut Self::DeviceBuffer<T>, val: T) {
        let values = vec![val; dst.buffer().len()];
        P::device()
            .write_buffer(dst.buffer(), &values)
            .expect("Hephaestus fill failed");
    }

    fn copy_to_device<T: Scalar>(&self, src: &[T], dst: &mut Self::DeviceBuffer<T>) {
        P::device()
            .write_buffer(dst.buffer(), src)
            .expect("Hephaestus host-to-device copy failed");
    }

    fn copy_to_host<T: Scalar>(&self, src: &Self::DeviceBuffer<T>, dst: &mut [T]) {
        P::device()
            .download(src.buffer(), dst)
            .expect("Hephaestus device-to-host copy failed");
    }
}

impl<P, T> coeus_ops::ReductionOps<T> for HephaestusBackend<P>
where
    P: ReductionProvider<T>,
    T: Scalar + leto_ops::Scalar,
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
        let input_layout = ranked::<2>("reduce", a_layout)?;
        let output_layout = ranked::<2>("reduce", c_layout)?;
        let provider_axis = ranked_axis::<2>("reduce", a_layout, axis)?;
        P::reduce(
            P::device(),
            op,
            RankedOperand {
                buffer: a.buffer(),
                layout: &input_layout,
            },
            provider_axis,
            RankedOperand {
                buffer: c.buffer(),
                layout: &output_layout,
            },
        )
        .map_err(|source| HephaestusBackendError::device("reduce", source))
    }

    fn cumsum(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error> {
        self.scan(ScanRequest {
            input: a,
            input_layout: a_layout,
            axis,
            operation_kind: ScanOperation::Sum,
            direction: ScanDirection::Forward,
            output: c,
            output_layout: c_layout,
            operation: "cumsum",
        })
    }

    fn suffix_sum(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error> {
        self.scan(ScanRequest {
            input: a,
            input_layout: a_layout,
            axis,
            operation_kind: ScanOperation::Sum,
            direction: ScanDirection::Reverse,
            output: c,
            output_layout: c_layout,
            operation: "suffix_sum",
        })
    }

    fn cumprod(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error> {
        self.scan(ScanRequest {
            input: a,
            input_layout: a_layout,
            axis,
            operation_kind: ScanOperation::Product,
            direction: ScanDirection::Forward,
            output: c,
            output_layout: c_layout,
            operation: "cumprod",
        })
    }

    fn suffix_prod(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error> {
        self.scan(ScanRequest {
            input: a,
            input_layout: a_layout,
            axis,
            operation_kind: ScanOperation::Product,
            direction: ScanDirection::Reverse,
            output: c,
            output_layout: c_layout,
            operation: "suffix_prod",
        })
    }
}

impl<P> HephaestusBackend<P>
where
    P: HephaestusProvider,
{
    fn scan<T>(&self, request: ScanRequest<'_, P, T>) -> Result<(), HephaestusBackendError>
    where
        P: ReductionProvider<T>,
        T: Scalar + leto_ops::Scalar,
    {
        let input_layout = ranked::<2>(request.operation, request.input_layout)?;
        let output_layout = ranked::<2>(request.operation, request.output_layout)?;
        let provider_axis =
            ranked_axis::<2>(request.operation, request.input_layout, request.axis)?;
        P::scan(
            P::device(),
            RankedOperand {
                buffer: request.input.buffer(),
                layout: &input_layout,
            },
            provider_axis,
            request.operation_kind,
            request.direction,
            RankedOperand {
                buffer: request.output.buffer(),
                layout: &output_layout,
            },
        )
        .map_err(|source| HephaestusBackendError::device(request.operation, source))
    }
}
