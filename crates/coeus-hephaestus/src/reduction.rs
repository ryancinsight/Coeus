use crate::{error::HephaestusBackendError, layout::ranked, storage::HephaestusStorage};
use coeus_core::{ComputeBackend, Layout, Scalar};
use coeus_ops::ReductionOp;
use hephaestus_core::{ComputeDevice, DeviceBuffer, ScanDirection};
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
}

/// Cumulative operation selected by a provider scan dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanOperation {
    /// Inclusive cumulative addition.
    Sum,
    /// Inclusive cumulative multiplication.
    Product,
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
    /// Reduce a rank-2 strided input into a keep-dimension output.
    fn reduce(
        device: &Self::Device,
        op: ReductionOp,
        input: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<T>, 2>,
        axis: usize,
        output: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<T>, 2>,
    ) -> hephaestus_core::Result<()>;

    /// Execute an inclusive prefix or suffix scan over a rank-2 strided input.
    fn scan(
        device: &Self::Device,
        input: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<T>, 2>,
        axis: usize,
        operation: ScanOperation,
        direction: ScanDirection,
        output: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<T>, 2>,
    ) -> hephaestus_core::Result<()>;
}

/// Generic Coeus backend implementation over one Hephaestus provider.
#[derive(Debug)]
pub struct HephaestusBackend<P>(std::marker::PhantomData<P>);

struct ScanRequest<'a, P, T>
where
    P: HephaestusProvider,
    T: bytemuck::Pod,
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

impl<P> coeus_core::backend::private::Sealed for HephaestusBackend<P> where P: HephaestusProvider {}

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
        P::reduce(
            P::device(),
            op,
            RankedOperand {
                buffer: a.buffer(),
                layout: &input_layout,
            },
            axis,
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
        P::scan(
            P::device(),
            RankedOperand {
                buffer: request.input.buffer(),
                layout: &input_layout,
            },
            request.axis,
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
