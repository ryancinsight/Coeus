/// Sealed-trait strategy encoding the dimension-specific behaviour of a
/// convolution layer: weight shape, spatial output computation, backend
/// dispatch, and autograd node construction.
///
/// Implementors: [`Dim1D`], [`Dim2D`], [`Dim3D`].
pub trait ConvDim: private::Sealed + 'static {
    /// Number of spatial axes handled by this strategy.
    const SPATIAL_RANK: usize;

    /// Weight tensor shape `[out_channels, in_channels, k...]`.
    fn weight_shape(oc: usize, ic: usize, k: usize) -> Vec<usize>;

    /// Output spatial dimension lengths.
    ///
    /// For each element `l` of `in_spatial` computes
    /// `(l + 2*padding - k_eff) / stride + 1`, or 0 on underflow.
    fn out_spatial(in_spatial: &[usize], k_eff: usize, stride: usize, padding: usize)
    -> Vec<usize>;

    /// Full output shape `[n, oc, out_spatial...]`.
    fn output_shape(n: usize, oc: usize, out_spatial: &[usize]) -> Vec<usize>;

    /// Invoke the correct backend convolution method.
    fn backend_conv<T: coeus_core::Scalar, B: coeus_ops::BackendOps<T>>(
        dispatch: ConvDispatch<'_, T, B>,
    ) -> Result<(), B::Error>;

    /// Invoke the correct autograd convolution function and return the output
    /// variable with its backward graph attached.
    fn autograd_conv<T: coeus_core::Float, B: coeus_ops::BackendOps<T> + Default>(
        input: &coeus_autograd::Var<T, B>,
        weight: &coeus_autograd::Var<T, B>,
        bias: &Option<coeus_autograd::Var<T, B>>,
        out_tensor: coeus_tensor::Tensor<T, B>,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> coeus_autograd::Var<T, B>;
}

mod private {
    pub trait Sealed {}
}

/// 1D convolution strategy.
#[derive(Clone, Default)]
pub struct Dim1D;
/// 2D convolution strategy.
#[derive(Clone, Default)]
pub struct Dim2D;
/// 3D convolution strategy.
#[derive(Clone, Default)]
pub struct Dim3D;

/// Borrowed backend convolution dispatch contract.
pub struct ConvDispatch<'a, T: coeus_core::Scalar, B: coeus_ops::BackendOps<T>> {
    /// Backend implementation that owns the concrete kernel dispatch.
    pub backend: &'a B,
    /// Input tensor storage.
    pub input_buf: &'a B::DeviceBuffer<T>,
    /// Input tensor layout.
    pub input_layout: &'a coeus_core::Layout,
    /// Weight tensor storage.
    pub weight_buf: &'a B::DeviceBuffer<T>,
    /// Weight tensor layout.
    pub weight_layout: &'a coeus_core::Layout,
    /// Optional bias tensor storage.
    pub bias: Option<&'a B::DeviceBuffer<T>>,
    /// Isotropic convolution stride.
    pub stride: usize,
    /// Symmetric zero padding.
    pub padding: usize,
    /// Dilation factor.
    pub dilation: usize,
    /// Output tensor storage.
    pub out_buf: &'a mut B::DeviceBuffer<T>,
    /// Output tensor layout.
    pub out_layout: &'a coeus_core::Layout,
}

impl private::Sealed for Dim1D {}
impl private::Sealed for Dim2D {}
impl private::Sealed for Dim3D {}

#[inline]
fn derive_out_spatial(
    in_spatial: &[usize],
    k_eff: usize,
    stride: usize,
    padding: usize,
) -> Vec<usize> {
    in_spatial
        .iter()
        .map(|&l| {
            (l + 2 * padding)
                .checked_sub(k_eff)
                .map(|n| n / stride + 1)
                .unwrap_or(0)
        })
        .collect()
}

#[inline]
fn derive_output_shape<D: ConvDim>(n: usize, oc: usize, out_spatial: &[usize]) -> Vec<usize> {
    let mut shape = Vec::with_capacity(2 + D::SPATIAL_RANK);
    shape.push(n);
    shape.push(oc);
    shape.extend_from_slice(out_spatial);
    shape
}

impl ConvDim for Dim1D {
    const SPATIAL_RANK: usize = 1;

    #[inline]
    fn weight_shape(oc: usize, ic: usize, k: usize) -> Vec<usize> {
        vec![oc, ic, k]
    }

    #[inline]
    fn out_spatial(
        in_spatial: &[usize],
        k_eff: usize,
        stride: usize,
        padding: usize,
    ) -> Vec<usize> {
        derive_out_spatial(in_spatial, k_eff, stride, padding)
    }

    #[inline]
    fn output_shape(n: usize, oc: usize, out_spatial: &[usize]) -> Vec<usize> {
        derive_output_shape::<Self>(n, oc, out_spatial)
    }

    #[inline]
    fn backend_conv<T: coeus_core::Scalar, B: coeus_ops::BackendOps<T>>(
        dispatch: ConvDispatch<'_, T, B>,
    ) -> Result<(), B::Error> {
        dispatch.backend.conv1d(
            dispatch.input_buf,
            dispatch.input_layout,
            dispatch.weight_buf,
            dispatch.weight_layout,
            dispatch.bias,
            dispatch.stride,
            dispatch.padding,
            dispatch.dilation,
            dispatch.out_buf,
            dispatch.out_layout,
        )
    }

    #[inline]
    fn autograd_conv<T: coeus_core::Float, B: coeus_ops::BackendOps<T> + Default>(
        input: &coeus_autograd::Var<T, B>,
        weight: &coeus_autograd::Var<T, B>,
        bias: &Option<coeus_autograd::Var<T, B>>,
        out_tensor: coeus_tensor::Tensor<T, B>,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> coeus_autograd::Var<T, B> {
        coeus_autograd::conv1d(input, weight, bias, out_tensor, stride, padding, dilation)
    }
}

impl ConvDim for Dim2D {
    const SPATIAL_RANK: usize = 2;

    #[inline]
    fn weight_shape(oc: usize, ic: usize, k: usize) -> Vec<usize> {
        vec![oc, ic, k, k]
    }

    #[inline]
    fn out_spatial(
        in_spatial: &[usize],
        k_eff: usize,
        stride: usize,
        padding: usize,
    ) -> Vec<usize> {
        derive_out_spatial(in_spatial, k_eff, stride, padding)
    }

    #[inline]
    fn output_shape(n: usize, oc: usize, out_spatial: &[usize]) -> Vec<usize> {
        derive_output_shape::<Self>(n, oc, out_spatial)
    }

    #[inline]
    fn backend_conv<T: coeus_core::Scalar, B: coeus_ops::BackendOps<T>>(
        dispatch: ConvDispatch<'_, T, B>,
    ) -> Result<(), B::Error> {
        dispatch.backend.conv2d(
            dispatch.input_buf,
            dispatch.input_layout,
            dispatch.weight_buf,
            dispatch.weight_layout,
            dispatch.bias,
            dispatch.stride,
            dispatch.padding,
            dispatch.dilation,
            dispatch.out_buf,
            dispatch.out_layout,
        )
    }

    #[inline]
    fn autograd_conv<T: coeus_core::Float, B: coeus_ops::BackendOps<T> + Default>(
        input: &coeus_autograd::Var<T, B>,
        weight: &coeus_autograd::Var<T, B>,
        bias: &Option<coeus_autograd::Var<T, B>>,
        out_tensor: coeus_tensor::Tensor<T, B>,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> coeus_autograd::Var<T, B> {
        coeus_autograd::conv2d(input, weight, bias, out_tensor, stride, padding, dilation)
    }
}

impl ConvDim for Dim3D {
    const SPATIAL_RANK: usize = 3;

    #[inline]
    fn weight_shape(oc: usize, ic: usize, k: usize) -> Vec<usize> {
        vec![oc, ic, k, k, k]
    }

    #[inline]
    fn out_spatial(
        in_spatial: &[usize],
        k_eff: usize,
        stride: usize,
        padding: usize,
    ) -> Vec<usize> {
        derive_out_spatial(in_spatial, k_eff, stride, padding)
    }

    #[inline]
    fn output_shape(n: usize, oc: usize, out_spatial: &[usize]) -> Vec<usize> {
        derive_output_shape::<Self>(n, oc, out_spatial)
    }

    #[inline]
    fn backend_conv<T: coeus_core::Scalar, B: coeus_ops::BackendOps<T>>(
        dispatch: ConvDispatch<'_, T, B>,
    ) -> Result<(), B::Error> {
        dispatch.backend.conv3d(
            dispatch.input_buf,
            dispatch.input_layout,
            dispatch.weight_buf,
            dispatch.weight_layout,
            dispatch.bias,
            dispatch.stride,
            dispatch.padding,
            dispatch.dilation,
            dispatch.out_buf,
            dispatch.out_layout,
        )
    }

    #[inline]
    fn autograd_conv<T: coeus_core::Float, B: coeus_ops::BackendOps<T> + Default>(
        input: &coeus_autograd::Var<T, B>,
        weight: &coeus_autograd::Var<T, B>,
        bias: &Option<coeus_autograd::Var<T, B>>,
        out_tensor: coeus_tensor::Tensor<T, B>,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> coeus_autograd::Var<T, B> {
        coeus_autograd::conv3d(input, weight, bias, out_tensor, stride, padding, dilation)
    }
}
