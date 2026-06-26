use crate::module::Module;
use crate::pool::out_dim;
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend, Scalar};
use coeus_tensor::Tensor;
use std::marker::PhantomData;

// ── AvgPool2d ──

#[derive(Clone)]
pub struct AvgPool2d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    _marker: PhantomData<(T, B)>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> AvgPool2d<T, B> {
    pub fn new(kernel_size: usize) -> Self {
        Self::with_params(kernel_size, kernel_size, 0, 1)
    }

    pub fn with_params(kernel_size: usize, stride: usize, padding: usize, dilation: usize) -> Self {
        assert!(
            stride >= 1 && dilation >= 1,
            "stride and dilation must be >= 1"
        );
        Self {
            kernel_size,
            stride,
            padding,
            dilation,
            _marker: PhantomData,
        }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for AvgPool2d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        let backend = B::default();

        let n = input.tensor.shape()[0];
        let c = input.tensor.shape()[1];
        let h = input.tensor.shape()[2];
        let w = input.tensor.shape()[3];
        let h_out = out_dim(
            h,
            self.kernel_size,
            self.padding,
            self.stride,
            self.dilation,
        );
        let w_out = out_dim(
            w,
            self.kernel_size,
            self.padding,
            self.stride,
            self.dilation,
        );
        assert!(
            h_out > 0 && w_out > 0,
            "AvgPool2d: kernel ({}) with dilation ({}) and padding ({}) \
             does not fit input spatial dims [{h}x{w}]; output would be [{h_out}x{w_out}]",
            self.kernel_size,
            self.dilation,
            self.padding,
        );

        let mut out_tensor = Tensor::zeros_on([n, c, h_out, w_out], &backend);
        let (out_storage, out_layout) = out_tensor.storage_mut_and_layout();

        backend.avg_pool2d(
            input.tensor.storage(),
            input.tensor.layout(),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            out_storage,
            out_layout,
        );

        coeus_autograd::avg_pool2d(
            input,
            out_tensor,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )
    }
}

// ── AvgPool3d ──

#[derive(Clone)]
pub struct AvgPool3d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    _marker: PhantomData<(T, B)>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> AvgPool3d<T, B> {
    pub fn new(kernel_size: usize) -> Self {
        Self::with_params(kernel_size, kernel_size, 0, 1)
    }

    pub fn with_params(kernel_size: usize, stride: usize, padding: usize, dilation: usize) -> Self {
        assert!(
            stride >= 1 && dilation >= 1,
            "stride and dilation must be >= 1"
        );
        Self {
            kernel_size,
            stride,
            padding,
            dilation,
            _marker: PhantomData,
        }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for AvgPool3d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        let backend = B::default();

        let n = input.tensor.shape()[0];
        let c = input.tensor.shape()[1];
        let d = input.tensor.shape()[2];
        let h = input.tensor.shape()[3];
        let w = input.tensor.shape()[4];
        let d_out = out_dim(
            d,
            self.kernel_size,
            self.padding,
            self.stride,
            self.dilation,
        );
        let h_out = out_dim(
            h,
            self.kernel_size,
            self.padding,
            self.stride,
            self.dilation,
        );
        let w_out = out_dim(
            w,
            self.kernel_size,
            self.padding,
            self.stride,
            self.dilation,
        );
        assert!(
            d_out > 0 && h_out > 0 && w_out > 0,
            "AvgPool3d: kernel ({}) with dilation ({}) and padding ({}) \
             does not fit input spatial dims [{d}x{h}x{w}]",
            self.kernel_size,
            self.dilation,
            self.padding,
        );

        let mut out_tensor = Tensor::zeros_on([n, c, d_out, h_out, w_out], &backend);
        let (out_storage, out_layout) = out_tensor.storage_mut_and_layout();

        backend.avg_pool3d(
            input.tensor.storage(),
            input.tensor.layout(),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            out_storage,
            out_layout,
        );

        coeus_autograd::avg_pool3d(
            input,
            out_tensor,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )
    }
}
