use coeus_core::{Scalar, Float, MoiraiBackend};
use coeus_tensor::Tensor;
use coeus_autograd::Var;
use crate::module::Module;

#[derive(Clone)]
pub struct Conv2d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    pub weight: Var<T, B>,
    pub bias: Option<Var<T, B>>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Conv2d<T, B> {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, bias: bool) -> Self {
        Self::with_params(in_channels, out_channels, kernel_size, 1, 0, 1, bias)
    }

    pub fn with_params(
        in_channels: usize, out_channels: usize, kernel_size: usize,
        stride: usize, padding: usize, dilation: usize, bias: bool,
    ) -> Self {
        assert!(stride >= 1 && dilation >= 1, "stride and dilation must be >= 1");
        let backend = B::default();
        let w_shape = [out_channels, in_channels, kernel_size, kernel_size];
        let weight = Var::new(Tensor::ones_on(w_shape, &backend), true);
        let bias_var = if bias {
            Some(Var::new(Tensor::zeros_on([out_channels], &backend), true))
        } else { None };
        Self { weight, bias: bias_var, in_channels, out_channels, kernel_size, stride, padding, dilation }
    }

    #[inline]
    fn k_eff(&self) -> usize { self.dilation * (self.kernel_size - 1) + 1 }

    #[inline]
    fn out_dim(&self, input_dim: usize) -> usize {
        let total = input_dim + 2 * self.padding;
        match total.checked_sub(self.k_eff()) {
            Some(numer) => numer / self.stride + 1,
            None => 0,
        }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Conv2d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        let mut p = vec![self.weight.clone()];
        if let Some(ref b) = self.bias { p.push(b.clone()); }
        p
    }

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        let backend = B::default();
        let h = input.tensor.shape()[2];
        let w = input.tensor.shape()[3];
        let h_out = self.out_dim(h);
        let w_out = self.out_dim(w);
        assert!(h_out > 0 && w_out > 0, "Conv2d: kernel does not fit input shape");

        let shape = [input.tensor.shape()[0], self.out_channels, h_out, w_out];
        let mut out_tensor = Tensor::zeros_on(shape, &backend);

        let (out_storage, out_layout) = out_tensor.storage_mut_and_layout();

        backend.conv2d(
            input.tensor.storage(),
            input.tensor.layout(),
            self.weight.tensor.storage(),
            self.weight.tensor.layout(),
            self.bias.as_ref().map(|b| b.tensor.storage()),
            self.stride,
            self.padding,
            self.dilation,
            out_storage,
            out_layout,
        );

        coeus_autograd::conv2d(
            input,
            &self.weight,
            &self.bias,
            out_tensor,
            self.stride,
            self.padding,
            self.dilation,
        )
    }
}
