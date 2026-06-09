use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar, Shape};
use coeus_tensor::Tensor;
use std::sync::{Arc, Mutex};

pub struct MaxPool2dNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<Mutex<Tensor<T, B>>>,
    pub inputs: Vec<Var<T, B>>,
    pub inp_clone: Tensor<T, B>,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for MaxPool2dNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "max_pool2d"
    }

    #[inline]
    fn output_grad(&self) -> &Arc<Mutex<Tensor<T, B>>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    #[inline]
    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<Mutex<Tensor<T, B>>>>]) {
        let backend = B::default();
        if let Some(Some(ref g_in)) = input_grads.get(0) {
            let mut grad_input = Tensor::zeros_on(self.inp_clone.shape_cloned(), &backend);
            let (gi_storage, gi_layout) = grad_input.storage_mut_and_layout();
            backend.max_pool2d_backward(
                grad_out.storage(),
                grad_out.layout(),
                self.inp_clone.storage(),
                self.inp_clone.layout(),
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                gi_storage,
                gi_layout,
            );
            let mut gl = g_in.lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &grad_input, &backend);
        }
    }
}

/// Tracked 2D Max Pooling.
pub fn max_pool2d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    out_tensor: Tensor<T, B>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Var<T, B> {
    let backend = B::default();
    let requires_grad = input.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(Mutex::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let inputs = vec![input.clone()];
        let inp_clone = input.tensor.clone();

        let node = MaxPool2dNode {
            output_grad,
            inputs,
            inp_clone,
            kernel_size,
            stride,
            padding,
            dilation,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };

    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}

pub struct AvgPool2dNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<Mutex<Tensor<T, B>>>,
    pub inputs: Vec<Var<T, B>>,
    pub inp_shape: Shape,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for AvgPool2dNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "avg_pool2d"
    }

    #[inline]
    fn output_grad(&self) -> &Arc<Mutex<Tensor<T, B>>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    #[inline]
    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<Mutex<Tensor<T, B>>>>]) {
        let backend = B::default();
        if let Some(Some(ref g_in)) = input_grads.get(0) {
            let mut grad_input = Tensor::zeros_on(self.inp_shape.clone(), &backend);
            let (gi_storage, gi_layout) = grad_input.storage_mut_and_layout();
            backend.avg_pool2d_backward(
                grad_out.storage(),
                grad_out.layout(),
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                gi_storage,
                gi_layout,
            );
            let mut gl = g_in.lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &grad_input, &backend);
        }
    }
}

/// Tracked 2D Average Pooling.
pub fn avg_pool2d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    out_tensor: Tensor<T, B>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Var<T, B> {
    let backend = B::default();
    let requires_grad = input.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(Mutex::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let inputs = vec![input.clone()];
        let inp_shape = input.tensor.shape_cloned();

        let node = AvgPool2dNode {
            output_grad,
            inputs,
            inp_shape,
            kernel_size,
            stride,
            padding,
            dilation,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };

    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}

pub struct MaxPool3dNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<Mutex<Tensor<T, B>>>,
    pub inputs: Vec<Var<T, B>>,
    pub inp_clone: Tensor<T, B>,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for MaxPool3dNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "max_pool3d"
    }

    #[inline]
    fn output_grad(&self) -> &Arc<Mutex<Tensor<T, B>>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    #[inline]
    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<Mutex<Tensor<T, B>>>>]) {
        let backend = B::default();
        if let Some(Some(ref g_in)) = input_grads.get(0) {
            let mut grad_input = Tensor::zeros_on(self.inp_clone.shape_cloned(), &backend);
            let grad_input_layout = grad_input.layout().clone();
            backend.max_pool3d_backward(
                grad_out.storage(),
                grad_out.layout(),
                self.inp_clone.storage(),
                self.inp_clone.layout(),
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                grad_input.storage_mut(),
                &grad_input_layout,
            );
            let mut gl = g_in.lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &grad_input, &backend);
        }
    }
}

pub fn max_pool3d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    out_tensor: Tensor<T, B>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Var<T, B> {
    let backend = B::default();
    let requires_grad = input.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(Mutex::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let inputs = vec![input.clone()];
        let inp_clone = input.tensor.clone();

        let node = MaxPool3dNode {
            output_grad,
            inputs,
            inp_clone,
            kernel_size,
            stride,
            padding,
            dilation,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };

    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}

pub struct AvgPool3dNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<Mutex<Tensor<T, B>>>,
    pub inputs: Vec<Var<T, B>>,
    pub inp_shape: Shape,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for AvgPool3dNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "avg_pool3d"
    }

    #[inline]
    fn output_grad(&self) -> &Arc<Mutex<Tensor<T, B>>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    #[inline]
    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<Mutex<Tensor<T, B>>>>]) {
        let backend = B::default();
        if let Some(Some(ref g_in)) = input_grads.get(0) {
            let mut grad_input = Tensor::zeros_on(self.inp_shape.clone(), &backend);
            let grad_input_layout = grad_input.layout().clone();
            backend.avg_pool3d_backward(
                grad_out.storage(),
                grad_out.layout(),
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                grad_input.storage_mut(),
                &grad_input_layout,
            );
            let mut gl = g_in.lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &grad_input, &backend);
        }
    }
}

pub fn avg_pool3d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    out_tensor: Tensor<T, B>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Var<T, B> {
    let backend = B::default();
    let requires_grad = input.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(Mutex::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let inputs = vec![input.clone()];
        let inp_shape = input.tensor.shape_cloned();

        let node = AvgPool3dNode {
            output_grad,
            inputs,
            inp_shape,
            kernel_size,
            stride,
            padding,
            dilation,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };

    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}
