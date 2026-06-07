use std::sync::{Arc, Mutex};
use coeus_core::{Scalar, Float};
use coeus_tensor::Tensor;
use crate::node::BackwardNode;
use crate::var::Var;

pub struct Conv1dNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<Mutex<Tensor<T, B>>>,
    pub inputs: Vec<Var<T, B>>,
    pub w_clone: Tensor<T, B>,
    pub inp_clone: Tensor<T, B>,
    pub has_bias: bool,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for Conv1dNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "conv1d"
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

        let mut grad_input = if input_grads.get(0).and_then(|g| g.as_ref()).is_some() {
            Some(Tensor::zeros_on(self.inp_clone.shape_cloned(), &backend))
        } else {
            None
        };

        let mut grad_weight = if input_grads.get(1).and_then(|g| g.as_ref()).is_some() {
            Some(Tensor::zeros_on(self.w_clone.shape_cloned(), &backend))
        } else {
            None
        };

        let mut grad_bias = if self.has_bias && input_grads.get(2).and_then(|g| g.as_ref()).is_some() {
            Some(Tensor::zeros_on([self.w_clone.shape()[0]], &backend))
        } else {
            None
        };

        let dummy_layout = grad_out.layout();

        let mut gi_storage = None;
        let mut gi_layout_val = None;
        if let Some(ref mut gi) = grad_input {
            let (store, lay) = gi.storage_mut_and_layout();
            gi_storage = Some(store);
            gi_layout_val = Some(lay);
        }
        let gi_layout_ref = gi_layout_val.unwrap_or(dummy_layout);

        let mut gw_storage = None;
        let mut gw_layout_val = None;
        if let Some(ref mut gw) = grad_weight {
            let (store, lay) = gw.storage_mut_and_layout();
            gw_storage = Some(store);
            gw_layout_val = Some(lay);
        }
        let gw_layout_ref = gw_layout_val.unwrap_or(dummy_layout);

        backend.conv1d_backward(
            grad_out.storage(),
            grad_out.layout(),
            self.inp_clone.storage(),
            self.inp_clone.layout(),
            self.w_clone.storage(),
            self.w_clone.layout(),
            gi_storage,
            gi_layout_ref,
            gw_storage,
            gw_layout_ref,
            grad_bias.as_mut().map(|gb| gb.storage_mut()),
            self.stride,
            self.padding,
            self.dilation,
        );

        if let Some(gi) = grad_input {
            let mut gl = input_grads[0].as_ref().unwrap().lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &gi, &backend);
        }
        if let Some(gw) = grad_weight {
            let mut gl = input_grads[1].as_ref().unwrap().lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &gw, &backend);
        }
        if let Some(gb) = grad_bias {
            let mut gl = input_grads[2].as_ref().unwrap().lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &gb, &backend);
        }
    }
}

/// Tracked 1D Convolution.
pub fn conv1d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    weight: &Var<T, B>,
    bias: &Option<Var<T, B>>,
    out_tensor: Tensor<T, B>,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Var<T, B> {
    let backend = B::default();
    let requires_grad = input.grad.is_some() || weight.grad.is_some()
        || bias.as_ref().map(|b| b.grad.is_some()).unwrap_or(false);

    let grad = if requires_grad {
        Some(Arc::new(Mutex::new(Tensor::zeros_on(out_tensor.shape_cloned(), &backend))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let inputs = {
            let mut v = vec![input.clone(), weight.clone()];
            if let Some(ref b) = bias { v.push(b.clone()); }
            v
        };
        let w_clone = weight.tensor.clone();
        let inp_clone = input.tensor.clone();
        let has_bias = bias.is_some();

        let node = Conv1dNode {
            output_grad,
            inputs,
            w_clone,
            inp_clone,
            has_bias,
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

pub struct Conv2dNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<Mutex<Tensor<T, B>>>,
    pub inputs: Vec<Var<T, B>>,
    pub w_clone: Tensor<T, B>,
    pub inp_clone: Tensor<T, B>,
    pub has_bias: bool,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for Conv2dNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "conv2d"
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

        let mut grad_input = if input_grads.get(0).and_then(|g| g.as_ref()).is_some() {
            Some(Tensor::zeros_on(self.inp_clone.shape_cloned(), &backend))
        } else {
            None
        };

        let mut grad_weight = if input_grads.get(1).and_then(|g| g.as_ref()).is_some() {
            Some(Tensor::zeros_on(self.w_clone.shape_cloned(), &backend))
        } else {
            None
        };

        let mut grad_bias = if self.has_bias && input_grads.get(2).and_then(|g| g.as_ref()).is_some() {
            Some(Tensor::zeros_on([self.w_clone.shape()[0]], &backend))
        } else {
            None
        };

        let dummy_layout = grad_out.layout();

        let mut gi_storage = None;
        let mut gi_layout_val = None;
        if let Some(ref mut gi) = grad_input {
            let (store, lay) = gi.storage_mut_and_layout();
            gi_storage = Some(store);
            gi_layout_val = Some(lay);
        }
        let gi_layout_ref = gi_layout_val.unwrap_or(dummy_layout);

        let mut gw_storage = None;
        let mut gw_layout_val = None;
        if let Some(ref mut gw) = grad_weight {
            let (store, lay) = gw.storage_mut_and_layout();
            gw_storage = Some(store);
            gw_layout_val = Some(lay);
        }
        let gw_layout_ref = gw_layout_val.unwrap_or(dummy_layout);

        backend.conv2d_backward(
            grad_out.storage(),
            grad_out.layout(),
            self.inp_clone.storage(),
            self.inp_clone.layout(),
            self.w_clone.storage(),
            self.w_clone.layout(),
            gi_storage,
            gi_layout_ref,
            gw_storage,
            gw_layout_ref,
            grad_bias.as_mut().map(|gb| gb.storage_mut()),
            self.stride,
            self.padding,
            self.dilation,
        );

        if let Some(gi) = grad_input {
            let mut gl = input_grads[0].as_ref().unwrap().lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &gi, &backend);
        }
        if let Some(gw) = grad_weight {
            let mut gl = input_grads[1].as_ref().unwrap().lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &gw, &backend);
        }
        if let Some(gb) = grad_bias {
            let mut gl = input_grads[2].as_ref().unwrap().lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &gb, &backend);
        }
    }
}

/// Tracked 2D Convolution.
pub fn conv2d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    weight: &Var<T, B>,
    bias: &Option<Var<T, B>>,
    out_tensor: Tensor<T, B>,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Var<T, B> {
    let backend = B::default();
    let requires_grad = input.grad.is_some() || weight.grad.is_some()
        || bias.as_ref().map(|b| b.grad.is_some()).unwrap_or(false);

    let grad = if requires_grad {
        Some(Arc::new(Mutex::new(Tensor::zeros_on(out_tensor.shape_cloned(), &backend))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let inputs = {
            let mut v = vec![input.clone(), weight.clone()];
            if let Some(ref b) = bias { v.push(b.clone()); }
            v
        };
        let w_clone = weight.tensor.clone();
        let inp_clone = input.tensor.clone();
        let has_bias = bias.is_some();

        let node = Conv2dNode {
            output_grad,
            inputs,
            w_clone,
            inp_clone,
            has_bias,
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

pub struct Conv3dNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<Mutex<Tensor<T, B>>>,
    pub inputs: Vec<Var<T, B>>,
    pub w_clone: Tensor<T, B>,
    pub inp_clone: Tensor<T, B>,
    pub has_bias: bool,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for Conv3dNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "conv3d"
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

        let mut grad_input = if input_grads.get(0).and_then(|g| g.as_ref()).is_some() {
            Some(Tensor::zeros_on(self.inp_clone.shape_cloned(), &backend))
        } else {
            None
        };

        let mut grad_weight = if input_grads.get(1).and_then(|g| g.as_ref()).is_some() {
            Some(Tensor::zeros_on(self.w_clone.shape_cloned(), &backend))
        } else {
            None
        };

        let mut grad_bias = if self.has_bias && input_grads.get(2).and_then(|g| g.as_ref()).is_some() {
            Some(Tensor::zeros_on([self.w_clone.shape()[0]], &backend))
        } else {
            None
        };

        let dummy_layout = grad_out.layout();

        let mut gi_storage = None;
        let mut gi_layout_val = None;
        if let Some(ref mut gi) = grad_input {
            let (store, lay) = gi.storage_mut_and_layout();
            gi_storage = Some(store);
            gi_layout_val = Some(lay);
        }
        let gi_layout_ref = gi_layout_val.unwrap_or(dummy_layout);

        let mut gw_storage = None;
        let mut gw_layout_val = None;
        if let Some(ref mut gw) = grad_weight {
            let (store, lay) = gw.storage_mut_and_layout();
            gw_storage = Some(store);
            gw_layout_val = Some(lay);
        }
        let gw_layout_ref = gw_layout_val.unwrap_or(dummy_layout);

        backend.conv3d_backward(
            grad_out.storage(),
            grad_out.layout(),
            self.inp_clone.storage(),
            self.inp_clone.layout(),
            self.w_clone.storage(),
            self.w_clone.layout(),
            gi_storage,
            gi_layout_ref,
            gw_storage,
            gw_layout_ref,
            grad_bias.as_mut().map(|gb| gb.storage_mut()),
            self.stride,
            self.padding,
            self.dilation,
        );

        if let Some(gi) = grad_input {
            let mut gl = input_grads[0].as_ref().unwrap().lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &gi, &backend);
        }
        if let Some(gw) = grad_weight {
            let mut gl = input_grads[1].as_ref().unwrap().lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &gw, &backend);
        }
        if let Some(gb) = grad_bias {
            let mut gl = input_grads[2].as_ref().unwrap().lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &gb, &backend);
        }
    }
}

/// Tracked 3D Convolution.
pub fn conv3d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    weight: &Var<T, B>,
    bias: &Option<Var<T, B>>,
    out_tensor: Tensor<T, B>,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Var<T, B> {
    let backend = B::default();
    let requires_grad = input.grad.is_some() || weight.grad.is_some()
        || bias.as_ref().map(|b| b.grad.is_some()).unwrap_or(false);

    let grad = if requires_grad {
        Some(Arc::new(Mutex::new(Tensor::zeros_on(out_tensor.shape_cloned(), &backend))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let inputs = {
            let mut v = vec![input.clone(), weight.clone()];
            if let Some(ref b) = bias { v.push(b.clone()); }
            v
        };
        let w_clone = weight.tensor.clone();
        let inp_clone = input.tensor.clone();
        let has_bias = bias.is_some();

        let node = Conv3dNode {
            output_grad,
            inputs,
            w_clone,
            inp_clone,
            has_bias,
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
