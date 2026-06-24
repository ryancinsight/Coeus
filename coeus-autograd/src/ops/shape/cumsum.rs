use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Scalar;
use coeus_tensor::Tensor;
use std::sync::Arc;

pub struct CumSumNode<T: Scalar + leto_ops::Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    pub dim: usize,
}

impl<T: Scalar + leto_ops::Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B>
    for CumSumNode<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "cumsum"
    }

    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    #[inline]
    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.first() {
            let suffix_grad = coeus_ops::suffix_sum(grad_out, self.dim);
            let gl = g.write();
            coeus_ops::add_assign(gl, &suffix_grad, &backend);
        }
    }
}

#[inline]
pub fn cumsum<T: Scalar + leto_ops::Scalar, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    dim: usize,
) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let out_tensor = coeus_ops::cumsum(&x.tensor, dim);

    let requires_grad = x.grad.is_some();
    if !requires_grad {
        return Var::new(out_tensor, false);
    }

    let output_grad = Arc::new(GradBuffer::new(Tensor::zeros_on(
        out_tensor.shape_cloned(),
        &backend,
    )));
    let grad = Some(output_grad.clone());

    let node = CumSumNode {
        output_grad,
        inputs: vec![x.clone()],
        dim,
    };
    let creator = Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>);

    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}
