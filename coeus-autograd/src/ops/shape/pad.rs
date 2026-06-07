use std::sync::{Arc, Mutex};
use coeus_core::Scalar;
use coeus_tensor::Tensor;
use crate::node::BackwardNode;
use crate::var::Var;

pub struct PadNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default>
where
    B::DeviceBuffer<T>: coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    pub output_grad: Arc<Mutex<Tensor<T, B>>>,
    pub inputs: Vec<Var<T, B>>,
    pub pads: Vec<(usize, usize)>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for PadNode<T, B>
where
    B::DeviceBuffer<T>: coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    #[inline]
    fn op_name(&self) -> &'static str { "pad" }

    #[inline]
    fn output_grad(&self) -> &Arc<Mutex<Tensor<T, B>>> { &self.output_grad }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] { &self.inputs }

    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<Mutex<Tensor<T, B>>>>]) {
        let backend = B::default();
        let Some(Some(ref acc)) = input_grads.first() else { return; };

        let ndim = grad_out.ndim();
        let in_shape = {
            let lock = acc.lock().unwrap();
            lock.shape_cloned()
        };

        let ranges: Vec<(usize, usize)> = (0..ndim)
            .map(|d| {
                let before = self.pads[d].0;
                (before, before + in_shape[d])
            })
            .collect();

        let sliced_grad = grad_out.slice(&ranges);

        let mut lock = acc.lock().unwrap();
        coeus_ops::add_assign(&mut *lock, &sliced_grad, &backend);
    }
}

#[inline]
pub fn pad<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    pads: &[(usize, usize)],
    value: T,
) -> Var<T, B>
where
    B::DeviceBuffer<T>: coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let out_tensor = coeus_ops::pad(&x.tensor, pads, value);

    let requires_grad = x.grad.is_some();
    if !requires_grad {
        return Var::new(out_tensor, false);
    }

    let output_grad = Arc::new(Mutex::new(Tensor::zeros_on(out_tensor.shape_cloned(), &backend)));
    let grad = Some(output_grad.clone());

    let node = PadNode {
        output_grad,
        inputs: vec![x.clone()],
        pads: pads.to_vec(),
    };
    let creator = Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>);

    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}
