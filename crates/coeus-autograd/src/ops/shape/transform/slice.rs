use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Scalar;
use coeus_tensor::Tensor;
use std::sync::Arc;

pub struct SliceNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    pub ranges: Vec<(usize, usize)>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for SliceNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "slice"
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
    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.first() {
            let parent_grad = g.write();
            let (parent_storage, parent_layout) = parent_grad.storage_mut_and_layout()?;
            let sliced_layout = parent_layout.slice(&self.ranges);

            let parent_storage_imm: &B::DeviceBuffer<T> =
                unsafe { &*(parent_storage as *const B::DeviceBuffer<T>) };
            backend.elementwise_binary(
                coeus_ops::BinaryOp::Add,
                parent_storage_imm,
                &sliced_layout,
                grad_out.storage(),
                grad_out.layout(),
                parent_storage,
                &sliced_layout,
            )?;
        }

        Ok(())
    }
}

/// Tracked slice operation.
#[inline]
pub fn slice<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    ranges: &[(usize, usize)],
) -> Result<Var<T, B>, B::Error> {
    let backend = B::default();
    let out_tensor = x.tensor.slice(ranges);

    let requires_grad = crate::grad_mode::should_track_var(x);
    if !requires_grad {
        return Var::new(out_tensor, false);
    }

    let output_grad = Arc::new(GradBuffer::new(Tensor::zeros_on(
        out_tensor.shape_cloned(),
        &backend,
    )?));
    let grad = Some(output_grad.clone());

    let node = SliceNode {
        output_grad,
        inputs: vec![x.clone()],
        ranges: ranges.to_vec(),
    };
    let creator = Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>);

    Ok(Var {
        tensor: out_tensor,
        grad,
        creator,
    })
}
