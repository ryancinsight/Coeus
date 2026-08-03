use coeus_autograd::Var;
use coeus_core::Float;
use coeus_tensor::Tensor;

/// Initialize weights with a constant value.
pub fn constant<T: Float, B: coeus_ops::BackendOps<T> + Default>(weight: &mut Var<T, B>, val: f64) {
    let shape = weight.tensor.shape_cloned();
    weight.tensor = Tensor::full_on(shape, T::from_f64(val), &B::default());
}

/// Initialize weights with zeros.
pub fn zeros<T: Float, B: coeus_ops::BackendOps<T> + Default>(weight: &mut Var<T, B>) {
    let shape = weight.tensor.shape_cloned();
    weight.tensor = Tensor::zeros_on(shape, &B::default());
}

/// Initialize weights with ones.
pub fn ones<T: Float, B: coeus_ops::BackendOps<T> + Default>(weight: &mut Var<T, B>) {
    let shape = weight.tensor.shape_cloned();
    weight.tensor = Tensor::ones_on(shape, &B::default());
}
