//! Finite-difference check through the owning NN classification-loss wrapper.

use super::{tensor, GradcheckScalar};
use coeus_autograd::gradcheck;
use coeus_core::MoiraiBackend;
use coeus_nn::cross_entropy_loss;

fn cross_entropy_case<T: GradcheckScalar>()
where
    MoiraiBackend: coeus_ops::CrossEntropyOps<T>,
{
    // Softmax-plus-log is smooth everywhere, so any finite logits with
    // in-range targets give a well-posed central difference. The targets
    // mirror `nll_loss_case` so the two classification losses share one
    // fixture shape; the check goes through the `coeus_nn` forward wrapper
    // so provider forward, target preparation, and graph assembly are all
    // inside the differentiated closure.
    let logits = tensor::<T>(&[3, 4], 0.29);
    let targets = [2usize, 0, 3];

    gradcheck(&[logits], |v| {
        cross_entropy_loss(&v[0], &targets)
            .expect("invariant: in-range targets satisfy the forward contract")
    })
    .expect("cross_entropy_loss backward must match central differences");
}

#[test]
fn cross_entropy_backward_matches_finite_differences() {
    cross_entropy_case::<f64>();
    cross_entropy_case::<f32>();
}
