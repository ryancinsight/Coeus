//! End-to-end training example: linear regression by gradient descent.
//!
//! Demonstrates the full Coeus training loop on the CPU `SequentialBackend`:
//! build a forward graph from autograd `Var`s, backpropagate, and update
//! parameters with the fused `SGD` optimizer — recovering a known
//! `y = X·w + b` from synthetic data.
//!
//! The forward pass is built from the optimizer's owned named parameters
//! (`opt.params[..]`), which `SGD::step` updates in place; their gradient
//! buffers are `Arc`-shared into the graph, so `backward()` populates exactly
//! the gradients the optimizer reads.
//!
//! Run with:  `cargo run -p coeus-optim --example linear_regression`

use coeus_autograd::{add, matmul, mean, mul, sub, Parameter, Var};
use coeus_core::SequentialBackend;
use coeus_optim::{Optimizer, SGD};
use coeus_tensor::Tensor;

type B = SequentialBackend;

const N: usize = 64; // samples
const D: usize = 3; // features

fn main() {
    // Ground-truth parameters to recover.
    let w_true = [2.0f32, -3.0, 0.5];
    let b_true = 1.0f32;

    // Deterministic, well-conditioned design matrix X via a small LCG so the
    // columns are decorrelated (full column rank) — the least-squares solution
    // is then unique and equal to `w_true`, which the optimizer must recover.
    let mut rng: u32 = 0x1234_5678;
    let mut next = || {
        rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (rng >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0 // uniform in [-1, 1)
    };
    let x_data: Vec<f32> = (0..N * D).map(|_| next()).collect();
    let y_data: Vec<f32> = (0..N)
        .map(|n| {
            let mut acc = b_true;
            for d in 0..D {
                acc += x_data[n * D + d] * w_true[d];
            }
            acc
        })
        .collect();

    // Constants (no grad) and trainable parameters (zero-initialized).
    let x = Var::new(Tensor::<f32, B>::from_slice(vec![N, D], &x_data), false);
    let y = Var::new(Tensor::<f32, B>::from_slice(vec![N, 1], &y_data), false);
    let w = Var::new(Tensor::<f32, B>::zeros(vec![D, 1]), true);
    let b = Var::new(Tensor::<f32, B>::zeros(vec![1, 1]), true); // broadcasts over [N, 1]

    let mut opt = SGD::new(
        vec![Parameter::new(w, "weight"), Parameter::new(b, "bias")],
        0.1f32,
        0.9f32,
    );

    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;
    for epoch in 0..200 {
        opt.zero_grad();
        // pred = X·w + b   (b: [1,1] broadcasts to [N,1])
        let pred = add(&matmul(&x, &opt.params[0]), &opt.params[1]);
        let diff = sub(&pred, &y);
        let loss = mean(&mul(&diff, &diff)); // MSE
        loss.backward();
        opt.step();

        last_loss = loss.tensor.as_slice()[0];
        if epoch == 0 {
            first_loss = last_loss;
        }
        if epoch % 40 == 0 || epoch == 199 {
            println!("epoch {epoch:3}: mse = {last_loss:.6}");
        }
    }

    println!("\nrecovered w = {:?}", opt.params[0].tensor.as_slice());
    println!("recovered b = {:?}", opt.params[1].tensor.as_slice());
    println!("true      w = {w_true:?}, b = {b_true}");

    assert!(
        last_loss < first_loss * 1e-3,
        "training did not converge: first {first_loss:.6}, last {last_loss:.6}"
    );
    // With a full-rank design the least-squares solution is unique, so the
    // optimizer must recover the ground-truth parameters (analytical oracle).
    let w_rec = opt.params[0].tensor.as_slice();
    let b_rec = opt.params[1].tensor.as_slice();
    for (got, want) in w_rec.iter().zip(&w_true) {
        assert!(
            (got - want).abs() < 1e-2,
            "w mismatch: got {got}, want {want}"
        );
    }
    assert!(
        (b_rec[0] - b_true).abs() < 1e-2,
        "b mismatch: got {}, want {b_true}",
        b_rec[0]
    );
    println!(
        "\nconverged: loss reduced {:.0}x; parameters recovered",
        first_loss / last_loss
    );
}
