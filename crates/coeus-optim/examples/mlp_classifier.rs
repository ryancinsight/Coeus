//! End-to-end training example: a 2-layer MLP classifier.
//!
//! Builds on `linear_regression` to show a fuller workflow: a hidden ReLU
//! layer, a proper cross-entropy objective (`nll_loss` over `log_softmax`),
//! and the `Adam` optimizer — classifying three 2-D Gaussian blobs.
//!
//! As in `linear_regression`, the forward graph is built from the optimizer's
//! owned named parameters (`opt.params[..]`), whose `Arc`-shared gradient
//! buffers connect `backward()` to the `Adam` update. Weights use small random
//! initialization (LCG) to break hidden-unit symmetry; biases start at zero.
//!
//! Run with:  `cargo run -p coeus-optim --example mlp_classifier`

use coeus_autograd::{add, log_softmax, matmul, nll_loss, relu, Parameter, Var};
use coeus_core::SequentialBackend;
use coeus_optim::{Adam, Optimizer};
use coeus_tensor::Tensor;

type B = SequentialBackend;

const PER_CLASS: usize = 40;
const CLASSES: usize = 3;
const N: usize = PER_CLASS * CLASSES; // samples
const D_IN: usize = 2; // input features
const D_HID: usize = 16; // hidden units

fn main() {
    // Class centers (well-separated) + small LCG noise -> linearly separable-ish
    // blobs that a small MLP classifies near-perfectly.
    let centers = [[2.0f32, 0.0], [-1.0, 2.0], [-1.0, -2.0]];
    let mut rng: u32 = 0x9E37_79B9;
    let mut uniform = move || {
        rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (rng >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0 // [-1, 1)
    };

    let mut x_data = vec![0.0f32; N * D_IN];
    let mut targets = vec![0usize; N];
    for (cls, center) in centers.iter().enumerate() {
        for s in 0..PER_CLASS {
            let n = cls * PER_CLASS + s;
            x_data[n * D_IN] = center[0] + 0.35 * uniform();
            x_data[n * D_IN + 1] = center[1] + 0.35 * uniform();
            targets[n] = cls;
        }
    }
    let x = Var::new(Tensor::<f32, B>::from_slice(vec![N, D_IN], &x_data), false);

    // Parameters: W1 [D_IN, D_HID], b1 [1, D_HID], W2 [D_HID, CLASSES], b2 [1, CLASSES].
    let scale = 0.5;
    let w1: Vec<f32> = (0..D_IN * D_HID).map(|_| scale * uniform()).collect();
    let w2: Vec<f32> = (0..D_HID * CLASSES).map(|_| scale * uniform()).collect();
    let w1 = Var::new(Tensor::<f32, B>::from_slice(vec![D_IN, D_HID], &w1), true);
    let b1 = Var::new(Tensor::<f32, B>::zeros(vec![1, D_HID]), true);
    let w2 = Var::new(
        Tensor::<f32, B>::from_slice(vec![D_HID, CLASSES], &w2),
        true,
    );
    let b2 = Var::new(Tensor::<f32, B>::zeros(vec![1, CLASSES]), true);

    let mut opt = Adam::new(
        vec![
            Parameter::new(w1, "layer1.weight"),
            Parameter::new(b1, "layer1.bias"),
            Parameter::new(w2, "layer2.weight"),
            Parameter::new(b2, "layer2.bias"),
        ],
        0.05f32,
        0.9,
        0.999,
        1e-8,
    );

    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;
    for epoch in 0..300 {
        opt.zero_grad();
        // h = relu(X·W1 + b1);  logits = h·W2 + b2
        let h = relu(&add(&matmul(&x, &opt.params[0]), &opt.params[1]));
        let logits = add(&matmul(&h, &opt.params[2]), &opt.params[3]);
        let loss = nll_loss(&log_softmax(&logits, 1), &targets);
        loss.backward()
            .expect("invariant: valid autograd fixture completes backward");
        opt.step();

        last_loss = loss.tensor.as_slice()[0];
        if epoch == 0 {
            first_loss = last_loss;
        }
        if epoch % 60 == 0 || epoch == 299 {
            println!(
                "epoch {epoch:3}: nll = {last_loss:.4}, acc = {:.1}%",
                100.0 * accuracy(&opt, &x, &targets)
            );
        }
    }

    let acc = accuracy(&opt, &x, &targets);
    println!("\nfinal accuracy = {:.1}%", 100.0 * acc);
    assert!(
        last_loss < first_loss * 0.5,
        "loss did not decrease: first {first_loss:.4}, last {last_loss:.4}"
    );
    assert!(acc > 0.9, "classifier did not learn: accuracy {acc:.3}");
    println!("converged: MLP separates the three classes");
}

/// Forward pass + argmax accuracy against the integer labels.
fn accuracy(opt: &Adam<f32, B>, x: &Var<f32, B>, targets: &[usize]) -> f32 {
    let h = relu(&add(&matmul(x, &opt.params[0]), &opt.params[1]));
    let logits = add(&matmul(&h, &opt.params[2]), &opt.params[3]);
    let data = logits.tensor.as_slice();
    let mut correct = 0usize;
    for (n, &t) in targets.iter().enumerate() {
        let row = &data[n * CLASSES..(n + 1) * CLASSES];
        let pred = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        if pred == t {
            correct += 1;
        }
    }
    correct as f32 / targets.len() as f32
}
