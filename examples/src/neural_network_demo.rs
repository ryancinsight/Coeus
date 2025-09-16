//! Executable example demonstrating neural network functionality

use coeus_nn::{Linear, Module, MseLoss, ReLU};
use coeus_tensor::Tensor;

pub fn run_neural_network_demo() {
    println!("🧠 Neural Network Forward Pass Demo");
    println!("====================================");

    // Simple neural network: 2 -> 4 -> 1
    println!("\n🏗️  Building Neural Network:");
    println!("---------------------------");

    let model = [
        Box::new(Linear::new(2, 4)) as Box<dyn Module<f32>>,
        Box::new(ReLU::new()),
        Box::new(Linear::new(4, 1)),
    ];

    println!("Network architecture: Linear(2→4) → ReLU → Linear(4→1)");

    // Test data: XOR problem inputs
    let inputs = [
        Tensor::from_vec(vec![0.0, 0.0], vec![2]),
        Tensor::from_vec(vec![0.0, 1.0], vec![2]),
        Tensor::from_vec(vec![1.0, 0.0], vec![2]),
        Tensor::from_vec(vec![1.0, 1.0], vec![2]),
    ];

    println!("\n🧪 Testing Neural Network Forward Pass:");
    println!("---------------------------------------");
    println!("Input → Network Output");
    println!("Note: This demonstrates the NN module functionality.");
    println!("Full training integration requires additional autograd work.");

    for (i, input) in inputs.iter().enumerate() {
        // Forward pass through all layers
        let mut output = input.clone();
        for (layer_idx, layer) in model.iter().enumerate() {
            output = layer.forward(&output).unwrap();
            if layer_idx == 0 {
                print!("Input {:?} → ", input.data());
            }
            if layer_idx < model.len() - 1 {
                print!("Layer {} → ", layer_idx + 1);
            }
        }

        println!("Output: {:.4}", output.data()[0]);

        // Expected XOR outputs for reference
        let expected = match i {
            0 => 0.0, // [0,0] → 0
            1 => 1.0, // [0,1] → 1
            2 => 1.0, // [1,0] → 1
            3 => 0.0, // [1,1] → 0
            _ => 0.0,
        };
        println!("           (Expected XOR output: {:.0})", expected);
        println!();
    }

    // Demonstrate loss computation separately
    println!("📊 Loss Function Demo:");
    println!("----------------------");

    let loss_fn = MseLoss::new();
    let pred = Tensor::from_vec(vec![0.8], vec![1]);
    let target = Tensor::from_vec(vec![1.0], vec![1]);

    let loss = loss_fn.forward(&pred, &target).unwrap();
    println!(
        "Prediction: {:.3}, Target: {:.3} → MSE Loss: {:.6}",
        pred.data()[0],
        target.data()[0],
        loss.data()[0]
    );

    println!("\n✅ Neural network demo completed!");
    println!("Note: Full training loop integration pending autograd enhancements.");
}
