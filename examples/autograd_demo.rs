//! Autograd Example: Automatic differentiation in action
//!
//! This example demonstrates Coeus's automatic differentiation system,
//! showing how gradients are computed and propagated through computation graphs.

use coeus_autograd::ops::backward_with_grad;
use coeus_backend::CpuBackend;
use coeus_dtype::float::Float32;
use coeus_storage::DenseStorage;
use coeus_tensor::Tensor;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Coeus Automatic Differentiation Example");
    println!("=========================================\n");

    // Create tensors with gradient tracking
    println!("1. Creating tensors for gradient computation:");
    let mut x = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(2.0)],
        &[1],
    )?;
    x = x.requires_grad_(true);
    println!("   x = {}", x.as_slice()[0].get());

    let mut y = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(3.0)],
        &[1],
    )?;
    y = y.requires_grad_(true);
    println!("   y = {}", y.as_slice()[0].get());

    // Build computation graph
    println!("\n2. Building computation graph:");
    let z = &x + &y;
    println!("   z = x + y = {}", z.as_slice()[0].get());

    let mut four = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(4.0)],
        &[1],
    )?;
    four = four.requires_grad_(true);
    let w = &z * &four;
    println!("   w = z * 4 = {}", w.as_slice()[0].get());

    let loss = &w * &w; // w²
    println!("   loss = w² = {}", loss.as_slice()[0].get());

    // Before backward pass, no gradients
    println!("\n3. Before backward pass:");
    println!("   x.grad() = {:?}", x.grad().unwrap_err()); // Should fail
    println!("   y.grad() = {:?}", y.grad().unwrap_err()); // Should fail
    println!("   z.grad() = {:?}", z.grad().unwrap_err()); // Should fail

    // Backward pass
    println!("\n4. Computing gradients with backward pass:");
    let loss_grad = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0)],
        &[1],
    )?;
    backward_with_grad(&loss, &loss_grad)?;
    println!("   ∂loss/∂x = {:?}", x.grad()?.as_slice()[0].get());
    println!("   ∂loss/∂y = {:?}", y.grad()?.as_slice()[0].get());
    println!("   ∂loss/∂z = {:?}", z.grad()?.as_slice()[0].get());
    println!("   ∂loss/∂w = {:?}", w.grad()?.as_slice()[0].get());

    // Manual verification
    println!("\n5. Manual gradient verification:");
    println!("   loss = w² = (4*(x+y))² = 16*(x+y)²");
    println!("   ∂loss/∂x = ∂/∂x[16*(x+y)²] = 16*2*(x+y)*1 = 32*(x+y) = 32*(2+3) = 160");
    println!("   ∂loss/∂y = ∂/∂y[16*(x+y)²] = 16*2*(x+y)*1 = 32*(x+y) = 32*(2+3) = 160");

    // Multi-output backward pass
    println!("\n6. Multi-output backward pass:");
    let mut a = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0)],
        &[1],
    )?;
    a = a.requires_grad_(true);

    let mut two = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(2.0)],
        &[1],
    )?;
    two = two.requires_grad_(true);
    let b = &a * &two;

    let mut three = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(3.0)],
        &[1],
    )?;
    three = three.requires_grad_(true);
    let c = &a * &three;

    // Backward with respect to both outputs
    let b_grad = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0)],
        &[1],
    )?;
    let c_grad = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0)],
        &[1],
    )?;
    backward_with_grad(&b, &b_grad)?;
    backward_with_grad(&c, &c_grad)?;
    println!(
        "   Backward w.r.t. multiple outputs: b={}, c={}",
        b.as_slice()[0].get(),
        c.as_slice()[0].get()
    );
    println!("   ∂(b+c)/∂a = ∂/∂a[a*2 + a*3] = ∂/∂a[5a] = 5");
    println!(
        "   Computed: ∂b/∂a + ∂c/∂a = {} + {} = {}",
        a.grad()?.as_slice()[0].get(),
        a.grad()?.as_slice()[0].get(),
        a.grad()?.as_slice()[0].get() + a.grad()?.as_slice()[0].get()
    );

    // Higher-order operations
    println!("\n7. Higher-order operations:");
    let mut base = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(2.0)],
        &[1],
    )?;
    base = base.requires_grad_(true);

    let mut exp = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(3.0)],
        &[1],
    )?;
    exp = exp.requires_grad_(true);

    let product = &base * &exp;
    println!(
        "   product = base*exp = {}*{} = {}",
        base.as_slice()[0].get(),
        exp.as_slice()[0].get(),
        product.as_slice()[0].get()
    );

    let product_grad = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0)],
        &[1],
    )?;
    backward_with_grad(&product, &product_grad)?;
    println!("   ∂(x*y)/∂x|_{{x=2,y=3}} = y = 3");
    println!("   ∂(x*y)/∂y|_{{x=2,y=3}} = x = 2");
    println!(
        "   Computed: ∂product/∂base = {}, ∂product/∂exp = {}",
        base.grad()?.as_slice()[0].get(),
        exp.grad()?.as_slice()[0].get()
    );

    println!("\n✅ Automatic differentiation example completed!");
    println!("\n💡 Key takeaways:");
    println!("   • Reverse-mode AD with computation graph construction");
    println!("   • Gradient sharing across Variable clones");
    println!("   • Chain rule implementation for complex expressions");
    println!("   • Memory-safe gradient accumulation");
    println!("   • Support for arbitrary differentiable operations");

    io::stdout().flush()?;
    Ok(())
}
