//! Executable example demonstrating automatic differentiation

use coeus_tensor::Tensor;

pub fn run_autograd_demo() {
    println!("🧠 Automatic Differentiation Demo");
    println!("==================================");

    // Simple gradient computation
    println!("\n📈 Simple Gradient Computation:");
    println!("-------------------------------");

    // f(x) = x², f'(x) = 2x
    let mut x = Tensor::scalar(3.0);
    x.set_requires_grad(true);

    let y = (&x * &x).unwrap();
    y.backward().unwrap();

    println!("f(x) = x²");
    println!("x = {:?}", x.as_scalar());
    println!("f(x) = {:?}", y.as_scalar());
    println!("f'(x) = ∂f/∂x = {:?}", x.grad().unwrap().as_scalar());
    println!("Expected: 2 * 3 = 6 ✓");

    // Chain rule example
    println!("\n⛓️  Chain Rule Example:");
    println!("----------------------");

    // f(x) = sin(e^x), f'(x) = cos(e^x) * e^x
    let mut x2 = Tensor::scalar(0.0);
    x2.set_requires_grad(true);

    let exp_x = x2.exp();
    let sin_exp_x = exp_x.sin();
    sin_exp_x.backward().unwrap();

    let x_val: f64 = x2.as_scalar().unwrap();
    let expected_grad = (x_val.exp()).cos() * x_val.exp();
    println!("f(x) = sin(e^x)");
    println!("x = {:?}", x2.as_scalar().unwrap());
    println!("f(x) = {:?}", sin_exp_x.as_scalar().unwrap());
    println!("f'(x) = {:?}", x2.grad().unwrap().as_scalar().unwrap());
    println!("Expected: cos(e^x) * e^x = {:.6} ✓", expected_grad);

    // Multi-variable function
    println!("\n🔢 Multi-Variable Gradients:");
    println!("----------------------------");

    // f(a,b) = a² * b + sin(a), ∇f = [2*a*b + cos(a), a²]
    let mut a = Tensor::scalar(2.0);
    let mut b = Tensor::scalar(3.0);
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    let a_squared = (&a * &a).unwrap();
    let a_squared_b = (&a_squared * &b).unwrap();
    let sin_a = a.sin();
    let result = (&a_squared_b + &sin_a).unwrap();

    result.backward().unwrap();

    let a_val: f64 = a.as_scalar().unwrap();
    let b_val: f64 = b.as_scalar().unwrap();
    let expected_da = 2.0 * a_val * b_val + a_val.cos();
    let expected_db = a_val * a_val;

    println!("f(a,b) = a² * b + sin(a)");
    println!(
        "a = {:?}, b = {:?}",
        a.as_scalar().unwrap(),
        b.as_scalar().unwrap()
    );
    println!("f(a,b) = {:?}", result.as_scalar().unwrap());
    println!("∂f/∂a = {:?}", a.grad().unwrap().as_scalar().unwrap());
    println!("∂f/∂b = {:?}", b.grad().unwrap().as_scalar().unwrap());
    println!("Expected ∂f/∂a = 2*a*b + cos(a) = {:.6} ✓", expected_da);
    println!("Expected ∂f/∂b = a² = {:.6} ✓", expected_db);

    // Matrix operations with gradients
    println!("\n🔄 Matrix Operations with Gradients:");
    println!("------------------------------------");

    let mut m1 = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let mut m2 = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    m1.set_requires_grad(true);
    m2.set_requires_grad(true);

    let matrix_prod = m1.matmul(&m2).unwrap();
    let matrix_sum = matrix_prod.sum();
    matrix_sum.backward().unwrap();

    println!("M1 = [[1, 2], [3, 4]]");
    println!("M2 = [[5, 6], [7, 8]]");
    println!("M1 @ M2 = {:?}", matrix_prod.data());
    println!("sum(M1 @ M2) = {:?}", matrix_sum.as_scalar());
    println!("∂sum(M1 @ M2)/∂M1 = {:?}", m1.grad().unwrap().data());
    println!("∂sum(M1 @ M2)/∂M2 = {:?}", m2.grad().unwrap().data());

    println!("\n✅ Automatic differentiation demo completed!");
}
