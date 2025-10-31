use backend::{cpu::CpuBackend, Backend};
use dtype::float::Float32;
use storage::{DenseStorage, Storage};

// Example demonstrating compute_squares functionality
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Coeus Compute Squares Example");
    println!("=============================");

    // Create CPU backend
    let backend = CpuBackend::<Float32>::new();

    // Create input tensor with values
    let mut data = vec![];
    for i in 0..10 {
        data.push(Float32::new(i as f32));
    }

    let input = DenseStorage::from_vec(data, &[10])?;
    println!("Input tensor: {:?}", input.as_slice().iter().map(|v| v.get()).collect::<Vec<f32>>());

    // Compute squares using backend
    let result = backend.compute_squares(&input)?;
    println!("Squares result: {:?}", result.as_slice().iter().map(|v| v.get()).collect::<Vec<f32>>());

    // Verify results
    for (i, val) in result.as_slice().iter().enumerate() {
        let expected = (i as f32) * (i as f32);
        assert!((val.get() - expected).abs() < 1e-6, "Result mismatch at index {}", i);
    }

    println!("✓ All results verified - compute_squares working correctly!");

    Ok(())
}

