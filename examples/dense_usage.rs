//! Dense Crate Usage Examples
//! 
//! This example demonstrates how to use the new dense crate for:
//! - Dense tensor operations
//! - Broadcasting operations
//! - Linear algebra operations
//! - Statistical operations
//! - Integration with storage primitives

use coeus_dense::{
    ops::{
        elementwise::{add, mul, exp, log},
        linear_algebra::{matmul, transpose, inverse},
        statistical::{sum, mean, max, min, std_dev},
        comparison::{eq, gt, lt},
        trigonometric::{sin, cos, tan},
    },
    algorithms::{
        broadcasting::{broadcast_shapes, broadcast_add},
        matrix::cache_friendly_matmul,
    },
    utils::{
        memory_utils::{is_contiguous, ensure_contiguous},
        shape_utils::validate_shapes,
    },
};
use coeus_storage::{DenseStorage, StorageFromVec};
use coeus_dtype::float::Float32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Coeus Dense Crate Examples ===\n");

    // Example 1: Basic Dense Operations
    basic_dense_operations_example()?;
    
    // Example 2: Broadcasting Operations
    broadcasting_example()?;
    
    // Example 3: Linear Algebra Operations
    linear_algebra_example()?;
    
    // Example 4: Statistical Operations
    statistical_operations_example()?;
    
    // Example 5: Comparison Operations
    comparison_operations_example()?;
    
    // Example 6: Trigonometric Functions
    trigonometric_functions_example()?;
    
    // Example 7: Memory Layout Optimization
    memory_optimization_example()?;
    
    // Example 8: Cache-Friendly Algorithms
    cache_friendly_example()?;

    println!("All dense crate examples completed successfully!");
    Ok(())
}

fn basic_dense_operations_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Basic Dense Operations");
    println!("=========================");
    
    // Create dense storage tensors
    let a = DenseStorage::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let b = DenseStorage::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;
    
    println!("Tensor A: {:?}", a.as_slice());
    println!("Tensor B: {:?}", b.as_slice());
    println!("Shape: {:?}", a.shape().dims());
    
    // Element-wise operations
    let sum = add(&a, &b)?;
    println!("A + B = {:?}", sum.as_slice());
    
    let product = mul(&a, &b)?;
    println!("A * B = {:?}", product.as_slice());
    
    // Mathematical functions
    let exp_a = exp(&a)?;
    println!("exp(A) = {:?}", exp_a.as_slice());
    
    let log_a = log(&a)?;
    println!("log(A) = {:?}", log_a.as_slice());
    
    println!();
    Ok(())
}

fn broadcasting_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Broadcasting Operations");
    println!("==========================");
    
    // Create tensors with different shapes for broadcasting
    let a = DenseStorage::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
    let b = DenseStorage::from_vec(vec![10.0, 20.0, 30.0], &[3])?;
    
    println!("Tensor A shape: {:?}", a.shape().dims());
    println!("Tensor A: {:?}", a.as_slice());
    println!("Tensor B shape: {:?}", b.shape().dims());
    println!("Tensor B: {:?}", b.as_slice());
    
    // Check if shapes are broadcastable
    let result_shape = broadcast_shapes(&[2, 3], &[3])?;
    println!("Broadcast result shape: {:?}", result_shape);
    
    // Perform broadcasted addition
    let result = broadcast_add(&a, &b)?;
    println!("Broadcasted A + B = {:?}", result.as_slice());
    
    // More complex broadcasting example
    let c = DenseStorage::from_vec(vec![1.0, 2.0], &[2, 1])?;
    let d = DenseStorage::from_vec(vec![10.0, 20.0, 30.0], &[1, 3])?;
    
    println!("\nComplex broadcasting:");
    println!("Tensor C shape: {:?}", c.shape().dims());
    println!("Tensor D shape: {:?}", d.shape().dims());
    
    let complex_result_shape = broadcast_shapes(&[2, 1], &[1, 3])?;
    println!("Complex broadcast result shape: {:?}", complex_result_shape);
    
    let complex_result = broadcast_add(&c, &d)?;
    println!("Complex broadcasted result: {:?}", complex_result.as_slice());
    
    println!();
    Ok(())
}

fn linear_algebra_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Linear Algebra Operations");
    println!("============================");
    
    // Create matrices for linear algebra operations
    let a = DenseStorage::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let b = DenseStorage::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;
    
    println!("Matrix A:");
    print_matrix(&a, 2, 2);
    println!("Matrix B:");
    print_matrix(&b, 2, 2);
    
    // Matrix multiplication
    let c = matmul(&a, &b)?;
    println!("A @ B (matrix multiplication):");
    print_matrix(&c, 2, 2);
    
    // Matrix transpose
    let a_t = transpose(&a, &[1, 0])?;
    println!("A^T (transpose):");
    print_matrix(&a_t, 2, 2);
    
    // Matrix inverse (for square matrices)
    let a_inv = inverse(&a)?;
    println!("A^(-1) (inverse):");
    print_matrix(&a_inv, 2, 2);
    
    // Verify A * A^(-1) = I
    let identity_check = matmul(&a, &a_inv)?;
    println!("A * A^(-1) (should be identity):");
    print_matrix(&identity_check, 2, 2);
    
    println!();
    Ok(())
}

fn statistical_operations_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Statistical Operations");
    println!("=========================");
    
    let data = DenseStorage::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 
        &[3, 4]
    )?;
    
    println!("Data matrix (3x4):");
    print_matrix(&data, 3, 4);
    
    // Reduce along all dimensions
    let total_sum = sum(&data, None)?;
    println!("Total sum: {:?}", total_sum.as_slice());
    
    let mean_value = mean(&data, None)?;
    println!("Mean value: {:?}", mean_value.as_slice());
    
    let max_value = max(&data, None)?;
    println!("Max value: {:?}", max_value.as_slice());
    
    let min_value = min(&data, None)?;
    println!("Min value: {:?}", min_value.as_slice());
    
    // Reduce along specific dimensions
    let row_sums = sum(&data, Some(1))?;  // Sum along columns (axis 1)
    println!("Row sums (sum along columns): {:?}", row_sums.as_slice());
    
    let col_means = mean(&data, Some(0))?; // Mean along rows (axis 0)
    println!("Column means (mean along rows): {:?}", col_means.as_slice());
    
    // Standard deviation
    let std_deviation = std_dev(&data, None, 0)?; // ddof=0 for population std
    println!("Standard deviation: {:?}", std_deviation.as_slice());
    
    println!();
    Ok(())
}

fn comparison_operations_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Comparison Operations");
    println!("========================");
    
    let a = DenseStorage::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4])?;
    let b = DenseStorage::from_vec(vec![1.0, 3.0, 2.0, 5.0], &[4])?;
    
    println!("Tensor A: {:?}", a.as_slice());
    println!("Tensor B: {:?}", b.as_slice());
    
    // Element-wise comparisons (return boolean tensors)
    let equal = eq(&a, &b)?;
    println!("A == B: {:?}", equal.as_slice());
    
    let greater = gt(&a, &b)?;
    println!("A > B:  {:?}", greater.as_slice());
    
    let less = lt(&a, &b)?;
    println!("A < B:  {:?}", less.as_slice());
    
    // Comparison with scalar (broadcasted)
    let scalar = DenseStorage::from_vec(vec![2.5], &[1])?;
    let greater_than_scalar = gt(&a, &scalar)?;
    println!("A > 2.5: {:?}", greater_than_scalar.as_slice());
    
    println!();
    Ok(())
}

fn trigonometric_functions_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("6. Trigonometric Functions");
    println!("==========================");
    
    // Create angles in radians
    let angles = DenseStorage::from_vec(
        vec![0.0, std::f32::consts::PI / 6.0, std::f32::consts::PI / 4.0, 
             std::f32::consts::PI / 3.0, std::f32::consts::PI / 2.0], 
        &[5]
    )?;
    
    println!("Angles (radians): {:?}", angles.as_slice());
    
    // Trigonometric functions
    let sin_values = sin(&angles)?;
    println!("sin(angles): {:?}", sin_values.as_slice());
    
    let cos_values = cos(&angles)?;
    println!("cos(angles): {:?}", cos_values.as_slice());
    
    let tan_values = tan(&angles)?;
    println!("tan(angles): {:?}", tan_values.as_slice());
    
    // Verify trigonometric identity: sin²(x) + cos²(x) = 1
    let sin_squared = mul(&sin_values, &sin_values)?;
    let cos_squared = mul(&cos_values, &cos_values)?;
    let identity_check = add(&sin_squared, &cos_squared)?;
    println!("sin²(x) + cos²(x) = {:?} (should be ~1.0)", identity_check.as_slice());
    
    println!();
    Ok(())
}

fn memory_optimization_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("7. Memory Layout Optimization");
    println!("=============================");
    
    // Create a tensor with non-contiguous memory layout (e.g., after transpose)
    let original = DenseStorage::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
    println!("Original tensor shape: {:?}", original.shape().dims());
    println!("Original is contiguous: {}", is_contiguous(&original));
    
    // Transpose creates non-contiguous layout
    let transposed = transpose(&original, &[1, 0])?;
    println!("Transposed shape: {:?}", transposed.shape().dims());
    println!("Transposed is contiguous: {}", is_contiguous(&transposed));
    
    // For optimal performance, ensure contiguous layout
    let contiguous = ensure_contiguous(&transposed)?;
    println!("After ensuring contiguous: {}", is_contiguous(&contiguous));
    
    // Demonstrate performance difference (conceptual)
    println!("Contiguous memory layout provides:");
    println!("  - Better cache locality");
    println!("  - SIMD optimization opportunities");
    println!("  - Faster element access");
    
    println!();
    Ok(())
}

fn cache_friendly_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("8. Cache-Friendly Algorithms");
    println!("============================");
    
    // Create larger matrices for cache-friendly demonstration
    let a = DenseStorage::from_vec((0..64).map(|i| i as f32).collect(), &[8, 8])?;
    let b = DenseStorage::from_vec((0..64).map(|i| (i * 2) as f32).collect(), &[8, 8])?;
    
    println!("Matrix A (8x8): first few elements {:?}...", &a.as_slice()[0..4]);
    println!("Matrix B (8x8): first few elements {:?}...", &b.as_slice()[0..4]);
    
    // Standard matrix multiplication
    let standard_result = matmul(&a, &b)?;
    println!("Standard matmul result (first few): {:?}...", &standard_result.as_slice()[0..4]);
    
    // Cache-friendly matrix multiplication with blocking
    let block_size = 4; // Smaller block size for demonstration
    let cache_friendly_result = cache_friendly_matmul(&a, &b, block_size)?;
    println!("Cache-friendly matmul result (first few): {:?}...", &cache_friendly_result.as_slice()[0..4]);
    
    // Verify results are identical
    let diff = sub(&standard_result, &cache_friendly_result)?;
    let max_diff = max(&diff, None)?;
    println!("Max difference between methods: {:?}", max_diff.as_slice());
    
    println!("Cache-friendly algorithms provide:");
    println!("  - Better memory access patterns");
    println!("  - Reduced cache misses");
    println!("  - Improved performance on large matrices");
    
    println!();
    Ok(())
}

// Helper function to print matrices in a readable format
fn print_matrix(storage: &DenseStorage<Float32>, rows: usize, cols: usize) {
    let data = storage.as_slice();
    for i in 0..rows {
        print!("  [");
        for j in 0..cols {
            print!("{:6.2}", data[i * cols + j]);
            if j < cols - 1 {
                print!(", ");
            }
        }
        println!("]");
    }
}

// Mock implementations for demonstration (actual implementations would be in dense crate)
use coeus_dense::ops::elementwise::sub;

// Additional mock functions that would be implemented in the actual dense crate
impl DenseStorage<Float32> {
    // Mock shape method
    pub fn shape(&self) -> &Shape {
        // Mock implementation
        &Shape::new(&[]).unwrap()
    }
}

// Mock Shape type
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: &[usize]) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Shape { dims: dims.to_vec() })
    }
    
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
}