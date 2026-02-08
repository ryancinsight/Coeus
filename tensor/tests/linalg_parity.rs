use tensor::{Tensor, Result};
use backend::CpuBackend;
use dtype::float::Float32;
use storage::DenseStorage;
use tensor::*;

#[test]
fn test_addr_parity() -> Result<()> {
    // addr: beta * input + alpha * (vec1 @ vec2)
    // Let input = [1.0, 1.0]
    //           [1.0, 1.0]
    // vec1 = [2.0, 3.0]
    // vec2 = [4.0, 5.0]
    // alpha = 2.0, beta = 0.5
    // outer = [8.0, 10.0]
    //         [12.0, 15.0]
    // term2 = alpha * outer = [16.0, 20.0]
    //                        [24.0, 30.0]
    // term1 = beta * input = [0.5, 0.5]
    //                       [0.5, 0.5]
    // result = [16.5, 20.5]
    //          [24.5, 30.5]

    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[2, 2]).unwrap();
    let vec1 = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32(2.0), Float32(3.0)], 
        &[2]
    ).unwrap();
    let vec2 = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32(4.0), Float32(5.0)], 
        &[2]
    ).unwrap();
    
    let res = tensor::ops::linalg::addr(&input, &vec1, &vec2, Float32(0.5), Float32(2.0)).unwrap();
    
    let data = res.as_slice();
    assert!((data[0].0 - 16.5).abs() < 1e-5);
    assert!((data[1].0 - 20.5).abs() < 1e-5);
    assert!((data[2].0 - 24.5).abs() < 1e-5);
    assert!((data[3].0 - 30.5).abs() < 1e-5);
    
    Ok(())
}

#[test]
fn test_outer_parity() -> Result<()> {
    // outer: vec1 @ vec2
    let vec1 = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32(2.0), Float32(3.0)], 
        &[2]
    ).unwrap();
    let vec2 = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32(4.0), Float32(5.0)], 
        &[2]
    ).unwrap();
    
    let res = tensor::ops::linalg::outer(&vec1, &vec2).unwrap();
    
    // [8, 10]
    // [12, 15]
    let data = res.as_slice();
    assert!((data[0].0 - 8.0).abs() < 1e-5);
    assert!((data[1].0 - 10.0).abs() < 1e-5);
    assert!((data[2].0 - 12.0).abs() < 1e-5);
    assert!((data[3].0 - 15.0).abs() < 1e-5);
    
    Ok(())
}

#[test]
fn test_addmm_parity() -> Result<()> {
    // addmm: beta * input + alpha * (mat1 @ mat2)
    // mat1 2x2: [[1, 2], [3, 4]]
    // mat2 2x2: [[1, 0], [0, 1]] (Identity) -> mat1 @ mat2 = mat1
    // input 2x2: [[1, 1], [1, 1]]
    // beta = 0.5, alpha = 2.0
    // Result = 0.5 * input + 2.0 * mat1
    // = [[0.5, 0.5], [0.5, 0.5]] + [[2, 4], [6, 8]]
    // = [[2.5, 4.5], [6.5, 8.5]]
    
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[2, 2]).unwrap();
    let mat1 = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32(1.0), Float32(2.0), Float32(3.0), Float32(4.0)],
        &[2, 2]
    ).unwrap();
    // Identity matrix: [1, 0, 0, 1]
    let mat2 = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32(1.0), Float32(0.0), Float32(0.0), Float32(1.0)],
        &[2, 2]
    ).unwrap();

    let res = tensor::ops::linalg::addmm(&input, &mat1, &mat2, Float32(0.5), Float32(2.0)).unwrap();
    
    let data = res.as_slice();
    // [2.5, 4.5]
    // [6.5, 8.5]
    assert!((data[0].0 - 2.5).abs() < 1e-5);
    assert!((data[1].0 - 4.5).abs() < 1e-5);
    assert!((data[2].0 - 6.5).abs() < 1e-5);
    assert!((data[3].0 - 8.5).abs() < 1e-5);
    
    Ok(())
}
