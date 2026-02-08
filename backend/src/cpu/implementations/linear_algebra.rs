use crate::Result;
use dtype::{num_traits, DataType};
use storage::{DenseStorage, Storage};
use crate::cpu::linear_algebra;

pub fn matmul_dense<T: DataType>(
    lhs: &DenseStorage<T>,
    rhs: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let lhs_shape = lhs.shape().dims();
    let rhs_shape = rhs.shape().dims();

    if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
        return Err(crate::BackendError::InvalidInput(
            "Matrix multiplication requires 2D tensors".to_string(),
        ));
    }

    let (m, k) = (lhs_shape[0], lhs_shape[1]);
    let (k2, n) = (rhs_shape[0], rhs_shape[1]);

    if k != k2 {
        return Err(crate::BackendError::InvalidInput(
            "Matrix dimensions don't match for multiplication".to_string(),
        ));
    }

    let lhs_slice = lhs.as_slice();
    let rhs_slice = rhs.as_slice();
    let mut result = vec![T::default(); m * n];

    linear_algebra::matmul_primitive(lhs_slice, rhs_slice, &mut result, m, k, n)?;

    DenseStorage::from_vec(result, &[m, n])
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn addmm_dense<T: DataType>(
    input: &DenseStorage<T>,
    mat1: &DenseStorage<T>,
    mat2: &DenseStorage<T>,
    beta: T,
    alpha: T,
) -> Result<DenseStorage<T>>
where
    T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Copy + Default + PartialEq,
{
    // C = beta * input + alpha * (mat1 @ mat2)
    let m1_shape = mat1.shape().dims();
    let m2_shape = mat2.shape().dims();

    if m1_shape.len() != 2 || m2_shape.len() != 2 {
        return Err(crate::BackendError::InvalidInput(
            "Matrix multiplication requires 2D tensors".to_string(),
        ));
    }

    let (m, k) = (m1_shape[0], m1_shape[1]);
    let (k2, n) = (m2_shape[0], m2_shape[1]);

    if k != k2 {
        return Err(crate::BackendError::InvalidInput(
            "Matrix dimensions don't match for multiplication".to_string(),
        ));
    }
    
    // Result shape is [m, n]
    let result_len = m * n;
    
    // Check input shape and handle broadcasting if necessary.
    // For now, we require input to be broadcastable to [m, n].
    // If input is exactly [m, n], we can use it directly.
    // If input is smaller, we must expand it.
    // If beta is zero, we ignore input! 
    
    let zero = T::default();
    let mut result_data: Vec<T>;
    
    if beta == zero {
        // If beta is 0, we don't need input values.
        result_data = vec![zero; result_len];
    } else {
        // We need input values scaled by beta.
        // If input matches shape, copy it.
        // If mismatched, use fallback or error?
        // PyTorch broadcasts.
        // Simple case: input matches shape.
        let input_dims = input.shape().dims();
       
        if input_dims == &[m, n] {
             result_data = input.as_slice().to_vec();
        } else {
             // Handle simple broadcasting or return error
             // For strict correctness in MVP, we can error if not matching or handle common cases (like 1D bias).
             // If input is [n], and we add to [m, n] (matrix). 
             // Typically addmm does: beta*input + alpha*mm.
             // If input is [m, n], perfectly fine.
             // If input is [1], fine.
             // If input is [n], it broadcasts to [m, n] (adds to each row).
             
             // Expand input to result_data
             if input.shape().size() == 1 {
                 // scalar broadcast
                 let val = input.as_slice()[0];
                 result_data = vec![val; result_len];
             } else if input_dims.len() == 1 && input_dims[0] == n {
                 // Broadcast vector [n] to [m, n]
                 result_data = Vec::with_capacity(result_len);
                 let in_slice = input.as_slice();
                 for _ in 0..m {
                     result_data.extend_from_slice(in_slice);
                 }
             } else {
                 return Err(crate::BackendError::InvalidInput(
                    format!("Input shape {:?} not broadcastable to result shape {:?}", input_dims, [m, n])
                 ));
             }
        }
    }
    
    // Now call gemm_primitive. 
    // gemm computes: C = alpha * (A @ B) + beta * C.
    // We already put `input` data into `result_data`.
    // Effectively `C_init = inputs`.
    // We want `final = alpha * (A @ B) + beta * input`.
    // gemm computes `beta_param * C_init + alpha_param * (A @ B)`.
    // So we pass `beta` as `beta_param`.
    // `result_data` contains `input`.
    // So `beta * inputs + alpha * (A @ B)`. Correct.
    
    linear_algebra::gemm_primitive(
        alpha,
        mat1.as_slice(),
        mat2.as_slice(),
        beta,
        &mut result_data,
        m, k, n
    )?;

    DenseStorage::from_vec(result_data, &[m, n])
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn addmv_dense<T: DataType>(
    input: &DenseStorage<T>,
    mat: &DenseStorage<T>,
    vec: &DenseStorage<T>,
    beta: T,
    alpha: T,
) -> Result<DenseStorage<T>>
where
    T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Copy + Default + PartialEq,
{
    // mat: [n, m], vec: [m] (treated as [m, 1]), result: [n] (treated as [n, 1])
    // input must broadcast to [n].
    
    let mat_shape = mat.shape().dims();
    let vec_shape = vec.shape().dims();
    
    // Validate mat 2D
    if mat_shape.len() != 2 {
         return Err(crate::BackendError::InvalidInput("mat must be 2D".into()));
    }
    let (n, m) = (mat_shape[0], mat_shape[1]);
    
    // Validate vec 1D
    if vec_shape.len() != 1 {
         return Err(crate::BackendError::InvalidInput("vec must be 1D".into()));
    }
    if vec_shape[0] != m {
         return Err(crate::BackendError::InvalidInput(format!("vec size {} mismatch mat cols {}", vec_shape[0], m)));
    }
    
    // Result size n
    let result_len = n;
    let zero = T::default();
    let mut result_data: Vec<T>;
    
    if beta == zero {
        result_data = vec![zero; result_len];
    } else {
        // Broadcast input to [n]
        let input_dims = input.shape().dims();
        if input_dims == &[n] {
             result_data = input.as_slice().to_vec();
        } else if input.shape().size() == 1 {
             let val = input.as_slice()[0];
             result_data = vec![val; result_len];
        } else {
             return Err(crate::BackendError::InvalidInput(
                format!("Input shape {:?} not broadcastable to result shape {:?}", input_dims, [n])
             ));
        }
    }
    
    // Use gemm: C [n, 1] = alpha * (A [n, m] @ B [m, 1]) + beta * C [n, 1]
    linear_algebra::gemm_primitive(
        alpha,
        mat.as_slice(),
        vec.as_slice(),
        beta,
        &mut result_data,
        n, m, 1
    )?;
    
    DenseStorage::from_vec(result_data, &[n])
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn cholesky_dense<T: DataType + num_traits::Float>(
    input: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    let shape = input.shape().dims();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(crate::BackendError::InvalidInput(
            "Cholesky requires a square matrix".to_string(),
        ));
    }

    let n = shape[0];
    let mut result = vec![T::default(); n * n];
    linear_algebra::cholesky_decomposition_primitive(input.as_slice(), &mut result, n)?;

    DenseStorage::from_vec(result, &[n, n])
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
}

pub fn qr_dense<T: DataType + num_traits::Float>(
    input: &DenseStorage<T>,
) -> Result<(DenseStorage<T>, DenseStorage<T>)> {
    let shape = input.shape().dims();
    if shape.len() != 2 {
        return Err(crate::BackendError::InvalidInput(
            "QR requires a 2D matrix".to_string(),
        ));
    }

    let (m, n) = (shape[0], shape[1]);
    let mut q = vec![T::default(); m * m];
    let mut r = vec![T::default(); m * n];

    linear_algebra::qr_decomposition_primitive(input.as_slice(), &mut q, &mut r, m, n)?;

    let q_storage = DenseStorage::from_vec(q, &[m, m])
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))?;
    let r_storage = DenseStorage::from_vec(r, &[m, n])
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))?;

    Ok((q_storage, r_storage))
}

pub fn svd_dense<T: DataType + num_traits::Float>(
    input: &DenseStorage<T>,
) -> Result<(DenseStorage<T>, DenseStorage<T>, DenseStorage<T>)> {
    let shape = input.shape().dims();
    if shape.len() != 2 {
        return Err(crate::BackendError::InvalidInput(
            "SVD requires a 2D matrix".to_string(),
        ));
    }

    let (m, n) = (shape[0], shape[1]);
    let mut u = vec![T::default(); m * m];
    let mut s = vec![T::default(); m.min(n)];
    let mut vt = vec![T::default(); n * n];

    linear_algebra::svd_decomposition_primitive(
        input.as_slice(),
        &mut u,
        &mut s,
        &mut vt,
        m,
        n,
    )?;

    let u_storage = DenseStorage::from_vec(u, &[m, m])
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))?;
    // S is usually returned as a 1D vector of singular values
    let s_storage = DenseStorage::from_vec(s, &[m.min(n)])
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))?;
    let vt_storage = DenseStorage::from_vec(vt, &[n, n])
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))?;

    Ok((u_storage, s_storage, vt_storage))
}
