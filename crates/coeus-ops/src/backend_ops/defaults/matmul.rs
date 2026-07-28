use crate::backend_ops::ops::BinaryOp;
use crate::backend_ops::traits::{ElementwiseOps, MatmulOps};
use coeus_core::{BackendError, Layout, Scalar, Shape, Strides};

fn shape3<B: MatmulOps<T>, T: Scalar>(shape: &[usize]) -> Result<[usize; 3], B::Error> {
    let [a, b, c] = shape else {
        return Err(B::Error::from(BackendError::UnsupportedRank {
            operation: "batched_matmul",
            rank: shape.len(),
            max_rank: 3,
        }));
    };
    Ok([*a, *b, *c])
}

/// Default: `c += a @ b` via temp + add.
pub fn matmul_accumulate<T: Scalar, B: MatmulOps<T> + ElementwiseOps<T>>(
    backend: &B,
    a: &B::DeviceBuffer<T>,
    a_layout: &Layout,
    b: &B::DeviceBuffer<T>,
    b_layout: &Layout,
    c: &mut B::DeviceBuffer<T>,
    c_layout: &Layout,
) -> Result<(), B::Error> {
    let temp_len = c_layout.shape().iter().product();
    let mut temp = backend.allocate::<T>(temp_len)?;
    let temp_layout =
        Layout::from_shape_strides(c_layout.shape_cloned(), c_layout.strides_cloned(), 0);
    backend.fill(&mut temp, T::zero())?;
    backend.matmul(a, a_layout, b, b_layout, &mut temp, &temp_layout)?;
    let c_ptr = c as *mut B::DeviceBuffer<T>;
    // SAFETY: `c_ptr` originates from the unique `&mut c` argument. The
    // temporary buffer is distinct, so the two storage references do not
    // alias during the backend elementwise update.
    unsafe {
        backend.elementwise_binary(
            BinaryOp::Add,
            &*c_ptr,
            c_layout,
            &temp,
            &temp_layout,
            &mut *c_ptr,
            c_layout,
        )?;
    }
    Ok(())
}

/// Default: rank-3 batched matmul via per-slice rank-2 dispatch.
pub fn batched_matmul<T: Scalar, B: MatmulOps<T>>(
    backend: &B,
    a: &B::DeviceBuffer<T>,
    a_layout: &Layout,
    b: &B::DeviceBuffer<T>,
    b_layout: &Layout,
    c: &mut B::DeviceBuffer<T>,
    c_layout: &Layout,
) -> Result<(), B::Error> {
    let [lhs_batch, m, lhs_k] = shape3::<B, T>(a_layout.shape())?;
    let [rhs_batch, rhs_k, n] = shape3::<B, T>(b_layout.shape())?;
    let [out_batch, out_m, out_n] = shape3::<B, T>(c_layout.shape())?;
    if !((lhs_batch == out_batch || lhs_batch == 1)
        && (rhs_batch == out_batch || rhs_batch == 1)
        && lhs_k == rhs_k
        && m == out_m
        && n == out_n)
    {
        return Err(B::Error::from(BackendError::ShapeMismatch {
            operation: "batched_matmul",
            lhs: a_layout.shape().to_vec(),
            rhs: b_layout.shape().to_vec(),
        }));
    }

    let lhs_batch_stride = if lhs_batch == 1 {
        0
    } else {
        a_layout.strides()[0]
    };
    let rhs_batch_stride = if rhs_batch == 1 {
        0
    } else {
        b_layout.strides()[0]
    };
    let out_batch_stride = c_layout.strides()[0];

    let lhs_shape = Shape::from([m, lhs_k].as_slice());
    let rhs_shape = Shape::from([rhs_k, n].as_slice());
    let out_shape = Shape::from([out_m, out_n].as_slice());
    let lhs_strides = Strides::from([a_layout.strides()[1], a_layout.strides()[2]].as_slice());
    let rhs_strides = Strides::from([b_layout.strides()[1], b_layout.strides()[2]].as_slice());
    let out_strides = Strides::from([c_layout.strides()[1], c_layout.strides()[2]].as_slice());

    for batch in 0..out_batch {
        let lhs_layout = Layout::from_shape_strides(
            lhs_shape.clone(),
            lhs_strides.clone(),
            a_layout.offset() + batch * lhs_batch_stride,
        );
        let rhs_layout = Layout::from_shape_strides(
            rhs_shape.clone(),
            rhs_strides.clone(),
            b_layout.offset() + batch * rhs_batch_stride,
        );
        let out_layout = Layout::from_shape_strides(
            out_shape.clone(),
            out_strides.clone(),
            c_layout.offset() + batch * out_batch_stride,
        );
        backend.matmul(a, &lhs_layout, b, &rhs_layout, c, &out_layout)?;
    }
    Ok(())
}

/// Default: `c += batched a @ b` via temp + add.
pub fn batched_matmul_accumulate<T: Scalar, B: MatmulOps<T> + ElementwiseOps<T>>(
    backend: &B,
    a: &B::DeviceBuffer<T>,
    a_layout: &Layout,
    b: &B::DeviceBuffer<T>,
    b_layout: &Layout,
    c: &mut B::DeviceBuffer<T>,
    c_layout: &Layout,
) -> Result<(), B::Error> {
    let temp_len = c_layout.shape().iter().product();
    let mut temp = backend.allocate::<T>(temp_len)?;
    let temp_layout =
        Layout::from_shape_strides(c_layout.shape_cloned(), c_layout.strides_cloned(), 0);
    backend.fill(&mut temp, T::zero())?;
    batched_matmul(backend, a, a_layout, b, b_layout, &mut temp, &temp_layout)?;
    let c_ptr = c as *mut B::DeviceBuffer<T>;
    // SAFETY: `c_ptr` originates from the unique `&mut c` argument. The
    // temporary buffer is distinct, so the two storage references do not
    // alias during the backend elementwise update.
    unsafe {
        backend.elementwise_binary(
            BinaryOp::Add,
            &*c_ptr,
            c_layout,
            &temp,
            &temp_layout,
            &mut *c_ptr,
            c_layout,
        )?;
    }
    Ok(())
}
