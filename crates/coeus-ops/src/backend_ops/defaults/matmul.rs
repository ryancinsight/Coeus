use crate::backend_ops::ops::BinaryOp;
use crate::backend_ops::traits::{ElementwiseOps, MatmulOps};
use coeus_core::{Layout, Scalar, Shape, Strides};

fn shape3(shape: &[usize], name: &str) -> [usize; 3] {
    assert_eq!(
        shape.len(),
        3,
        "batched_matmul: {name} shape must have rank 3"
    );
    [shape[0], shape[1], shape[2]]
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
    let mut temp = backend.allocate_zeroed::<T>(temp_len);
    let temp_layout =
        Layout::from_shape_strides(c_layout.shape_cloned(), c_layout.strides_cloned(), 0);
    backend.matmul(a, a_layout, b, b_layout, &mut temp, &temp_layout)?;
    let c_ptr = c as *mut B::DeviceBuffer<T>;
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
    assert_eq!(a_layout.ndim(), 3, "batched_matmul: lhs must be rank 3");
    assert_eq!(b_layout.ndim(), 3, "batched_matmul: rhs must be rank 3");
    assert_eq!(c_layout.ndim(), 3, "batched_matmul: out must be rank 3");

    let [lhs_batch, m, lhs_k] = shape3(a_layout.shape(), "lhs");
    let [rhs_batch, rhs_k, n] = shape3(b_layout.shape(), "rhs");
    let [out_batch, out_m, out_n] = shape3(c_layout.shape(), "out");
    assert!(
        (lhs_batch == out_batch || lhs_batch == 1)
            && (rhs_batch == out_batch || rhs_batch == 1)
            && lhs_k == rhs_k
            && m == out_m
            && n == out_n,
        "batched_matmul: incompatible shapes {:?}, {:?}, {:?}",
        a_layout.shape(),
        b_layout.shape(),
        c_layout.shape(),
    );

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
    let mut temp = backend.allocate_zeroed::<T>(temp_len);
    let temp_layout =
        Layout::from_shape_strides(c_layout.shape_cloned(), c_layout.strides_cloned(), 0);
    batched_matmul(backend, a, a_layout, b, b_layout, &mut temp, &temp_layout)?;
    let c_ptr = c as *mut B::DeviceBuffer<T>;
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
