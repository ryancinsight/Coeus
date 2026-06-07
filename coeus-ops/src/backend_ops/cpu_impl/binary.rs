use coeus_core::{Scalar, Layout, Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use crate::ptr::{Ptr, MutPtr};
use crate::backend_ops::{BinaryOp, UnaryOp};
use crate::backend_ops::{compute_broadcast_offsets, compute_unary_offset};

// ── Binary operations monomorphization traits ──

pub trait BinaryKernelOp<T: Scalar> {
    fn apply(x: T, y: T) -> T;
}

pub struct AddOp;
impl<T: Scalar> BinaryKernelOp<T> for AddOp {
    #[inline(always)]
    fn apply(x: T, y: T) -> T { x + y }
}

pub struct SubOp;
impl<T: Scalar> BinaryKernelOp<T> for SubOp {
    #[inline(always)]
    fn apply(x: T, y: T) -> T { x - y }
}

pub struct MulOp;
impl<T: Scalar> BinaryKernelOp<T> for MulOp {
    #[inline(always)]
    fn apply(x: T, y: T) -> T { x * y }
}

pub struct DivOp;
impl<T: Scalar> BinaryKernelOp<T> for DivOp {
    #[inline(always)]
    fn apply(x: T, y: T) -> T { x / y }
}

// ── Unary operations monomorphization traits ──

pub trait UnaryKernelOp<T: Scalar> {
    fn apply(x: T) -> T;
}

pub struct ReluOp;
impl<T: Scalar> UnaryKernelOp<T> for ReluOp {
    #[inline(always)]
    fn apply(x: T) -> T {
        if x > T::zero() { x } else { T::zero() }
    }
}

pub struct ReluGradOp;
impl<T: Scalar> UnaryKernelOp<T> for ReluGradOp {
    #[inline(always)]
    fn apply(x: T) -> T {
        if x > T::zero() { T::one() } else { T::zero() }
    }
}

pub struct SigmoidOp;
impl<T: Scalar> UnaryKernelOp<T> for SigmoidOp {
    #[inline(always)]
    fn apply(x: T) -> T {
        x.sigmoid_op()
    }
}

pub struct SigmoidGradOp;
impl<T: Scalar> UnaryKernelOp<T> for SigmoidGradOp {
    #[inline(always)]
    fn apply(y: T) -> T {
        y * (T::one() - y)
    }
}

pub struct TanhOp;
impl<T: Scalar> UnaryKernelOp<T> for TanhOp {
    #[inline(always)]
    fn apply(x: T) -> T {
        x.tanh_op()
    }
}

pub struct TanhGradOp;
impl<T: Scalar> UnaryKernelOp<T> for TanhGradOp {
    #[inline(always)]
    fn apply(y: T) -> T {
        T::one() - y * y
    }
}

pub struct GeluOp;
impl<T: Scalar> UnaryKernelOp<T> for GeluOp {
    #[inline(always)]
    fn apply(x: T) -> T {
        x.gelu_op()
    }
}

pub struct GeluGradOp;
impl<T: Scalar> UnaryKernelOp<T> for GeluGradOp {
    #[inline(always)]
    fn apply(x: T) -> T {
        let half = T::from_f64(0.5);
        let one = T::one();
        let c1 = T::from_f64(0.7978845608);
        let c2 = T::from_f64(0.044715);
        let c3 = T::from_f64(0.134145);

        let x2 = x * x;
        let v = c1 * (x + c2 * x * x2);
        let t = v.tanh_op();
        let dy = c1 * (one + c3 * x2);
        half * (one + t) + half * x * (one - t * t) * dy
    }
}

pub struct SinOp;
impl<T: Scalar> UnaryKernelOp<T> for SinOp {
    #[inline(always)]
    fn apply(x: T) -> T {
        x.sin_op()
    }
}

pub struct CosOp;
impl<T: Scalar> UnaryKernelOp<T> for CosOp {
    #[inline(always)]
    fn apply(x: T) -> T {
        x.cos_op()
    }
}

pub struct ExpOp;
impl<T: Scalar> UnaryKernelOp<T> for ExpOp {
    #[inline(always)]
    fn apply(x: T) -> T {
        x.exp_op()
    }
}

pub struct LogOp;
impl<T: Scalar> UnaryKernelOp<T> for LogOp {
    #[inline(always)]
    fn apply(x: T) -> T {
        x.log_op()
    }
}

pub struct NegOp;
impl<T: Scalar> UnaryKernelOp<T> for NegOp {
    #[inline(always)]
    fn apply(x: T) -> T {
        T::zero() - x
    }
}

pub struct AbsOp;
impl<T: Scalar> UnaryKernelOp<T> for AbsOp {
    #[inline(always)]
    fn apply(x: T) -> T {
        x.abs_val()
    }
}

pub struct SqrtOp;
impl<T: Scalar> UnaryKernelOp<T> for SqrtOp {
    #[inline(always)]
    fn apply(x: T) -> T {
        x.sqrt_val()
    }
}

pub struct SiluOp;
impl<T: Scalar> UnaryKernelOp<T> for SiluOp {
    #[inline(always)]
    fn apply(x: T) -> T {
        x * x.sigmoid_op()
    }
}

pub struct SiluGradOp;
impl<T: Scalar> UnaryKernelOp<T> for SiluGradOp {
    #[inline(always)]
    fn apply(x: T) -> T {
        let s = x.sigmoid_op();
        s * (T::one() + x * (T::one() - s))
    }
}

pub struct MishOp;
impl<T: Scalar> UnaryKernelOp<T> for MishOp {
    #[inline(always)]
    fn apply(x: T) -> T {
        let sp = (T::one() + x.exp_op()).log_op();
        x * sp.tanh_op()
    }
}

pub struct MishGradOp;
impl<T: Scalar> UnaryKernelOp<T> for MishGradOp {
    #[inline(always)]
    fn apply(x: T) -> T {
        let sp = (T::one() + x.exp_op()).log_op();
        let w = sp.tanh_op();
        let sig = x.sigmoid_op();
        w + x * (T::one() - w * w) * sig
    }
}

pub struct EluOp;
impl<T: Scalar> UnaryKernelOp<T> for EluOp {
    #[inline(always)]
    fn apply(x: T) -> T {
        if x >= T::zero() { x } else { x.exp_op() - T::one() }
    }
}

pub struct EluGradOp;
impl<T: Scalar> UnaryKernelOp<T> for EluGradOp {
    #[inline(always)]
    fn apply(x: T) -> T {
        // Input x is the original input value (not ELU output)
        if x >= T::zero() { T::one() } else { x.exp_op() }
    }
}

pub struct SoftplusOp;
impl<T: Scalar> UnaryKernelOp<T> for SoftplusOp {
    #[inline(always)]
    fn apply(x: T) -> T {
        (T::one() + x.exp_op()).log_op()
    }
}

pub struct SoftplusGradOp;
impl<T: Scalar> UnaryKernelOp<T> for SoftplusGradOp {
    #[inline(always)]
    fn apply(x: T) -> T {
        x.sigmoid_op()
    }
}

pub struct GeluTanhOp;
impl<T: Scalar> UnaryKernelOp<T> for GeluTanhOp {
    #[inline(always)]
    fn apply(x: T) -> T {
        // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let c1 = T::from_f64(0.7978845608); // sqrt(2/pi)
        let c2 = T::from_f64(0.044715);
        let half = T::from_f64(0.5);
        let one = T::one();
        let v = c1 * (x + c2 * x * x * x);
        half * x * (one + v.tanh_op())
    }
}

pub struct GeluTanhGradOp;
impl<T: Scalar> UnaryKernelOp<T> for GeluTanhGradOp {
    #[inline(always)]
    fn apply(x: T) -> T {
        let c1 = T::from_f64(0.7978845608);
        let c2 = T::from_f64(0.044715);
        let c3 = T::from_f64(0.134145); // 3 * 0.044715
        let half = T::from_f64(0.5);
        let one = T::one();
        let v = c1 * (x + c2 * x * x * x);
        let t = v.tanh_op();
        let dt = c1 * (one + c3 * x * x); // d/dx of the argument to tanh
        half * (one + t) + half * x * (one - t * t) * dt
    }
}


// ── Optimized generic execution runners ──

#[inline(always)]
fn run_binary_op<T: Scalar, B: Backend, O: BinaryKernelOp<T>>(
    backend: &B,
    a: &B::DeviceBuffer<T>,
    a_layout: &Layout,
    b: &B::DeviceBuffer<T>,
    b_layout: &Layout,
    c: &mut B::DeviceBuffer<T>,
    c_layout: &Layout,
) where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    let a_slice = a.as_slice();
    let b_slice = b.as_slice();
    let c_slice = c.as_mut_slice();

    let out_shape = c_layout.shape();
    let out_numel = out_shape.iter().product::<usize>();

    let a_ptr = Ptr(a_slice.as_ptr());
    let b_ptr = Ptr(b_slice.as_ptr());
    let c_ptr = MutPtr(c_slice.as_mut_ptr());

    let a_off = a_layout.offset();
    let b_off = b_layout.offset();
    let c_off = c_layout.offset();

    if a_layout.shape() == b_layout.shape()
        && a_layout.is_contiguous()
        && b_layout.is_contiguous()
        && c_layout.is_contiguous()
    {
        // Contiguous fast path: monomorphized static loop
        let a_off = a_layout.offset();
        let b_off = b_layout.offset();
        let c_off = c_layout.offset();
        backend.parallel_for(0, out_numel, move |i| unsafe {
            // SAFETY: The raw pointers point to valid contiguous device memory buffers of size out_numel.
            // Loop index i is in [0, out_numel), which is within safe bounds.
            c_ptr.write(c_off + i, O::apply(a_ptr.read(a_off + i), b_ptr.read(b_off + i)));
        });
        return;
    }

    // Strided broadcasting path
    let out_strides = c_layout.strides_cloned();
    let out_shape_vec = c_layout.shape_cloned();

    let a_shape_v = a_layout.shape_cloned();
    let b_shape_v = b_layout.shape_cloned();
    let a_strides_v = a_layout.strides_cloned();
    let b_strides_v = b_layout.strides_cloned();

    if c_layout.is_contiguous() {
        backend.parallel_for(0, out_numel, move |i| {
            let (off_a, off_b) = compute_broadcast_offsets(
                i,
                &out_shape_vec,
                &out_strides,
                &a_shape_v,
                &a_strides_v,
                a_off,
                &b_shape_v,
                &b_strides_v,
                b_off,
            );

            unsafe {
                // SAFETY: The raw pointers point to valid storage, and compute_broadcast_offsets
                // calculates the correct, in-bounds physical offsets according to broadcasting rules.
                c_ptr.write(c_off + i, O::apply(a_ptr.read(off_a), b_ptr.read(off_b)));
            }
        });
    } else {
        let c_strides_v = c_layout.strides_cloned();
        backend.parallel_for(0, out_numel, move |i| {
            let (off_a, off_b) = compute_broadcast_offsets(
                i,
                &out_shape_vec,
                &out_strides,
                &a_shape_v,
                &a_strides_v,
                a_off,
                &b_shape_v,
                &b_strides_v,
                b_off,
            );
            let off_c = compute_unary_offset(
                i,
                &out_strides,
                &out_shape_vec,
                &c_strides_v,
                c_off,
            );

            unsafe {
                c_ptr.write(off_c, O::apply(a_ptr.read(off_a), b_ptr.read(off_b)));
            }
        });
    }
}

#[inline(always)]
fn run_unary_op<T: Scalar, B: Backend, O: UnaryKernelOp<T>>(
    backend: &B,
    a: &B::DeviceBuffer<T>,
    a_layout: &Layout,
    c: &mut B::DeviceBuffer<T>,
    c_layout: &Layout,
) where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    let a_slice = a.as_slice();
    let c_slice = c.as_mut_slice();

    let out_shape = c_layout.shape();
    let out_numel = out_shape.iter().product::<usize>();

    let a_ptr = Ptr(a_slice.as_ptr());
    let c_ptr = MutPtr(c_slice.as_mut_ptr());

    let a_off = a_layout.offset();
    let c_off = c_layout.offset();

    if a_layout.is_contiguous() && c_layout.is_contiguous() {
        backend.parallel_for(0, out_numel, move |i| unsafe {
            // SAFETY: The raw pointers point to valid contiguous buffers of size out_numel,
            // and the loop index i is within safe bounds.
            let val = a_ptr.read(a_off + i);
            c_ptr.write(c_off + i, O::apply(val));
        });
    } else {
        let in_strides = a_layout.strides_cloned();
        let in_shape = a_layout.shape_cloned();
        let out_strides = c_layout.strides_cloned();

        if c_layout.is_contiguous() {
            backend.parallel_for(0, out_numel, move |i| {
                let physical_index = compute_unary_offset(
                    i,
                    &out_strides,
                    &in_shape,
                    &in_strides,
                    a_off,
                );
                unsafe {
                    let val = a_ptr.read(physical_index);
                    c_ptr.write(c_off + i, O::apply(val));
                }
            });
        } else {
            let c_strides_v = c_layout.strides_cloned();
            let out_shape_vec = c_layout.shape_cloned();
            backend.parallel_for(0, out_numel, move |i| {
                let physical_index = compute_unary_offset(
                    i,
                    &out_strides,
                    &in_shape,
                    &in_strides,
                    a_off,
                );
                let physical_out = compute_unary_offset(
                    i,
                    &out_strides,
                    &out_shape_vec,
                    &c_strides_v,
                    c_off,
                );
                unsafe {
                    let val = a_ptr.read(physical_index);
                    c_ptr.write(physical_out, O::apply(val));
                }
            });
        }
    }
}

// ── Public elements routing ──

#[inline]
pub(crate) fn elementwise_binary<T: Scalar, B: Backend>(
    backend: &B,
    op: BinaryOp,
    a: &B::DeviceBuffer<T>,
    a_layout: &Layout,
    b: &B::DeviceBuffer<T>,
    b_layout: &Layout,
    c: &mut B::DeviceBuffer<T>,
    c_layout: &Layout,
) where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    match op {
        BinaryOp::Add => run_binary_op::<T, B, AddOp>(backend, a, a_layout, b, b_layout, c, c_layout),
        BinaryOp::Sub => run_binary_op::<T, B, SubOp>(backend, a, a_layout, b, b_layout, c, c_layout),
        BinaryOp::Mul => run_binary_op::<T, B, MulOp>(backend, a, a_layout, b, b_layout, c, c_layout),
        BinaryOp::Div => run_binary_op::<T, B, DivOp>(backend, a, a_layout, b, b_layout, c, c_layout),
    }
}

#[inline]
pub(crate) fn elementwise_unary<T: Scalar, B: Backend>(
    backend: &B,
    op: UnaryOp,
    a: &B::DeviceBuffer<T>,
    a_layout: &Layout,
    c: &mut B::DeviceBuffer<T>,
    c_layout: &Layout,
) where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    match op {
        UnaryOp::Relu => run_unary_op::<T, B, ReluOp>(backend, a, a_layout, c, c_layout),
        UnaryOp::ReluGrad => run_unary_op::<T, B, ReluGradOp>(backend, a, a_layout, c, c_layout),
        UnaryOp::Sigmoid => run_unary_op::<T, B, SigmoidOp>(backend, a, a_layout, c, c_layout),
        UnaryOp::SigmoidGrad => run_unary_op::<T, B, SigmoidGradOp>(backend, a, a_layout, c, c_layout),
        UnaryOp::Tanh => run_unary_op::<T, B, TanhOp>(backend, a, a_layout, c, c_layout),
        UnaryOp::TanhGrad => run_unary_op::<T, B, TanhGradOp>(backend, a, a_layout, c, c_layout),
        UnaryOp::Gelu => run_unary_op::<T, B, GeluOp>(backend, a, a_layout, c, c_layout),
        UnaryOp::GeluGrad => run_unary_op::<T, B, GeluGradOp>(backend, a, a_layout, c, c_layout),
        UnaryOp::Sin => run_unary_op::<T, B, SinOp>(backend, a, a_layout, c, c_layout),
        UnaryOp::Cos => run_unary_op::<T, B, CosOp>(backend, a, a_layout, c, c_layout),
        UnaryOp::Exp => run_unary_op::<T, B, ExpOp>(backend, a, a_layout, c, c_layout),
        UnaryOp::Log => run_unary_op::<T, B, LogOp>(backend, a, a_layout, c, c_layout),
        UnaryOp::Neg => run_unary_op::<T, B, NegOp>(backend, a, a_layout, c, c_layout),
        UnaryOp::Abs => run_unary_op::<T, B, AbsOp>(backend, a, a_layout, c, c_layout),
        UnaryOp::Sqrt => run_unary_op::<T, B, SqrtOp>(backend, a, a_layout, c, c_layout),
        UnaryOp::Silu => run_unary_op::<T, B, SiluOp>(backend, a, a_layout, c, c_layout),
        UnaryOp::SiluGrad => run_unary_op::<T, B, SiluGradOp>(backend, a, a_layout, c, c_layout),
        UnaryOp::Mish => run_unary_op::<T, B, MishOp>(backend, a, a_layout, c, c_layout),
        UnaryOp::MishGrad => run_unary_op::<T, B, MishGradOp>(backend, a, a_layout, c, c_layout),
        UnaryOp::Elu => run_unary_op::<T, B, EluOp>(backend, a, a_layout, c, c_layout),
        UnaryOp::EluGrad => run_unary_op::<T, B, EluGradOp>(backend, a, a_layout, c, c_layout),
        UnaryOp::Softplus => run_unary_op::<T, B, SoftplusOp>(backend, a, a_layout, c, c_layout),
        UnaryOp::SoftplusGrad => run_unary_op::<T, B, SoftplusGradOp>(backend, a, a_layout, c, c_layout),
        UnaryOp::GeluTanh => run_unary_op::<T, B, GeluTanhOp>(backend, a, a_layout, c, c_layout),
        UnaryOp::GeluTanhGrad => run_unary_op::<T, B, GeluTanhGradOp>(backend, a, a_layout, c, c_layout),
        UnaryOp::LeakyRelu(slope_bits) => {
            let slope = T::from_f64(f64::from_bits(slope_bits));
            let a_slice = a.as_slice();
            let c_slice = c.as_mut_slice();
            let numel = c_layout.shape().iter().product::<usize>();
            let a_off = a_layout.offset();
            let c_off = c_layout.offset();
            let a_ptr = Ptr(a_slice.as_ptr());
            let c_ptr = MutPtr(c_slice.as_mut_ptr());
            if a_layout.is_contiguous() && c_layout.is_contiguous() {
                backend.parallel_for(0, numel, move |i| unsafe {
                    let x = a_ptr.read(a_off + i);
                    let y = if x >= T::zero() { x } else { slope * x };
                    c_ptr.write(c_off + i, y);
                });
            } else {
                let in_strides = a_layout.strides_cloned();
                let in_shape = a_layout.shape_cloned();
                let out_strides = c_layout.strides_cloned();
                let out_shape = c_layout.shape_cloned();
                backend.parallel_for(0, numel, move |i| {
                    let physical_in = compute_unary_offset(i, &out_strides, &in_shape, &in_strides, a_off);
                    let physical_out = compute_unary_offset(i, &out_strides, &out_shape, &out_strides, c_off);
                    unsafe {
                        let x = a_ptr.read(physical_in);
                        let y = if x >= T::zero() { x } else { slope * x };
                        c_ptr.write(physical_out, y);
                    }
                });
            }
        },
        UnaryOp::LeakyReluGrad(slope_bits) => {
            let slope = T::from_f64(f64::from_bits(slope_bits));
            let a_slice = a.as_slice();
            let c_slice = c.as_mut_slice();
            let numel = c_layout.shape().iter().product::<usize>();
            let a_off = a_layout.offset();
            let c_off = c_layout.offset();
            let a_ptr = Ptr(a_slice.as_ptr());
            let c_ptr = MutPtr(c_slice.as_mut_ptr());
            if a_layout.is_contiguous() && c_layout.is_contiguous() {
                backend.parallel_for(0, numel, move |i| unsafe {
                    let x = a_ptr.read(a_off + i);
                    let y = if x >= T::zero() { T::one() } else { slope };
                    c_ptr.write(c_off + i, y);
                });
            } else {
                let in_strides = a_layout.strides_cloned();
                let in_shape = a_layout.shape_cloned();
                let out_strides = c_layout.strides_cloned();
                let out_shape = c_layout.shape_cloned();
                backend.parallel_for(0, numel, move |i| {
                    let physical_in = compute_unary_offset(i, &out_strides, &in_shape, &in_strides, a_off);
                    let physical_out = compute_unary_offset(i, &out_strides, &out_shape, &out_strides, c_off);
                    unsafe {
                        let x = a_ptr.read(physical_in);
                        let y = if x >= T::zero() { T::one() } else { slope };
                        c_ptr.write(physical_out, y);
                    }
                });
            }
        },
    }
}
