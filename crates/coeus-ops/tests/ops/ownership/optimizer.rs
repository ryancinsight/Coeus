use coeus_core::{ComputeBackend, Float, Layout, Scalar};
use coeus_ops::OptimizerOps;

pub(crate) fn upload<T: Scalar, B: ComputeBackend>(
    backend: &B,
    values: &[T],
) -> B::DeviceBuffer<T> {
    let mut buffer = backend.allocate(values.len());
    backend.copy_to_device(values, &mut buffer);
    buffer
}

pub(crate) fn assert_values<T: Scalar, B: ComputeBackend>(
    backend: &B,
    buffer: &B::DeviceBuffer<T>,
    expected: &[T],
) {
    let mut actual = vec![T::zero(); expected.len()];
    backend.copy_to_host(buffer, &mut actual);
    assert_eq!(
        actual,
        expected,
        "{} / {}",
        core::any::type_name::<B>(),
        core::any::type_name::<T>()
    );
}

pub(crate) fn adam_preserves_all_state_clones<T: Float, B: OptimizerOps<T>>(backend: &B, one: T) {
    let _span = tracing::info_span!("adam_state_ownership").entered();
    let two = one + one;
    let three = two + one;
    let four = two + two;
    let five = four + one;
    let half = one / two;
    let quarter = half / two;
    let layout = Layout::new([2].into());
    let parameter = upload(backend, &[four, five]);
    let first = upload(backend, &[three, T::zero() - three]);
    let second = upload(backend, &[five, five]);
    let gradient = upload(backend, &[one, T::zero() - one]);
    let mut parameter_write = parameter.clone();
    let mut first_write = first.clone();
    let mut second_write = second.clone();
    // The last state exceeds its allocation; earlier states must remain intact.
    let invalid_second_layout = Layout::new([3].into());
    backend
        .adam_step(
            &mut parameter_write,
            &layout,
            &gradient,
            &layout,
            &mut first_write,
            &layout,
            &mut second_write,
            &invalid_second_layout,
            half,
            half,
            three / four,
            four,
            1,
        )
        .expect_err("the second moment layout exceeds its two-element allocation");
    assert_values(backend, &parameter, &[four, five]);
    assert_values(backend, &parameter_write, &[four, five]);
    assert_values(backend, &first, &[three, T::zero() - three]);
    assert_values(backend, &first_write, &[three, T::zero() - three]);
    assert_values(backend, &second, &[five, five]);
    assert_values(backend, &second_write, &[five, five]);
    assert_values(backend, &gradient, &[one, T::zero() - one]);

    parameter_write = parameter.clone();
    first_write = first.clone();
    second_write = second.clone();

    // beta1=1/2, beta2=3/4, step=1 yield m'=±2, v'=4,
    // corrected moments ±4 and 16, and update (1/2)*±4/(4+4)=±1/4.
    // All intermediate values are exactly representable in every real scalar.
    backend
        .adam_step(
            &mut parameter_write,
            &layout,
            &gradient,
            &layout,
            &mut first_write,
            &layout,
            &mut second_write,
            &layout,
            half,
            half,
            three / four,
            four,
            1,
        )
        .expect("valid two-element Adam update");
    assert_values(backend, &parameter, &[four, five]);
    assert_values(backend, &first, &[three, T::zero() - three]);
    assert_values(backend, &second, &[five, five]);
    assert_values(backend, &gradient, &[one, T::zero() - one]);
    assert_values(backend, &parameter_write, &[four - quarter, five + quarter]);
    assert_values(backend, &first_write, &[two, T::zero() - two]);
    assert_values(backend, &second_write, &[four, four]);

    // All three writable snapshots may initially share one allocation.
    // m'=v'=2, corrected moments 4 and 4 give update (1/2)*4/(2+2)=1/2.
    let common = upload(backend, &[three, three]);
    let mut parameter_write = common.clone();
    let mut first_write = common.clone();
    let mut second_write = common.clone();
    let gradient = upload(backend, &[one, one]);
    backend
        .adam_step(
            &mut parameter_write,
            &layout,
            &gradient,
            &layout,
            &mut first_write,
            &layout,
            &mut second_write,
            &layout,
            half,
            half,
            half,
            two,
            1,
        )
        .expect("valid mutually shared Adam destinations");
    assert_values(backend, &common, &[three, three]);
    assert_values(backend, &parameter_write, &[three - half, three - half]);
    assert_values(backend, &first_write, &[two, two]);
    assert_values(backend, &second_write, &[two, two]);
}
