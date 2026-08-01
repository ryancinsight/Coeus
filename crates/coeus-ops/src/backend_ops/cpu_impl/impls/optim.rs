use super::super::CpuBackend;
use crate::backend_ops::cpu_impl::error::map_leto_error;
use crate::backend_ops::traits::OptimizerOps;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Layout, Scalar};
use coeus_leto::{
    stateful_update, ReadOperand, StatefulUpdateOperands, StatefulUpdateState, WriteOperand,
};
use leto_ops::{
    AdaGrad, AdaGradParameters, Adam, AdamParameters, AdamW, AdamWParameters, RmsProp,
    RmsPropParameters, Sgd, SgdParameters,
};

fn read<'a, T>(layout: &'a Layout, data: &'a [T]) -> ReadOperand<'a, T> {
    ReadOperand { layout, data }
}

fn write<'a, T>(layout: &'a Layout, data: &'a mut [T]) -> WriteOperand<'a, T> {
    WriteOperand { layout, data }
}

fn one_state<'a, T>(
    param: WriteOperand<'a, T>,
    grad: ReadOperand<'a, T>,
    state: WriteOperand<'a, T>,
) -> StatefulUpdateOperands<'a, T> {
    StatefulUpdateOperands {
        parameter: param,
        gradient: grad,
        state: StatefulUpdateState::One(state),
    }
}

fn two_states<'a, T>(
    param: WriteOperand<'a, T>,
    grad: ReadOperand<'a, T>,
    first: WriteOperand<'a, T>,
    second: WriteOperand<'a, T>,
) -> StatefulUpdateOperands<'a, T> {
    StatefulUpdateOperands {
        parameter: param,
        gradient: grad,
        state: StatefulUpdateState::Two(first, second),
    }
}

fn step_number(operation: &'static str, step: usize) -> Result<u64, coeus_core::BackendError> {
    u64::try_from(step).map_err(|_| coeus_core::BackendError::Overflow {
        operation,
        reason: "optimizer step exceeds u64 range",
    })
}

impl<T: Scalar + leto_ops::RealScalar, B: CpuBackend> OptimizerOps<T> for B
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    fn sgd_step(
        &self,
        param: &mut Self::DeviceBuffer<T>,
        param_layout: &Layout,
        grad: &Self::DeviceBuffer<T>,
        grad_layout: &Layout,
        velocity: &mut Self::DeviceBuffer<T>,
        velocity_layout: &Layout,
        lr: T,
        momentum: T,
    ) -> Result<(), Self::Error> {
        let parameters =
            SgdParameters::new(lr, momentum).map_err(|error| map_leto_error("SGD step", error))?;
        stateful_update::<T, Sgd>(
            one_state(
                write(param_layout, param.as_mut_slice()),
                read(grad_layout, grad.as_slice()),
                write(velocity_layout, velocity.as_mut_slice()),
            ),
            parameters,
        )
        .map_err(|error| map_leto_error("SGD step", error))
    }

    fn adam_step(
        &self,
        param: &mut Self::DeviceBuffer<T>,
        param_layout: &Layout,
        grad: &Self::DeviceBuffer<T>,
        grad_layout: &Layout,
        m: &mut Self::DeviceBuffer<T>,
        m_layout: &Layout,
        v: &mut Self::DeviceBuffer<T>,
        v_layout: &Layout,
        lr: T,
        beta1: T,
        beta2: T,
        eps: T,
        t: usize,
    ) -> Result<(), Self::Error> {
        let parameters = AdamParameters::new(lr, beta1, beta2, eps, step_number("Adam step", t)?)
            .map_err(|error| map_leto_error("Adam step", error))?;
        stateful_update::<T, Adam>(
            two_states(
                write(param_layout, param.as_mut_slice()),
                read(grad_layout, grad.as_slice()),
                write(m_layout, m.as_mut_slice()),
                write(v_layout, v.as_mut_slice()),
            ),
            parameters,
        )
        .map_err(|error| map_leto_error("Adam step", error))
    }

    fn rmsprop_step(
        &self,
        param: &mut Self::DeviceBuffer<T>,
        param_layout: &Layout,
        grad: &Self::DeviceBuffer<T>,
        grad_layout: &Layout,
        v: &mut Self::DeviceBuffer<T>,
        v_layout: &Layout,
        lr: T,
        alpha: T,
        eps: T,
    ) -> Result<(), Self::Error> {
        let parameters = RmsPropParameters::new(lr, alpha, eps)
            .map_err(|error| map_leto_error("RMSProp step", error))?;
        stateful_update::<T, RmsProp>(
            one_state(
                write(param_layout, param.as_mut_slice()),
                read(grad_layout, grad.as_slice()),
                write(v_layout, v.as_mut_slice()),
            ),
            parameters,
        )
        .map_err(|error| map_leto_error("RMSProp step", error))
    }

    fn adamw_step(
        &self,
        param: &mut Self::DeviceBuffer<T>,
        param_layout: &Layout,
        grad: &Self::DeviceBuffer<T>,
        grad_layout: &Layout,
        m: &mut Self::DeviceBuffer<T>,
        m_layout: &Layout,
        v: &mut Self::DeviceBuffer<T>,
        v_layout: &Layout,
        lr: T,
        beta1: T,
        beta2: T,
        eps: T,
        weight_decay: T,
        t: usize,
    ) -> Result<(), Self::Error> {
        let parameters = AdamWParameters::new(
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step_number("AdamW step", t)?,
        )
        .map_err(|error| map_leto_error("AdamW step", error))?;
        stateful_update::<T, AdamW>(
            two_states(
                write(param_layout, param.as_mut_slice()),
                read(grad_layout, grad.as_slice()),
                write(m_layout, m.as_mut_slice()),
                write(v_layout, v.as_mut_slice()),
            ),
            parameters,
        )
        .map_err(|error| map_leto_error("AdamW step", error))
    }

    fn adagrad_step(
        &self,
        param: &mut Self::DeviceBuffer<T>,
        param_layout: &Layout,
        grad: &Self::DeviceBuffer<T>,
        grad_layout: &Layout,
        history: &mut Self::DeviceBuffer<T>,
        history_layout: &Layout,
        lr: T,
        eps: T,
    ) -> Result<(), Self::Error> {
        let parameters = AdaGradParameters::new(lr, eps)
            .map_err(|error| map_leto_error("AdaGrad step", error))?;
        stateful_update::<T, AdaGrad>(
            one_state(
                write(param_layout, param.as_mut_slice()),
                read(grad_layout, grad.as_slice()),
                write(history_layout, history.as_mut_slice()),
            ),
            parameters,
        )
        .map_err(|error| map_leto_error("AdaGrad step", error))
    }
}

#[cfg(test)]
mod tests {
    use coeus_core::{CpuAddressableStorage, CpuStorage, Shape};

    use super::*;

    fn storage(values: &[f32]) -> CpuStorage<f32> {
        CpuStorage::from_slice(values)
    }

    fn layout() -> Layout {
        Layout::new(Shape::from(vec![2]))
    }

    fn assert_unchanged(storage: &CpuStorage<f32>, expected: &[f32]) {
        assert_eq!(storage.as_slice(), expected);
    }

    #[test]
    fn invalid_parameters_leave_storage_unchanged() {
        let backend = coeus_core::SequentialBackend;
        let layout = layout();
        let gradient = storage(&[0.25, -0.5]);

        let mut parameter = storage(&[2.0, 3.0]);
        let mut velocity = storage(&[0.1, 0.2]);
        backend
            .sgd_step(
                &mut parameter,
                &layout,
                &gradient,
                &layout,
                &mut velocity,
                &layout,
                -0.1,
                0.9,
            )
            .expect_err("negative SGD learning rate must fail before mutation");
        assert_unchanged(&parameter, &[2.0, 3.0]);
        assert_unchanged(&velocity, &[0.1, 0.2]);

        let mut first = storage(&[0.1, 0.2]);
        let mut second = storage(&[0.3, 0.4]);
        backend
            .adam_step(
                &mut parameter,
                &layout,
                &gradient,
                &layout,
                &mut first,
                &layout,
                &mut second,
                &layout,
                0.1,
                0.9,
                0.999,
                1.0e-8,
                0,
            )
            .expect_err("zero Adam step must fail before mutation");
        assert_unchanged(&parameter, &[2.0, 3.0]);
        assert_unchanged(&first, &[0.1, 0.2]);
        assert_unchanged(&second, &[0.3, 0.4]);

        let mut average = storage(&[0.1, 0.2]);
        backend
            .rmsprop_step(
                &mut parameter,
                &layout,
                &gradient,
                &layout,
                &mut average,
                &layout,
                0.1,
                0.99,
                0.0,
            )
            .expect_err("zero RMSProp epsilon must fail before mutation");
        assert_unchanged(&parameter, &[2.0, 3.0]);
        assert_unchanged(&average, &[0.1, 0.2]);

        backend
            .adamw_step(
                &mut parameter,
                &layout,
                &gradient,
                &layout,
                &mut first,
                &layout,
                &mut second,
                &layout,
                0.1,
                0.9,
                0.999,
                1.0e-8,
                -0.1,
                1,
            )
            .expect_err("negative AdamW weight decay must fail before mutation");
        assert_unchanged(&parameter, &[2.0, 3.0]);
        assert_unchanged(&first, &[0.1, 0.2]);
        assert_unchanged(&second, &[0.3, 0.4]);

        let mut history = storage(&[0.1, 0.2]);
        backend
            .adagrad_step(
                &mut parameter,
                &layout,
                &gradient,
                &layout,
                &mut history,
                &layout,
                0.1,
                0.0,
            )
            .expect_err("zero AdaGrad epsilon must fail before mutation");
        assert_unchanged(&parameter, &[2.0, 3.0]);
        assert_unchanged(&history, &[0.1, 0.2]);
    }

    #[test]
    fn rank_eight_dispatches_through_leto() {
        let backend = coeus_core::SequentialBackend;
        let layout = Layout::new(Shape::from(vec![1; 8]));
        let gradient = storage(&[1.0]);
        let mut parameter = storage(&[2.0]);
        let mut velocity = storage(&[0.0]);

        backend
            .sgd_step(
                &mut parameter,
                &layout,
                &gradient,
                &layout,
                &mut velocity,
                &layout,
                0.1,
                0.0,
            )
            .expect("rank-eight Leto dispatch");

        assert_eq!(parameter.as_slice(), &[1.9]);
        assert_eq!(velocity.as_slice(), &[1.0]);
    }
}
