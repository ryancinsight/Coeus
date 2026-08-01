use super::super::CpuBackend;
use crate::backend_ops::cpu_impl::error::map_leto_error;
use crate::backend_ops::traits::{
    OptimizerOps, OptimizerStateRef, OptimizerStepRule, OptimizerStepValidation,
};
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Layout, Scalar};
use coeus_leto::{
    stateful_update, validate_stateful_update, ReadOperand, StatefulUpdateOperands,
    StatefulUpdateState, StatefulUpdateValidation, StatefulUpdateValidationState, WriteOperand,
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

fn validate_one<T>(
    param: ReadOperand<'_, T>,
    grad: ReadOperand<'_, T>,
    state: ReadOperand<'_, T>,
) -> leto::Result<()> {
    validate_stateful_update(StatefulUpdateValidation {
        parameter: param,
        gradient: grad,
        state: StatefulUpdateValidationState::One(state),
    })
}

fn validate_two<T>(
    param: ReadOperand<'_, T>,
    grad: ReadOperand<'_, T>,
    first: ReadOperand<'_, T>,
    second: ReadOperand<'_, T>,
) -> leto::Result<()> {
    validate_stateful_update(StatefulUpdateValidation {
        parameter: param,
        gradient: grad,
        state: StatefulUpdateValidationState::Two(first, second),
    })
}

fn step_number(operation: &'static str, step: usize) -> Result<u64, coeus_core::BackendError> {
    i32::try_from(step)
        .map(|step| step as u64)
        .map_err(|_| coeus_core::BackendError::Overflow {
            operation,
            reason: "optimizer step exceeds i32::MAX provider limit",
        })
}

impl<T: Scalar + leto_ops::RealScalar, B: CpuBackend> OptimizerOps<T> for B
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    fn validate_optimizer_step(
        &self,
        validation: OptimizerStepValidation<'_, T, Self>,
    ) -> Result<(), Self::Error> {
        let (operation, expected_states) = match validation.rule {
            OptimizerStepRule::Sgd {
                learning_rate,
                momentum,
            } => {
                SgdParameters::new(learning_rate, momentum)
                    .map_err(|error| map_leto_error("SGD step", error))?;
                ("SGD step", 1)
            }
            OptimizerStepRule::Adam {
                learning_rate,
                beta_one,
                beta_two,
                epsilon,
                step,
            } => {
                AdamParameters::new(
                    learning_rate,
                    beta_one,
                    beta_two,
                    epsilon,
                    step_number("Adam step", step)?,
                )
                .map_err(|error| map_leto_error("Adam step", error))?;
                ("Adam step", 2)
            }
            OptimizerStepRule::RmsProp {
                learning_rate,
                alpha,
                epsilon,
            } => {
                RmsPropParameters::new(learning_rate, alpha, epsilon)
                    .map_err(|error| map_leto_error("RMSProp step", error))?;
                ("RMSProp step", 1)
            }
            OptimizerStepRule::AdamW {
                learning_rate,
                beta_one,
                beta_two,
                epsilon,
                weight_decay,
                step,
            } => {
                AdamWParameters::new(
                    learning_rate,
                    beta_one,
                    beta_two,
                    epsilon,
                    weight_decay,
                    step_number("AdamW step", step)?,
                )
                .map_err(|error| map_leto_error("AdamW step", error))?;
                ("AdamW step", 2)
            }
            OptimizerStepRule::AdaGrad {
                learning_rate,
                epsilon,
            } => {
                AdaGradParameters::new(learning_rate, epsilon)
                    .map_err(|error| map_leto_error("AdaGrad step", error))?;
                ("AdaGrad step", 1)
            }
        };
        let (parameter, parameter_layout) = validation.parameter;
        let (gradient, gradient_layout) = validation.gradient;
        let result = match (expected_states, validation.state) {
            (1, OptimizerStateRef::One(state, state_layout)) => validate_one(
                read(parameter_layout, parameter.as_slice()),
                read(gradient_layout, gradient.as_slice()),
                read(state_layout, state.as_slice()),
            ),
            (2, OptimizerStateRef::Two(first, first_layout, second, second_layout)) => {
                validate_two(
                    read(parameter_layout, parameter.as_slice()),
                    read(gradient_layout, gradient.as_slice()),
                    read(first_layout, first.as_slice()),
                    read(second_layout, second.as_slice()),
                )
            }
            _ => Err(leto::LetoError::InvalidInput(
                "optimizer rule and persistent state cardinality disagree".to_string(),
            )),
        };
        result.map_err(|error| map_leto_error(operation, error))
    }

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
    fn ranks_zero_through_eight_dispatch_through_leto() {
        let backend = coeus_core::SequentialBackend;
        for rank in 0..=8 {
            let layout = Layout::new(Shape::from(vec![1; rank]));
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
                .unwrap_or_else(|error| panic!("rank-{rank} Leto dispatch failed: {error}"));

            assert_eq!(parameter.as_slice(), &[1.9], "rank {rank}");
            assert_eq!(velocity.as_slice(), &[1.0], "rank {rank}");
        }
    }

    #[test]
    fn adam_step_rejects_values_beyond_accelerator_domain() {
        let backend = coeus_core::SequentialBackend;
        let layout = Layout::new(Shape::from(vec![1]));
        let gradient = storage(&[1.0]);
        let parameter = storage(&[2.0]);
        let first = storage(&[0.0]);
        let second = storage(&[0.0]);

        backend
            .validate_optimizer_step(OptimizerStepValidation {
                parameter: (&parameter, &layout),
                gradient: (&gradient, &layout),
                state: OptimizerStateRef::Two(&first, &layout, &second, &layout),
                rule: OptimizerStepRule::Adam {
                    learning_rate: 0.1,
                    beta_one: 0.9,
                    beta_two: 0.999,
                    epsilon: 1.0e-8,
                    step: i32::MAX as usize + 1,
                },
            })
            .expect_err("CPU Adam domain must match accelerator i32 step limit");
        assert_unchanged(&parameter, &[2.0]);
        assert_unchanged(&first, &[0.0]);
        assert_unchanged(&second, &[0.0]);
    }
}
