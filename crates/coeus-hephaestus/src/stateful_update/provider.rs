use crate::HephaestusProvider;
use coeus_core::{backend::ComputeBackend, Layout, Scalar};
use coeus_ops::{OptimizerStateRef, OptimizerStepRule, OptimizerStepValidation};
use hephaestus_core::{
    AdaGrad, AdaGradParameters, AdaGradParametersF64, Adam, AdamParameters, AdamParametersF64,
    AdamW, AdamWParameters, AdamWParametersF64, ComputeDevice, HephaestusError, RmsProp,
    RmsPropParameters, RmsPropParametersF64, Sgd, SgdParameters, SgdParametersF64,
    StatefulUpdateOps,
};

/// Provider-owned stateful-update operation marker.
pub trait StatefulUpdateProvider: HephaestusProvider {
    /// Monomorphized Hephaestus operation selected by this provider.
    type Operations: StatefulUpdateOps<Self::Device> + Default;
}

/// Projects one Coeus backend into its provider-owned stateful-update path.
pub trait StatefulUpdateBackend: ComputeBackend {
    /// Hephaestus provider selected by this Coeus backend.
    type Provider: StatefulUpdateProvider;

    #[doc(hidden)]
    fn stateful_update_buffer<T: Scalar>(
        storage: &Self::DeviceBuffer<T>,
    ) -> &<<Self::Provider as HephaestusProvider>::Device as ComputeDevice>::Buffer<T>;

    #[doc(hidden)]
    fn stateful_update_error(operation: &'static str, source: HephaestusError) -> Self::Error;

    #[doc(hidden)]
    fn validate_optimizer_step(
        &self,
        validation: OptimizerStepValidation<'_, f32, Self>,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        let (parameter, parameter_layout) = validation.parameter;
        let (gradient, gradient_layout) = validation.gradient;
        match (validation.rule, validation.state) {
            (
                OptimizerStepRule::Sgd {
                    learning_rate,
                    momentum,
                },
                OptimizerStateRef::One(state, state_layout),
            ) => {
                SgdParameters::new(learning_rate, momentum)
                    .map_err(|source| Self::stateful_update_error("SGD step", source))?;
                super::dispatch::validate_one::<Self, Sgd, f32>(
                    "SGD step",
                    parameter,
                    parameter_layout,
                    gradient,
                    gradient_layout,
                    state,
                    state_layout,
                )
            }
            (
                OptimizerStepRule::Adam {
                    learning_rate,
                    beta_one,
                    beta_two,
                    epsilon,
                    step,
                },
                OptimizerStateRef::Two(first, first_layout, second, second_layout),
            ) => {
                AdamParameters::new(learning_rate, beta_one, beta_two, epsilon, step)
                    .map_err(|source| Self::stateful_update_error("Adam step", source))?;
                super::dispatch::validate_two::<Self, Adam, f32>(
                    "Adam step",
                    parameter,
                    parameter_layout,
                    gradient,
                    gradient_layout,
                    first,
                    first_layout,
                    second,
                    second_layout,
                )
            }
            (
                OptimizerStepRule::RmsProp {
                    learning_rate,
                    alpha,
                    epsilon,
                },
                OptimizerStateRef::One(state, state_layout),
            ) => {
                RmsPropParameters::new(learning_rate, alpha, epsilon)
                    .map_err(|source| Self::stateful_update_error("RMSProp step", source))?;
                super::dispatch::validate_one::<Self, RmsProp, f32>(
                    "RMSProp step",
                    parameter,
                    parameter_layout,
                    gradient,
                    gradient_layout,
                    state,
                    state_layout,
                )
            }
            (
                OptimizerStepRule::AdamW {
                    learning_rate,
                    beta_one,
                    beta_two,
                    epsilon,
                    weight_decay,
                    step,
                },
                OptimizerStateRef::Two(first, first_layout, second, second_layout),
            ) => {
                AdamWParameters::new(
                    learning_rate,
                    beta_one,
                    beta_two,
                    epsilon,
                    weight_decay,
                    step,
                )
                .map_err(|source| Self::stateful_update_error("AdamW step", source))?;
                super::dispatch::validate_two::<Self, AdamW, f32>(
                    "AdamW step",
                    parameter,
                    parameter_layout,
                    gradient,
                    gradient_layout,
                    first,
                    first_layout,
                    second,
                    second_layout,
                )
            }
            (
                OptimizerStepRule::AdaGrad {
                    learning_rate,
                    epsilon,
                },
                OptimizerStateRef::One(state, state_layout),
            ) => {
                AdaGradParameters::new(learning_rate, epsilon)
                    .map_err(|source| Self::stateful_update_error("AdaGrad step", source))?;
                super::dispatch::validate_one::<Self, AdaGrad, f32>(
                    "AdaGrad step",
                    parameter,
                    parameter_layout,
                    gradient,
                    gradient_layout,
                    state,
                    state_layout,
                )
            }
            _ => Err(Self::stateful_update_error(
                "optimizer step preflight",
                HephaestusError::InvalidConfiguration {
                    message: "optimizer rule and persistent state cardinality disagree".to_string(),
                },
            )),
        }
    }

    #[doc(hidden)]
    #[expect(clippy::too_many_arguments, reason = "mirrors OptimizerOps")]
    fn dispatch_sgd_step(
        &self,
        parameter: &mut Self::DeviceBuffer<f32>,
        parameter_layout: &Layout,
        gradient: &Self::DeviceBuffer<f32>,
        gradient_layout: &Layout,
        velocity: &mut Self::DeviceBuffer<f32>,
        velocity_layout: &Layout,
        learning_rate: f32,
        momentum: f32,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        let parameters = SgdParameters::new(learning_rate, momentum)
            .map_err(|source| Self::stateful_update_error("SGD step", source))?;
        super::dispatch::one::<Self, Sgd, f32>(
            "SGD step",
            parameter,
            parameter_layout,
            gradient,
            gradient_layout,
            velocity,
            velocity_layout,
            parameters,
        )
    }

    #[doc(hidden)]
    #[expect(clippy::too_many_arguments, reason = "mirrors OptimizerOps")]
    fn dispatch_adam_step(
        &self,
        parameter: &mut Self::DeviceBuffer<f32>,
        parameter_layout: &Layout,
        gradient: &Self::DeviceBuffer<f32>,
        gradient_layout: &Layout,
        first: &mut Self::DeviceBuffer<f32>,
        first_layout: &Layout,
        second: &mut Self::DeviceBuffer<f32>,
        second_layout: &Layout,
        learning_rate: f32,
        beta_one: f32,
        beta_two: f32,
        epsilon: f32,
        step: usize,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        let parameters = AdamParameters::new(learning_rate, beta_one, beta_two, epsilon, step)
            .map_err(|source| Self::stateful_update_error("Adam step", source))?;
        super::dispatch::two::<Self, Adam, f32>(
            "Adam step",
            parameter,
            parameter_layout,
            gradient,
            gradient_layout,
            first,
            first_layout,
            second,
            second_layout,
            parameters,
        )
    }

    #[doc(hidden)]
    #[expect(clippy::too_many_arguments, reason = "mirrors OptimizerOps")]
    fn dispatch_rmsprop_step(
        &self,
        parameter: &mut Self::DeviceBuffer<f32>,
        parameter_layout: &Layout,
        gradient: &Self::DeviceBuffer<f32>,
        gradient_layout: &Layout,
        average: &mut Self::DeviceBuffer<f32>,
        average_layout: &Layout,
        learning_rate: f32,
        alpha: f32,
        epsilon: f32,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        let parameters = RmsPropParameters::new(learning_rate, alpha, epsilon)
            .map_err(|source| Self::stateful_update_error("RMSProp step", source))?;
        super::dispatch::one::<Self, RmsProp, f32>(
            "RMSProp step",
            parameter,
            parameter_layout,
            gradient,
            gradient_layout,
            average,
            average_layout,
            parameters,
        )
    }

    #[doc(hidden)]
    #[expect(clippy::too_many_arguments, reason = "mirrors OptimizerOps")]
    fn dispatch_adamw_step(
        &self,
        parameter: &mut Self::DeviceBuffer<f32>,
        parameter_layout: &Layout,
        gradient: &Self::DeviceBuffer<f32>,
        gradient_layout: &Layout,
        first: &mut Self::DeviceBuffer<f32>,
        first_layout: &Layout,
        second: &mut Self::DeviceBuffer<f32>,
        second_layout: &Layout,
        learning_rate: f32,
        beta_one: f32,
        beta_two: f32,
        epsilon: f32,
        weight_decay: f32,
        step: usize,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        let parameters = AdamWParameters::new(
            learning_rate,
            beta_one,
            beta_two,
            epsilon,
            weight_decay,
            step,
        )
        .map_err(|source| Self::stateful_update_error("AdamW step", source))?;
        super::dispatch::two::<Self, AdamW, f32>(
            "AdamW step",
            parameter,
            parameter_layout,
            gradient,
            gradient_layout,
            first,
            first_layout,
            second,
            second_layout,
            parameters,
        )
    }

    #[doc(hidden)]
    #[expect(clippy::too_many_arguments, reason = "mirrors OptimizerOps")]
    fn dispatch_adagrad_step(
        &self,
        parameter: &mut Self::DeviceBuffer<f32>,
        parameter_layout: &Layout,
        gradient: &Self::DeviceBuffer<f32>,
        gradient_layout: &Layout,
        history: &mut Self::DeviceBuffer<f32>,
        history_layout: &Layout,
        learning_rate: f32,
        epsilon: f32,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        let parameters = AdaGradParameters::new(learning_rate, epsilon)
            .map_err(|source| Self::stateful_update_error("AdaGrad step", source))?;
        super::dispatch::one::<Self, AdaGrad, f32>(
            "AdaGrad step",
            parameter,
            parameter_layout,
            gradient,
            gradient_layout,
            history,
            history_layout,
            parameters,
        )
    }

    #[doc(hidden)]
    fn validate_optimizer_step_f64(
        &self,
        validation: OptimizerStepValidation<'_, f64, Self>,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        let (parameter, parameter_layout) = validation.parameter;
        let (gradient, gradient_layout) = validation.gradient;
        match (validation.rule, validation.state) {
            (
                OptimizerStepRule::Sgd {
                    learning_rate,
                    momentum,
                },
                OptimizerStateRef::One(state, state_layout),
            ) => {
                SgdParametersF64::new(learning_rate, momentum)
                    .map_err(|source| Self::stateful_update_error("SGD step", source))?;
                super::dispatch::validate_one::<Self, Sgd, f64>(
                    "SGD step",
                    parameter,
                    parameter_layout,
                    gradient,
                    gradient_layout,
                    state,
                    state_layout,
                )
            }
            (
                OptimizerStepRule::Adam {
                    learning_rate,
                    beta_one,
                    beta_two,
                    epsilon,
                    step,
                },
                OptimizerStateRef::Two(first, first_layout, second, second_layout),
            ) => {
                AdamParametersF64::new(learning_rate, beta_one, beta_two, epsilon, step)
                    .map_err(|source| Self::stateful_update_error("Adam step", source))?;
                super::dispatch::validate_two::<Self, Adam, f64>(
                    "Adam step",
                    parameter,
                    parameter_layout,
                    gradient,
                    gradient_layout,
                    first,
                    first_layout,
                    second,
                    second_layout,
                )
            }
            (
                OptimizerStepRule::RmsProp {
                    learning_rate,
                    alpha,
                    epsilon,
                },
                OptimizerStateRef::One(state, state_layout),
            ) => {
                RmsPropParametersF64::new(learning_rate, alpha, epsilon)
                    .map_err(|source| Self::stateful_update_error("RMSProp step", source))?;
                super::dispatch::validate_one::<Self, RmsProp, f64>(
                    "RMSProp step",
                    parameter,
                    parameter_layout,
                    gradient,
                    gradient_layout,
                    state,
                    state_layout,
                )
            }
            (
                OptimizerStepRule::AdamW {
                    learning_rate,
                    beta_one,
                    beta_two,
                    epsilon,
                    weight_decay,
                    step,
                },
                OptimizerStateRef::Two(first, first_layout, second, second_layout),
            ) => {
                AdamWParametersF64::new(
                    learning_rate,
                    beta_one,
                    beta_two,
                    epsilon,
                    weight_decay,
                    step,
                )
                .map_err(|source| Self::stateful_update_error("AdamW step", source))?;
                super::dispatch::validate_two::<Self, AdamW, f64>(
                    "AdamW step",
                    parameter,
                    parameter_layout,
                    gradient,
                    gradient_layout,
                    first,
                    first_layout,
                    second,
                    second_layout,
                )
            }
            (
                OptimizerStepRule::AdaGrad {
                    learning_rate,
                    epsilon,
                },
                OptimizerStateRef::One(state, state_layout),
            ) => {
                AdaGradParametersF64::new(learning_rate, epsilon)
                    .map_err(|source| Self::stateful_update_error("AdaGrad step", source))?;
                super::dispatch::validate_one::<Self, AdaGrad, f64>(
                    "AdaGrad step",
                    parameter,
                    parameter_layout,
                    gradient,
                    gradient_layout,
                    state,
                    state_layout,
                )
            }
            _ => Err(Self::stateful_update_error(
                "optimizer step preflight",
                HephaestusError::InvalidConfiguration {
                    message: "optimizer rule and persistent state cardinality disagree".to_string(),
                },
            )),
        }
    }

    #[doc(hidden)]
    #[expect(clippy::too_many_arguments, reason = "mirrors OptimizerOps")]
    fn dispatch_sgd_step_f64(
        &self,
        parameter: &mut Self::DeviceBuffer<f64>,
        parameter_layout: &Layout,
        gradient: &Self::DeviceBuffer<f64>,
        gradient_layout: &Layout,
        velocity: &mut Self::DeviceBuffer<f64>,
        velocity_layout: &Layout,
        learning_rate: f64,
        momentum: f64,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        let parameters = SgdParametersF64::new(learning_rate, momentum)
            .map_err(|source| Self::stateful_update_error("SGD step", source))?;
        super::dispatch::one::<Self, Sgd, f64>(
            "SGD step",
            parameter,
            parameter_layout,
            gradient,
            gradient_layout,
            velocity,
            velocity_layout,
            parameters,
        )
    }

    #[doc(hidden)]
    #[expect(clippy::too_many_arguments, reason = "mirrors OptimizerOps")]
    fn dispatch_adam_step_f64(
        &self,
        parameter: &mut Self::DeviceBuffer<f64>,
        parameter_layout: &Layout,
        gradient: &Self::DeviceBuffer<f64>,
        gradient_layout: &Layout,
        first: &mut Self::DeviceBuffer<f64>,
        first_layout: &Layout,
        second: &mut Self::DeviceBuffer<f64>,
        second_layout: &Layout,
        learning_rate: f64,
        beta_one: f64,
        beta_two: f64,
        epsilon: f64,
        step: usize,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        let parameters = AdamParametersF64::new(learning_rate, beta_one, beta_two, epsilon, step)
            .map_err(|source| Self::stateful_update_error("Adam step", source))?;
        super::dispatch::two::<Self, Adam, f64>(
            "Adam step",
            parameter,
            parameter_layout,
            gradient,
            gradient_layout,
            first,
            first_layout,
            second,
            second_layout,
            parameters,
        )
    }

    #[doc(hidden)]
    #[expect(clippy::too_many_arguments, reason = "mirrors OptimizerOps")]
    fn dispatch_rmsprop_step_f64(
        &self,
        parameter: &mut Self::DeviceBuffer<f64>,
        parameter_layout: &Layout,
        gradient: &Self::DeviceBuffer<f64>,
        gradient_layout: &Layout,
        average: &mut Self::DeviceBuffer<f64>,
        average_layout: &Layout,
        learning_rate: f64,
        alpha: f64,
        epsilon: f64,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        let parameters = RmsPropParametersF64::new(learning_rate, alpha, epsilon)
            .map_err(|source| Self::stateful_update_error("RMSProp step", source))?;
        super::dispatch::one::<Self, RmsProp, f64>(
            "RMSProp step",
            parameter,
            parameter_layout,
            gradient,
            gradient_layout,
            average,
            average_layout,
            parameters,
        )
    }

    #[doc(hidden)]
    #[expect(clippy::too_many_arguments, reason = "mirrors OptimizerOps")]
    fn dispatch_adamw_step_f64(
        &self,
        parameter: &mut Self::DeviceBuffer<f64>,
        parameter_layout: &Layout,
        gradient: &Self::DeviceBuffer<f64>,
        gradient_layout: &Layout,
        first: &mut Self::DeviceBuffer<f64>,
        first_layout: &Layout,
        second: &mut Self::DeviceBuffer<f64>,
        second_layout: &Layout,
        learning_rate: f64,
        beta_one: f64,
        beta_two: f64,
        epsilon: f64,
        weight_decay: f64,
        step: usize,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        let parameters = AdamWParametersF64::new(
            learning_rate,
            beta_one,
            beta_two,
            epsilon,
            weight_decay,
            step,
        )
        .map_err(|source| Self::stateful_update_error("AdamW step", source))?;
        super::dispatch::two::<Self, AdamW, f64>(
            "AdamW step",
            parameter,
            parameter_layout,
            gradient,
            gradient_layout,
            first,
            first_layout,
            second,
            second_layout,
            parameters,
        )
    }

    #[doc(hidden)]
    #[expect(clippy::too_many_arguments, reason = "mirrors OptimizerOps")]
    fn dispatch_adagrad_step_f64(
        &self,
        parameter: &mut Self::DeviceBuffer<f64>,
        parameter_layout: &Layout,
        gradient: &Self::DeviceBuffer<f64>,
        gradient_layout: &Layout,
        history: &mut Self::DeviceBuffer<f64>,
        history_layout: &Layout,
        learning_rate: f64,
        epsilon: f64,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        let parameters = AdaGradParametersF64::new(learning_rate, epsilon)
            .map_err(|source| Self::stateful_update_error("AdaGrad step", source))?;
        super::dispatch::one::<Self, AdaGrad, f64>(
            "AdaGrad step",
            parameter,
            parameter_layout,
            gradient,
            gradient_layout,
            history,
            history_layout,
            parameters,
        )
    }
}
