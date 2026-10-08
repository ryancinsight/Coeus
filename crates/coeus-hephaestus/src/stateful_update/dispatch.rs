use super::{StatefulUpdateBackend, StatefulUpdateProvider};
use crate::{layout::ranked, HephaestusProvider};
use coeus_core::{BackendError, Layout, Scalar, StorageMut};
use hephaestus_core::{
    plan_stateful_update, ComputeDevice, HephaestusError, StatefulUpdateAliasing,
    StatefulUpdateOperands, StatefulUpdateOps, StatefulUpdateRule, StridedView,
};

type Operations<B> = <<B as StatefulUpdateBackend>::Provider as StatefulUpdateProvider>::Operations;
type Provider<B> = <B as StatefulUpdateBackend>::Provider;
type Device<B> = <Provider<B> as HephaestusProvider>::Device;
type Dialect<B> = <Operations<B> as StatefulUpdateOps<Device<B>>>::Dialect;

/// Width-indexed optimizer entry point.
///
/// Selects `stateful_update` (f32) or `stateful_update_f64` (f64) plus the
/// matching parameter struct, so every dispatch body below runs for both
/// widths from one generic body instead of twinned f32/f64 modules.
pub(super) trait Width: Scalar {
    /// Backend parameter struct for `Rule` at this width.
    type Parameters<B, Rule>
    where
        B: StatefulUpdateBackend,
        Rule: StatefulUpdateRule<Dialect<B>>;

    /// Dispatch one update at this width.
    fn execute<B, Rule, const N: usize>(
        operands: StatefulUpdateOperands<'_, <Device<B> as ComputeDevice>::Buffer<Self>, N>,
        parameters: Self::Parameters<B, Rule>,
    ) -> Result<(), HephaestusError>
    where
        B: StatefulUpdateBackend,
        Rule: StatefulUpdateRule<Dialect<B>>;
}

impl Width for f32 {
    type Parameters<B, Rule>
        = Rule::Parameters
    where
        B: StatefulUpdateBackend,
        Rule: StatefulUpdateRule<Dialect<B>>;

    fn execute<B, Rule, const N: usize>(
        operands: StatefulUpdateOperands<'_, <Device<B> as ComputeDevice>::Buffer<Self>, N>,
        parameters: Self::Parameters<B, Rule>,
    ) -> Result<(), HephaestusError>
    where
        B: StatefulUpdateBackend,
        Rule: StatefulUpdateRule<Dialect<B>>,
    {
        Operations::<B>::default().stateful_update::<Rule, N>(
            <Provider<B> as HephaestusProvider>::device(),
            operands,
            parameters,
        )
    }
}

impl Width for f64 {
    type Parameters<B, Rule>
        = Rule::ParametersF64
    where
        B: StatefulUpdateBackend,
        Rule: StatefulUpdateRule<Dialect<B>>;

    fn execute<B, Rule, const N: usize>(
        operands: StatefulUpdateOperands<'_, <Device<B> as ComputeDevice>::Buffer<Self>, N>,
        parameters: Self::Parameters<B, Rule>,
    ) -> Result<(), HephaestusError>
    where
        B: StatefulUpdateBackend,
        Rule: StatefulUpdateRule<Dialect<B>>,
    {
        Operations::<B>::default().stateful_update_f64::<Rule, N>(
            <Provider<B> as HephaestusProvider>::device(),
            operands,
            parameters,
        )
    }
}

struct Request<'a, B: StatefulUpdateBackend, T: Scalar> {
    operation: &'static str,
    parameter: &'a B::DeviceBuffer<T>,
    parameter_layout: &'a Layout,
    gradient: &'a B::DeviceBuffer<T>,
    gradient_layout: &'a Layout,
    states: State<'a, B, T>,
}

enum State<'a, B: StatefulUpdateBackend, T: Scalar> {
    One(&'a B::DeviceBuffer<T>, &'a Layout),
    Two(
        &'a B::DeviceBuffer<T>,
        &'a Layout,
        &'a B::DeviceBuffer<T>,
        &'a Layout,
    ),
}

pub(super) fn validate_one<B, Rule, T>(
    operation: &'static str,
    parameter: &B::DeviceBuffer<T>,
    parameter_layout: &Layout,
    gradient: &B::DeviceBuffer<T>,
    gradient_layout: &Layout,
    state: &B::DeviceBuffer<T>,
    state_layout: &Layout,
) -> Result<(), B::Error>
where
    B: StatefulUpdateBackend,
    Rule: StatefulUpdateRule<Dialect<B>>,
    T: Width,
{
    validate::<B, Rule, T>(Request {
        operation,
        parameter,
        parameter_layout,
        gradient,
        gradient_layout,
        states: State::One(state, state_layout),
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "validates two-state provider operands"
)]
pub(super) fn validate_two<B, Rule, T>(
    operation: &'static str,
    parameter: &B::DeviceBuffer<T>,
    parameter_layout: &Layout,
    gradient: &B::DeviceBuffer<T>,
    gradient_layout: &Layout,
    first: &B::DeviceBuffer<T>,
    first_layout: &Layout,
    second: &B::DeviceBuffer<T>,
    second_layout: &Layout,
) -> Result<(), B::Error>
where
    B: StatefulUpdateBackend,
    Rule: StatefulUpdateRule<Dialect<B>>,
    T: Width,
{
    validate::<B, Rule, T>(Request {
        operation,
        parameter,
        parameter_layout,
        gradient,
        gradient_layout,
        states: State::Two(first, first_layout, second, second_layout),
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "assembles one-state provider operands"
)]
pub(super) fn one<B, Rule, T>(
    operation: &'static str,
    parameter: &mut B::DeviceBuffer<T>,
    parameter_layout: &Layout,
    gradient: &B::DeviceBuffer<T>,
    gradient_layout: &Layout,
    state: &mut B::DeviceBuffer<T>,
    state_layout: &Layout,
    parameters: T::Parameters<B, Rule>,
) -> Result<(), B::Error>
where
    B: StatefulUpdateBackend,
    Rule: StatefulUpdateRule<Dialect<B>>,
    T: Width,
{
    validate_one::<B, Rule, T>(
        operation,
        parameter,
        parameter_layout,
        gradient,
        gradient_layout,
        state,
        state_layout,
    )?;
    parameter.make_unique();
    state.make_unique();
    dispatch::<B, Rule, T>(
        Request {
            operation,
            parameter: &*parameter,
            parameter_layout,
            gradient,
            gradient_layout,
            states: State::One(&*state, state_layout),
        },
        parameters,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "assembles two-state provider operands"
)]
pub(super) fn two<B, Rule, T>(
    operation: &'static str,
    parameter: &mut B::DeviceBuffer<T>,
    parameter_layout: &Layout,
    gradient: &B::DeviceBuffer<T>,
    gradient_layout: &Layout,
    first: &mut B::DeviceBuffer<T>,
    first_layout: &Layout,
    second: &mut B::DeviceBuffer<T>,
    second_layout: &Layout,
    parameters: T::Parameters<B, Rule>,
) -> Result<(), B::Error>
where
    B: StatefulUpdateBackend,
    Rule: StatefulUpdateRule<Dialect<B>>,
    T: Width,
{
    validate_two::<B, Rule, T>(
        operation,
        parameter,
        parameter_layout,
        gradient,
        gradient_layout,
        first,
        first_layout,
        second,
        second_layout,
    )?;
    parameter.make_unique();
    first.make_unique();
    second.make_unique();
    dispatch::<B, Rule, T>(
        Request {
            operation,
            parameter: &*parameter,
            parameter_layout,
            gradient,
            gradient_layout,
            states: State::Two(&*first, first_layout, &*second, second_layout),
        },
        parameters,
    )
}

fn dispatch<B, Rule, T>(
    request: Request<'_, B, T>,
    parameters: T::Parameters<B, Rule>,
) -> Result<(), B::Error>
where
    B: StatefulUpdateBackend,
    Rule: StatefulUpdateRule<Dialect<B>>,
    T: Width,
{
    match request.parameter_layout.ndim() {
        0 => execute::<B, Rule, T, 0>(request, parameters),
        1 => execute::<B, Rule, T, 1>(request, parameters),
        2 => execute::<B, Rule, T, 2>(request, parameters),
        3 => execute::<B, Rule, T, 3>(request, parameters),
        4 => execute::<B, Rule, T, 4>(request, parameters),
        5 => execute::<B, Rule, T, 5>(request, parameters),
        6 => execute::<B, Rule, T, 6>(request, parameters),
        7 => execute::<B, Rule, T, 7>(request, parameters),
        8 => execute::<B, Rule, T, 8>(request, parameters),
        rank => Err(BackendError::UnsupportedRank {
            operation: request.operation,
            rank,
            max_rank: 8,
        }
        .into()),
    }
}

fn validate<B, Rule, T>(request: Request<'_, B, T>) -> Result<(), B::Error>
where
    B: StatefulUpdateBackend,
    Rule: StatefulUpdateRule<Dialect<B>>,
    T: Width,
{
    match request.parameter_layout.ndim() {
        0 => validate_rank::<B, Rule, T, 0>(request),
        1 => validate_rank::<B, Rule, T, 1>(request),
        2 => validate_rank::<B, Rule, T, 2>(request),
        3 => validate_rank::<B, Rule, T, 3>(request),
        4 => validate_rank::<B, Rule, T, 4>(request),
        5 => validate_rank::<B, Rule, T, 5>(request),
        6 => validate_rank::<B, Rule, T, 6>(request),
        7 => validate_rank::<B, Rule, T, 7>(request),
        8 => validate_rank::<B, Rule, T, 8>(request),
        rank => Err(BackendError::UnsupportedRank {
            operation: request.operation,
            rank,
            max_rank: 8,
        }
        .into()),
    }
}

fn validate_rank<B, Rule, T, const N: usize>(request: Request<'_, B, T>) -> Result<(), B::Error>
where
    B: StatefulUpdateBackend,
    Rule: StatefulUpdateRule<Dialect<B>>,
    T: Width,
{
    let parameter_layout = ranked::<N>(request.operation, request.parameter_layout)?;
    let gradient_layout = ranked::<N>(request.operation, request.gradient_layout)?;
    let parameter = StridedView::new(
        B::stateful_update_buffer(request.parameter),
        &parameter_layout,
    );
    let gradient = StridedView::new(
        B::stateful_update_buffer(request.gradient),
        &gradient_layout,
    );
    let result = match request.states {
        State::One(state, layout) => {
            let layout = ranked::<N>(request.operation, layout)?;
            let states = [StridedView::new(B::stateful_update_buffer(state), &layout)];
            plan_stateful_update(
                StatefulUpdateOperands {
                    parameter,
                    gradient,
                    states: &states,
                },
                Rule::STATE_COUNT,
                StatefulUpdateAliasing::default(),
            )
        }
        State::Two(first, first_layout, second, second_layout) => {
            let first_layout = ranked::<N>(request.operation, first_layout)?;
            let second_layout = ranked::<N>(request.operation, second_layout)?;
            let states = [
                StridedView::new(B::stateful_update_buffer(first), &first_layout),
                StridedView::new(B::stateful_update_buffer(second), &second_layout),
            ];
            plan_stateful_update(
                StatefulUpdateOperands {
                    parameter,
                    gradient,
                    states: &states,
                },
                Rule::STATE_COUNT,
                StatefulUpdateAliasing::default(),
            )
        }
    };
    result
        .map(|_| ())
        .map_err(|source| B::stateful_update_error(request.operation, source))
}

fn execute<B, Rule, T, const N: usize>(
    request: Request<'_, B, T>,
    parameters: T::Parameters<B, Rule>,
) -> Result<(), B::Error>
where
    B: StatefulUpdateBackend,
    Rule: StatefulUpdateRule<Dialect<B>>,
    T: Width,
{
    let parameter_layout = ranked::<N>(request.operation, request.parameter_layout)?;
    let gradient_layout = ranked::<N>(request.operation, request.gradient_layout)?;
    let parameter = StridedView::new(
        B::stateful_update_buffer(request.parameter),
        &parameter_layout,
    );
    let gradient = StridedView::new(
        B::stateful_update_buffer(request.gradient),
        &gradient_layout,
    );

    match request.states {
        State::One(state, layout) => {
            let layout = ranked::<N>(request.operation, layout)?;
            let states = [StridedView::new(B::stateful_update_buffer(state), &layout)];
            T::execute::<B, Rule, N>(
                StatefulUpdateOperands {
                    parameter,
                    gradient,
                    states: &states,
                },
                parameters,
            )
            .map_err(|source| B::stateful_update_error(request.operation, source))
        }
        State::Two(first, first_layout, second, second_layout) => {
            let first_layout = ranked::<N>(request.operation, first_layout)?;
            let second_layout = ranked::<N>(request.operation, second_layout)?;
            let states = [
                StridedView::new(B::stateful_update_buffer(first), &first_layout),
                StridedView::new(B::stateful_update_buffer(second), &second_layout),
            ];
            T::execute::<B, Rule, N>(
                StatefulUpdateOperands {
                    parameter,
                    gradient,
                    states: &states,
                },
                parameters,
            )
            .map_err(|source| B::stateful_update_error(request.operation, source))
        }
    }
}
