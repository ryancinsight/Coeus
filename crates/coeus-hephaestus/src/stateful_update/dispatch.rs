use super::{StatefulUpdateBackend, StatefulUpdateProvider};
use crate::{layout::ranked, HephaestusProvider};
use coeus_core::{BackendError, Layout};
use hephaestus_core::{StatefulUpdateOperands, StatefulUpdateOps, StatefulUpdateRule, StridedView};

type Operations<B> = <<B as StatefulUpdateBackend>::Provider as StatefulUpdateProvider>::Operations;
type Provider<B> = <B as StatefulUpdateBackend>::Provider;
type Dialect<B> =
    <Operations<B> as StatefulUpdateOps<<Provider<B> as HephaestusProvider>::Device>>::Dialect;
type Parameters<B, Rule> = <Rule as StatefulUpdateRule<Dialect<B>>>::Parameters;

struct Request<'a, B: StatefulUpdateBackend> {
    operation: &'static str,
    parameter: &'a B::DeviceBuffer<f32>,
    parameter_layout: &'a Layout,
    gradient: &'a B::DeviceBuffer<f32>,
    gradient_layout: &'a Layout,
    states: State<'a, B>,
}

enum State<'a, B: StatefulUpdateBackend> {
    One(&'a B::DeviceBuffer<f32>, &'a Layout),
    Two(
        &'a B::DeviceBuffer<f32>,
        &'a Layout,
        &'a B::DeviceBuffer<f32>,
        &'a Layout,
    ),
}

#[expect(
    clippy::too_many_arguments,
    reason = "assembles one-state provider operands"
)]
pub(super) fn one<B, Rule>(
    operation: &'static str,
    parameter: &mut B::DeviceBuffer<f32>,
    parameter_layout: &Layout,
    gradient: &B::DeviceBuffer<f32>,
    gradient_layout: &Layout,
    state: &mut B::DeviceBuffer<f32>,
    state_layout: &Layout,
    parameters: Parameters<B, Rule>,
) -> Result<(), B::Error>
where
    B: StatefulUpdateBackend,
    Rule: StatefulUpdateRule<Dialect<B>>,
{
    dispatch::<B, Rule>(
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
pub(super) fn two<B, Rule>(
    operation: &'static str,
    parameter: &mut B::DeviceBuffer<f32>,
    parameter_layout: &Layout,
    gradient: &B::DeviceBuffer<f32>,
    gradient_layout: &Layout,
    first: &mut B::DeviceBuffer<f32>,
    first_layout: &Layout,
    second: &mut B::DeviceBuffer<f32>,
    second_layout: &Layout,
    parameters: Parameters<B, Rule>,
) -> Result<(), B::Error>
where
    B: StatefulUpdateBackend,
    Rule: StatefulUpdateRule<Dialect<B>>,
{
    dispatch::<B, Rule>(
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

fn dispatch<B, Rule>(
    request: Request<'_, B>,
    parameters: Parameters<B, Rule>,
) -> Result<(), B::Error>
where
    B: StatefulUpdateBackend,
    Rule: StatefulUpdateRule<Dialect<B>>,
{
    match request.parameter_layout.ndim() {
        0 => execute::<B, Rule, 0>(request, parameters),
        1 => execute::<B, Rule, 1>(request, parameters),
        2 => execute::<B, Rule, 2>(request, parameters),
        3 => execute::<B, Rule, 3>(request, parameters),
        4 => execute::<B, Rule, 4>(request, parameters),
        5 => execute::<B, Rule, 5>(request, parameters),
        6 => execute::<B, Rule, 6>(request, parameters),
        7 => execute::<B, Rule, 7>(request, parameters),
        8 => execute::<B, Rule, 8>(request, parameters),
        rank => Err(BackendError::UnsupportedRank {
            operation: request.operation,
            rank,
            max_rank: 8,
        }
        .into()),
    }
}

fn execute<B, Rule, const N: usize>(
    request: Request<'_, B>,
    parameters: Parameters<B, Rule>,
) -> Result<(), B::Error>
where
    B: StatefulUpdateBackend,
    Rule: StatefulUpdateRule<Dialect<B>>,
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
            Operations::<B>::default()
                .stateful_update::<Rule, N>(
                    <Provider<B> as HephaestusProvider>::device(),
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
            Operations::<B>::default()
                .stateful_update::<Rule, N>(
                    <Provider<B> as HephaestusProvider>::device(),
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
