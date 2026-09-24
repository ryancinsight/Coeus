//! TCP collective contracts: cross-rank element-count mismatches are typed
//! errors that poison the links; local caller errors panic.

use super::super::super::support::loopback_meshes;
use super::super::super::support::single_rank_tcp_mesh;
use coeus_core::SequentialBackend;
use coeus_dist::Communicator;
use coeus_dist::Sum;
use coeus_dist::TcpCommunicator;
use coeus_dist::TcpMeshError;
use coeus_tensor::Tensor;
use std::thread;

/// One rank's collective result and the result of its next barrier.
type RankOutcome = (Result<(), TcpMeshError>, Result<(), TcpMeshError>);

/// Run `collective` on both ranks of a two-rank loopback cluster, then a
/// barrier on each, and return each rank's pair of results.
fn on_two_ranks<F>(collective: F) -> Vec<RankOutcome>
where
    F: Fn(usize, &TcpCommunicator, &SequentialBackend) -> Result<(), TcpMeshError> + Sync,
{
    let collective = &collective;
    thread::scope(|scope| {
        let ranks = loopback_meshes(2)
            .into_iter()
            .enumerate()
            .map(|(rank, mesh)| {
                scope.spawn(move || {
                    let mut comm = TcpCommunicator::new(mesh);
                    let backend = SequentialBackend::new();
                    let outcome = collective(rank, &comm, &backend);
                    let next = comm.barrier();
                    comm.shutdown();
                    (outcome, next)
                })
            })
            .collect::<Vec<_>>();
        ranks.into_iter().map(|rank| rank.join().unwrap()).collect()
    })
}

/// Both ranks' next collective fails without touching the wire.
fn assert_poisoned(rank: usize, next: &Result<(), TcpMeshError>) {
    assert!(
        matches!(next, Err(TcpMeshError::LinkPoisoned { rank: r, .. }) if *r == rank),
        "rank {rank}: expected LinkPoisoned after the mismatch, got {next:?}"
    );
}

/// A rooted collective whose ranks hold `numels` elements: the root reports
/// the other rank's count, the other rank reports the root's status 0.
fn assert_rooted_mismatch(outcomes: &[RankOutcome], root: usize, numels: [u64; 2]) {
    let other = 1 - root;
    match &outcomes[root].0 {
        Err(TcpMeshError::NumelMismatch {
            rank,
            peer,
            expected,
            received,
            ..
        }) => assert_eq!(
            (*rank, *peer, *expected, *received),
            (root, other, numels[root], numels[other])
        ),
        result => panic!("root {root}: expected NumelMismatch, got {result:?}"),
    }
    match &outcomes[other].0 {
        Err(TcpMeshError::PeerReportedMismatch { rank, peer, .. }) => {
            assert_eq!((*rank, *peer), (other, root));
        }
        result => panic!("rank {other}: expected PeerReportedMismatch, got {result:?}"),
    }
    for (rank, (_, next)) in outcomes.iter().enumerate() {
        assert_poisoned(rank, next);
    }
}

/// An all-gather whose ranks hold `numels` elements: each rank reports the
/// other's count.
fn assert_pairwise_mismatch(outcomes: &[RankOutcome], numels: [u64; 2]) {
    for (rank, (outcome, next)) in outcomes.iter().enumerate() {
        let other = 1 - rank;
        match outcome {
            Err(TcpMeshError::NumelMismatch {
                rank: reporting,
                peer,
                expected,
                received,
                ..
            }) => assert_eq!(
                (*reporting, *peer, *expected, *received),
                (rank, other, numels[rank], numels[other])
            ),
            result => panic!("rank {rank}: expected NumelMismatch, got {result:?}"),
        }
        assert_poisoned(rank, next);
    }
}

/// A tensor of `lens[rank]` zeros.
fn per_rank(
    rank: usize,
    lens: [usize; 2],
    backend: &SequentialBackend,
) -> Tensor<f32, SequentialBackend> {
    Tensor::zeros_on([lens[rank]], backend)
}

#[test]
fn test_tcp_all_reduce_mismatched_numel_is_a_typed_mismatch() {
    let outcomes = on_two_ranks(|rank, comm, backend| {
        let mut tensor = per_rank(rank, [2, 1], backend);
        comm.all_reduce::<f32, _, Sum>(&mut tensor, backend)
    });
    assert_rooted_mismatch(&outcomes, 0, [2, 1]);
}

#[test]
fn test_tcp_all_reduce_zero_numel_mismatched_numel_is_a_typed_mismatch() {
    let outcomes = on_two_ranks(|rank, comm, backend| {
        let mut tensor = per_rank(rank, [0, 1], backend);
        comm.all_reduce::<f32, _, Sum>(&mut tensor, backend)
    });
    assert_rooted_mismatch(&outcomes, 0, [0, 1]);
}

#[test]
fn test_tcp_broadcast_mismatched_numel_is_a_typed_mismatch() {
    let outcomes = on_two_ranks(|rank, comm, backend| {
        let mut tensor = per_rank(rank, [2, 1], backend);
        comm.broadcast(&mut tensor, 0, backend)
    });
    assert_rooted_mismatch(&outcomes, 0, [2, 1]);
}

#[test]
fn test_tcp_all_gather_mismatched_peer_numel_is_a_typed_mismatch() {
    let outcomes = on_two_ranks(|rank, comm, backend| {
        let lens = [2, 1];
        let tensor = per_rank(rank, lens, backend);
        let mut output = vec![per_rank(rank, lens, backend), per_rank(rank, lens, backend)];
        comm.all_gather(&tensor, &mut output, backend)
    });
    assert_pairwise_mismatch(&outcomes, [2, 1]);
}

#[test]
fn test_tcp_all_gather_zero_numel_mismatched_peer_numel_is_a_typed_mismatch() {
    let outcomes = on_two_ranks(|rank, comm, backend| {
        let lens = [0, 1];
        let tensor = per_rank(rank, lens, backend);
        let mut output = vec![per_rank(rank, lens, backend), per_rank(rank, lens, backend)];
        comm.all_gather(&tensor, &mut output, backend)
    });
    assert_pairwise_mismatch(&outcomes, [0, 1]);
}

#[test]
fn test_tcp_reduce_mismatched_numel_is_a_typed_mismatch() {
    let outcomes = on_two_ranks(|rank, comm, backend| {
        let mut tensor = per_rank(rank, [2, 1], backend);
        comm.reduce::<f32, _, Sum>(&mut tensor, 0, backend)
    });
    assert_rooted_mismatch(&outcomes, 0, [2, 1]);
}

#[test]
fn test_tcp_gather_mismatched_peer_numel_is_a_typed_mismatch() {
    let outcomes = on_two_ranks(|rank, comm, backend| {
        let lens = [2, 1];
        let tensor = per_rank(rank, lens, backend);
        let mut output = vec![per_rank(rank, lens, backend), per_rank(rank, lens, backend)];
        comm.gather(&tensor, &mut output, 1, backend)
    });
    assert_rooted_mismatch(&outcomes, 1, [2, 1]);
}

#[test]
fn test_tcp_gather_zero_numel_mismatched_peer_numel_is_a_typed_mismatch() {
    let outcomes = on_two_ranks(|rank, comm, backend| {
        let lens = [0, 1];
        let tensor = per_rank(rank, lens, backend);
        let mut output = vec![per_rank(rank, lens, backend), per_rank(rank, lens, backend)];
        comm.gather(&tensor, &mut output, 1, backend)
    });
    assert_rooted_mismatch(&outcomes, 1, [0, 1]);
}

#[test]
fn test_tcp_scatter_mismatched_target_numel_is_a_typed_mismatch() {
    let outcomes = on_two_ranks(|rank, comm, backend| {
        let lens = [2, 1];
        let mut tensor = per_rank(rank, lens, backend);
        let input = if rank == 0 {
            vec![per_rank(0, lens, backend), per_rank(0, lens, backend)]
        } else {
            Vec::new()
        };
        comm.scatter(&mut tensor, &input, 0, backend)
    });
    assert_rooted_mismatch(&outcomes, 0, [2, 1]);
}

#[test]
fn test_tcp_scatter_zero_numel_mismatched_target_numel_is_a_typed_mismatch() {
    let outcomes = on_two_ranks(|rank, comm, backend| {
        let lens = [0, 1];
        let mut tensor = per_rank(rank, lens, backend);
        let input = if rank == 0 {
            vec![per_rank(0, lens, backend), per_rank(0, lens, backend)]
        } else {
            Vec::new()
        };
        comm.scatter(&mut tensor, &input, 0, backend)
    });
    assert_rooted_mismatch(&outcomes, 0, [0, 1]);
}

#[test]
#[should_panic(expected = "all_gather output numel mismatch")]
fn test_tcp_all_gather_mismatched_output_numel_panics() {
    let mesh = single_rank_tcp_mesh();
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();

    let tensor = Tensor::from_slice_on([2], &[1.0f32, 2.0], &backend);
    let mut output = vec![Tensor::zeros_on([1], &backend)];
    comm.all_gather(&tensor, &mut output, &backend).unwrap();
}

#[test]
#[should_panic(expected = "all_gather output length mismatch")]
fn test_tcp_all_gather_zero_numel_output_len_mismatch_panics() {
    let mesh = single_rank_tcp_mesh();
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();

    let tensor = Tensor::<f32, _>::zeros_on([0], &backend);
    let mut output: Vec<Tensor<f32, SequentialBackend>> = vec![];
    comm.all_gather(&tensor, &mut output, &backend).unwrap();
}

#[test]
#[should_panic(expected = "all_gather output numel mismatch")]
fn test_tcp_all_gather_zero_numel_output_numel_mismatch_panics() {
    let mesh = single_rank_tcp_mesh();
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();

    let tensor = Tensor::<f32, _>::zeros_on([0], &backend);
    let mut output = vec![Tensor::zeros_on([1], &backend)];
    comm.all_gather(&tensor, &mut output, &backend).unwrap();
}

#[test]
#[should_panic(expected = "scatter input numel mismatch")]
fn test_tcp_scatter_mismatched_input_numel_panics() {
    let mesh = single_rank_tcp_mesh();
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();

    let mut tensor = Tensor::zeros_on([2], &backend);
    let input = vec![Tensor::from_slice_on([1], &[3.0f32], &backend)];
    comm.scatter(&mut tensor, &input, 0, &backend).unwrap();
}

#[test]
#[should_panic(expected = "collective root out of bounds")]
fn test_tcp_broadcast_root_out_of_bounds_panics() {
    let mesh = single_rank_tcp_mesh();
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();
    let mut tensor = Tensor::from_slice_on([1], &[1.0f32], &backend);
    comm.broadcast(&mut tensor, 1, &backend).unwrap();
}

#[test]
#[should_panic(expected = "collective root out of bounds")]
fn test_tcp_reduce_root_out_of_bounds_panics() {
    let mesh = single_rank_tcp_mesh();
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();
    let mut tensor = Tensor::from_slice_on([1], &[1.0f32], &backend);
    comm.reduce::<f32, _, Sum>(&mut tensor, 1, &backend)
        .unwrap();
}

#[test]
#[should_panic(expected = "collective root out of bounds")]
fn test_tcp_gather_root_out_of_bounds_panics() {
    let mesh = single_rank_tcp_mesh();
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();
    let tensor = Tensor::from_slice_on([1], &[1.0f32], &backend);
    let mut output = vec![Tensor::zeros_on([1], &backend)];
    comm.gather(&tensor, &mut output, 1, &backend).unwrap();
}

#[test]
#[should_panic(expected = "collective root out of bounds")]
fn test_tcp_scatter_root_out_of_bounds_panics() {
    let mesh = single_rank_tcp_mesh();
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();
    let mut tensor = Tensor::zeros_on([1], &backend);
    let input = vec![Tensor::from_slice_on([1], &[1.0f32], &backend)];
    comm.scatter(&mut tensor, &input, 1, &backend).unwrap();
}

#[test]
#[should_panic(expected = "gather output length mismatch on root")]
fn test_tcp_gather_zero_numel_output_len_mismatch_panics() {
    let mesh = single_rank_tcp_mesh();
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();
    let tensor = Tensor::<f32, _>::zeros_on([0], &backend);
    let mut output: Vec<Tensor<f32, SequentialBackend>> = vec![];
    comm.gather(&tensor, &mut output, 0, &backend).unwrap();
}

#[test]
#[should_panic(expected = "gather output numel mismatch")]
fn test_tcp_gather_mismatched_output_numel_panics() {
    let mesh = single_rank_tcp_mesh();
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();
    let tensor = Tensor::from_slice_on([2], &[1.0f32, 2.0], &backend);
    let mut output = vec![Tensor::zeros_on([1], &backend)];
    comm.gather(&tensor, &mut output, 0, &backend).unwrap();
}

#[test]
#[should_panic(expected = "gather output numel mismatch")]
fn test_tcp_gather_zero_numel_output_numel_mismatch_panics() {
    let mesh = single_rank_tcp_mesh();
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();
    let tensor = Tensor::<f32, _>::zeros_on([0], &backend);
    let mut output = vec![Tensor::zeros_on([1], &backend)];
    comm.gather(&tensor, &mut output, 0, &backend).unwrap();
}

#[test]
#[should_panic(expected = "scatter input length mismatch on root")]
fn test_tcp_scatter_zero_numel_input_len_mismatch_panics() {
    let mesh = single_rank_tcp_mesh();
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();
    let mut tensor = Tensor::<f32, _>::zeros_on([0], &backend);
    let input: Vec<Tensor<f32, SequentialBackend>> = vec![];
    comm.scatter(&mut tensor, &input, 0, &backend).unwrap();
}

#[test]
#[should_panic(expected = "scatter input numel mismatch")]
fn test_tcp_scatter_zero_numel_input_numel_mismatch_panics() {
    let mesh = single_rank_tcp_mesh();
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();
    let mut tensor = Tensor::<f32, _>::zeros_on([0], &backend);
    let input = vec![Tensor::zeros_on([1], &backend)];
    comm.scatter(&mut tensor, &input, 0, &backend).unwrap();
}
