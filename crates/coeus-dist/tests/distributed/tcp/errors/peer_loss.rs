//! A rank lost before a collective leaves every surviving rank with a typed
//! error instead of a panic or an unbounded wait.

use coeus_core::SequentialBackend;
use coeus_dist::{Communicator, MeshDeadlines, Sum, TcpCommunicator, TcpMesh, TcpMeshError};
use coeus_tensor::Tensor;
use std::io::ErrorKind;
use std::num::NonZeroUsize;
use std::thread;
use std::time::Duration;

/// Survivors' I/O bound: far above loopback delivery of a close, and short
/// enough that a survivor left waiting fails with `RecvTimedOut` before the
/// 60 s nextest termination.
const SURVIVOR_IO: Duration = Duration::from_secs(20);

/// Rank 2 of 3 is shut down before an all-reduce. Rank 0, the reduce root,
/// reads rank 1's element count, then finds rank 2's stream at its end; it
/// poisons its links, which closes rank 1's pending wait for the handshake
/// status. Both survivors report `Recv` with `UnexpectedEof`: rank 0 had read
/// everything rank 1 sent, so its close is a FIN rather than a reset.
#[test]
fn all_reduce_after_a_peer_is_lost_fails_on_every_survivor() {
    let mut meshes = TcpMesh::create_loopback_cluster(
        NonZeroUsize::new(3).unwrap(),
        MeshDeadlines::DEFAULT.with_io(SURVIVOR_IO),
    )
    .unwrap();
    let mut lost = meshes.pop().unwrap();
    lost.shutdown();
    drop(lost);
    let mut survivors = meshes
        .into_iter()
        .map(TcpCommunicator::new)
        .collect::<Vec<_>>();

    let outcomes = thread::scope(|scope| {
        let pending = survivors
            .iter()
            .map(|comm| {
                scope.spawn(move || {
                    let backend = SequentialBackend::new();
                    let mut tensor = Tensor::from_slice_on([2], &[1.0_f32, 2.0], &backend);
                    comm.all_reduce::<f32, _, Sum>(&mut tensor, &backend)
                })
            })
            .collect::<Vec<_>>();
        pending
            .into_iter()
            .map(|rank| rank.join().unwrap())
            .collect::<Vec<_>>()
    });

    for (rank, (outcome, expected_peer)) in outcomes.into_iter().zip([2, 0]).enumerate() {
        match outcome {
            Err(TcpMeshError::Recv {
                rank: reporting,
                peer,
                source,
                ..
            }) => {
                assert_eq!(reporting, rank);
                assert_eq!(peer, expected_peer, "rank {rank} names the peer it lost");
                assert_eq!(source.kind(), ErrorKind::UnexpectedEof, "rank {rank}");
            }
            other => panic!("rank {rank}: expected Recv, got {other:?}"),
        }
    }

    // The failed collective poisoned rank 0's links; later collectives fail
    // at once instead of exchanging misaligned bytes.
    assert!(matches!(
        survivors[0].barrier(),
        Err(TcpMeshError::LinkPoisoned { rank: 0, .. })
    ));
    survivors.iter_mut().for_each(TcpCommunicator::shutdown);
}
