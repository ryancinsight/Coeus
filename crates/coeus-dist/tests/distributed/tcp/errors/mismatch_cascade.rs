//! An element-count mismatch in a three-rank collective is a typed error on
//! the ranks that see it, and the links it poisons end every other rank's
//! wait at once instead of after the I/O bound.

use coeus_core::SequentialBackend;
use coeus_dist::{Communicator, MeshDeadlines, TcpCommunicator, TcpMesh, TcpMeshError};
use coeus_tensor::Tensor;
use std::io::ErrorKind;
use std::net::Ipv4Addr;
use std::num::NonZeroUsize;
use std::thread;
use std::time::Duration;

/// Every rank's I/O bound: far above loopback delivery, and short enough that
/// a rank left waiting fails with `RecvTimedOut` before the 60 s nextest
/// termination.
const IO_BOUND: Duration = Duration::from_secs(20);

fn three_meshes() -> Vec<TcpMesh> {
    TcpMesh::create_loopback_cluster(
        NonZeroUsize::new(3).unwrap(),
        MeshDeadlines::DEFAULT.with_io(IO_BOUND),
    )
    .unwrap()
}

fn assert_poisoned(comm: &TcpCommunicator, rank: usize) {
    let next = comm.barrier();
    assert!(
        matches!(next, Err(TcpMeshError::LinkPoisoned { rank: r, .. }) if r == rank),
        "rank {rank}: expected LinkPoisoned, got {next:?}"
    );
}

/// Rank 1 is a raw mesh announcing `u64::MAX` elements to broadcast root 0.
/// The root reports the mismatch, rank 2 reports the root's status 0, and
/// rank 1 reads that status and then end of stream from the closed root.
#[test]
fn a_hostile_element_count_fails_the_root_and_every_rank_promptly() {
    const ELEMENTS: usize = 4;
    let mut meshes = three_meshes();
    let rank_2 = TcpCommunicator::new(meshes.pop().unwrap());
    let mut hostile = meshes.pop().unwrap();
    let root = TcpCommunicator::new(meshes.pop().unwrap());

    let [root_outcome, rank_2_outcome] = thread::scope(|scope| {
        let pending = [&root, &rank_2].map(|comm| {
            scope.spawn(move || {
                let backend = SequentialBackend::new();
                let mut tensor = Tensor::<f32, _>::zeros_on([ELEMENTS], &backend);
                comm.broadcast(&mut tensor, 0, &backend)
            })
        });
        hostile.send(0, &u64::MAX.to_le_bytes()).unwrap();
        let mut status = [9u8; 1];
        hostile.recv(0, &mut status).unwrap();
        assert_eq!(status, [0], "the root answers a mismatch with status 0");
        let mut byte = [0u8; 1];
        match hostile.recv(0, &mut byte) {
            Err(TcpMeshError::Recv { source, .. }) => {
                assert_eq!(source.kind(), ErrorKind::UnexpectedEof);
            }
            other => panic!("rank 1: expected end of stream from the root, got {other:?}"),
        }
        pending.map(|rank| rank.join().unwrap())
    });

    match root_outcome {
        Err(TcpMeshError::NumelMismatch {
            rank,
            peer,
            address,
            expected,
            received,
        }) => {
            assert_eq!(
                (rank, peer, expected, received),
                (0, 1, ELEMENTS as u64, u64::MAX)
            );
            assert_eq!(address.ip(), Ipv4Addr::LOCALHOST);
        }
        other => panic!("root: expected NumelMismatch, got {other:?}"),
    }
    match rank_2_outcome {
        Err(TcpMeshError::PeerReportedMismatch { rank, peer, .. }) => {
            assert_eq!((rank, peer), (2, 0));
        }
        other => panic!("rank 2: expected PeerReportedMismatch, got {other:?}"),
    }
    assert_poisoned(&root, 0);
    assert_poisoned(&rank_2, 2);
    for mut comm in [root, rank_2] {
        comm.shutdown();
    }
    hostile.shutdown();
}

/// Rank 1 all-gathers 3 elements against 4 on ranks 0 and 2. Ranks 0 and 1
/// exchange counts first and both report the mismatch; rank 2, waiting on
/// rank 0, reads end of stream when rank 0 poisons its links.
#[test]
fn an_all_gather_mismatch_ends_the_uninvolved_rank_s_wait() {
    let comms = three_meshes()
        .into_iter()
        .map(TcpCommunicator::new)
        .collect::<Vec<_>>();
    let lens = [4, 3, 4];

    let outcomes = thread::scope(|scope| {
        let pending = comms
            .iter()
            .enumerate()
            .map(|(rank, comm)| {
                scope.spawn(move || {
                    let backend = SequentialBackend::new();
                    let tensor = Tensor::<f32, _>::zeros_on([lens[rank]], &backend);
                    let mut output = (0..3)
                        .map(|_| Tensor::zeros_on([lens[rank]], &backend))
                        .collect::<Vec<_>>();
                    comm.all_gather(&tensor, &mut output, &backend)
                })
            })
            .collect::<Vec<_>>();
        pending
            .into_iter()
            .map(|rank| rank.join().unwrap())
            .collect::<Vec<_>>()
    });

    for (rank, peer) in [(0, 1), (1, 0)] {
        match &outcomes[rank] {
            Err(TcpMeshError::NumelMismatch {
                rank: reporting,
                peer: reported,
                address,
                expected,
                received,
            }) => {
                assert_eq!(
                    (*reporting, *reported, *expected, *received),
                    (rank, peer, lens[rank] as u64, lens[peer] as u64)
                );
                assert_eq!(address.ip(), Ipv4Addr::LOCALHOST);
            }
            other => panic!("rank {rank}: expected NumelMismatch, got {other:?}"),
        }
    }
    match &outcomes[2] {
        Err(TcpMeshError::Recv {
            rank, peer, source, ..
        }) => {
            assert_eq!((*rank, *peer), (2, 0));
            assert_eq!(source.kind(), ErrorKind::UnexpectedEof);
        }
        other => panic!("rank 2: expected Recv, got {other:?}"),
    }
    for (rank, comm) in comms.iter().enumerate() {
        assert_poisoned(comm, rank);
    }
    for mut comm in comms {
        comm.shutdown();
    }
}
