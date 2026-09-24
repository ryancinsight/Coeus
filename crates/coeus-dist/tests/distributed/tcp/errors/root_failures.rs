//! A collective root that misbehaves on the wire: a raw [`TcpMesh`] plays
//! rank 0 and speaks the broadcast protocol by hand, so each failure is
//! placed at an exact byte.

use coeus_core::SequentialBackend;
use coeus_dist::{Communicator, MeshDeadlines, TcpCommunicator, TcpMesh, TcpMeshError};
use coeus_tensor::Tensor;
use std::io::ErrorKind;
use std::net::Ipv4Addr;
use std::num::NonZeroUsize;
use std::thread;
use std::time::Duration;

/// Rank 1's I/O bound: far above loopback delivery, and short enough that a
/// rank left waiting fails with `RecvTimedOut` before the 60 s nextest
/// termination.
const RANK_1_IO: Duration = Duration::from_secs(20);

/// Four `f32` elements: the broadcast payload is 16 bytes.
const ELEMENTS: usize = 4;

/// A raw rank-0 mesh and a rank-1 communicator.
fn hand_driven_root() -> (TcpMesh, TcpCommunicator) {
    let mut meshes = TcpMesh::create_loopback_cluster(
        NonZeroUsize::new(2).unwrap(),
        MeshDeadlines::DEFAULT.with_io(RANK_1_IO),
    )
    .unwrap();
    let rank_1 = TcpCommunicator::new(meshes.pop().unwrap());
    (meshes.pop().unwrap(), rank_1)
}

/// Run rank 1's broadcast from root 0 while `root` drives rank 0's side.
fn broadcast_against(
    rank_1: &TcpCommunicator,
    root: impl FnOnce() + Send,
) -> Result<(), TcpMeshError> {
    thread::scope(|scope| {
        let pending = scope.spawn(|| {
            let backend = SequentialBackend::new();
            let mut tensor = Tensor::<f32, _>::zeros_on([ELEMENTS], &backend);
            rank_1.broadcast(&mut tensor, 0, &backend)
        });
        root();
        pending.join().unwrap()
    })
}

/// Read rank 1's element count, as the broadcast root does first.
fn read_element_count(root: &TcpMesh) {
    let mut count = [0u8; 8];
    root.recv(1, &mut count).unwrap();
    assert_eq!(u64::from_le_bytes(count), ELEMENTS as u64);
}

#[test]
fn an_invalid_handshake_status_is_a_typed_error_that_poisons_the_links() {
    const STATUS: u8 = 7;
    let (mut root, mut rank_1) = hand_driven_root();

    let outcome = broadcast_against(&rank_1, || {
        read_element_count(&root);
        root.send(1, &[STATUS]).unwrap();
    });
    match outcome {
        Err(TcpMeshError::InvalidStatus {
            rank,
            peer,
            address,
            status,
        }) => {
            assert_eq!((rank, peer, status), (1, 0, STATUS));
            // Rank 1 accepted rank 0, so this is rank 0's outgoing endpoint.
            assert_eq!(address.ip(), Ipv4Addr::LOCALHOST);
        }
        other => panic!("expected InvalidStatus, got {other:?}"),
    }

    // Rank 1 closed its links: the root sees end of stream, and rank 1's
    // next collective fails without touching the wire.
    let mut byte = [0u8; 1];
    assert!(matches!(
        root.recv(1, &mut byte),
        Err(TcpMeshError::Recv { source, .. }) if source.kind() == ErrorKind::UnexpectedEof
    ));
    assert!(matches!(
        rank_1.barrier(),
        Err(TcpMeshError::LinkPoisoned {
            rank: 1,
            peer: 0,
            ..
        })
    ));
    rank_1.shutdown();
    root.shutdown();
}

#[test]
fn a_root_lost_mid_payload_is_a_typed_error_on_the_receiver() {
    let (mut root, mut rank_1) = hand_driven_root();

    let outcome = broadcast_against(&rank_1, || {
        read_element_count(&root);
        root.send(1, &[1]).unwrap();
        // Half of the 16-byte payload, then the root goes away.
        root.send(1, &[0u8; 8]).unwrap();
        root.shutdown();
    });
    match outcome {
        Err(TcpMeshError::Recv {
            rank, peer, source, ..
        }) => {
            assert_eq!((rank, peer), (1, 0));
            assert_eq!(source.kind(), ErrorKind::UnexpectedEof);
        }
        other => panic!("expected Recv, got {other:?}"),
    }
    assert!(matches!(
        rank_1.barrier(),
        Err(TcpMeshError::LinkPoisoned { rank: 1, .. })
    ));
    rank_1.shutdown();
}
