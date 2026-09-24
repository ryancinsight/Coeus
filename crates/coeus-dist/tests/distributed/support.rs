//! Shared concurrency and loopback-mesh helpers for distributed contracts.

use coeus_dist::{MeshDeadlines, TcpMesh};
use std::num::NonZeroUsize;

pub(super) fn loopback_meshes(world_size: usize) -> Vec<TcpMesh> {
    let world_size =
        NonZeroUsize::new(world_size).expect("TCP test cluster requires a non-zero world size");
    TcpMesh::create_loopback_cluster(world_size, MeshDeadlines::DEFAULT)
        .expect("loopback TCP cluster setup")
}

pub(super) fn single_rank_tcp_mesh() -> TcpMesh {
    loopback_meshes(1)
        .into_iter()
        .next()
        .expect("one-rank loopback cluster must contain its rank")
}
