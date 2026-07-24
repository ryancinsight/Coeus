//! TCP mesh-construction and rank-boundary panic contracts.

use super::super::super::support::single_rank_tcp_mesh;
use coeus_dist::TcpMesh;

#[test]
#[should_panic(expected = "send peer must differ from local rank")]
fn test_tcp_mesh_send_self_panics() {
    let mesh = single_rank_tcp_mesh();
    mesh.send(0, &[1u8]);
}

#[test]
#[should_panic(expected = "recv peer must differ from local rank")]
fn test_tcp_mesh_recv_self_panics() {
    let mesh = single_rank_tcp_mesh();
    let mut byte = [0u8; 1];
    mesh.recv(0, &mut byte);
}

#[test]
#[should_panic(expected = "send peer out of bounds")]
fn test_tcp_mesh_send_out_of_bounds_panics() {
    let mesh = single_rank_tcp_mesh();
    mesh.send(1, &[1u8]);
}

#[test]
#[should_panic(expected = "recv peer out of bounds")]
fn test_tcp_mesh_recv_out_of_bounds_panics() {
    let mesh = single_rank_tcp_mesh();
    let mut byte = [0u8; 1];
    mesh.recv(1, &mut byte);
}

#[test]
#[should_panic(expected = "rank must be less than world size")]
fn test_tcp_mesh_new_rank_out_of_bounds_panics() {
    let _mesh = TcpMesh::new(1, 1, &[]);
}

#[test]
#[should_panic(expected = "world size must be > 0")]
fn test_tcp_mesh_new_zero_world_size_panics() {
    let addresses: Vec<std::net::SocketAddr> = vec![];
    let _mesh = TcpMesh::new(0, 0, &addresses);
}

#[test]
#[should_panic(expected = "addresses list length must match world size")]
fn test_tcp_mesh_new_addresses_len_mismatch_panics() {
    let _mesh = TcpMesh::new(0, 2, &[]);
}
