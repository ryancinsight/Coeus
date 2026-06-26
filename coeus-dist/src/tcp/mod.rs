/// Socket-based collective operations implementing [`Communicator`](crate::Communicator).
pub mod collectives;
/// Fully-connected mesh of TCP streams connecting all ranks.
pub mod mesh;

pub use collectives::TcpCommunicator;
pub use mesh::TcpMesh;
