/// Socket-based collective operations implementing [`Communicator`](crate::Communicator).
pub mod collectives;
/// Typed failures of mesh setup and peer I/O.
pub mod error;
/// Fully-connected mesh of TCP streams connecting all ranks.
pub mod mesh;
/// Bounded mesh setup: the setup deadline and connection retry policy.
pub mod setup;

pub use collectives::TcpCommunicator;
pub use error::TcpMeshError;
pub use mesh::TcpMesh;
pub use setup::SetupDeadline;
