/// Socket-based collective operations implementing [`Communicator`](crate::Communicator).
pub mod collectives;
/// Setup and peer I/O deadlines and the connection retry policy.
pub mod deadlines;
/// Typed failures of mesh setup and peer I/O.
pub mod error;
/// Fully-connected mesh of TCP streams connecting all ranks.
pub mod mesh;

pub use collectives::TcpCommunicator;
pub use deadlines::MeshDeadlines;
pub use error::{StreamStep, TcpMeshError};
pub use mesh::TcpMesh;
