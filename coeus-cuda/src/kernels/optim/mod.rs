pub mod sgd;
pub mod adam;
pub mod rmsprop;
pub mod adagrad;

pub use sgd::launch_sgd_step;
pub use adam::launch_adam_step;
pub use rmsprop::launch_rmsprop_step;
pub use adagrad::launch_adagrad_step;
