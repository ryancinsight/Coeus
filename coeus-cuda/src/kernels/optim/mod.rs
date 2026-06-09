pub mod adagrad;
pub mod adam;
pub mod adamw;
pub mod rmsprop;
pub mod sgd;

pub use adagrad::launch_adagrad_step;
pub use adam::launch_adam_step;
pub use adamw::launch_adamw_step;
pub use rmsprop::launch_rmsprop_step;
pub use sgd::launch_sgd_step;
