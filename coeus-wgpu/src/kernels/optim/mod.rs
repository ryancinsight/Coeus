pub mod sgd;
pub mod adam;
pub mod adamw;
pub mod rmsprop;
pub mod adagrad;

pub use sgd::dispatch_sgd_step;
pub use adam::dispatch_adam_step;
pub use adamw::dispatch_adamw_step;
pub use rmsprop::dispatch_rmsprop_step;
pub use adagrad::dispatch_adagrad_step;

