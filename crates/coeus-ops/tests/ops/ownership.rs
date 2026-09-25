//! Output ownership across operation families and CPU backends.

#[path = "ownership/accumulated_outputs.rs"]
mod accumulated_outputs;
#[path = "ownership/cpu/device_outputs.rs"]
mod cpu_device_outputs;
#[path = "ownership/cpu/state_updates.rs"]
mod cpu_state_updates;
#[path = "ownership/device_outputs.rs"]
mod device_outputs;
#[path = "ownership/optimizer.rs"]
mod optimizer;
#[path = "ownership/staggered.rs"]
mod staggered;
