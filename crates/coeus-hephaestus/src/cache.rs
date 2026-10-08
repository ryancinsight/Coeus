use std::sync::OnceLock;

use hephaestus_core::{HephaestusError, Result};

/// Borrow a process-wide cached value, initializing it on first use.
///
/// This is `OnceLock::get_or_try_init`, which stays unstable (`once_cell_try`),
/// so the bridge keeps one copy of the protocol instead of one per backend:
/// a fast read, a fallible init whose failure is never cached (the next call
/// retries), and a race-tolerant publish where a lost `set` still returns the
/// winner through the final read. `unavailable` names the value for the
/// defensive [`HephaestusError::DeviceUnavailable`]; that arm is unreachable —
/// a `set` either publishes or proves a winner exists — but the type needs a
/// total function.
pub fn get_or_try_init<T: Send + Sync>(
    cache: &'static OnceLock<T>,
    unavailable: &'static str,
    init: impl FnOnce() -> Result<T>,
) -> Result<&'static T> {
    if let Some(value) = cache.get() {
        return Ok(value);
    }
    let candidate = init()?;
    let _ = cache.set(candidate);
    cache
        .get()
        .ok_or_else(|| HephaestusError::DeviceUnavailable {
            message: unavailable.to_owned(),
        })
}
