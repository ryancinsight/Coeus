//! Global runtime configuration for Coeus backends
//!
//! This module provides centralized management for global settings such as
//! the random seed and the number of compute threads.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::sync::OnceLock;

/// Global seed for reproducible results
static GLOBAL_SEED: AtomicU64 = AtomicU64::new(0);

/// Number of threads for CPU operations
static NUM_THREADS: OnceLock<Mutex<usize>> = OnceLock::new();

fn get_num_threads_mutex() -> &'static Mutex<usize> {
    NUM_THREADS.get_or_init(|| Mutex::new(num_cpus::get()))
}

/// Set the random seed for reproducible results
pub fn set_manual_seed(seed: u64) {
    GLOBAL_SEED.store(seed, Ordering::SeqCst);
    // In the future, this should also re-initialize any global RNG instances
}

/// Get the current random seed
pub fn get_manual_seed() -> u64 {
    GLOBAL_SEED.load(Ordering::SeqCst)
}

/// Set the number of threads for CPU operations
pub fn set_num_threads(num: usize) {
    if let Ok(mut threads) = get_num_threads_mutex().lock() {
        *threads = num;
        
        // If rayon is being used, we should ideally reconfigure the global thread pool.
        // Note: Rayon global pool can only be initialized once.
        // For a more flexible approach, we might need a custom thread pool.
    }
}

/// Get the current number of threads for CPU operations
pub fn get_num_threads() -> usize {
    get_num_threads_mutex().lock().map(|t| *t).unwrap_or_else(|_| num_cpus::get())
}
