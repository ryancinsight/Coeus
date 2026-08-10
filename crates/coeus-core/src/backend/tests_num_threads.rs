// ── num_threads cache invariant tests ──
//
// Verify the cached hardware-parallelism value (once per process) is
// identical to the live Themis/syscall probe result, and that repeated
// `num_threads()` reads are stable across calls.

#[cfg(test)]
mod tests {
    use crate::backend::{ComputeBackend, MoiraiBackend, SequentialBackend};
    use std::collections::HashSet;

    #[test]
    fn moirai_num_threads_matches_available_parallelism() {
        let backend = MoiraiBackend::new();
        let cached = backend.num_threads();
        let live = themis::CpuTopology::detect()
            .map(|topology| topology.logical_processors())
            .or_else(|| std::thread::available_parallelism().ok().map(usize::from))
            .unwrap_or(1)
            .max(1);
        assert_eq!(
            cached, live,
            "MoiraiBackend::num_threads() ({cached}) must equal topology-probed parallelism ({live})"
        );
    }

    #[test]
    fn moirai_num_threads_is_stable_across_calls() {
        let backend = MoiraiBackend::new();
        let first = backend.num_threads();
        // Repeated reads return the same cached value (relaxed load), even
        // under contention. Implicit invariant: `available_parallelism()`
        // is monotonic within a process.
        let mut seen = HashSet::new();
        for _ in 0..1024 {
            seen.insert(backend.num_threads());
        }
        assert_eq!(
            seen.len(),
            1,
            "MoiraiBackend::num_threads() must return a single stable value; saw {seen:?}"
        );
        assert_eq!(*seen.iter().next().unwrap(), first);
    }

    #[test]
    fn sequential_num_threads_is_one() {
        let backend = SequentialBackend::new();
        assert_eq!(backend.num_threads(), 1);
    }

    #[test]
    fn moirai_num_threads_is_stable_under_concurrent_load() {
        use std::sync::{Arc, Barrier};
        use std::thread;
        let backend = MoiraiBackend::new();
        let first = backend.num_threads();
        // Spawn 8 threads that simultaneously hammer `num_threads()` so
        // any non-atomic snapshot would race against the writer.
        let threads = 8;
        let barrier = Arc::new(Barrier::new(threads));
        let mut handles = Vec::with_capacity(threads);
        let results = Arc::new(std::sync::Mutex::new(Vec::new()));
        for _ in 0..threads {
            let b = Arc::clone(&barrier);
            let res = Arc::clone(&results);
            handles.push(thread::spawn(move || {
                let backend = MoiraiBackend::new();
                b.wait();
                let mut local = Vec::with_capacity(2048);
                for _ in 0..2048 {
                    local.push(backend.num_threads());
                }
                res.lock().unwrap().extend(local);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let all: HashSet<usize> = results.lock().unwrap().iter().copied().collect();
        assert_eq!(
            all.len(),
            1,
            "MoiraiBackend::num_threads() must be consistent under concurrent hammer; saw {all:?}"
        );
        assert_eq!(*all.iter().next().unwrap(), first);
    }

    #[test]
    fn moirai_num_threads_is_at_least_one() {
        let backend = MoiraiBackend::new();
        assert!(backend.num_threads() >= 1);
    }
}
