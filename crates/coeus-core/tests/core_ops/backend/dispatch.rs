//! Real backend completion on callback panic, including parallel admission.

use coeus_core::{Backend, MoiraiBackend, SequentialBackend};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tracing_subscriber::fmt::format::FmtSpan;

struct Invocation<'a> {
    active: &'a AtomicUsize,
    finished: &'a AtomicUsize,
}

impl Drop for Invocation<'_> {
    fn drop(&mut self) {
        self.finished.fetch_add(1, Ordering::SeqCst);
        self.active.fetch_sub(1, Ordering::SeqCst);
    }
}

fn panic_completion<B: Backend>(backend: B) {
    let _trace = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .without_time()
            .with_test_writer()
            .with_span_events(FmtSpan::FULL)
            .finish(),
    );
    let _case =
        tracing::info_span!("backend_unwind", backend = std::any::type_name::<B>()).entered();
    let (visits, active, finished) = {
        let _phase = tracing::info_span!("arrange").entered();
        // 4097 crosses the actual MoiraiBackend threshold of 4096, retaining
        // the real scheduler/admission path. Miri uses the same input domain.
        (
            Arc::new((0..4097).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>()),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
        )
    };
    let panic = {
        let _phase = tracing::info_span!("act").entered();
        let worker_visits = Arc::clone(&visits);
        let worker_active = Arc::clone(&active);
        let worker_finished = Arc::clone(&finished);
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            backend.parallel_for(0, visits.len(), move |row| {
                worker_active.fetch_add(1, Ordering::SeqCst);
                let _invocation = Invocation {
                    active: &worker_active,
                    finished: &worker_finished,
                };
                worker_visits[row].fetch_add(1, Ordering::SeqCst);
                assert_ne!(row, 3, "callback unwind witness");
            });
        }))
    };
    let _phase = tracing::info_span!("assert").entered();
    assert!(panic.is_err(), "real callback panic must propagate");
    assert_eq!(
        active.load(Ordering::SeqCst),
        0,
        "no admitted callback escapes unwind"
    );
    let admitted: usize = visits
        .iter()
        .map(|count| count.load(Ordering::SeqCst))
        .sum();
    assert!(visits.iter().all(|count| count.load(Ordering::SeqCst) <= 1));
    assert_eq!(visits[3].load(Ordering::SeqCst), 1);
    assert_eq!(finished.load(Ordering::SeqCst), admitted);
    assert_eq!(
        Arc::strong_count(&visits),
        1,
        "backend retains no callback capture"
    );
    assert_eq!(
        Arc::strong_count(&active),
        1,
        "backend retains no active-work capture"
    );
    assert_eq!(
        Arc::strong_count(&finished),
        1,
        "backend retains no completion capture"
    );
}

#[test]
fn sequential_callbacks_complete_before_unwind_returns() {
    panic_completion(SequentialBackend::new());
}

#[test]
fn moirai_callbacks_complete_before_unwind_returns() {
    panic_completion(MoiraiBackend::new());
    // Nextest gives Miri an isolated test process. Retire its real global
    // workers only after the callback/completion/capture assertions: doing
    // this earlier could hide a callback that outlives parallel_for. Native
    // libtest can share the runtime with other cases and must keep it usable.
    #[cfg(miri)]
    moirai::global().shutdown();
}
