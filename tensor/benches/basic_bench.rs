use criterion::{criterion_group, criterion_main, Criterion};

fn basic_benchmark(c: &mut Criterion) {
    c.bench_function("basic_test", |b| {
        b.iter(|| {
            let result = 2 + 2;
            assert_eq!(result, 4);
        });
    });
}

criterion_group!(benches, basic_benchmark);
criterion_main!(benches);
