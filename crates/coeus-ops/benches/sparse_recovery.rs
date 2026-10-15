//! Counterbalanced, value-checked comparisons against Coeus cd8e984e.
//!
//! The owned bridge below is the exact former dispatch: three Vec copies,
//! CsrMatrix validation, a dense view, then the same Leto sparse kernel.
//! Forward comparisons include validation and output allocation on both sides.
//! Inputs are dyadic, so every expected operation is exact at these sizes.

use std::hint::black_box;
use std::time::{Duration, Instant};

use coeus_core::{
    Backend, CpuAddressableStorage, CpuAddressableStorageMut, MoiraiBackend, SequentialBackend,
};
use coeus_leto::CsrDispatch;
use coeus_sparse::CsrTensor;
use coeus_tensor::Tensor;
use leto::{ArrayView1, ArrayView2, Layout};
use leto_ops::{CsrMatrix, CsrView};

struct Fixture {
    rows: usize,
    shared: usize,
    width: usize,
    values: Vec<f64>,
    columns: Vec<usize>,
    offsets: Vec<usize>,
    dense: Vec<f64>,
    expected: Vec<f64>,
}

impl Fixture {
    fn new(rows: usize, shared: usize, entries: usize, width: usize) -> Self {
        let mut values = Vec::new();
        let mut columns = Vec::new();
        let mut offsets = vec![0];
        for row in 0..rows {
            if row % 7 != 0 {
                let start = row % (shared - entries + 1);
                for local in 0..entries {
                    values.push(
                        f64::from(i32::try_from((row + local) % 9).expect("small residue")) / 8.0
                            - 0.5,
                    );
                    columns.push(start + local);
                }
            }
            offsets.push(values.len());
        }
        let dense: Vec<_> = (0..shared * width)
            .map(|index| f64::from(i32::try_from(index % 19).expect("small residue")) / 8.0 - 1.0)
            .collect();
        let mut expected = vec![0.0; rows * width];
        for row in 0..rows {
            for column in 0..width {
                for p in offsets[row]..offsets[row + 1] {
                    expected[row * width + column] +=
                        values[p] * dense[columns[p] * width + column];
                }
            }
        }
        Self {
            rows,
            shared,
            width,
            values,
            columns,
            offsets,
            dense,
            expected,
        }
    }

    fn descriptor(&self) -> CsrDispatch<'_, f64> {
        CsrDispatch::new(
            &self.values,
            &self.columns,
            &self.offsets,
            self.rows,
            self.shared,
        )
    }

    fn owned_bridge(&self, output: &mut [f64]) {
        let owned = CsrMatrix::from_parts(
            self.values.clone(),
            self.columns.clone(),
            self.offsets.clone(),
            self.rows,
            self.shared,
        )
        .expect("canonical CSR");
        if self.width == 1 {
            let dense = ArrayView1::try_new(
                Layout::c_contiguous([self.shared]).expect("vector layout"),
                &self.dense,
            )
            .expect("vector view");
            leto_ops::spmv_into(&owned, &dense, output).expect("owned vector product");
        } else {
            let dense = ArrayView2::try_new(
                Layout::c_contiguous([self.shared, self.width]).expect("matrix layout"),
                &self.dense,
            )
            .expect("matrix view");
            leto_ops::spmm_into(&owned, &dense, output).expect("owned matrix product");
        }
    }

    fn borrowed_bridge(&self, output: &mut [f64]) {
        if self.width == 1 {
            coeus_leto::spmv_into(self.descriptor(), &self.dense, output)
                .expect("borrowed vector product");
        } else {
            coeus_leto::spmm_into(self.descriptor(), &self.dense, self.width, output)
                .expect("borrowed matrix product");
        }
    }

    fn tensor_inputs<B: Backend>(&self, backend: &B) -> (CsrTensor<f64, B>, Tensor<f64, B>) {
        let columns: Vec<_> = self
            .columns
            .iter()
            .map(|&i| i64::try_from(i).expect("fixture indices fit signed storage"))
            .collect();
        let offsets: Vec<_> = self
            .offsets
            .iter()
            .map(|&i| i64::try_from(i).expect("fixture offsets fit signed storage"))
            .collect();
        let a = CsrTensor::new(
            [self.rows, self.shared].into(),
            Tensor::from_slice_on([self.values.len()], &self.values, backend)
                .expect("values tensor"),
            Tensor::from_slice_on([columns.len()], &columns, backend).expect("columns tensor"),
            Tensor::from_slice_on([offsets.len()], &offsets, backend).expect("offsets tensor"),
        );
        let shape = if self.width == 1 {
            vec![self.shared]
        } else {
            vec![self.shared, self.width]
        };
        let dense = Tensor::from_slice_on(shape, &self.dense, backend).expect("dense tensor");
        (a, dense)
    }
}

fn time(operation: &mut impl FnMut(), iterations: usize) -> f64 {
    let started = Instant::now();
    for _ in 0..iterations {
        operation();
    }
    started.elapsed().as_secs_f64()
        / f64::from(u32::try_from(iterations).expect("bounded calibration"))
}

fn calibrate(operation: &mut impl FnMut()) -> usize {
    let mut iterations = 1;
    while iterations < 1 << 20 {
        let elapsed = time(operation, iterations);
        if elapsed * f64::from(u32::try_from(iterations).expect("bounded calibration"))
            >= Duration::from_millis(20).as_secs_f64()
        {
            break;
        }
        iterations *= 2;
    }
    iterations
}

fn paired(label: &str, mut baseline: impl FnMut(), mut candidate: impl FnMut()) {
    let baseline_iterations = calibrate(&mut baseline);
    let candidate_iterations = calibrate(&mut candidate);
    let mut baseline_samples = Vec::new();
    let mut candidate_samples = Vec::new();
    let mut ratios = Vec::new();
    for round in 0..21 {
        let (base, next) = if round % 2 == 0 {
            (
                time(&mut baseline, baseline_iterations),
                time(&mut candidate, candidate_iterations),
            )
        } else {
            let next = time(&mut candidate, candidate_iterations);
            (time(&mut baseline, baseline_iterations), next)
        };
        baseline_samples.push(base);
        candidate_samples.push(next);
        ratios.push(next / base);
    }
    baseline_samples.sort_by(f64::total_cmp);
    candidate_samples.sort_by(f64::total_cmp);
    ratios.sort_by(f64::total_cmp);
    tracing::info!(
        label,
        baseline_us = baseline_samples[10] * 1.0e6,
        candidate_us = candidate_samples[10] * 1.0e6,
        ratio = ratios[10],
        ratio_quartile_lower = ratios[5],
        ratio_quartile_upper = ratios[15],
        "21 alternating-order paired samples; ratio = candidate / baseline"
    );
}

fn bridge_comparison(fixture: &Fixture) {
    let mut base = vec![f64::NAN; fixture.expected.len()];
    let mut next = base.clone();
    fixture.owned_bridge(&mut base);
    fixture.borrowed_bridge(&mut next);
    assert_eq!(base, fixture.expected);
    assert_eq!(next, fixture.expected);
    paired(
        &format!(
            "bridge/{}x{}/rhs{}",
            fixture.rows, fixture.shared, fixture.width
        ),
        || {
            fixture.owned_bridge(&mut base);
            black_box(&base);
        },
        || {
            fixture.borrowed_bridge(&mut next);
            black_box(&next);
        },
    );
    assert_eq!(base, fixture.expected);
    assert_eq!(next, fixture.expected);
}

fn forward_product<B: Backend>(
    a: &CsrTensor<f64, B>,
    dense: &Tensor<f64, B>,
    width: usize,
    backend: &B,
) -> Tensor<f64, B>
where
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64>,
{
    if width == 1 {
        coeus_ops::spmv(a, dense, backend).expect("current Coeus SpMV")
    } else {
        coeus_ops::spmm(a, dense, backend).expect("current Coeus SpMM")
    }
}

fn forward_comparison<B: Backend>(fixture: &Fixture, backend: &B, backend_name: &str)
where
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64>,
{
    let (a, dense) = fixture.tensor_inputs(backend);
    let current = forward_product(&a, &dense, fixture.width, backend);
    assert_eq!(current.as_slice(), fixture.expected);
    let provider = || {
        let mut output = vec![0.0; fixture.expected.len()];
        fixture.borrowed_bridge(&mut output);
        output
    };
    assert_eq!(provider(), fixture.expected);
    paired(
        &format!(
            "forward/{backend_name}/{}x{}/rhs{}",
            fixture.rows, fixture.shared, fixture.width
        ),
        || {
            black_box(forward_product(&a, &dense, fixture.width, backend));
        },
        || {
            black_box(provider());
        },
    );
}

fn rounding_counterexample() {
    let values = [-1.0, 1.0 + f64::EPSILON];
    let columns = [0, 1];
    let offsets = [0, 2];
    let view = CsrView::from_slices(&values, &columns, &offsets, 1, 2).expect("canonical CSR");
    let mut dense = [1.0; 32];
    dense[16..].fill(1.0 - f64::EPSILON);
    let b = ArrayView2::try_new(Layout::c_contiguous([2, 16]).expect("dense layout"), &dense)
        .expect("dense view");
    let mut output = [0.0; 16];
    leto_ops::spmm_view_into(&view, &b, &mut output).expect("provider product");
    let separate = (values[1] * dense[16]) - 1.0;
    assert_eq!(separate.to_bits(), 0);
    tracing::info!(
        coeus_separate_bits = separate.to_bits(),
        leto_first_bits = output[0].to_bits(),
        leto_first = output[0],
        "rounding counterexample; separate result is +0"
    );
}

fn main() {
    tracing_subscriber::fmt()
        .with_ansi(false)
        .without_time()
        .init();
    tracing::info!(
        base = "cd8e984e0d9c4f3621bd89f7830646de845a22e5",
        architecture = std::env::consts::ARCH,
        threads = std::thread::available_parallelism()
            .expect("hardware thread count")
            .get(),
        "sparse recovery comparison"
    );
    rounding_counterexample();
    for (rows, shared, entries, width) in [
        (1024, 1024, 8, 1),
        (16384, 16384, 16, 1),
        (1024, 1024, 8, 16),
        (4096, 4096, 16, 64),
    ] {
        let fixture = Fixture::new(rows, shared, entries, width);
        bridge_comparison(&fixture);
        forward_comparison(&fixture, &SequentialBackend, "sequential");
        forward_comparison(&fixture, &MoiraiBackend, "moirai");
    }
}
