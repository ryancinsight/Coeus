use pyo3::prelude::*;
use pyo3::types::PyDict;

#[test]
fn test_autodiff_comparison_pytorch() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        // Create the module and populate it.
        let pycoeus_module = pyo3::types::PyModule::new(py, "pycoeus").unwrap();
        pycoeus::pycoeus(&pycoeus_module).unwrap();

        // Inject the module into sys.modules so Python code can import it
        let sys = py.import("sys").unwrap();
        let modules_any = sys.getattr("modules").unwrap();
        let modules = modules_any.downcast::<PyDict>().unwrap();
        modules.set_item("pycoeus", &pycoeus_module).unwrap();

        // Run python comparison script
        let compare_script = cr"
import pycoeus
import sys

def main():
    try:
        import torch
    except ImportError:
        print('WARNING: PyTorch is not installed. Skipping autodiff comparison test.', file=sys.stderr)
        return

    import time

    try:
        # 1. Warm-up and basic correctness check
        data_x = [float(i) * 0.01 for i in range(32 * 64)]
        data_target = [1.0] * (32 * 10)

        # Create pycoeus inputs
        x_pyc = pycoeus.Tensor(data_x, [32, 64], requires_grad=True)
        linear_pyc = pycoeus.Linear(64, 10)
        
        # Extract weight & bias to synchronize with PyTorch
        w_data = linear_pyc.weight.data
        b_data = linear_pyc.bias.data if linear_pyc.bias else [0.0] * 10

        # Create PyTorch inputs
        x_torch = torch.tensor(data_x, dtype=torch.float64).reshape(32, 64).requires_grad_(True)
        w_torch = torch.tensor(w_data, dtype=torch.float64).reshape(10, 64).requires_grad_(True)
        b_torch = torch.tensor(b_data, dtype=torch.float64).requires_grad_(True)

        # Forward & Backward Core
        def run_pycoeus():
            out = linear_pyc.forward(x_pyc)
            act = pycoeus.relu(out)
            target = pycoeus.Tensor(data_target, [32, 10])
            loss = pycoeus.mse_loss(act, target)
            loss.backward()
            return loss

        def run_torch():
            out = torch.nn.functional.linear(x_torch, w_torch, b_torch)
            act = torch.relu(out)
            target = torch.tensor(data_target, dtype=torch.float64).reshape(32, 10)
            loss = torch.nn.functional.mse_loss(act, target)
            loss.backward()
            return loss

        # Synchronize gradients & verify parity
        loss_pyc = run_pycoeus()
        loss_torch = run_torch()

        assert abs(loss_pyc.data[0] - loss_torch.item()) < 1e-4, f'Loss mismatch: {loss_pyc.data[0]} vs {loss_torch.item()}'

        # Verify gradients
        x_pyc_grad = x_pyc.grad
        linear_pyc_w_grad = linear_pyc.weight.grad
        linear_pyc_b_grad = linear_pyc.bias.grad if linear_pyc.bias else None

        # Check parity of B's gradients
        for i in range(len(linear_pyc_b_grad)):
            assert abs(linear_pyc_b_grad[i] - b_torch.grad[i].item()) < 1e-4

        # Check parity of W's gradients
        for i in range(len(linear_pyc_w_grad)):
            assert abs(linear_pyc_w_grad[i] - w_torch.grad.flatten()[i].item()) < 1e-4

        # Check parity of input X's gradients
        for i in range(len(x_pyc_grad)):
            assert abs(x_pyc_grad[i] - x_torch.grad.flatten()[i].item()) < 1e-4

        print('\n[BENCHMARK] AUTODIFF PARITY VERIFIED SUCCESSFULLY!', flush=True)

        # 2. Timing benchmarks
        iters = 200

        # Benchmark PyTorch
        start_torch = time.perf_counter()
        for _ in range(iters):
            x_torch.grad = None
            w_torch.grad = None
            b_torch.grad = None
            loss = run_torch()
        end_torch = time.perf_counter()
        time_torch = (end_torch - start_torch) / iters

        # Benchmark PyCoeus
        start_pyc = time.perf_counter()
        for _ in range(iters):
            x_pyc.zero_grad()
            linear_pyc.weight.zero_grad()
            if linear_pyc.bias:
                linear_pyc.bias.zero_grad()
            loss = run_pycoeus()
        end_pyc = time.perf_counter()
        time_pyc = (end_pyc - start_pyc) / iters

        print('\n======================================================', flush=True)
        print('          AUTODIFF COMPARISON BENCHMARK RESULT        ', flush=True)
        print('======================================================', flush=True)
        print(f'PyTorch average step time: {time_torch * 1000.0:.3f} ms', flush=True)
        print(f'PyCoeus average step time: {time_pyc * 1000.0:.3f} ms', flush=True)
        speedup = time_torch / time_pyc
        print(f'Monomorphized PyCoeus vs PyTorch Speedup: {speedup:.2f}x', flush=True)
        print('======================================================\n', flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

main()
";

        if let Err(e) = py.run(compare_script, None, None) {
            panic!("Python comparison script failed: {:?}", e);
        }
    });
}
