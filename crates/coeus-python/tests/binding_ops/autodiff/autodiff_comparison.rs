fn find_working_python() -> Option<(String, String)> {
    let candidates = [
        "C:\\Users\\RyanClanton\\AppData\\Local\\Programs\\Python\\Python313\\python.exe",
        "python",
        "C:\\Users\\RyanClanton\\.unsloth\\studio\\unsloth_studio\\Scripts\\python.exe",
    ];

    for candidate in candidates {
        let original_path = std::env::var("PATH").unwrap_or_default();
        let clean_path = {
            let mut paths: Vec<String> = original_path.split(';').map(|s| s.to_string()).collect();
            if candidate.contains("unsloth") {
                paths.insert(
                    0,
                    "C:\\Users\\RyanClanton\\.unsloth\\studio\\unsloth_studio\\Scripts".to_string(),
                );
            }
            if !paths
                .iter()
                .any(|p| p.contains("ucrt64\\bin") || p.contains("ucrt64/bin"))
            {
                paths.push("D:\\msys64\\ucrt64\\bin".to_string());
            }
            paths.join(";")
        };

        let mut cmd = std::process::Command::new(candidate);
        cmd.args([
            "-c",
            "import torch; x = torch.tensor([1.0], requires_grad=True); x.sum().backward()",
        ]);
        cmd.env("PATH", &clean_path);
        cmd.env("CUDA_VISIBLE_DEVICES", "-1");
        cmd.env("KMP_DUPLICATE_LIB_OK", "TRUE");

        match cmd.output() {
            Ok(output) => {
                if output.status.success() {
                    return Some((candidate.to_string(), clean_path));
                } else {
                    println!(
                        "Candidate {} failed. stdout: {}, stderr: {}",
                        candidate,
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    );
                }
            }
            Err(e) => {
                println!("Candidate {} could not be executed: {:?}", candidate, e);
            }
        }
    }
    None
}

#[test]
fn test_autodiff_comparison_pytorch() {
    let Some((python_exe, clean_path)) = find_working_python() else {
        println!(
            "WARNING: No working PyTorch environment found. Skipping autodiff parity comparison."
        );
        return;
    };

    let temp_dir = std::env::temp_dir().join(format!("pycoeus_test_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&temp_dir);

    let exe_dir = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let target_debug_dir = exe_dir.parent().unwrap().to_path_buf();

    #[cfg(target_os = "windows")]
    {
        let src_dll = exe_dir.join("pycoeus.dll");
        let src_dll_parent = target_debug_dir.join("pycoeus.dll");
        let dst_pyd = temp_dir.join("pycoeus.pyd");
        if src_dll.exists() {
            let _ = std::fs::copy(&src_dll, &dst_pyd);
        } else if src_dll_parent.exists() {
            let _ = std::fs::copy(&src_dll_parent, &dst_pyd);
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        let src_so = exe_dir.join("libpycoeus.so");
        let src_so_parent = target_debug_dir.join("libpycoeus.so");
        let src_dylib = exe_dir.join("libpycoeus.dylib");
        let src_dylib_parent = target_debug_dir.join("libpycoeus.dylib");
        let dst_so = temp_dir.join("pycoeus.so");
        if src_so.exists() {
            let _ = std::fs::copy(&src_so, &dst_so);
        } else if src_so_parent.exists() {
            let _ = std::fs::copy(&src_so_parent, &dst_so);
        } else if src_dylib.exists() {
            let _ = std::fs::copy(&src_dylib, &dst_so);
        } else if src_dylib_parent.exists() {
            let _ = std::fs::copy(&src_dylib_parent, &dst_so);
        }
    }

    let python_path = temp_dir.to_str().unwrap().to_string();

    // Write compare script to a temp file
    let temp_file_path = std::env::temp_dir().join("coeus_autodiff_compare.py");
    let compare_script = r#"
import os
import sys

if sys.platform == 'win32':
    for path in os.environ.get('PATH', '').split(';'):
        if path and os.path.isdir(path) and ('msys64' in path.lower() or 'mingw' in path.lower()):
            try:
                os.add_dll_directory(path)
            except Exception:
                pass

import pycoeus

def main():
    try:
        import torch
    except ImportError:
        print('WARNING: PyTorch is not installed. Skipping autodiff comparison test.', file=sys.stderr)
        return

    import time

    try:
        # 1. Warm-up and basic correctness check
        data_x = [float(i) * 0.01 for i in range(128 * 256)]
        data_target = [1.0] * (128 * 64)

        # Create pycoeus inputs
        x_pyc = pycoeus.Tensor(data_x, [128, 256], requires_grad=True)
        linear_pyc = pycoeus.Linear(256, 64)
        
        # Extract weight & bias to synchronize with PyTorch
        w_data = linear_pyc.weight.data
        b_data = linear_pyc.bias.data if linear_pyc.bias else [0.0] * 64

        # Create PyTorch inputs
        x_torch = torch.tensor(data_x, dtype=torch.float64).reshape(128, 256).requires_grad_(True)
        w_torch = torch.tensor(w_data, dtype=torch.float64).reshape(64, 256).requires_grad_(True)
        b_torch = torch.tensor(b_data, dtype=torch.float64).requires_grad_(True)

        target_pyc_val = pycoeus.Tensor(data_target, [128, 64])
        target_torch_val = torch.tensor(data_target, dtype=torch.float64).reshape(128, 64)

        # Forward & Backward Core
        def run_pycoeus():
            out = linear_pyc.forward(x_pyc)
            act = pycoeus.relu(out)
            loss = pycoeus.mse_loss(act, target_pyc_val)
            loss.backward()
            return loss

        def run_torch():
            out = torch.nn.functional.linear(x_torch, w_torch, b_torch)
            act = torch.relu(out)
            loss = torch.nn.functional.mse_loss(act, target_torch_val)
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
        iters = 100

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
    finally:
        try:
            pycoeus.shutdown()
        except Exception:
            pass

main()
"#;

    std::fs::write(&temp_file_path, compare_script).unwrap();

    println!("[DIAGNOSTIC] Chosen Python: {}", python_exe);
    println!("[DIAGNOSTIC] Clean Path: {}", clean_path);
    println!("[DIAGNOSTIC] Python Path (temp_dir): {}", python_path);

    let mut cmd = std::process::Command::new(&python_exe);
    cmd.arg(&temp_file_path);
    cmd.env("PATH", &clean_path);
    cmd.env("PYTHONPATH", &python_path);
    cmd.env("CUDA_VISIBLE_DEVICES", "-1");
    cmd.env("KMP_DUPLICATE_LIB_OK", "TRUE");

    let output = cmd
        .output()
        .expect("Failed to execute python comparison child process");

    // Clean up temp files
    let _ = std::fs::remove_file(temp_file_path);
    let _ = std::fs::remove_dir_all(temp_dir);

    println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    println!("stderr: {}", String::from_utf8_lossy(&output.stderr));

    assert!(
        output.status.success(),
        "Python out-of-process comparison script failed"
    );
}
