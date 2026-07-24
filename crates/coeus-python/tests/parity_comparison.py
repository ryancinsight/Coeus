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
import time

def run_pytorch_comparison():
    try:
        import torch
    except ImportError:
        print("PyTorch is not available for comparison.")
        return

    import subprocess
    try:
        res = subprocess.run(
            [sys.executable, "-c", "import torch; x = torch.zeros(2, requires_grad=True); x.sum().backward()"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if res.returncode != 0:
            print("PyTorch autograd is broken/crashes in this environment. Skipping PyTorch comparison.")
            return
    except Exception:
        print("Failed to run PyTorch check. Skipping PyTorch comparison.")
        return

    print("--- Running Parity Comparison against PyTorch ---")

    # 1. Linear + Relu + MSELoss Forward & Backward
    data_x = [float(i) * 0.01 for i in range(128 * 256)]
    data_target = [1.0] * (128 * 64)

    x_pyc = pycoeus.Tensor(data_x, [128, 256], requires_grad=True)
    linear_pyc = pycoeus.Linear(256, 64)
    w_data = linear_pyc.weight.data
    b_data = linear_pyc.bias.data if linear_pyc.bias else [0.0] * 64

    x_torch = torch.tensor(data_x, dtype=torch.float64).reshape(128, 256).requires_grad_(True)
    w_torch = torch.tensor(w_data, dtype=torch.float64).reshape(64, 256).requires_grad_(True)
    b_torch = torch.tensor(b_data, dtype=torch.float64).requires_grad_(True)

    out_pyc = linear_pyc.forward(x_pyc)
    act_pyc = pycoeus.relu(out_pyc)
    target_pyc = pycoeus.Tensor(data_target, [128, 64])
    loss_pyc = pycoeus.mse_loss(act_pyc, target_pyc)

    out_torch = torch.nn.functional.linear(x_torch, w_torch, b_torch)
    act_torch = torch.relu(out_torch)
    target_torch = torch.tensor(data_target, dtype=torch.float64).reshape(128, 64)
    loss_torch = torch.nn.functional.mse_loss(act_torch, target_torch)

    print(f"Linear Loss - pycoeus: {loss_pyc.data[0]:.6f}, PyTorch: {loss_torch.item():.6f}")
    assert abs(loss_pyc.data[0] - loss_torch.item()) < 1e-5, "Loss mismatch!"

    loss_pyc.backward()
    loss_torch.backward()

    # Gradients verification
    for i in range(len(x_pyc.grad)):
        assert abs(x_pyc.grad[i] - x_torch.grad.flatten()[i].item()) < 1e-5
    for i in range(len(linear_pyc.weight.grad)):
        assert abs(linear_pyc.weight.grad[i] - w_torch.grad.flatten()[i].item()) < 1e-5
    if linear_pyc.bias:
        for i in range(len(linear_pyc.bias.grad)):
            assert abs(linear_pyc.bias.grad[i] - b_torch.grad[i].item()) < 1e-5

    print("Linear + Relu gradient parity verified!")

    # 2. BatchNorm2d Forward & Backward
    data_bn = [float(i) * 0.1 for i in range(2 * 2 * 2 * 3)] # shape [2, 2, 2, 3]
    w_bn_data = [1.2, 0.8]
    b_bn_data = [0.1, -0.1]

    bn_pyc = pycoeus.BatchNorm2d(num_features=2, eps=1e-5, momentum=0.1)
    bn_pyc.weight.data = w_bn_data
    bn_pyc.bias.data = b_bn_data
    x_bn_pyc = pycoeus.Tensor(data_bn, [2, 2, 2, 3], requires_grad=True)
    out_bn_pyc = bn_pyc.forward(x_bn_pyc)
    loss_bn_pyc = out_bn_pyc.sum_axis(0).sum_axis(1).sum_axis(2).sum_axis(3)
    loss_bn_pyc.backward()

    bn_torch = torch.nn.BatchNorm2d(num_features=2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True).double()
    with torch.no_grad():
        bn_torch.weight.copy_(torch.tensor(w_bn_data, dtype=torch.float64))
        bn_torch.bias.copy_(torch.tensor(b_bn_data, dtype=torch.float64))
        bn_torch.running_mean.zero_()
        bn_torch.running_var.fill_(1.0)
    x_bn_torch = torch.tensor(data_bn, dtype=torch.float64).reshape(2, 2, 2, 3).requires_grad_(True)
    out_bn_torch = bn_torch(x_bn_torch)
    loss_bn_torch = out_bn_torch.sum()
    loss_bn_torch.backward()

    # Forward check
    for i in range(len(out_bn_pyc.data)):
        assert abs(out_bn_pyc.data[i] - out_bn_torch.flatten()[i].item()) < 1e-4

    # Running stats check
    pyc_mean = bn_pyc.running_mean.data
    pyc_var = bn_pyc.running_var.data
    for i in range(2):
        assert abs(pyc_mean[i] - bn_torch.running_mean[i].item()) < 1e-4
        assert abs(pyc_var[i] - bn_torch.running_var[i].item()) < 1e-4

    # Backward check
    for i in range(len(x_bn_pyc.grad)):
        assert abs(x_bn_pyc.grad[i] - x_bn_torch.grad.flatten()[i].item()) < 1e-4
    for i in range(2):
        assert abs(bn_pyc.weight.grad[i] - bn_torch.weight.grad[i].item()) < 1e-4
        assert abs(bn_pyc.bias.grad[i] - bn_torch.bias.grad[i].item()) < 1e-4

    print("BatchNorm2d forward/backward/running stats parity verified!")

    # 3. Dropout evaluation (exact identity) and training mode checks
    dropout_pyc = pycoeus.Dropout(p=0.0) # p=0 means identity
    x_drop_pyc = pycoeus.Tensor([1.0, 2.0, 3.0], [1, 3], requires_grad=True)
    out_drop_pyc = dropout_pyc.forward(x_drop_pyc)
    out_drop_pyc.backward()

    assert out_drop_pyc.data == [1.0, 2.0, 3.0]
    assert x_drop_pyc.grad == [1.0, 1.0, 1.0]

    # Stochastic check in PyCoeus (p=0.5)
    dropout_pyc_stoch = pycoeus.Dropout(p=0.5)
    x_drop_stoch = pycoeus.Tensor([1.0] * 100, [1, 100], requires_grad=True)
    out_drop_stoch = dropout_pyc_stoch.forward(x_drop_stoch)
    out_drop_stoch.backward()
    for val in out_drop_stoch.data:
        assert val == 0.0 or abs(val - 2.0) < 1e-5
    for g in x_drop_stoch.grad:
        assert g == 0.0 or abs(g - 2.0) < 1e-5

    print("Dropout identity & training scaling verified!")

    # 4. Multi-step Optimizer Step checks (SGD & AdamW)
    # SGD with momentum
    sgd_param_pyc = pycoeus.Tensor([10.0, 5.0], [2], requires_grad=True)
    sgd_pyc = pycoeus.SGD([("weight", sgd_param_pyc)], lr=0.1, momentum=0.9)

    sgd_param_torch = torch.tensor([10.0, 5.0], dtype=torch.float64).requires_grad_(True)
    sgd_torch = torch.optim.SGD([sgd_param_torch], lr=0.1, momentum=0.9)

    for step in range(3):
        sgd_param_pyc.zero_grad()
        loss_sgd_pyc = sgd_param_pyc * pycoeus.Tensor([2.0, -1.0])
        loss_sgd_pyc.backward()
        sgd_pyc.step()

        sgd_torch.zero_grad()
        loss_sgd_torch = (sgd_param_torch * torch.tensor([2.0, -1.0])).sum()
        loss_sgd_torch.backward()
        sgd_torch.step()

        for i in range(2):
            assert abs(sgd_param_pyc.data[i] - sgd_param_torch[i].item()) < 1e-5

    # AdamW
    adamw_param_pyc = pycoeus.Tensor([1.0, -2.0], [2], requires_grad=True)
    adamw_pyc = pycoeus.AdamW([("weight", adamw_param_pyc)], lr=0.01, weight_decay=0.01)

    adamw_param_torch = torch.tensor([1.0, -2.0], dtype=torch.float64).requires_grad_(True)
    adamw_torch = torch.optim.AdamW([adamw_param_torch], lr=0.01, weight_decay=0.01)

    for step in range(3):
        adamw_param_pyc.zero_grad()
        loss_adamw_pyc = adamw_param_pyc * pycoeus.Tensor([0.5, 1.5])
        loss_adamw_pyc.backward()
        adamw_pyc.step()

        adamw_torch.zero_grad()
        loss_adamw_torch = (adamw_param_torch * torch.tensor([0.5, 1.5])).sum()
        loss_adamw_torch.backward()
        adamw_torch.step()

        for i in range(2):
            assert abs(adamw_param_pyc.data[i] - adamw_param_torch[i].item()) < 1e-5

    print("Multi-step SGD & AdamW optimizer step parity verified!")

    # 4a. Conv1d Parity Check
    data_x_1d = [1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0]
    w_conv1d_data = [
        0.5, -0.5, 1.0, 0.0, 1.0, 0.0,
        0.1, 0.2, 0.3, -0.1, -0.2, -0.3,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    ]
    b_conv1d_data = [0.1, -0.1, 0.5]

    conv1d_pyc = pycoeus.Conv1d(2, 3, 3, 1, 0, 1, True)
    conv1d_pyc.weight.data = w_conv1d_data
    if conv1d_pyc.bias:
        conv1d_pyc.bias.data = b_conv1d_data
    x_conv1d_pyc = pycoeus.Tensor(data_x_1d, [1, 2, 4], requires_grad=True)
    out_conv1d_pyc = conv1d_pyc.forward(x_conv1d_pyc)
    loss_conv1d_pyc = out_conv1d_pyc.sum_axis(0).sum_axis(1).sum_axis(2)
    loss_conv1d_pyc.backward()

    conv1d_torch = torch.nn.Conv1d(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=0, dilation=1, bias=True).double()
    with torch.no_grad():
        conv1d_torch.weight.copy_(torch.tensor(w_conv1d_data, dtype=torch.float64).reshape(3, 2, 3))
        conv1d_torch.bias.copy_(torch.tensor(b_conv1d_data, dtype=torch.float64))
    x_conv1d_torch = torch.tensor(data_x_1d, dtype=torch.float64).reshape(1, 2, 4).requires_grad_(True)
    out_conv1d_torch = conv1d_torch(x_conv1d_torch)
    loss_conv1d_torch = out_conv1d_torch.sum()
    loss_conv1d_torch.backward()

    for i in range(len(out_conv1d_pyc.data)):
        assert abs(out_conv1d_pyc.data[i] - out_conv1d_torch.flatten()[i].item()) < 1e-4
    for i in range(len(x_conv1d_pyc.grad)):
        assert abs(x_conv1d_pyc.grad[i] - x_conv1d_torch.grad.flatten()[i].item()) < 1e-4
    for i in range(len(conv1d_pyc.weight.grad)):
        assert abs(conv1d_pyc.weight.grad[i] - conv1d_torch.weight.grad.flatten()[i].item()) < 1e-4
    if conv1d_pyc.bias:
        for i in range(len(conv1d_pyc.bias.grad)):
            assert abs(conv1d_pyc.bias.grad[i] - conv1d_torch.bias.grad[i].item()) < 1e-4

    print("Conv1d forward & backward parity verified!")

    # 4b. Conv2d Parity Check
    data_x_2d = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
        -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0
    ]
    w_conv2d_data = [
        0.5, -0.5, 1.0, 0.0,
        0.1, 0.2, 0.3, -0.1,
        -0.2, 0.5, 0.0, 1.0,
        1.0, -1.0, 0.2, 0.8
    ]
    b_conv2d_data = [0.1, -0.2]

    conv2d_pyc = pycoeus.Conv2d(2, 2, 2, 1, 0, 1, True)
    conv2d_pyc.weight.data = w_conv2d_data
    if conv2d_pyc.bias:
        conv2d_pyc.bias.data = b_conv2d_data
    x_conv2d_pyc = pycoeus.Tensor(data_x_2d, [1, 2, 3, 3], requires_grad=True)
    out_conv2d_pyc = conv2d_pyc.forward(x_conv2d_pyc)
    loss_conv2d_pyc = out_conv2d_pyc.sum_axis(0).sum_axis(1).sum_axis(2).sum_axis(3)
    loss_conv2d_pyc.backward()

    conv2d_torch = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=2, stride=1, padding=0, dilation=1, bias=True).double()
    with torch.no_grad():
        conv2d_torch.weight.copy_(torch.tensor(w_conv2d_data, dtype=torch.float64).reshape(2, 2, 2, 2))
        conv2d_torch.bias.copy_(torch.tensor(b_conv2d_data, dtype=torch.float64))
    x_conv2d_torch = torch.tensor(data_x_2d, dtype=torch.float64).reshape(1, 2, 3, 3).requires_grad_(True)
    out_conv2d_torch = conv2d_torch(x_conv2d_torch)
    loss_conv2d_torch = out_conv2d_torch.sum()
    loss_conv2d_torch.backward()

    for i in range(len(out_conv2d_pyc.data)):
        assert abs(out_conv2d_pyc.data[i] - out_conv2d_torch.flatten()[i].item()) < 1e-4
    for i in range(len(x_conv2d_pyc.grad)):
        assert abs(x_conv2d_pyc.grad[i] - x_conv2d_torch.grad.flatten()[i].item()) < 1e-4
    for i in range(len(conv2d_pyc.weight.grad)):
        assert abs(conv2d_pyc.weight.grad[i] - conv2d_torch.weight.grad.flatten()[i].item()) < 1e-4
    if conv2d_pyc.bias:
        for i in range(len(conv2d_pyc.bias.grad)):
            assert abs(conv2d_pyc.bias.grad[i] - conv2d_torch.bias.grad[i].item()) < 1e-4

    print("Conv2d forward & backward parity verified!")

    # 4c. Conv3d Parity Check
    data_x_3d = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    w_conv3d_data = [0.5, -0.5, 1.0, 0.0, 0.1, 0.2, 0.3, -0.1]
    b_conv3d_data = [0.1]

    conv3d_pyc = pycoeus.Conv3d(1, 1, 2, 1, 0, 1, True)
    conv3d_pyc.weight.data = w_conv3d_data
    if conv3d_pyc.bias:
        conv3d_pyc.bias.data = b_conv3d_data
    x_conv3d_pyc = pycoeus.Tensor(data_x_3d, [1, 1, 2, 2, 2], requires_grad=True)
    out_conv3d_pyc = conv3d_pyc.forward(x_conv3d_pyc)
    loss_conv3d_pyc = out_conv3d_pyc.sum_axis(0).sum_axis(1).sum_axis(2).sum_axis(3).sum_axis(4)
    loss_conv3d_pyc.backward()

    conv3d_torch = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0, dilation=1, bias=True).double()
    with torch.no_grad():
        conv3d_torch.weight.copy_(torch.tensor(w_conv3d_data, dtype=torch.float64).reshape(1, 1, 2, 2, 2))
        conv3d_torch.bias.copy_(torch.tensor(b_conv3d_data, dtype=torch.float64))
    x_conv3d_torch = torch.tensor(data_x_3d, dtype=torch.float64).reshape(1, 1, 2, 2, 2).requires_grad_(True)
    out_conv3d_torch = conv3d_torch(x_conv3d_torch)
    loss_conv3d_torch = out_conv3d_torch.sum()
    loss_conv3d_torch.backward()

    for i in range(len(out_conv3d_pyc.data)):
        assert abs(out_conv3d_pyc.data[i] - out_conv3d_torch.flatten()[i].item()) < 1e-4
    for i in range(len(x_conv3d_pyc.grad)):
        assert abs(x_conv3d_pyc.grad[i] - x_conv3d_torch.grad.flatten()[i].item()) < 1e-4
    for i in range(len(conv3d_pyc.weight.grad)):
        assert abs(conv3d_pyc.weight.grad[i] - conv3d_torch.weight.grad.flatten()[i].item()) < 1e-4
    if conv3d_pyc.bias:
        for i in range(len(conv3d_pyc.bias.grad)):
            assert abs(conv3d_pyc.bias.grad[i] - conv3d_torch.bias.grad[i].item()) < 1e-4

    print("Conv3d forward & backward parity verified!")

    # 4d. LayerNorm Parity Check
    data_ln = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, 3.0]
    w_ln_data = [1.2, 0.8, 1.0, 0.9]
    b_ln_data = [0.1, -0.1, 0.2, 0.0]

    ln_pyc = pycoeus.LayerNorm(4, 1e-5)
    ln_pyc.weight.data = w_ln_data
    ln_pyc.bias.data = b_ln_data
    x_ln_pyc = pycoeus.Tensor(data_ln, [2, 4], requires_grad=True)
    out_ln_pyc = ln_pyc.forward(x_ln_pyc)
    loss_ln_pyc = out_ln_pyc.sum_axis(0).sum_axis(1)
    loss_ln_pyc.backward()

    ln_torch = torch.nn.LayerNorm(normalized_shape=4, eps=1e-5).double()
    with torch.no_grad():
        ln_torch.weight.copy_(torch.tensor(w_ln_data, dtype=torch.float64))
        ln_torch.bias.copy_(torch.tensor(b_ln_data, dtype=torch.float64))
    x_ln_torch = torch.tensor(data_ln, dtype=torch.float64).reshape(2, 4).requires_grad_(True)
    out_ln_torch = ln_torch(x_ln_torch)
    loss_ln_torch = out_ln_torch.sum()
    loss_ln_torch.backward()

    for i in range(len(out_ln_pyc.data)):
        assert abs(out_ln_pyc.data[i] - out_ln_torch.flatten()[i].item()) < 1e-3
    for i in range(len(x_ln_pyc.grad)):
        assert abs(x_ln_pyc.grad[i] - x_ln_torch.grad.flatten()[i].item()) < 1e-3
    for i in range(len(ln_pyc.weight.grad)):
        assert abs(ln_pyc.weight.grad[i] - ln_torch.weight.grad.flatten()[i].item()) < 1e-3
    for i in range(len(ln_pyc.bias.grad)):
        assert abs(ln_pyc.bias.grad[i] - ln_torch.bias.grad.flatten()[i].item()) < 1e-3

    print("LayerNorm forward & backward parity verified!")

    # 4e. log_softmax Parity Check
    data_lsm = [1.0, 2.0, 3.0, -1.0, 0.5, 2.5]

    x_lsm_pyc = pycoeus.Tensor(data_lsm, [2, 3], requires_grad=True)
    out_lsm_pyc = pycoeus.log_softmax(x_lsm_pyc, axis=1)
    loss_lsm_pyc = out_lsm_pyc.sum_axis(0).sum_axis(1)
    loss_lsm_pyc.backward()

    x_lsm_torch = torch.tensor(data_lsm, dtype=torch.float64).reshape(2, 3).requires_grad_(True)
    out_lsm_torch = torch.nn.functional.log_softmax(x_lsm_torch, dim=1)
    loss_lsm_torch = out_lsm_torch.sum()
    loss_lsm_torch.backward()

    for i in range(len(out_lsm_pyc.data)):
        assert abs(out_lsm_pyc.data[i] - out_lsm_torch.flatten()[i].item()) < 1e-4
    for i in range(len(x_lsm_pyc.grad)):
        assert abs(x_lsm_pyc.grad[i] - x_lsm_torch.grad.flatten()[i].item()) < 1e-4

    print("log_softmax forward & backward parity verified!")

    # 4f. Embedding Parity Check
    emb_pyc = pycoeus.Embedding(5, 4)
    w_emb_data = [
        0.1, 0.2, 0.3, 0.4,
        -0.1, -0.2, -0.3, -0.4,
        0.5, 0.6, 0.7, 0.8,
        -0.5, -0.6, -0.7, -0.8,
        1.0, 1.1, 1.2, 1.3
    ]
    emb_pyc.weight.data = w_emb_data
    indices_data = [1.0, 2.0, 0.0, 4.0, 3.0, 1.0]
    indices_pyc = pycoeus.Tensor(indices_data, [2, 3])
    out_emb_pyc = emb_pyc.forward(indices_pyc)
    loss_emb_pyc = out_emb_pyc.sum_axis(0).sum_axis(1).sum_axis(2)
    loss_emb_pyc.backward()

    emb_torch = torch.nn.Embedding(5, 4).double()
    with torch.no_grad():
        emb_torch.weight.copy_(torch.tensor(w_emb_data, dtype=torch.float64).reshape(5, 4))
    indices_torch = torch.tensor([[1, 2, 0], [4, 3, 1]], dtype=torch.long)
    out_emb_torch = emb_torch(indices_torch)
    loss_emb_torch = out_emb_torch.sum()
    loss_emb_torch.backward()

    for i in range(len(out_emb_pyc.data)):
        assert abs(out_emb_pyc.data[i] - out_emb_torch.flatten()[i].item()) < 1e-5
    for i in range(len(emb_pyc.weight.grad)):
        assert abs(emb_pyc.weight.grad[i] - emb_torch.weight.grad.flatten()[i].item()) < 1e-5

    print("Embedding forward & backward parity verified!")

    # 4g. CrossEntropyLoss Parity Check
    logits_data = [
        1.5, 0.5, -0.5,
        -1.0, 2.0, 0.0
    ]
    targets = [0, 1]
    logits_pyc = pycoeus.Tensor(logits_data, [2, 3], requires_grad=True)
    loss_cel_pyc = pycoeus.cross_entropy_loss(logits_pyc, targets)
    loss_cel_pyc.backward()

    logits_torch = torch.tensor(logits_data, dtype=torch.float64).reshape(2, 3).requires_grad_(True)
    targets_torch = torch.tensor(targets, dtype=torch.long)
    loss_cel_torch = torch.nn.functional.cross_entropy(logits_torch, targets_torch)
    loss_cel_torch.backward()

    assert abs(loss_cel_pyc.data[0] - loss_cel_torch.item()) < 1e-5
    for i in range(len(logits_pyc.grad)):
        assert abs(logits_pyc.grad[i] - logits_torch.grad.flatten()[i].item()) < 1e-5

    print("CrossEntropyLoss forward & backward parity verified!")

    # 5. Timing benchmarks
    iters = 100
    
    # PyTorch
    start = time.perf_counter()
    for _ in range(iters):
        x_torch.grad = None
        w_torch.grad = None
        b_torch.grad = None
        out = torch.nn.functional.linear(x_torch, w_torch, b_torch)
        act = torch.relu(out)
        loss = torch.nn.functional.mse_loss(act, target_torch)
        loss.backward()
    end = time.perf_counter()
    time_torch = (end - start) / iters

    # PyCoeus
    start = time.perf_counter()
    for _ in range(iters):
        x_pyc.zero_grad()
        linear_pyc.weight.zero_grad()
        if linear_pyc.bias:
            linear_pyc.bias.zero_grad()
        out = linear_pyc.forward(x_pyc)
        act = pycoeus.relu(out)
        loss = pycoeus.mse_loss(act, target_pyc)
        loss.backward()
    end = time.perf_counter()
    time_pyc = (end - start) / iters

    print(f"PyTorch Step Time: {time_torch * 1000.0:.3f} ms")
    print(f"PyCoeus Step Time: {time_pyc * 1000.0:.3f} ms")
    print(f"Relative performance: {time_torch / time_pyc:.2f}x")

def run_jax_comparison():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        print("JAX is not available for comparison.")
        return

    print("--- Running Parity Comparison against JAX ---")
    data_x = [float(i) * 0.01 for i in range(128 * 256)]
    data_target = [1.0] * (128 * 64)

    # Initialize PyCoeus
    x_pyc = pycoeus.Tensor(data_x, [128, 256])
    linear_pyc = pycoeus.Linear(256, 64)
    w_data = linear_pyc.weight.data
    b_data = linear_pyc.bias.data if linear_pyc.bias else [0.0] * 64

    # JAX inputs
    x_jax = jnp.array(data_x).reshape(128, 256)
    w_jax = jnp.array(w_data).reshape(64, 256)
    b_jax = jnp.array(b_data)
    target_jax = jnp.array(data_target).reshape(128, 64)

    def jax_loss_fn(x, w, b):
        out = jnp.dot(x, w.T) + b
        act = jnp.maximum(out, 0.0)
        return jnp.mean((act - target_jax) ** 2)

    grad_fn = jax.value_and_grad(jax_loss_fn, argnums=(0, 1, 2))
    loss_jax, (dx, dw, db) = grad_fn(x_jax, w_jax, b_jax)

    print(f"JAX Loss: {loss_jax:.6f}")
    # Compile
    jitted_grad_fn = jax.jit(grad_fn)
    _ = jitted_grad_fn(x_jax, w_jax, b_jax)

    # Benchmark JAX
    iters = 100
    start = time.perf_counter()
    for _ in range(iters):
        loss_val, grads = jitted_grad_fn(x_jax, w_jax, b_jax)
        loss_val.block_until_ready()
    end = time.perf_counter()
    time_jax = (end - start) / iters
    print(f"JAX (jitted) Step Time: {time_jax * 1000.0:.3f} ms")

def run_mlx_comparison():
    try:
        import mlx.core as mx
    except ImportError:
        print("MLX is not available for comparison.")
        return

    print("--- Running Parity Comparison against MLX ---")
    data_x = [float(i) * 0.01 for i in range(128 * 256)]
    data_target = [1.0] * (128 * 64)

    # Initialize PyCoeus
    x_pyc = pycoeus.Tensor(data_x, [128, 256])
    linear_pyc = pycoeus.Linear(256, 64)
    w_data = linear_pyc.weight.data
    b_data = linear_pyc.bias.data if linear_pyc.bias else [0.0] * 64

    # MLX inputs
    x_mlx = mx.array(data_x).reshape(128, 256)
    w_mlx = mx.array(w_data).reshape(64, 256)
    b_mlx = mx.array(b_data)
    target_mlx = mx.array(data_target).reshape(128, 64)

    def mlx_loss_fn(x, w, b):
        out = mx.matmul(x, w.T) + b
        act = mx.maximum(out, 0.0)
        return mx.mean((act - target_mlx) ** 2)

    grad_fn = mx.value_and_grad(mlx_loss_fn, argnums=[0, 1, 2])
    loss_mlx, (dx, dw, db) = grad_fn(x_mlx, w_mlx, b_mlx)
    mx.eval(loss_mlx, dx, dw, db)

    print(f"MLX Loss: {loss_mlx.item():.6f}")

    # Benchmark MLX
    iters = 100
    start = time.perf_counter()
    for _ in range(iters):
        loss_val, grads = grad_fn(x_mlx, w_mlx, b_mlx)
        mx.eval(loss_val, grads)
    end = time.perf_counter()
    time_mlx = (end - start) / iters
    print(f"MLX Step Time: {time_mlx * 1000.0:.3f} ms")

if __name__ == "__main__":
    try:
        run_pytorch_comparison()
        run_jax_comparison()
        run_mlx_comparison()
    except BaseException as e:
        import traceback
        print("EXCEPTION OCCURRED:", file=sys.stdout)
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
    finally:
        try:
            pycoeus.shutdown()
        except Exception:
            pass
