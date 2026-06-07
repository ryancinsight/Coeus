use pyo3::prelude::*;
use pyo3::types::PyDict;

#[test]
fn test_pycoeus_bindings() {
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

        // Run python verification script
        let test_script = c"
import pycoeus
import sys
import traceback

try:
    # 1. Tensor creation and operations
    x = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2], requires_grad=True)
    assert x.shape == [2, 2], f'x.shape is {x.shape}'
    assert x.data == [1.0, 2.0, 3.0, 4.0], f'x.data is {x.data}'

    # 2. Math methods
    y = x.exp()
    assert y.shape == [2, 2], f'y.shape is {y.shape}'

    z = x.log()
    assert z.shape == [2, 2], f'z.shape is {z.shape}'

    s = x.sum_axis(0)
    assert s.shape == [1, 2], f's.shape is {s.shape}'

    m = x.mean_axis(1)
    assert m.shape == [2, 1], f'm.shape is {m.shape}'

    # 3. Activations and backward pass
    out = pycoeus.relu(x)
    loss = pycoeus.mse_loss(out, pycoeus.Tensor([2.0, 2.0, 2.0, 2.0], [2, 2]))
    loss.backward()

    # Test silu activation
    out_silu = pycoeus.silu(x)
    assert out_silu.shape == [2, 2], f'out_silu.shape is {out_silu.shape}'
    loss_silu = pycoeus.mse_loss(out_silu, pycoeus.Tensor([2.0, 2.0, 2.0, 2.0], [2, 2]))
    loss_silu.backward()

    # Test mish activation
    out_mish = pycoeus.mish(x)
    assert out_mish.shape == [2, 2], f'out_mish.shape is {out_mish.shape}'
    loss_mish = pycoeus.mse_loss(out_mish, pycoeus.Tensor([2.0, 2.0, 2.0, 2.0], [2, 2]))
    loss_mish.backward()

    # Test elu activation
    out_elu = pycoeus.elu(x)
    assert out_elu.shape == [2, 2], f'out_elu.shape is {out_elu.shape}'
    loss_elu = pycoeus.mse_loss(out_elu, pycoeus.Tensor([2.0, 2.0, 2.0, 2.0], [2, 2]))
    loss_elu.backward()

    # Test softplus activation
    out_softplus = pycoeus.softplus(x)
    assert out_softplus.shape == [2, 2], f'out_softplus.shape is {out_softplus.shape}'
    loss_softplus = pycoeus.mse_loss(out_softplus, pycoeus.Tensor([2.0, 2.0, 2.0, 2.0], [2, 2]))
    loss_softplus.backward()

    # Test gelu_tanh activation
    out_gelu_tanh = pycoeus.gelu_tanh(x)
    assert out_gelu_tanh.shape == [2, 2], f'out_gelu_tanh.shape is {out_gelu_tanh.shape}'
    loss_gelu_tanh = pycoeus.mse_loss(out_gelu_tanh, pycoeus.Tensor([2.0, 2.0, 2.0, 2.0], [2, 2]))
    loss_gelu_tanh.backward()

    # Test leaky_relu activation
    out_leaky = pycoeus.leaky_relu(x, negative_slope=0.1)
    assert out_leaky.shape == [2, 2], f'out_leaky.shape is {out_leaky.shape}'
    loss_leaky = pycoeus.mse_loss(out_leaky, pycoeus.Tensor([2.0, 2.0, 2.0, 2.0], [2, 2]))
    loss_leaky.backward()

    # Test binary_cross_entropy loss
    pred_bce = pycoeus.Tensor([0.1, 0.9, 0.8, 0.2], [2, 2], requires_grad=True)
    target_bce = pycoeus.Tensor([0.0, 1.0, 1.0, 0.0], [2, 2])
    loss_bce = pycoeus.binary_cross_entropy(pred_bce, target_bce)
    loss_bce.backward()
    assert pred_bce.grad is not None

    # Test nll_loss
    log_probs = pycoeus.Tensor([-0.1, -2.0, -1.5, -0.2], [2, 2], requires_grad=True)
    loss_nll = pycoeus.nll_loss(log_probs, [0, 1])
    loss_nll.backward()
    assert log_probs.grad is not None

    # Test huber_loss
    pred_huber = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2], requires_grad=True)
    target_huber = pycoeus.Tensor([1.1, 1.9, 3.5, 3.8], [2, 2])
    loss_huber = pycoeus.huber_loss(pred_huber, target_huber, delta=1.0)
    loss_huber.backward()
    assert pred_huber.grad is not None

    # Verify grad exists and matches shape
    assert x.grad is not None, 'x.grad is None'
    assert len(x.grad) == 4, f'len(x.grad) is {len(x.grad)}'

    # 4. Neural Network Modules
    linear = pycoeus.Linear(2, 3)
    x_in = pycoeus.Tensor([1.0, 2.0], [1, 2])
    y_linear = linear.forward(x_in)
    assert y_linear.shape == [1, 3], f'y_linear.shape is {y_linear.shape}'
    assert linear.weight is not None, 'linear.weight is None'
    assert linear.bias is not None, 'linear.bias is None'

    # Conv1d
    conv1d = pycoeus.Conv1d(in_channels=2, out_channels=4, kernel_size=3)
    x_1d = pycoeus.Tensor([1.0]*10, [1, 2, 5]) # batch=1, in_channels=2, length=5
    y_1d = conv1d.forward(x_1d)
    assert y_1d.shape[0] == 1, f'y_1d.shape[0] is {y_1d.shape[0]}'
    assert y_1d.shape[1] == 4, f'y_1d.shape[1] is {y_1d.shape[1]}'

    # Conv2d
    conv2d = pycoeus.Conv2d(in_channels=2, out_channels=4, kernel_size=3)
    x_2d = pycoeus.Tensor([1.0]*50, [1, 2, 5, 5]) # batch=1, in_channels=2, height=5, width=5
    y_2d = conv2d.forward(x_2d)
    assert y_2d.shape[0] == 1, f'y_2d.shape[0] is {y_2d.shape[0]}'
    assert y_2d.shape[1] == 4, f'y_2d.shape[1] is {y_2d.shape[1]}'

    # 5. Optimizers
    param = pycoeus.Tensor([10.0], requires_grad=True)
    sgd = pycoeus.SGD([param], lr=0.1)
    loss = param * pycoeus.Tensor([2.0])
    loss.backward()
    sgd.step()
    assert param.data[0] < 10.0, f'SGD step failed, param.data[0] is {param.data[0]}'

    # Adam
    param_adam = pycoeus.Tensor([10.0], requires_grad=True)
    adam = pycoeus.Adam([param_adam], lr=0.1)
    loss_adam = param_adam * pycoeus.Tensor([2.0])
    loss_adam.backward()
    adam.step()
    assert param_adam.data[0] < 10.0, f'Adam step failed, param_adam.data[0] is {param_adam.data[0]}'

    # AdamW
    param_adamw = pycoeus.Tensor([10.0], requires_grad=True)
    adamw = pycoeus.AdamW([param_adamw], lr=0.1, weight_decay=0.01)
    loss_adamw = param_adamw * pycoeus.Tensor([2.0])
    loss_adamw.backward()
    adamw.step()
    assert param_adamw.data[0] < 10.0, f'AdamW step failed, param_adamw.data[0] is {param_adamw.data[0]}'

    # RMSProp
    param_rmsprop = pycoeus.Tensor([10.0], requires_grad=True)
    rmsprop = pycoeus.RMSProp([param_rmsprop], lr=0.1)
    loss_rmsprop = param_rmsprop * pycoeus.Tensor([2.0])
    loss_rmsprop.backward()
    rmsprop.step()
    assert param_rmsprop.data[0] < 10.0, f'RMSProp step failed, param_rmsprop.data[0] is {param_rmsprop.data[0]}'

    # GroupNorm
    gn = pycoeus.GroupNorm(num_groups=2, num_features=4)
    x_gn = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 4], requires_grad=True)
    y_gn = gn.forward(x_gn)
    assert y_gn.shape == [2, 4]
    loss_gn = y_gn.sum_axis(0).sum_axis(1)
    loss_gn.backward()
    assert x_gn.grad is not None

    # InstanceNorm1d
    inst1d = pycoeus.InstanceNorm1d(num_features=2)
    x_inst1d = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2], requires_grad=True)
    y_inst1d = inst1d.forward(x_inst1d)
    assert y_inst1d.shape == [2, 2]
    loss_inst1d = y_inst1d.sum_axis(0).sum_axis(1)
    loss_inst1d.backward()
    assert x_inst1d.grad is not None

    # InstanceNorm2d
    inst2d = pycoeus.InstanceNorm2d(num_features=2)
    x_inst2d = pycoeus.Tensor([1.0]*8, [1, 2, 2, 2], requires_grad=True)
    y_inst2d = inst2d.forward(x_inst2d)
    assert y_inst2d.shape == [1, 2, 2, 2]
    loss_inst2d = y_inst2d.sum_axis(0).sum_axis(1).sum_axis(2).sum_axis(3)
    loss_inst2d.backward()
    assert x_inst2d.grad is not None

    # MultiHeadAttention
    mha = pycoeus.MultiHeadAttention(d_model=4, num_heads=2)
    x_mha = pycoeus.Tensor([1.0]*8, [1, 2, 4], requires_grad=True)
    y_mha = mha.forward(x_mha)
    assert y_mha.shape == [1, 2, 4]
    loss_mha = y_mha.sum_axis(0).sum_axis(1).sum_axis(2)
    loss_mha.backward()
    assert x_mha.grad is not None

    # log_softmax
    x_lsm = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2], requires_grad=True)
    y_lsm = pycoeus.log_softmax(x_lsm, axis=1)
    assert y_lsm.shape == [2, 2]
    loss_lsm = y_lsm.sum_axis(0).sum_axis(1)
    loss_lsm.backward()
    assert x_lsm.grad is not None

    # cat
    x_c1 = pycoeus.Tensor([1.0, 2.0], [1, 2], requires_grad=True)
    x_c2 = pycoeus.Tensor([3.0, 4.0], [1, 2], requires_grad=True)
    y_cat = pycoeus.cat([x_c1, x_c2], dim=0)
    assert y_cat.shape == [2, 2]
    loss_cat = y_cat.sum_axis(0).sum_axis(1)
    loss_cat.backward()
    assert x_c1.grad is not None
    assert x_c2.grad is not None

    # split
    x_sp = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2], requires_grad=True)
    y_sp = pycoeus.split(x_sp, chunk_size=1, dim=0)
    assert len(y_sp) == 2
    assert y_sp[0].shape == [1, 2]
    assert y_sp[1].shape == [1, 2]
    loss_sp = (y_sp[0] + y_sp[1]).sum_axis(0).sum_axis(1)
    loss_sp.backward()
    assert x_sp.grad is not None

    # 6. Distributed Communication & Collectives
    import threading

    world_size = 3
    comms = pycoeus.create_mock_cluster(world_size)
    assert len(comms) == world_size

    # Test all_reduce
    results_reduce = [None] * world_size
    def run_all_reduce(rank):
        comm = comms[rank]
        t = pycoeus.Tensor([float(rank + 1.0), float(rank + 2.0)])
        comm.all_reduce(t)
        results_reduce[rank] = t.data

    threads = []
    for r in range(world_size):
        t = threading.Thread(target=run_all_reduce, args=(r,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    for r in range(world_size):
        assert results_reduce[r] == [6.0, 9.0], f'Rank {r} got {results_reduce[r]}'

    # Test broadcast
    results_broadcast = [None] * world_size
    def run_broadcast(rank):
        comm = comms[rank]
        if rank == 1:
            t = pycoeus.Tensor([42.0, 100.0])
        else:
            t = pycoeus.Tensor([0.0, 0.0])
        comm.broadcast(t, 1)
        results_broadcast[rank] = t.data

    threads = []
    for r in range(world_size):
        t = threading.Thread(target=run_broadcast, args=(r,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    for r in range(world_size):
        assert results_broadcast[r] == [42.0, 100.0], f'Rank {r} got {results_broadcast[r]}'

    # Test all_gather
    results_gather = [None] * world_size
    def run_all_gather(rank):
        comm = comms[rank]
        t = pycoeus.Tensor([float(rank * 10.0)])
        out = [pycoeus.Tensor([0.0]) for _ in range(world_size)]
        comm.all_gather(t, out)
        results_gather[rank] = [o.data[0] for o in out]

    threads = []
    for r in range(world_size):
        t = threading.Thread(target=run_all_gather, args=(r,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    for r in range(world_size):
        assert results_gather[r] == [0.0, 10.0, 20.0], f'Rank {r} got {results_gather[r]}'

    # Test DDP gradient synchronization
    results_grad_sync = [None] * world_size
    def run_grad_sync(rank):
        comm = comms_sync[rank]
        p = pycoeus.Tensor([0.0, 0.0], requires_grad=True)
        loss = p * pycoeus.Tensor([float(rank + 1.0), float(rank + 10.0)])
        loss.backward()
        pycoeus.synchronize_gradients([p], comm)
        results_grad_sync[rank] = p.grad

    threads = []
    world_size_sync = 2
    comms_sync = pycoeus.create_mock_cluster(world_size_sync)
    for r in range(world_size_sync):
        t = threading.Thread(target=run_grad_sync, args=(r,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    for r in range(world_size_sync):
        assert results_grad_sync[r] is not None
        assert abs(results_grad_sync[r][0] - 1.5) < 1e-5, f'Rank {r} grad[0] is {results_grad_sync[r][0]}'
        assert abs(results_grad_sync[r][1] - 10.5) < 1e-5, f'Rank {r} grad[1] is {results_grad_sync[r][1]}'

    # Test reduce
    results_reduce_only = [None] * world_size
    def run_reduce(rank):
        comm = comms[rank]
        t = pycoeus.Tensor([float(rank + 1.0), float(rank + 2.0)])
        comm.reduce(t, 1)
        results_reduce_only[rank] = t.data

    threads = []
    for r in range(world_size):
        t = threading.Thread(target=run_reduce, args=(r,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    assert results_reduce_only[1] == [6.0, 9.0], f'Rank 1 got {results_reduce_only[1]}'

    # Test gather
    results_gather_only = [None] * world_size
    def run_gather(rank):
        comm = comms[rank]
        t = pycoeus.Tensor([float(rank * 10.0)])
        out = [pycoeus.Tensor([0.0]) for _ in range(world_size)] if rank == 2 else []
        comm.gather(t, out, 2)
        if rank == 2:
            results_gather_only[rank] = [o.data[0] for o in out]

    threads = []
    for r in range(world_size):
        t = threading.Thread(target=run_gather, args=(r,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    assert results_gather_only[2] == [0.0, 10.0, 20.0], f'Rank 2 got {results_gather_only[2]}'

    # Test scatter
    results_scatter_only = [None] * world_size
    def run_scatter(rank):
        comm = comms[rank]
        t = pycoeus.Tensor([0.0])
        inp = [pycoeus.Tensor([100.0]), pycoeus.Tensor([200.0]), pycoeus.Tensor([300.0])] if rank == 0 else []
        comm.scatter(t, inp, 0)
        results_scatter_only[rank] = t.data

    threads = []
    for r in range(world_size):
        t = threading.Thread(target=run_scatter, args=(r,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    for r in range(world_size):
        assert results_scatter_only[r] == [(r + 1.0) * 100.0], f'Rank {r} got {results_scatter_only[r]}'

    # 6.5. TCP Mesh & TcpCommunicator Collectives
    def get_free_ports(count):
        import socket
        ports = []
        sockets = []
        for _ in range(count):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('127.0.0.1', 0))
            ports.append(f'127.0.0.1:{s.getsockname()[1]}')
            sockets.append(s)
        for s in sockets:
            s.close()
        return ports

    # Test all_reduce
    tcp_addresses_all_reduce = get_free_ports(2)
    tcp_results_reduce = [None] * 2
    def run_tcp_all_reduce(rank):
        mesh = pycoeus.TcpMesh(rank, 2, tcp_addresses_all_reduce)
        comm = pycoeus.TcpCommunicator(mesh)
        t = pycoeus.Tensor([float(rank + 1.0), float(rank + 2.0)])
        comm.all_reduce(t)
        tcp_results_reduce[rank] = t.data

    threads = []
    for r in range(2):
        t = threading.Thread(target=run_tcp_all_reduce, args=(r,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    for r in range(2):
        assert tcp_results_reduce[r] == [3.0, 5.0], f'TCP all_reduce Rank {r} got {tcp_results_reduce[r]}'

    # Test broadcast
    tcp_addresses_broadcast = get_free_ports(2)
    tcp_results_broadcast = [None] * 2
    def run_tcp_broadcast(rank):
        mesh = pycoeus.TcpMesh(rank, 2, tcp_addresses_broadcast)
        comm = pycoeus.TcpCommunicator(mesh)
        if rank == 0:
            t = pycoeus.Tensor([10.0, 20.0])
        else:
            t = pycoeus.Tensor([0.0, 0.0])
        comm.broadcast(t, 0)
        tcp_results_broadcast[rank] = t.data

    threads = []
    for r in range(2):
        t = threading.Thread(target=run_tcp_broadcast, args=(r,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    for r in range(2):
        assert tcp_results_broadcast[r] == [10.0, 20.0], f'TCP broadcast Rank {r} got {tcp_results_broadcast[r]}'

    # Test all_gather
    tcp_addresses_all_gather = get_free_ports(2)
    tcp_results_all_gather = [None] * 2
    def run_tcp_all_gather(rank):
        mesh = pycoeus.TcpMesh(rank, 2, tcp_addresses_all_gather)
        comm = pycoeus.TcpCommunicator(mesh)
        t = pycoeus.Tensor([float(rank * 100.0)])
        out = [pycoeus.Tensor([0.0]) for _ in range(2)]
        comm.all_gather(t, out)
        tcp_results_all_gather[rank] = [o.data[0] for o in out]

    threads = []
    for r in range(2):
        t = threading.Thread(target=run_tcp_all_gather, args=(r,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    for r in range(2):
        assert tcp_results_all_gather[r] == [0.0, 100.0], f'TCP all_gather Rank {r} got {tcp_results_all_gather[r]}'

    # Test reduce
    tcp_addresses_reduce = get_free_ports(2)
    tcp_results_reduce_only = [None] * 2
    def run_tcp_reduce(rank):
        mesh = pycoeus.TcpMesh(rank, 2, tcp_addresses_reduce)
        comm = pycoeus.TcpCommunicator(mesh)
        t = pycoeus.Tensor([float(rank + 1.0), float(rank + 2.0)])
        comm.reduce(t, 1)
        tcp_results_reduce_only[rank] = t.data

    threads = []
    for r in range(2):
        t = threading.Thread(target=run_tcp_reduce, args=(r,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    assert tcp_results_reduce_only[1] == [3.0, 5.0], f'TCP reduce Rank 1 got {tcp_results_reduce_only[1]}'

    # Test gather
    tcp_addresses_gather = get_free_ports(2)
    tcp_results_gather_only = [None] * 2
    def run_tcp_gather(rank):
        mesh = pycoeus.TcpMesh(rank, 2, tcp_addresses_gather)
        comm = pycoeus.TcpCommunicator(mesh)
        t = pycoeus.Tensor([float(rank * 100.0)])
        out = [pycoeus.Tensor([0.0]) for _ in range(2)] if rank == 1 else []
        comm.gather(t, out, 1)
        if rank == 1:
            tcp_results_gather_only[rank] = [o.data[0] for o in out]

    threads = []
    for r in range(2):
        t = threading.Thread(target=run_tcp_gather, args=(r,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    assert tcp_results_gather_only[1] == [0.0, 100.0], f'TCP gather Rank 1 got {tcp_results_gather_only[1]}'

    # Test scatter
    tcp_addresses_scatter = get_free_ports(2)
    tcp_results_scatter_only = [None] * 2
    def run_tcp_scatter(rank):
        mesh = pycoeus.TcpMesh(rank, 2, tcp_addresses_scatter)
        comm = pycoeus.TcpCommunicator(mesh)
        t = pycoeus.Tensor([0.0])
        inp = [pycoeus.Tensor([100.0]), pycoeus.Tensor([200.0])] if rank == 0 else []
        comm.scatter(t, inp, 0)
        tcp_results_scatter_only[rank] = t.data

    threads = []
    for r in range(2):
        t = threading.Thread(target=run_tcp_scatter, args=(r,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    for r in range(2):
        assert tcp_results_scatter_only[r] == [(r + 1.0) * 100.0], f'TCP scatter Rank {r} got {tcp_results_scatter_only[r]}'

    # 7. Normalization & Pooling layers
    # Test LayerNorm
    ln = pycoeus.LayerNorm(normalized_shape=4, eps=1e-5)
    x_ln = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 4], requires_grad=True)
    out_ln = ln.forward(x_ln)
    assert out_ln.shape == [2, 4], f'LayerNorm shape is {out_ln.shape}'
    out_ln.backward()
    assert x_ln.grad is not None
    assert ln.weight.grad is not None
    assert ln.bias.grad is not None

    # Test RMSNorm
    rms = pycoeus.RMSNorm(normalized_shape=4, eps=1e-8)
    x_rms = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 4], requires_grad=True)
    out_rms = rms.forward(x_rms)
    assert out_rms.shape == [2, 4], f'RMSNorm shape is {out_rms.shape}'
    out_rms.backward()
    assert x_rms.grad is not None
    assert rms.weight.grad is not None

    # Test AvgPool2d
    avg_pool = pycoeus.AvgPool2d(kernel_size=2, stride=2, padding=0)
    x_pool = pycoeus.Tensor([1.0] * 16, [1, 1, 4, 4], requires_grad=True)
    out_avg = avg_pool.forward(x_pool)
    assert out_avg.shape == [1, 1, 2, 2], f'AvgPool2d shape is {out_avg.shape}'
    out_avg.backward()
    assert x_pool.grad is not None

    # Test MaxPool2d
    max_pool = pycoeus.MaxPool2d(kernel_size=2, stride=2, padding=0)
    out_max = max_pool.forward(x_pool)
    assert out_max.shape == [1, 1, 2, 2], f'MaxPool2d shape is {out_max.shape}'
    out_max.backward()
    assert x_pool.grad is not None

    # Test Conv3d
    conv3d = pycoeus.Conv3d(in_channels=2, out_channels=4, kernel_size=3)
    x_3d = pycoeus.Tensor([1.0]*250, [1, 2, 5, 5, 5]) # batch=1, in_channels=2, depth=5, height=5, width=5
    y_3d = conv3d.forward(x_3d)
    assert y_3d.shape[0] == 1
    assert y_3d.shape[1] == 4
    assert y_3d.shape[2] == 3
    assert y_3d.shape[3] == 3
    assert y_3d.shape[4] == 3

    # Test BatchNorm3d
    bn3d = pycoeus.BatchNorm3d(num_features=2, eps=1e-5, momentum=0.1)
    x_bn = pycoeus.Tensor([1.0]*16, [1, 2, 2, 2, 2], requires_grad=True)
    out_bn = bn3d.forward(x_bn)
    assert out_bn.shape == [1, 2, 2, 2, 2]
    out_bn.backward()
    assert x_bn.grad is not None
    assert bn3d.weight.grad is not None
    assert bn3d.bias.grad is not None

    # Test AvgPool3d
    avg_pool3d = pycoeus.AvgPool3d(kernel_size=2, stride=2, padding=0)
    x_pool3d = pycoeus.Tensor([1.0]*64, [1, 1, 4, 4, 4], requires_grad=True)
    out_avg3d = avg_pool3d.forward(x_pool3d)
    assert out_avg3d.shape == [1, 1, 2, 2, 2]
    out_avg3d.backward()
    assert x_pool3d.grad is not None

    # Test MaxPool3d
    max_pool3d = pycoeus.MaxPool3d(kernel_size=2, stride=2, padding=0)
    out_max3d = max_pool3d.forward(x_pool3d)
    assert out_max3d.shape == [1, 1, 2, 2, 2]
    out_max3d.backward()
    assert x_pool3d.grad is not None

    # Test Embedding
    emb = pycoeus.Embedding(num_embeddings=10, embedding_dim=4)
    indices = pycoeus.Tensor([1.0, 2.0, 3.0, 0.0], [2, 2], requires_grad=False)
    out_emb = emb.forward(indices)
    assert out_emb.shape == [2, 2, 4], f'Embedding shape is {out_emb.shape}'
    out_emb.backward()
    assert emb.weight.grad is not None, 'Embedding weight grad is None'

    # Test Dropout
    dropout = pycoeus.Dropout(p=0.5)
    x_drop = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
    out_drop_train = dropout.forward(x_drop)
    assert out_drop_train.shape == [2, 2]
    dropout.train(False)
    out_drop_eval = dropout.forward(x_drop)
    assert out_drop_eval.shape == [2, 2]
    assert out_drop_eval.data == [1.0, 2.0, 3.0, 4.0]

    # Test BatchNorm1d
    bn1d = pycoeus.BatchNorm1d(num_features=2, eps=1e-5, momentum=0.1)
    x_bn1d = pycoeus.Tensor([1.0]*12, [2, 2, 3], requires_grad=True)
    out_bn1d = bn1d.forward(x_bn1d)
    assert out_bn1d.shape == [2, 2, 3]
    out_bn1d.backward()
    assert x_bn1d.grad is not None
    assert bn1d.weight.grad is not None
    assert bn1d.bias.grad is not None

    # Test BatchNorm2d
    bn2d = pycoeus.BatchNorm2d(num_features=2, eps=1e-5, momentum=0.1)
    x_bn2d = pycoeus.Tensor([1.0]*24, [2, 2, 2, 3], requires_grad=True)
    out_bn2d = bn2d.forward(x_bn2d)
    assert out_bn2d.shape == [2, 2, 2, 3]
    out_bn2d.backward()
    assert x_bn2d.grad is not None
    assert bn2d.weight.grad is not None
    assert bn2d.bias.grad is not None
    # 8. GroupNorm, InstanceNorm, MultiHeadAttention
    # Test GroupNorm (num_groups=2, num_features=4)
    gn = pycoeus.GroupNorm(num_groups=2, num_features=4)
    x_gn = pycoeus.Tensor(list(range(1, 9)), [1, 4, 2], requires_grad=True)
    out_gn = gn.forward(x_gn)
    assert out_gn.shape == [1, 4, 2], f'GroupNorm shape is {out_gn.shape}'
    out_gn.backward()
    assert x_gn.grad is not None, 'GroupNorm input grad is None'
    assert gn.weight.grad is not None, 'GroupNorm weight grad is None'
    assert gn.bias.grad is not None, 'GroupNorm bias grad is None'

    # Test GroupNorm num_groups=1 (equivalent to LayerNorm over all features)
    gn1 = pycoeus.GroupNorm(num_groups=1, num_features=4)
    x_gn1 = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 4], requires_grad=True)
    out_gn1 = gn1.forward(x_gn1)
    assert out_gn1.shape == [2, 4], f'GroupNorm(G=1) shape is {out_gn1.shape}'
    out_gn1.backward()
    assert x_gn1.grad is not None

    # Test InstanceNorm1d: [N=2, C=3, L=4]
    in1d = pycoeus.InstanceNorm1d(num_features=3)
    x_in1d = pycoeus.Tensor([float(i) for i in range(24)], [2, 3, 4], requires_grad=True)
    out_in1d = in1d.forward(x_in1d)
    assert out_in1d.shape == [2, 3, 4], f'InstanceNorm1d shape is {out_in1d.shape}'
    out_in1d.backward()
    assert x_in1d.grad is not None
    assert in1d.weight.grad is not None
    assert in1d.bias.grad is not None

    # Test InstanceNorm2d: [N=1, C=2, H=3, W=3]
    in2d = pycoeus.InstanceNorm2d(num_features=2)
    x_in2d = pycoeus.Tensor([float(i) for i in range(18)], [1, 2, 3, 3], requires_grad=True)
    out_in2d = in2d.forward(x_in2d)
    assert out_in2d.shape == [1, 2, 3, 3], f'InstanceNorm2d shape is {out_in2d.shape}'
    out_in2d.backward()
    assert x_in2d.grad is not None
    assert in2d.weight.grad is not None
    assert in2d.bias.grad is not None

    # Test MultiHeadAttention (self-attention): d_model=8, num_heads=4
    mha = pycoeus.MultiHeadAttention(d_model=8, num_heads=4)
    x_mha = pycoeus.Tensor([0.1 * i for i in range(40)], [1, 5, 8], requires_grad=True)
    out_mha = mha.forward(x_mha)
    assert out_mha.shape == [1, 5, 8], f'MHA self-attention shape is {out_mha.shape}'
    out_mha.backward()
    assert x_mha.grad is not None, 'MHA input grad is None'
    assert mha.w_q.grad is not None, 'MHA w_q grad is None'
    assert mha.w_k.grad is not None, 'MHA w_k grad is None'
    assert mha.w_v.grad is not None, 'MHA w_v grad is None'
    assert mha.w_o.grad is not None, 'MHA w_o grad is None'

    # Test MultiHeadAttention cross-attention: query=[1,3,8], key/value=[1,5,8]
    mha2 = pycoeus.MultiHeadAttention(d_model=8, num_heads=2)
    q_mha = pycoeus.Tensor([0.1 * i for i in range(24)], [1, 3, 8], requires_grad=True)
    k_mha = pycoeus.Tensor([0.1 * i for i in range(40)], [1, 5, 8], requires_grad=False)
    v_mha = pycoeus.Tensor([0.1 * i for i in range(40)], [1, 5, 8], requires_grad=False)
    out_cross = mha2.forward_cross(q_mha, k_mha, v_mha)
    assert out_cross.shape == [1, 3, 8], f'MHA cross-attention shape is {out_cross.shape}'
    out_cross.backward()
    assert q_mha.grad is not None, 'MHA cross-attention query grad is None'

    # 9. log_softmax, cat, split
    # Test log_softmax: exp(output) must sum to 1 along axis=1
    x_lsm = pycoeus.Tensor([1.0, 2.0, 3.0, 0.5, 1.5, 2.5], [2, 3], requires_grad=True)
    out_lsm = pycoeus.log_softmax(x_lsm, 1)
    assert out_lsm.shape == [2, 3], f'log_softmax shape is {out_lsm.shape}'
    import math
    row0_sum = sum(math.exp(v) for v in out_lsm.data[:3])
    assert abs(row0_sum - 1.0) < 1e-5, f'log_softmax row0 sum={row0_sum}'
    row1_sum = sum(math.exp(v) for v in out_lsm.data[3:])
    assert abs(row1_sum - 1.0) < 1e-5, f'log_softmax row1 sum={row1_sum}'
    # Backward
    loss_lsm = pycoeus.mse_loss(out_lsm, pycoeus.Tensor([0.0, 1.0, 0.0, 1.0, 0.0, 0.0], [2, 3]))
    loss_lsm.backward()
    assert x_lsm.grad is not None

    # Test cat along dim=1: [2,3] ++ [2,4] → [2,7]
    a_cat = pycoeus.Tensor([1.0]*6, [2, 3], requires_grad=True)
    b_cat = pycoeus.Tensor([2.0]*8, [2, 4], requires_grad=True)
    out_cat = pycoeus.cat([a_cat, b_cat], 1)
    assert out_cat.shape == [2, 7], f'cat shape is {out_cat.shape}'
    out_cat.backward()
    assert a_cat.grad is not None
    assert b_cat.grad is not None
    assert len(a_cat.grad) == 6, f'a_cat.grad len={len(a_cat.grad)}'
    assert len(b_cat.grad) == 8, f'b_cat.grad len={len(b_cat.grad)}'

    # Test cat along dim=0: [2,3] ++ [3,3] → [5,3]
    p_cat = pycoeus.Tensor([0.0]*6, [2, 3], requires_grad=True)
    q_cat = pycoeus.Tensor([1.0]*9, [3, 3], requires_grad=True)
    out_cat2 = pycoeus.cat([p_cat, q_cat], 0)
    assert out_cat2.shape == [5, 3], f'cat(dim=0) shape is {out_cat2.shape}'
    out_cat2.backward()
    assert p_cat.grad is not None
    assert q_cat.grad is not None

    # Test split: [1,6] into chunks of 2 along dim=1 → 3 chunks of [1,2]
    x_split = pycoeus.Tensor(list(range(6)), [1, 6], requires_grad=True)
    chunks = pycoeus.split(x_split, 2, 1)
    assert len(chunks) == 3, f'split chunk count={len(chunks)}'
    for ch in chunks:
        assert ch.shape == [1, 2], f'chunk shape={ch.shape}'

    # split backward: drive loss from chunk 0 only; grad in positions 2-5 must be zero
    x_split2 = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 6], requires_grad=True)
    chunks2 = pycoeus.split(x_split2, 2, 1)
    loss_split = pycoeus.mse_loss(chunks2[0], pycoeus.Tensor([0.0, 0.0], [1, 2]))
    loss_split.backward()
    assert x_split2.grad is not None
    g_split = x_split2.grad
    assert g_split[2] == 0.0, f'split: grad[2] should be 0 but is {g_split[2]}'
    assert g_split[3] == 0.0, f'split: grad[3] should be 0 but is {g_split[3]}'
    assert g_split[4] == 0.0, f'split: grad[4] should be 0 but is {g_split[4]}'
    assert g_split[5] == 0.0, f'split: grad[5] should be 0 but is {g_split[5]}'
    assert g_split[0] != 0.0 or g_split[1] != 0.0, 'split: grad[0]/[1] should be non-zero'

    # 10. Tensor shape methods: reshape, permute, squeeze, unsqueeze, t, contiguous
    # Test reshape: [2, 3] -> [1, 6]
    x_shape = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], requires_grad=True)
    x_reshaped = x_shape.reshape([1, 6])
    assert x_reshaped.shape == [1, 6], f'reshape shape={x_reshaped.shape}'
    x_reshaped.backward()
    assert x_shape.grad is not None, 'reshape backward failed'

    # Test permute: [2, 3] -> [3, 2]
    x_shape2 = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], requires_grad=True)
    x_permuted = x_shape2.permute([1, 0])
    assert x_permuted.shape == [3, 2], f'permute shape={x_permuted.shape}'
    # checking data mapping
    assert x_permuted.data == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0], f'permuted data={x_permuted.data}'
    x_permuted.backward()
    assert x_shape2.grad is not None, 'permute backward failed'

    # Test unsqueeze: [2, 3] -> [2, 1, 3]
    x_shape3 = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], requires_grad=True)
    x_unsqueezed = x_shape3.unsqueeze(1)
    assert x_unsqueezed.shape == [2, 1, 3], f'unsqueeze shape={x_unsqueezed.shape}'
    x_unsqueezed.backward()
    assert x_shape3.grad is not None, 'unsqueeze backward failed'

    # Test squeeze (specific axis): [2, 1, 3] -> [2, 3]
    x_shape4 = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 1, 3], requires_grad=True)
    x_squeezed = x_shape4.squeeze(1)
    assert x_squeezed.shape == [2, 3], f'squeeze shape={x_squeezed.shape}'
    x_squeezed.backward()
    assert x_shape4.grad is not None, 'squeeze backward failed'

    # Test squeeze (all): [1, 2, 1, 3] -> [2, 3]
    x_shape5 = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 2, 1, 3], requires_grad=True)
    x_squeezed_all = x_shape5.squeeze()
    assert x_squeezed_all.shape == [2, 3], f'squeeze all shape={x_squeezed_all.shape}'
    x_squeezed_all.backward()
    assert x_shape5.grad is not None, 'squeeze all backward failed'

    # Test transpose (t): [2, 3] -> [3, 2]
    x_shape6 = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], requires_grad=True)
    x_t = x_shape6.t()
    assert x_t.shape == [3, 2], f't shape={x_t.shape}'
    assert x_t.data == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0], f't data={x_t.data}'
    x_t.backward()
    assert x_shape6.grad is not None, 't backward failed'

    # Test contiguous
    x_shape7 = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], requires_grad=True)
    x_transposed = x_shape7.t()
    x_cont = x_transposed.contiguous()
    assert x_cont.shape == [3, 2]
    x_cont.backward()
    assert x_shape7.grad is not None, 'contiguous backward failed'

    # 11. Rotary Positional Embedding (RoPE)
    rope = pycoeus.RotaryEmbedding(max_len=16, d_head=4, base=10000.0)
    assert rope.max_len == 16, f'rope.max_len is {rope.max_len}'
    assert rope.d_head == 4, f'rope.d_head is {rope.d_head}'
    assert rope.base == 10000.0, f'rope.base is {rope.base}'

    x_rope = pycoeus.Tensor([1.0]*32, [2, 4, 1, 4], requires_grad=True)
    y_rope = rope.forward(x_rope)
    assert y_rope.shape == [2, 4, 1, 4], f'y_rope.shape is {y_rope.shape}'
    loss_rope = y_rope.sum_axis(0).sum_axis(1).sum_axis(2).sum_axis(3)
    loss_rope.backward()
    assert x_rope.grad is not None, 'x_rope.grad is None'
    assert len(x_rope.grad) == 32

    # 12. General Transpose Method
    x_tr = pycoeus.Tensor([float(i) for i in range(24)], [2, 3, 4], requires_grad=True)
    y_tr = x_tr.transpose(0, 2)
    assert y_tr.shape == [4, 3, 2], f'y_tr.shape is {y_tr.shape}'
    loss_tr = y_tr.sum_axis(0).sum_axis(1).sum_axis(2)
    loss_tr.backward()
    assert x_tr.grad is not None, 'x_tr.grad is None'
    assert all(abs(g - 1.0) < 1e-5 for g in x_tr.grad)

    # 13. AdaGrad Optimizer
    param_adagrad = pycoeus.Tensor([10.0], requires_grad=True)
    adagrad = pycoeus.AdaGrad([param_adagrad], lr=0.1)
    loss_adagrad = param_adagrad * pycoeus.Tensor([2.0])
    loss_adagrad.backward()
    adagrad.step()
    assert abs(param_adagrad.data[0] - 9.9) < 1e-5, f'AdaGrad step failed, data is {param_adagrad.data[0]}'

    # 14. CumSum Method & Function
    x_cs = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [4], requires_grad=True)
    y_cs = x_cs.cumsum(0)
    assert y_cs.data == [1.0, 3.0, 6.0, 10.0], f'y_cs.data is {y_cs.data}'
    y_cs_fn = pycoeus.cumsum(x_cs, 0)
    assert y_cs_fn.data == [1.0, 3.0, 6.0, 10.0]

    # test backward
    loss_cs = y_cs.sum_axis(0)
    loss_cs.backward()
    assert x_cs.grad is not None, 'x_cs.grad is None'
    assert x_cs.grad == [4.0, 3.0, 2.0, 1.0], f'x_cs.grad is {x_cs.grad}'

except Exception as e:
    traceback.print_exc()
    sys.exit(1)
";

        if let Err(e) = py.run(test_script, None, None) {
            panic!("Python execution failed: {:?}", e);
        }
    });
}
