use pyo3::prelude::*;
use pyo3::types::PyDict;

#[test]
fn test_pycoeus_dist() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let pycoeus_module = pyo3::types::PyModule::new(py, "pycoeus").unwrap();
        pycoeus::pycoeus(&pycoeus_module).unwrap();

        let sys = py.import("sys").unwrap();
        let modules_any = sys.getattr("modules").unwrap();
        let modules = modules_any.downcast::<PyDict>().unwrap();
        modules.set_item("pycoeus", &pycoeus_module).unwrap();

        let test_script = c"
import pycoeus
import sys
import traceback
import threading

try:
    # 6. Distributed Communication & Collectives
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

except Exception as e:
    traceback.print_exc()
    sys.exit(1)
";

        if let Err(e) = py.run(test_script, None, None) {
            panic!("Python execution failed: {:?}", e);
        }
    });
}
