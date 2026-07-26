use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::ffi::CString;

use crate::common;

fn run_pycoeus_script(script: &str) {
    let _guard = common::python_test_lock()
        .lock()
        .expect("python test lock poisoned");
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let script = CString::new(script).unwrap();
        let pycoeus_module = pyo3::types::PyModule::new(py, "pycoeus").unwrap();
        pycoeus::pycoeus(&pycoeus_module).unwrap();

        let sys = py.import("sys").unwrap();
        let modules_any = sys.getattr("modules").unwrap();
        let modules = modules_any.downcast::<PyDict>().unwrap();
        modules.set_item("pycoeus", &pycoeus_module).unwrap();
        let globals = PyDict::new(py);
        globals.set_item("pycoeus", &pycoeus_module).unwrap();

        let result = py.run(script.as_c_str(), Some(&globals), None);
        modules
            .del_item("pycoeus")
            .unwrap_or_else(|e| panic!("failed to remove pycoeus test module: {e:?}"));
        result.unwrap_or_else(|e| panic!("Python execution failed: {e:?}"));
    });
}

#[test]
fn test_pycoeus_local_collectives() {
    run_pycoeus_script(
        r#"
import pycoeus
import threading

world_size = 3
comms = pycoeus.create_local_cluster(world_size)
assert len(comms) == world_size

def run_collective(call):
    threads = []
    for rank in range(world_size):
        thread = threading.Thread(target=call, args=(rank,))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

results_reduce = [None] * world_size
def all_reduce(rank):
    tensor = pycoeus.Tensor([float(rank + 1.0), float(rank + 2.0)])
    comms[rank].all_reduce(tensor)
    results_reduce[rank] = tensor.data

run_collective(all_reduce)
for rank in range(world_size):
    assert results_reduce[rank] == [6.0, 9.0], f'rank {rank} all_reduce {results_reduce[rank]}'

results_broadcast = [None] * world_size
def broadcast(rank):
    tensor = pycoeus.Tensor([42.0, 100.0]) if rank == 1 else pycoeus.Tensor([0.0, 0.0])
    comms[rank].broadcast(tensor, 1)
    results_broadcast[rank] = tensor.data

run_collective(broadcast)
for rank in range(world_size):
    assert results_broadcast[rank] == [42.0, 100.0], f'rank {rank} broadcast {results_broadcast[rank]}'

results_gather = [None] * world_size
def all_gather(rank):
    tensor = pycoeus.Tensor([float(rank * 10.0)])
    out = [pycoeus.Tensor([0.0]) for _ in range(world_size)]
    comms[rank].all_gather(tensor, out)
    results_gather[rank] = [item.data[0] for item in out]

run_collective(all_gather)
for rank in range(world_size):
    assert results_gather[rank] == [0.0, 10.0, 20.0], f'rank {rank} all_gather {results_gather[rank]}'

results_reduce_only = [None] * world_size
def reduce(rank):
    tensor = pycoeus.Tensor([float(rank + 1.0), float(rank + 2.0)])
    comms[rank].reduce(tensor, 1)
    results_reduce_only[rank] = tensor.data

run_collective(reduce)
assert results_reduce_only[1] == [6.0, 9.0], f'root reduce {results_reduce_only[1]}'

results_gather_only = [None] * world_size
def gather(rank):
    tensor = pycoeus.Tensor([float(rank * 10.0)])
    out = [pycoeus.Tensor([0.0]) for _ in range(world_size)] if rank == 2 else []
    comms[rank].gather(tensor, out, 2)
    if rank == 2:
        results_gather_only[rank] = [item.data[0] for item in out]

run_collective(gather)
assert results_gather_only[2] == [0.0, 10.0, 20.0], f'root gather {results_gather_only[2]}'

results_scatter_only = [None] * world_size
def scatter(rank):
    tensor = pycoeus.Tensor([0.0])
    inputs = [pycoeus.Tensor([100.0]), pycoeus.Tensor([200.0]), pycoeus.Tensor([300.0])] if rank == 0 else []
    comms[rank].scatter(tensor, inputs, 0)
    results_scatter_only[rank] = tensor.data

run_collective(scatter)
for rank in range(world_size):
    assert results_scatter_only[rank] == [(rank + 1.0) * 100.0], f'rank {rank} scatter {results_scatter_only[rank]}'
"#,
    );
}

#[test]
fn test_pycoeus_gradient_synchronization() {
    run_pycoeus_script(
        r#"
import pycoeus
import threading

world_size = 2
comms = pycoeus.create_local_cluster(world_size)
results = [None] * world_size

def run(rank):
    param = pycoeus.Tensor([0.0, 0.0], requires_grad=True)
    loss = param * pycoeus.Tensor([float(rank + 1.0), float(rank + 10.0)])
    loss.backward()
    pycoeus.synchronize_gradients([param], comms[rank])
    results[rank] = param.grad

threads = [threading.Thread(target=run, args=(rank,)) for rank in range(world_size)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()

for rank in range(world_size):
    assert results[rank] is not None
    assert abs(results[rank][0] - 1.5) < 1e-5, f'rank {rank} grad[0] {results[rank][0]}'
    assert abs(results[rank][1] - 10.5) < 1e-5, f'rank {rank} grad[1] {results[rank][1]}'
"#,
    );
}

#[test]
fn test_pycoeus_tcp_mesh_single_rank() {
    run_pycoeus_script(
        r#"
mesh = pycoeus.TcpMesh(0, 1, ["127.0.0.1:0"])
comm = pycoeus.TcpCommunicator(mesh)
assert comm.rank() == 0
assert comm.size() == 1
"#,
    );
}

#[test]
fn test_pycoeus_tcp_loopback_cluster_rejects_zero_world_size() {
    run_pycoeus_script(
        r#"
try:
    pycoeus.create_tcp_loopback_cluster(0)
except ValueError as error:
    assert str(error) == "world_size must be greater than zero"
else:
    raise AssertionError("zero-sized TCP loopback cluster was accepted")
"#,
    );
}

fn run_pycoeus_tcp_script(body: &str) {
    let mut script = String::from(
        r#"
import pycoeus
import threading

def run_tcp(call):
    comms = pycoeus.create_tcp_loopback_cluster(2)
    results = [None] * 2
    failures = []
    def run(rank):
        try:
            results[rank] = call(rank, comms[rank])
        except BaseException as error:
            failures.append((rank, error))

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not failures, f'tcp rank failures: {failures}'
    return results

"#,
    );
    script.push_str(body);
    run_pycoeus_script(&script);
}

#[test]
fn test_pycoeus_tcp_all_reduce() {
    run_pycoeus_tcp_script(
        r#"
def all_reduce(rank, comm):
    tensor = pycoeus.Tensor([float(rank + 1.0), float(rank + 2.0)])
    comm.all_reduce(tensor)
    return tensor.data

all_reduce_results = run_tcp(all_reduce)
for rank in range(2):
    assert all_reduce_results[rank] == [3.0, 5.0], f'tcp all_reduce rank {rank} {all_reduce_results[rank]}'
"#,
    );
}

#[test]
fn test_pycoeus_tcp_broadcast() {
    run_pycoeus_tcp_script(
        r#"
def broadcast(rank, comm):
    tensor = pycoeus.Tensor([10.0, 20.0]) if rank == 0 else pycoeus.Tensor([0.0, 0.0])
    comm.broadcast(tensor, 0)
    return tensor.data

broadcast_results = run_tcp(broadcast)
for rank in range(2):
    assert broadcast_results[rank] == [10.0, 20.0], f'tcp broadcast rank {rank} {broadcast_results[rank]}'
"#,
    );
}

#[test]
fn test_pycoeus_tcp_all_gather() {
    run_pycoeus_tcp_script(
        r#"
def all_gather(rank, comm):
    tensor = pycoeus.Tensor([float(rank * 100.0)])
    out = [pycoeus.Tensor([0.0]) for _ in range(2)]
    comm.all_gather(tensor, out)
    return [item.data[0] for item in out]

all_gather_results = run_tcp(all_gather)
for rank in range(2):
    assert all_gather_results[rank] == [0.0, 100.0], f'tcp all_gather rank {rank} {all_gather_results[rank]}'
"#,
    );
}

#[test]
fn test_pycoeus_tcp_reduce() {
    run_pycoeus_tcp_script(
        r#"
def reduce(rank, comm):
    tensor = pycoeus.Tensor([float(rank + 1.0), float(rank + 2.0)])
    comm.reduce(tensor, 1)
    return tensor.data

reduce_results = run_tcp(reduce)
assert reduce_results[1] == [3.0, 5.0], f'tcp reduce root {reduce_results[1]}'
"#,
    );
}

#[test]
fn test_pycoeus_tcp_gather() {
    run_pycoeus_tcp_script(
        r#"
def gather(rank, comm):
    tensor = pycoeus.Tensor([float(rank * 100.0)])
    out = [pycoeus.Tensor([0.0]) for _ in range(2)] if rank == 1 else []
    comm.gather(tensor, out, 1)
    return [item.data[0] for item in out] if rank == 1 else None

gather_results = run_tcp(gather)
assert gather_results[1] == [0.0, 100.0], f'tcp gather root {gather_results[1]}'
"#,
    );
}

#[test]
fn test_pycoeus_tcp_scatter() {
    run_pycoeus_tcp_script(
        r#"
def scatter(rank, comm):
    tensor = pycoeus.Tensor([0.0])
    inputs = [pycoeus.Tensor([100.0]), pycoeus.Tensor([200.0])] if rank == 0 else []
    comm.scatter(tensor, inputs, 0)
    return tensor.data

scatter_results = run_tcp(scatter)
for rank in range(2):
    assert scatter_results[rank] == [(rank + 1.0) * 100.0], f'tcp scatter rank {rank} {scatter_results[rank]}'
"#,
    );
}
