//! Demonstrates Python comparison operators on `pycoeus.Tensor`.
//!
//! Run with: `cargo run -p coeus-python --example python_comparison_operators`

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::ffi::CString;

fn main() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let pycoeus_module = pyo3::types::PyModule::new(py, "pycoeus")
            .expect("invariant: pycoeus module allocation");
        pycoeus::pycoeus(&pycoeus_module).expect("invariant: pycoeus module initialization");
        let sys = py.import("sys").expect("invariant: stdlib sys import");
        let modules_any = sys
            .getattr("modules")
            .expect("invariant: sys.modules attribute");
        let modules = modules_any
            .downcast::<PyDict>()
            .expect("invariant: sys.modules is a dict");
        modules
            .set_item("pycoeus", &pycoeus_module)
            .expect("invariant: register pycoeus in sys.modules");
        let globals = PyDict::new(py);
        globals
            .set_item("pycoeus", &pycoeus_module)
            .expect("invariant: inject pycoeus into globals");

        let script = r#"
a = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [4])
b = pycoeus.Tensor([2.0, 2.0, 1.0, 5.0], [4])

eq = a == b
assert eq.data == [0.0, 1.0, 0.0, 0.0], f"eq failed: {eq.data}"

ne = a != b
assert ne.data == [1.0, 0.0, 1.0, 1.0], f"ne failed: {ne.data}"

lt = a < b
assert lt.data == [1.0, 0.0, 0.0, 1.0], f"lt failed: {lt.data}"

le = a <= b
assert le.data == [1.0, 1.0, 0.0, 1.0], f"le failed: {le.data}"

gt = a > b
assert gt.data == [0.0, 0.0, 1.0, 0.0], f"gt failed: {gt.data}"

ge = a >= b
assert ge.data == [0.0, 1.0, 1.0, 0.0], f"ge failed: {ge.data}"

assert (a == 2.0).data == [0.0, 1.0, 0.0, 0.0]
assert (a < 3.0).data == [1.0, 1.0, 0.0, 0.0]
assert (a > 2.0).data == [0.0, 0.0, 1.0, 1.0]
assert (a != 4.0).data == [1.0, 1.0, 1.0, 0.0]
print("python_comparison_operators: ok")
"#;
        let script = CString::new(script).expect("example script must not contain interior NUL");
        let result = py.run(script.as_c_str(), Some(&globals), None);
        modules
            .del_item("pycoeus")
            .unwrap_or_else(|e| panic!("failed to remove pycoeus example module: {e:?}"));
        result.unwrap_or_else(|e| panic!("Python comparison example failed:\n{e}"));
    });
}
