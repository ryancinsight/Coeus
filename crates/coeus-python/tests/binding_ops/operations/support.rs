//! Shared Python interpreter setup for operation-family contracts.
#![expect(clippy::unwrap_used, reason = "ratchet COEUS-UNWRAP-1")]

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::ffi::CString;

use crate::common;

pub(super) fn run_script(script: &str) {
    let _guard = common::python_test_lock()
        .lock()
        .expect("python test lock poisoned");
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let pycoeus_module = pyo3::types::PyModule::new(py, "pycoeus").unwrap();
        pycoeus::pycoeus(&pycoeus_module).unwrap();
        let sys = py.import("sys").unwrap();
        let modules_any = sys.getattr("modules").unwrap();
        let modules = modules_any.downcast::<PyDict>().unwrap();
        modules.set_item("pycoeus", &pycoeus_module).unwrap();
        let globals = PyDict::new(py);
        globals.set_item("pycoeus", &pycoeus_module).unwrap();
        let script = CString::new(script).expect("test script must not contain interior NUL");
        let result = py.run(script.as_c_str(), Some(&globals), None);
        modules
            .del_item("pycoeus")
            .unwrap_or_else(|e| panic!("failed to remove pycoeus test module: {e:?}"));
        result.unwrap_or_else(|e| panic!("Python script failed:\n{e}"));
    });
}
