use pyo3::prelude::*;
use pyo3::types::PyDict;

#[test]
fn test_pycoeus_optim() {
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

try:
    # 5. Optimizers
    param = pycoeus.Tensor([10.0], requires_grad=True)
    sgd = pycoeus.SGD([('weight', param)], lr=0.1)
    loss = param * pycoeus.Tensor([2.0])
    loss.backward()
    sgd.step()
    assert param.data[0] < 10.0, f'SGD step failed, param.data[0] is {param.data[0]}'

    # Adam
    param_adam = pycoeus.Tensor([10.0], requires_grad=True)
    adam = pycoeus.Adam([('weight', param_adam)], lr=0.1)
    loss_adam = param_adam * pycoeus.Tensor([2.0])
    loss_adam.backward()
    adam.step()
    assert param_adam.data[0] < 10.0, f'Adam step failed, param_adam.data[0] is {param_adam.data[0]}'

    # AdamW
    param_adamw = pycoeus.Tensor([10.0], requires_grad=True)
    adamw = pycoeus.AdamW([('weight', param_adamw)], lr=0.1, weight_decay=0.01)
    loss_adamw = param_adamw * pycoeus.Tensor([2.0])
    loss_adamw.backward()
    adamw.step()
    assert param_adamw.data[0] < 10.0, f'AdamW step failed, param_adamw.data[0] is {param_adamw.data[0]}'

    # RMSProp
    param_rmsprop = pycoeus.Tensor([10.0], requires_grad=True)
    rmsprop = pycoeus.RMSProp([('weight', param_rmsprop)], lr=0.1)
    loss_rmsprop = param_rmsprop * pycoeus.Tensor([2.0])
    loss_rmsprop.backward()
    rmsprop.step()
    assert param_rmsprop.data[0] < 10.0, f'RMSProp step failed, param_rmsprop.data[0] is {param_rmsprop.data[0]}'

    # AdaGrad Optimizer
    param_adagrad = pycoeus.Tensor([10.0], requires_grad=True)
    adagrad = pycoeus.AdaGrad([('weight', param_adagrad)], lr=0.1)
    loss_adagrad = param_adagrad * pycoeus.Tensor([2.0])
    loss_adagrad.backward()
    adagrad.step()
    assert abs(param_adagrad.data[0] - 9.9) < 1e-5, f'AdaGrad step failed, data is {param_adagrad.data[0]}'

except Exception as e:
    traceback.print_exc()
    sys.exit(1)
";

        if let Err(e) = py.run(test_script, None, None) {
            panic!("Python execution failed: {:?}", e);
        }
    });
}
