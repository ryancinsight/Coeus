#![expect(clippy::unwrap_used, reason = "ratchet COEUS-UNWRAP-1")]
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[test]
fn test_pycoeus_activations() {
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

    # CumSum Method & Function
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
