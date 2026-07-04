use pyo3::prelude::*;
use pyo3::types::PyDict;

#[test]
fn test_pycoeus_nn() {
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

    # Conv3d
    conv3d = pycoeus.Conv3d(in_channels=2, out_channels=4, kernel_size=3)
    x_3d = pycoeus.Tensor([1.0]*250, [1, 2, 5, 5, 5]) # batch=1, in_channels=2, depth=5, height=5, width=5
    y_3d = conv3d.forward(x_3d)
    assert y_3d.shape[0] == 1
    assert y_3d.shape[1] == 4
    assert y_3d.shape[2] == 3
    assert y_3d.shape[3] == 3
    assert y_3d.shape[4] == 3

    # GroupNorm
    gn = pycoeus.GroupNorm(num_groups=2, num_channels=4)
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

    # Test GroupNorm
    gn = pycoeus.GroupNorm(num_groups=2, num_channels=4)
    x_gn = pycoeus.Tensor(list(range(1, 9)), [1, 4, 2], requires_grad=True)
    out_gn = gn.forward(x_gn)
    assert out_gn.shape == [1, 4, 2], f'GroupNorm shape is {out_gn.shape}'
    out_gn.backward()
    assert x_gn.grad is not None, 'GroupNorm input grad is None'
    assert gn.weight.grad is not None, 'GroupNorm weight grad is None'
    assert gn.bias.grad is not None, 'GroupNorm bias grad is None'

    # Test GroupNorm G=1
    gn1 = pycoeus.GroupNorm(num_groups=1, num_channels=4)
    x_gn1 = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 4], requires_grad=True)
    out_gn1 = gn1.forward(x_gn1)
    assert out_gn1.shape == [2, 4], f'GroupNorm(G=1) shape is {out_gn1.shape}'
    out_gn1.backward()
    assert x_gn1.grad is not None

    # Test InstanceNorm1d
    in1d = pycoeus.InstanceNorm1d(num_features=3)
    x_in1d = pycoeus.Tensor([float(i) for i in range(24)], [2, 3, 4], requires_grad=True)
    out_in1d = in1d.forward(x_in1d)
    assert out_in1d.shape == [2, 3, 4], f'InstanceNorm1d shape is {out_in1d.shape}'
    out_in1d.backward()
    assert x_in1d.grad is not None
    assert in1d.weight.grad is not None
    assert in1d.bias.grad is not None

    # Test InstanceNorm2d
    in2d = pycoeus.InstanceNorm2d(num_features=2)
    x_in2d = pycoeus.Tensor([float(i) for i in range(18)], [1, 2, 3, 3], requires_grad=True)
    out_in2d = in2d.forward(x_in2d)
    assert out_in2d.shape == [1, 2, 3, 3], f'InstanceNorm2d shape is {out_in2d.shape}'
    out_in2d.backward()
    assert x_in2d.grad is not None
    assert in2d.weight.grad is not None
    assert in2d.bias.grad is not None

    # Test MultiHeadAttention (self-attention)
    mha = pycoeus.MultiHeadAttention(d_model=8, num_heads=4)
    x_mha = pycoeus.Tensor([0.1 * i for i in range(40)], [1, 5, 8], requires_grad=True)
    out_mha = mha.forward(x_mha)
    out_mha_cross = mha.forward_cross(x_mha, x_mha, x_mha)
    assert out_mha.shape == [1, 5, 8], f'MHA self-attention shape is {out_mha.shape}'
    for a, b in zip(out_mha.data, out_mha_cross.data):
        assert abs(a - b) < 1e-9, f'MHA self/cross SSOT mismatch: {a} vs {b}'
    out_mha.backward()
    assert x_mha.grad is not None, 'MHA input grad is None'
    assert mha.w_q.grad is not None, 'MHA w_q grad is None'
    assert mha.w_k.grad is not None, 'MHA w_k grad is None'
    assert mha.w_v.grad is not None, 'MHA w_v grad is None'
    assert mha.w_o.grad is not None, 'MHA w_o grad is None'

    # Test MultiHeadAttention cross-attention
    mha2 = pycoeus.MultiHeadAttention(d_model=8, num_heads=2)
    q_mha = pycoeus.Tensor([0.1 * i for i in range(24)], [1, 3, 8], requires_grad=True)
    k_mha = pycoeus.Tensor([0.1 * i for i in range(40)], [1, 5, 8], requires_grad=False)
    v_mha = pycoeus.Tensor([0.1 * i for i in range(40)], [1, 5, 8], requires_grad=False)
    out_cross = mha2.forward_cross(q_mha, k_mha, v_mha)
    assert out_cross.shape == [1, 3, 8], f'MHA cross-attention shape is {out_cross.shape}'
    out_cross.backward()
    assert q_mha.grad is not None, 'MHA cross-attention query grad is None'

    # Rotary Positional Embedding (RoPE)
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

    # General Transpose Method
    x_tr = pycoeus.Tensor([float(i) for i in range(24)], [2, 3, 4], requires_grad=True)
    y_tr = x_tr.transpose(0, 2)
    assert y_tr.shape == [4, 3, 2], f'y_tr.shape is {y_tr.shape}'
    loss_tr = y_tr.sum_axis(0).sum_axis(1).sum_axis(2)
    loss_tr.backward()
    assert x_tr.grad is not None, 'x_tr.grad is None'
    assert all(abs(g - 1.0) < 1e-5 for g in x_tr.grad)

    # Tensor shape methods
    x_shape = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], requires_grad=True)
    x_reshaped = x_shape.reshape([1, 6])
    assert x_reshaped.shape == [1, 6], f'reshape shape={x_reshaped.shape}'
    x_reshaped.backward()
    assert x_shape.grad is not None

    x_shape2 = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], requires_grad=True)
    x_permuted = x_shape2.permute([1, 0])
    assert x_permuted.shape == [3, 2]
    assert x_permuted.data == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    x_permuted.backward()
    assert x_shape2.grad is not None

    x_shape3 = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], requires_grad=True)
    x_unsqueezed = x_shape3.unsqueeze(1)
    assert x_unsqueezed.shape == [2, 1, 3]
    x_unsqueezed.backward()
    assert x_shape3.grad is not None

    x_shape4 = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 1, 3], requires_grad=True)
    x_squeezed = x_shape4.squeeze(1)
    assert x_squeezed.shape == [2, 3]
    x_squeezed.backward()
    assert x_shape4.grad is not None

    x_shape5 = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 2, 1, 3], requires_grad=True)
    x_squeezed_all = x_shape5.squeeze()
    assert x_squeezed_all.shape == [2, 3]
    x_squeezed_all.backward()
    assert x_shape5.grad is not None

    x_shape6 = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], requires_grad=True)
    x_t = x_shape6.t()
    assert x_t.shape == [3, 2]
    assert x_t.data == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    x_t.backward()
    assert x_shape6.grad is not None

    x_shape7 = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], requires_grad=True)
    x_transposed = x_shape7.t()
    x_cont = x_transposed.contiguous()
    assert x_cont.shape == [3, 2]
    x_cont.backward()
    assert x_shape7.grad is not None

    # Module parameters() and zero_grad() tests
    linear_test = pycoeus.Linear(2, 3)
    params = linear_test.parameters()
    assert len(params) == 2, f'Expected 2 parameters, got {len(params)}'

    # Run forward/backward to populate grads
    x_test = pycoeus.Tensor([1.0, 2.0], [1, 2], requires_grad=True)
    y_test = linear_test.forward(x_test)
    loss_test = y_test.sum_axis(0).sum_axis(1)
    loss_test.backward()

    assert linear_test.weight.grad is not None
    assert linear_test.bias.grad is not None

    linear_test.zero_grad()
    assert all(g == 0.0 for g in linear_test.weight.grad)
    assert all(g == 0.0 for g in linear_test.bias.grad)

    # Check non-learnable modules have empty parameters and no-op zero_grad
    dp_test = pycoeus.Dropout(0.5)
    assert dp_test.parameters() == []
    dp_test.zero_grad()

    avgpool_test = pycoeus.AvgPool2d(2)
    assert avgpool_test.parameters() == []
    avgpool_test.zero_grad()

    # GlobalAvgPool1d: [N, C, L] -> [N, C, 1]
    gap1d = pycoeus.GlobalAvgPool1d()
    assert gap1d.parameters() == []
    x_gap1d = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 2, 3], requires_grad=True)
    y_gap1d = gap1d.forward(x_gap1d)
    assert y_gap1d.shape == [1, 2, 1], f'GlobalAvgPool1d shape: {y_gap1d.shape}'
    # channel 0 mean = (1+2+3)/3 = 2.0, channel 1 mean = (4+5+6)/3 = 5.0
    assert abs(y_gap1d.data[0] - 2.0) < 1e-6, f'gap1d ch0: {y_gap1d.data[0]}'
    assert abs(y_gap1d.data[1] - 5.0) < 1e-6, f'gap1d ch1: {y_gap1d.data[1]}'
    y_gap1d.backward()
    assert x_gap1d.grad is not None

    # GlobalAvgPool2d: [N, C, H, W] -> [N, C, 1, 1]
    gap2d = pycoeus.GlobalAvgPool2d()
    x_gap2d = pycoeus.Tensor([1.0]*8, [1, 2, 2, 2], requires_grad=True)
    y_gap2d = gap2d.forward(x_gap2d)
    assert y_gap2d.shape == [1, 2, 1, 1], f'GlobalAvgPool2d shape: {y_gap2d.shape}'
    assert abs(y_gap2d.data[0] - 1.0) < 1e-6, f'gap2d ch0: {y_gap2d.data[0]}'
    y_gap2d.backward()
    assert x_gap2d.grad is not None

    # GlobalAvgPool3d: [N, C, D, H, W] -> [N, C, 1, 1, 1]
    gap3d = pycoeus.GlobalAvgPool3d()
    x_gap3d = pycoeus.Tensor([1.0]*16, [1, 2, 2, 2, 2], requires_grad=True)
    y_gap3d = gap3d.forward(x_gap3d)
    assert y_gap3d.shape == [1, 2, 1, 1, 1], f'GlobalAvgPool3d shape: {y_gap3d.shape}'
    y_gap3d.backward()
    assert x_gap3d.grad is not None

    # GlobalMaxPool2d: [N, C, H, W] -> [N, C, 1, 1]
    gmp2d = pycoeus.GlobalMaxPool2d()
    x_gmp2d = pycoeus.Tensor([1.0, 3.0, 2.0, 4.0], [1, 1, 2, 2], requires_grad=True)
    y_gmp2d = gmp2d.forward(x_gmp2d)
    assert y_gmp2d.shape == [1, 1, 1, 1], f'GlobalMaxPool2d shape: {y_gmp2d.shape}'
    assert abs(y_gmp2d.data[0] - 4.0) < 1e-6, f'gmp2d value: {y_gmp2d.data[0]}'
    y_gmp2d.backward()
    assert x_gmp2d.grad is not None

    # GlobalMaxPool3d: [N, C, D, H, W] -> [N, C, 1, 1, 1]
    gmp3d = pycoeus.GlobalMaxPool3d()
    x_gmp3d = pycoeus.Tensor([1.0]*16, [1, 2, 2, 2, 2], requires_grad=True)
    y_gmp3d = gmp3d.forward(x_gmp3d)
    assert y_gmp3d.shape == [1, 2, 1, 1, 1], f'GlobalMaxPool3d shape: {y_gmp3d.shape}'
    y_gmp3d.backward()
    assert x_gmp3d.grad is not None

    rope_test = pycoeus.RotaryEmbedding(10, 4)
    assert rope_test.parameters() == []
    rope_test.zero_grad()

except Exception as e:
    traceback.print_exc()
    sys.exit(1)
";

        if let Err(e) = py.run(test_script, None, None) {
            panic!("Python execution failed: {:?}", e);
        }
    });
}
