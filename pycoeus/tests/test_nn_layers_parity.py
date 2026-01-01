import torch
import coeus
import numpy as np
import pytest

def test_maxpool2d_parity():
    input_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
    input_t = torch.from_numpy(input_np)
    input_c = coeus.tensor(input_np)
    
    pool_t = torch.nn.MaxPool2d(2, stride=2)
    pool_c = coeus.nn.MaxPool2d(2, stride=2)
    
    out_t = pool_t(input_t).detach().numpy()
    out_c = pool_c(input_c).numpy()
    
    assert np.allclose(out_t, out_c, atol=1e-5)

def test_avgpool2d_parity():
    input_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
    input_t = torch.from_numpy(input_np)
    input_c = coeus.tensor(input_np)
    
    pool_t = torch.nn.AvgPool2d(2, stride=2)
    pool_c = coeus.nn.AvgPool2d(2, stride=2)
    
    out_t = pool_t(input_t).detach().numpy()
    out_c = pool_c(input_c).numpy()
    
    assert np.allclose(out_t, out_c, atol=1e-5)

def test_adaptive_avgpool2d_parity():
    input_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
    input_t = torch.from_numpy(input_np)
    input_c = coeus.tensor(input_np)
    
    pool_t = torch.nn.AdaptiveAvgPool2d((4, 4))
    pool_c = coeus.nn.AdaptiveAvgPool2d((4, 4))
    
    out_t = pool_t(input_t).detach().numpy()
    out_c = pool_c(input_c).numpy()
    
    assert np.allclose(out_t, out_c, atol=1e-5)

def test_layernorm_parity():
    input_np = np.random.randn(2, 10, 20).astype(np.float32)
    input_t = torch.from_numpy(input_np)
    input_c = coeus.tensor(input_np)
    
    norm_t = torch.nn.LayerNorm(20)
    norm_c = coeus.nn.LayerNorm(20)
    
    # Copy weights/bias for parity (both initialize to 1s and 0s by default)
    # But let's be explicit
    with torch.no_grad():
        norm_t.weight.fill_(1.0)
        norm_t.bias.fill_(0.0)
    
    out_t = norm_t(input_t).detach().numpy()
    out_c = norm_c(input_c).numpy()
    
    assert np.allclose(out_t, out_c, atol=1e-4)

def test_rnn_parity():
    input_np = np.random.randn(5, 3, 10).astype(np.float32) # (seq, batch, input)
    input_t = torch.from_numpy(input_np)
    input_c = coeus.tensor(input_np)
    
    rnn_t = torch.nn.RNN(10, 20, 1)
    rnn_c = coeus.nn.RNN(10, 20, 1)
    
    # Output comparison (shapes first)
    out_t, h_t = rnn_t(input_t)
    out_c, h_c = rnn_c(input_c)
    
    assert out_t.shape == tuple(out_c.shape)
    assert h_t.shape == tuple(h_c.shape)

def test_lstm_parity():
    input_np = np.random.randn(5, 3, 10).astype(np.float32) # (seq, batch, input)
    input_t = torch.from_numpy(input_np)
    input_c = coeus.tensor(input_np)
    
    lstm_t = torch.nn.LSTM(10, 20, 1)
    lstm_c = coeus.nn.LSTM(10, 20, 1)
    
    out_t, (h_t, c_t) = lstm_t(input_t)
    out_c, (h_c, c_c) = lstm_c(input_c)
    
    assert out_t.shape == tuple(out_c.shape)
    assert h_t.shape == tuple(h_c.shape)
    assert c_t.shape == tuple(c_c.shape)

def test_gru_parity():
    input_np = np.random.randn(5, 3, 10).astype(np.float32) # (seq, batch, input)
    input_t = torch.from_numpy(input_np)
    input_c = coeus.tensor(input_np)
    
    gru_t = torch.nn.GRU(10, 20, 1)
    gru_c = coeus.nn.GRU(10, 20, 1)
    
    assert gru_t.weight_ih_l0.shape == tuple(gru_c.parameters()[0].shape)
    
    out_t, h_t = gru_t(input_t)
    out_c, h_c = gru_c(input_c)
    
    assert out_t.shape == tuple(out_c.shape)
    assert h_t.shape == tuple(h_c.shape)

def test_activations_parity():
    input_np = np.random.randn(5, 5).astype(np.float32)
    input_t = torch.from_numpy(input_np)
    input_c = coeus.tensor(input_np)
    
    # ReLU
    assert np.allclose(torch.nn.ReLU()(input_t).detach().numpy(), coeus.nn.ReLU()(input_c).numpy())
    # GeLU
    assert np.allclose(torch.nn.GELU()(input_t).detach().numpy(), coeus.nn.GELU()(input_c).numpy(), atol=1e-3)
    # SiLU
    assert np.allclose(torch.nn.SiLU()(input_t).detach().numpy(), coeus.nn.SiLU()(input_c).numpy(), atol=1e-3)
    # Tanh
    assert np.allclose(torch.nn.Tanh()(input_t).detach().numpy(), coeus.nn.Tanh()(input_c).numpy())
    # Sigmoid
    assert np.allclose(torch.nn.Sigmoid()(input_t).detach().numpy(), coeus.nn.Sigmoid()(input_c).numpy())

if __name__ == "__main__":
    pytest.main([__file__])
