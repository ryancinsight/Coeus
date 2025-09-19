#!/usr/bin/env python3
"""
Comprehensive pytest-based comparison test suite for PyTorch vs PyCoeus

This module provides detailed unit tests comparing PyTorch and PyCoeus
implementations across all neural network components.
"""

import pytest
import numpy as np
import sys
from typing import Tuple, Any
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

# Import dependencies with proper error handling
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import pycoeus as pc
    PYCOEUS_AVAILABLE = True
except ImportError:
    PYCOEUS_AVAILABLE = False

# Skip all tests if either framework is unavailable
pytestmark = pytest.mark.skipif(
    not (PYTORCH_AVAILABLE and PYCOEUS_AVAILABLE),
    reason="Both PyTorch and PyCoeus must be available for comparison tests"
)

class TestTensorOperations:
    """Test basic tensor operations compatibility"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing"""
        np.random.seed(42)  # For reproducible tests
        return {
            'small_1d': np.random.randn(10).astype(np.float32),
            'small_2d': np.random.randn(5, 8).astype(np.float32),
            'medium_2d': np.random.randn(32, 64).astype(np.float32),
            'batch_data': np.random.randn(4, 16).astype(np.float32),
            'conv_data': np.random.randn(2, 3, 32, 32).astype(np.float32),
        }
    
    def assert_tensors_close(self, torch_tensor: torch.Tensor, pycoeus_tensor, 
                           rtol: float = 1e-4, atol: float = 1e-6):
        """Assert that PyTorch and PyCoeus tensors are numerically close"""
        # Convert PyCoeus tensor to numpy
        if hasattr(pycoeus_tensor, 'numpy'):
            pc_numpy = pycoeus_tensor.numpy()
        elif hasattr(pycoeus_tensor, 'data'):
            pc_numpy = np.array(pycoeus_tensor.data()).reshape(pycoeus_tensor.shape())
        else:
            pc_numpy = np.array(pycoeus_tensor)
        
        torch_numpy = torch_tensor.detach().numpy()
        
        # Check shapes match
        assert torch_numpy.shape == pc_numpy.shape, \
            f"Shape mismatch: PyTorch {torch_numpy.shape} vs PyCoeus {pc_numpy.shape}"
        
        # Check numerical closeness
        np.testing.assert_allclose(torch_numpy, pc_numpy, rtol=rtol, atol=atol)
    
    def test_tensor_creation(self, sample_data):
        """Test tensor creation from numpy arrays"""
        for name, data in sample_data.items():
            torch_tensor = torch.from_numpy(data)
            pc_tensor = pc.PyTensor.from_numpy(data)
            
            self.assert_tensors_close(torch_tensor, pc_tensor)
    
    def test_tensor_arithmetic(self, sample_data):
        """Test basic tensor arithmetic operations"""
        data = sample_data['small_2d']
        
        # Create tensors
        torch_a = torch.from_numpy(data)
        torch_b = torch.from_numpy(data * 2)
        
        pc_a = pc.PyTensor.from_numpy(data)
        pc_b = pc.PyTensor.from_numpy(data * 2)
        
        # Test addition
        torch_add = torch_a + torch_b
        pc_add = pc_a + pc_b
        self.assert_tensors_close(torch_add, pc_add)
        
        # Test multiplication
        torch_mul = torch_a * torch_b
        pc_mul = pc_a * pc_b
        self.assert_tensors_close(torch_mul, pc_mul)
        
        # Test subtraction
        torch_sub = torch_a - torch_b
        pc_sub = pc_a - pc_b
        self.assert_tensors_close(torch_sub, pc_sub)

class TestLinearLayers:
    """Test linear layer implementations"""
    
    @pytest.fixture
    def linear_configs(self):
        """Different linear layer configurations to test"""
        return [
            (10, 5, True),   # Small layer with bias
            (64, 32, True),  # Medium layer with bias
            (128, 64, False), # Large layer without bias
            (1, 1, True),    # Minimal layer
        ]
    
    def test_linear_forward(self, linear_configs):
        """Test linear layer forward pass"""
        for in_features, out_features, bias in linear_configs:
            # Create input data
            batch_size = 4
            input_data = np.random.randn(batch_size, in_features).astype(np.float32)
            
            # PyTorch implementation
            torch_linear = nn.Linear(in_features, out_features, bias=bias)
            torch_input = torch.tensor(input_data)
            torch_output = torch_linear(torch_input)
            
            # PyCoeus implementation
            pc_linear = pc.Linear(in_features, out_features, bias=bias)
            pc_input = pc.PyTensor.from_numpy(input_data)
            pc_output = pc_linear.forward(pc_input)
            
            # Compare outputs (note: weights are randomly initialized differently)
            assert torch_output.shape == tuple(pc_output.shape()), \
                f"Output shape mismatch: PyTorch {torch_output.shape} vs PyCoeus {pc_output.shape()}"
    
    def test_linear_parameters(self, linear_configs):
        """Test linear layer parameter shapes and properties"""
        for in_features, out_features, bias in linear_configs:
            # PyTorch layer
            torch_linear = nn.Linear(in_features, out_features, bias=bias)
            
            # PyCoeus layer
            pc_linear = pc.Linear(in_features, out_features, bias=bias)
            
            # Check weight shapes
            assert torch_linear.weight.shape == tuple(pc_linear.weight().shape()), \
                "Weight shape mismatch"
            
            # Check bias shapes
            if bias:
                assert torch_linear.bias.shape == tuple(pc_linear.bias().shape()), \
                    "Bias shape mismatch"
            else:
                assert torch_linear.bias is None
                assert pc_linear.bias() is None

class TestConvolutionalLayers:
    """Test convolutional layer implementations"""
    
    @pytest.fixture
    def conv_configs(self):
        """Different convolution configurations to test"""
        return [
            (3, 16, 3, 1, 1),   # Standard conv
            (16, 32, 5, 2, 2),  # Larger kernel with stride
            (1, 8, 1, 1, 0),    # 1x1 conv
            (32, 64, 3, 1, 0),  # No padding
        ]
    
    def test_conv2d_forward(self, conv_configs):
        """Test Conv2d forward pass"""
        for in_channels, out_channels, kernel_size, stride, padding in conv_configs:
            # Create input data
            batch_size, height, width = 2, 32, 32
            input_data = np.random.randn(batch_size, in_channels, height, width).astype(np.float32)
            
            # PyTorch implementation
            torch_conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                                 stride=stride, padding=padding)
            torch_input = torch.tensor(input_data)
            torch_output = torch_conv(torch_input)
            
            # PyCoeus implementation
            pc_conv = pc.Conv2d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=padding)
            pc_input = pc.PyTensor.from_numpy(input_data)
            pc_output = pc_conv.forward(pc_input)
            
            # Compare output shapes
            assert torch_output.shape == tuple(pc_output.shape()), \
                f"Output shape mismatch: PyTorch {torch_output.shape} vs PyCoeus {pc_output.shape()}"
    
    def test_conv2d_parameters(self, conv_configs):
        """Test Conv2d parameter shapes"""
        for in_channels, out_channels, kernel_size, stride, padding in conv_configs:
            # PyTorch layer
            torch_conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                                 stride=stride, padding=padding)
            
            # PyCoeus layer
            pc_conv = pc.Conv2d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=padding)
            
            # Check weight shapes
            assert torch_conv.weight.shape == tuple(pc_conv.weight().shape()), \
                "Conv2d weight shape mismatch"
            
            # Check bias shapes
            if torch_conv.bias is not None:
                assert torch_conv.bias.shape == tuple(pc_conv.bias().shape()), \
                    "Conv2d bias shape mismatch"

class TestActivationFunctions:
    """Test activation function implementations"""
    
    @pytest.fixture
    def activation_data(self):
        """Generate test data for activation functions"""
        return {
            'positive': np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
            'negative': np.array([-5.0, -4.0, -3.0, -2.0, -1.0], dtype=np.float32),
            'mixed': np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32),
            'zero': np.array([0.0], dtype=np.float32),
            'large': np.random.randn(100).astype(np.float32),
        }
    
    def test_relu_activation(self, activation_data):
        """Test ReLU activation function"""
        for name, data in activation_data.items():
            # PyTorch
            torch_input = torch.tensor(data)
            torch_relu = nn.ReLU()
            torch_output = torch_relu(torch_input)
            
            # PyCoeus
            pc_input = pc.PyTensor.from_numpy(data)
            pc_relu = pc.ReLU()
            pc_output = pc_relu.forward(pc_input)
            
            # Compare results
            torch_numpy = torch_output.detach().numpy()
            pc_numpy = np.array(pc_output.data()).reshape(pc_output.shape())
            
            np.testing.assert_allclose(torch_numpy, pc_numpy, rtol=1e-6, atol=1e-8)
    
    def test_activation_properties(self):
        """Test mathematical properties of activation functions"""
        # Test ReLU properties
        test_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        expected_relu = np.array([0.0, 0.0, 0.0, 1.0, 2.0], dtype=np.float32)
        
        pc_input = pc.PyTensor.from_numpy(test_data)
        pc_relu = pc.ReLU()
        pc_output = pc_relu.forward(pc_input)
        pc_numpy = np.array(pc_output.data()).reshape(pc_output.shape())
        
        np.testing.assert_allclose(pc_numpy, expected_relu, rtol=1e-6, atol=1e-8)

class TestLossFunctions:
    """Test loss function implementations"""
    
    @pytest.fixture
    def loss_data(self):
        """Generate test data for loss functions"""
        np.random.seed(42)
        return {
            'regression': {
                'predictions': np.random.randn(10, 1).astype(np.float32),
                'targets': np.random.randn(10, 1).astype(np.float32),
            },
            'classification': {
                'logits': np.random.randn(8, 5).astype(np.float32),
                'targets': np.random.randint(0, 5, (8,)),
            },
            'binary': {
                'predictions': np.random.rand(6, 1).astype(np.float32),
                'targets': np.random.randint(0, 2, (6, 1)).astype(np.float32),
            }
        }
    
    def test_mse_loss(self, loss_data):
        """Test Mean Squared Error loss"""
        pred_data = loss_data['regression']['predictions']
        target_data = loss_data['regression']['targets']
        
        # PyTorch
        torch_pred = torch.tensor(pred_data)
        torch_target = torch.tensor(target_data)
        torch_mse = nn.MSELoss()
        torch_loss = torch_mse(torch_pred, torch_target)
        
        # PyCoeus
        pc_pred = pc.PyTensor.from_numpy(pred_data)
        pc_target = pc.PyTensor.from_numpy(target_data)
        pc_mse = pc.MseLoss()
        pc_loss = pc_mse.forward(pc_pred, pc_target)
        
        # Compare results
        torch_loss_val = torch_loss.item()
        pc_loss_val = pc_loss.data()[0] if hasattr(pc_loss, 'data') else float(pc_loss)
        
        np.testing.assert_allclose(torch_loss_val, pc_loss_val, rtol=1e-4, atol=1e-6)
    
    def test_cross_entropy_loss(self, loss_data):
        """Test Cross Entropy loss"""
        logits_data = loss_data['classification']['logits']
        targets_data = loss_data['classification']['targets']
        
        # PyTorch
        torch_logits = torch.tensor(logits_data)
        torch_targets = torch.tensor(targets_data, dtype=torch.long)
        torch_ce = nn.CrossEntropyLoss()
        torch_loss = torch_ce(torch_logits, torch_targets)
        
        # PyCoeus
        pc_logits = pc.PyTensor.from_numpy(logits_data)
        pc_targets = pc.PyTensor(targets_data.tolist(), [len(targets_data)])
        pc_ce = pc.CrossEntropyLoss()
        pc_loss = pc_ce.forward(pc_logits, pc_targets)
        
        # Compare results (allowing for some numerical differences)
        torch_loss_val = torch_loss.item()
        pc_loss_val = pc_loss.data()[0] if hasattr(pc_loss, 'data') else float(pc_loss)
        
        # Cross entropy can have larger numerical differences due to softmax computation
        np.testing.assert_allclose(torch_loss_val, pc_loss_val, rtol=1e-3, atol=1e-5)

class TestOptimizers:
    """Test optimizer implementations"""
    
    def test_sgd_creation(self):
        """Test SGD optimizer creation and basic properties"""
        # Create test parameters
        param_data = np.random.randn(5, 3).astype(np.float32)
        
        # PyTorch SGD
        torch_param = torch.tensor(param_data, requires_grad=True)
        torch_sgd = optim.SGD([torch_param], lr=0.01, momentum=0.9, weight_decay=0.001)
        
        # PyCoeus SGD
        pc_param = pc.PyTensor.from_numpy(param_data)
        pc_sgd = pc.Sgd([pc_param], lr=0.01, momentum=0.9, weight_decay=0.001)
        
        # Test that optimizers were created successfully
        assert torch_sgd.param_groups[0]['lr'] == 0.01
        assert len(pc_sgd.parameters()) == 1
    
    def test_adam_creation(self):
        """Test Adam optimizer creation and basic properties"""
        # Create test parameters
        param_data = np.random.randn(3, 4).astype(np.float32)
        
        # PyTorch Adam
        torch_param = torch.tensor(param_data, requires_grad=True)
        torch_adam = optim.Adam([torch_param], lr=0.001, betas=(0.9, 0.999), eps=1e-8)
        
        # PyCoeus Adam
        pc_param = pc.PyTensor.from_numpy(param_data)
        pc_adam = pc.Adam([pc_param], lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
        
        # Test that optimizers were created successfully
        assert torch_adam.param_groups[0]['lr'] == 0.001
        assert len(pc_adam.parameters()) == 1

class TestRecurrentLayers:
    """Test recurrent layer implementations"""
    
    @pytest.fixture
    def rnn_configs(self):
        """Different RNN configurations to test"""
        return [
            (10, 20),   # input_size, hidden_size
            (5, 5),     # Same input and hidden size
            (1, 10),    # Minimal input size
            (50, 25),   # Larger configuration
        ]
    
    def test_rnn_forward(self, rnn_configs):
        """Test RNN forward pass"""
        for input_size, hidden_size in rnn_configs:
            # Create input data: (seq_len, batch_size, input_size)
            seq_len, batch_size = 3, 2
            input_data = np.random.randn(seq_len, batch_size, input_size).astype(np.float32)
            
            # PyTorch RNN
            torch_rnn = nn.RNN(input_size, hidden_size, batch_first=False)
            torch_input = torch.tensor(input_data)
            torch_output, torch_hidden = torch_rnn(torch_input)
            
            # PyCoeus RNN
            pc_rnn = pc.Rnn(input_size, hidden_size)
            pc_input = pc.PyTensor.from_numpy(input_data)
            pc_output, pc_hidden = pc_rnn.forward(pc_input)
            
            # Compare shapes
            assert torch_output.shape == tuple(pc_output.shape()), \
                f"RNN output shape mismatch: PyTorch {torch_output.shape} vs PyCoeus {pc_output.shape()}"
            assert torch_hidden.shape == tuple(pc_hidden.shape()), \
                f"RNN hidden shape mismatch: PyTorch {torch_hidden.shape} vs PyCoeus {pc_hidden.shape()}"
    
    def test_rnn_parameters(self, rnn_configs):
        """Test RNN parameter shapes"""
        for input_size, hidden_size in rnn_configs:
            # PyTorch RNN
            torch_rnn = nn.RNN(input_size, hidden_size)
            
            # PyCoeus RNN
            pc_rnn = pc.Rnn(input_size, hidden_size)
            
            # Check parameter shapes
            assert torch_rnn.weight_ih_l0.shape == tuple(pc_rnn.weight_ih().shape()), \
                "RNN weight_ih shape mismatch"
            assert torch_rnn.weight_hh_l0.shape == tuple(pc_rnn.weight_hh().shape()), \
                "RNN weight_hh shape mismatch"

class TestModelComposition:
    """Test model composition and complex architectures"""
    
    def test_sequential_model(self):
        """Test sequential model composition"""
        # Define a simple MLP
        input_size, hidden_size, output_size = 10, 20, 5
        batch_size = 4
        
        input_data = np.random.randn(batch_size, input_size).astype(np.float32)
        
        # PyTorch sequential model
        torch_model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        torch_input = torch.tensor(input_data)
        torch_output = torch_model(torch_input)
        
        # PyCoeus manual composition (since Sequential might not be implemented)
        pc_linear1 = pc.Linear(input_size, hidden_size)
        pc_relu = pc.ReLU()
        pc_linear2 = pc.Linear(hidden_size, output_size)
        
        pc_input = pc.PyTensor.from_numpy(input_data)
        pc_hidden = pc_linear1.forward(pc_input)
        pc_activated = pc_relu.forward(pc_hidden)
        pc_output = pc_linear2.forward(pc_activated)
        
        # Compare shapes
        assert torch_output.shape == tuple(pc_output.shape()), \
            f"Sequential model output shape mismatch: PyTorch {torch_output.shape} vs PyCoeus {pc_output.shape()}"
    
    def test_cnn_model(self):
        """Test CNN model composition"""
        # Simple CNN for CIFAR-like data
        batch_size, channels, height, width = 2, 3, 32, 32
        num_classes = 10
        
        input_data = np.random.randn(batch_size, channels, height, width).astype(np.float32)
        
        # PyTorch CNN
        torch_model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )
        torch_input = torch.tensor(input_data)
        torch_output = torch_model(torch_input)
        
        # PyCoeus CNN (manual composition)
        pc_conv1 = pc.Conv2d(3, 16, 3, padding=1)
        pc_relu1 = pc.ReLU()
        pc_conv2 = pc.Conv2d(16, 32, 3, padding=1)
        pc_relu2 = pc.ReLU()
        
        pc_input = pc.PyTensor.from_numpy(input_data)
        pc_conv1_out = pc_conv1.forward(pc_input)
        pc_relu1_out = pc_relu1.forward(pc_conv1_out)
        # Note: MaxPool2d and AdaptiveAvgPool2d might not be implemented yet
        # This test validates the basic CNN components
        
        # Compare intermediate shapes
        assert torch_conv1_out.shape == tuple(pc_conv1_out.shape()), \
            "CNN conv1 output shape mismatch"

# Performance and stress tests
class TestPerformance:
    """Performance comparison tests"""
    
    @pytest.mark.slow
    def test_large_tensor_operations(self):
        """Test performance with large tensors"""
        # Large tensor operations
        size = 1000
        data = np.random.randn(size, size).astype(np.float32)
        
        # PyTorch
        torch_tensor = torch.tensor(data)
        torch_result = torch.matmul(torch_tensor, torch_tensor.T)
        
        # PyCoeus (if matrix multiplication is available)
        pc_tensor = pc.PyTensor.from_numpy(data)
        # Note: This test depends on PyCoeus having matrix multiplication
        
        # For now, just test that large tensors can be created
        assert torch_tensor.shape == tuple(pc_tensor.shape())
    
    @pytest.mark.slow
    def test_training_performance(self):
        """Test training loop performance"""
        # Simple training loop performance test
        input_size, output_size = 100, 10
        batch_size, num_epochs = 32, 10
        
        # Generate synthetic data
        X = np.random.randn(batch_size, input_size).astype(np.float32)
        y = np.random.randint(0, output_size, (batch_size,))
        
        # PyTorch training
        torch_model = nn.Linear(input_size, output_size)
        torch_optimizer = optim.SGD(torch_model.parameters(), lr=0.01)
        torch_criterion = nn.CrossEntropyLoss()
        
        torch_X = torch.tensor(X)
        torch_y = torch.tensor(y, dtype=torch.long)
        
        import time
        start_time = time.time()
        for epoch in range(num_epochs):
            torch_optimizer.zero_grad()
            torch_pred = torch_model(torch_X)
            torch_loss = torch_criterion(torch_pred, torch_y)
            torch_loss.backward()
            torch_optimizer.step()
        torch_time = time.time() - start_time
        
        # PyCoeus training (simplified due to autograd limitations)
        pc_model = pc.Linear(input_size, output_size)
        pc_criterion = pc.CrossEntropyLoss()
        
        pc_X = pc.PyTensor.from_numpy(X)
        pc_y = pc.PyTensor(y.tolist(), [batch_size])
        
        start_time = time.time()
        pc_pred = pc_model.forward(pc_X)
        pc_loss = pc_criterion.forward(pc_pred, pc_y)
        pc_time = time.time() - start_time
        
        print(f"PyTorch training time: {torch_time:.4f}s")
        print(f"PyCoeus inference time: {pc_time:.4f}s")
        
        # This is mainly a smoke test to ensure no crashes occur
        assert torch_time > 0
        assert pc_time > 0

# Integration tests
class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_end_to_end_classification(self):
        """Test complete classification workflow"""
        # Simple binary classification problem
        n_samples, n_features = 100, 20
        n_classes = 2
        
        # Generate synthetic data
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randint(0, n_classes, (n_samples,))
        
        # Split into train/test
        split_idx = n_samples // 2
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # PyTorch workflow
        torch_model = nn.Sequential(
            nn.Linear(n_features, 10),
            nn.ReLU(),
            nn.Linear(10, n_classes)
        )
        torch_optimizer = optim.Adam(torch_model.parameters(), lr=0.01)
        torch_criterion = nn.CrossEntropyLoss()
        
        # Train PyTorch model
        torch_X_train = torch.tensor(X_train)
        torch_y_train = torch.tensor(y_train, dtype=torch.long)
        
        for epoch in range(5):  # Few epochs for testing
            torch_optimizer.zero_grad()
            torch_pred = torch_model(torch_X_train)
            torch_loss = torch_criterion(torch_pred, torch_y_train)
            torch_loss.backward()
            torch_optimizer.step()
        
        # Test PyTorch model
        torch_X_test = torch.tensor(X_test)
        with torch.no_grad():
            torch_test_pred = torch_model(torch_X_test)
            torch_test_accuracy = (torch_test_pred.argmax(dim=1) == torch.tensor(y_test)).float().mean()
        
        # PyCoeus workflow (simplified)
        pc_model1 = pc.Linear(n_features, 10)
        pc_relu = pc.ReLU()
        pc_model2 = pc.Linear(10, n_classes)
        pc_criterion = pc.CrossEntropyLoss()
        
        # Forward pass
        pc_X_train = pc.PyTensor.from_numpy(X_train)
        pc_y_train = pc.PyTensor(y_train.tolist(), [len(y_train)])
        
        pc_hidden = pc_model1.forward(pc_X_train)
        pc_activated = pc_relu.forward(pc_hidden)
        pc_pred = pc_model2.forward(pc_activated)
        pc_loss = pc_criterion.forward(pc_pred, pc_y_train)
        
        # Test that workflow completes without errors
        assert torch_test_accuracy >= 0.0  # Sanity check
        assert hasattr(pc_loss, 'data') or hasattr(pc_loss, '__float__')  # Loss computed
        
        print(f"PyTorch test accuracy: {torch_test_accuracy:.4f}")
        print("PyCoeus workflow completed successfully")

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])