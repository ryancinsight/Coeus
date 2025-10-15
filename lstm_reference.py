#!/usr/bin/env python3
"""
Generate PyTorch LSTM reference outputs for numerical validation tests.

This script creates deterministic LSTM computations that can be compared
against Coeus implementation for numerical accuracy validation.
"""

import torch
import numpy as np

def create_deterministic_lstm(input_size=4, hidden_size=6, num_layers=1, bidirectional=False, bias=True):
    """Create LSTM with deterministic weights for reproducible testing."""

    # Set random seed for deterministic weights
    torch.manual_seed(42)

    lstm = torch.nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=False,  # Use (seq, batch, feature) format
        bidirectional=bidirectional
    )

    return lstm

def generate_reference_outputs():
    """Generate reference outputs for various LSTM configurations."""

    print("Generating PyTorch LSTM reference outputs...")

    # Test Case 1: Basic unidirectional LSTM
    print("\n1. Basic unidirectional LSTM (input_size=4, hidden_size=6, seq_len=3, batch=2)")

    lstm1 = create_deterministic_lstm(4, 6, 1, False, True)

    # Create deterministic input
    input1 = torch.tensor([
        [[0.5, 1.0, -0.5, 0.8], [0.3, -0.2, 0.9, -0.4]],  # seq=0, batch elements
        [[0.1, 0.6, -0.3, 0.7], [-0.1, 0.4, -0.8, 0.2]],  # seq=1
        [[-0.6, 0.9, 0.2, -0.5], [0.7, -0.3, 0.1, 0.8]]   # seq=2
    ], dtype=torch.float32)  # shape: (3, 2, 4)

    with torch.no_grad():
        output1, (h1, c1) = lstm1(input1)

    print(f"Input shape: {input1.shape}")
    print(f"Output shape: {output1.shape}")
    print(f"Hidden shape: {h1.shape}")
    print(f"Cell shape: {c1.shape}")
    print(f"Output values (first few): {output1.flatten()[:10].tolist()}")

    # Test Case 2: Bidirectional LSTM
    print("\n2. Bidirectional LSTM (same input)")

    lstm2 = create_deterministic_lstm(4, 6, 1, True, True)

    with torch.no_grad():
        output2, (h2, c2) = lstm2(input1)

    print(f"Output shape: {output2.shape}")
    print(f"Hidden shape: {h2.shape}")
    print(f"Cell shape: {c2.shape}")
    print(f"Output values (first few): {output2.flatten()[:10].tolist()}")

    # Test Case 3: Multi-layer LSTM
    print("\n3. Multi-layer LSTM (num_layers=2)")

    lstm3 = create_deterministic_lstm(4, 6, 2, False, True)

    with torch.no_grad():
        output3, (h3, c3) = lstm3(input1)

    print(f"Output shape: {output3.shape}")
    print(f"Hidden shape: {h3.shape}")
    print(f"Cell shape: {c3.shape}")
    print(f"Output values (first few): {output3.flatten()[:10].tolist()}")

    # Test Case 4: No bias
    print("\n4. LSTM without bias")

    lstm4 = create_deterministic_lstm(4, 6, 1, False, False)

    with torch.no_grad():
        output4, (h4, c4) = lstm4(input1)

    print(f"Output shape: {output4.shape}")
    print(f"Output values (first few): {output4.flatten()[:10].tolist()}")

    # Test Case 5: Single timestep
    print("\n5. Single timestep LSTM")

    input5 = input1[0:1]  # Take first timestep only, shape: (1, 2, 4)
    lstm5 = create_deterministic_lstm(4, 6, 1, False, True)

    with torch.no_grad():
        output5, (h5, c5) = lstm5(input5)

    print(f"Input shape: {input5.shape}")
    print(f"Output shape: {output5.shape}")
    print(f"Output values: {output5.flatten().tolist()}")

    # Save reference data for Rust tests
    reference_data = {
        'basic_unidirectional': {
            'input': input1.numpy().flatten().tolist(),
            'input_shape': list(input1.shape),
            'output': output1.numpy().flatten().tolist(),
            'output_shape': list(output1.shape),
            'hidden': h1.numpy().flatten().tolist(),
            'cell': c1.numpy().flatten().tolist()
        },
        'bidirectional': {
            'output': output2.numpy().flatten().tolist(),
            'output_shape': list(output2.shape),
            'hidden': h2.numpy().flatten().tolist(),
            'cell': c2.numpy().flatten().tolist()
        },
        'multilayer': {
            'output': output3.numpy().flatten().tolist(),
            'output_shape': list(output3.shape),
            'hidden': h3.numpy().flatten().tolist(),
            'cell': c3.numpy().flatten().tolist()
        },
        'no_bias': {
            'output': output4.numpy().flatten().tolist(),
            'output_shape': list(output4.shape)
        },
        'single_timestep': {
            'input': input5.numpy().flatten().tolist(),
            'input_shape': list(input5.shape),
            'output': output5.numpy().flatten().tolist(),
            'output_shape': list(output5.shape),
            'hidden': h5.numpy().flatten().tolist(),
            'cell': c5.numpy().flatten().tolist()
        }
    }

    # Print Rust-compatible arrays
    print("\n" + "="*50)
    print("RUST TEST DATA (copy to nn/src/rnn.rs)")
    print("="*50)

    for test_name, data in reference_data.items():
        print(f"\n// {test_name.upper()} REFERENCE DATA")
        if 'input' in data:
            print(f"const {test_name.upper()}_INPUT: &[f32] = &{data['input']};")
        if 'output' in data:
            print(f"const {test_name.upper()}_OUTPUT: &[f32] = &{data['output']};")
        if 'hidden' in data:
            print(f"const {test_name.upper()}_HIDDEN: &[f32] = &{data['hidden']};")
        if 'cell' in data:
            print(f"const {test_name.upper()}_CELL: &[f32] = &{data['cell']};")

    return reference_data

if __name__ == "__main__":
    reference_data = generate_reference_outputs()
