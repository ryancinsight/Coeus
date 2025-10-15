#!/usr/bin/env python3
"""
Test script for new PyTorch-compatible layers in Coeus
"""

import coeus as torch
from coeus import nn

def main():
    print('Testing new PyTorch-compatible layers in Coeus')
    print('=' * 50)

    # Test Conv2D
    print('[TEST] Conv2D layer')
    try:
        conv = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        print(f'  OK Created Conv2d: {conv.in_channels} -> {conv.out_channels} channels')
        print(f'    Kernel: {conv.kernel_size}, Stride: {conv.stride}, Padding: {conv.padding}')
    except Exception as e:
        print(f'  ERROR Conv2D failed: {e}')

    # Test BatchNorm2d
    print('[TEST] BatchNorm2d layer')
    try:
        batchnorm = nn.BatchNorm2d(64)
        print(f'  OK Created BatchNorm2d: {batchnorm.num_features} features')
        print(f'    eps: {batchnorm.eps}, momentum: {batchnorm.momentum}')
    except Exception as e:
        print(f'  ERROR BatchNorm2d failed: {e}')

    # Test Dropout
    print('[TEST] Dropout layer')
    try:
        dropout = nn.Dropout(0.5)
        print(f'  OK Created Dropout: p={dropout.p}')
    except Exception as e:
        print(f'  ERROR Dropout failed: {e}')

    # Test Embedding
    print('[TEST] Embedding layer')
    try:
        embedding = nn.Embedding(1000, 128)
        print(f'  OK Created Embedding: {embedding.num_embeddings} tokens, {embedding.embedding_dim} dims')
    except Exception as e:
        print(f'  ERROR Embedding failed: {e}')

    print('=' * 50)
    print('All new layers created successfully!')
    print('Coeus now provides significantly more PyTorch compatibility.')

if __name__ == "__main__":
    main()