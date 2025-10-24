#!/usr/bin/env python3
"""
Simple test script for PyCoeus basic functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pycoeus', 'python'))

try:
    import coeus
    print("✅ PyCoeus imported successfully")
except ImportError as e:
    print(f"❌ Failed to import PyCoeus: {e}")
    print("Note: Make sure to build the extension with 'cd pycoeus && pip install -e .'")
    sys.exit(1)

try:
    import numpy as np
    print("✅ NumPy imported successfully")

    # Test basic tensor creation
    print("Testing tensor creation...")
    tensor = coeus.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
    print(f"Created tensor with shape: {tensor.shape()}")

    # Test NumPy conversion
    print("Testing NumPy conversion...")
    numpy_array = np.array(tensor)
    print(f"Converted to NumPy array: {numpy_array}")
    print(f"Array shape: {numpy_array.shape}")
    print(f"Array dtype: {numpy_array.dtype}")

    # Test that the data is correct
    expected = np.array([[1.0, 2.0], [3.0, 4.0]])
    if np.allclose(numpy_array, expected):
        print("✅ NumPy conversion data matches expected values")
    else:
        print(f"❌ Data mismatch. Expected: {expected}, Got: {numpy_array}")
        sys.exit(1)

    # Test tensor operations
    print("Testing tensor operations...")
    tensor2 = coeus.Tensor([5.0, 6.0, 7.0, 8.0], [2, 2])
    result = tensor + tensor2
    numpy_result = np.array(result)
    expected_result = np.array([[6.0, 8.0], [10.0, 12.0]])

    if np.allclose(numpy_result, expected_result):
        print("✅ Tensor addition works correctly")
    else:
        print(f"❌ Addition failed. Expected: {expected_result}, Got: {numpy_result}")
        sys.exit(1)

    print("✅ All PyCoeus tests passed!")

except ImportError as e:
    print(f"❌ NumPy import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test failed with exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
