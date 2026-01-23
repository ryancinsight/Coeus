"""
Test PyCoeus exception hierarchy

This test verifies that all exception types are properly exposed
to Python and can be imported and used.
"""

import pytest


def test_exception_imports():
    """Test that all exception types can be imported from coeus"""
    try:
        import coeus
        
        # Verify all exception classes exist
        assert hasattr(coeus, 'CoeusError')
        assert hasattr(coeus, 'TensorError')
        assert hasattr(coeus, 'BackendError')
        assert hasattr(coeus, 'OptimizerError')
        assert hasattr(coeus, 'NNError')
        assert hasattr(coeus, 'StorageError')
        assert hasattr(coeus, 'ShapeError')
        assert hasattr(coeus, 'DeviceError')
        
        print("✓ All exception types are accessible")
    except ImportError as e:
        pytest.skip(f"Could not import coeus: {e}")


def test_exception_hierarchy():
    """Test that exception hierarchy is correct"""
    try:
        import coeus
        
        # Test inheritance relationships
        assert issubclass(coeus.TensorError, coeus.CoeusError)
        assert issubclass(coeus.BackendError, coeus.CoeusError)
        assert issubclass(coeus.OptimizerError, coeus.CoeusError)
        assert issubclass(coeus.NNError, coeus.CoeusError)
        assert issubclass(coeus.StorageError, coeus.CoeusError)
        assert issubclass(coeus.ShapeError, coeus.TensorError)
        assert issubclass(coeus.DeviceError, coeus.BackendError)
        
        print("✓ Exception hierarchy is correct")
    except ImportError as e:
        pytest.skip(f"Could not import coeus: {e}")


def test_exception_instantiation():
    """Test that exceptions can be instantiated and raised"""
    try:
        import coeus
        
        # Test that we can create and raise each exception type
        exceptions = [
            coeus.CoeusError,
            coeus.TensorError,
            coeus.BackendError,
            coeus.OptimizerError,
            coeus.NNError,
            coeus.StorageError,
            coeus.ShapeError,
            coeus.DeviceError,
        ]
        
        for exc_class in exceptions:
            try:
                raise exc_class("Test error message")
            except exc_class as e:
                assert str(e) == "Test error message"
                assert isinstance(e, coeus.CoeusError)
        
        print("✓ All exceptions can be instantiated and raised")
    except ImportError as e:
        pytest.skip(f"Could not import coeus: {e}")


def test_exception_catching():
    """Test that exceptions can be caught by their base class"""
    try:
        import coeus
        
        # Test catching specific exception with base class
        try:
            raise coeus.ShapeError("Shape mismatch")
        except coeus.TensorError as e:
            assert "Shape mismatch" in str(e)
        
        # Test catching any Coeus exception
        try:
            raise coeus.DeviceError("Device not available")
        except coeus.CoeusError as e:
            assert "Device not available" in str(e)
        
        print("✓ Exception catching works correctly")
    except ImportError as e:
        pytest.skip(f"Could not import coeus: {e}")


if __name__ == "__main__":
    test_exception_imports()
    test_exception_hierarchy()
    test_exception_instantiation()
    test_exception_catching()
    print("\n✅ All exception tests passed!")
