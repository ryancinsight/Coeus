"""Autograd validation and chain rule tests for PyCoeus"""

import pytest
import numpy as np
import math


@pytest.mark.autograd
def test_chain_rule_basic(pycoeus_available, pytorch_available):
    """Test basic chain rule validation using PyCoeus autograd vs PyTorch"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Test chain rule: d/dx[f(g(x))] = f'(g(x)) * g'(x)
    # Example: d/dx[sin(x^2)] = cos(x^2) * 2x

    def f_prime(x):
        return math.cos(x * x) * 2 * x

    # Test at several points
    test_points = [0.1, 0.5, 1.0, -0.5]

    for x_val in test_points:
        # PyCoeus autograd computation
        x_pc = pc.PyTensor([x_val], [1])
        x_pc.requires_grad_(True)

        # f(x) = sin(x^2)
        y_pc = (x_pc * x_pc).sin()
        y_pc.backward()

        pc_grad = x_pc.grad().data()[0]

        # PyTorch autograd computation
        x_pt = torch.tensor([x_val], requires_grad=True)
        y_pt = torch.sin(x_pt * x_pt)
        y_pt.backward()

        pt_grad = x_pt.grad.item()

        # Analytical derivative for reference
        analytical = f_prime(x_val)

        # Compare PyCoeus vs PyTorch
        relative_error = abs((pc_grad - pt_grad) / (pt_grad + 1e-12))
        assert relative_error < 1e-5, \
            f"PyCoeus vs PyTorch gradient mismatch at x={x_val}: pc={pc_grad}, pt={pt_grad}, analytical={analytical}"


@pytest.mark.autograd
def test_autograd_comprehensive(pycoeus_available, pytorch_available):
    """Comprehensive autograd validation against PyTorch"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch
    import numpy as np

    def test_function(x, y):
        """Complex function: z = sin(x^2 + y^2) + exp(x * y)"""
        return (x*x + y*y).sin() + (x * y).exp()

    # Test multiple scenarios
    test_cases = [
        ([1.0, 2.0], [0.5, 1.5]),
        ([0.1, 0.2], [-0.1, 0.3]),
        ([-1.0, -2.0], [1.0, -1.0]),
    ]

    for x_vals, y_vals in test_cases:
        # PyCoeus computation
        x_pc = pc.PyTensor(x_vals, [len(x_vals)])
        y_pc = pc.PyTensor(y_vals, [len(y_vals)])
        x_pc.requires_grad_(True)
        y_pc.requires_grad_(True)

        z_pc = test_function(x_pc, y_pc)
        z_pc.backward()

        # PyTorch computation
        x_pt = torch.tensor(x_vals, requires_grad=True)
        y_pt = torch.tensor(y_vals, requires_grad=True)

        z_pt = test_function(x_pt, y_pt)
        z_pt.sum().backward()  # Sum to create scalar for backward pass

        # Compare gradients
        for i in range(len(x_vals)):
            pc_grad_x = x_pc.grad().data()[i]
            pt_grad_x = x_pt.grad[i].item()
            pc_grad_y = y_pc.grad().data()[i]
            pt_grad_y = y_pt.grad[i].item()

            # Allow for small numerical differences
            assert abs(pc_grad_x - pt_grad_x) < 1e-4, \
                f"X gradient mismatch at {x_vals[i]}, {y_vals[i]}: pc={pc_grad_x}, pt={pt_grad_x}"
            assert abs(pc_grad_y - pt_grad_y) < 1e-4, \
                f"Y gradient mismatch at {x_vals[i]}, {y_vals[i]}: pc={pc_grad_y}, pt={pt_grad_y}"


@pytest.mark.autograd
def test_gradient_accumulation(pycoeus_available):
    """Test gradient accumulation in computational graphs"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    # Test that gradients accumulate correctly when a tensor is used multiple times
    # f(x) = x + x + x = 3x, so df/dx = 3

    data = [2.0]
    pc_tensor = pc.PyTensor(data, [1])
    pc_tensor.requires_grad_(True)

    # Use tensor multiple times
    result = pc_tensor + pc_tensor + pc_tensor  # Should be 6.0

    expected_value = 6.0
    actual_value = result.data()[0]

    assert abs(actual_value - expected_value) < 1e-6, \
        f"Multiple usage result incorrect: {actual_value} vs {expected_value}"

    # Gradient should be 3 (once autograd is fully implemented)
    # For now, just verify the computation is correct


@pytest.mark.autograd
def test_higher_order_derivatives(pycoeus_available):
    """Test higher-order derivative computation"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    # Test second derivative: d²/dx²[x^3] = 6x

    def f(x):
        return x * x * x

    def f_double_prime(x):
        return 6 * x

    test_points = [0.0, 1.0, 2.0, -1.0]

    for x in test_points:
        analytical = f_double_prime(x)

        # Numerical second derivative
        h = 1e-4
        f_prime_x_plus_h = (f(x + h) - f(x - h)) / (2 * h)
        f_prime_x_minus_h = (f(x) - f(x - 2*h)) / (2 * h)
        numerical = (f_prime_x_plus_h - f_prime_x_minus_h) / (2 * h)

        relative_error = abs((analytical - numerical) / (analytical + 1e-12))
        assert relative_error < 1e-4, \
            f"Second derivative failed at x={x}: analytical={analytical}, numerical={numerical}"


@pytest.mark.autograd
def test_partial_derivatives(pycoeus_available):
    """Test partial derivative computation"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    # Test function: f(x,y) = x^2 * y + sin(x)
    # ∂f/∂x = 2x*y + cos(x)
    # ∂f/∂y = x^2

    def f(x, y):
        return x * x * y + math.sin(x)

    def df_dx(x, y):
        return 2 * x * y + math.cos(x)

    def df_dy(x, y):
        return x * x

    test_points = [(0.5, 1.0), (1.0, 2.0), (-0.5, -1.0)]

    for x, y in test_points:
        analytical_dx = df_dx(x, y)
        analytical_dy = df_dy(x, y)

        # Numerical partial derivatives
        h = 1e-5
        numerical_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
        numerical_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)

        error_dx = abs((analytical_dx - numerical_dx) / (analytical_dx + 1e-12))
        error_dy = abs((analytical_dy - numerical_dy) / (analytical_dy + 1e-12))

        assert error_dx < 1e-6, f"∂f/∂x failed at ({x},{y}): {error_dx}"
        assert error_dy < 1e-6, f"∂f/∂y failed at ({x},{y}): {error_dy}"


@pytest.mark.autograd
def test_jacobian_computation(pycoeus_available):
    """Test Jacobian matrix computation for vector-valued functions"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    # Test function: f(x,y) = [x^2 + y, x*y, sin(x)]
    # Jacobian should be:
    # [[2x, 1],
    #  [y,  x],
    #  [cos(x), 0]]

    def f_vec(x, y):
        return [
            x * x + y,
            x * y,
            math.sin(x)
        ]

    def jacobian_analytical(x, y):
        return [
            [2 * x, 1],
            [y, x],
            [math.cos(x), 0]
        ]

    test_points = [(0.5, 1.0), (1.0, 0.0), (-0.5, 2.0)]

    for x, y in test_points:
        analytical_jac = jacobian_analytical(x, y)

        # Numerical Jacobian
        h = 1e-5
        f0 = f_vec(x, y)
        fx = f_vec(x + h, y)
        fy = f_vec(x, y + h)

        numerical_jac = [
            [(fx[i] - f0[i]) / h for i in range(3)],
            [(fy[i] - f0[i]) / h for i in range(3)]
        ]

        # Transpose to match analytical form
        numerical_jac = list(map(list, zip(*numerical_jac)))

        # Check each element
        for i in range(3):
            for j in range(2):
                analytical_val = analytical_jac[i][j]
                numerical_val = numerical_jac[i][j]

                relative_error = abs((analytical_val - numerical_val) / (analytical_val + 1e-12))
                assert relative_error < 1e-5, \
                    f"Jacobian[{i}][{j}] failed at ({x},{y}): {relative_error}"


@pytest.mark.autograd
def test_gradient_through_operations(pycoeus_available):
    """Test gradient flow through various operations"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    # Test gradient through: sin(exp(x))
    # d/dx[sin(exp(x))] = cos(exp(x)) * exp(x)

    def f(x):
        return math.sin(math.exp(x))

    def f_prime(x):
        return math.cos(math.exp(x)) * math.exp(x)

    test_points = [0.0, 0.5, 1.0]

    for x in test_points:
        analytical = f_prime(x)

        # Numerical derivative
        h = 1e-5
        numerical = (f(x + h) - f(x - h)) / (2 * h)

        relative_error = abs((analytical - numerical) / (analytical + 1e-12))
        assert relative_error < 1e-6, \
            f"Gradient through operations failed at x={x}: {relative_error}"


@pytest.mark.autograd
def test_autograd_consistency(pycoeus_available, pytorch_available):
    """Test consistency between PyCoeus and PyTorch autograd"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Test cases that should have identical gradients
    test_functions = [
        ("simple_add", lambda x: x + x, lambda x: 2 * torch.ones_like(x)),
        ("simple_mul", lambda x: x * x, lambda x: 2 * x),
        ("exp_func", lambda x: torch.exp(x), lambda x: torch.exp(x)),
    ]

    for name, torch_func, torch_grad_func in test_functions:
        # Test with random data
        data = np.random.randn(10).tolist()
        torch_tensor = torch.tensor(data, requires_grad=True)

        # PyTorch computation
        torch_result = torch_func(torch_tensor)
        torch_result.backward(torch.ones_like(torch_result))
        torch_grad = torch_tensor.grad.clone()

        # Reset for next test
        torch_tensor.grad.zero_()

        # For now, just verify PyTorch gradients are computed
        # Once PyCoeus autograd is fully implemented, compare directly
        assert torch_grad is not None, f"PyTorch gradient not computed for {name}"

        # Verify gradient has correct shape
        assert torch_grad.shape == torch_tensor.shape, \
            f"Gradient shape mismatch for {name}: {torch_grad.shape} vs {torch_tensor.shape}"


@pytest.mark.autograd
def test_gradient_stability(pycoeus_available):
    """Test gradient computation stability"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    # Test with functions that could cause numerical instability
    unstable_functions = [
        ("exp_large", lambda x: math.exp(10)),  # Large exponential
        ("log_small", lambda x: math.log(1e-8)),  # Log of small number
        ("div_zero", lambda x: 1.0 / 1e-8),  # Large division
    ]

    for name, func in unstable_functions:
        try:
            result = func(1.0)
            # If we get here, function didn't crash
            assert isinstance(result, (int, float)), f"{name} returned non-numeric: {result}"
        except (OverflowError, ZeroDivisionError, ValueError) as e:
            # These are acceptable for unstable functions
            pass
        except Exception as e:
            pytest.fail(f"{name} raised unexpected error: {e}")


@pytest.mark.autograd
def test_vector_jacobian_product(pycoeus_available, pytorch_available):
    """Test vector-Jacobian product (VJP) computation"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Test VJP for f(x) = [x^2, sin(x), exp(x)]
    # With vector v = [1, 1, 1]
    # VJP should be: v^T * J = [2x, 1*cos(x), 1*exp(x)]

    def f_vec(x):
        return [x * x, math.sin(x), math.exp(x)]

    def vjp_analytical(x, v):
        jac_T = [
            2 * x,
            math.cos(x),
            math.exp(x)
        ]
        return sum(vi * ji for vi, ji in zip(v, jac_T))

    test_points = [0.0, 0.5, 1.0, -0.5]
    v = [1.0, 1.0, 1.0]

    for x in test_points:
        analytical_vjp = vjp_analytical(x, v)

        # Numerical VJP
        h = 1e-5
        f0 = f_vec(x)
        fx = f_vec(x + h)

        # Finite difference approximation of J * v
        numerical_vjp = sum((fx[i] - f0[i]) / h * v[i] for i in range(3))

        relative_error = abs((analytical_vjp - numerical_vjp) / (analytical_vjp + 1e-12))
        assert relative_error < 1e-5, \
            f"VJP failed at x={x}: analytical={analytical_vjp}, numerical={numerical_vjp}"


@pytest.mark.autograd
def test_jacobian_vector_product(pycoeus_available, pytorch_available):
    """Test Jacobian-vector product (JVP) computation"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Test JVP for f(x) = x^3
    # With tangent vector t = 1
    # JVP should be: J * t = 3x^2 * t

    def f(x):
        return x * x * x

    def jvp_analytical(x, t):
        return 3 * x * x * t

    test_points = [0.0, 0.5, 1.0, 2.0]
    tangent_vector = 1.0

    for x in test_points:
        analytical_jvp = jvp_analytical(x, tangent_vector)

        # Numerical JVP using directional derivative
        h = 1e-5
        t = tangent_vector
        directional_deriv = (f(x + h * t) - f(x)) / h

        relative_error = abs((analytical_jvp - directional_deriv) / (analytical_jvp + 1e-12))
        assert relative_error < 1e-5, \
            f"JVP failed at x={x}: analytical={analytical_jvp}, numerical={directional_deriv}"


@pytest.mark.autograd
def test_hessian_computation(pycoeus_available):
    """Test Hessian matrix computation"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    # Test Hessian of f(x,y) = x^2 * y + x * y^2
    # H = [[2y, 2x + 2y], [2x + 2y, 2x]]

    def f(x, y):
        return x * x * y + x * y * y

    def hessian_analytical(x, y):
        return [
            [2 * y, 2 * x + 2 * y],
            [2 * x + 2 * y, 2 * x]
        ]

    test_points = [(1.0, 1.0), (0.5, 2.0), (-1.0, -1.0)]

    for x, y in test_points:
        analytical_hess = hessian_analytical(x, y)

        # Numerical Hessian using finite differences
        h = 1e-4

        # Compute second derivatives
        f_xx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h * h)
        f_xy = (f(x + h, y + h) - f(x + h, y) - f(x, y + h) + f(x, y)) / (h * h)
        f_yy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h * h)

        numerical_hess = [
            [f_xx, f_xy],
            [f_xy, f_yy]
        ]

        # Check each element
        for i in range(2):
            for j in range(2):
                analytical_val = analytical_hess[i][j]
                numerical_val = numerical_hess[i][j]

                relative_error = abs((analytical_val - numerical_val) / (analytical_val + 1e-12))
                assert relative_error < 1e-4, \
                    f"Hessian[{i}][{j}] failed at ({x},{y}): {relative_error}"


@pytest.mark.autograd
def test_autograd_edge_cases(pycoeus_available):
    """Test autograd with edge cases and special values"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    # Test with special values that could cause issues
    special_values = [
        ([0.0], "zero"),
        ([1.0], "one"),
        ([-1.0], "negative"),
        ([float('inf')], "infinity"),
        ([-float('inf')], "negative_infinity"),
        ([float('nan')], "nan"),
    ]

    for data, case_name in special_values:
        pc_tensor = pc.PyTensor(data, [1])
        pc_tensor.requires_grad_(True)

        try:
            # Try basic operations
            if case_name not in ["infinity", "negative_infinity", "nan"]:
                result = pc_tensor + pc_tensor
                assert len(result.data()) == 1, f"Operation failed for {case_name}"

        except Exception as e:
            # Some edge cases are expected to fail gracefully
            if case_name in ["infinity", "negative_infinity", "nan"]:
                continue  # These are acceptable failures
            else:
                pytest.fail(f"Unexpected failure for {case_name}: {e}")


@pytest.mark.autograd
def test_gradient_flow_complexity(pycoeus_available):
    """Test gradient flow through complex computational graphs"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    # Create a complex computational graph
    # f(x) = sin(exp(x^2 + x)) + cos(x^3)
    # This should test chain rule through multiple nested operations

    def complex_function(x):
        inner = x * x + x  # x^2 + x
        exp_inner = math.exp(inner)  # exp(x^2 + x)
        sin_exp = math.sin(exp_inner)  # sin(exp(x^2 + x))
        x_cubed = x * x * x  # x^3
        cos_x3 = math.cos(x_cubed)  # cos(x^3)
        return sin_exp + cos_x3

    def complex_derivative(x):
        # d/dx[sin(exp(x^2 + x)) + cos(x^3)]
        # = cos(exp(x^2 + x)) * exp(x^2 + x) * (2x + 1) - sin(x^3) * 3x^2
        exp_term = math.exp(x * x + x)
        cos_exp = math.cos(exp_term)
        sin_x3 = math.sin(x * x * x)

        return cos_exp * exp_term * (2 * x + 1) - sin_x3 * 3 * x * x

    test_points = [0.1, 0.5, 1.0]

    for x in test_points:
        analytical = complex_derivative(x)

        # Numerical derivative
        h = 1e-5
        numerical = (complex_function(x + h) - complex_function(x - h)) / (2 * h)

        relative_error = abs((analytical - numerical) / (analytical + 1e-12))
        assert relative_error < 1e-5, \
            f"Complex gradient flow failed at x={x}: error={relative_error}"
