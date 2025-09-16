"""
Parameter Optimization for Iterative Methods.

This module provides functions to calculate optimal parameters
for various iterative methods.

Author: Tomer Caspi with the assistance of Cursor.ai.
"""

from typing import Optional
import numpy as np
from numpy import linalg as LA


def calculate_optimal_richardson_parameter(matrix: np.ndarray) -> float:
    """
    Calculate the optimal relaxation parameter for Richardson's method.

    For Richardson's method x^(k+1) = x^(k) + ω(b - Ax^(k)),
    the optimal parameter is ω = 2/(λ_min + λ_max) where λ_min and λ_max
    are the minimum and maximum eigenvalues of matrix A.

    Parameters
    ----------
    matrix : np.ndarray
        The coefficient matrix A

    Returns
    -------
    float
        Optimal relaxation parameter ω
    """
    try:
        eigenvalues = LA.eigvals(matrix)
        lambda_min = np.min(eigenvalues.real)
        lambda_max = np.max(eigenvalues.real)

        if lambda_min + lambda_max == 0:
            raise ValueError("Sum of eigenvalues is zero"
                             " - cannot compute optimal parameter")
        # Optimal parameter: ω = 2/(λ_min + λ_max)
        optimal_omega = 2.0 / (lambda_min + lambda_max)

        print(
            f"Matrix eigenvalues: λ_min = {lambda_min:.4f}, "
            f"λ_max = {lambda_max:.4f}")
        print(f"Optimal Richardson parameter: ω = {optimal_omega:.4f}")

        return optimal_omega

    except LA.LinAlgError as e:
        print("Linear algebra error calculating"
              f" optimal Richardson parameter: {e}")
        return 0.1
    except ValueError as e:
        print(f"Value error calculating optimal Richardson parameter: {e}")
        return 0.1
    except TypeError as e:
        print(f"Type error calculating optimal Richardson parameter: {e}")
    return 0.1


def calculate_optimal_sor_parameter(matrix: np.ndarray) -> Optional[float]:
    """
    Calculate the optimal relaxation parameter for SOR method.

    For SOR, the optimal parameter is approximately:
    ω_opt ≈ 2 / (1 + sqrt(1 - ρ(B)^2))
    where ρ(B) is the spectral radius of the Jacobi iteration matrix.

    Parameters
    ----------
    matrix : np.ndarray
        The coefficient matrix A

    Returns
    -------
    float
        Optimal relaxation parameter ω
    """
    try:
        # Calculate Jacobi iteration matrix: B = D^(-1) * (L + U)
        D = np.diag(np.diag(matrix))  # Diagonal part
        L_plus_U = matrix - D  # Lower + Upper triangular parts

        # Jacobi iteration matrix
        B = np.linalg.inv(D) @ (-L_plus_U)

        # Calculate spectral radius of Jacobi matrix
        eigenvalues = np.linalg.eigvals(B)
        rho_B = np.max(np.abs(eigenvalues))

        # Optimal SOR parameter
        omega_opt = 2.0 / (1.0 + np.sqrt(1.0 - rho_B**2))

        print(f"Jacobi spectral radius: ρ(B) = {rho_B:.4f}")
        print(f"Optimal SOR parameter: ω_opt = {omega_opt:.4f}")

        return omega_opt

    except Exception as e:
        print(f"Could not calculate optimal SOR parameter: {e}")
        return 1.2  # Default fallback value
