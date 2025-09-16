"""
Iterative Methods for Solving Linear Systems

This module provides implementations of Jacobi, Gauss-Seidel, and SOR methods
for solving the linear system Ax = b, with comprehensive documentation and
eliminated code repetition.

Author: Tomer Caspi with the assistance of Cursor.ai
Based on "Numerical Analysis" by Burden, 10th Edition
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy import linalg as LA
from parameter_optimization import (calculate_optimal_sor_parameter,
                                    calculate_optimal_richardson_parameter)


np.set_printoptions(suppress=True)


class MatrixDecomposition:
    """
    Utility class for common matrix decompositions used in iterative methods.
    """

    @staticmethod
    def get_diagonal(matrix: np.ndarray) -> np.ndarray:
        """Extract diagonal matrix D from A."""
        return np.diag(np.diag(matrix))

    @staticmethod
    def get_off_diagonal(matrix: np.ndarray) -> np.ndarray:
        """Extract off-diagonal matrix (L + U) from A."""
        return matrix - MatrixDecomposition.get_diagonal(matrix)

    @staticmethod
    def get_lower_triangular(matrix: np.ndarray,
                             include_diagonal: bool = True) -> np.ndarray:
        """
        Extract lower triangular matrix L from A.

        Parameters
        ----------
        matrix : np.ndarray
            Input matrix A
        include_diagonal : bool, default=False
            If True, includes diagonal (D + L). If False, only L.
        """
        if include_diagonal:
            return np.tril(matrix)  # D + L
        return np.tril(matrix, k=-1)  # L only

    @staticmethod
    def get_upper_triangular(matrix: np.ndarray,
                             include_diagonal: bool = False) -> np.ndarray:
        """Extract upper triangular matrix U from A.

        Parameters
        ----------
        matrix : np.ndarray
            Input matrix A
        include_diagonal : bool, default=False
            If True, includes diagonal (D + U). If False, only U.
        """
        if include_diagonal:
            return np.triu(matrix)  # D + U
        return np.triu(matrix, k=1)  # U only


@dataclass
class SolverConfig:
    """Configuration parameters for iterative solvers."""
    tolerance: float = 1e-5
    max_iterations: int = 100
    verbose: bool = True


@dataclass
class LinearSystem:
    """Represents the linear system Ax = b."""
    matrix: np.ndarray
    constant_vector: np.ndarray
    initial_vector: np.ndarray

    def __post_init__(self):
        """Validate the linear system dimensions"""
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Matrix must be square")
        if len(self.constant_vector) != self.matrix.shape[0]:
            raise ValueError("Constant vector dimensions mismatch")
        if len(self.initial_vector) != self.matrix.shape[0]:
            raise ValueError("Initial vector dimensions mismatch")


class IterativeMethod(ABC):
    """Abstract base class for iterative methods"""

    def __init__(self, config: SolverConfig):
        self.config = config

    @abstractmethod
    def compute_iteration(self, system: LinearSystem,
                          current_vector: np.ndarray) -> np.ndarray:
        """Compute one iteration of the method."""

    @abstractmethod
    def check_convergence(self, system: LinearSystem) -> bool:
        """Check if the method will converge for the given system."""

    @abstractmethod
    def get_method_name(self) -> str:
        """Return the name of the method."""

    def get_spectral_radius(self, system: LinearSystem) -> Optional[float]:
        """Get spectral radius of iteration matrix"""
        try:
            iteration_matrix = self.get_iteration_matrix(system)
            if iteration_matrix is None:
                return None

            eigenvalues = LA.eigvals(iteration_matrix)
            return np.max(np.abs(eigenvalues))

        except LA.LinAlgError:
            return None

    @abstractmethod
    def get_iteration_matrix(self,
                             system: LinearSystem) -> Optional[np.ndarray]:
        """ Get the iteration matrix for this method.
        To be implemented by subclasses."""

    def _check_spectral_radius_convergence(self, iteration_matrix: np.ndarray,
                                           method_name: str) -> bool:
        """Common spectral radius convergence check for all methods."""
        try:
            eigenvalues = LA.eigvals(iteration_matrix)
            spectral_radius = np.max(np.abs(eigenvalues))

            if spectral_radius >= 1:
                if self.config.verbose:
                    print(f"{method_name} does not converge: "
                          f"ρ(T) = {spectral_radius}")
                return False

            if self.config.verbose:
                print(f"{method_name} will converge: ρ(T) = {spectral_radius}")
            return True

        except LA.LinAlgError:
            if self.config.verbose:
                print("Cannot compute spectral radius"
                      "- matrix may be singular")
            return False

    def solve(self, system: LinearSystem) -> Optional[np.ndarray]:
        """Solve the linear system using this iterative method"""
        if not self.check_convergence(system):
            return None

        current_vector = system.initial_vector.copy()

        for iteration in range(1, self.config.max_iterations):
            new_vector = self.compute_iteration(system, current_vector)

            if LA.norm(new_vector - current_vector) < self.config.tolerance:
                if self.config.verbose:
                    print(f"{self.get_method_name()}: converged after "
                          f"{iteration} iterations: {new_vector}")
                return new_vector

            if self.config.verbose:
                print(f"Iteration {iteration}: {new_vector}")

            current_vector = new_vector.copy()
        if self.config.verbose:
            print(f"{self.get_method_name()}: no convergence after "
                  f"{self.config.max_iterations} iterations")
        return None


class JacobiMethod(IterativeMethod):
    """Jacobi iterative method implementation"""

    def compute_iteration(self, system: LinearSystem,
                          current_vector: np.ndarray) -> np.ndarray:
        """Compute one Jacobi iteration"""
        new_vector = np.zeros_like(current_vector)
        matrix = system.matrix
        constant_vector = system.constant_vector

        for i, _ in enumerate(current_vector):
            mat_row = matrix[i, :]
            # Sum of off-diagonal terms using previous iteration values
            off_diag_sum = (mat_row[:i] @ current_vector[:i]
                            + mat_row[i+1:] @ current_vector[i+1:])
            new_vector[i] = (constant_vector[i] - off_diag_sum) / matrix[i, i]
        return new_vector

    def check_convergence(self, system: LinearSystem) -> bool:
        """Check Jacobi convergence using spectral radius"""
        matrix = system.matrix
        diag_matrix = MatrixDecomposition.get_diagonal(matrix)
        off_diag_matrix = MatrixDecomposition.get_off_diagonal(matrix)
        iteration_matrix = LA.inv(diag_matrix) @ (-off_diag_matrix)

        return self._check_spectral_radius_convergence(iteration_matrix,
                                                       "Jacobi")

    def get_method_name(self) -> str:
        return "Jacobi"

    def get_iteration_matrix(self,
                             system: LinearSystem) -> Optional[np.ndarray]:
        matrix = system.matrix
        diag_matrix = MatrixDecomposition.get_diagonal(matrix)
        off_diag_matrix = MatrixDecomposition.get_off_diagonal(matrix)
        iteration_matrix = LA.inv(diag_matrix) @ (-off_diag_matrix)
        return iteration_matrix


class DampedJacobiMethod(IterativeMethod):
    """Damped Jacobi iterative method implementation"""

    def __init__(self, relaxation_param: float, config: SolverConfig):
        super().__init__(config)
        if not 0 < relaxation_param:
            raise ValueError("Relaxation parameter must be greater than 0")
        self.relaxation_param = relaxation_param

    def compute_iteration(self, system: LinearSystem,
                          current_vector: np.ndarray) -> np.ndarray:
        """Compute one Damped Jacobi iteration"""
        new_vector = np.zeros_like(current_vector)
        matrix = system.matrix
        constant_vector = system.constant_vector
        omega = self.relaxation_param
        for i, _ in enumerate(current_vector):
            mat_row = matrix[i, :]
            # Sum of off-diagonal terms using previous iteration values
            off_diag_sum = (mat_row[:i] @ current_vector[:i]
                            + mat_row[i+1:] @ current_vector[i+1:])
            jacobi_value = (constant_vector[i] - off_diag_sum) / matrix[i, i]
            new_vector[i] = ((1-omega) * current_vector[i]
                             + omega * jacobi_value)
        return new_vector

    def check_convergence(self, system: LinearSystem) -> bool:
        """Check Damped Jacobi convergence using spectral radius"""
        matrix = system.matrix
        matrix_order = matrix.shape[0]
        diag_matrix = MatrixDecomposition.get_diagonal(matrix)
        off_diag_matrix = MatrixDecomposition.get_off_diagonal(matrix)
        omega = self.relaxation_param

        iteration_matrix = ((1 - omega) * np.eye(matrix_order)
                            - omega * LA.inv(diag_matrix) @ off_diag_matrix)

        return self._check_spectral_radius_convergence(iteration_matrix,
                                                       "Damped Jacobi")

    def get_method_name(self) -> str:
        return f"Damped Jacobi (ω={self.relaxation_param:.3f})"

    def get_iteration_matrix(self,
                             system: LinearSystem) -> Optional[np.ndarray]:
        matrix = system.matrix
        matrix_order = matrix.shape[0]
        diag_matrix = MatrixDecomposition.get_diagonal(matrix)
        off_diag_matrix = MatrixDecomposition.get_off_diagonal(matrix)
        omega = self.relaxation_param

        iteration_matrix = ((1 - omega) * np.eye(matrix_order)
                            - omega * LA.inv(diag_matrix) @ off_diag_matrix)
        return iteration_matrix


class GaussSeidelMethod(IterativeMethod):
    """Gauss-Seidel iterative method implementation"""

    def compute_iteration(self, system: LinearSystem,
                          current_vector: np.ndarray) -> np.ndarray:
        """Compute one Gauss-Seidel iteration"""
        new_vector = np.zeros_like(current_vector)
        matrix = system.matrix
        constant_vector = system.constant_vector

        for i, _ in enumerate(current_vector):
            mat_row = matrix[i, :]
            # Sum of off-diagonal terms using previous iteration values
            off_diag_sum = (mat_row[:i] @ new_vector[:i]
                            + mat_row[i+1:] @ current_vector[i+1:])
            new_vector[i] = (constant_vector[i] - off_diag_sum) / matrix[i, i]
        return new_vector

    def check_convergence(self, system: LinearSystem) -> bool:
        """Check Gauss-Seidel convergence using spectral radius"""
        matrix = system.matrix
        lower_trian_mat = MatrixDecomposition.get_lower_triangular(
            matrix,
            include_diagonal=True)
        upper_trian_mat = MatrixDecomposition.get_upper_triangular(
            matrix, include_diagonal=False)
        iteration_matrix = LA.inv(lower_trian_mat) @ upper_trian_mat

        return self._check_spectral_radius_convergence(iteration_matrix,
                                                       "Gauss-Seidel")

    def get_method_name(self) -> str:
        return "Gauss-Seidel"

    def get_iteration_matrix(self,
                             system: LinearSystem) -> Optional[np.ndarray]:
        matrix = system.matrix
        lower_trian_mat = MatrixDecomposition.get_lower_triangular(
            matrix,
            include_diagonal=True)
        upper_trian_mat = MatrixDecomposition.get_upper_triangular(
            matrix, include_diagonal=False)
        iteration_matrix = LA.inv(lower_trian_mat) @ upper_trian_mat

        return iteration_matrix


class RichardsonMethod(IterativeMethod):
    """Richardson iterative method implementation"""

    def __init__(self, relaxation_param: float, config: SolverConfig):
        super().__init__(config)
        if not 0 < relaxation_param:
            raise ValueError("Relaxation parameter must be greater than 0")
        self.relaxation_param = relaxation_param

    def compute_iteration(self, system: LinearSystem,
                          current_vector: np.ndarray) -> np.ndarray:
        """Compute one Richardson iteration."""
        new_vector = current_vector.copy()
        matrix = system.matrix
        constant_vector = system.constant_vector
        omega = self.relaxation_param

        new_vector = (omega * (constant_vector - matrix @ current_vector)
                      + current_vector)

        return new_vector

    def check_convergence(self, system: LinearSystem) -> bool:
        """Check Richardson convergence using spectral radius."""
        matrix = system.matrix
        matrix_order = matrix.shape[0]
        omega = self.relaxation_param
        iteration_matrix = np.eye(matrix_order) - omega * matrix

        return self._check_spectral_radius_convergence(
            iteration_matrix,
            f"Richardson (ω = {omega})")

    def get_method_name(self) -> str:
        return f"Richardson's method (ω={self.relaxation_param:.3f})"

    def get_iteration_matrix(self,
                             system: LinearSystem) -> Optional[np.ndarray]:
        matrix = system.matrix
        matrix_order = matrix.shape[0]
        omega = self.relaxation_param
        iteration_matrix = np.eye(matrix_order) - omega * matrix

        return iteration_matrix


class SORMethod(IterativeMethod):
    """Successive Over-Relaxation method implementation."""

    def __init__(self, relaxation_param: float, config: SolverConfig):
        super().__init__(config)
        if not 0 < relaxation_param < 2:
            raise ValueError("Relaxation parameter must be in (0,2)")
        self.relaxation_param = relaxation_param

    def compute_iteration(self, system: LinearSystem,
                          current_vector: np.ndarray) -> np.ndarray:
        """Compute one SOR iteration."""
        new_vector = current_vector.copy()
        matrix = system.matrix
        constant_vector = system.constant_vector
        omega = self.relaxation_param

        for i, _ in enumerate(current_vector):
            off_diagonal_sum = (matrix[i, :i] @ new_vector[:i] +
                                matrix[i, i+1:] @ current_vector[i+1:])
            gauss_seidel_value = (
                constant_vector[i] - off_diagonal_sum) / matrix[i, i]
            new_vector[i] = ((1 - omega) * current_vector[i] +
                             omega * gauss_seidel_value)

        return new_vector

    def check_convergence(self, system: LinearSystem) -> bool:
        """Check SOR convergence using spectral radius."""
        matrix = system.matrix
        omega = self.relaxation_param
        diag_matrix = MatrixDecomposition.get_diagonal(matrix)
        lower_triangular = MatrixDecomposition.get_lower_triangular(
            matrix, include_diagonal=False)
        upper_triangular = MatrixDecomposition.get_upper_triangular(
            matrix, include_diagonal=False)

        iteration_matrix = (
            np.linalg.inv(diag_matrix - omega * lower_triangular) @
            ((1 - omega) * diag_matrix + omega * upper_triangular)
        )

        return self._check_spectral_radius_convergence(iteration_matrix,
                                                       f"SOR (ω = {omega})")

    def get_method_name(self) -> str:
        return f"SOR (ω={self.relaxation_param:.3f})"

    def get_iteration_matrix(self,
                             system: LinearSystem) -> Optional[np.ndarray]:
        matrix = system.matrix
        omega = self.relaxation_param
        diag_matrix = MatrixDecomposition.get_diagonal(matrix)
        lower_triangular = MatrixDecomposition.get_lower_triangular(
            matrix, include_diagonal=False)
        upper_triangular = MatrixDecomposition.get_upper_triangular(
            matrix, include_diagonal=False)

        iteration_matrix = (
            np.linalg.inv(diag_matrix - omega * lower_triangular) @
            ((1 - omega) * diag_matrix + omega * upper_triangular)
        )
        return iteration_matrix


def jacobi(system: LinearSystem,
           config: SolverConfig) -> Optional[np.ndarray]:
    """
    Solve the linear system Ax = b using Jacobi's iterative method.

    Based on Algorithm 7.1 from "Numerical Analysis" by Burden, 10th Edition

    Parameters
    ----------
    system : LinearSystem
        Linear System Ax = b, with matrix, initial_vector, and constant_vector.
    config : SolverConfig
        Solver configuration with tolerance, max_iterations,
        and verbose settings

    Returns
    -------
    np.ndarray or None
        Approximate solution vector if convergence is achieved, None otherwise.
        The solution is printed to console upon convergence.

    Notes
    -----
    - The method requires that the diagonal elements of A are non-zero.
    - Convergence is guaranteed for strictly diagonally dominant matrices.
    - The method may not converge for arbitrary matrices.
    - Uses Euclidean for convergence checking.
    """
    solver = JacobiMethod(config)
    return solver.solve(system)


def damped_jacobi(system: LinearSystem, config: SolverConfig,
                  relaxation_param: float) -> Optional[np.ndarray]:
    """
    Solve the linear system Ax = b using Damped Jacobi method.

    Parameters
    ----------
    system : LinearSystem
        Linear System Ax = b, with matrix, initial_vector, and constant_vector.
    config : SolverConfig
        Solver configuration with tolerance, max_iterations,
        and verbose settings

    Returns
    -------
    np.ndarray or None
        Solution vector if converged, None otherwise
    """
    solver = DampedJacobiMethod(relaxation_param, config)
    return solver.solve(system)


def gauss_seidel(system: LinearSystem,
                 config: SolverConfig) -> Optional[np.ndarray]:
    """
    Solve the linear system Ax = b using Gauss-Seidel iterative method.

    Based on Algorithm 7.2 from "Numerical Analysis" by Burden, 10th Edition
    Parameters
    ----------
    system : LinearSystem
        Linear System Ax = b, with matrix, initial_vector, and constant_vector.
    config : SolverConfig
        Solver configuration with tolerance, max_iterations,
        and verbose settings

    Returns
    -------
    np.ndarray or None
        Approximate solution vector if convergence is achieved, None otherwise.
        The solution is printed to console upon convergence.

    Notes
    -----
    - The method requires that the diagonal elements of A are non-zero.
    - Convergence is guaranteed for strictly diagonally dominant matrices.
    - Uses Euclidean norm for convergence checking.
    """
    solver = GaussSeidelMethod(config)
    return solver.solve(system)


def richardson(system: LinearSystem, config: SolverConfig,
               relaxation_param: float = None) -> Optional[np.ndarray]:
    """
    Solve Ax = b using Richardson's method.

    Richardson's method uses the iteration formula:
    x^(k+1) = x^(k) + α(b - Ax^(k))
    where α is the relaxation parameter.

    Parameters
    ----------
    system : LinearSystem
        Linear System defined by a matrix, initial vector, and constant vector.
    config : SolverConfig
        Solver configuration
    relaxation_param : float, optional
        Relaxation parameter.
        If none, return optimal relaxation parameter.

    Returns
    -------
    np.ndarray or None
        Solution vector if converged, None otherwise
    """
    matrix = system.matrix
    if relaxation_param is None:
        relaxation_param = calculate_optimal_richardson_parameter(matrix)
    solver = RichardsonMethod(relaxation_param, config)
    return solver.solve(system)


def sor(system: LinearSystem, config: SolverConfig,
        relaxation_param: float = None) -> Optional[np.ndarray]:
    """
    Solve the linear system Ax = b
    using Successive Over-Relaxation (SOR) method.

    SOR method is a generalization of Gauss-Seidel method
    with a relaxation parameter ω.

    Based on Algorithm 7.3 from "Numerical Analysis" by Burden, 10th Edition

    Parameters
    ----------
    system : LinearSystem
        Linear System Ax = b, with matrix, initial_vector, and constant_vector.
    config : SolverConfig
        Solver configuration with tolerance, max_iterations,
        and verbose settings

    Returns
    -------
    np.ndarray or None
        Approximate solution vector if convergence is achieved, None otherwise.
        The solution is printed to console upon convergence.

    Notes
    -----
    - The method requires that the diagonal elements of A are non-zero.
    - Optimal relaxation parameter depends on the matrix properties.
    - Convergence is guaranteed for strictly diagonally dominant matrices
        with 0 < ω < 2.
    - Uses Euclidean norm for convergence checking.

    """
    matrix = system.matrix
    if relaxation_param is None:
        relaxation_param = calculate_optimal_sor_parameter(matrix)
    solver = SORMethod(relaxation_param, config)
    return solver.solve(system)
