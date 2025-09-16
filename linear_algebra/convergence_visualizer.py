"""
Convergence Visualization for Iterative Methods

This module provides comprehensive visualization tools
for analyzing the convergence behavior of iterative methods
 for solving linear systems Ax = b.

Author: Tomer Caspi with the assistance of Cursor.ai
Based on "Numerical Analysis" by Burden, 10th Edition
"""
from typing import List, Dict, Optional
from dataclasses import dataclass
import sys
import os
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from stationary_iterative_methods import (
    IterativeMethod, JacobiMethod, GaussSeidelMethod, SORMethod,
    RichardsonMethod, DampedJacobiMethod, SolverConfig, LinearSystem
)
from parameter_optimization import (
    calculate_optimal_richardson_parameter, calculate_optimal_sor_parameter)


# Add the current directory to the path to import iterative_methods
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class ConvergenceData:
    """Data structure to store convergence information for a method."""
    method_name: str
    iterations: List[int]
    errors: List[float]
    solutions: List[np.ndarray]
    converged: bool
    final_iteration: int
    spectral_radius: Optional[float] = None


class ConvergenceTracker(IterativeMethod):
    """Extended iterative method that tracks convergence data."""

    def __init__(self, base_method: IterativeMethod) -> None:
        # Call the parent class constructor with the base method's config
        super().__init__(base_method.config)
        self.base_method = base_method
        self.convergence_data = ConvergenceData(
            method_name=base_method.get_method_name(),
            iterations=[],
            errors=[],
            solutions=[],
            converged=False,
            final_iteration=0
        )

    def compute_iteration(self, system: LinearSystem,
                          current_vector: np.ndarray) -> np.ndarray:
        """Compute one iteration using the base method."""
        return self.base_method.compute_iteration(system, current_vector)

    def check_convergence(self, system: LinearSystem) -> bool:
        """Check convergence using the base method."""
        return self.base_method.check_convergence(system)

    def get_method_name(self) -> str:
        """Get method name from base method."""
        return self.base_method.get_method_name()

    def get_iteration_matrix(self,
                             system: LinearSystem) -> Optional[np.ndarray]:
        """Get iteration matrix by delegating to the base method."""
        return self.base_method.get_iteration_matrix(system)

    def solve_with_tracking(
            self, system: LinearSystem,
            true_solution: Optional[np.ndarray] = None) -> ConvergenceData:
        """
        Solve the linear system while tracking convergence data.

        Parameters
        ----------
        system : LinearSystem
            The linear system to solve
        true_solution : np.ndarray, optional
            The true solution for error calculation.
            If None, uses residual error.

        Returns
        -------
        ConvergenceData
            Complete convergence tracking data
        """
        if not self.check_convergence(system):
            return self.convergence_data

        current_vector = system.initial_vector.copy()

        # Calculate spectral radius for analysis
        try:
            self.convergence_data.spectral_radius = (
                self._calculate_spectral_radius(system))
        except (LA.LinAlgError, ValueError, TypeError) as e:
            if self.config.verbose:
                print(f"Warning: Could not calculate spectral radius: {e}")
            self.convergence_data.spectral_radius = None

        for iteration in range(1, self.config.max_iterations + 1):
            new_vector = self.compute_iteration(system, current_vector)

            # Calculate error
            if true_solution is not None:
                error = np.linalg.norm(new_vector - true_solution)
            else:
                # Use residual error: ||Ax - b||
                residual = system.matrix @ new_vector - system.constant_vector
                error = np.linalg.norm(residual)

            # Store convergence data
            self.convergence_data.iterations.append(iteration)
            self.convergence_data.errors.append(error)
            self.convergence_data.solutions.append(new_vector.copy())

            # Check convergence
            if (np.linalg.norm(new_vector - current_vector)
                    < self.config.tolerance):
                self.convergence_data.converged = True
                self.convergence_data.final_iteration = iteration
                break

            current_vector = new_vector.copy()

        if not self.convergence_data.converged:
            self.convergence_data.final_iteration = self.config.max_iterations

        return self.convergence_data

    def _calculate_spectral_radius(self,
                                   system: LinearSystem) -> Optional[float]:
        """Calculate spectral radius for the iteration matrix."""
        try:
            return self.base_method.get_spectral_radius(system)
        except (LA.LinAlgError, ValueError, AttributeError) as e:
            if self.config.verbose:
                print(f"Warning: Spectral radius calculation falied: {e}")
            return None


class ConvergenceVisualizer:
    """Main class for visualizing iterative method convergence."""

    def __init__(self, figsize=(12, 8)):
        """
        Initialize the visualizer with plotting style.

        Parameters
        ----------
        figsize : tuple
            Default figure size
        """
        self.figsize = figsize
        self._setup_plotting_style()

    def _setup_plotting_style(self):
        """Setup consistent plotting style following project standards."""
        plt.rcParams.update({
            'figure.figsize': self.figsize,
            'font.size': 12,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 2,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'legend.fontsize': 12,
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })

    def plot_convergence_comparison(
            self, convergence_data_list: List[ConvergenceData],
            title: str = "Iterative Methods Convergence Comparison",
            log_scale: bool = True,
            save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive convergence comparison plot.

        Parameters
        ----------
        convergence_data_list : List[ConvergenceData]
            List of convergence data for different methods
        title : str
            Title for the plot
        log_scale : bool
            Whether to use logarithmic scale for y-axis
        save_path : str, optional
            Path to save the plot (only saves if provided)

        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Define colors for different methods
        colors = ['blue', 'red', 'green', 'orange',
                  'purple', 'brown', 'pink', 'gray']

        # Plot 1: Error vs Iterations
        ax1 = self._plot_error_vs_iterations(convergence_data_list, colors,
                                             ax1, log_scale)

        # Plot 2: Convergence Summary
        ax2 = self._plot_convergence_summary(convergence_data_list, colors,
                                             ax2)

        plt.suptitle(title, fontsize=18, fontweight='bold')
        plt.tight_layout()

        # Only save if explicitly requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        # Simple show - no complex backend handling
        plt.show()
        return fig

    def _plot_error_vs_iterations(self, convergence_data_list,
                                  colors, ax1, log_scale):
        """Helper method to plot error vs iterations"""
        for i, data in enumerate(convergence_data_list):
            color = colors[i % len(colors)]
            ax1.plot(data.iterations, data.errors,
                     color=color, linewidth=2, marker='o', markersize=4,
                     label=f"{data.method_name} (ρ={data.spectral_radius:.3f})"
                     if data.spectral_radius else data.method_name)

        ax1.set_xlabel('Iteration Number')
        ax1.set_ylabel('Error')
        ax1.set_title('Convergence Rate Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        if log_scale:
            ax1.set_yscale('log')

        return ax1

    def _plot_convergence_summary(self, convergence_data_list,
                                  colors, ax2):
        """Helper method to plot convergence summary bar chart."""
        method_names = [data.method_name for data in convergence_data_list]
        final_errors = [data.errors[-1] if data.errors else float('inf')
                        for data in convergence_data_list]
        iterations_to_converge = [
            data.final_iteration for data in convergence_data_list]

        x_pos = np.arange(len(method_names))
        bars = ax2.bar(x_pos, iterations_to_converge,
                       color=colors[:len(method_names)], alpha=0.7)

        # Add value labels on bars
        for bar_rect, error in zip(bars, final_errors):
            height = bar_rect.get_height()
            ax2.text(bar_rect.get_x() + bar_rect.get_width()/2.,
                     height + 0.5,
                     f'{height}\n(ε={error:.2e})',
                     ha='center', va='bottom', fontsize=10)

        ax2.set_xlabel('Method')
        ax2.set_ylabel('Iterations to Converge')
        ax2.set_title('Convergence Speed Comparison')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(method_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        return ax2

    def plot_error_evolution(self, convergence_data: ConvergenceData,
                             title: Optional[str] = None,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot detailed error evolution for a single method.

        Parameters
        ----------
        convergence_data : ConvergenceData
            Convergence data for the method
        title : str, optional
            Title for the plot
        save_path : str, optional
            Path to save the plot (only saves if provided)

        Returns
        -------
        plt.Figure
            The created figure
        """
        if title is None:
            title = f"Error Evolution: {convergence_data.method_name}"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot 1: Error vs Iterations
        ax1.plot(convergence_data.iterations, convergence_data.errors,
                 'b-o', linewidth=2, markersize=4)
        ax1.set_xlabel('Iteration Number')
        ax1.set_ylabel('Error')
        ax1.set_title(f'Error Evolution: {convergence_data.method_name}')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)

        # Add convergence info
        convergence_text = (
            f"Converged: {'Yes' if convergence_data.converged else 'No'}\n"
            f"Iterations: {convergence_data.final_iteration}\n"
            f"Final Error: {convergence_data.errors[-1]:.2e}")
        if convergence_data.spectral_radius:
            convergence_text += (
                f"\nSpectral Radius: {convergence_data.spectral_radius:.3f}")

        ax1.text(0.02, 0.98, convergence_text, transform=ax1.transAxes,
                 verticalalignment='top',
                 bbox={'boxstyle': "round,pad=0.3",
                       'facecolor': "white",
                       'alpha': 0.8})

        # Plot 2: Error Reduction Rate
        if len(convergence_data.errors) > 1:
            error_reduction = np.diff(np.log(convergence_data.errors))
            ax2.plot(convergence_data.iterations[1:], error_reduction,
                     'r-s', linewidth=2, markersize=4)
            ax2.set_xlabel('Iteration Number')
            ax2.set_ylabel('Log Error Reduction Rate')
            ax2.set_title('Error Reduction Rate')
            ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Only save if explicitly requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        # Simple show - no complex backend handling
        plt.show()
        return fig


def analyze_convergence(
        matrix: np.ndarray,
        constant_vector: np.ndarray,
        initial_vector: np.ndarray,
        relaxation_params:
        Optional[Dict[str, float]] = None) -> List[ConvergenceData]:
    """
    Analyze convergence for all iterative methods on a given problem.

    Parameters
    ----------
    matrix : np.ndarray
        Coefficient matrix A
    constant_vector : np.ndarray
        Right-hand side vector b
    initial_vector : np.ndarray
        Initial guess x0
    relaxation_params : dict, optional
        Dictionary of relaxation parameters for methods that need them

    Returns
    -------
    List[ConvergenceData]
        List of convergence data for all methods
    """
    if relaxation_params is None:
        relaxation_params = {
            'sor': calculate_optimal_sor_parameter(matrix),
            'richardson': calculate_optimal_richardson_parameter(matrix),
            'damped_jacobi': 2/3  # Optimal value for Damped Jacobi
        }

    config = SolverConfig(tolerance=1e-8, max_iterations=100, verbose=False)
    system = LinearSystem(matrix, constant_vector, initial_vector)

    # Calculate true solution for error analysis
    try:
        true_solution = np.linalg.solve(matrix, constant_vector)
    except LA.LinAlgError as e:
        print(f"Warning: Could not calculate true solution: {e}")
        true_solution = None
    except ValueError as e:
        print(f"Warning: Invalid matrix/vector dimensions: {e}")
        true_solution = None

    convergence_data_list = []

    # Test all methods
    methods_to_test = [
        ('Jacobi', JacobiMethod(config)),
        ('Gauss-Seidel', GaussSeidelMethod(config)),
        ('SOR', SORMethod(relaxation_params['sor'], config)),
        ('Richardson (Optimal)', RichardsonMethod(
            relaxation_params['richardson'], config)),
        ('Damped Jacobi', DampedJacobiMethod(
            relaxation_params['damped_jacobi'], config))
    ]

    for method_name, method in methods_to_test:
        try:
            tracker = ConvergenceTracker(method)
            convergence_data = tracker.solve_with_tracking(
                system, true_solution)
            convergence_data_list.append(convergence_data)
            print(f"{method_name}:"
                  f"{'Converged' if convergence_data.converged else 'Failed'}"
                  f"after {convergence_data.final_iteration} iterations")
        except LA.LinAlgError as e:
            print(f"{method_name}: Failed with linear algebra error - {e}")
        except (ValueError, TypeError, AttributeError) as e:
            print(f"{method_name}: Failed with parameter error - {e}")
        except RuntimeError as e:
            print(f"{method_name}: Failed with runtime error - {e}")

    return convergence_data_list
