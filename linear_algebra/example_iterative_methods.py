"""
Example: Jacobi, Gauss-Seidel, and SOR Convergence Analysis

This example demonstrates how to:
1. Set up a specific linear system Ax = b
2. Solve it using Jacobi, Gauss-Seidel, and SOR methods
3. Calculate optimal SOR parameter
4. Visualize the convergence behavior

Author: Tomer Caspi with the assistance of Cursor.ai
"""

import numpy as np
import stationary_iterative_methods
from stationary_iterative_methods import (
    JacobiMethod, GaussSeidelMethod, SORMethod, SolverConfig, LinearSystem,
)
from convergence_visualizer import ConvergenceTracker, ConvergenceVisualizer
from parameter_optimization import calculate_optimal_sor_parameter


def solve_and_visualize():
    """Solve the given system and visualize convergence."""
    # Define the system

    n = 10  # Size of the matrix

    main_diag = np.full(n, 3)  # Main diagonal elements
    super_diag = np.full(n - 1, -1)  # Superdiagonal elements
    sub_diag = np.full(n - 1, -1)  # Subdiagonal elements

    A = np.diag(main_diag) + \
        np.diag(super_diag, k=1) + \
        np.diag(sub_diag, k=-1)

    b = np.ones(n)
    x_0 = np.zeros(n)

    print("System Setup:")
    print("A =")
    print(A)
    print(f"b = {b}")
    print(f"x_0 = {x_0}")
    print()

    # Calculate true solution for error analysis
    true_solution = np.linalg.solve(A, b)
    print(f"True solution: {true_solution}")
    print()

    # Calculate optimal SOR parameter
    optimal_sor_omega = calculate_optimal_sor_parameter(A)
    print()

    # Configure solver
    config = SolverConfig(tolerance=1e-8, max_iterations=100, verbose=False)
    system = LinearSystem(A, b, x_0)

    # Create methods
    jacobi_method = JacobiMethod(config)
    gauss_seidel_method = GaussSeidelMethod(config)
    sor_method = SORMethod(optimal_sor_omega, config)

    # Create convergence trackers
    jacobi_tracker = ConvergenceTracker(jacobi_method)
    gauss_seidel_tracker = ConvergenceTracker(gauss_seidel_method)
    sor_tracker = ConvergenceTracker(sor_method)

    print("Solving with Jacobi method...")
    jacobi_data = jacobi_tracker.solve_with_tracking(system, true_solution)
    print(f"Jacobi: {'Converged' if jacobi_data.converged else 'Failed'} "
          f"after {jacobi_data.final_iteration} iterations")
    print(f"Final error: {jacobi_data.errors[-1]:.2e}")
    print(f"Spectral radius: {jacobi_data.spectral_radius:.4f}")
    print()

    print("Solving with Gauss-Seidel method...")
    gauss_seidel_data = gauss_seidel_tracker.solve_with_tracking(
        system, true_solution)
    print("Gauss-Seidel: "
          f"{'Converged' if gauss_seidel_data.converged else 'Failed'} "
          f"after {gauss_seidel_data.final_iteration} iterations")
    print(f"Final error: {gauss_seidel_data.errors[-1]:.2e}")
    print(f"Spectral radius: {gauss_seidel_data.spectral_radius:.4f}")
    print()

    print("Solving with SOR method (optimal ω)...")
    sor_data = sor_tracker.solve_with_tracking(system, true_solution)
    print(f"SOR: {'Converged' if sor_data.converged else 'Failed'} "
          f"after {sor_data.final_iteration} iterations")
    print(f"Final error: {sor_data.errors[-1]:.2e}")
    print(f"Spectral radius: {sor_data.spectral_radius:.4f}")
    print()

    # Create visualizer
    visualizer = ConvergenceVisualizer()

    # Plot convergence comparison
    print("Creating convergence comparison plot...")
    visualizer.plot_convergence_comparison(
        [jacobi_data, gauss_seidel_data, sor_data],
        title="Jacobi vs Gauss-Seidel vs SOR Convergence Comparison"
    )

    # Plot individual error evolution for each method
    print("Creating Jacobi error evolution plot...")
    visualizer.plot_error_evolution(
        jacobi_data,
        title="Jacobi Method: Error Evolution"
    )

    print("Creating Gauss-Seidel error evolution plot...")
    visualizer.plot_error_evolution(
        gauss_seidel_data,
        title="Gauss-Seidel Method: Error Evolution"
    )

    print("Creating SOR error evolution plot...")
    visualizer.plot_error_evolution(
        sor_data,
        title=f"SOR Method (ω={optimal_sor_omega:.4f}): Error Evolution"
    )

    # Print final solutions
    print("\nFinal Solutions:")
    print(f"Jacobi solution: \n {jacobi_data.solutions[-1]} \n")
    print(f"Gauss-Seidel solution: \n {gauss_seidel_data.solutions[-1]} \n")
    print(f"SOR solution: \n {sor_data.solutions[-1]} \n")
    print(f"True solution: \n {true_solution}")

    # Print convergence summary
    print("\nConvergence Summary:")
    print(f"{'Method':<15} {'Iterations':<12} "
          f"{'Final Error':<15} {'Spectral Radius':<15}")
    print("-" * 60)
    print(
        f"{'Jacobi':<15} {jacobi_data.final_iteration:<12} "
        f"{jacobi_data.errors[-1]:<15.2e} "
        f"{jacobi_data.spectral_radius:<15.4f}")
    print(
        f"{'Gauss-Seidel':<15} {gauss_seidel_data.final_iteration:<12} "
        f"{gauss_seidel_data.errors[-1]:<15.2e} "
        f"{gauss_seidel_data.spectral_radius:<15.4f}")
    print(
        f"{'SOR (optimal)':<15} {sor_data.final_iteration:<12}"
        f"{sor_data.errors[-1]:<15.2e} {sor_data.spectral_radius:<15.4f}")

    return jacobi_data, gauss_seidel_data, sor_data


if __name__ == "__main__":
    jacobi_data, gauss_seidel_data, sor_data = solve_and_visualize()
