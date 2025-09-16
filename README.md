# Numerical Analysis Methods

A comprehensive collection of numerical analysis methods implemented in Python for educational and research purposes.

## ⚠️ Project Status: Work in Progress

**IMPORTANT:** This repository contains code at various stages of development and quality. While some modules (particularly the newer ones) are well-documented and follow best practices, other parts of the codebase are legacy implementations that need significant improvement.

### Current State
- **Well-documented modules**: `stationary_iterative_methods`, `convergence_visualizer`, `example_iterative_methods`
- **Legacy modules**: Various older implementations with inconsistent naming conventions and documentation
- **Active development**: Continuously improving code quality, documentation, and adding new methods

## Available Methods

### Linear Algebra - Iterative Methods
**Status: Production Ready**
- **Jacobi Method**: Stationary iterative method for solving linear systems
- **Gauss-Seidel Method**: Improved convergence over Jacobi
- **SOR (Successive Over-Relaxation)**: Optimized relaxation parameter calculation
- **Richardson Method**: With optimal parameter calculation
- **Damped Jacobi**: Enhanced Jacobi with relaxation parameter

**Features:**
- Object-oriented design with abstract base classes
- Comprehensive convergence analysis
- Automatic parameter optimization
- Detailed visualization tools
- Full documentation with mathematical background

### Other Methods
**Status: Legacy/Under Development**
- Eigenvalue Methods  
- Nonlinear Solvers
- ODE Solvers
- Boundary Value Problems

*Note: These modules contain older implementations that are being refactored and improved.*

### Example: Solving Linear Systems
```python
from stationary_iterative_methods import JacobiMethod, SolverConfig, LinearSystem
import numpy as np

# Define system: Ax = b
A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])
b = np.array([1, 1, 1])
x0 = np.zeros(3)

# Configure and solve
config = SolverConfig(tolerance=1e-6, max_iterations=100)
system = LinearSystem(A, b, x0)
jacobi = JacobiMethod(config)
solution = jacobi.solve(system)
```

### Example: Convergence Analysis
```python
from convergence_visualizer import ConvergenceVisualizer, analyze_convergence

# Analyze all methods
convergence_data = analyze_convergence(A, b, x0)

# Visualize results
visualizer = ConvergenceVisualizer()
visualizer.plot_convergence_comparison(convergence_data)
```

## 🤝 Contributing

Contributions are welcome! Please note:

1. **For new features**: Follow the coding standards established in the `linear_algebra` modules
2. **For legacy code**: Focus on improving documentation and refactoring
3. **Issues**: Report bugs or suggest improvements for any module
4. **Pull requests**: Include tests and documentation updates

### Coding Standards
- Use lowercase with underscores for file names
- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Add type hints where appropriate
## References

Based on "Numerical Analysis" by Burden & Faires, 10th Edition.

## Author

**Tomer Caspi** - Developed with assistance from Cursor.ai

**Note**: This project is actively maintained. The iterative methods in the `linear_algebra` directory represent the current development standards, while other modules are being updated to match this quality level.
