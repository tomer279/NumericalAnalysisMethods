# Created by Tomer Caspi

import numpy as np
from numpy import linalg as LA

# Gauss-Seidel method to numerically solve the matrix equation Ax=b
# The method returns an approximate solution vector of the equation
# The code also checks if the method converges by calculating the spectral radius of the
# matrix method T = (D-L)^-1*U.
# if the spectral radius is larger than 1, the method does not converge.
# The code is based on algorithm 7.2 from the book "Numerical Analysis" by Burden, 10th Ed
# ----------------------------------------------------------------------
# Parameters:
# A -> The coefficient matrix of the equation Ax=b
# b -> the constant vector b of the equation Ax=b
# vec -> the initial approximation x^(0) to the method
# tol -> the tolerance of the method based on ||x - x0||_inf < tol
# N -> number of iterations for the method
def gauss_seidel(A: np.matrix, b: np.array, x: np.array, tol: float, N: int):
    if __check_convergence__(A) is False:
        return
    n = A.shape[0]
    y = np.zeros(n)
    for k in range(N):
        for i in range(0, n):
            sum1 = 0
            sum2 = 0
            for j in range(0, n):
                if j < i:
                    sum1 += A[i, j] * y[j]
                elif j > i:
                    sum2 += A[i, j] * x[j]
            y[i] = (b[i] - sum1 - sum2) / A[i, i]
        if LA.norm(x - y, np.inf) < tol:  # checking if the approximation is close enough
            print("Gauss-Seidel: Result after", k, "iterations:")
            print(y)
            return y
        x = y.copy()
    print("maximum number of iterations exceeded")
    return


# a private function to calculate the spectral radius of the matrix T = (D-L)^-1 * U
# if the matrix has a spectral radius higher or equal than 1, the method does not converge, and will return false
# -------------------------------------------------------------------------------------------------
# Parameter: A-> the matrix for which T is defined
def __check_convergence__(A: np.matrix):
    n = A.shape[0]
    D = np.zeros((n, n))
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(0, n):
        D[i, i] = A[i, i]
    for i in range(0, n - 1):
        for j in range(1, n):
            if i < j:
                U[i, j] = -A[i, j]
    for i in range(1, n):
        for j in range(0, n - 1):
            if i > j:
                L[i, j] = -A[i, j]
    T = LA.inv(D - L) @ U
    Q = LA.eigvals(T)  # vector containing the eigenvalues of T
    if LA.norm(Q, np.inf) >= 1:  # finding what is the spectral radius of T by finding the max norm of Q
        print("Method does not converge. \u03C1(T)=", LA.norm(Q, np.inf))
        return False
    return True


# implementation of the method:
A_mat = np.matrix([
    [4, 2, -2],
     [0, 4, 2],
     [1, 0, 4]], dtype=float)
b_vec = np.array([7, 2, 5], dtype=float)
x_vec = np.array([0, 0, 0], dtype=float)
TOL = 10 ** -6
gauss_seidel(A_mat, b_vec, x_vec, TOL, 100)
