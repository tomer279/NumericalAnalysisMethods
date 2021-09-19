import numpy as np
from numpy import linalg as LA


# a private function to calculate the spectral radius of the matrix T = (D-L)^-1 * U
# if the matrix has a spectral radius higher or equal than 1, the method does not converge, and will return false
# -------------------------
# Parameter: A-> the matrix for which T is defined
def __check_convergence__(A: np.matrix):
    D = np.matrix([
        [A[0, 0], 0, 0],
        [0, A[1, 1], 0],
        [0, 0, A[2, 2]]
    ])
    L = np.matrix([
        [0, 0, 0],
        [-A[1, 0], 0, 0],
        [-A[2, 0], -A[2, 1], 0]
    ])
    U = np.matrix([
        [0, -A[0, 1], -A[0, 2]],
        [0, 0, -A[1, 2]],
        [0, 0, 0]
    ])
    T = np.matmul(LA.inv((D - L)), U)
    Q = LA.eigvals(T)
    if LA.norm(Q, np.inf) >= 1: # finding what is the spectral radius of T
        print("Method does not converge. \u03C1(T)=", LA.norm(Q, np.inf))
        return False
    return True


# Gauss Seidel method to numerically solve a 3x3 matrix equation Ax=b
# The method returns an approximate solution vector of the equation
# -----------------------------------
# Parameters:
# A -> The coefficient matrix of the equation Ax=b
# b -> the constant vector b of the equation Ax=b
# vec -> the initial approximation x^(0) to the method
# tol -> the tolerance of the method based on ||x - x0||_inf < tol
# N -> number of iterations for the method
def gauss_seidel(A: np.matrix, b: np.array, vec: np.array, tol: float, N: int):
    if __check_convergence__(A) is False: 
        return
    for i in range(N):  # calculating elements approximation vector
        x0 = (1 / A[0, 0]) * (-A[0, 1] * vec[1] - A[0, 2] * vec[2] + b[0])
        x1 = (1 / A[1, 1]) * (-A[1, 0] * x0 - A[1, 2] * vec[2] + b[1])
        x2 = (1 / A[2, 2]) * (-A[2, 0] * x0 - A[2, 1] * x1 + b[2])
        new_vec = np.array([x0, x1, x2])
        if LA.norm(vec - new_vec, np.inf) < tol:  # checking if the approximation is good enough
            print("Gauss-Seidel: Result after", i, "iterations:")
            print(new_vec)
            return new_vec
        vec = new_vec
    print("maximum number of iterations exceeded")
    return


A = np.matrix([
    [2, -1, 1],
    [2, 2, 4],
    [-1, -1, 2]])
b = np.array([-1, 4, -5])
x = np.array([0, 0, 0])
tol = 10 ** -4
gauss_seidel(A, b, x, tol, 20)
