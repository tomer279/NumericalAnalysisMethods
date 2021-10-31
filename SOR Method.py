# Created by Tomer Caspi

import numpy as np
from numpy import linalg as LA
import sys

# SOR method to numerically solve the matrix equation Ax=b given a parameter w
# The method returns an approximate solution vector of the equation
# The code is based on algorithm 7.3 from the book "Numerical Analysis" by Burden, 10th Ed
# ----------------------------------------------------------------------
# Parameters:
# A -> The coefficient matrix of the equation Ax=b
# b -> the constant vector b of the equation Ax=b
# x -> the initial approximation x^(0) to the method
# w -> the parameter which is connected to the SOR method.
# tol -> the tolerance of the method based on ||x - x0||_inf < tol
def sor(A: np.matrix, b: np.array, x: np.array, w: float, tol: float):
    n = A.shape[0]
    y = np.zeros(n)
    for k in range(1, sys.maxsize):
        for i in range(0, n):
            sum1 = 0
            sum2 = 0
            for j in range(0, n):
                if j < i:
                    sum1 += A[i, j] * y[j]
                elif j > i:
                    sum2 += A[i, j] * x[j]
            y[i] = (1 - w) * x[i] + (w / A[i, i]) * (b[i] - sum1 - sum2)
        if LA.norm(y - x) < tol:
            print("SOR method: result after", k, "iterations: ")
            print(y)
            return
        x = y.copy()
    print("No solution has been found")
    return


# input parameters:

A_mat = np.matrix([[4, 1, 1, 0, 1],
                   [-1, -3, 1, 1, 0],
                   [2, 1, 5, -1, -1],
                   [-1, -1, -1, 4, 0],
                   [0, 2, -1, 1, 4]], dtype=float)
b_vec = np.array([6, 6, 6,6,6], dtype=float)
x_vec = np.array([0, 0, 0,0,0], dtype=float)
TOL = 10 ** -10
omega = 1.1

sor(A_mat, b_vec, x_vec, omega, TOL)
