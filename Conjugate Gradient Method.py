
import numpy as np
from numpy import linalg as LA
import sys

# Conjugate Gradient method with preconditioning
# to numerically solve the matrix equation Ax=b.
# The method returns an approximate solution vector of the equation
# The code allows the use of a preconditioning matrix C^-1
# The code is based on algorithm 7.3 from the book "Numerical Analysis" by Burden, 10th Ed
# ----------------------------------------------------------------------
# Parameters:
# A -> The coefficient matrix of the equation Ax=b
# b -> the constant vector b of the equation Ax=b
# C -> the preconditioned matrix. Notice that C is meant to be C^-1.
# x -> the initial approximation x^(0) to the method
# tol -> the tolerance of the method based on ||x - x0||_inf < tol

def CG_method(A: np.matrix, b: np.array, C: np.matrix, x: np.array, tol: float):
    r = b - np.ravel(A.dot(x))
    w = np.ravel(C.dot(r))
    v = np.ravel(C.T.dot(w))
    alph = np.sum(np.power(w, 2))
    for i in range(0, sys.maxsize):
        if LA.norm(v, np.inf) < tol:
            print("Solution vector: ", x)
            print("with residual: ", r)
            return
        u = np.ravel(A.dot(v))
        t = alph / u.dot(v)
        x += t * v
        r = r - t * u
        w = np.ravel(C.dot(r))
        beta = np.sum(np.power(w, 2))
        if beta < tol and LA.norm(r, np.inf) < tol:
            print("Solution vector: ", x)
            print("with residual: ", r)
            print("Approximation arrived after ", i, "iterations")
            return
        s = beta / alph
        v = np.ravel(C.T.dot(w)) + s * v
        alph = beta
    print("No solution was found")
    return


# input parameters:

A_mat = np.matrix([[1, 0.5, 1 / 3],
                   [1 / 2, 1 / 3, 1 / 4],
                   [1 / 3, 1 / 4, 1 / 5]])
b_vec = np.array([5 / 6, 5 / 12, 17 / 60])
C_mat = np.matrix([[1., 0., 0.],
                   [0., np.sqrt(3), 0.],
                   [0., 0., np.sqrt(5)]])
x_vec = np.array([0., 0., 0.])
TOL = 10 ** -4

CG_method(A_mat, b_vec, C_mat, x_vec, TOL)
