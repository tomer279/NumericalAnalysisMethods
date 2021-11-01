import numpy as np
from numpy import linalg as LA


# Inverse power method for approximating an eigenvalue and associated eigenvector of a square matrix
# The code is based on algorithm 9.3 from the book "Numerical Analysis" by Burden, 10th Ed
# PARAMETERS:
# A -> The matrix for which we want to find the dominant eigenvalue
# x -> initial vector
# TOL -> tolerance for the method
# N -> maximum number of iterations
# t -> an approximation for the eigenvalue

def inversepowermethod(A: np.matrix, x: np.array, TOL: float, N: int, t: float):
    k = 1
    xp = LA.norm(x, np.inf)
    p = np.where(xp == np.absolute(x))[0][0] # finding the smallest natural p where |x[p]| = ||x||_inf
    x = x / xp
    while k <= N:
        B = A - (t * np.identity(A.shape[0]))  # B is the matrix A-tI
        if np.linalg.det(B) == 0.0:  # calculating if the determinant of B is 0.
            # if so, the system (A-tI)x = 0 is not solvable.
            print('t is the eigenvalue ', t)
            return
        y = np.linalg.solve(B, x)  # solving the linear system By = x
        mu = y[p]
        p = np.where(LA.norm(y, np.inf) == np.absolute(y))[0][0]  # finding the smallest natural p
        # where |y[p]| = ||y||_inf
        err = LA.norm(x - (y / y[p]), np.inf)
        x = y / y[p]
        if err < TOL:
            mu = (1 / mu) + t
            print("Inverse Power Method: Result after", k, "iterations")
            return np.array([mu, k])
        k = k + 1

    print("Maximum number of iterations exceeded")
    return


# input Parameters:
mat = np.matrix([[2, 1, 3, 4],
               [1, -3, 1, 5],
               [3, 1, 6, -2],
               [4, 5, -2, -1]])
vec = np.array([1, 2.5, -0.75, -2.5])
tol = 10 ** - 6
n = 1000
l = -8

print(inversepowermethod(mat, vec, tol, n, l))
