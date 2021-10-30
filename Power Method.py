# Created by Tomer Caspi

import numpy as np
from numpy import linalg as LA


# Power method for approximating the dominant eigenvalue and associated eigenvector of a square matrix.
# The code is based on algorithm 9.1 from the book "Numerical Analysis" by Burden, 10th Ed
# PARAMETERS:
# A -> The matrix for which we want to find the dominant eigenvalue
# x -> initial vector
# TOL -> tolerance for the method
# N -> maximum number of iterations

def powermethod(A: np.matrix, x: np.array, TOL: float, N: int):
    k = 1
    xp = LA.norm(x, np.inf)
    p = np.where(xp == np.absolute(x))[0][0]  # finding the smallest natural p where |x[p]| = ||x||_inf
    x = x / xp
    while k <= N:
        y = np.ravel(A.dot(x))
        mu = y[p]
        p = np.where(LA.norm(y, np.inf) == np.absolute(y))[0][0]  # finding the smallest natural p
        # where |y[p]| = ||y||_inf
        if y[p] == 0:
            print("eigenvector" + x)
            print("A has the eigenvalue 0, select a new vector x and restart")
            return
        err = LA.norm(x - (y / y[p]), np.inf)
        x = y / y[p]
        if err < TOL:
            print("Power Method: Result after", k, "iterations")
            return np.array([mu, k])
        k = k + 1
    print("The maximum number of iterations exceeded")
    return


# input Parameters:


mat = np.matrix([[2, 1, 3, 4],
               [1, -3, 1, 5],
               [3, 1, 6, -2],
               [4, 5, -2, -1]])
vec = np.array([1, 2.5, -0.75, -2.5])
tol = 10 ** - 6
n = 1000

print(powermethod(mat, vec, tol, n))
