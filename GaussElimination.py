import numpy as np


# Gaussian elimination with backward substitution
# to solve the linear equation Ax = b , such that A
# is a matrix of order n, and b is a column vector.
# The method receives matrix of order n x n+1, where the n+1 column
# represents the b vector
# the method returns a vector which is the solution for the linear equation.
# if there is no solution ,the method returns an appropriate print and stops.
# the method is based on algorithm 6.1 from "Numerical Analysis" by Burden, 10th Ed
# PARAMETERS:
# A -> a matrix of order n x (n+1), such that the first n columns create
# the coefficient matrix, and the last n+1 column is the vector b.
def gauss_elimination(A: np.matrix):
    n = A.shape[0]
    p = -1
    for i in range(0, n - 1):
        for p in range(i, n + 1):
            if p == n:
                print("no unique solution exists")
                return
            if A[p, i] != 0:
                break
        if p != i:
            A[[p, i]] = A[[i, p]]
        for j in range(i + 1, n):
            A[j] = A[j] - (A[j, i] / A[i, i]) * A[i]
    if A[n - 1, n - 1] == 0:
        print("no unique solution exists")
        return
    x = np.zeros(n)
    x[n - 1] = A[n - 1, n] / A[n - 1, n - 1]
    for i in range(n - 1, -1, -1):
        sig = 0
        for j in range(i + 1, n):
            sig += A[i, j] * x[j]
        x[i] = (A[i, n] - sig) / A[i, i]
    print(x)
    return


A_mat = np.matrix([[1, 1, 0, 3, 4],
                   [2, 1, -1, 1, 1],
                   [3, -1, -1, 2, -3],
                   [-1, 2, 3, -1, 4]], dtype=float)
gauss_elimination(A_mat)
