import numpy as np


# LU factorization of a square matrix A,
# such that A = LU, where L is a lower
# triangular matrix, and U is an upper triangular matrix.
# The method assumes that the diagonal elements of L are all 1.
# If factorization is impossible, the method will print an appropriate response,
# and stops.
# the method is based on algorithm 6.4 from the book "Numerical Analysis" by Burden, 10th Ed
# PARAMETERS:
# A -> a square matrix of order n
def LUfactor(A: np.matrix):
    n = A.shape[0]
    L = np.zeros((n, n))
    for i in range(0, n):
        L[i, i] = 1
    U = np.zeros((n, n))
    if A[0, 0] == 0:
        print("factorization impossible")
        return
    U[0, 0] = A[0, 0]
    for j in range(1, n):
        U[0, j] = A[0, j]  # first row of U
        L[j, 0] = A[j, 0] / U[0, 0]  # first row of L
    for i in range(1, n - 1):
        sig = 0
        for k in range(0, i):
            sig += L[i, k] * U[k, i]
        U[i, i] = (A[i, i] - sig) / L[i, i]
        if U[i, i] == 0:
            print("Factorization impossible")
            return
        for j in range(i + 1, n):
            sig1 = 0
            sig2 = 0
            for k in range(0, i):
                sig1 += L[i, k] * U[k, j]
                sig2 += L[j, k] * U[k, i]
            U[i, j] = (A[i, j] - sig1) / L[i, i]  # ith row of U
            L[j, i] = (A[j, i] - sig2) / U[i, i]  # ith row of L
    sig = 0
    for k in range(0, n - 1):
        sig += L[n - 1, k] * U[k, n - 1]
    U[n - 1, n - 1] = (A[n - 1, n - 1] - sig) / L[n - 1, n - 1]
    print("L = ")
    print(L)
    print("U = ")
    print(U)
    return [L, U]


# an example of how LU factorization helps to solve linear equations Ax =b
# such that A is a square matrix, and b is a vector.
# if A = LU, than LUx = b, so we solve Ly = b, and then solve Ux = y, and find x.
# the method prints the LU factorization of A,
# and returns a solution vector x
# PARAMETERS:
# A -> a square matrix
# b -> a vector
def lin_solve(A: np.matrix, b: np.array):
    lu_vec = LUfactor(A)
    L = lu_vec[0]
    U = lu_vec[1]
    n = A.shape[0]
    y = np.zeros(n)
    y[0] = b[0] / L[0, 0]
    for i in range(1, n): # solving the equation Ly = b
        sig = 0
        for j in range(0, i):
            sig += L[i, j] * y[j]
        y[i] = (b[i] - sig) / L[i, i]
    x = np.zeros(n)
    x[n - 1] = y[n - 1] / U[n - 1, n - 1] # solving the equation Ux = y
    for i in range(n - 2, -1, -1):
        sig = 0
        for j in range(i+1, n):
            sig += U[i, j] * x[j]
        x[i] = (y[i] - sig) / U[i, i]
    return x


A_mat = np.matrix([[1, 1, 0, 3],
                   [2, 1, -1, 1],
                   [3, -1, -1, 2],
                   [-1, 2, 3, -1]], dtype=float)
b_vec = np.array([8,7,14,-7], dtype = float)
print(lin_solve(A_mat,b_vec))