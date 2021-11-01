import numpy as np

np.set_printoptions(suppress=True)


# Householder method for matrices to obtain a matrix that is similar to that matrix and "simple".
# if the matrix is symmetric, than the method returns a tridiagonal symmetric matrix that is similar to it.
# if not, the method returns a hessenberg matrix which is similar to the matrix as well.
# the code is based on algorithm 9.5 from the book "Numerical Analysis" by Burden, 10th Ed.

# PARAMETERS:
# A -> a matrix.
def householder(A: np.matrix):
    n = A.shape[0]
    for k in range(0, n - 2):
        q = 0
        for j in range(k + 1, n):
            q += A[j, k] ** 2
        if A[k + 1, k] == 0:
            alph = -(q ** 0.5)
        else:
            alph = - (q ** 0.5) * A[k + 1, k] / np.abs(A[k + 1, k])
        rsq = alph ** 2 - alph * A[k + 1, k]
        v = np.zeros(n)
        v[k + 1] = A[k + 1, k] - alph
        for j in range(k + 2, n):
            v[j] = A[j, k]
        if np.array_equal(A, A.T):
            u = (1 / rsq) * np.ravel(A.dot(v))
            z = u - (1/(2*rsq)) * np.ravel(v.dot(u)) * v
            A = A - np.outer(v, z) - np.outer(z, v)
        else:
            u = (1 / rsq) * np.ravel(A.dot(v))
            y = np.zeros(n)
            for j in range(0, n):
                sig = 0
                for i in range(k + 1, n):
                    sig += A[i, j] * v[i]
                y[j] = sig / rsq
            prod = np.ravel(u.dot(v))
            z = np.zeros(n)
            for j in range(0, n):
                z[j] = u[j] - (prod / rsq) * v[j]
            B = A.copy()
            for l in range(k + 1, n):
                for j in range(0, k + 1):
                    A[j, l] = B[j, l] - z[j] * v[l]
                    A[l, j] = B[l, j] - y[j] * v[l]
                for j in range(k + 1, n):
                    A[j, l] = B[j, l] - z[j] * v[l] - y[l] * v[j]
    return A

# input parameters


A_mat = np.matrix([[2, -1, 3],
                   [2, 0, 1],
                   [-2, 1, 4]], dtype=float)
B_mat = np.matrix([[4,1,-2,2],
                   [1,2,0,1],
                   [-2,0,3,-2],
                   [2,1,-2,-1]], dtype=float)


print(householder(A_mat))
print(householder(B_mat))
