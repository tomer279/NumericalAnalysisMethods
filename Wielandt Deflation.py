
import numpy as np
from numpy import linalg as LA
import sys


# wielandt deflation to approximate the second most dominant eigenvalue, and the associated eigenvector,
# of a matrix of order n.
# the method uses a slightly modified power method as a private function for the purpose of this method.
# the code is based on algorithm 9.4 from the book "Numerical Analysis" by Burden, 10th Ed

# PARAMETERS:
# A -> a square matrix of order n
# q -> approximation of (or actual) first dominant eigenvalue
# v -> approximation of (or actual) the associated eigenvector of q
# x -> a vector of order n-1. must not be zero
# tol -> the tolerance for the method
def wielandt(A: np.matrix, q: float, v: np.array, x: np.array, tol: float):
    n = A.shape[0]
    i = np.where(LA.norm(v, np.inf) == np.absolute(v))[0][0]
    B = np.asmatrix(np.zeros((n - 1, n - 1)))
    if i != 0:
        for k in range(0, i):
            for j in range(0, i):
                B[k, j] = A[k, j] - (v[k] / v[i]) * A[i, j]
    if i != 0 and i != n - 1:
        for k in range(i, n - 1):
            for j in range(0, i):
                B[k, j] = A[k + 1, j] - (v[k + 1] / v[i]) * A[i, j]
                B[j, k] = A[j, k + 1] - (v[j] / v[i]) * A[i, k + 1]
    if i != n - 1:
        for k in range(i, n - 1):
            for j in range(i, n - 1):
                B[k, j] = A[k + 1, j + 1] - (v[k + 1] / v[i]) * A[i, j + 1]
    pm = __powermethod(B, x, tol)
    mu = pm[0]
    w_tag = pm[1]
    w = np.zeros(n)
    if i != 0:
        for k in range(1, i - 1):
            w[k] = w_tag[k]
    w[i] = 0
    if i != n - 1:
        for k in range(i + 1, n):
            w[k] = w_tag[k - 1]
    sig = 0
    for j in range(0, n):
        sig += A[i, j] * w[j]
    u = np.zeros(n)
    for k in range(0, n):
        u[k] = (mu - q) * w[k] + sig * (v[k] / v[i])
    print(mu,u)
    return


# power method for the deflation method.
# it is the exact same method that I wrote, but instead of returning a printed format,
# it returns an array containing the dominant eigenvalue and its eigenvector
# details about the parameters are listed in the original code for the power method
def __powermethod(A: np.matrix, x: np.array, TOL: float):
    k = 1
    xp = LA.norm(x, np.inf)
    p = np.where(xp == np.absolute(x))[0][0]
    x = x / xp
    while k <= sys.maxsize:
        y = np.ravel(A.dot(x))
        mu = y[p]
        p = np.where(LA.norm(y, np.inf) == np.absolute(y))[0][0]
        if y[p] == 0:
            print("eigenvector" + x)
            print("A has the eigenvalue 0, select a new vector x and restart")
            return False
        err = LA.norm(x - (y / y[p]), np.inf)
        x = y / y[p]
        if err < TOL:
            return [mu, x, k]
        k = k + 1
    return False


# input parameters
A_mat = np.matrix([[1, 1, 1],
                   [1, 1, 0],
                   [1, 0, 1]], dtype=float)
q_val = 2.414
v_vec = np.array([1, 0.707, 0.707], dtype=float)
x_vec = np.array([1,1], dtype=float)  # must not be 0 vector
tol_val = 10**-4

wielandt(A_mat, q_val, v_vec, x_vec, tol_val)
