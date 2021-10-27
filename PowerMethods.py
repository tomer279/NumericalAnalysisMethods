import numpy as np
from numpy import linalg as LA


def powermethod(A: np.matrix, x: np.array, TOL: float, N: int):
    k = 1
    xp = LA.norm(x, np.inf)
    p = np.where(xp == np.absolute(x))[0][0]
    x = x / xp
    while k <= N:
        y = np.ravel(A.dot(x))
        mu = y[p]
        p = np.where(LA.norm(y, np.inf) == np.absolute(y))[0][0]
        if y[p] == 0:
            print("eigenvector" + x)
            print("A has the eigenvalue 0, select a new vector x and restart")
            return
        err = LA.norm(x - (y / y[p]), np.inf)
        x = y / y[p]
        if err < TOL:
            print("Power Method: Result after", k, "iterations")
            return mu
        k = k + 1
    print("The maximum number of iterations exceeded")
    return


def inversepowermethod(A: np.matrix, x: np.array, TOL: float, N: int, t: float):
    k = 1
    xp = LA.norm(x, np.inf)
    p = np.where(xp == np.absolute(x))[0][0]
    x = x / xp
    while k <= N:
        B = A - (t * np.identity(A.shape[0]))
        if np.linalg.det(B) == 0.0:
            print('t is the eigenvalue ', t)
            return
        y = np.linalg.solve(B, x)
        mu = y[p]
        p = np.where(LA.norm(y, np.inf) == np.absolute(y))[0][0]
        err = LA.norm(x - (y / y[p]), np.inf)
        x = y / y[p]
        if err < TOL:
            mu = (1 / mu) + t
            print("Inverse Power Method: Result after", k, "iterations")
            return mu
        k = k + 1

    print("Maximum number of iterations exceeded")
    return


A = np.matrix([[2, 1, 3, 4], [1, -3, 1, 5], [3, 1, 6, -2], [4, 5, -2, -1]])
x = np.array([1, 2.5, -0.75, -2.5])
TOL = 10 ** -3
N = 200
t = -8
powermethod(A,x,TOL,N)
print(inversepowermethod(A, x, TOL, N, t))
