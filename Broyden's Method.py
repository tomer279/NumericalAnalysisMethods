# Created by Tomer Caspi

import numpy as np
from numpy import linalg as LA
from numpy import log, sin, cos, exp, pi
import sys

np.set_printoptions(suppress=True)


# Broyden's method for solving a non-linear system F(x) = 0, where F is a function from R^n to R^n,
# x is a vector of R^n, and 0 is the zero vector of R^n, and n = 2 or n = 3.
# The method returns a vector approximation to the system which has an infinity norm less than the tolerance
# The method is based on algorithm 10.2 from the book "Numerical Analysis" by Burden, 10th Ed.

# PARAMETERS:
# F -> The function which creates the non-linear system. F can be of length 2 or 3.
# J -> THe jacobian of the function F. The user must calculate the jacobian first manually.
# x -> the initial approximation vector. the vector must be of length 2 or 3, and have the same length as F
# tol -> the tolerance for the method, such that an approximation returns when ||x||_(inf) < tol

def broyden(F, J, x: list, tol: float):
    if len(x) == 2:
        A0 = J(x[0], x[1], 0)
        v = F(x[0], x[1], 0)
    else:
        A0 = J(x[0], x[1], x[2])
        v = F(x[0], x[1], x[2])
    A = LA.inv(A0)
    s = -np.ravel(A.dot(v))
    x = x + s
    for k in range(1,sys.maxsize):
        w = v
        if len(v) == 2:
            v = F(x[0], x[1], 0)
        else:
            v = F(x[0], x[1], x[2])
        y = v - w
        z = -np.ravel(A.dot(y))
        p = -np.ravel(s.dot(z))
        u = np.ravel(A.T.dot(s))
        A = A + (1 / p) * np.outer(s + z, u.T)
        s = -np.ravel(A.dot(v))
        x = x + s
        if LA.norm(s, np.inf) < tol:
            print('Broydens method: the number of iterations:', k)
            return x
        print("iteration", k, ":", x)  # this command can be removed if you want to see only the final result
    print("The procedure was unsuccessful")
    return


# Writing down the function F: the function can have 2 or 3 variables.
# The length of the array must be the same as the number of variables.
# The jacobian for the function must be written manually by the user.
def function_a(x: float, y: float, z: float):  # write the desired function
    return np.array([log(x ** 2 + y ** 2) - sin(x * y) - log(2) - log(pi), exp(x - y) + cos(x * y)])


def jacobian_a(x: float, y: float, z: float):  # write the jacobian of the function
    return np.matrix([[(2 * x) / (x ** 2 + y ** 2) - y * cos(x * y), (2 * y) / (x ** 2 + y ** 2) - x * cos(x * y)],
                      [exp(x - y) - y * sin(x * y), -exp(x - y) - x * sin(x * y)]])


def function_b(x: float, y: float, z: float):
    return np.array([3 * x - cos(y * z) - 0.5, x ** 2 - 625 * y ** 2 - 0.25, exp(-x * y) + 20 * z + (10 * pi - 3) / 3])


def jacobian_b(x: float, y: float, z: float):
    return np.matrix([[3, z * sin(y * z), y * sin(y * z)]
                         , [2 * x, -1250 * y, 0]
                         , [-y * exp(-x * y), -x * exp(-x * y), 20]])


print(broyden(function_b, jacobian_b, [1.0,1.0,-0.1], 10 ** -6))
