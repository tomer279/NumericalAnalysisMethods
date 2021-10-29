# Created by Tomer Caspi

import numpy as np
from numpy import linalg as LA
import sys
from numpy import log, sin, cos, exp, pi

np.set_printoptions(suppress=True)


# Steepest Descent for solving a non-linear system F(x) = 0, where F is a function from R^n to R^n,
# x is a vector of R^n, and 0 is the zero vector of R^n, and n = 2 or n = 3.
# The method approximates a solution to minimize the function g = sum_(i=1)^(n) F(x)[i]**2
# The gradiant of g is found directly by the jacobian of F
# The method returns a vector approximation to the system which has an infinity norm less than the tolerance
# and the value of g(x) on the approximation
# The method is based on algorithm 10.3 from the book "Numerical Analysis" by Burden, 10th Ed.

# PARAMETERS:
# F -> The function which creates the non-linear system. F must be of length 2 or 3.
# J -> The jacobian of the function F.
# g -> The function g = sum_(i=1)^(n)  F(x)[i]**2
# vec -> The initial approximation vector,
# tol -> the tolerance for the method, such that an approximation returns when ||x||_(inf) < tol
def steepest_descent_direct_grad(F: np.array, J, g, vec: np.array, tol: float):
    for k in range(1, sys.maxsize):
        g_1 = g(vec, F)  # g_1 = g(x^((k)))
        # z = grad(g(x^((k))) = 2 * J(x^((k))) * F(x^((k)))
        if len(vec) == 2:
            z = np.ravel(2 * np.dot(J(vec[0], vec[1], 0).T, F(vec[0], vec[1], 0)))
        else:
            z = np.ravel(2 * np.dot(J(vec[0], vec[1], vec[2]).T, F(vec[0], vec[1], vec[2])))
        z_0 = LA.norm(z)
        if z_0 == 0:
            print("zero gradient")
            print(vec, g_1)
            return
        z = z / z_0  # make z to a unit vector
        a_3 = 1
        g_3 = g(vec - a_3 * z, F)
        while g_3 >= g_1:
            a_3 = a_3 / 2
            g_3 = g(vec - a_3 * z, F)
            if a_3 < tol / 2:
                print("No likely improvement")
                print(vec, g_1)
                return
        a_2 = a_3 / 2
        g_2 = g(vec - a_2 * z, F)
        # calculating Newton's interpolating polynomial P(a) = g1 + h1*a + h3*a*(a-a2)
        # that interpolates h(a) at a = 0, a = a2 and a = a3
        h_1 = (g_2 - g_1) / a_2
        h_2 = (g_3 - g_2) / (a_3 - a_2)
        h_3 = (h_2 - h_1) / a_3
        a_0 = 0.5 * (a_2 - (h_1 / h_3))  # critical point of P occurs at a0
        g_0 = g(vec - a_0 * z, F)
        if g_0 == min(g_0, g_3):
            a = a_0
        else:
            a = a_3
        vec = vec - a * z
        if np.abs(g(vec, F) - g_1) < tol:
            print("Steepest Descent with numerical gradiant: the number of iterations:", k)
            print(vec, g(vec, F))  # procedure is successful
            return
        # the command below can be removed if you want to see only the final result
        print("iteration", k, ":", vec, g(vec, F))
    print('Maximum iterations exceeded')
    return


# Input functions:

def function_a(x: float, y: float, z: float):
    return np.array([log(x ** 2 + y ** 2) - sin(x * y) - log(2) - log(pi), exp(x - y) + cos(x * y)])


def jacobian_a(x: float, y: float, z: float):
    return np.matrix([[(2 * x) / (x ** 2 + y ** 2) - y * cos(x * y), (2 * y) / (x ** 2 + y ** 2) - x * cos(x * y)],
                      [exp(x - y) - y * sin(x * y), -exp(x - y) - x * sin(x * y)]])


def function_b(x: float, y: float, z: float):
    return np.array([3 * x - cos(y * z) - 0.5,
                     x ** 2 - 625 * y ** 2 - 0.25,
                     exp(-x * y) + 20 * z + (10 * pi - 3) / 3])


def jacobian_b(x: float, y: float, z: float):
    return np.matrix([[3, z * sin(y * z), y * sin(y * z)]
                         ,[2 * x, -1250 * y, 0]
                         ,[-y * exp(-x * y), -x * exp(-x * y), 20]])


def g_func(vec: np.array, f):
    if len(vec) == 2:
        return f(vec[0], vec[1], 0)[0] ** 2 + f(vec[0], vec[1], 0)[1] ** 2
    else:
        return f(vec[0], vec[1], vec[2])[0] ** 2 + f(vec[0], vec[1], vec[2])[1] ** 2 + \
               f(vec[0], vec[1], vec[2])[2] ** 2


steepest_descent_direct_grad(function_a, jacobian_a, g_func, np.array([2.0, 2.0]), 10 ** -6)
