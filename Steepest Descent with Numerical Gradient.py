import numpy as np
from numpy import linalg as LA
import sys
from numpy import log, sin, cos, exp, pi

np.set_printoptions(suppress=True)


# Steepest Descent for solving a non-linear system F(x) = 0, where F is a function from R^n to R^n,
# x is a vector of R^n, and 0 is the zero vector of R^n, and n = 2 or n = 3.
# The method approximates a solution to minimize the function g = sum_(i=1)^(n) F(x)[i]**2
# The gradiant of g is evaluated by a 5-point midpoint derivative formula
# The method returns a vector approximation to the system which has an infinity norm less than the tolerance
# and the value of g(x) on the approximation
# The method is based on algorithm 10.3 from the book "Numerical Analysis" by Burden, 10th Ed.

# PARAMETERS:
# F -> The function which creates the non-linear system. F must be of length 2 or 3.
# g -> The function g = sum_(i=1)^(n)  F(x)[i]**2
# vec -> The initial approximation vector
# tol -> the tolerance for the method, such that an approximation returns when ||x||_(inf) < tol
def steepest_descent_numerical_grad(F: np.array, g, vec: np.array, tol: float):
    for k in range(1, sys.maxsize):
        g_1 = g(vec, F)  # g_1 = g(x^((k)))
        z = __midpoint_rule(F, vec)  # z = grad(g(x^(k)))
        z_0 = LA.norm(z)
        if z_0 == 0:
            print("zero gradient")
            print(vec, g_1)  # procedure completed, might have minimum
            return
        z = z / z_0  # make z a vector unit
        a_3 = 1
        g_3 = g(vec - a_3 * z, F)
        while g_3 >= g_1:
            a_3 = a_3 / 2
            g_3 = g(vec - a_3 * z, F)
            if a_3 < tol / 2:
                print("No likely improvement")  # procedure completed, might have minimum
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
    print('Maximum iterations exceeded')  # procedure is unsuccessful
    return


# Private method for calculating the gradiant of g using 5-point midpoint rule
# The method calculates the approximating derivatives of F, and returns the approximate gradiant
# The method works only on F of length 2 or 3.

# PARAMETERS:
# F -> The function F of the non-linear system
# vec -> a vector in which the approximation is calculated

def __midpoint_rule(F, vec: list):
    x = vec[0]
    y = vec[1]
    if len(vec) == 2:
        return np.array([2 * F(x, y, 0)[0] * (5 / 6) *
                         (F(x - 0.2, y, 0)[0] - 8 * F(x - 0.1, y, 0)[0] + 8 * F(x + 0.1, y, 0)[0] - F(x + 0.2, y, 0)[0])
                         + 2 * F(x, y, 0)[1] * (5 / 6) * (
                                 F(x - 0.2, y, 0)[1] - 8 * F(x - 0.1, y, 0)[1] + 8 * F(x + 0.1, y, 0)[1] -
                                 F(x + 0.2, y, 0)[1]),
                         2 * F(x, y, 0)[0] * (5 / 6) *
                         (F(x, y - 0.2, 0)[0] - 8 * F(x, y - 0.1, 0)[0] + 8 * F(x, y + 0.1, 0)[0] - F(x, y + 0.2, 0)[0])
                         + 2 * F(x, y, 0)[1] * (5 / 6) *
                         (F(x, y - 0.2, 0)[1] - 8 * F(x, y - 0.1, 0)[1] + 8 * F(x, y + 0.1, 0)[1] - F(x, y + 0.2, 0)[
                             1])], dtype=float)
    else:
        z = vec[2]
        return np.array([2 * F(x, y, z)[0] * (5 / 6) * (
                F(x - 0.2, y, z)[0] - 8 * F(x - 0.1, y, z)[0] + 8 * F(x + 0.1, y, z)[0] - F(x + 0.2, y, z)[0])
                         + 2 * F(x, y, z)[1] * (5 / 6) * (
                                 F(x - 0.2, y, z)[1] - 8 * F(x - 0.1, y, z)[1] + 8 * F(x + 0.1, y, z)[1] -
                                 F(x + 0.2, y, z)[1])
                         + 2 * F(x, y, z)[2] * (5 / 6) * (
                                 F(x - 0.2, y, z)[2] - 8 * F(x - 0.1, y, z)[2] + 8 * F(x + 0.1, y, z)[2] -
                                 F(x + 0.2, y, z)[2]),

                         2 * F(x, y, z)[0] * (5 / 6) * (
                                 F(x, y - 0.2, z)[0] - 8 * F(x, y - 0.1, z)[0] + 8 * F(x, y + 0.1, z)[0] -
                                 F(x, y + 0.2, z)[0])
                         + 2 * F(x, y, z)[1] * (5 / 6) * (
                                 F(x, y - 0.2, z)[1] - 8 * F(x, y - 0.1, z)[1] + 8 * F(x, y + 0.1, z)[1] -
                                 F(x, y + 0.2, z)[1])
                         + 2 * F(x, y, z)[2] * (5 / 6) * (
                                 F(x, y - 0.2, z)[2] - 8 * F(x, y - 0.1, z)[2] + 8 * F(x, y + 0.1, z)[2] -
                                 F(x, y + 0.2, z)[2]),

                         2 * F(x, y, z)[0] * (5 / 6) * (
                                 F(x, y, z - 0.2)[0] - 8 * F(x, y, z - 0.1)[0] + 8 * F(x, y, z + 0.1)[0] -
                                 F(x, y, z + 0.2)[0])
                         + 2 * F(x, y, z)[1] * (5 / 6) * (
                                 F(x, y, z - 0.2)[1] - 8 * F(x, y, z - 0.1)[1] + 8 * F(x, y, z + 0.1)[1] -
                                 F(x, y, z + 0.2)[1])
                         + 2 * F(x, y, z)[2] * (5 / 6) * (
                                 F(x, y, z - 0.2)[2] - 8 * F(x, y, z - 0.1)[2] + 8 * F(x, y, z + 0.1)[2] -
                                 F(x, y, z + 0.2)[2])
                         ], dtype=float)


# Input functions:

def function_a(x: float, y: float, z: float):  # write the desired function
    return np.array([log(x ** 2 + y ** 2) - sin(x * y) - log(2) - log(pi), exp(x - y) + cos(x * y)])


def function_b(x: float, y: float, z: float):
    return np.array([3 * x - cos(y * z) - 0.5, x ** 2 - 625 * y ** 2 - 0.25, exp(-x * y) + 20 * z + (10 * pi - 3) / 3])


def g(vec: np.array, f):
    if len(vec) == 2:
        return f(vec[0], vec[1], 0)[0] ** 2 + f(vec[0], vec[1], 0)[1] ** 2
    else:
        return f(vec[0], vec[1], vec[2])[0] ** 2 + f(vec[0], vec[1], vec[2])[1] ** 2 + \
               f(vec[0], vec[1], vec[2])[2] ** 2


steepest_descent_numerical_grad(function_a, g, np.array([2.0, 2.0]), 10 ** -6)
