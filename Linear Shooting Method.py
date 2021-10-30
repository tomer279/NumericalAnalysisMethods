# Created by Tomer Caspi

import numpy as np


# Linear shooting method with Newton's method for approximating the solution to the boundary problem:
# -y'' + p(x)y' + q(x)y + r(x) = 0,  a <= x <= b, y(a) = alph, y(b) =beta
# over an equally spaced grid x[i]
# the method prints a list of the form [x[i],w[1,i],w[2,i]], where x[i] is the x value,
# w[1,i] is the approximation to y(x[i]), and w[2,i] is the approximation to y'(x[i])

# PARAMETERS:
# a-> the right endpoint of a <= x <= b
# b-> the left endpoint f a <= x <= b
# alph -> initial value y(a) = alph
# beta -> initial value y(b) = beta
# p -> the function p(x) of the problem
# q -> the function q(x) of the problem
# r -> the function r(x) of the problem
# h -> the step-size. for better approximations, choose lower values. suggested values at 10**(-k)
# where k is natural

def linear_shooting(a: float, b: float, alph: float, beta: float, p, q, r, h: float):
    N = int((b - a) / h)
    u_10 = alph
    u_20 = 0
    v_10 = 0
    v_20 = 1
    u = np.matrix([[0], [u_10], [u_20]])
    v = np.matrix([[0], [v_10], [v_20]])
    for i in range(0, N):  # Runge-Kutta method for systems
        x = a + i * h
        k_11 = h * u[2, i]
        k_12 = h * (p(x) * u[2, i] + q(x) * u[1, i] + r(x))
        k_21 = h * (u[2, i] + 0.5 * k_12)
        k_22 = h * (p(x + h / 2) * (u[2, i] + 0.5 * k_12) + q(x + h / 2) * (u[1, i] + 0.5 * k_11) + r(
            x + h / 2))
        k_31 = h * (u[2, i] + 0.5 * k_22)
        k_32 = h * (p(x + h / 2) * (u[2, i] + 0.5 * k_22) + q(x + h / 2) * (u[1, i] + 0.5 * k_21) + r(
            x + h / 2))
        k_41 = h * (u[2, i] + k_32)
        k_42 = h * (p(x + h) * (u[2, i] + k_32) + q(x + h) * (u[1, i] + k_31) + r(x + h))
        u_new1 = u[1, i] + (1 / 6) * (k_11 + 2 * k_21 + 2 * k_31 + k_41)
        u_new2 = u[2, i] + (1 / 6) * (k_12 + 2 * k_22 + 2 * k_32 + k_42)
        u = np.append(u, [[0], [u_new1], [u_new2]], axis=1)
        l_11 = h * v[2, i]
        l_12 = h * (p(x) * v[2, i] + q(x) * v[1, i])
        l_21 = h * (v[2, i] + 0.5 * l_12)
        l_22 = h * (p(x + h / 2) * (v[2, i] + 0.5 * l_12) + q(x + h / 2) * (v[1, i] + 0.5 * l_11))
        l_31 = h * (v[2, i] + 0.5 * l_22)
        l_32 = h * (p(x + h / 2) * (v[2, i] + 0.5 * l_22) + q(x + h / 2) * (v[1, i] + 0.5 * l_21))
        l_41 = h * (v[2, i] + l_32)
        l_42 = h * (p(x + h) * (v[2, i] + l_32) + q(x + h) * (v[1, i] + l_31))
        v_new1 = v[1, i] + (1 / 6) * (l_11 + 2 * l_21 + 2 * l_31 + l_41)
        v_new2 = v[2, i] + (1 / 6) * (l_12 + 2 * l_22 + 2 * l_32 + l_42)
        v = np.append(v, [[0], [v_new1], [v_new2]], axis=1)
    w_10 = alph
    w_20 = (beta - u[1, N]) / v[1, N]
    # it can be optional to print the absolute / relative error if the solution is known.
    # if not, you can remove it.
    print([a, w_10,w_20, np.abs(np.exp(-10 * a) - w_10)])
    for i in range(1, N + 1):
        W1 = u[1, i] + w_20 * v[1, i]
        W2 = u[2, i] + w_20 * v[2, i]
        x = a + i * h
        # it can be optional to print the absolute / relative error if the solution is known.
        # if not, you can remove it.
        print([float('%g' % x), W1, W2, np.abs(np.exp(-10 * x) - W1)])
    return


# Input functions:

def p_func(x: float):
    return 0.0


def q_func(x: float):
    return 100.0


def r_func(x: float):
    return 0.0


linear_shooting(0.0, 1.0, 1.0, np.exp(-10), p_func, q_func, r_func, 0.1)
