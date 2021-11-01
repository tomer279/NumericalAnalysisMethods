import numpy as np


# Nonlinear shooting method for approximating the solution to the boundary problem:
# y'' = f(x,y,y'),  a <= x <= b, y(a) = alph, y(b) =beta
# over an equally spaced grid x[i]
# the method prints a list of the form [x[i],w[1,i],w[2,i]], where x[i] is the x value,
# w[1,i] is the approximation to y(x[i]), and w[2,i] is the approximation to y'(x[i])

# PARAMETERS:
# a-> the right endpoint of a <= x <= b
# b-> the left endpoint f a <= x <= b
# alph -> initial value y(a) = alph
# beta -> initial value y(b) = beta
# f -> the function f of the problem
# f_y -> the derivative of f on y
# f_ydiff -> the derivative of f on y'
# h -> the step-size. for better approximations, choose lower values. suggested values at 10**(-k)
# tol -> the tolerance for which |w[1,N]-beta| <= tol
def non_linear_shooting(a: float, b: float, alph: float, beta: float, f, f_y, f_ydiff, h: float, tol: float, M: int):
    N = int((b - a) / h)
    tk = (beta - alph) / (b - a)  # slope of the straight line through (a,alph) and (b,beta)
    for j in range(1, M):
        w = np.matrix([[0.0], [alph], [tk]])
        u = [0, 0.0, 1.0]
        for i in range(1, N + 1):  # Runge-Kutta method for systems
            x = a + (i - 1) * h
            k_11 = h * w[2, i - 1]
            k_12 = h * f(x, w[1, i - 1], w[2, i - 1])
            k_21 = h * (w[2, i - 1] + 0.5 * k_12)
            k_22 = h * f(x + h / 2, w[1, i - 1] + 0.5 * k_11, w[2, i - 1] + 0.5 * k_12)
            k_31 = h * (w[2, i - 1] + 0.5 * k_22)
            k_32 = h * f(x + h / 2, w[1, i - 1] + 0.5 * k_21, w[2, i - 1] + 0.5 * k_22)
            k_41 = h * (w[2, i - 1] + k_32)
            k_42 = h * f(x + h, w[1, i - 1] + k_31, w[2, i - 1] + k_32)
            w_new1 = w[1, i - 1] + (1 / 6) * (k_11 + 2 * k_21 + 2 * k_31 + k_41)
            w_new2 = w[2, i - 1] + (1 / 6) * (k_12 + 2 * k_22 + 2 * k_32 + k_42)
            w = np.append(w, [[0], [w_new1], [w_new2]], axis=1)
            l_11 = h * u[2]
            l_12 = h * (f_y(x, w[1, i - 1], w[2, i - 1]) * u[1] + f_ydiff(x, w[1, i - 1], w[2, i - 1]) * 2)
            l_21 = h * (u[2] + 0.5 * l_12)
            l_22 = h * (f_y(x + h / 2, w[1, i - 1], w[2, i - 1]) * (u[1] + 0.5 * l_11)
                        + f_ydiff(x + h / 2, w[1, i - 1], w[2, i - 1]) * (u[2] + 0.5 * l_12))
            l_31 = h * (u[2] + 0.5 * l_22)
            l_32 = h * (f_y(x + h / 2, w[1, i - 1], w[2, i - 1]) * (u[1] + 0.5 * l_21)
                        + f_ydiff(x + h / 2, w[1, i - 1], w[2, i - 1]) * (u[2] + 0.5 * l_22))
            l_41 = h * (u[2] + l_32)
            l_42 = h * (f_y(x + h, w[1, i - 1], w[2, i - 1]) * (u[1] + l_31)
                        + f_ydiff(x + h, w[1, i - 1], w[2, i - 1]) * (u[2] + l_32))
            u[1] += (1 / 6) * (l_11 + 2 * l_21 + 2 * l_31 + l_41)
            u[2] += (1 / 6) * (l_12 + 2 * l_22 + 2 * l_32 + l_42)
        if np.abs(w[1, N] - beta) <= tol:  # procedure is complete
            for i in range(0, N + 1):
                x = a + i * h
                print([x, w[1, i], w[2, i - 1], np.abs(w[1, i] - 1 / (x + 3))])
            return
        tk -= (w[1, N] - beta) / u[1]  # Newton's method for computing tk
    print("Maximum number of iterations exceeded") # procedure is unsuccessful
    return


# input functions:


def func(x: float, y: float, z: float):
    return 2 * (y ** 3)


def func_y(x: float, y: float, z: float):
    return 6 * (y ** 2)


def func_ydiff(x: float, y: float, z: float):
    return 0.0


non_linear_shooting(-1.0, 0.0, 1 / 2, 1 / 3, func, func_y, func_ydiff, 0.25, 10 ** -4, 1000)
