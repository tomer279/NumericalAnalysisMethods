import numpy as np
from sympy import *


def __rayleigh_ritz__(p, q, f, n: int):
    t = Symbol('t')
    h = float(1 / (n + 1))
    x = [round(h * i, 1) for i in range(n + 2)]
    Q = np.zeros((7, n + 3))
    alph_arr = [0]
    beta_arr = [0]
    b_arr = [0]
    a_arr = [0]
    zeta_arr = [0]
    z_arr = [0]
    for i in range(1, n):
        Q[1, i] = ((1 / h) ** 2) * integrate((x[i + 1] - t) * (t - x[i]) * q(t), (t, x[i], x[i + 1]))
        Q[2, i] = ((1 / h) ** 2) * integrate(((t - x[i - 1]) ** 2) * q(t), (t, x[i - 1], x[i]))
        Q[3, i] = ((1 / h) ** 2) * integrate(((x[i + 1] - t) ** 2) * q(t), (t, x[i], x[i + 1]))
        Q[4, i] = ((1 / h) ** 2) * integrate(p(t), (t, x[i - 1], x[i]))
        Q[5, i] = (1 / h) * integrate((t - x[i - 1]) * f(t), (t, x[i - 1], x[i]))
        Q[6, i] = (1 / h) * integrate((x[i + 1] - t) * f(t), (t, x[i], x[i + 1]))
    Q[2, n] = ((1 / h) ** 2) * integrate(((t - x[n - 1]) ** 2) * q(t), (t, x[n - 1], x[n]))
    Q[3, n] = ((1 / h) ** 2) * integrate(((x[n + 1] - t) ** 2) * q(t), (t, x[n], x[n + 1]))
    Q[4, n] = ((1 / h) ** 2) * integrate(p(t), (t, x[n - 1], x[n]))
    Q[5, n] = (1 / h) * integrate((t - x[n - 1]) * f(t), (t, x[n - 1], x[n]))
    Q[6, n] = (1 / h) * integrate((x[n + 1] - t) * f(t), (t, x[n], x[n + 1]))
    Q[4, n + 1] = ((1 / h) ** 2) * integrate(p(t), (t, x[n - 1], x[n]))
    for i in range(1, n):
        alph_arr.append(Q[4, i] + Q[4, i + 1] + Q[2, i] + Q[3, i])
        beta_arr.append(Q[1, i] - Q[4, i + 1])
        b_arr.append(Q[5, i] + Q[6, i])
    alph_arr.append(Q[4, n] + Q[4, n + 1] + Q[2, n] + Q[3, n])
    b_arr.append(Q[5, n] + Q[6, n])
    a_arr.append(alph_arr[1])
    zeta_arr.append(beta_arr[1] / alph_arr[1])
    z_arr.append(b_arr[1] / a_arr[1])
    for i in range(2, n):
        a_arr.append(alph_arr[i] - beta_arr[i - 1] * zeta_arr[i - 1])
        zeta_arr.append(beta_arr[i] / a_arr[i])
        z_arr.append((b_arr[i] - beta_arr[i - 1] * z_arr[i - 1]) / a_arr[i])
    a_arr.append(alph_arr[n] - beta_arr[n - 1] * zeta_arr[n - 1])
    z_arr.append((b_arr[n] - beta_arr[n - 1] * z_arr[n - 1]) / a_arr[n])
    c_arr = [0] * (n + 1)
    c_arr[n] = z_arr[n]
    # print([x[n], c_arr[n]])
    for i in range(n - 1, 0, -1):
        c_arr[i] = z_arr[i] - zeta_arr[i] * c_arr[i + 1]
    # print([x[i], c_arr[i]])
    return c_arr


def __linear_piecewise_basis__(n: int, t: float):
    h = float(1 / (n + 1))
    x = [round(h * i, 1) for i in range(n + 2)]
    phi = np.zeros(n + 1)
    for i in range(0, n + 1):
        if 0 <= t <= x[i - 1] or x[i + 1] < t <= 1:
            phi[i] = 0
        elif x[i - 1] < t <= x[i]:
            phi[i] = (1 / h) * (t - x[i - 1])
        elif x[i] < t <= x[i + 1]:
            phi[i] = (1 / h) * (x[i + 1] - t)
    return phi


def __calculate_linear_function__(p, q, f, n: int):
    h = float(1 / (n + 1))
    x = [round(h * i, 1) for i in range(n + 2)]
    c = __rayleigh_ritz__(p, q, f, n)
    psi = np.zeros(n + 2)
    sum = 0
    for i in range(1, n + 1):
        phi = __linear_piecewise_basis__(9, x[i])
        for j in range(0, n + 1):
            sum += c[j] * phi[j]
        psi[i] = sum
        sum = 0
    return psi


def calculate_differential_equation(p, q, f, n: int, alph: float, beta: float):
    h = float(1 / (n + 1))
    x = [round(h * i, 1) for i in range(n + 2)]
    psi = __calculate_linear_function__(p, q, f, n)
    y = [psi[i] + beta * x[i] + alph * (1 - x[i]) for i in range(0, n + 1)]
    # use w if the actual solution is known
    # w = [x[i] + np.exp(-x[i]) for i in range(0, n + 1)]
    for i in range(0, n + 1):
        print([i, x[i], y[i]])


def px(t):
    return 1


def qx(t):
    return 1


def fx(t):
    return t - t * exp(-1) - 1


calculate_differential_equation(px, qx, fx, 9, 1, 1 + np.exp(-1))
