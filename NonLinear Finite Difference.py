import numpy as np
from numpy import linalg as LA


# Nonlinear finite difference method for approximating the solution to the boundary problem:
# y'' = f(x,y,y'),  a <= x <= b, y(a) = alph, y(b) =beta
# over an equally spaced grid x[i]
# the method prints a list of the form [x[i],w[1,i]], where x[i] is the x value,
# and w[1,i] is the approximation to y(x[i])

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
# M -> maximum number of iterations

def nonlinear_finite_diff(a: float, b: float, alph: float, beta: float, f , f_y , f_ydiff , h: float, tol: float, M: int):
    N = int((b - a) / h - 1)
    w = [alph]
    for i in range(1, N + 1):
        w.append(alph + i * (beta - alph) / (b - a) * h)
    w.append(beta)
    for j in range(1, M):
        x = a + h
        t = (w[2] - alph) / (2 * h)
        a_arr = [0, 2 + (h ** 2) * f_y(x, w[1], t)]
        b_arr = [0, -1 + (h / 2) * (f_ydiff(x, w[1], t))]
        c_arr = [0, 0]
        d_arr = [0, -(2 * w[1] - w[2] - alph - (h ** 2) * f(x, w[1], t))]
        for i in range(2, N):
            x = a + i * h
            t = (w[i + 1] - w[i - 1]) / (2 * h)
            a_arr.append(2 + (h ** 2) * f_y(x, w[i], t))
            b_arr.append(-1 + (h / 2) * f_ydiff(x, w[i], t))
            c_arr.append(-1 - (h / 2) * f_ydiff(x, w[i], t))
            d_arr.append(-(2 * w[i] - w[i + 1] - w[i - 1] + (h ** 2) * f(x, w[i], t)))
        x = b - h
        t = (beta - w[N - 1]) / (2 * h)
        a_arr.append(2 + (h ** 2) * f_y(x, w[N], t))
        c_arr.append(-1 - (h / 2) * f_ydiff(x, w[N], t))
        d_arr.append(-(2 * w[N] - w[N - 1] - beta + (h ** 2) * f(x, w[N], t)))
        # solving a tridiagonal linear system
        l_arr = [0, a_arr[1]]
        u_arr = [0, b_arr[1] / a_arr[1]]
        z_arr = [0, d_arr[1] / l_arr[1]]
        for i in range(2, N):
            l_arr.append(a_arr[i] - c_arr[i] * u_arr[i - 1])
            u_arr.append(b_arr[i] / l_arr[i])
            z_arr.append((d_arr[i] - c_arr[i] * z_arr[i - 1]) / l_arr[i])
        l_arr.append(a_arr[N] - c_arr[N] * u_arr[N - 1])
        z_arr.append((d_arr[N] - c_arr[N] * z_arr[N - 1]) / l_arr[N])
        v_arr = np.zeros(N)
        v_arr = np.append(v_arr, z_arr[N])
        w[N] += v_arr[N]
        for i in range(N - 1, 0, -1):
            v_arr[i] = z_arr[i] - u_arr[i] * v_arr[i + 1]
            w[i] = w[i] + v_arr[i]
        if LA.norm(v_arr, np.inf) <= tol:
            for i in range(0, N + 2):
                x = a + i * h
                print([float('%g' % x), w[i],1/(x+1) , np.abs(w[i]-1/(x+1))])
            return
    print("Maximum number of iterations exceeded")
    return


# input functions:


def f_func(x: float, y: float, z: float):
    return y ** 3 - y * z


def f_y_func(x: float, y: float, z: float):
    return 3 * (y ** 2) - z


def f_ydiff_func(x: float, y: float, z: float):
    return -y


nonlinear_finite_diff(1.0, 2.0, 1 / 2, 1 / 3, f_func,f_y_func,f_ydiff_func, 0.1, 10 ** -4, 1000)
