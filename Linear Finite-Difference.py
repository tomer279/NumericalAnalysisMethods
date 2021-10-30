import numpy as np


# Linear Finite-Difference method for approximating the solution to the boundary problem:
# -y'' + p(x)y' + q(x)y + r(x) = 0,  a <= x <= b, y(a) = alph, y(b) =beta
# over an equally spaced grid x[i]
# the method prints a list of the form [x[i],w[1,i]], where x[i] is the x value,
# and w[1,i] is the approximation to y(x[i]).

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
def linear_finite_diff(a: float, b: float, alph: float, beta: float, p, q, r, h: float):
    N = int((b - a) / h - 1)
    x = a + h
    a1 = 2 + (h ** 2) * q(x)
    b1 = -1 + (h / 2) * p(x)
    d1 = - (h ** 2) * r(x) + (1 + (h / 2) * p(x)) * alph
    a_arr = [0, a1]
    b_arr = [0, b1]
    c_arr = [0, 0]
    d_arr = [0, d1]
    for i in range(2, N):
        x = a + i * h
        a_arr.append(2 + (h ** 2) * q(x))
        b_arr.append(-1 + (h / 2) * p(x))
        c_arr.append(-1 - (h / 2) * p(x))
        d_arr.append(-(h ** 2) * r(x))
    x = b - h
    a_arr.append(2 + (h ** 2) * q(x))
    c_arr.append(-1 - (h / 2) * p(x))
    d_arr.append(-(h ** 2) * r(x) + (1 - (h / 2) * p(x)) * beta)
    # the following is used to solve a tridiagonal linear system
    l_arr = [0, a1]
    u_arr = [0, b1 / a1]
    z_arr = [0, d1 / l_arr[1]]
    for i in range(2, N):
        l_arr.append(a_arr[i] - c_arr[i] * u_arr[i - 1])
        u_arr.append(b_arr[i] / l_arr[i])
        z_arr.append((d_arr[i] - c_arr[i] * z_arr[i - 1]) / l_arr[i])
    l_arr.append(a_arr[N] - c_arr[N] * u_arr[N - 1])
    z_arr.append((d_arr[N] - c_arr[N] * z_arr[N - 1]) / l_arr[N])
    w_arr = [alph]
    for i in range(1,N):
        w_arr.append(0)
    w_arr.append(z_arr[N])
    w_arr.append(beta)
    for i in range(N - 1, 0, -1):
        w_arr[i] = z_arr[i] - u_arr[i] * w_arr[i+1]
    for i in range(0, N + 2):
        x = a+i*h
        print([float('%g' % x), w_arr[i] , np.abs(np.exp(-10*x) - w_arr[i])])
    return


def p_func(x: float):
    return 0.0


def q_func(x: float):
    return 100.0


def r_func(x: float):
    return 0.0


linear_finite_diff(0.0, 1.0, 1.0, np.exp(-10), p_func, q_func, r_func, 0.1)
