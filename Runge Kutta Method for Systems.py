# created by Tomer Caspi

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)


# Runge-Kutta method for numerically solving a 3th-order system of ordinary differential equations:
# x'(t) = f_1(x,y,z,t)
# y'(t) = f_2(x,y,z,t)
# z'(t) = f_3(x,y,z,t)
# for a <= t <=b, and with initial conditions:
# x(a) = alph_1 , y(a) = alpha_2 , z(a) = alph_3
# the method returns a printed table of the approximations to x,y,z at (h/b) values of t, with the form:
# [t,x(t),y(t),z(t)]
# the method also creates a plot based on the table above.
# The code is based on algorithm 5.7 from the book "Numerical Analysis" by Burden, 10th Ed.


# PARAMETERS:
# f_1, f_2,f_3 -> the functions which make the system listed above.
# u0 -> the initial conditions vector, such that u0_j(a) = alpha_j
# h -> the step size for the method. for better results, enter small values of 10**(-k)
# b -> the left endpoint of the t variable for the system. the right endpoint of the system is 0.
def rk_system(f_1, f_2, f_3, u0: list, h: float, b: float):
    N = int(b / h)
    t = 0
    w = np.array([t])  # the approximation array
    # the following is for the creation of the plot, marking the x,y,z values at t:
    x = np.array([])
    y = np.array([])
    z = np.array([])
    for j in range(1, 4):
        w = np.append(w, u0[j - 1])  # adding the initial conditions
    print(w)
    for i in range(1, N + 1):
        k11 = h * f_1(w[1], w[2], w[3], t)
        k12 = h * f_2(w[1], w[2], w[3], t)
        k13 = h * f_3(w[1], w[2], w[3], t)
        k21 = h * f_1(w[1] + 0.5 * k11, w[2] + 0.5 * k12, w[3] + 0.5 * k13, t + h / 2)
        k22 = h * f_2(w[1] + 0.5 * k11, w[2] + 0.5 * k12, w[3] + 0.5 * k13, t + h / 2)
        k23 = h * f_3(w[1] + 0.5 * k11, w[2] + 0.5 * k12, w[3] + 0.5 * k13, t + h / 2)
        k31 = h * f_1(w[1] + 0.5 * k21, w[2] + 0.5 * k22, w[3] + 0.5 * k23, t + h / 2)
        k32 = h * f_2(w[1] + 0.5 * k21, w[2] + 0.5 * k22, w[3] + 0.5 * k23, t + h / 2)
        k33 = h * f_3(w[1] + 0.5 * k21, w[2] + 0.5 * k22, w[3] + 0.5 * k23, t + h / 2)
        k41 = h * f_1(w[1] + k31, w[2] + k32, w[3] + k33, t + h)
        k42 = h * f_2(w[1] + k31, w[2] + k32, w[3] + k33, t + h)
        k43 = h * f_3(w[1] + k31, w[2] + k32, w[3] + k33, t + h)
        w[1] += (k11 + 2 * k21 + 2 * k31 + k41) / 6.0
        w[2] += (k12 + 2 * k22 + 2 * k32 + k42) / 6.0
        w[3] += (k13 + 2 * k23 + 2 * k33 + k43) / 6.0
        t = i * h
        w[0] = t
        # print(w)   # this command can be removed if you want the plot to show quicker
        x = np.append(x, w[1])
        y = np.append(y, w[2])
        z = np.append(z, w[3])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(x, y, z)
    plt.show()
    return


# functions x,y,z for the methods:

# Lorentz system:
def func_1(x: float, y: float, z: float, t: float):
    return 10 * (y - x)


def func_2(x: float, y: float, z: float, t: float):
    return x * (27 - z) - y


def func_3(x: float, y: float, z: float, t: float):
    return x * y - (8 / 3) * z


# Rossler System:
def ros_1(x: float, y: float, z: float, t: float):
    return -y - z


def ros_2(x: float, y: float, z: float, t: float):
    return x + 0.1 * y


# you can change the c parameter in (x-c) for different results (values between 4-16 are preferred)
def ros_3(x: float, y: float, z: float, t: float):
    return 0.1 + z * (x - 9)


# running the method:
rk_system(func_1, func_2, func_3, [2.0, 5.0, 0.1], 0.01, 200)
rk_system(ros_1, ros_2, ros_3, [2.0, 5.0, 0.1], 0.01, 200)
