# Created by Tomer Caspi

import numpy as np
import sys
import matplotlib.pyplot as plt


# Euler's method used for approximating a solution for the ODE y'(t) = f(t,y(t)),
# where  a <= t <= b and y(a) = alpha, at equally spaced nodes.
# The method returns an approximation vector of the form [t[i],w[i]), where t[i] is the t value,
# and w[i] is the approximation for y(t[i])
# The method also incorporates the plotting of the values on a graph,
# where it is shown at the end of the program

# PARAMETERS:
# func -> The function f(t,y) of the ODE
# a -> the right endpoint of t values (a <= t)
# alph -> initial value where y(a) = alpha
# b -> the left endpoint of t value (t <= b)
# h -> the step size for the method. for better approximations, use lower values
# of the form 10^(-k) where k is natural
def euler_method(func, a: float, alph: float, b: float, h: float):
    t = a
    w = alph
    t_array = np.array([t])
    w_array = np.array([w])
    print("initial values:", np.array([t, w]))
    for i in range(1, sys.maxsize):
        w = w + h * func(t, w)  # compute w[i]
        t = a + i * h  # compute t[i]
        if t > b:
            plt.plot(t_array, w_array)
            return
        t_array = np.append(t_array, t)
        w_array = np.append(w_array, w)
        print("iteration ", i, ":", np.array([t, w]))
    return


# input function
def f(x: float, y: float):
    return 1 / (x ** 2) - y / x - y ** 2


euler_method(f, 1, -1, 2, 0.5)
euler_method(f, 1, -1, 2, 0.05)
# the following below is optional
t = np.linspace(1.0, 2.0)  # must enter (a,b)
plt.plot(t, -1 / t)  # solution for the ODE y' = f(t,y) where f is listed above.
plt.show()
