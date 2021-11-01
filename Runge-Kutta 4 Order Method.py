
import numpy as np
import matplotlib.pyplot as plt


# Runge-Kutta method of order 4, used to approximate the solution of the initial-value problem:
# y' = f(t,y), a <= t <= b, y(a) = alpha, at equally spaced nodes.
# The method also incorporates the plotting of the values on a graph,
# where it is shown at the end of the program

# PARAMETERS:
# func -> The function f(t,y) of the ODE
# a -> the right endpoint of t values (a <= t)
# alph -> initial value where y(a) = alpha
# b -> the left endpoint of t value (t <= b)
# h -> the step size for the method. for better approximations, use lower values
# of the form 10^(-k) where k is natural
def rk4_method(func, a: float, alph: float, b: float, h: float):
    N = int((b - a) / h)
    t = a
    w = alph
    t_array = np.array([t])  # this line and the next one are used to plot the approximating values. This is optional
    w_array = np.array([w])
    print([t, w])
    for i in range(1, N + 1):
        k1 = h * func(t, w)
        k2 = h * func(t + h / 2, w + k1 / 2)
        k3 = h * func(t + h / 2, w + k2 / 2)
        k4 = h * func(t + h, w + k3)
        w += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t = a + i * h
        t_array = np.append(t_array, t)  # this line and the next one are used to plot the approximating values.
                                         # this is optional.
        w_array = np.append(w_array, w)
        print([float('%g' % t), w])
    # the following is used to plot the approximating values. This is optional.
    # the h value is based on different values of h, and is manipulated by the user
    if h == 0.05:
        plt.plot(t_array, w_array, 'ro')
    else:
        plt.plot(t_array, w_array)
    return


# input function
def f(x: float, y: float):
    return 1 / (x ** 2) - y / x - y ** 2


rk4_method(f, 1, -1, 2, 0.5)
rk4_method(f, 1, -1, 2, 0.05)
# the following below is optional
t = np.linspace(1.0, 2.0)  # must enter (a,b)
plt.plot(t, -1 / t)  # solution for the ODE y' = f(t,y) where f is listed above.
plt.show()
