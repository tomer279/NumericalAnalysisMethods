import numpy as np
from math import factorial
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)


# Taylor method to solve an ODE y'(x) = f(x,y), such that a<=x<=b, y(a) = alph.
# The method returns a printed scale [t_i ,w_i] such that t is the x values, and w_i is the approximation
# for y(t_i)
# in order for the method to work, the user needs to enter manually the function f and the desired derivatives
# such that (d/dx)f(x,y,y').
# after writing the functions, the user needs to insert them by order of the derivatives in the "func_list" variable
def func(x: float, y: float):
    return 1 / (x ** 2) - y / x - y ** 2


def func_diff(x: float, y: float):
    return (-3 / x ** 3) + (3 * y ** 2) / x + 2 * y ** 3


def func_diff2(x: float, y: float):
    return (9 / (x ** 4)) + (6 * y / (x ** 3)) - (3 * (y ** 2)) / (x ** 2) - 12 * (y ** 3) / x - 6 * (y ** 4)


func_list = [func, func_diff, func_diff2]  # write the above functions by order


# PARAMETERS:
# a -> the left endpoint for which a <= x <= b
# alph -> the initial value y(a) = alph
# b -> the right endpoint for which a <= x <= b
# h -> the step size
# f_diff -> function list which includes the function f and its derivatives by x.
def taylor_method(a: float, alph: float, b: float, h: float, f_diff: list):
    N = int((b - a) / h)
    w = np.zeros(N + 1)
    t = np.zeros(N + 1)
    n = len(f_diff)
    w[0] = alph
    t[0] = a
    print("initial values:", [t[0], w[0]])
    for i in range(0, N):
        T = 0
        for j in range(0, n):
            T += ((h ** j) / factorial(j)) * f_diff[j](t[i], w[i])
        t[i + 1] = a + (i + 1) * h
        w[i + 1] = w[i] + h * T
        print("iteration ", i + 1, ":", np.array([t[i + 1], w[i + 1]]))
    plt.plot(t, w, label="h = " + str(h))
    return


# input parameters:
a_val = 1.0
alph_val = -1.0
b_val = 2.0
h_val = 0.5
h_val2 = 0.05
n_val = 2
taylor_method(a_val, alph_val, b_val, h_val, func_list)
taylor_method(a_val, alph_val, b_val, h_val2, func_list)
# the code above is meant the plotting of the method
t = np.linspace(a_val, b_val)  # must enter (a,b)
plt.plot(t, -1 / t, label="y(t) = -1/t")  # solution for the ODE y' = f(t,y) where f is listed above.
plt.legend()
plt.title("Taylor method")
plt.show()
