import numpy as np
import sys
import matplotlib.pyplot as plt


def euler_method(func, a: float, alph: float, b: float, h: float):
    t = a
    w = alph
    t_array = np.array([t])
    w_array = np.array([w])
    print("initial values:", np.array([t, w]))
    for i in range(1, sys.maxsize):
        w = w + h * func(t, w)
        t = a + i * h
        if t > b:
            plt.plot(t_array,w_array)
            return
        t_array = np.append(t_array , t)
        w_array = np.append(w_array , w)
        print("iteration ", i, ":", np.array([t, w]))
    return


def func(t: float, y: float):
    return 1 / (t ** 2) - y / t - y ** 2


euler_method(func,1,-1,2,0.5)
euler_method(func, 1, -1, 2, 0.05)
t= np.linspace(1.0,2.0)
plt.plot(t, -1/t)
plt.show()