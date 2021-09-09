import numpy as np


def rkf_method(f, a: float, b: float, alph: float, tol: float, hmax: float, hmin: float):
    t = a
    w = alph
    h = hmax
    flag = 1
    print("Initial value:", np.array([t, w]))
    while flag == 1:
        k1 = h * f(t, w)
        k2 = h * f(t + (1 / 4) * h, w + (1 / 4) * k1)
        k3 = h * f(t + (3 / 8) * h, w + (3 / 32) * k1 + (9 / 32) * k2)
        k4 = h * f(t + (12 / 13) * h, w + (1932 / 2197) * k1 - (7200 / 2197) * k2 + (7296 / 2197) * k3)
        k5 = h * f(t + h, w + (439 / 216) * k1 - 8 * k2 + (3680 / 513) * k3 - (845 / 4104) * k4)
        k6 = h * f(t + (1 / 2) * h,
                   w - (8 / 27) * k1 + 2 * k2 - (3544 / 2565) * k3 + (1859 / 4104) * k4 - (11 / 40) * k5)
        R = (1 / h) * np.abs((1 / 360) * k1 - (128 / 4275) * k3 - (2197 / 75240) * k4 + (1 / 50) * k5 + (2 / 55) * k6)
        if R <= tol:
            t = t + h
            w = w + (25 / 216) * k1 + (1408 / 2565) * k3 + (2197 / 4104) * k4 - (1 / 5) * k5
            print(np.array([t, w, h]))
        delta = 0.84 * ((tol / R) ** 0.25)
        if delta <= 0.1:
            h = 0.1 * h
        elif delta >= 4:
            h = 4 * h
        else:
            h = delta * h
        if h > hmax:
            h = hmax
        if t >= b:
            flag = 0
        elif t + h > b:
            h = b - t
        elif h < hmin:
            flag = 0
            print('minimum h exceeded')
    return


def func(t: float, y: float):
    return -1 * ((y/t) ** 2) + (y/t)


rkf_method(func, 1, 4, 1, 10**-6, 0.5, 0.05)
