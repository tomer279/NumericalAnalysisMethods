import numpy as np
from numpy import linalg as LA


def broyden(F, J, x: float, y: float, z: float, tol: float):
    A0 = J(x, y, z)
    v = F(x, y, z)
    if v.shape[0] == 2:
        vec = np.array([x, y])
    if v.shape[0] == 3:
        vec = np.array([x, y, z])
    A = LA.inv(A0)
    s = np.ravel(-v.dot(A))
    vec = vec + s
    k = 2
    while k <= np.iinfo(np.int32).max:
        w = v
        if vec.shape[0] == 2:
            v = F(vec[0], vec[1], 0)
        if vec.shape[0] == 3:
            v = F(vec[0], vec[1], vec[2])
        y = v - w
        z = -1 * np.ravel(y.dot(A))
        p = -1 * np.ravel(s.dot(z))
        u = np.ravel(s.dot(A.T))
        if p < 10**-6:
            return vec
        A = A + (1 / p) * np.ravel((s + z).dot(u))
        s = -np.ravel(v.dot(A))
        vec = vec + s
        if LA.norm(s) < tol:
            return vec
        k = k + 1
    print("The procedure was unsuccessful")
    return


def function(x: float, y: float, z: float):  # write the desired function
    return np.array(
        [3 * x - np.cos(y * z) - 0.5,
         x ** 2 - 81 * (y + 0.1) ** 2 + np.sin(z) + 1.06,
         np.e ** (-x * y) + 20 * z + (10 * np.pi - 3) / 3])


def jacobian(x: float, y: float, z: float):
    return np.matrix(
        [[3, z * np.sin(y*z) , y * np.sin(y*z)],
         [2*x , -162*(y+0.1) , np.cos(z)],
         [-y * np.e**(-y*x) , -x*np.e**(-x*y), 20]])


print(broyden(function, jacobian, 0.1, 0.1, -0.1, 10 ** -1))
