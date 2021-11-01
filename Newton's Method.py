
import numpy as np
from numpy import linalg as LA
from numpy import log, sin, cos, exp, pi

np.set_printoptions(suppress=True)


# Newton's method for solving a non-linear system F(x) = 0, where F is a function from R^n to R^n,
# x is a vector of R^n, and 0 is the zero vector of R^n, and n = 2 or n = 3.
# The method returns a vector approximation to the system which has an infinity norm less than the tolerance
# The method is based on algorithm 10.1 from the book "Numerical Analysis" by Burden, 10th Ed.

# PARAMETERS:
# F -> The function which creates the non-linear system. F can be of length 2 or 3.
# J -> THe jacobian of the function F. The user must calculate the jacobian first manually.
# vec -> the initial approximation vector. the vector must be of length 2 or 3, and have the same length as F
# tol -> the tolerance for the method, such that an approximation returns when ||vec||_(inf) < tol.
# tolerance should have values 10**(-k), where k is a natural number
# N -> the number of maximum iterations.


def non_linear_newton(F: np.array, J: np.array, vec: list, tol: float, N: int):
    for i in range(1, N + 1):
        if len(vec) == 2:
            u = F(vec[0], vec[1], 0)
            A = J(vec[0], vec[1], 0)
        elif len(vec) == 3:
            u = F(vec[0], vec[1], vec[2])
            A = J(vec[0], vec[1], vec[2])
        else:
            print("Only 2x2 or 3x3 systems are allowed")
            return
        y = LA.solve(A, -u)  # solving the linear system J(vec)*y = -F(vec)
        vec = vec + y
        if LA.norm(y, np.inf) < tol:  # procedure is successful
            print("Newtons method: the number of iterations:", i)
            return vec
        print("iteration", i, ":", vec)  # this command can be removed if you want to see only the final result
    print("maximum number of iterations exceeded")  # procedure is unsuccessful
    return


# Writing down the function F: the function can have 2 or 3 variables.
# The length of the array must be the same as the number of variables.
# The jacobian for the function must be written manually by the user.

def function_a(x: float, y: float, z: float):
    return np.array([log(x ** 2 + y ** 2) - sin(x * y) - log(2) - log(pi),
                     exp(x - y) + cos(x * y)])


def jacobian_a(x: float, y: float, z: float):  # write the jacobian of the function
    return np.matrix([[(2 * x) / (x ** 2 + y ** 2) - y * cos(x * y), (2 * y) / (x ** 2 + y ** 2) - x * cos(x * y)],
                      [exp(x - y) - y * sin(x * y), -exp(x - y) - x * sin(x * y)]])


def function_b(x: float, y: float, z: float):
    return np.array([3 * x - cos(y * z) - 0.5,
                     x ** 2 - 625 * y ** 2 - 0.25,
                     exp(-x * y) + 20 * z + (10 * pi - 3) / 3])


def jacobian_b(x: float, y: float, z: float):
    return np.matrix([[3, z * sin(y * z), y * sin(y * z)]
                         , [2 * x, -1250 * y, 0]
                         , [-y * exp(-x * y), -x * exp(-x * y), 20]])


print(non_linear_newton(function_b, jacobian_b, [1.0, 1.0, 0.0], 10 ** -3, 400))
