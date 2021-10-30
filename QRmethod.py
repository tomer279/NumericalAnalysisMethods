# Created by Tomer Caspi

import numpy as np
import sys


# QR method to obtain the eigenvalues of a symmetric, tridiagonal matrix.
# The code is based on algorithm 9.6 from the book "Numerical Analysis" by Burden, 10th Ed

# Parameters:
# A -> a symmetric and tridiagonal matrix
# tol -> the tolerance for the method


def qr_method(A: np.matrix, tol: float):
    if A != A.T:
        print("Matrix is not symmetric")
        return
    a_vec = np.array([0], dtype=float)  # The vector which contains all the diagonal elements of A
    b_vec = np.array([0, 0], dtype=float)  # The vector which contains the upper diagonal elements of A
    n = len(A)
    for i in range(0, n):
        a_vec = np.append(a_vec, A[i, i])
        if i != n - 1:
            b_vec = np.append(b_vec, A[i, i + 1])  # defining the "b" elements
    shift = 0  # accumulated shift
    for i in range(1, sys.maxsize):  # no k is defined. The method continues up until we reach final result.
        if np.abs(b_vec[n]) <= tol:
            print(a_vec[n] + shift)
            n -= 1
        if np.abs(b_vec[2]) <= tol:
            print(a_vec[1] + shift)
            n -= 1
            a_vec[1] = a_vec[2]
            for j in range(2, n + 1):
                a_vec[j] = a_vec[j + 1]
                b_vec[j] = b_vec[j + 1]
        if n == 0:
            return
        if n == 1:
            print(a_vec[1] + shift)
            return
        for j in range(3, n):
            if abs(b_vec[j]) <= tol:
                print("Split into", [a_vec[k] for k in range(1, j)], [b_vec[k] for k in range(2, j)])
                print("and", [a_vec[k] for k in range(j, n + 1)], [b_vec[k] for k in range(j + 1, n + 1)])
                return
        # lines 46 - 48 compute shift
        b = -(a_vec[n - 1] + a_vec[n])
        c = a_vec[n] * a_vec[n - 1] - b_vec[n] ** 2
        d = (b ** 2 - 4 * c) ** 0.5
        if b > 0:
            mu_1 = (-2 * c) / (b + d)
            mu_2 = (-b + d) / 2
        else:
            mu_1 = (d - b) / 2
            mu_2 = 2 * c / (d - b)
        if n == 2:
            print(mu_1 + shift)
            print(mu_2 + shift)
            return
        if np.abs(mu_1 - a_vec[n]) < np.abs(mu_2 - a_vec[n]):  # choosing sig such that
            # |sig - a[n] | = min{|mu_1 - a[n]|,|mu_2 - a[n]|}
            sig = mu_1
        else:
            sig = mu_2
        shift += sig  # accumulate the shift
        d_vec = np.array([0])
        for j in range(1, n + 1):  # perform shift
            d_vec = np.append(d_vec, a_vec[j] - sig)
        x_1 = d_vec[1]
        y_1 = b_vec[2]
        z_vec = np.array([0])
        c_vec = np.array([0, 0])
        sig_vec = np.array([0, 0])
        q_vec = np.array([0])
        x_vec = np.array([0, x_1])
        y_vec = np.array([0, y_1])
        r_vec = np.array([0])
        for j in range(2, n + 1):
            z_vec = np.append(z_vec, (x_vec[j - 1] ** 2 + b_vec[j] ** 2) ** 0.5)
            c_vec = np.append(c_vec, x_vec[j - 1] / z_vec[j - 1])
            sig_vec = np.append(sig_vec, b_vec[j] / z_vec[j - 1])
            q_vec = np.append(q_vec, c_vec[j] * y_vec[j - 1] + sig_vec[j] * d_vec[j])
            x_vec = np.append(x_vec, -sig_vec[j] * y_vec[j - 1] + c_vec[j] * d_vec[j])
            if j != n:
                r_vec = np.append(r_vec, sig_vec[j] * b_vec[j + 1])
                y_vec = np.append(y_vec, c_vec[j] * b_vec[j + 1])
        z_vec = np.append(z_vec, x_vec[n])
        a_vec[1] = sig_vec[2] * q_vec[1] + c_vec[2] * z_vec[1]
        b_vec[2] = sig_vec[2] * z_vec[2]
        for j in range(2, n):
            np.put(a_vec, j, sig_vec[j + 1] * q_vec[j] + c_vec[j] * c_vec[j + 1] * z_vec[j])
            np.put(b_vec, j + 1, sig_vec[j + 1] * z_vec[j + 1])
        np.put(a_vec, n, c_vec[n] * z_vec[n])


# input parameters:
mat = np.matrix([[1, 1, 0, 0],
               [1, 2, -1, 0],
               [0, -1, 3, 1],
               [0, 0, 1, 4]])
TOL = 10**-2
qr_method(mat, TOL)
