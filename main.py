# Michał Piekarski 175456

import numpy as np
import numpy.linalg as la
import time

"""
    function that calculates system of linear equations using Jacobi method
    @param A matrix of coefficients
    @param b vector of free variables
    @param size size of square matrix and vector
    @return tuple of: solution vector, iterations needed to convergence
"""


def Jacobi(A, b, size):
    x = np.ones(shape=[size, 1])   # vector of approximated solution
    norm = la.norm(np.dot(A, x) - b)
    L = np.tril(A, -1)  # lower triangle A
    U = np.triu(A, 1)   # upper triangle A
    D = np.diag(np.diag(A))
    Db = la.solve(D, b)  # constant part of next solution
    DLU = la.solve(-D, L + U)   # constant part of next solution
    error = 10 ** -9
    iter = 0

    start = time.time()
    while norm > error and iter < 1000:
        x = np.dot(DLU, x) + Db
        norm = la.norm(np.dot(A, x) - b)
        iter += 1
    end = time.time()
    return x, iter, end - start


"""
    function that calculates system of linear equations using Jacobi method
    @param A matrix of coefficients
    @param b vector of free variables
    @param size size of square matrix and vector
    @return tuple of: solution vector, iterations needed to convergence
"""


def Gauss_Seidel(A, b, size):
    x = np.ones(shape=[size, 1])   # vector of approximated solution
    norm = la.norm(np.dot(A, x) - b)
    L = np.tril(A, -1)  # lower triangle A
    U = np.triu(A, 1)   # upper triangle A
    D = np.diag(np.diag(A))
    DLb = la.solve(D + L, b)  # constant part of next solution
    error = 10 ** -9
    iter = 0

    start = time.time()
    while norm > error and iter < 1000:
        x = la.solve(-(D + L), np.dot(U, x)) + DLb
        norm = la.norm(np.dot(A, x) - b)
        print(norm)
        iter += 1
    end = time.time()
    return x, iter, end - start


c = 5   # penultimate number of index
d = 6   # last number of index
e = 4   # 4th number of index
f = 5   # 3rd number of index

# task A data
a1 = 5 + e
a2 = a3 = -1

N = 900 + 10 * c + d   # size of matrix A

A = np.zeros(shape=[N, N])
b = np.empty(shape=[N, 1])

# creation of band matrix A and vector B
for i in range(N):
    A[i][i] = a1  # creation of diagonal
    b[i] = np.sin((i + 1) * (f + 1))  # angle in radians
    if i < N - 1:
        A[i + 1][i] = a2
    if i < N - 2:
        A[i + 2][i] = a3


Gauss_sol, Gauss_iter, Gauss_time = Gauss_Seidel(A, b, N)
Jacobi_sol, Jacobi_iter, Jacobi_time = Jacobi(A, b, N)
