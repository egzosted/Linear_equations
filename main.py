# Michał Piekarski 175456

import numpy as np
import numpy.linalg as la
import time

"""
    function that calculates norm of vector
    @param vec vector that norm we want to calculate
    @return scalar, norm of vector
"""


def vector_norm(vec):
    norm = 0
    sum = 0
    for i in vec:
        sum += i * i
    norm = np.sqrt(sum)
    return norm


"""
    function that calculates system of linear equations using Jacobi method
    @param A matrix of coefficients
    @param b vector of free variables
    @param size size of square matrix and vector
    @return tuple of: solution vector, iterations needed to convergence, time of solution
"""


def Jacobi(A, b, size):
    x = np.zeros(shape=[size, 1])   # vector of approximated solution
    temp_x = np.ones(shape=[size, 1])  # vector of approximated solution in next iteration
    norm = vector_norm(np.dot(A, x) - b)
    error = 10 ** -9
    iter = 0

    start = time.time()
    while norm > error and iter < 1000:
        for i in range(size):
            before_diagonal = 0
            after_diagonal = 0
            for j in range(i):
                before_diagonal += A[i][j] * x[j]
            for j in range(i + 1, size):
                after_diagonal += A[i][j] * x[j]
            temp_x[i] = (b[i] - before_diagonal - after_diagonal) / A[i][i]
        for i in range(size):
            x[i] = temp_x[i]

        norm = vector_norm(np.dot(A, x) - b)
        iter += 1
    end = time.time()
    return x, iter, end - start


"""
    function that calculates system of linear equations using Jacobi method
    @param A matrix of coefficients
    @param b vector of free variables
    @param size size of square matrix and vector
    @return tuple of: solution vector, iterations needed to convergence, time of solution
"""


def Gauss_Seidel(A, b, size):
    x = np.zeros(shape=[size, 1])   # vector of approximated solution
    temp_x = np.ones(shape=[size, 1])  # vector of approximated solution in next iteration
    norm = vector_norm(np.dot(A, x) - b)
    error = 10 ** -9
    iter = 0

    start = time.time()
    while norm > error and iter < 1000:
        for i in range(size):
            before_diagonal = 0
            after_diagonal = 0
            for j in range(i):
                before_diagonal += A[i][j] * temp_x[j]
            for j in range(i + 1, size):
                after_diagonal += A[i][j] * x[j]
            temp_x[i] = (b[i] - before_diagonal - after_diagonal) / A[i][i]
        for i in range(size):
            x[i] = temp_x[i]

        norm = vector_norm(np.dot(A, x) - b)
        iter += 1
    end = time.time()
    return x, iter, end - start

    """
        function that calculates system of linear equations using LU factorization
        @param A matrix of coefficients
        @param b vector of free variables
        @param size size of square matrix and vector
        @return tuple of: solution vector, time of solution
    """


def LU(A, b, size):


c = 5   # penultimate number of index
d = 6   # last number of index
e = 4   # 4th number of index
f = 5   # 3rd number of index

# task A data
a1 = 3
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
        A[i][i + 1] = a2
    if i < N - 2:
        A[i + 2][i] = a3
        A[i][i + 2] = a3


# Gauss_sol, Gauss_iter, Gauss_time = Gauss_Seidel(A, b, N)
# Jacobi_sol, Jacobi_iter, Jacobi_time = Jacobi(A, b, N)

# print(
#    f'Metoda Gaussa-Seidela rozwiązała zadanie w czasie {Gauss_time} i potrzebowala {Gauss_iter} iteracji')
# print(
#    f'Metoda Jacobiego rozwiązała zadanie w czasie {Jacobi_time} i potrzebowala {Jacobi_iter} iteracji')
