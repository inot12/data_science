#! /home/toni/.pyenv/shims/python3
"""
Created on Sep 14, 2020

@author:toni
"""

import numpy as np


def np_tests(a):
    """Test basic numpy commands."""
    print('a =\n', a)
    b = a * a
    c = a ** 2
    print('b =\n', b, '\n\n', 'c =\n', c)
    print('a @ a =\n', a @ a)  # matrix multiplication
    print('Nesting level of a:\n', a.ndim)
    print('Number of elements in a:\n', a.size)
    print('Dimensions of matrix a:\n', a.shape)
    print(a.shape[0])
    print(a[-1])
    print(a[-1, -1])

    print('Second column of a:\n', a[:, 1])
    print('Third row of a:\n', a[2, :])
    print(f'a =\n{a}')
    # notice the :,i1:i2 slice when a submatrix is sliced
    print('Submatrix of a:\n', a[0:2][:, 0:2])
    # select rows and cols to form a submatrix
    print('Selected rows and columns of a:\n', a[np.ix_([0, 2], [0, 2])])
    print('a.T:\n', a.T)
    print('Transpose of conjugate (for complex matrices) of a:\n', a.conj().T)
    # avoid the following statement because it is easy do get zero division
    print('a\\b=\n', a / b)
    print('Show which elements of a greater than 2:\n', a > 2)
    # Return the indices of the elements that are non-zero.
    print('Nonzero a:\n', np.nonzero(a > 2))
    # zero out all elements less than 5 (5 excluded)
    print('Zero out all ai < 5:\n', a * (a > 5))
    # zero out all elements greater than 5 (5 included)
    print('Zero out all ai >= 5:\n', a * (a < 5))

    # aliasing results in: change of d changes a too, therefore using copy
    # no need to import the copy module since np has a method copy
    t = np.copy(a)
    d = a
    d[:] = 3
    print('Aliasing a to d, change d')
    print('d =\n', d)
    print('a =\n', a)
    a = np.copy(t)
    print('Copy a to d, change d')
    d = a.copy()
    d[:] = 3
    print('d =\n', d)
    print('a =\n', a)
    d = np.copy(a)
    d[:] = 3
    print('Using np.copy(a), change d')
    print('d =\n', d)
    print('a =\n', a)

    y = a[1, :].copy()  # copy 2nd row of a to y
    print('Copy of 2nd row of a:\n', y)
    # create a 1-dim vector from a matrix, copies a which means a is unchanged
    z = a.flatten()
    print('Create a 1-dim vector from a matrix', z)

    print('Example uses of np.arrange')
    print(np.arange(5))
    print(np.arange(1, 11))
    print(np.arange(1, 11)[: np.newaxis])  # this should be a column vector
    print('np.zeros use')
    print(np.zeros((3, 5)))
    print(np.zeros((2, 2, 3)))
    print('np.ones use')
    print(np.ones((5, 3)))
    print('identity matrix\n', np.eye(3))
    print('diagonal matrix\n', np.diag(np.array([1, 2, 3])))
    print('random matrix\n', np.random.rand(3, 4))
    print('np.linspace use\n', np.linspace(1, 10, 6))

    # two 2D arrays: one of x values, the other of y values
    print('mgrid method\n', np.mgrid[0:9., 0:6.])
    print('meshgrid method\n', np.meshgrid([1, 2, 4], [2, 4, 5]))
    print('Matrix result of adding a and b horizontally:\n', np.hstack((a, b)))
    print('Matrix result of adding a and b vertically:\n', np.vstack((a, b)))
    print('np.tile examples:')
    print(np.tile(a, (2, 3)))  # create a matrix like [[a, a, a], [a, a, a]]
    print(np.tile(a, (2, 2)))
    print('Maximum value of a: ', a.max())
    print('List of maximum values of each column of a', a.max(0))
    print('List of maximum values of each row', a.max(1))
    print('b =\n', b)
    print('a @ a =\n', a @ a)
    print('Compare elements of a@a and b and from a matrix with max values\n',
          np.maximum(a @ a, b))
    v = np.array([1, 2, 3])
    # norm of a vector, works only for vectors, not matrices
    print('Norm of vector v: ', np.sqrt(v @ v))
    print('a =\n', a)
    print('norm of matrix a\n', np.linalg.norm(a))

    print('a =\n', a, '\nb =\n', b)
    print('Logical and\n', np.logical_and(a, b))  # logical and; logical_or
    print('Bitwise and\n', a & b)  # bitwise and; a | b bit-wise or

    print('Determinant of a\n', np.linalg.det(a))  # determinant of a
    print('Inverse of a\n', np.linalg.inv(a))  # inverse of a
    print('Pseudo-inverse of a\n', np.linalg.pinv(a))  # pseudo-inverse of a
    print('Rank(a):', np.linalg.matrix_rank(a))  # rank of a
    # ax = v if a square; a\v
    print('x from ax=v:', np.linalg.solve(a, v), '\n')
    # ax = b if generally
    print('ax = b, general case :\n', np.linalg.lstsq(a, v, rcond=None), '\n')
    print('a.T, v.T, b//v\n', np.linalg.solve(a.T, v.T))  # b/v
    print('Single value decomposition of a:')
    U, S, Vh = np.linalg.svd(a)
    V = Vh.T  # single value decomposition.
    print(U, '\n\n', S, '\n\n', Vh, '\n\n', V)
    D, V = np.linalg.eig(a)
    print('Eigenvalues and vectors:\n', D, '\n\n', V)
    Q, R = np.linalg.qr(a)  # QR decomposition
    print('QR decomposition:\n', Q, '\n', R)
#     L,U = np.scipy.linalg.lu(a)
#     print('\n',L,'\n',U)
    print("Sorting:", a.sort(), '\na sorted\n', a)

    print('Unique elements of a', np.unique(a))
    print('a =\n', a)
    # squeeze should remove unnecessary levels of a
    print('Squeeze of a:\n', a.squeeze())  # what does that do?
    print('Ending np_tests...')

    grades = np.array([1, 3, 4, 2, 5, 5])
    print(grades)

    res = np.where(grades > 3)
    print(f'The where function finds element positions that satisfy'
          f'a certain condition: {res}')

    print(f'argmin returns index of smallest element: {np.argmin(grades)}\n')
    print(f'argmax returns index of smallest element: {np.argmax(grades)}\n')
    print(f'argsort returns indices of sorted elements:{np.argsort(grades)}\n')

    arr1 = np.array([3, 1, 2, 5, 5])
    arr2 = np.array([3, 4, 4, 2, 4])
    print(f'Intersection of arr1 and arr2: {np.intersect1d(arr1, arr2)}')


def main():
    a = np.array([[2, 5, 1], [7, 1, 5], [8, 4, 9]])
    np_tests(a)


if __name__ == "__main__":
    main()
