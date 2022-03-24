#! /home/toni/.pyenv/shims/python3
"""
Created on Jul 23, 2020

@author:toni
"""

from timeit import default_timer as timer

from numba import jit, cuda
import numpy as np


# normal function to run on cpu
def func(a):
    for i in range(10000000):
        a[i] += 1


# function optimized to run on gpu
@jit(target_backend="cuda")
def func2(a):
    for i in range(10000000):
        a[i] += 1


def main():
    n = 10000000
    a = np.ones(n, dtype=np.float64)
    b = np.ones(n, dtype=np.float32)

    start = timer()
    func(a)
    print("without GPU:", timer() - start)

    start = timer()
    func2(a)
    print("with GPU:", timer() - start)


if __name__ == "__main__":
    main()
