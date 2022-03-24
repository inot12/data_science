#! /home/toni/.pyenv/shims/python3
"""
Created on Jul 23, 2020

@author:toni
"""

from timeit import default_timer as timer

from numba import vectorize
import numpy as np


def powar(a, b):
    return a ** b


@vectorize(['float32(float32, float32)'], target='cuda')
def powah(a, b):
    return a ** b


def main():
    vec_size = 100000000

    a = b = np.array(np.random.sample(vec_size), dtype=np.float32)
    c = np.zeros(vec_size, dtype=np.float32)
    d = np.zeros(vec_size, dtype=np.float32)

    start = timer()
    c = powar(a, b)
    print("normal:", timer() - start)

    start = timer()
    d = powah(a, b)
    print("vectorize:", timer() - start)


if __name__ == "__main__":
    main()
