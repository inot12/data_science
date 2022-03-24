#! /home/toni/.pyenv/shims/python3
"""
Created on Sep 17, 2020

@author:toni
"""

# import scipy
# import numpy as np
from scipy.spatial.transform import Rotation as R

a = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
r = R.from_matrix(a)
print(r.as_matrix())
print(r.as_quat())


def main():
    pass


if __name__ == "__main__":
    main()
