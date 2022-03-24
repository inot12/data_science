"""
Created on Jan 17, 2022

@author: inot
"""

import multiprocessing
import random
from functools import reduce


def func(number_):
    random_list = random.sample(range(1000000), number_)
    return reduce(lambda x, y: x * y, random_list)


number = 50000
process1 = multiprocessing.Process(target=func, args=(number,))
process2 = multiprocessing.Process(target=func, args=(number,))

process1.start()
process2.start()

process1.join()
process2.join()
