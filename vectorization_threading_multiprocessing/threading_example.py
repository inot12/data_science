#! ~/python-environments/3.8.5/bin/python3
"""
Created on Jan 17, 2022

@author:inot
"""

import threading
import random
from functools import reduce


def func(number_):
    random_list = random.sample(range(1000000), number_)
    return reduce(lambda x, y: x * y, random_list)


number = 50000
thread1 = threading.Thread(target=func, args=(number,))
thread2 = threading.Thread(target=func, args=(number,))

thread1.start()
thread2.start()

thread1.join()
thread2.join()
