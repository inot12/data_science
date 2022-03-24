"""
Runtime optimization through vectorization and parallelization.
Script 1: Unoptimized script with row-wise apply of haversine calculation.

January 2018
Markus Konrad <markus.konrad@wzb.eu>

# example with real places:

coords = np.array([
    [52.516667, 13.388889, 51.507222, -0.1275],   # Berlin-London
    [52.516667, 13.388889, 55.75, 37.616667],     # Berlin-Moscow
    [55.75, 37.616667, 51.507222, -0.1275],       # Moscow-London
])

np.apply_along_axis(haversine, 1, coords)

####

labels = [
    'Berlin-London',
    'Berlin-Moscow',
    'Moscow-London',
]

df_coords = pd.DataFrame(coords, index=labels,
                         columns=['origin_lat', 'origin_lng',
                                  'destination_lat', 'destination_lng'])

df_coords.apply(haversine, axis=1)
"""

import math
import sys

import pandas as pd
import numpy as np


def haversine(row):
    """
    haversine:
    calculate great circle distance between two points on earth in km
    """
    a_lat, a_lng, b_lat, b_lng = row

    R = 6371     # earth radius in km

    a_lat = math.radians(a_lat)
    a_lng = math.radians(a_lng)
    b_lat = math.radians(b_lat)
    b_lng = math.radians(b_lng)

    d_lat = b_lat - a_lat
    d_lng = b_lng - a_lng

    a = (math.pow(math.sin(d_lat / 2), 2) + math.cos(a_lat)
         * math.cos(b_lat) * math.pow(math.sin(d_lng / 2), 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


if __name__ == '__main__':
    script_signature = (f'{sys.argv[0]} <haversine|hash> <array|dataframe>'
                        ' <num. test rows>')

    if len(sys.argv) != 4:
        print('run as: ', script_signature, file=sys.stderr)
        exit(1)

    operation = sys.argv[1]
    assert operation in ('haversine', 'hash')

    datastructure = sys.argv[2]
    assert datastructure in ('array', 'dataframe')

    n_values = int(sys.argv[3])

    coords = np.array([np.random.uniform(-90, 90, n_values),       # random origin latitude
                       np.random.uniform(-180, 180, n_values),     # random origin longitude
                       np.random.uniform(-90, 90, n_values),       # random destination latitude
                       np.random.uniform(-180, 180, n_values)]).T  # random destination longitude

    if operation == 'haversine':
        fn = haversine
    else:
        fn = lambda row: hash(tuple(row))

    if datastructure == 'array':
        result = np.apply_along_axis(fn, 1, coords)
    else:
        df_coords = pd.DataFrame(coords)
        result = df_coords.apply(fn, axis=1)


"""
In [3]: %timeit %run unoptimized.py array 10000
53.8 ms ± 269 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

In [4]: %timeit %run unoptimized.py dataframe 10000
174 ms ± 6.14 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
"""