"""
Runtime optimization through vectorization and parallelization.
Script 2: Vectorized calculation of haversine distance.

January 2018
Markus Konrad <markus.konrad@wzb.eu>

# example with real places:

coords = np.array([
    [52.516667, 13.388889, 51.507222, -0.1275],   # Berlin-London
    [52.516667, 13.388889, 55.75, 37.616667],     # Berlin-Moscow
    [55.75, 37.616667, 51.507222, -0.1275],       # Moscow-London
])

vec_haversine(coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3])

####

labels = [
    'Berlin-London',
    'Berlin-Moscow',
    'Moscow-London',
]

df_coords = pd.DataFrame(coords, index=labels,
                         columns=['origin_lat', 'origin_lng',
                                  'destination_lat', 'destination_lng'])

vec_haversine(df_coords.origin_lat, df_coords.origin_lng,
              df_coords.destination_lat, df_coords.destination_lng)

"""

import sys

import numpy as np
import pandas as pd


def vec_haversine(a_lat, a_lng, b_lat, b_lng):
    """
    Vectorized version of haversine / great circle distance for point to point
    distances on earth. Calculates the haversine distance in kilometers for a
    bunch of geocoordinate pairs between the points defined by `a_lat`, `a_lng`
    and `b_lat`, `b_lng`.
    """
    R = 6371  # earth radius in km

    a_lat = np.radians(a_lat)
    a_lng = np.radians(a_lng)
    b_lat = np.radians(b_lat)
    b_lng = np.radians(b_lng)

    d_lat = b_lat - a_lat
    d_lng = b_lng - a_lng

    d_lat_sq = np.sin(d_lat / 2) ** 2
    d_lng_sq = np.sin(d_lng / 2) ** 2

    a = d_lat_sq + np.cos(a_lat) * np.cos(b_lat) * d_lng_sq
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c  # returns distance between a and b in km


if __name__ == '__main__':
    script_signature = f'{sys.argv[0]} <array|dataframe> <num. test rows>'

    if len(sys.argv) != 3:
        print('run as: ', script_signature, file=sys.stderr)
        exit(1)

    datastructure = sys.argv[1]
    assert datastructure in ('array', 'dataframe')

    n_values = int(sys.argv[2])

    coords = np.array([np.random.uniform(-90, 90, n_values),  # random origin latitude
                       np.random.uniform(-180, 180, n_values),  # random origin longitude
                       np.random.uniform(-90, 90, n_values),  # random destination latitude
                       np.random.uniform(-180, 180, n_values)]).T  # random destination longitude

    if datastructure == 'array':
        result = vec_haversine(coords[:, 0], coords[:, 1],
                               coords[:, 2], coords[:, 3])
    else:
        df_coords = pd.DataFrame(
            coords, columns=['origin_lat', 'origin_lng',
                             'destination_lat', 'destination_lng'])
        result = vec_haversine(df_coords.origin_lat, df_coords.origin_lng,
                               df_coords.destination_lat,
                               df_coords.destination_lng)

"""
In [8]: %timeit %run vectorized.py array 10000
2.9 ms ± 72.3 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

In [9]: %timeit %run vectorized.py dataframe 10000
4.83 ms ± 134 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
"""
