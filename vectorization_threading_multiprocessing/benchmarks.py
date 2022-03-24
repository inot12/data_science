#! ~/python-environments/3.8.5/bin/python3
"""
Created on Nov 25, 2021

@author:inot
"""

import multiprocessing as mp
import os
import random
import threading
import time
from timeit import default_timer as timer

import numpy as np
import pandas as pd


def vec_haversine(a_lat, a_lng, b_lat, b_lng):
    """Calculate the great circle distance (GCD) of two points on earth by
    using the haversine formula."""
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

    return R * c  # return distance between a and b in km


def process_chunk(proc_chunk):
    """Calculate the GCD for a chunk of data. This result will be used in
    parallel processing."""
    chunk_res = vec_haversine(proc_chunk.origin_lat,
                              proc_chunk.origin_lng,
                              proc_chunk.destination_lat,
                              proc_chunk.destination_lng)
    chunk_res.index = proc_chunk.index

    return chunk_res


def main():
    # vectorization is used for numpy array operations
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([6, 7, 8, 9, 10])
    print(f'a * b = {a*b}')

    # python's list comprehension
    a = [random.randint(1, 100) for _ in range(1000000)]
    b = [random.randint(1, 100) for _ in range(1000000)]
    start = timer()
    (lambda a, b: [x * y for x, y in zip(a, b)])(a, b)
    end = timer()
    print(f'Time (list comprehension): {end - start}')

    # numpy vectorization
    a = np.random.randint(1, 100, 1000000)
    b = np.random.randint(1, 100, 1000000)
    start = timer()
    _ = a * b
    end = timer()
    print(f'Time (numpy): {end - start}')

    # MULTIPROCESSING

    # data initialization
    coords = np.array([
        # orig_lat, orig_lng,  dest_lat,  dest_lng
        [52.516667, 13.388889, 51.507222, -0.1275],  # Berlin-London
        [52.516667, 13.388889, 55.75, 37.616667],  # Berlin-Moscow
        [55.75, 37.616667, 51.507222, -0.1275],  # Moscow-London
    ])
    labels = ['Berlin-London', 'Berlin-Moscow', 'Moscow-London', ]
    df_coords = pd.DataFrame(coords, index=labels,
                             columns=['origin_lat', 'origin_lng',
                                      'destination_lat', 'destination_lng'])

    # use vectorized haversine function to calculate GCD
    res = vec_haversine(df_coords.origin_lat, df_coords.origin_lng,
                        df_coords.destination_lat, df_coords.destination_lng)
    print(res)

    n_proc = mp.cpu_count()  # use all CPU cores in your machine
    chunksize = len(df_coords) // n_proc  # determine the size of each chunk

    # slice the dataframe
    # the last process always gets the chunk with the remainder
    proc_chunks = []
    for i_proc in range(n_proc):
        chunkstart = i_proc * chunksize
        # make sure to include the division remainder for the last process
        chunkend = (i_proc + 1) * chunksize if i_proc < n_proc - 1 else None

        proc_chunks.append(df_coords.iloc[slice(chunkstart, chunkend)])
    assert sum(map(len, proc_chunks)) == len(df_coords)

    with mp.Pool(processes=n_proc) as pool:
        # start the sub-processes without blocking
        # pass the chunk to each worker process
        proc_results = [pool.apply_async(process_chunk, args=(chunk,))
                        for chunk in proc_chunks]
        # with apply_async we distribute the data and start the processes
        # simultaneously ("non-blocking") without waiting for individual
        # processes to finish

        # block until all results are fetched
        result_chunks = [r.get() for r in proc_results]

    # combine the partial results of each process into a single final result
    results = pd.concat(result_chunks)
    results = pd.concat((df_coords, results), axis=1)
    print(results)
    # make sure we got a result for each coordinate pair:
    assert len(results) == len(df_coords)

    # a simple example

    def info(title):
        print(title)
        print('module name:', __name__)
        print('parent process:', os.getppid())
        print('process id:', os.getpid())

    def f(name):
        info('function f')
        print('hello', name)

    if __name__ == '__main__':
        info('main line')
        p = mp.Process(target=f, args=('bob',))
        p.start()
        p.join()

    # THREADING

    # with this threading example, the code does not execute sequentially
    # this may reduce execution time
    def greet_them(people):
        for person in people:
            print("Hello Dear " + person + ". How are you?")
            time.sleep(0.5)

    def assign_id(people):
        i = 1
        for person in people:
            print(f"Hey! {person}, your id is {i}.")
            i += 1
            time.sleep(0.5)

    people = ['Richard', 'Dinesh', 'Elrich', 'Gilfoyle', 'Gevin']

    t = time.time()

    # Created the Threads
    t1 = threading.Thread(target=greet_them, args=(people,))
    t2 = threading.Thread(target=assign_id, args=(people,))

    # Started the threads
    t1.start()
    t2.start()

    # Joined the threads
    t1.join()
    t2.join()

    print("Woaahh!! My work is finished..")
    print("I took " + str(time.time() - t))


if __name__ == "__main__":
    main()
