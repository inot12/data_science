"""
Runtime optimization through vectorization and parallelization.
Script 3: Parallel and vectorized calculation of haversine distance.

Please note that this might be slower than the single-core vectorized version
because of the overhead that is caused by multiprocessing.

January 2018
Markus Konrad <markus.konrad@wzb.eu>
"""

import sys
import os
import multiprocessing as mp

import numpy as np
import pandas as pd

# we use the same vectorized haversine function
from vectorized import vec_haversine

operation = 'haversine'  # haversine or hash


def process_chunk(proc_chunk):
    """
    Process the partial array/dataframe `proc_chunk` passed to this process
    worker. Return an array/dataframe of the same length and same indices.
    """
    # pid = os.getpid()
    # print('process %d: got chunk of size %d with rows %d to %d'
    #       % (pid, len(proc_chunk), proc_chunk.index[0], proc_chunk.index[-1]))

    if operation == 'haversine':
        chunk_res = vec_haversine(
            proc_chunk.origin_lat, proc_chunk.origin_lng,
            proc_chunk.destination_lat, proc_chunk.destination_lng)
        chunk_res.index = proc_chunk.index
    else:  # hash
        chunk_res = proc_chunk.apply(lambda row: hash(tuple(row)), axis=1)

    return chunk_res


if __name__ == '__main__':
    script_signature = f'{sys.argv[0]} <haversine|hash> <num. test rows>'

    if len(sys.argv) != 3:
        print('run as: ', script_signature, file=sys.stderr)
        exit(1)

    operation = sys.argv[1]
    assert operation in ('haversine', 'hash')

    n_values = int(sys.argv[2])

    coords = np.array([np.random.uniform(-90, 90, n_values),  # random origin latitude
                       np.random.uniform(-180, 180, n_values),  # random origin longitude
                       np.random.uniform(-90, 90, n_values),  # random destination latitude
                       np.random.uniform(-180, 180, n_values)]).T  # random destination longitude

    df_coords = pd.DataFrame(
        coords,
        columns=['origin_lat', 'origin_lng', 'destination_lat',
                 'destination_lng'])

    # set the number of processes
    n_proc = mp.cpu_count()

    # this often can't be devided evenly (handle this in the for-loop below)
    chunksize = len(coords) // n_proc

    # devide into chunks
    proc_chunks = []
    for i_proc in range(n_proc):
        chunkstart = i_proc * chunksize
        # make sure to include the division remainder for the last process
        chunkend = (i_proc + 1) * chunksize if i_proc < n_proc - 1 else None

        proc_chunks.append(df_coords.iloc[slice(chunkstart, chunkend)])

    # make sure all data is in the chunks
    assert sum(map(len, proc_chunks)) == len(df_coords)

    # distribute work to the worker processes
    with mp.Pool(processes=n_proc) as pool:
        # starts the sub-processes without blocking
        # pass the chunk to each worker process
        proc_results = [pool.apply_async(process_chunk, args=(chunk,))
                        for chunk in proc_chunks]

        # blocks until all results are fetched
        result_chunks = [r.get() for r in proc_results]

    # concatenate results from worker processes
    results = pd.concat(result_chunks)
    results = pd.concat((df_coords, results), axis=1)
    # make sure we got a result for each coordinate pair
    assert len(results) == len(df_coords)

    # print('results:')
    # print(results)

"""
haversine is not faster with parallel processing:

In [10]: %timeit %run vectorized.py dataframe 1000000
230 ms ± 10.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

In [11]: %timeit %run parallelized.py haversine 1000000
388 ms ± 70.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

hashing is faster:

In [8]: %timeit %run unoptimized.py hash dataframe 1000000
16.4 s ± 270 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

In [9]: %timeit %run parallelized.py hash 1000000
8.37 s ± 115 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
"""
