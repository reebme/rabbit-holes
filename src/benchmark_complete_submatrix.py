#!/usr/bin/python

from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import time
from datetime import datetime

from submatrix import adaptive_prune_to_complete, extract_mask, apply_missingness

big_dims = [
    (5, 5),          # tiny square sanity check

    (23, 17),        # small tall-ish
    (17, 29),        # small wide-ish
    (47, 43),        # small near-square

    (137, 79),       # medium tall
    (83, 157),       # medium wide
    (149, 149),      # medium square

#    (503, 241),      # large tall
#    (257, 509),      # large wide
#    (487, 461),      # large near-square

#    (1237, 719),     # very large tall
#    (733, 1511),     # very large wide

#    (2000, 3000),    # max benchmark
]

complete_dims = [
    (5, 5),          # tiny square

    (7, 5),          # small tall
    (5, 7),          # small wide
    (9, 9),          # small square

    (23, 11),        # medium-ish tall
    (13, 29),        # medium-ish wide
    (37, 31),        # medium-ish near-square

   (89, 47),        # large tall
   (53, 101),       # large wide
   (137, 137),      # large square

 #   (311, 173),      # very large tall
 #   (197, 389),      # very large wide
]

# missingness to test
missingness = [0.3, 0.5, 0.7, 0.85, 0.95]

# parameters to search through
#shs = np.arange(1, 10, 1)
Cs = np.hstack((np.arange(1, 10), np.arange(10, 100, 10), np.arange(100, 500, 20)))

def run_benchmark(task):
    # (int, int), (int, int), float
    big_dims, complete_dims, miss, C, seed = task

    if complete_dims[0] > big_dims[0] or complete_dims[1] > big_dims[1]:
        raise ValueError("Complete matrix cannot be larger than the big matrix.")

    rng = np.random.default_rng(seed)

    # generate complete matrix to be planted
    complete = rng.standard_normal(size=complete_dims)

    # generate big encompassing matrix
    big = rng.standard_normal(size=big_dims)

    if big.size - complete.size < big.size * miss:
        raise ValueError("Level of missingness impossible to apply.")

    # plant a complete matrix
    rows = np.sort(rng.choice(big_dims[0], size=complete_dims[0], replace=False))
    cols = np.sort(rng.choice(big_dims[1], size=complete_dims[1], replace=False))

    big[np.ix_(rows,cols)] = complete

    # apply missingness
    available_mask = extract_mask(big, rows, cols)
    apply_missingness(big, available_mask, miss, rng)
        
    df = pd.DataFrame(big)

    #max_iter = np.max(big_dims)

    start = time.perf_counter()
    found_rows, found_cols, iter_info = adaptive_prune_to_complete(df, max_iter=1000, shrink_coeff=C)
    elapsed = time.perf_counter() - start

    if found_rows is not None and found_cols is not None:
        # check if a found matrix is indeed complete
        if df.loc[found_rows, found_cols].isna().sum().sum() != 0:
            raise AssertionError(
                f"Returned submatrix contains {missing_count} missing entries; "
                f"big_dim={big_dim}, complete_dim={complete_dim}, "
                f"missingness={miss}, C={C}"
            )
        
        return {
            "big_rows": big_dims[0],
            "big_cols": big_dims[1],
            "complete_rows": complete_dims[0],
            "complete_cols": complete_dims[1],
            "missingness": miss,
            "C": C,
            "seed": seed,
            "row_cnt": len(found_rows),
            "col_cnt": len(found_cols),
            "found_area": df.loc[found_rows, found_cols].size,
            "no_iter": iter_info["i"].max(),
            "time_sec": elapsed,
        }
            
    else:
        return {
            "big_rows": big_dims[0],
            "big_cols": big_dims[1],
            "complete_rows": complete_dims[0],
            "complete_cols": complete_dims[1],
            "missingness": miss,
            "C": C,
            "seed": seed,
            "row_cnt": -1,
            "col_cnt": -1,
            "found_area": -1,
            "no_iter": iter_info["i"].max(),
            "time_sec": elapsed,
        }

if __name__ == "__main__":
    base_tasks = list(product(big_dims, complete_dims, missingness))
    tasks = [
        (big_dim, complete_dim, miss, C, seed)
        for seed, (big_dim, complete_dim, miss) in enumerate(base_tasks, start=1)
        for C in Cs
    ]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(f"../results/submatrix_planted_results_{timestamp}.txt").resolve()

    first_result = True

    with ProcessPoolExecutor(max_workers=2) as executor:
        future_to_task = {
            executor.submit(run_benchmark, task): task
            for task in tasks
        }

        for future in as_completed(future_to_task):
            task = future_to_task[future]

            try:
                result = future.result()  # one DataFrame from one completed benchmark
            except AssertionError as ae:
                print(f"Failed {task}:\n{ae}")
                continue
            except ValueError as ve:
                continue

            pd.DataFrame([result]).to_csv(
                results_file,
                mode="w" if first_result else "a",
                header=first_result,
                index=False,
            )
            first_result = False

            #print(f"Finished {task}.")
