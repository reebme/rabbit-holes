#!/usr/bin/python

from concurrent.futures import ProcessPoolExecutor, as_completed

from pathlib import Path

import numpy as np
import pandas as pd

import time
from datetime import datetime

import logging

import submatrix

# short_sides = [30, 60, 120, 250, 500, 1000]
short_sides = [30, 60, 120, 250]

# missingness to test
missingness = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]

# parameters to search through
# it's a computational cost vs solutions diversity compromise
# larger Cs still produce distinct solutions
Cs = np.hstack((np.arange(1, 10), np.arange(10, 100, 10), np.arange(100, 301, 20)))

# R = n/d, r = qR
# qs
planted_relative_aspects = [
    0.35,     # relatively wide
    1.0,      # same relative shape as the host
    2.8,      # relatively tall
]

# P(element belongs to the planted block | element is observed) = alpha
alpha = 0.5

T = 10

def run_benchmark(task):
    # a task is (big_dims, complete_dims, missingness, t, seed, Cs)
    big_dims, complete_dims, miss, _, seed, Cs = task

    if complete_dims[0] > big_dims[0] or complete_dims[1] > big_dims[1]:
        raise ValueError("Complete matrix cannot be larger than the big matrix.")

    big = submatrix.generate_matrix_with_planted(big_dims[0], big_dims[1],
                                                complete_dims[0], complete_dims[1],
                                                seed, miss)
    df = pd.DataFrame(big)

    results = []

    ### adaptive pruning ###

    # max_iter = np.max(big_dims)
    # iterate over Cs
    for C in Cs:
        start = time.perf_counter()
        found_rows, found_cols, iter_info = submatrix.adaptive_prune_to_complete(df, max_iter=3000, shrink_coeff=C)
        elapsed = time.perf_counter() - start

        if found_rows is not None and found_cols is not None:
            # check if a found matrix is indeed complete
            missing_count = df.loc[found_rows, found_cols].isna().sum().sum()
            if missing_count != 0:
                raise AssertionError(
                    f"Returned submatrix contains {missing_count} missing entries; "
                    f"big_dim={big_dims}, complete_dim={complete_dims}, "
                    f"missingness={miss}, C={C}"
                )
            
            results.append({
                "big_rows": big_dims[0],
                "big_cols": big_dims[1],
                "complete_rows": complete_dims[0],
                "complete_cols": complete_dims[1],
                "missingness": miss,
                "C": C,
                "seed": seed,
                "row_cnt": len(found_rows),
                "col_cnt": len(found_cols),
                "found_area": len(found_rows) * len(found_cols),
                "no_iter": iter_info["i"].max() + 1,
                "time_sec": elapsed,
            })  
        else:
            results.append({
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
                "no_iter": iter_info["i"].max() + 1,
                "time_sec": elapsed,
            })

    ### basic prune to complete ###

    start = time.perf_counter()
    found_rows, found_cols, iter_info = submatrix.basic_prune_to_complete(big)
    elapsed = time.perf_counter() - start
    
    if found_rows is not None and found_cols is not None:
        # check if a found matrix is indeed complete
        missing_count = df.loc[found_rows, found_cols].isna().sum().sum()
        if missing_count != 0:
            raise AssertionError(
                f"Returned submatrix contains {missing_count} missing entries; "
                f"big_dim={big_dims}, complete_dim={complete_dims}, "
                f"missingness={miss}, C={C}"
            )
        
        results.append({
            "big_rows": big_dims[0],
            "big_cols": big_dims[1],
            "complete_rows": complete_dims[0],
            "complete_cols": complete_dims[1],
            "missingness": miss,
            "C": -1,
            "seed": seed,
            "row_cnt": len(found_rows),
            "col_cnt": len(found_cols),
            "found_area": len(found_rows) * len(found_cols),
            "no_iter": iter_info["i"].max() + 1,
            "time_sec": elapsed,
        })  
    else:
        results.append({
            "big_rows": big_dims[0],
            "big_cols": big_dims[1],
            "complete_rows": complete_dims[0],
            "complete_cols": complete_dims[1],
            "missingness": miss,
            "C": -1,
            "seed": seed,
            "row_cnt": -1,
            "col_cnt": -1,
            "found_area": -1,
            "no_iter": iter_info["i"].max() + 1,
            "time_sec": elapsed,
        })

    return results

if __name__ == "__main__":
    ### prepare the out file ###
    
    # the results are written to repo/results/benchmarking_submatrices
    # it is assumed the script is in repo/src
    results_folder = (
        Path(__file__).resolve().parent.parent
        / "results"
        / "benchmarking_submatrices"
    )
    # make sure the folder exists
    results_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(results_folder / f"submatrix_planted_results_{timestamp}.csv").resolve()

    ### prepare the log file ###

    log_file = Path(results_folder / f"submatrix_planted_results_{timestamp}_err_log.txt").resolve()
    
    # configure logging
    logging.basicConfig(
        filename=log_file,
        level=logging.ERROR,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logger = logging.getLogger(__name__)
    
    ### prepare the big matrices dimensions ###

    big_dims_to_test = submatrix.prepare_big_dims(short_sides)
    # big_dims_to_test.extend([
    #    (2000, 3000),
    #    (3000, 2000),
    # ])

    start_seed = 1
    first_result = True
   
    with ProcessPoolExecutor(max_workers=2) as executor:
        # loop over big_dims not to keep all futures in memory 
        for big_dims in big_dims_to_test:
            ### prepare tasks ###

            complete_dims = {}
            
            # the encompassing matrix is n x d
            n, d = big_dims
            for m in missingness:
                temp = []
                for a in planted_relative_aspects:
                    temp_compl_dims = submatrix.calculate_complete_dims(alpha, m, a, n, d)
                    if temp_compl_dims is not None:
                        temp.append(temp_compl_dims)

                complete_dims[(n, d), m] = temp

            # big dims, complete dims, missingness define one matrix
            tasks = [(big, complete, missing) 
                        for (big, missing), complete_dims_list in complete_dims.items() 
                        for complete in complete_dims_list
                    ]
            print(f"No of tested matrices: {len(tasks)}.")
            # Repeat each matrix experiment T times
            tasks = [(*task, t) for task in tasks for t in range(T)]
            print(f"No of experiments (unique matrices): {len(tasks)}.")
            # assign each unique matrix a seed
            tasks = [(*task, seed, Cs) 
                        for seed, task in enumerate(tasks, start = start_seed)]
            start_seed += len(tasks)
            
            # a task is (big_dims, complete_dims, missingness, t, seed, Cs)
            
            ### execute tasks with max_workers processess ###
            
            future_to_task = {
                executor.submit(run_benchmark, task): task
                for task in tasks
            }

            for future in as_completed(future_to_task):
                task = future_to_task[future]

                try:
                    result = future.result()
                except Exception:
                    big, complete, miss, repetition, seed, _ = task

                    logger.exception(
                        "Task failed | big=%s | complete=%s | "
                        "missingness=%s | repetition=%s | seed=%s\n"
                        "%s",
                        big,
                        complete,
                        miss,
                        repetition,
                        seed
                    )
                    continue

                pd.DataFrame(result).to_csv(
                    results_file,
                    mode="w" if first_result else "a",
                    header=first_result,
                    index=False,
                )
                first_result = False
