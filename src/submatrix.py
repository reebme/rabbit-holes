import pandas as pd
import numpy as np

import warnings

def prepare_big_dims(short_sides):
    '''
    Prepare a list of big matrix dimensions given a list(int) of s values.

    Returns a list of tuples (row_number, col_number).
    '''
    big_dims = []
    
    for s in short_sides:
        big_dims.extend([
            (s, s),
            (2 * s, s),
            (s, 2 * s),
        ])

    return big_dims

def calculate_complete_dims(alpha, p, q, n, d):
    '''
    alpha: P(element belongs in the planted block | element observed) = alpha
    p: P(element missing) = p
    q: realtive aspect ratio
    n: no of rows in the big matrix
    d: no of columns in the big matrix

    Complete matrix has x rows and y columns.
    
    Area of the big matrix A_p = nd
    Aspect ratio of the big matrix R = n/d
    Aspect ratio of the planted matrix r = qR = x/y

    Function returns (x,y) if it is possible to be planted in (n,d) matrix.
    '''
    # feasibility condition
    beta = alpha * (1 - p)
    if not beta <= q <= 1/beta:
        return None
    
    y = d * np.sqrt(alpha * (1 - p) / q)
    x = n * np.sqrt(alpha * q * (1 - p))

    return (int(x), int(y))

def generate_matrix_with_planted(n, d,
                                 x, y,
                                 seed,
                                 frac_missing
                                ):
    '''
    n: no of rows
    d: no of cols
    seed: the seed for the random no generator
    frac_missing: P(element is missing) = frac_missing

    Planted matrix has x rows and y columns.
    '''
    rng = np.random.default_rng(seed)
    
    big = rng.standard_normal(size=(n, d))
    complete = rng.standard_normal(size=(x, y))

    rows = np.sort(rng.choice(n, x, replace=False))
    cols = np.sort(rng.choice(d, y, replace=False))
    
    big[np.ix_(rows, cols)] = complete

    not_planted_mask = extract_mask(big, rows, cols)
    apply_missingness(big, not_planted_mask, frac_missing, rng)

    return big

def basic_prune_to_complete(matrix: np.ndarray):
    # convert to boolean missing
    matrix = np.isnan(matrix)
    # extract dimensions
    no_rows, no_cols = matrix.shape

    # copy the matrix to be pruned
    current = matrix.copy()

    # the total number of missing elements
    no_missing = current.sum()

    row_idx = np.arange(no_rows)
    col_idx = np.arange(no_cols)

    iteration_info = []
   
    i = 0 
    while no_rows > 0 and no_cols > 0 and no_missing > 0:
        # missing elements across rows
        rows_missing = current.sum(axis=1)
        # missing elements across columns
        cols_missing = current.sum(axis=0)
        # check if contamination is worse in rows or columns
        # remove a column
        if max(rows_missing / no_cols) <= max(cols_missing / no_rows):
            # extract the worst column index
            worst_col = np.argmax(cols_missing)
            # remove from index
            col_idx = np.delete(col_idx, worst_col)
            
        # remove a row
        else:
            # extract the worst row index
            worst_row = np.argmax(rows_missing)
            row_idx = np.delete(row_idx, worst_row)

        current = matrix[np.ix_(row_idx, col_idx)]
        no_rows, no_cols = current.shape
        no_missing = current.sum()
        
        iteration_info.append({
            "i": i,
            "no_pruned_rows": matrix.shape[0] - no_rows,
            "no_pruned_cols": matrix.shape[1] - no_cols,
            "eff_lr": -1,
            "missing_no": no_missing,
            "missing_frac": no_missing/matrix.size
        })

        i+= 1
        
    if len(row_idx) > 0 and len(col_idx) > 0:
        return row_idx, col_idx, pd.DataFrame(iteration_info)
    else:
        return None, None, pd.DataFrame(iteration_info)

def basic_prune_to_complete_with_completion(matrix: np.ndarray):
    # convert to boolean missing
    matrix = np.isnan(matrix)
    # extract dimensions
    no_rows, no_cols = matrix.shape

    # copy the matrix to be pruned
    current = matrix.copy()

    # the total number of missing elements
    no_missing = current.sum()

    row_idx = np.arange(no_rows)
    col_idx = np.arange(no_cols)

    iteration_info = []
   
    i = 0 
    while no_rows > 0 and no_cols > 0 and no_missing > 0:
        # missing elements across rows
        rows_missing = current.sum(axis=1)
        # missing elements across columns
        cols_missing = current.sum(axis=0)
        # check if contamination is worse in rows or columns
        # remove a column
        if max(rows_missing / no_cols) <= max(cols_missing / no_rows):
            # extract the worst column index
            worst_col = np.argmax(cols_missing)
            # remove from index
            col_idx = np.delete(col_idx, worst_col)
            
        # remove a row
        else:
            # extract the worst row index
            worst_row = np.argmax(rows_missing)
            row_idx = np.delete(row_idx, worst_row)

        # reintroduce rows complete over the column subset
        row_complete_mask = matrix[:, col_idx].sum(axis=1) == 0
        row_idx = np.union1d(row_idx, np.flatnonzero(row_complete_mask))

        # reintroduce columns complete over the row subset
        col_complete_mask = matrix[row_idx, :].sum(axis=0) == 0
        col_idx = np.union1d(col_idx, np.flatnonzero(col_complete_mask))

        current = matrix[np.ix_(row_idx, col_idx)]
        no_missing = current.sum()
        
        iteration_info.append({
            "i": i,
            "no_pruned_rows": no_rows - current.shape[0],
            "no_pruned_cols": no_cols - current.shape[1],
            "eff_lr": -1,
            "missing_no": no_missing,
            "missing_frac": no_missing/matrix.size
        })

        no_rows, no_cols = current.shape

        i+= 1
        
    if len(row_idx) > 0 and len(col_idx) > 0:
        return row_idx, col_idx, pd.DataFrame(iteration_info)
    else:
        return None, None, pd.DataFrame(iteration_info)

def adaptive_prune_to_complete(
    data: pd.DataFrame,
    max_iter:int = 100,
    shrink_coeff:float = 10
):
    n_rows, n_cols = data.shape
    if n_rows == 0 or n_cols == 0:
        raise ValueError("Supplied matrix is degenerate")
    
    if shrink_coeff <= 0:
        raise ValueError("shrink_coeff must be positive")

    if shrink_coeff < 1:
        raise ValueError("shrink_coeff should be >= 1")

    if max_iter <= 0:
        raise ValueError("max_iter must be > 0")

    if not data.index.is_unique:
        raise ValueError("data index must be unique")

    if not data.columns.is_unique:
        raise ValueError("data columns must be unique")

    X = data.copy(deep=False)

    # container for benchmarking info
    # i: iteration no
    # no_pruned_cols: no of pruned columns
    # no_pruned_rows: no of pruned rows
    # eff_lr: effective learning rate
    iteration_info = []
    
    # iteration count
    i = 1
    while(i <= max_iter):
        ### remove columns ###

        # pick threshold
        # fraction of elements in the least populated column
        col_avail = X.notna().sum(axis=0)
        min_fraction_col = (col_avail/n_rows).min()
        # fraction of missing + shrunk learning_rate
        col_threshold = min_fraction_col + (1 - min_fraction_col)/(shrink_coeff * i)

        # save the number of old columns
        n_cols_old = X.shape[1]
        
        # prune columns
        X = X.loc[:, col_avail >= col_threshold * n_rows]
        
        # update column count
        n_cols = X.shape[1]
        # if all columns were pruned
        if n_cols == 0:
            iteration_info.append({
                "i": i,
                "no_pruned_rows": 0,
                "no_pruned_cols": n_cols_old - n_cols,
                "eff_lr": 1/(shrink_coeff * i)
            })
            iteration_info = pd.DataFrame(iteration_info)
            return None, None, iteration_info

        ### remove rows ###
        
        # pick threshold
        # fraction of elements in the least populated row
        row_avail = X.notna().sum(axis=1)
        min_fraction_row = (row_avail/n_cols).min()
        # fraction of missing + shrunk learning_rate
        row_threshold = min_fraction_row + (1 - min_fraction_row)/(shrink_coeff * i)

        n_rows_old = X.shape[0]
        
        # prune rows
        X = X.loc[row_avail >= row_threshold * n_cols, :]

        # update row count
        n_rows = X.shape[0]
        # if all rows were pruned
        if n_rows == 0:
            iteration_info.append({
                "i": i,
                "no_pruned_rows": n_rows_old - n_rows,
                "no_pruned_cols": n_cols_old - n_cols,
                "eff_lr": 1/(shrink_coeff * i)
            })
            iteration_info = pd.DataFrame(iteration_info)
            return None, None, iteration_info

        missing = X.isna().sum().sum()
        total = X.shape[0] * X.shape[1]

        iteration_info.append({
            "i": i,
            "no_pruned_rows": n_rows_old - n_rows,
            "no_pruned_cols": n_cols_old - n_cols,
            "eff_lr": 1/(shrink_coeff * i),
            "missing_no": missing,
            "missing_frac": missing/total
        })

        if missing == 0: break
        
        i += 1

    iteration_info = pd.DataFrame(iteration_info)
    if i <= max_iter: 
        return X.index.copy(), X.columns.copy(), iteration_info
    else:
        warnings.warn(
            f"{max_iter} iterations reached without convergence. "
            f"Remaining matrix has shape {X.shape} and {missing} missing values.",
            RuntimeWarning
        )
        return None, None, iteration_info

def adaptive_prune_to_complete_with_completion(
    data: pd.DataFrame,
    max_iter:int = 100,
    shrink_coeff:float = 10
):
    n_rows, n_cols = data.shape
    if n_rows == 0 or n_cols == 0:
        raise ValueError("Supplied matrix is degenerate")
    
    if shrink_coeff <= 0:
        raise ValueError("shrink_coeff must be positive")

    if shrink_coeff < 1:
        raise ValueError("shrink_coeff should be >= 1")

    if max_iter <= 0:
        raise ValueError("max_iter must be > 0")

    if not data.index.is_unique:
        raise ValueError("data index must be unique")

    if not data.columns.is_unique:
        raise ValueError("data columns must be unique")

    X = data.copy(deep=False)

    # container for benchmarking info
    # i: iteration no
    # no_pruned_cols: no of pruned columns
    # no_pruned_rows: no of pruned rows
    # eff_lr: effective learning rate
    iteration_info = []
    
    # iteration count
    i = 1
    while(i <= max_iter):
        n_rows_old, n_cols_old = X.shape
        
        ### remove columns ###

        # pick threshold
        # fraction of elements in the least populated column
        col_avail = X.notna().sum(axis=0)
        min_fraction_col = (col_avail/n_rows).min()
        # fraction of missing + shrunk learning_rate
        col_threshold = min_fraction_col + (1 - min_fraction_col)/(shrink_coeff * i)

        # prune columns
        X = X.loc[:, col_avail >= col_threshold * n_rows]

        # update column count
        n_cols = X.shape[1]
        # if all columns were pruned
        if n_cols == 0:
            iteration_info.append({
                "i": i,
                "no_pruned_rows": 0,
                "no_pruned_cols": n_cols_old - n_cols,
                "eff_lr": 1/(shrink_coeff * i)
            })
            iteration_info = pd.DataFrame(iteration_info)
            return None, None, iteration_info


        # reintroduce rows which have availability over the subset of columns
        complete_row_mask = data.loc[:, X.columns].notna().all(axis=1)
        row_idx = X.index.union(data.index[complete_row_mask], sort=False)

        # reconstruct X from the original data
        X = data.loc[row_idx, X.columns]

        n_rows, n_cols = X.shape
        
        ### remove rows ###
        
        # pick threshold
        # fraction of elements in the least populated row
        row_avail = X.notna().sum(axis=1)
        min_fraction_row = (row_avail/n_cols).min()
        # fraction of missing + shrunk learning_rate
        row_threshold = min_fraction_row + (1 - min_fraction_row)/(shrink_coeff * i)
        
        # prune rows
        X = X.loc[row_avail >= row_threshold * n_cols, :]

        # update row count
        n_rows = X.shape[0]
        # if all rows were pruned
        if n_rows == 0:
            iteration_info.append({
                "i": i,
                "no_pruned_rows": n_rows_old - n_rows,
                "no_pruned_cols": n_cols_old - n_cols,
                "eff_lr": 1/(shrink_coeff * i)
            })
            iteration_info = pd.DataFrame(iteration_info)
            return None, None, iteration_info

        
        # reintroduce columns which are complete over the subset of rows
        col_complete_mask = data.loc[X.index, :].notna().all(axis=0)
        col_idx = X.columns.union(data.columns[col_complete_mask], sort=False)

        # reconstruct X from the original data
        X = data.loc[X.index, col_idx]

        n_rows, n_cols = X.shape

        missing = X.isna().sum().sum()
        total = X.size

        iteration_info.append({
            "i": i,
            "no_pruned_rows": n_rows_old - n_rows,
            "no_pruned_cols": n_cols_old - n_cols,
            "eff_lr": 1/(shrink_coeff * i),
            "missing_no": missing,
            "missing_frac": missing/total
        })

        if missing == 0: break
        
        i += 1

    iteration_info = pd.DataFrame(iteration_info)
    if i <= max_iter: 
        return X.index.copy(), X.columns.copy(), iteration_info
    else:
        warnings.warn(
            f"{max_iter} iterations reached without convergence. "
            f"Remaining matrix has shape {X.shape} and {missing} missing values.",
            RuntimeWarning
        )
        return None, None, iteration_info

def extract_mask(matrix, excluded_rows, excluded_cols):
    excluded_mask = np.zeros(matrix.shape, dtype=bool)
    excluded_mask[np.ix_(excluded_rows, excluded_cols)] = True
    return ~excluded_mask

def apply_missingness(matrix, available_mask, missingness, rng):
    # assumes applying missingness is possible
    no_missing = int(round(missingness * matrix.size))
    available_idx = np.flatnonzero(available_mask)

    missing_idx = rng.choice(available_idx, size = no_missing, replace = False)
    matrix.flat[missing_idx] = np.nan

# experiment assessment helper
def distinct_solutions(matrices):
    ds = matrices[matrices["found_area"] > 0]
    ds = ds.sort_values(by=['row_cnt','col_cnt', 'C', 'time_sec'], ascending=[False, False, True, True])
    return ds.groupby(by='row_cnt', as_index=False).first()

# experiment assessment helper
def Pareto_frontier(matrices):
    ds = distinct_solutions(matrices)
    ds = ds.sort_values(by=['row_cnt', 'col_cnt'], ascending=[False, False])
    ds.reset_index(inplace=True)
    pf_idx = np.zeros(len(ds), dtype=bool)
    no_rows = 0
    no_cols = 0
    for solution in ds.itertuples():
        if (solution.row_cnt > no_rows) or (solution.col_cnt > no_cols):
            no_rows = solution.row_cnt
            no_cols = solution.col_cnt
            pf_idx[solution.Index] = True
    return ds.loc[pf_idx]
