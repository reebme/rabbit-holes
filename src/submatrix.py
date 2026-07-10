import pandas as pd
import numpy as np

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
