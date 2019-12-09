'''

'''
import numpy as np
import logging


def sld(n_objectives, h=99) -> np.ndarray:
    sequence = [i/h for i in np.arange(h)] + [h/h]
    sequences = [sequence for _ in range(n_objectives-1)]

    n_rows = len(sequence) ** (n_objectives-1)
    if n_rows > 5000:
        logging.warn(f"The following configuration: sld({n_objectives}, {h})"
                     f" will generate a very large number of subproblems (={n_rows})."
                     "This may considerally slow down the program.")
    combinations = np.array(np.meshgrid(*sequences))\
                     .reshape(n_objectives-1, n_rows).T

    last_col = np.expand_dims(
        np.repeat(1, n_rows) - np.sum(combinations, axis=1), axis=0
    ).T
    sub_problems = np.concatenate((combinations, last_col), axis=1)
    valid_sub_problems = sub_problems[sub_problems[:, -1] >= 0]

    return valid_sub_problems
