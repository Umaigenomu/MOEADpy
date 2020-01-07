'''

'''
import numpy as np
import logging
from . import Component


class Sld(Component):
    def __init__(self, h=99):
        self.h = h

    def __call__(self, n_objectives) -> np.ndarray:
        sequence = [i / self.h for i in np.arange(self.h)] + [self.h / self.h]
        sequences = [sequence for _ in range(n_objectives - 1)]

        n_rows = len(sequence) ** (n_objectives - 1)
        if n_rows > 5000:
            logging.warn(f"The following configuration: sld({n_objectives}, {self.h})"
                         f" will generate a very large number of subproblems (~={n_rows})."
                          " Considering the algorithm, this may significantly slow down the program.")
        combinations = np.array(np.meshgrid(*sequences))\
            .reshape(n_objectives - 1, n_rows).T

        last_col = np.expand_dims(
            np.repeat(1, n_rows) - np.sum(combinations, axis=1), axis=0
        ).T
        sub_problems = np.concatenate((combinations, last_col), axis=1)
        valid_sub_problems = sub_problems[sub_problems[:, -1] >= 0]

        return valid_sub_problems
