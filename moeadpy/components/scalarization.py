from . import Component
import numpy as np


class Scalarization(Component):
    def __call__(self, eval, weights, min_points, max_points):
        '''
        Every Scalarization class must take into account the 4 parameters above
        when implementing __call__
        '''
        pass


class WeightedTchebycheff(Scalarization):
    def __init__(self, eps=1/10**16):
        self.eps = eps

    def __call__(self, evaluations: np.ndarray, weights: np.ndarray, min_points: np.ndarray, *args):
        '''
        return: 1D ndarray the length of evaluations.shape[0]
        '''
        min_p_matrix = np.tile(min_points, (evaluations.shape[0], 1))
        weighted_scaled_eva = weights * (evaluations - min_p_matrix + self.eps)
        return np.max(weighted_scaled_eva, axis=1)
