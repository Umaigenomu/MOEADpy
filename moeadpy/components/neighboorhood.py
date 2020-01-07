from . import Component
from sklearn.neighbors import NearestNeighbors
import numpy as np

class Neighborhood(Component):
    def get_neighbor(self, index):
        raise NotImplementedError("This Neighboorhood class hasn't implemenented 'get_neighbor' yet.")


class TNearestNeighbors(Neighborhood):
    def __init__(self, T, delta_prob, mode="weights"):
        self.T = T
        self.delta_prob = delta_prob
        self.mode = mode
        if mode not in ("weights", "solutions"):
            raise ValueError(
                "TNearestNeighbors's 'mode' attribute only"
                " accepts the following values: 'weights', 'solutions'")
        self.neighbors = None
        self.neighbor_sets = None

    def __call__(self, data: np.ndarray, iter: int) -> np.ndarray:
        if self.mode == "weights" and iter > 1:
            return self.neighbors

        nbr_indexes = NearestNeighbors(n_neighbors=self.T, n_jobs=-1)\
                        .fit(data)\
                        .kneighbors(return_distance=False)

        self.neighbors = nbr_indexes
        self.neighbor_sets = [set(neighbors) for neighbors in np.array(nbr_indexes)]
        return nbr_indexes

    def get_neighbor(self, ind):
        if self.delta_prob == 1 or np.random.random() <= self.delta_prob:
            return np.random.choice(self.neighbors[ind])
        else:
            i = np.random.choice(self.neighbors.shape[0])
            while i in self.neighbor_sets[ind]:
                i = np.random.choice(self.neighbors.shape[0])
            return i
