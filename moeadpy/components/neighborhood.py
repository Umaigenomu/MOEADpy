from . import Component
from sklearn.neighbors import NearestNeighbors
import numpy as np


class Neighborhood(Component):
    def get_neighbor(self, index):
        raise NotImplementedError(
            "This Neighboorhood class hasn't implemented 'get_neighbor' yet.")
    
    def get_neighbor_probability_matrix(self):
        raise NotImplementedError(
            "This Neighboorhood class hasn't implemented 'get_neighbor_probability_matrix' yet.")


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
        self.neighbor_prob = None

    def __call__(self, data: np.ndarray, iteration: int) -> np.ndarray:
        if self.mode == "weights" and iteration > 1:
            return self.neighbors

        nbr_indexes = NearestNeighbors(n_neighbors=self.T, n_jobs=-1)\
            .fit(data)\
            .kneighbors(return_distance=False)

        self.neighbors = nbr_indexes
        self.neighbor_sets = [set(neighbors) for neighbors in nbr_indexes]
        return nbr_indexes

    def get_neighbor(self, ind):
        if self.delta_prob == 1 or np.random.random() <= self.delta_prob:
            return np.random.choice(self.neighbors[ind])
        else:
            i = np.random.choice(self.neighbors.shape[0])
            while i in self.neighbor_sets[ind]:
                i = np.random.choice(self.neighbors.shape[0])
            return i

    def get_neighbor_probability_matrix(self):
        if self.neighbor_prob is not None:
            return self.neighbor_prob
        pop_size = self.neighbors.shape[0]
        self.neighbor_prob = np.array([
                [self.delta_prob / self.T
                if i in self.neighbor_sets[ni]
                else (1 - self.delta_prob) / (pop_size - 1 - self.T)
                for i in range(pop_size)]
            for ni in range(pop_size)])
        for ni in range(pop_size):
            self.neighbor_prob[ni, ni] = 0.0
        return self.neighbor_prob
