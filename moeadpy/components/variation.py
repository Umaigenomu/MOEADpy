import numpy as np
from . import Component, neighborhood, SingleItemIterator


class Crossover(SingleItemIterator):
    def generate_crossover_pairs(self, population, neighbor_selec_probs):
        pop_indexes = np.array([i for i in range(population.shape[0])])
        # 2D; shape = (population.shape[0], 2)
        selected_neigh_inds = np.array([np.random.choice(pop_indexes, size=2, replace=False, p=probs)
                                        for probs in neighbor_selec_probs])
        selected_neighs0 = population[selected_neigh_inds[:, 0]]
        selected_neighs1 = population[selected_neigh_inds[:, 1]]
        return selected_neighs0, selected_neighs1


    def __call__(self, population, neighbors):
        # A 'Crossover' class must implement __call__ with two parameters:
        # one for the population and one for the neighboorhood."
        pass


class SimulatedBinaryCrossover(Crossover):
    def __init__(self, eta, pc, eps=1/10**6, use_prob_matrix=True):
        self.eta = eta
        self.pc = pc
        self.eps = eps
        self.use_prob_matrix = use_prob_matrix

    def apply_sbx_vectors(self, v1, v2):  # always inplace
        for i in range(len(v1)):
            x1, x2 = v1[i], v2[i]
            if np.random.random() <= self.pc and (np.abs(x1 - x2) > self.eps):
                rand = np.random.random()
                if rand <= 0.5:
                    beta = 2. * rand
                else:
                    beta = 1. / (2. * (1. - rand))
                beta **= 1. / (self.eta + 1.)
                v1[i] = 0.5 * (((1 + beta) * x1) + ((1 - beta) * x2))
                v2[i] = 0.5 * (((1 - beta) * x1) + ((1 + beta) * x2))
        return v1, v2

    def apply_sbx_matrices(self, selected_neighs_0, selected_neighs_1):
        rand = np.random.rand(selected_neighs_0.shape[0], selected_neighs_0.shape[1])
        not_recombined = (rand > self.pc) | (
                np.abs(selected_neighs_0-selected_neighs_1) < self.eps)
        recombined = (not_recombined == False)

        beta = np.random.rand(selected_neighs_0.shape[0], selected_neighs_0.shape[1])
        beta[beta <= 0.5] = 2. * beta[beta <= 0.5]
        beta[beta > 0.5] = 1. / (2. * (1 - beta[beta > 0.5]))
        beta **= 1. / (self.eta + 1.)

        recomb_neighs_0 = selected_neighs_0 * not_recombined \
                          + (0.5 * (((1 + beta) * selected_neighs_0) + ((1 - beta) * selected_neighs_1))) \
                              * recombined
        recomb_neighs_1 = selected_neighs_1 * not_recombined \
                          + (0.5 * (((1 - beta) * selected_neighs_0) + ((1 + beta) * selected_neighs_1))) \
                              * recombined

        return recomb_neighs_0, recomb_neighs_1


    def __call__(self, population: np.ndarray, neighbors: neighborhood.Neighborhood):
        if not self.use_prob_matrix:
            for i in range(population.shape[0]):
                n1 = neighbors.get_neighbor(i)
                n2 = neighbors.get_neighbor(i)
                while n2 == n1:
                    n2 = neighbors.get_neighbor(i)
                population[n1], population[n2] = self.apply_sbx_vectors(population[n1], population[n2])
            return population
        else:
            # 2D; (population.shape[0], population.shape[0])
            neigh_selec_probs = neighbors.get_neighbor_probability_matrix()
            # Each 2D, population's shape
            selected_neighs_0, selected_neighs_1 = self.generate_crossover_pairs(population, neigh_selec_probs)
            recomb_neighs_0, recomb_neighs_1 = self.apply_sbx_matrices(selected_neighs_0, selected_neighs_1)

            choose_0 = np.random.rand(population.shape[0]).reshape((population.shape[0], 1)) <= 0.5
            return (recomb_neighs_0 * choose_0) + (recomb_neighs_1 * (choose_0 == False))


class PolynomialMutation(SingleItemIterator):
    def __init__(self, eta, prob_m="n"):
        self.eta = eta
        self.prob_m = prob_m

    def _mutate(self, x):
        # x is always scaled to [0, 1]
        delta_1 = x
        delta_2 = 1 - x
        rand = np.random.random()
        mut_pow = 1.0 / (self.eta + 1.)
        if rand < 0.5:
            delta_q = (2.0 * rand + (1.0 - 2.0 * rand) * (1.0 - delta_1) ** (self.eta + 1)) ** mut_pow - 1.0
        else:
            delta_q = 1.0 - (2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (1.0 - delta_2) ** (self.eta + 1)) ** mut_pow
        x += delta_q
        # x = min(max(x, 0), 1)
        return x

    def __call__(self, population: np.ndarray):
        mut_pow = 1.0 / (self.eta + 1.)
        if self.prob_m == "n":
            self.prob_m = 1 / population.shape[1]  # number of columns (solution length)

        mutate_or_not = np.random.rand(population.shape[0], population.shape[1]) 
        delta_l_or_u = np.random.rand(population.shape[0], population.shape[1])

        delta_l = delta_l_or_u[(mutate_or_not <= self.prob_m) & (delta_l_or_u < 0.5)]
        population[(mutate_or_not <= self.prob_m) & (delta_l_or_u < 0.5)] += \
                ((2.0 * delta_l + (1.0 - 2.0 * delta_l) \
                    * (1.0 - population[(mutate_or_not <= self.prob_m) & (delta_l_or_u < 0.5)]) \
                        ** (self.eta + 1)) ** mut_pow - 1.0)
                
        delta_u = delta_l_or_u[(mutate_or_not <= self.prob_m) & (delta_l_or_u >= 0.5)]
        population[(mutate_or_not <= self.prob_m) & (delta_l_or_u >= 0.5)] += \
                (1.0 - (2.0 * (1.0 - delta_u) + 2.0 * (delta_u - 0.5) \
                    * (1.0 - (1 - population[(mutate_or_not <= self.prob_m) & (delta_l_or_u >= 0.5)])) \
                        ** (self.eta + 1)) ** mut_pow)

        return population


class ApplyMaxMin(SingleItemIterator):
    def __init__(self, min=0, max=1):
        self.min = min
        self.max = max

    def __call__(self, population):
        population[population < self.min] = self.min
        population[population > self.max] = self.max
        return population
