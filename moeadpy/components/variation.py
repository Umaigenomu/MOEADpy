import numpy as np
from . import Component, neighboorhood

# Every variator/localsearch class in this file must inherit this (composite pattern)
class SingleVariatorList(Component):
    def __iter__(self):
        self.iterated = False
        return self

    def __next__(self):
        if not self.iterated:
            self.iterated = True
            return self
        else:
            raise StopIteration


class Crossover(SingleVariatorList):
    def __call__(self, population, neighbors):
        # A 'Crossover' class must implement __call__ with two parameters:
        # one for the population and one for the neighboorhood."
        pass


class SimulatedBinaryCrossover(Crossover):
    def __init__(self, eta, pc, eps):
        self.eta = eta
        self.pc = pc
        self.eps = eps

    def apply_crossover(self, v1, v2):  # always inplace
        for i, (x1, x2) in enumerate(zip(v1, v2)):
            if np.random.random() <= self.pc and (np.abs(v1[i] - v2[i]) > self.eps):
                rand = np.random.random()
                if rand <= 0.5:
                    beta = 2. * rand
                else:
                    beta = 1. / (2. * (1. - rand))
                beta **= 1. / (self.eta + 1.)
                v1[i] = 0.5 * (((1 + beta) * x1) + ((1 - beta) * x2))
                v2[i] = 0.5 * (((1 - beta) * x1) + ((1 + beta) * x2))
        return v1, v2

    def __call__(self, population: np.ndarray, neighbors: neighboorhood.Neighborhood):
        for i in range(population.shape[0]):
            neighbor = neighbors.get_neighbor(i)
            population[i], neighbor = self.apply_crossover(population[i], neighbor)


class PolynomialMutation(SingleVariatorList):
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


class ApplyMaxMin(SingleVariatorList):
    def __init__(self, min=0, max=1):
        self.min = min
        self.max = max

    def __call__(self, population):
        population[population < self.min] = self.min
        population[population > self.max] = self.max
