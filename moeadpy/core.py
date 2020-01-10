import numpy as np
from .components import \
    constraints, \
    decomposition, \
    neighboorhood, \
    scalarization, \
    stopping, \
    update, \
    variation

class _AliasDict(dict):
    def __setitem__(self, key, value):
        for ikey in key.split(";"):
            super().__setitem__(ikey, value)

    def __missing__(self, key):
        return key


aliases = _AliasDict({
    "simplex lattice design;Simplex-Lattice Design": "Sld",

})


def _invoke(component, method: str, *args, **kwargs):
    try:
        f = getattr(component, aliases[method])
    except AttributeError:
        raise AttributeError(
            f"The method '{method}' is not included in '{component}'. "
            "Please re-check your spelling.")
    return f(*args, **kwargs)


def _denormalize_pop(pop, lower_limit, value_range):
    return lower_limit + (pop * value_range)


class Moead:
    def __init__(self,
                 problem,
                 n_objectives: int,
                 x_min: np.ndarray,
                 x_max: np.ndarray,
                 decomp=decomposition.Sld(h=99),
                 neighborhood=neighboorhood.TNearestNeighbors(
                     T=20, delta_prob=1),
                 variation=(
                     variation.SimulatedBinaryCrossover(eta=20, pc=1),
                     variation.PolynomialMutation(eta=20, prob_m="n"),
                     variation.ApplyMaxMin(min=0, max=1)
                    ),
                 solution_scaling=None,
                 scalarization=scalarization.WeightedTchebycheff(),  # scalar aggregation function
                 constraint=None,
                 update=update.StandardUpdate(),
                 stop_criteria=stopping.MaxIter(200),
                 seed=None,
                 ):
        self.problem = problem
        self.n_objectives = n_objectives

        self.x_min = x_min
        self.x_max = x_max
        self._lower_limit = None
        self._upper_limit = None
        self._value_range = None
        if len(x_max) != len(x_min):
            raise ValueError("'x_max' and 'x_min' don't share the same length."
                             " Please re-check your parameters.")
        self.solution_length = len(x_max)

        self.decomp = decomp
        self.scalarization = scalarization
        self.neighborhood = neighborhood
        self.variation = variation
        self.update = update
        self.constraint = constraint
        self.solution_scaling = solution_scaling
        self.stop_criteria = stop_criteria
        self.seed = seed
        self.results = None

    def _set_lower_upper_limits(self, n_rows):
        self._lower_limit = np.tile(self.x_min, (n_rows, 1))
        self._upper_limit = np.tile(self.x_max, (n_rows, 1))
        self._value_range = self._upper_limit - self._lower_limit

    def evaluate_solutions(self, population, constraints=None):
        '''
        returns: matrix =>
                    columns -> each subproblem
                    rows -> each candidate solution
        '''
        if not self._lower_limit:
            self._set_lower_upper_limits(population.shape[0])
        denorm_pop = _denormalize_pop(
            population, self._lower_limit, self._value_range)
        eva: np.ndarray = None
        try:
            iter(self.problem)  # check if it's a list of problems
        except TypeError:
            try:
                # eva = np.array([self.problem(indiv, weights)
                #                 for indiv, weights in zip(denorm_pop, weight_matrix)])
                eva = np.asarray(self.problem(denorm_pop))  # shape = popul_size x n_objectives
            except TypeError:
                raise TypeError(
                    "This moead object's 'problem'(s) is either not an array-like of callable(s) or just simply wrong!")
        else:
            # eva = np.array([[f(indiv) for f in self.problem] for indiv in denorm_pop])
            eva = np.asarray([f(denorm_pop) for f in self.problem]).T
        return eva

    def compute(self):
        np.random.seed(self.seed)

        # 1. Generate initial population, weights, and evaluations
        weights: np.ndarray = self.decomp(self.n_objectives)
        population_size = weights.shape[0]

        self._set_lower_upper_limits(population_size)
        population = np.random.rand(population_size, self.solution_length)

        new_eva = self.evaluate_solutions(population, self.constraint)

        stop = False
        iteration = 0
        while not stop:
            iteration += 1
        # 2. Define or update neighborhoods
            neighb: np.ndarray = None
            if self.neighborhood.mode == "weights":
                neighb = self.neighborhood(weights, iter)
            else:
                neighb = self.neighborhood(population, iter)
        # 3. Copy the incumbent solution in preparation for the new one
            new_population = np.array(population)
        # 4. Variation operators
            for variator in self.variation:
                if isinstance(variator, variation.Crossover):
                    new_population = variator(new_population, self.neighborhood)
                else:
                    new_population = variator(new_population)
        # 5. Re-evaluation and scalar aggregation functions
            old_eva = new_eva
            new_eva = self.evaluate_solutions(new_population)
            combined_eva = np.concatenate((old_eva, new_eva), axis=0)

            min_points = np.min(combined_eva, axis=0)  # ideal points
            max_points = np.max(combined_eva, axis=0)  # nadir points

            flattened_neighborhood_ind = neighb.flatten() # len = neigh.shape[0] * neigh.shape[1]
            neighborhood_evaluations = new_eva[flattened_neighborhood_ind]
            # Replicate each weight vector T (neighborhood size) times
            replicated_weights = np.repeat(weights, neighb.shape[1], axis=0)
            # neighborhood_evaluations.shape == replicated_weights.shape

            Z_neigh = self.scalarization(neighborhood_evaluations, replicated_weights, min_points, max_points)
            Z_neigh = Z_neigh.reshape((neighb.shape[0], neighb.shape[1]))
            Z_old_eva = self.scalarization(old_eva, weights, min_points, max_points)  # len = neighb.shape[0]
            Z_old_eva = Z_old_eva.reshape((1, len(Z_old_eva))).T
            # Z_full.shape = (neighb.shape[0], neighb.shape[1] + 1)
            # => coefficients for all the evaluation matrices made above
            Z_full = np.concatenate((Z_neigh, Z_old_eva), axis=1)
        # 6. Constraints and index ordering
            Z_sort_ind = np.argsort(Z_full, axis=1)  # shape's the same as Z_full; indexes
        # 7. Update
            population, eva = self.update(new_population, population, new_eva, old_eva, neighb, Z_sort_ind) 
        # 8. Termination check
            stop = self.stop_criteria(iteration)
        self.results = population, eva
        return population, eva
