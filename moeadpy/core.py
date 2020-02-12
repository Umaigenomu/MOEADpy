import numpy as np
import logging
from components import \
    constraints, \
    decomposition, \
    neighborhood, \
    scalarization, \
    stopping, \
    update, \
    variation, \
    MoeadSet


# def _invoke(component, method: str, *args, **kwargs):
#     try:
#         f = getattr(component, aliases[method])
#     except AttributeError:
#         raise AttributeError(
#             f"The method '{method}' is not included in '{component}'. "
#             "Please re-check your spelling.")
#     return f(*args, **kwargs)


def _denormalize_pop(pop, lower_limit, value_range):
    return lower_limit + (pop * value_range)


class Moead:
    def __init__(self,
                 problem,
                 n_objectives: int,
                 x_min: np.ndarray,
                 x_max: np.ndarray,
                 decomp=decomposition.Sld(h=99),
                 neighborhood=neighborhood.TNearestNeighbors(
                     T=20, delta_prob=1),
                 variation=(
                     variation.SimulatedBinaryCrossover(eta=20, pc=1),
                     variation.PolynomialMutation(eta=20, prob_m="n"),
                     variation.ApplyMaxMin(min=0, max=1)
                    ),
                 solution_scaling=None,
                 scalarization=scalarization.WeightedTchebycheff(),  # scalar aggregation function
                 constraint=None,
                 constraint_method=None,
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
        self.constraint_method = constraint_method
        self.solution_scaling = solution_scaling
        self.stop_criteria = stop_criteria
        self.seed = seed
        self.mset = MoeadSet()
        self.results = None

    def _set_lower_upper_limits(self, n_rows):
        self._lower_limit = np.tile(self.x_min, (n_rows, 1))
        self._upper_limit = np.tile(self.x_max, (n_rows, 1))
        self._value_range = self._upper_limit - self._lower_limit

    def _update_mset(self, new_population, population, eva, old_eva, v, vt, neighb, Z_full, Z_sort_ind, iteration):
        self.mset.x = new_population
        self.mset.xt = population
        self.mset.y = eva
        self.mset.yt = old_eva
        self.mset.v = v
        self.mset.vt = vt
        self.mset.neighb = neighb
        self.mset.Z_full = Z_full
        self.mset.sort_inds = Z_sort_ind
        self.mset.iteration = iteration

    def evaluate_solutions(self, population):
        '''
        returns: matrix =>
                    columns -> each subproblem
                    rows -> each candidate solution
        '''
        if self._lower_limit is None:
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

        if self.constraint:
            violations = self.constraint(denorm_pop)
        else:
            violations = None

        return eva, violations

    def sort_by_selection_quality(self, Z_full, vt, v, neighb):
        if self.constraint is None or self.constraint_method is None:
            Z_sort_ind = np.argsort(Z_full, axis=1)
        else:
            vsum = np.sum(v, axis=1)  # shape = neighb.shape[0]
            V_new = vsum[neighb.flatten()]
            V_new = V_new.reshape((neighb.shape[0], neighb.shape[1]))

            V_old = np.sum(vt, axis=1)
            V_old = V_old.reshape((1, neighb.shape[0])).T
            
            V_full = np.concatenate((V_new, V_old), axis=1)

            Z_sort_ind = self.constraint_method(Z_full, V_full, neighb)
        return Z_sort_ind

    def compute(self):
        np.random.seed(self.seed)

        # 1. Generate initial population, weights, and evaluations
        weights: np.ndarray = self.decomp(self.n_objectives)
        population_size = weights.shape[0]

        self._set_lower_upper_limits(population_size)
        population = np.random.rand(population_size, self.solution_length)

        eva, v = self.evaluate_solutions(population)
        if eva.shape[1] != self.n_objectives:
            logging.warn(f"User defined 'n_objectives' and the actual number of objectives inferred from the problem "
                          "differ. Changing n_objectives to {new_eva.shape[1]} and regenerating weights.")
            self.n_objectives = eva.shape[1]
            weights = self.decomp(self.n_objectives)
        
        self.mset.weights = weights

        stop = False
        iteration = 0
        while not stop:
            iteration += 1
        # 2. Define or update neighborhoods
            neighb: np.ndarray = None
            if self.neighborhood.mode == "weights":
                neighb = self.neighborhood(self.mset.weights)
            else:
                neighb = self.neighborhood(population, iteration)
        # 3. Copy the incumbent solution in preparation for the new one
            new_population = np.array(population)
        # 4. Variation operators
            for variator in self.variation:
                if isinstance(variator, variation.Crossover):
                    new_population = variator(new_population, self.neighborhood)
                else:
                    new_population = variator(new_population)
        # 5. Re-evaluation and scalar aggregation functions
            old_eva = eva
            vt = v
            eva, v = self.evaluate_solutions(new_population)
            combined_eva = np.concatenate((old_eva, eva), axis=0)

            min_points = np.min(combined_eva, axis=0)  # ideal points
            max_points = np.max(combined_eva, axis=0)  # nadir points

            flattened_neighborhood_ind = neighb.flatten() # len = neigh.shape[1] * neigh.shape[0]
            neighborhood_evaluations = eva[flattened_neighborhood_ind]
            # Replicate each weight vector T (neighborhood size) times
            replicated_weights = np.repeat(self.mset.weights, neighb.shape[1], axis=0)

            # neighborhood_evaluations.shape == replicated_weights.shape
            Z_neigh = self.scalarization(neighborhood_evaluations, replicated_weights, min_points, max_points)
            Z_neigh = Z_neigh.reshape((neighb.shape[0], neighb.shape[1]))
            Z_old_eva = self.scalarization(old_eva, self.mset.weights, min_points, max_points)  # len = neighb.shape[0]
            Z_old_eva = Z_old_eva.reshape((1, Z_old_eva.shape[0])).T
            # Z_full.shape = (population size, neighb.shape[1] + 1)
            # => coefficients for all the evaluation matrices made above
            Z_full = np.concatenate((Z_neigh, Z_old_eva), axis=1)
        # 6. Constraints and index ordering
            Z_sort_ind = self.sort_by_selection_quality(Z_full, vt, v, neighb)  # shape's the same as Z_full (indexes)
        # 7. Update
            self._update_mset(
                new_population, population, eva, old_eva, v, vt,
                neighb, Z_full, Z_sort_ind, iteration
                )
            population, eva = self.update(self.mset)
        # 8. Termination check
            for criteria in self.stop_criteria:
                stop = stop or criteria(self.mset)
        self.results = population, eva
        return population, eva
