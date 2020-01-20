from . import Component
import numpy as np


class StandardUpdate(Component):
    def __call__(self, new_pop, old_pop, new_eval, old_eval, neighb, sel_ind):
        best_ind = sel_ind[:, 0]
        next_pop = np.array([new_pop[neighb[i, best_ind[i]]]
                                if best_ind[i] < neighb.shape[1]
                                else old_pop[i]
                                for i in range(new_pop.shape[0])])
        next_eval = np.array([new_eval[neighb[i, best_ind[i]]]
                                if best_ind[i] < neighb.shape[1]
                                else old_eval[i]
                                for i in range(new_eval.shape[0])])
        return next_pop, next_eval
