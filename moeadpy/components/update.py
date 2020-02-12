from . import Component
from . import MoeadSet
import numpy as np


class Update(Component):
    def __call__(self, mset: MoeadSet):
        # An 'Update' class must implement __call__ with this param
        pass


class StandardUpdate(Update):
    def __call__(self, mset:MoeadSet):
        best_ind = mset.sort_inds[:, 0]
        next_pop = np.array([mset.x[mset.neighb[i, best_ind[i]]]
                                if best_ind[i] < mset.neighb.shape[1]
                                else mset.xt[i]
                                for i in range(mset.x.shape[0])])
        next_eval = np.array([mset.y[mset.neighb[i, best_ind[i]]]
                                if best_ind[i] < mset.neighb.shape[1]
                                else mset.yt[i]
                                for i in range(mset.y.shape[0])])
        return next_pop, next_eval
