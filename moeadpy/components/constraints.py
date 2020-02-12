from . import Component
import numpy as np

class Constraint(Component):
    def __init__(self, gs: list, hs: list, hepsilon=0):
        self.gs = gs  # inequality constr funcs => g(x) <= 0
        self.hs = hs  # equality constr funcs ==> h(x) = 0
        self.hepsilon = hepsilon  # tolerance for equality constr
        self.violations = None

    def __call__(self, population):
        g_all = None
        h_all = None

        if self.gs is not None:
            g_all = []
            for g in self.gs:
                g_res = g(population)
                g_all.append(g_res)
            g_all = np.asarray(g_all).reshape((1, len(self.gs))).T
            g_all[g_all <= 0] = 0

        if self.hs is not None:
            h_all = []
            for h in self.hs:
                h_res = h(population)
                h_all.append(h_res)
            h_all = np.asarray(h_all).reshape((1, len(self.hs))).T
            h_all = np.abs(h_all) - self.hepsilon
            h_all[h_all <= 0] = 0

        if g_all is not None or h_all is not None:
            if h_all is None:
                self.violations = g_all
            elif g_all is None:
                self.violations = h_all
            else:
                self.violations = np.concatenate((g_all, h_all), axis=1)
        return self.violations


class ConstraintMethod(Component):
    def __call__(self, Z_full, V_full, neighb):
        # Any class that inherits this must include the above parameters in the above order
        pass


class Penalty(ConstraintMethod):
    def __init__(self, beta=1):
        self.beta = beta

    def __call__(self, Z_full, V_full, neighb):
        real_z = Z_full + self.beta * V_full
        return np.argsort(real_z, axis=1)


class ViolationBasedRanking(ConstraintMethod):
    def __init__(self, criterion=None):
        self.criterion = criterion
        if self.criterion is None:
            self.criterion = self.violation_thresh

    def violation_thresh(self, V_full):
        Vsum = np.sum(V_full, axis=1)
        thresh = ((V_full.shape[1] - np.count_nonzero(V_full, axis=1)) * Vsum) / (V_full.shape[1] ** 2)
        thresh = thresh.reshape((1, thresh.shape[0])).T
        thresh = np.tile(thresh, (1, V_full.shape[1]))
        return V_full <= thresh
    
    def __call__(self, Z_full, V_full, neighb):
        criteria = self.criterion(V_full)
        real_z = np.zeros(Z_full.shape)
        feasible = (V_full == 0) | (criteria)
        
        real_z[feasible] = Z_full[feasible]
        real_z[feasible == False] = np.inf
        Z_sort_ind = np.argsort(real_z, axis=1)

        return Z_sort_ind
