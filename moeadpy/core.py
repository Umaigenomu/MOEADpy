import numpy as np
import components


class _AliasDict(dict):
    def __setitem__(self, key, value):
        for ikey in key.split(";"):
            super().__setitem__(ikey, value)

    def __missing__(self, key):
        return key


aliases = _AliasDict({
    "simplex lattice design;Simplex-Lattice Design": "sld",

})


def invoke(component, method: str, *args, **kwargs):
    try:
        f = getattr(component, aliases[method])
    except AttributeError:
        raise AttributeError(
            f"The method '{method}' is not included in '{component}'. "
            "Please re-check your spelling.")
    return f(*args, **kwargs)


class Moead:
    def __init__(self,
                 decomp=("sld", 99),  # decomposition strategy
                 scalarization="wt",  # scalar aggregation function
                 neighborhood=("lambda", 20, 1),  # neighborhood assignment
                 variation=None,  # variation operators
                 update=None,  # update method
                 constraint=None,  # constraint handling method
                 scaling=None,  # objective scaling strategy
                 terminantion=None,  # stop criteria
                 showpars=None,  # echoing behavior
                 seed=None,       # Seed for PRNG
                 ):
        self.problem = None
        self.decomp = decomp
        self.scalarization = scalarization
        self.neighborhood = neighborhood
        self.variation = variation
        self.update = update
        self.constraint = constraint
        self.scaling = scaling
        self.terminantion = terminantion
        self.showpars = showpars
        self.seed = seed
        pass

    def set_problem(problem):

        pass

    def compute():
        # 1. Generate initial population and weights
        
        # 2. Define or update neighborhoods
        # 3. Copy incumbent the solution in preparation for the new one
        # 4. Variation operators
        # 5. Aggregation functions
        # 6. Constraints
        # 7. Update
        # 8. Termination check
        pass
