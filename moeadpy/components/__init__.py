__all__ = ["constraints", "decomposition", "neighborhood",
           "scalarization", "stopping", "update", "variation"]

class Component:
    def __call__(self):
        raise NotImplementedError("This 'Component' class hasn't implemented the method '__call__' yet.")
        pass


# Every variator, localsearch, stopcriteria class 
# in this file must inherit this (composite pattern)
class SingleItemIterator(Component):
    def __iter__(self):
        self.iterated = False
        return self

    def __next__(self):
        if not self.iterated:
            self.iterated = True
            return self
        else:
            raise StopIteration


class MoeadSet:
    def __init__(self, weights=None, neighb=None, x=None, xt=None,
                     y=None, yt=None, v=None, vt=None, sort_inds=None,
                     iteration=None):
        self.weights = weights
        self.neighb = neighb
        self.x = x
        self.xt = xt
        self.y = y
        self.yt = yt
        self.v = v
        self.vt = vt
        self.sort_inds = sort_inds
        self.iteration = iteration
    
    def update(self, **kwargs):
        for attr, value in kwargs.items():
            if not hasattr(self, attr):
                raise ValueError(f"'MoeadSet' does not contain the attribute '{attr}''.")
            setattr(self, attr, value)