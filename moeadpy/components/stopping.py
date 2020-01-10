from . import Component

class MaxIter:
    def __init__(self, max_iter=200):
        self.max_iter = max_iter

    def __call__(self, iter, *args):
        return iter < self.max_iter 
        