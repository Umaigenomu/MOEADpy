from . import SingleItemIterator
from . import MoeadSet


class MaxIter(SingleItemIterator):
    def __init__(self, max_iter=200):
        self.max_iter = max_iter

    def __call__(self, mset: MoeadSet):
        return mset.iteration >= self.max_iter
        