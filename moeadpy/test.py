import unittest
from . import core
import numpy as np


def zd1(population):
    D = population.shape[1]
    return np.array(
        [[
            x[0],
            (1 + 9 * np.sum(x[1:D] / (D-1))) * (1 - np.sqrt(x[0] / (1 + 9 * np.sum(x[1:D] / (D-1)))))
        ] for x in population]
        )


class TestMainClass(unittest.TestCase):

    def test_run(self):
        m = core.Moead(zd1, 2, np.zeros(10), np.ones(10), seed=4)
        print(
            m.compute()
        )
        pass


if __name__ == "__main__":
    unittest.main()
