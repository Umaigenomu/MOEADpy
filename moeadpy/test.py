import unittest
import core
import numpy as np
from components import neighborhood, stopping, decomposition
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')



def zd1(population):
    D = population.shape[1]
    return np.array(
        [[
            x[0],
            (1 + 9 * np.sum(x[1:D] / (D-1))) * (1 - np.sqrt(x[0] / (1 + 9 * np.sum(x[1:D] / (D-1)))))
        ] for x in population]
        )

def plot_front(evaluations):
    length = evaluations.shape[0]
    plt.figure()
    plt.scatter(evaluations[:, 0].reshape(length), evaluations[:, 1].reshape(length))
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    plt.savefig("test.png")
    plt.close()

class TestMainClass(unittest.TestCase):

    def test_run(self):
        m = core.Moead(zd1, 2, np.zeros(10), np.ones(10),
                       decomp=decomposition.Sld(h=99),
                       neighborhood=neighborhood.TNearestNeighbors(T=20, delta_prob=0.85),
                       seed=10,
                       stop_criteria=stopping.MaxIter(200))
        m.compute()
        print(
            m.results[0]
        )
        print(
            m.results[1]
        )
        plot_front(m.results[1])


if __name__ == "__main__":
    unittest.main()
