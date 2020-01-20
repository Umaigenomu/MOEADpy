import unittest
import core
import numpy as np
from components import neighborhood, stopping, decomposition, scalarization
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
np.set_printoptions(threshold=np.inf)



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
    plt.xlim(0, np.max(evaluations[:, 0]))
    plt.ylim(0, np.max(evaluations[:, 1]))
    plt.savefig("test.png")
    plt.close()

class TestMainClass(unittest.TestCase):

    @unittest.skip("Unnecessary")
    def test_original(self):
        m = core.Moead(zd1, 2, np.zeros(30), np.ones(30),
                       seed=None,
                       stop_criteria=stopping.MaxIter(200))
        m.compute()
        print(m.results[0].round(2))
        print(m.results[1].round(2))
        plot_front(m.results[1])
    
    # @unittest.skip("")
    def test_variations(self):
        m = core.Moead(zd1, 2, np.zeros(30), np.ones(30),
                       decomp=decomposition.Sld(h=99),
                       scalarization=scalarization.WeightedTchebycheff(),
                       seed=24,
                       stop_criteria=stopping.MaxIter(600))
        m.compute()
        print(m.results[0].round(2))
        print(m.results[1].round(2))
        plot_front(m.results[1])


if __name__ == "__main__":
    unittest.main()
