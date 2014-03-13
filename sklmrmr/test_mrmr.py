import unittest
import numpy as np
from sklmrmr import MRMR


class TestMRMR(unittest.TestCase):
    def test_mrmr(self):
        X = np.zeros((10, 10))
        X[5:, 0] = 1
        y = np.zeros(10)
        y[5:] = 1
        model = MRMR(k=1)
        model.fit(X, y)
        assert model.selected_[0] == 0
        assert model.n_features_ == 1

    def test_continuous_features(self):
        X = np.zeros((10, 10))
        X[0, 0] = 1.5
        y = np.zeros(10)
        model = MRMR()
        self.assertRaises(ValueError, model.fit, X, y)

    def test_continuous_labels(self):
        X = np.zeros((10, 10))
        y = np.zeros(10)
        y[0] = 1.5
        model = MRMR()
        self.assertRaises(ValueError, model.fit, X, y)


if __name__ == '__main__':
    unittest.main()
