import numpy as np
from scipy.stats import multivariate_normal


class two_multivariate_normals:
    mvn1 = None
    mvn2 = None

    def __init__(self, mu, var):
        self.mvn1 = multivariate_normal([mu, mu], [[var, 0], [0, var]])
        self.mvn2 = multivariate_normal([-mu, -mu], [[var, 0], [0, var]])

    def pdf(self, pos):
        return 0.5 * (self.mvn1.pdf(pos) + self.mvn2.pdf(pos))

    def rvs(self, num_samples):
        weights = np.random.rand(num_samples)
        return np.tile((weights <= 0.5), (2, 1)).T * self.mvn1.rvs(num_samples) + np.tile(
            (weights > 0.5), (2, 1)
        ).T * self.mvn2.rvs(num_samples)
