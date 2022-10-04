import numpy as np
from scipy.stats import multivariate_normal


class two_multivariate_normals:
    mvn1 = None
    mvn2 = None

    def __init__(self, mu, var, cov):
        self.mvn1 = multivariate_normal([mu, mu], [[var, cov], [cov, var]])
        self.mvn2 = multivariate_normal([-mu, -mu], [[var, cov], [cov, var]])

    def pdf(self, pos):
        return 0.5 * (self.mvn1.pdf(pos) + self.mvn2.pdf(pos))

    def rvs(self, num_samples):
        weights = np.random.rand(num_samples)
<<<<<<< HEAD
        return np.tile((weights <= 0.5), (2, 1)).T * self.mvn1.rvs(num_samples) + np.tile(
            (weights > 0.5), (2, 1)
        ).T * self.mvn2.rvs(num_samples)

=======
        return np.tile((weights <= 0.5), (2, 1)).T * self.mvn1.rvs(
            num_samples
        ) + np.tile((weights > 0.5), (2, 1)).T * self.mvn2.rvs(num_samples)
>>>>>>> b1986d53dff3b26a374c4cd43c9e217075962518
