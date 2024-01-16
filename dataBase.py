import numpy as np
from scipy.stats import multivariate_normal

class Feature:
    def __init__(self, mean_para, variance_para, dim):
        self.mean = mean_para
        self.variance_para = variance_para
        self.dim = dim

    def get_variance(self):
        variance = np.zeros((self.dim, self.dim))
        for j in range(self.dim):
            for k in range(self.dim):
                variance[j][k] = np.power(self.variance_para, np.abs(j-k))
        return variance

    def get_mean(self):
        mean = [self.mean for i in range(self.dim)]
        return mean

    def get_sample(self, n, random_seed=None):
        return multivariate_normal.rvs(self.get_mean(), self.get_variance(), size=n, random_state=random_seed)


if __name__ == "__main__":
    mean = 0
    var = 0.5

    feature = Feature(mean, var, 10)
    # print(feature.get_variance_variance())
    print(feature.get_sample(2, 2024))
