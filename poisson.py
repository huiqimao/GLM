import numpy as np
import pandas as pd
from scipy.stats import norm, poisson
from dataBase import Feature


class Poisson_simulate():
    def __init__(self, theta, tau, eta, gamma, n_sample):
        self.theta = theta
        self.tau = tau
        self.eta = eta
        self.gamma = gamma
        self.n_sample = n_sample

    # feature: n_sample * n_dim /5*10
    def get_price_para(self, feature, random_seed):
        epsilon = norm.rvs(0, 9, (self.n_sample, 1), random_state=random_seed)
        price_feature = 50 + np.dot(feature, self.gamma) + epsilon
        return price_feature

    def get_poisson_mean(self, feature, random_seed):
        para1 = self.theta[0] + np.dot(self.theta[1:5], feature[:, 1:5].T)
        para1 = para1.reshape(-1, 1)
        para2 = np.concatenate([feature,
                                np.power(feature[:,1],2).reshape(-1,1),
                                (feature[:, 2] * feature[:, 3]).reshape(-1,1),
                                (feature[:, 3] * feature[:, 4]).reshape(-1,1),
                                (feature[:, 4] * feature[:, 5]).reshape(-1,1),
                                ], axis=1) # n_sample * n_dim
        prices = self.get_price_para(feature, random_seed)
        para3 = prices*para1 + self.tau + np.dot(para2, self.eta) # n_sample * 1
        poisson_mean = np.exp(para3)
        return prices, poisson_mean

    def get_sample(self, poisson_mean_list, n_sample, random_seed):
        samples = np.array([poisson.rvs(mu=m, size=n_sample, random_state=random_seed)
                   for m in poisson_mean_list]).T
        return samples


def generate_feature(n_sample, poisson_times, file=None):
    mean = 0
    var = 0.5
    random_seed = 2024
    f = Feature(mean, var, 10)
    feature = f.get_sample(n_sample, random_seed)  # dim:5*10

    gamma = np.array([[3]] * 10)
    tau = 1.2
    eta = np.array([[0.1]] * 14)
    theta = np.array([-0.02, -0.005, -0.005, -0.005, -0.005]).T
    poisson = Poisson_simulate(theta, tau, eta, gamma, n_sample)
    prices, poisson_mean = poisson.get_poisson_mean(feature, random_seed)
    samples = poisson.get_sample(poisson_mean, poisson_times, random_seed).T
    data = np.concatenate([np.concatenate([feature,
                                           prices,
                                           poisson_mean,
                                           samples[:, i].reshape(-1, 1),],
                                          axis=1)
                           for i in range(poisson_times)])

    column_name = ['X_' + str(i) for i in range(10)] + ['price', 'poisson_mean', 'poisson_sample']
    df = pd.DataFrame(data, columns=column_name)
    if file:
        df.to_csv(file, index=False)
    return df


if __name__ == "__main__":
    file = "/Users/huiqiangmao/PycharmProjects/GLM/data/poisson_samples_test.csv"
    data = generate_feature(1000, 10, file)


