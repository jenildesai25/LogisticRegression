import numpy as np


class LogisticRegression:

    @classmethod
    def generate_test_data(cls):
        mu_1 = [1, 0]
        mu_2 = [0, 1.5]

        sigma_1 = [[1, 0.75], [0.75, 1]]
        sigma_2 = [[1, 0.75], [0.75, 1]]

        set_0 = np.append(np.random.multivariate_normal(mu_1, sigma_1, 1000), np.zeros((1000, 1)), axis=1)
        set_1 = np.append(np.random.multivariate_normal(mu_2, sigma_2, 1000), np.ones((1000, 1)), axis=1)
        return set_0, set_1

