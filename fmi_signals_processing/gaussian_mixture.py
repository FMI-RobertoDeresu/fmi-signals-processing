import numpy as np
from scipy.stats import norm


class GaussianMixture:
    def __init__(self, c, mu, sigma):
        self.n = len(c)
        self.c = c
        self.mu = mu
        self.sigma = sigma
        self.generator = self._get_generator()

    def _get_generator(self):
        while True:
            gaussian_index = np.random.choice(range(self.n), p=self.c)
            mu, sigma = self.mu[gaussian_index], self.sigma[gaussian_index]
            observation = norm.rvs(mu, sigma)
            yield observation

    def generate(self):
        return next(self.generator)

    def component_pdf(self, k, observation):
        c, mu, sigma = self.c[k], self.mu[k], self.sigma[k]
        # pdf = c * norm.pdf(observation, loc=mu, scale=sigma)
        pdf = c * norm.pdf((observation - mu) / sigma)
        return pdf

    def pdf(self, observation):
        pdf = 0
        for k in range(self.n):
            pdf += self.component_pdf(k, observation)

        return pdf
