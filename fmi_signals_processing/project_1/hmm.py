import numpy as np
from project_1.gaussian_mixture import GaussianMixture


class HMM:
    def __init__(self, pi, a, b):
        self.n = len(pi)
        self.m = np.shape(b)[1]
        self.pi = np.array(pi)
        self.a = np.array(a)
        self.b = [GaussianMixture(*[list(param) for param in zip(*params)]) for params in b]

    def generate(self, t):
        obs = np.zeros(t)
        state = np.random.choice(range(self.n), p=self.pi)

        for t in range(t):
            obs[t] = self.b[state].generate()
            state = np.random.choice(range(self.n), p=self.a[state])

        return obs

    # alpha(t, i) is the probability that the HMM is in state i having generated partial observation t
    def _get_alpha(self, obs):
        obs_size = len(obs)
        alpha = np.zeros((obs_size, self.n))

        for i in range(self.n):
            alpha[0, i] = self.pi[i] * self.b[i].pdf(obs[0])

        for t in range(1, obs_size):
            for i in range(self.n):
                alpha[t, i] = np.sum(alpha[t-1] * self.a[:, i]) * self.b[i].pdf(obs[t])

        return alpha

    # beta(t, i) is the probability of generating partial observation from t+1 to the end,
    # given that the HMM is in state i at time t
    def _get_beta(self, obs):
        observations_size = len(obs)
        beta = np.zeros((observations_size, self.n))
        beta[-1] = np.full(self.n, 1)

        for t in range(observations_size-2, -1, -1):
            for i in range(self.n):
                for j in range(self.n):
                    beta[t, i] += self.a[i, j] * self.b[j].pdf(obs[t+1]) * beta[t+1][j]

        return beta

    # gamma(t, i, j) is the probability of taking the transition from state i to state j at time t
    def _get_gamma(self, obs):
        obs_size = len(obs)
        gamma = np.zeros((obs_size, self.n, self.n))

        alpha = self._get_alpha(obs)
        beta = self._get_beta(obs)

        for t in range(1, obs_size):
            for i in range(self.n):
                for j in range(self.n):
                    gamma[t, i, j] = (alpha[t-1, i] * self.a[i, j] * self.b[j].pdf(obs[t]) * beta[t, j] /
                                      np.sum(alpha[t]))

        return gamma

    # xi(t, i, k) is the probability of component (gaussian) k at time t in state i
    def _get_xi(self, obs):
        obs_size = len(obs)
        xi = np.zeros((obs_size, self.n, self.m))

        alpha = self._get_alpha(obs)
        beta = self._get_beta(obs)

        for t in range(1, obs_size):
            for i in range(self.n):
                for k in range(self.m):
                    xi[t, i, k] = (np.sum(alpha[t-1] * self.a[:, i] * self.b[i].component_pdf(k, obs[t]) * beta[t, i]) /
                                   np.sum(alpha[t]))

        return xi

    def forward(self, obs):
        alpha = self._get_alpha(obs)
        probability = np.sum(alpha[-1])
        return probability

    def viterbi(self, obs):
        obs_size = len(obs)

        # (v[t, i], b[t, i]) is the (probability, path) of the most likely state sequence at time t,
        # which has generated the observation until time t and ends in state i
        v = np.zeros((obs_size, self.n))
        b = np.zeros((obs_size, self.n))

        for i in range(self.n):
            v[0, i] = self.pi[i] * self.b[i].pdf(obs[0])

        for t in range(1, obs_size):
            for i in range(self.n):
                v[t, i] = np.max(v[t-1] * self.a[:, i]) * self.b[i].pdf(obs[t])
                b[t, i] = np.argmax(v[t-1] * self.a[:, i])

        probability = np.max(v[-1])

        path = np.zeros(obs_size, dtype=int)
        path[obs_size-1] = np.argmax(v[obs_size-1])
        for t in range(obs_size-1, 0, -1):
            path[t - 1] = b[t][path[t]]

        return probability, path

    def baum_welch(self, obs, steps):
        obs_size = len(obs)
        train_results = np.zeros(steps)

        for step in range(steps):
            gamma = self._get_gamma(obs)
            xi = self._get_xi(obs)

            a = np.zeros((self.n, self.n))
            for i in range(self.n):
                for j in range(self.n):
                    a = np.sum(gamma[:, i, j]) / np.sum(gamma[:, i, :])

            for j in range(self.n):
                for k in range(self.m):
                    c = np.sum(xi[:, j, k]) / np.sum(xi[:, j, :])
                    mu = np.sum(xi[:, j, k] * obs) / np.sum(xi[:, j, k])
                    sigma = np.sum(xi[:, j, k] * np.power(obs - self.b[j].mu[k], 2)) / np.sum(xi[:, j, k])

                    self.b[j].c[k] = c
                    self.b[j].mu[k] = mu
                    self.b[i].sigma[k] = sigma

            self.a[i, j] = a
            # train_results[step] = self.forward(obs)

            # IN PROGRESS!!

        return train_results
