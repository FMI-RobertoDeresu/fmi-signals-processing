import numpy as np
import utils


def obs_generation_algorithm(T, N, M, A, B, pi):
    i = np.random.choice(range(N), p=pi)
    x = np.zeros(T, dtype=int)
    for t in range(T):
        x[t] = np.random.choice(range(M), p=B[i])
        i = np.random.choice(range(N), p=A[i])

    return x


def forward_algorithm(T, N, X, A, B, pi):
    alpha = np.zeros((T, N))
    alpha[0] = pi * np.transpose(B)[X[0]]
    for t in range(1, T):
        for j in range(N):
            alpha[t][j] = sum(alpha[t-1] * np.transpose(A)[j]) * B[j][X[t]]

    p = sum(alpha[T-1])

    return p


def viterbi_algorithm(T, N, X, A, B, pi):
    V = np.zeros((T, N))
    VP = np.zeros((T, N))

    V[0] = pi * np.transpose(B)[X[0]]

    for t in range(1, T):
        for j in range(N):
            V[t][j] = max(V[t-1] * np.transpose(A)[j] * B[j][X[t]])
            VP[t][j] = np.argmax(V[t-1] * np.transpose(A)[j] * B[j][X[t]])

    p = max(V[T-1])
    s = np.zeros(T, dtype=int)
    s[T-1] = np.argmax(V[T-1])
    for t in range(T-1, 0, -1):
        s[t-1] = VP[t][s[t]]

    return p, s


def baum_welch_algorithm(T, N, M, A, B, pi):
    # Generate observation
    X = obs_generation_algorithm(T, N, M, A, B, pi)

    # Model perturbation
    A = utils.normalize_array(A + [np.random.uniform(0, .2, N) for _ in range(N)])
    B = utils.normalize_array(B + [np.random.uniform(0, .2, M) for _ in range(N)])
    pi = utils.normalize_array(pi + np.random.uniform(0, .2, N))

    repeat = 1
    while repeat:
        # forward
        # alpha(t, i) - is the probability that the HMM is in state i having generated partial observation Xt
        alpha = np.zeros((T, N))
        alpha[0] = pi * np.transpose(B)[X[0]]
        for t in range(1, T):
            alpha[t] = (alpha[t-1] * np.transpose(A)).sum(1) * B.transpose()[X[t]]

        # backward
        # beta(t, i) - is the probability of generating partial observation Xt+1 (from t+1 to the end)
        # given that the HMM is in state i at time t
        beta = np.zeros((T, N))
        beta[T-1] = np.full(N, 1/N)
        for t in range(T-2, -1, -1):
            beta[t] = np.sum(A * np.transpose(B)[X[t+1]] * beta[t+1], 1)

        # Update
        # gamma(t, i) - is the probability of taking the transition from state i to state j at time t
        gamma = np.zeros((T, N, N))
        for t in range(T):
            gamma[t] = (alpha[t-1].repeat(N).reshape((N,N)) * A * np.transpose(B)[X[t]] * beta[t])/(sum(alpha[t]))

        A = gamma.sum(0)/gamma.sum((0, 2)).repeat(N).reshape(N, N)
        B = np.array([gamma[x] for x in X]).sum((0, 1)).repeat(M).reshape(N, M)/gamma.sum((0, 1)).repeat(M).reshape(N, M)

        B1 = np.zeros((N, M))
        for j in range(N):
            for k in range(M):
                sum1 = 0
                for t in X:
                    for i in range(N):
                        sum1 += gamma[t, i, j]

                sum2 = 0
                for t in range(T):
                    for i in range(N):
                        sum2 += gamma[t, i, j]
                B1[j, k] = sum1 / sum2

        sameB = np.all(np.round(B, 5) == np.round(B1, 5))

        repeat -= 1

    return A, B, pi
