import numpy as np
import hmm_algorithms
import os


def get_obs_generation_params(inputFile):
    with open(inputFile, "r") as f:
        (T, N) = [int(x) for x in f.readline().split()]
        A = np.array([[float(x) for x in f.readline().split()] for _ in range(N)])
        B = np.array([[float(x) for x in f.readline().split()] for _ in range(N)])
        pi = np.array([float(x) for x in f.readline().split()])

        return T, N, len(B[0]), A, B, pi


def get_forward_params(inputFile):
    with open(inputFile, "r") as f:
        N = int(f.readline())
        X = np.array([int(x) for x in f.readline().split()])
        A = np.array([[float(x) for x in f.readline().split()] for _ in range(N)])
        B = np.array([[float(x) for x in f.readline().split()] for _ in range(N)])
        pi = np.array([float(x) for x in f.readline().split()])

        return len(X), N, X, A, B, pi


def get_viterbi_params(inputFile):
    with open(inputFile, "r") as f:
        N = int(f.readline())
        X = np.array([int(x) for x in f.readline().split()])
        A = np.array([[float(x) for x in f.readline().split()] for _ in range(N)])
        B = np.array([[float(x) for x in f.readline().split()] for _ in range(N)])
        pi = np.array([float(x) for x in f.readline().split()])

        return len(X), N, X, A, B, pi


def get_baum_welch_params(inputFile):
    with open(inputFile, "r") as f:
        T = int(f.readline())
        N = int(f.readline())
        A = np.array([[float(x) for x in f.readline().split()] for _ in range(N)])
        B = np.array([[float(x) for x in f.readline().split()] for _ in range(N)])
        pi = np.array([float(x) for x in f.readline().split()])

        return T, N, len(B[0]), A, B, pi


if __name__ == '__main__':
    obs_generation_params = get_obs_generation_params('input\\obs_generation.txt')
    observations = hmm_algorithms.obs_generation_algorithm(*obs_generation_params)
    print(f'Algoritmul de generare de observatii:\n'
          f'Secventa: {" ".join([str(x) for x in observations])}\n')

    forward_params = get_forward_params('input\\forward.txt')
    forward_probability = hmm_algorithms.forward_algorithm(*forward_params)
    print(f'Algoritmul forward:\n'
          f'Probabilitate: {forward_probability:.3f}\n')

    viterbi_params = get_viterbi_params('input\\viterbi.txt')
    (viterbi_probability, viterbi_sequence) = hmm_algorithms.viterbi_algorithm(*viterbi_params)
    print(f'Algoritmul Viterbi:\n'
          f'Probabilitate: {viterbi_probability:.3f}\n'
          f'Secventa: {" ".join([str(x) for x in viterbi_sequence])}\n')

    baum_welch_params = get_baum_welch_params('input\\baum_welch.txt')
    (baum_welch_A, baum_welch_B, baum_welch_pi) = hmm_algorithms.baum_welch_algorithm(*baum_welch_params)
    print(f'Algoritmul Baum-Welch:\n'
          f'A:\n{os.linesep.join([" ".join(["{0:.2f}".format(y) for y in x]) for x in baum_welch_A])}\n\n'
          f'B:\n{os.linesep.join([" ".join(["{0:.16f}".format(y) for y in x]) for x in baum_welch_B])}\n\n'
          f'Pi:\n{" ".join(["{0:.2f}".format(x) for x in baum_welch_pi])}\n')
