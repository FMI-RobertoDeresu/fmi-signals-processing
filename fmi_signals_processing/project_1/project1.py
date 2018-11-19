import os
import numpy as np
from project_1.hmm import HMM


def print_observations(hmm: HMM, t):
    observations = hmm.generate(t)
    print(f'Algoritmul de generare de observatii:\n'
          f'Secventa: {", ".join(["{0:.10f}".format(observation) for observation in observations])}\n')

    return observations


def print_forward(hmm: HMM, test):
    forward_probability = hmm.forward(test)
    print(f'Algoritmul forward:\n'
          f'Probabilitate: {str(forward_probability)}\n')


def print_viterbi(hmm: HMM, test):
    (viterbi_probability, viterbi_sequence) = hmm.viterbi(test)
    print(f'Algoritmul Viterbi:\n'
          f'Probabilitate: {str(viterbi_probability)}\n'
          f'Secventa: {" ".join([str(x) for x in viterbi_sequence])}\n')


def print_baum_welch(hmm: HMM, test, steps):
    train_results = hmm.baum_welch(np.array(test), steps)
    print(f'Algoritmul Baum-Welch:\n'
          f'Train results:\n{os.linesep.join(str(x) for x in train_results)}\n'
          f'A:\n{os.linesep.join([", ".join(["{0:.10f}".format(y) for y in x]) for x in hmm.a])}\n'
          f'B:\n{os.linesep.join([", ".join(["({0:.10f}, {1:.10f}, {2:.10f})".format(*y) for y in zip(x.c, x.mu, x.sigma)]) for x in hmm.b])}\n\n')


def run(input_data):
    hmm_1 = HMM(**input_data['hmm_1'])
    hmm_2 = HMM(**input_data['hmm_2'])
    hmm_3 = HMM(**input_data['hmm_3'])

    test = hmm_1.generate(input_data['t'])

    print_forward(hmm_1, test)
    print_forward(hmm_2, test)
    print_forward(hmm_3, test)

    print_viterbi(hmm_1, test)
    print_viterbi(hmm_2, test)
    print_viterbi(hmm_3, test)
