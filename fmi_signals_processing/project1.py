import os
import numpy as np
from hmm import HMM
from input import input_data


def project1(input):
    hmm = HMM(**input['hmm'])

    observations = hmm.generate(input['t'])
    print(f'Algoritmul de generare de observatii:\n'
          f'Secventa: {", ".join(["{0:.10f}".format(observation) for observation in observations])}\n')

    forward_probability = hmm.forward(input['test'])
    print(f'Algoritmul forward:\n'
          f'Probabilitate: {str(forward_probability)}\n')

    (viterbi_probability, viterbi_sequence) = hmm.viterbi(input['test'])
    print(f'Algoritmul Viterbi:\n'
          f'Probabilitate: {str(viterbi_probability)}\n'
          f'Secventa: {" ".join([str(x) for x in viterbi_sequence])}\n')

    return

    hmm_baum_welch = HMM(**input['hmm_baum_welch_2'])
    train_results = hmm_baum_welch.baum_welch(np.array(input['test']), input['hmm_baum_welch_steps'])
    print(f'Algoritmul Baum-Welch:\n'
          f'Train results:\n{os.linesep.join(str(x) for x in train_results)}\n'
          f'A:\n{os.linesep.join([", ".join(["{0:.10f}".format(y) for y in x]) for x in hmm_baum_welch.a])}\n'
          f'B:\n{os.linesep.join([", ".join(["({0:.10f}, {1:.10f}, {2:.10f})".format(*y) for y in zip(x.c, x.mu, x.sigma)]) for x in hmm_baum_welch.b])}\n\n')


if __name__ == '__main__':
    project1(input_data)
