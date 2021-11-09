import matplotlib.pyplot as plt
import numpy as np
from evaluator import evaluate
import threading
import sys
import time
import math

from utils import load_ast


def get_stream(obj):
    while True:
        yield evaluate(obj)[0]


def get_samples(exp, num_samples):
    samples = []
    stream = get_stream(exp)
    for i in range(int(num_samples)):
        samples.append(next(stream))

    return np.array([elem.numpy() for elem in samples])


def data_generator_wrapper():
    for i in range(1, 4):
        # exp = daphne(['desugar-hoppl', '-i', '../../HW5/programs/{}.daphne'.format(i)])
        exp = load_ast('programs/saved_tests/{}.daphne'.format(i))
        print('\n\n\nSample of prior of program {}:'.format(i))

        start = time.time()
        samples = get_samples(exp, 10000)
        end = time.time()
        print('Took {0:.2f} seconds to finish Program {1}'.format(end - start, i))

        np.save('data/program{}.npy'.format(i), samples)

if __name__ == '__main__':

        # sys.setrecursionlimit(100000)
        # threading.stack_size(200000000)
        # thread = threading.Thread(target=data_generator_wrapper)
        # thread.start()

        # program1 = np.load('data/program1.npy')
        # plt.hist(program1,
        #          bins=math.ceil((np.max(program1.flatten()) - np.min(program1.flatten()))/50))
        # plt.title('Program 2\nPrior Exp: {0:.2f}    Prior Var: {1:.2f}'.format(np.mean(program1), np.var(program1)))
        # plt.xlabel('value')
        # plt.ylabel('frequency')
        # plt.savefig('figures/program1.png')
        # plt.clf()
        #
        # program2 = np.load('data/program2.npy')
        # plt.hist(program2,
        #          bins=math.ceil((np.max(program2.flatten()) - np.min(program2.flatten()))/2))
        # plt.title('Program 3\nPrior Exp: {0:.2f}    Prior Var: {1:.2f}'.format(np.mean(program2), np.var(program2)))
        # plt.xlabel('value')
        # plt.ylabel('frequency')
        # plt.savefig('figures/program2.png')
        # plt.clf()

        program3 = np.load('data/program3.npy')

        for i in range(program3.shape[1]):
            plt.hist(program3[:, i])
            plt.title('Program 4 - Observation {}'.format(i))
            plt.xlabel('state')
            plt.ylabel('frequency')
            plt.savefig('figures/program3_obs{}'.format(i))
            plt.clf()
