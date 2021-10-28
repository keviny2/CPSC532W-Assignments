from evaluation_based_sampling import load_ast, evaluate_program
from graph_based_sampling import sample_from_joint
import matplotlib.pyplot as plt
import numpy as np
from utils import tasks


def create_plots(sampling_type, task_num, num_samples=1000, save_plot=False):
    samples = get_samples(sampling_type, task_num, num_samples)
    if task_num == 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('marginal expectation: {:.2f}'.format(np.mean(samples), 2))
        ax.hist(samples)
        ax.set(ylabel='frequency', xlabel='mu')
        path = 'report/figures/{0}_{1}'.format(sampling_type, task_num)
        plt.suptitle("{0} - {1} based sampling".format(tasks[task_num - 1], sampling_type))

    if task_num == 2:
        fig, axs = plt.subplots(2, figsize=(8, 6))

        axs[0].set_title('marginal expectation for slope: {:.2f}'.format(np.mean(samples[:, 0]), 2))
        axs[0].hist(samples[:, 0])
        axs[0].set(ylabel='frequency', xlabel='slope')

        axs[1].set_title('marginal expectation for bias: {:.2f}'.format(np.mean(samples[:, 1]), 2))
        axs[1].hist(samples[:, 1])
        axs[1].set(ylabel='frequency', xlabel='bias')


        path = 'report/figures/{0}_{1}'.format(sampling_type, task_num)
        plt.suptitle("{0} - {1} based sampling".format(tasks[task_num - 1], sampling_type))
        plt.tight_layout()

    if task_num == 3:
        fig, axs = plt.subplots(figsize=(8, 6))

        counts = np.apply_along_axis(np.bincount, 1, samples)
        sum_of_rows = counts.sum(axis=1)
        normalized_counts = counts / sum_of_rows[:, np.newaxis]

        axs.set_title('marginal expectations \n'
                      'state 1: {0:.2f}\n'
                      'state 2: {1:.2f}\n'
                      'state 3: {2:.2f}'.format(np.mean(normalized_counts[:, 0]),
                                                np.mean(normalized_counts[:, 1]),
                                                np.mean(normalized_counts[:, 2])))
        axs.hist(samples)
        axs.set(ylabel='frequency', xlabel='states')


        path = 'report/figures/{0}_{1}'.format(sampling_type, task_num)
        plt.suptitle("{0} - {1} based sampling".format(tasks[task_num - 1], sampling_type))
        plt.tight_layout()

    if task_num == 4:
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))

        axs[0, 0].set_title('marginal expectation for W_0: {:.2f}'.format(np.mean(samples[:, 0]), 2))
        axs[0, 0].hist(samples[:, 0])
        axs[0, 0].set(ylabel='frequency', xlabel='W_0')

        axs[1, 0].set_title('marginal expectation for b_0: {:.2f}'.format(np.mean(samples[:, 1]), 2))
        axs[1, 0].hist(samples[:, 1])
        axs[1, 0].set(ylabel='frequency', xlabel='b_0')

        axs[0, 1].set_title('marginal expectation for W_1: {:.2f}'.format(np.mean(samples[:, 2]), 2))
        axs[0, 1].hist(samples[:, 2])
        axs[0, 1].set(ylabel='frequency', xlabel='W_1')

        axs[1, 1].set_title('marginal expectation for b_1: {:.2f}'.format(np.mean(samples[:, 3]), 2))
        axs[1, 1].hist(samples[:, 3])
        axs[1, 1].set(ylabel='frequency', xlabel='b_1')

        path = 'report/figures/{0}_{1}'.format(sampling_type, task_num)
        plt.suptitle("{0} - {1} based sampling".format(tasks[task_num - 1], sampling_type))
        plt.tight_layout()

    if save_plot:
        plt.savefig(path)
        print("\nFigure saved!", path)

    plt.show()


def get_samples(sampling_type, task_num, num_samples):
    samples = []
    if sampling_type == 'evaluation':
        ast = load_ast('programs/saved_asts/hw2/daphne{}_ast.pkl'.format(task_num))

        stream = get_stream(ast, sampling_type)
        for i in range(int(num_samples)):
            samples.append(next(stream))

    if sampling_type == 'graph':
        graph = load_ast('programs/saved_asts/hw2/daphne_graph{}.pkl'.format(task_num))

        stream = get_stream(graph, sampling_type)
        for i in range(int(num_samples)):
            samples.append(next(stream))

    if task_num == 4:
        return np.array([[np.mean(sub_elem.numpy().flatten()) for sub_elem in elem] for elem in samples])

    return np.array([elem.numpy() for elem in samples])


def get_stream(obj, sampling_type):
    if sampling_type == 'evaluation':
        while True:
            yield evaluate_program(obj)[0]
    if sampling_type == 'graph':
        while True:
            yield sample_from_joint(obj)


if __name__ == '__main__':

    for task in range(4, 5):
        create_plots('evaluation', task)
        create_plots('graph', task)