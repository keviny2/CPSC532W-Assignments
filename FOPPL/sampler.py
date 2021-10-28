from abc import ABC, abstractmethod
import re
import torch
from utils import load_ast, substitute_sampled_vertices
from graph_based_sampling import deterministic_eval


class Sampler(ABC):
    def __init__(self, method):
        self.method = method
        self.posterior_exp = {}
        self.posterior_var = {}

    @abstractmethod
    def sample(self, num_samples, num):
        """
        abstract sampling procedure that is algorithm specific

        :param num_samples: number of samples
        :param num: number corresponding to which daphne program was loaded; will affect how we compute summary
        statistics and construct plots because programs have different numbers of return values
        :param summary: bool
        :param plot: bool
        """
        raise NotImplementedError('subclasses must override this method!')

    @abstractmethod
    def compute_statistics(self, samples, parameter_names):
        """
        method to compute the posterior expectation and variance for a sequence of samples and sample weights

        :param samples: samples obtained from the sampling procedure
        :param parameter_names: parameter names for printing / plotting
        :return:
        """
        raise NotImplementedError('subclasses must override this method!')

    @abstractmethod
    def plot_values(self, samples, parameter_names):
        """
        method to construct likelihood plots and histograms of the posterior

        :param samples: samples obtained from the sampling procedure
        :param parameter_names: parameter names for printing / plotting
        :return:
        """
        raise NotImplementedError('subclasses must override this method!')

    def summary(self, num, samples):
        """
        computes posterior expectation and variance

        :param num: number corresponding to which daphne program was loaded; will affect how we compute summary
        :param samples: list of observations obtained from the sampling procedure
        """

        if num == 1:
            self.compute_statistics(samples, ['mu'])

        if num == 2:
            self.compute_statistics(samples, ['slope', 'bias'])

        if num == 5:
            self.compute_statistics(samples, ['z[1] == z[2]'])

        if num == 6:
            self.compute_statistics(samples, ['is-raining'])

        if num == 7:
            self.compute_statistics(samples, ['x', 'y'])

    def plot(self, num, samples, num_points, save_plot):
        """
        constructs plots

        :param num: number corresponding to which daphne program was loaded; will affect how we compute summary
        :param samples: list of observations obtained from the sampling procedure
        :param num_points: number of points to plot
        :param save_plot: True if we save the plot
        """
        if num == 1:
            self.plot_values(samples, ['mu'], num_points, save_plot, num)

        if num == 2:
            self.plot_values(samples, ['slope', 'bias'], num_points, save_plot, num)

        if num == 5:
            self.plot_values(samples,
                             ['z[1] == z[2]'],
                             num_points,
                             save_plot,
                             num)

        if num == 6:
            self.plot_values(samples, ['is-raining'], num_points, save_plot, num)

        if num == 7:
            self.plot_values(samples, ['x', 'y'], num_points, save_plot, num)

    @staticmethod
    def compute_log_density(samples, num, ignore=None):
        """
        computes log density for a set of observations
        :param samples: set of observations
        :param num: identification for which program to run
        :param ignore: list of indexes that indicate which parameters we want to ignore from computation
        :return: trace of log densities
        """
        # load the corresponding program
        graph = load_ast('programs/saved_asts/hw3/program{}_graph.pkl'.format(num))

        regex_obs = re.compile(r'observe*')
        regex_lat = re.compile(r'sample*')

        # iterate over every node in the graph to obtain the joint
        log_ps = []  # accumulator

        # iterate over each observation
        for x in samples:
            # assign parameter values
            if x.size() == torch.Size([]) or x.size() == torch.Size([1]):
                graph[1]['Y'][graph[2]] = x
            else:
                # assuming that the x's will be in the same order as the naming
                # (e.g. sample1, sample2 NOT sample2, sample1)
                for i in range(len(x)):
                    graph[1]['Y']['sample{}'.format(i + 1)] = x[i]

            # iterate over nodes and compute log likelihood
            log_p = 0  # accumulator
            for node in graph[1]['V']:

                # if ignore is specified, check if we want to ignore the node
                if ignore is not None:
                    if node in ['sample{}'.format(i) for i in ignore]:
                        continue

                # substitute variables with their values
                raw_expression = graph[1]['P'][node]
                expression = substitute_sampled_vertices(raw_expression, graph[1]['Y'])

                # different behavior when it is a sample or observed node
                if regex_obs.match(node):
                    log_p += deterministic_eval(expression)
                elif regex_lat.match(node):
                    expression[0] = 'observe*'
                    expression.append(graph[1]['Y'][node])
                    log_p += deterministic_eval(expression)
                else:
                    raise KeyError('Cannot determine expression type')

            log_ps.append(log_p)

        return torch.FloatTensor(log_ps)


