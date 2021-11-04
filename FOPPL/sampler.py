from abc import ABC, abstractmethod
import re
import torch
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math
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

    def compute_statistics(self, samples, parameter_names):
        """
        method to compute the posterior expectation and variance for a sequence of samples and sample weights

        :param samples: samples obtained from the sampling procedure
        :param parameter_names: parameter names for printing / plotting
        :return:
        """
        if self.method in ['IS', 'BBVI']:
            self.compute_statistics_weights(samples, parameter_names)
        elif self.method in ['MH', 'HMC']:
            self.compute_statistics_trace(samples, parameter_names)
        else:
            raise ValueError('Invalid sampler type')

    def compute_statistics_weights(self, samples, parameter_names):
        """
        method to compute the posterior expectation and variance for a sequence of samples and sample weights

        :param samples: samples obtained from the sampling procedure
        :param parameter_names: parameter names for printing / plotting
        :return:
        """
        # separate parameter observations and weights from samples
        temp = [elem[0] for elem in samples]
        weights = torch.FloatTensor([elem[1] for elem in samples])

        # initialize empty list that will contain lists of parameter observations
        parameter_traces = []

        # checks if samples only contains a single parameter
        if temp[0].size() == torch.Size([]):
            parameter_traces.append(torch.FloatTensor(temp))
        else:
            for i in range(len(parameter_names)):
                parameter_traces.append(torch.FloatTensor([elem[i] for elem in temp]))

        flag = False  # flag to check if covariance already reported
        for i, obs in enumerate(parameter_traces):
            # always want to get the posterior expectation of the latents
            posterior_exp = torch.dot(obs, torch.exp(weights)) / torch.sum(torch.exp(weights))
            self.posterior_exp[parameter_names[i]] = posterior_exp
            print('Posterior Expectation {}:'.format(parameter_names[i]), posterior_exp)

            if parameter_names == ['slope', 'bias']:
                # if covariance already reported, continue, no need to print it out again!
                if flag:
                    continue

                # compute covariance in the case of bayesian regression program
                covariance = np.cov(np.array([parameter_traces[i].numpy() for i in range(len(parameter_traces))]),
                                    aweights=np.exp(weights.numpy()))

                print('posterior covariance of slope and bias:\n', covariance)
                flag = True
                continue

            posterior_var = torch.dot(torch.pow((obs - posterior_exp), 2), torch.exp(weights)) / \
                            torch.sum(torch.exp(weights))
            self.posterior_var[parameter_names[i]] = posterior_var
            print('Posterior Variance {}:'.format(parameter_names[i]), posterior_var)

    def compute_statistics_trace(self, samples, parameter_names):
        # initialize empty list that will contain lists of parameter observations
        parameter_traces = []

        # checks if samples only contains a single parameter
        if samples[0].size() == torch.Size([]):
            parameter_traces.append(torch.FloatTensor(samples))
        else:
            for i in range(len(parameter_names)):
                parameter_traces.append(torch.FloatTensor([elem[i] for elem in samples]))

        flag = False
        for i, obs in enumerate(parameter_traces):
            posterior_exp = torch.mean(obs)
            self.posterior_exp[parameter_names[i]] = posterior_exp
            print('Posterior Expectation {}:'.format(parameter_names[i]), posterior_exp)

            if parameter_names == ['slope', 'bias']:
                # if covariance already reported, continue, no need to print it out again!
                if flag:
                    continue

                # compute covariance in the case of bayesian regression program
                covariance = np.cov(np.array([parameter_traces[i].numpy() for i in range(len(parameter_traces))]))

                print('posterior covariance of slope and bias:\n', covariance)
                flag = True
                continue

            posterior_var = torch.var(obs)
            self.posterior_var[parameter_names[i]] = posterior_var
            print('Posterior Variance {}:'.format(parameter_names[i]), posterior_var)


    def plot_values(self, samples, parameter_names, num_points, save_plot, num, program_num, trace):
        """
        method to construct likelihood plots and histograms of the posterior

        :param samples: samples obtained from the sampling procedure
        :param parameter_names: parameter names for printing / plotting
        :return:
        """
        if self.method in ['IS', 'BBVI']:
            self.plot_values_weights(samples, parameter_names, num_points, save_plot, num, program_num, trace)
        elif self.method in ['MH', 'HMC']:
            self.plot_values_trace(samples, parameter_names, num_points, save_plot, num, program_num)
        else:
            raise NotImplementedError('subclasses must override this method!')

    def plot_values_trace(self, samples, parameter_names, num_points, save_plot, num, program_num):
        parameter_traces = []

        # checks if samples only contains a single parameter
        if samples[0].size() == torch.Size([]):
            parameter_traces.append(torch.FloatTensor(samples))
        else:
            for i in range(len(parameter_names)):
                parameter_traces.append(torch.FloatTensor([elem[i] for elem in samples]))

        fig, axs = plt.subplots(len(parameter_names), figsize=(8, 6))
        if len(parameter_names) == 1:
            axs = [axs]

        # histograms
        for i, obs in enumerate(parameter_traces):
            if parameter_names == ['slope', 'bias']:
                axs[i].set_title('{1} posterior exp: {0:.2f}'.format(self.posterior_exp[parameter_names[i]],
                                                                     parameter_names[i]))
            else:
                axs[i].set_title(
                    '{2} posterior exp: {0:.2f}    var: {1:.2f}'.format(self.posterior_exp[parameter_names[i]],
                                                                        self.posterior_var[parameter_names[i]],
                                                                        parameter_names[i]))
            axs[i].hist(obs.numpy().flatten())
            axs[i].set(ylabel='frequency', xlabel=parameter_names[i])

        plt.suptitle('Histogram for Program {0} using {1}'.format(num, self.method))
        plt.tight_layout()

        if save_plot:
            plt.savefig('report/HW3/figures/{0}_program_{1}'.format(self.method, num))

        if num in [1, 2, 7]:

            # trace plots
            fig, axs = plt.subplots(len(parameter_names), figsize=(8, 6))
            if len(parameter_names) == 1:
                axs = [axs]

            for i, obs in enumerate(parameter_traces):
                axs[i].plot(obs[-num_points:].numpy().flatten())
                axs[i].set(ylabel=parameter_names[i], xlabel='iterations')

            plt.suptitle('Trace Plots for Program {0} using {1}'.format(num, self.method))
            plt.tight_layout()

            if save_plot:
                plt.savefig('report/HW3/figures/trace_{0}_program_{1}'.format(self.method, num))

            plt.clf()

            # log joint density
            fig, axs = plt.subplots(len(parameter_names) + 1, figsize=(8, 6))

            # ============== FULL JOINT =======================
            # axs[0].set_title('Full Joint')
            axs[0].set(xlabel='iterations', ylabel='log joint density')

            log_p = self.compute_log_density(samples, num)
            axs[0].plot(log_p.numpy().flatten())

            # =============== INDIVIDUAL JOINTS =======================
            # for i in range(1, len(parameter_names) + 1):
            #     axs[i].set_title('Log joint for {}'.format(parameter_names[i - 1]))
            #     axs[i].set(xlabel='iterations', ylabel='log joint density')
            #
            #     # ignore is a list which will tell us which parameters we want to ignore
            #     # ex. say we only want to find P(slope | data) and ignore bias. ignore=[2] since sample2==bias
            #     ignore = list(range(1, len(parameter_names) + 1))
            #     ignore.remove(i)
            #     log_p = self.compute_log_density(samples, num, ignore=ignore)
            #     axs[i].plot(log_p.numpy().flatten())

            if num == 7:
                plt.suptitle('Log Joint Density Plots for Program {0} using {1}'.format(5, self.method))
            else:
                plt.suptitle('Log Joint Density Plots for Program {0} using {1}'.format(num, self.method))
            plt.tight_layout()

            if save_plot:
                plt.savefig('report/HW3/figures/log_joint_{0}_program_{1}'.format(self.method, num))

    def plot_values_weights(self, samples, parameter_names, num_points, save_plot, num, program_num, trace):

        # separate parameter observations and weights from samples
        temp = [elem[0] for elem in samples]
        weights = torch.FloatTensor([elem[1] for elem in samples])

        # initialize empty list that will contain lists of parameter observations
        parameter_traces = []

        # checks if samples only contains a single parameter
        if temp[0].size() == torch.Size([]):
            parameter_traces.append(torch.FloatTensor(temp))
        else:
            for i in range(len(parameter_names)):
                parameter_traces.append(torch.FloatTensor([elem[i] for elem in temp]))

        fig, axs = plt.subplots(len(parameter_names), figsize=(8, 6))
        if len(parameter_names) == 1:
            axs = [axs]

        # only need to plot histograms of posterior for IS
        for i, obs in enumerate(parameter_traces):
            if parameter_names == ['slope', 'bias']:
                axs[i].set_title('{1} posterior exp: {0:.2f}'.format(self.posterior_exp[parameter_names[i]],
                                                                     parameter_names[i]))
            else:
                axs[i].set_title('{2} posterior exp: {0:.2f}    var: {1:.2f}'.format(self.posterior_exp[parameter_names[i]],
                                                                                     self.posterior_var[parameter_names[i]],
                                                                                     parameter_names[i]))

            if num == 5 or num == 6:
                bin_size = 5
            else:
                bin_size = 2

            if trace:
                axs[i].plot(obs.numpy().flatten())
                axs[i].set(ylabel=parameter_names[i], xlabel='iterations')
            else:
                axs[i].hist(obs.numpy().flatten(),
                            weights=torch.exp(weights).numpy().flatten(),
                            bins=bin_size * math.ceil(np.max(obs.numpy().flatten()) - np.min(obs.numpy().flatten())))
                axs[i].set(ylabel='frequency', xlabel=parameter_names[i])

        if trace:
            plt.suptitle('Trace plots for Program {0} using {1}'.format(program_num, self.method))
            plt.tight_layout()
            if save_plot:
                plt.savefig('report/HW4/figures/{0}_program_{1}_trace'.format(self.method, program_num))
        else:
            plt.suptitle('Histogram for Program {0} using {1}'.format(program_num, self.method))
            plt.tight_layout()
            if save_plot:
                plt.savefig('report/HW4/figures/{0}_program_{1}'.format(self.method, program_num))




    def summary(self, num, samples):
        """
        computes posterior expectation and variance

        :param num: number corresponding to which daphne program was loaded; will affect how we compute summary
        :param samples: list of observations obtained from the sampling procedure
        """

        if num == 1:
            self.compute_statistics(samples, ['mu', 'sigma'])

        if num == 2:
            self.compute_statistics(samples, ['slope', 'bias'])

        if num == 5:
            self.compute_statistics(samples, ['z[1] == z[2]'])

        if num == 6:
            self.compute_statistics(samples, ['is-raining'])

        if num == 7:
            self.compute_statistics(samples, ['x', 'y'])

    def plot(self, num, samples, num_points, save_plot, program_num, trace=False):
        """
        constructs plots

        :param num: number corresponding to which daphne program was loaded; will affect how we compute summary
        :param samples: list of observations obtained from the sampling procedure
        :param num_points: number of points to plot
        :param save_plot: True if we save the plot
        """
        if num == 1:
            self.plot_values(samples, ['mu', 'sigma'], num_points, save_plot, num, program_num, trace)
            self.plot_values(samples, ['mu', 'sigma'], num_points, save_plot, num, program_num, ~trace)

        if num == 2:
            self.plot_values(samples, ['slope', 'bias'], num_points, save_plot, num, program_num, trace)

        if num == 4:
            temp = [elem[0] for elem in samples]
            processed_samples = []
            for obs in temp:
                processed_sample = [torch.squeeze(param) for param in obs]
                processed_samples.append(processed_sample)

            self.plot_heatmap(processed_samples, ['W_0', 'b_0', 'W_1', 'b_1'], save_plot, program_num)

        if num == 5:
            self.plot_values(samples,
                             ['z[1] == z[2]'],
                             num_points,
                             save_plot,
                             num,
                             program_num,
                             trace)

        if num == 6:
            self.plot_values(samples, ['is-raining'], num_points, save_plot, num, program_num)

        if num == 7:
            self.plot_values(samples, ['x', 'y'], num_points, save_plot, num, program_num)

    def plot_heatmap(self, samples, parameter_names, save_plot, program_num):

        parameter_dict = {}
        for idx, parameter_name in enumerate(parameter_names):
            parameter_dict[parameter_name] = np.array([param[idx].numpy() for param in samples])

        # ========= EXPECTATION ==========
        for i, param in enumerate(list(parameter_dict.keys())):
            fig, ax = plt.subplots(figsize=(8, 6))
            plt.title('Heatmap for {} posterior expectation'.format(param))

            if param == 'W_1':
                sns.heatmap(np.mean(parameter_dict[param], axis=0), annot=True, robust=True)
            else:
                sns.heatmap(np.mean(parameter_dict[param], axis=0, keepdims=True), annot=True, robust=True)

            if save_plot:
                plt.savefig('report/HW4/figures/heatmap_exp_{}'.format(param))

        # ========= VARIANCE ==========
        for i, param in enumerate(list(parameter_dict.keys())):
            fig, ax = plt.subplots(figsize=(8, 6))
            plt.title('Heatmap for {} posterior variance'.format(param))

            if param == 'W_1':
                sns.heatmap(np.var(parameter_dict[param], axis=0), annot=True, robust=True)
            else:
                sns.heatmap(np.var(parameter_dict[param], axis=0, keepdims=True), annot=True, robust=True)

            if save_plot:
                plt.savefig('report/HW4/figures/heatmap_var_{}'.format(param))

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


