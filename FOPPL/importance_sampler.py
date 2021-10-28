from evaluation_based_sampling import evaluate_program
from sampler import Sampler
from utils import load_ast
import matplotlib.pyplot as plt
import copy
import torch
import numpy as np
import math


class ImportanceSampler(Sampler):

    def __init__(self, method):
        super().__init__(method)

    def sample(self, num_samples, num):
        """
        importance sampling procedure
        """

        print('=' * 10, 'Likelihood Weighting', '=' * 10)

        ast = load_ast('programs/saved_asts/hw3/program{}.pkl'.format(num))
        sig = {'logW': 0}
        samples = []
        for i in range(num_samples):
            r_i, sig_i = evaluate_program(ast, sig, 'IS')
            logW_i = copy.deepcopy(sig['logW'])
            samples.append([r_i, logW_i])
            sig['logW'] = 0

        return samples

    def compute_statistics(self, samples, parameter_names):
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

    def plot_values(self, samples, parameter_names, num_points, save_plot, num=None):

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

            axs[i].hist(obs.numpy().flatten(),
                        weights=torch.exp(weights).numpy().flatten(),
                        bins=bin_size * math.ceil(np.max(obs.numpy().flatten()) - np.min(obs.numpy().flatten())))
            axs[i].set(ylabel='frequency', xlabel=parameter_names[i])

        plt.suptitle('Histogram for Program {0} using {1}'.format(num, self.method))
        plt.tight_layout()

        if save_plot:
            plt.savefig('report/HW3/figures/{0}_program_{1}'.format(self.method, num))



