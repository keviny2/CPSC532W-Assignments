from evaluation_based_sampling import evaluate_program
from sampler import Sampler
from utils import load_ast
import matplotlib.pyplot as plt
import copy
import torch
import numpy as np
import math


class ImportanceSampler(Sampler):

    def __init__(self):
        super().__init__('IS')

    def sample(self, num_samples, num):
        """
        importance sampling procedure
        """

        print('=' * 10, 'Likelihood Weighting', '=' * 10)

        ast = load_ast('programs/saved_asts/hw3/program{}.pkl'.format(num))
        sig = {'logW': 0}
        samples = []
        for i in range(num_samples):
            r_i, sig_i = evaluate_program(ast, sig, self.method)
            logW_i = copy.deepcopy(sig['logW'])
            samples.append([r_i, logW_i])
            sig['logW'] = 0

        return samples

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



