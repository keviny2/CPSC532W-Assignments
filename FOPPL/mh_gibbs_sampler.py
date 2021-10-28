from graph_based_sampling import sample_from_joint, deterministic_eval
from sampler import Sampler
from utils import load_ast, substitute_sampled_vertices
import re
import copy
import torch
import matplotlib.pyplot as plt
import numpy as np


class MHGibbsSampler(Sampler):
    def __init__(self, method):
        super().__init__(method)
        self.Q = {}

    def sample(self, num_samples, num, summary=True, plot=True):
        """
        metropolis within gibbs sampler

        """
        print('=' * 10, 'Metropolis within Gibbs', '=' * 10)
        graph = load_ast('programs/saved_asts/hw3/program{}_graph.pkl'.format(num))

        # initialize latent variables
        _ = sample_from_joint(graph)

        return self.gibbs(graph, num_samples)

    def gibbs(self, graph, num_samples):
        # make a deep copy of the input graph
        curr_graph = copy.deepcopy(graph)

        # samples will store the observations
        samples = []
        for s in range(num_samples):
            # run the gibbs sampler
            temp_graph = self.gibbs_step(copy.deepcopy(curr_graph))

            # evaluate to obtain a sample
            raw_expression = temp_graph[2]  # this will get the return expression of the graph
            variable_bindings = temp_graph[1]['Y']
            expression = substitute_sampled_vertices(raw_expression, variable_bindings)
            samples.append(deterministic_eval(expression))

            # reassign the current graph
            curr_graph = temp_graph

        return samples


    def gibbs_step(self, graph):
        """
        performs one entire gibbs sweep

        :param graph: current state of the graph
        :return:
        """
        # get a list of the latent variables
        regex = re.compile(r'observe*')
        latent_vars = [vertex for vertex in graph[1]['V'] if not regex.match(vertex)]

        # populate the proposal map self.Q
        self.populate_proposals(graph, latent_vars)

        for latent_var in latent_vars:
            # get the prior distribution for node latent_var
            d = deterministic_eval(substitute_sampled_vertices(self.Q[latent_var], graph[1]['Y']))

            # make a copy of the graph (line 15 of algorithm 1 on pg.82 in the text)
            graph_propose = copy.deepcopy(graph)

            # sample a new latent_var from the proposal distribution (just the prior for now)
            graph_propose[1]['Y'][latent_var] = d.sample()

            # compute acceptance ratio
            alpha = self.accept(latent_var, graph_propose, graph)

            # MH step
            u = torch.rand(1)
            if u < alpha:
                graph = copy.deepcopy(graph_propose)

        return graph

    def accept(self, latent_var, graph_propose, graph):
        """
        compute acceptance probability

        :param latent_var: node of interest
        :param graph_propose: new proposed graph
        :param graph: original graph
        :return:
        """
        # both transition kernels are the same because we're just sampling from the prior
        d_old = deterministic_eval(substitute_sampled_vertices(self.Q[latent_var], graph[1]['Y']))
        d_new = deterministic_eval(substitute_sampled_vertices(self.Q[latent_var], graph_propose[1]['Y']))

        # create variables for the old and new parameter values
        old_value = graph[1]['Y'][latent_var]
        new_value = graph_propose[1]['Y'][latent_var]

        # first part of the log prob
        log_alpha = d_new.log_prob(old_value) - d_old.log_prob(new_value)

        # v_x contains all the children of node latent_var
        v_x = graph[1]['A'][latent_var]

        # compute the ratio of the joint likelihoods
        variable_bindings = graph[1]['Y']
        variable_bindings_propose = graph_propose[1]['Y']
        for v in v_x:
            # substitute variable bindings
            raw_expression = graph_propose[1]['P'][v]
            expression = substitute_sampled_vertices(raw_expression, variable_bindings_propose)
            log_alpha += deterministic_eval(expression)

            # substitute variable bindings
            raw_expression = graph[1]['P'][v]
            expression = substitute_sampled_vertices(raw_expression, variable_bindings)
            log_alpha -= deterministic_eval(expression)

        # NEED TO INCLUDE THE LATENT VARIABLE ITSELF TOO!!!!
        # likelihood under new model
        raw_expression = ['observe*', graph_propose[1]['P'][latent_var][1], graph_propose[1]['Y'][latent_var]]
        expression = substitute_sampled_vertices(raw_expression, variable_bindings_propose)
        log_alpha += deterministic_eval(expression)

        # likelihood under old model
        raw_expression = ['observe*', graph[1]['P'][latent_var][1], graph[1]['Y'][latent_var]]
        expression = substitute_sampled_vertices(raw_expression, variable_bindings)
        log_alpha -= deterministic_eval(expression)

        # exponentiate to exit log-space
        return torch.exp(log_alpha)

    def populate_proposals(self, graph, latent_vars):
        for latent_var in latent_vars:
            self.Q[latent_var] = graph[1]['P'][latent_var][1]

    def compute_statistics(self, samples, parameter_names):

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

    def plot_values(self, samples, parameter_names, num_points, save_plot, num):

        # initialize empty list that will contain lists of parameter observations
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
