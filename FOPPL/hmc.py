import copy

from sampler import Sampler
from graph_based_sampling import sample_from_joint, deterministic_eval
from utils import load_ast, substitute_sampled_vertices
import matplotlib.pyplot as plt
import torch
import re
import numpy as np
import math


class HMCSampler(Sampler):

    def __init__(self, method, T, epsilon):
        """

        :param method: sampling type
        :param T: number of leapfrog steps
        :param epsilon: step size
        :param M: prior for sampling momentum
        """
        super().__init__(method)
        self.T = T
        self.epsilon = epsilon
        self.M = None
        self.latent_vars = []

    def sample(self, num_samples, num, summary=True, plot=True):
        """
        hamiltonian monte carlo

        """

        print('=' * 10, 'Hamiltonian Monte Carlo', '=' * 10)
        graph = load_ast('programs/saved_asts/hw3/program{}_graph.pkl'.format(num))

        # find latent variables and store in class variable self.latent_vars
        regex_lat = re.compile(r'sample*')
        self.latent_vars = [vertex for vertex in graph[1]['V'] if regex_lat.match(vertex)]

        # set M since we have the dimensions now
        self.M = torch.eye(len(self.latent_vars))

        # initialize latent variables
        _ = sample_from_joint(graph)

        # get initial state
        X = {}
        Y = {}

        for node in graph[1]['V']:
            val = graph[1]['Y'][node]
            if node in self.latent_vars:
                X[node] = val
                X[node].requires_grad = True
            else:
                Y[node] = val

        return self.hmc(X, Y, graph, num_samples)


    def hmc(self, X, Y, graph, num_samples):

        graph_old = copy.deepcopy(graph)
        X_old = copy.deepcopy(X)

        samples = []
        for s in range(num_samples):

            R_old = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(len(self.latent_vars)),
                                                                                  self.M).sample()

            # X_new is a dictionary containing the proposed new variable bindings
            X_new, R_new = self.leapfrog(X_old, Y, R_old, graph)

            # MH step
            u = torch.rand(1)
            alpha = torch.exp(-self.H(X_new, Y, graph, R_new) + self.H(X_old, Y, graph_old, R_old))
            if u < alpha:
                X_old = X_new

            samples.append(deterministic_eval(substitute_sampled_vertices(graph[2], X_old)))

        return samples


    def leapfrog(self, X, Y, R, graph):
        """
        perform leapfrog integration

        :param X: dictionary containing latent nodes and their corresponding values
        :param Y: dictionary containing observed nodes and their corresponding values
        :param graph: graphical model
        :param R: momentum
        :return: updated values for: X, R_half
        """

        R_half = R - 0.5 * self.epsilon * self.grad_U(X, Y, graph)

        for t in range(self.T - 1):
            # update latent variables
            for idx, node in enumerate(self.latent_vars):
                X[node] = X[node].detach() + (self.epsilon * R_half[idx])
                X[node].requires_grad = True

            R_half = R_half.detach() - self.epsilon * self.grad_U(X, Y, graph)

        # one last update
        for idx, node in enumerate(self.latent_vars):
            X[node] = X[node].detach() + (self.epsilon * R_half[idx])
            X[node].requires_grad = True

        R_half = R_half.detach() - 0.5 * self.epsilon * self.grad_U(X, Y, graph)

        return X, R_half

    def grad_U(self, X, Y, graph):
        """
        obtain the gradient of U given current state

        :param X: dictionary containing latent nodes and their corresponding values
        :param Y: dictionary containing observed nodes and their corresponding values
        :param graph: graphical model
        :return: tensor containing the list of gradients
        """

        # compute gradient
        U = self.U(X, Y, graph)

        # backpropagation
        U.backward(gradient=torch.ones(U.size()))  # BUG: the sampler never accepts, maybe it has something to do with gradients getting messed up

        # construct a list of the gradients for each latent variable
        grad = [X[node].grad for node in self.latent_vars]
        return torch.FloatTensor(grad)

    @staticmethod
    def U(X, Y, graph):
        """
        returns potential energy U (log of the joint)

        :param graph: graphical model
        :return:
        """
        # iterate over every node in the graph to obtain the joint
        log_gamma = 0  # accumulator

        # compute log likelihood for latent nodes
        for node in list(X.keys()):
            # substitute variables with their values
            expression = substitute_sampled_vertices(graph[1]['P'][node], {**X, **Y})

            # if latent compute likelihood using prior distribution
            expression[0] = 'observe*'
            expression.append(X[node])
            log_gamma += deterministic_eval(expression)

        # compute log likelihood for observed nodes
        for node in list(Y.keys()):
            expression = substitute_sampled_vertices(graph[1]['P'][node], {**X, **Y})
            log_gamma += deterministic_eval(expression)

        return -log_gamma

    def H(self, X, Y, graph, R):
        """
        use auxiliary variable technique

        :param graph: graphical model
        :param R: momentum
        :return: exp{-U(X) + 0.5*R.T*inv(M)*R}
        """
        if self.M.size() == torch.Size([]):
            return torch.exp(-self.U(X, Y, graph) +
                             0.5 * torch.matmul(R.T,
                                                torch.FloatTensor([torch.matmul(torch.FloatTensor([self.M]), R)])))

        return torch.exp(-self.U(X, Y, graph) +
                         0.5 * torch.matmul(R.T,
                                            torch.matmul(torch.inverse(self.M), R)))

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

            if num == 2:
                axs[i].hist(obs.numpy().flatten(),
                            bins=5 * math.ceil(np.max(obs.numpy().flatten()) - np.min(obs.numpy().flatten())))
            else:
                axs[i].hist(obs.numpy().flatten())

            axs[i].set(ylabel='frequency', xlabel=parameter_names[i])

        plt.suptitle('Histogram for Program {0} using {1}'.format(num, self.method))
        plt.tight_layout()

        if save_plot:
            plt.savefig('report/HW3/figures/{0}_program_{1}'.format(self.method, num))

        if num in [1, 2, 7]:

            fig, axs = plt.subplots(len(parameter_names), figsize=(8, 6))
            if len(parameter_names) == 1:
                axs = [axs]

            # trace plots
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

