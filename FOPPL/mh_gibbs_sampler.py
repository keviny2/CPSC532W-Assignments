from graph_based_sampling import sample_from_joint, deterministic_eval
from sampler import Sampler
from utils import load_ast, substitute_sampled_vertices
import re
import copy
import torch


class MHGibbsSampler(Sampler):
    def __init__(self):
        super().__init__('MH')
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

