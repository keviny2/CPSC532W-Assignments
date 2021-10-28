import pickle
import torch
from torch.distributions import normal, beta, exponential, uniform, categorical, bernoulli, gamma, dirichlet
from dirac import Dirac

distributions = ['normal', 'beta', 'exponential', 'uniform', 'discrete', 'flip', 'gamma', 'dirichlet', 'dirac']

tasks = ['Gaussian unknown mean problem',
         'Bayesian linear regression',
         'Hidden Markov Model',
         'Bayesian Neural Net']


def get_distribution(dist_type, parameters):
    params = []
    for param in parameters:
        if type(param) is torch.Tensor:
            try:
                params.append(param.numpy().item())
            except:
                params.append(param)
        else:
            params.append(param)

    if dist_type == 'normal':
        return normal.Normal(params[0], params[1])
    if dist_type == 'beta':
        return beta.Beta(params[0], params[1])
    if dist_type == 'exponential':
        return exponential.Exponential(params[0])
    if dist_type == 'uniform':
        return uniform.Uniform(params[0], params[1])
    if dist_type == 'discrete':
        return categorical.Categorical(probs=params[0])
    if dist_type == 'flip':
        return bernoulli.Bernoulli(params[0])
    if dist_type == 'gamma':
        return gamma.Gamma(params[0], params[1])
    if dist_type == 'dirichlet':
        return dirichlet.Dirichlet(params[0])
    if dist_type == 'dirac':
        # TODO: implement the dirac function
        # IDEA: can try to use sympy.functions.special.delta_functions.DiracDelta()
        return Dirac(params[0])


def save_ast(file_name, my_ast):
    """
    saves an ast compiled by daphne in a pickle file

    :param file_name: path to where the saved ast will be saved
    :param my_ast: ast object to be stored
    :return:
    """

    open_file = open(file_name, "wb")
    pickle.dump(my_ast, open_file)
    open_file.close()


def load_ast(file_name):
    """
    loads a saved ast

    :param file_name: path to the pickle file
    :return:
    """

    open_file = open(file_name, 'rb')
    ret = pickle.load(open_file)
    open_file.close()
    return ret


def substitute_sampled_vertices(expression, variable_bindings):
    """
    given an ast, substitute all variable strings with their corresponding values

    :param expression: an ast
    :param variable_bindings: a map between variable names and variable values
    :return:
    """
    if type(expression) is not list:
        if isinstance(expression, str):
            if expression in list(variable_bindings.keys()):
                return variable_bindings[expression]
        return expression

    return [substitute_sampled_vertices(sub_expression, variable_bindings) for sub_expression in expression]


