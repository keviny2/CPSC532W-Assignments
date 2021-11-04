import pickle
import torch
import random
import string
from torch.distributions import beta, exponential
from distributions import Normal, Bernoulli, Gamma, Dirichlet, Categorical, UniformContinuous
from dirac import Dirac

distributions = ['normal', 'beta', 'exponential', 'uniform-continuous', 'discrete', 'flip', 'gamma', 'dirichlet', 'dirac']

tasks = ['Gaussian unknown mean problem',
         'Bayesian linear regression',
         'Hidden Markov Model',
         'Bayesian Neural Net']

# dictionary of ordinal numbers for printing purposes
nth = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth"
}


def clone(dict):
    """
    utility function to clone a dictionary containing lists of tensors for bbvi alg.
    :param dict: dictionary to clone
    :return: cloned dictionary
    """

    ret = {}
    keys = list(dict.keys())
    for key in keys:
        curr = dict[key]

        if curr.size() == torch.Size([]):
            cloned = curr.clone().detach()
        else:
            cloned = torch.FloatTensor([elem.clone().detach() for elem in curr])
        ret[key] = cloned

    return ret


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
        return Normal(torch.FloatTensor([params[0]]), torch.FloatTensor([params[1]]))
    if dist_type == 'beta':
        return beta.Beta(params[0], params[1])
    if dist_type == 'exponential':
        return exponential.Exponential(params[0])
    if dist_type == 'uniform-continuous':
        return UniformContinuous(low=torch.FloatTensor([params[0]]), high=torch.FloatTensor([params[1]]))
    if dist_type == 'discrete':
        return Categorical(probs=torch.FloatTensor(params[0]))
    if dist_type == 'flip':
        return Bernoulli(torch.FloatTensor([params[0]]))
    if dist_type == 'gamma':
        return Gamma(torch.FloatTensor([params[0]]), torch.FloatTensor([params[1]]))
    if dist_type == 'dirichlet':
        return Dirichlet(torch.FloatTensor(params[0]))
    if dist_type == 'dirac':
        return Dirac(params[0])

    raise RuntimeError('{} is not a valid distribution'.format(ast))


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


def generate_random_string(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))


def create_fresh_variables(expression, length=10):
    """

    given an expression, substitute sample and observe statements so that they include a fresh variable
    :param expression: an ast
    :return:
    """
    if type(expression) is not list:
        return expression
    if expression[0] == 'sample':
        expression.insert(1, generate_random_string(length))
    if expression[0] == 'observe':
        expression.insert(1, generate_random_string(length))

    return [create_fresh_variables(sub_expression) for sub_expression in expression]
