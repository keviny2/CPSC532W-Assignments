import torch
import torch.distributions as dist
from graph import topological_sort

from daphne import daphne

import primitives
from tests import is_tol, run_prob_test,load_truth
from utils import load_ast, substitute_sampled_vertices
from dirac import Dirac

# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
env = {'normal': dist.Normal,
       'beta': dist.Beta,
       'exponential': dist.Exponential,
       'uniform': dist.Uniform,
       'discrete': dist.Categorical,
       'flip': dist.Bernoulli,
       'dirichlet': dist.Dirichlet,
       'gamma': dist.Gamma,
       'dirac': Dirac,
       '+': primitives.add,
       '*': primitives.multiply,
       '-': primitives.minus,
       '/': primitives.divide,
       'sqrt': torch.sqrt,
       'vector': primitives.vector,
       'sample*': primitives.sample,
       'observe*': primitives.observe,
       'hash-map': primitives.hashmap,
       'get': primitives.get,
       'put': primitives.put,
       'first': primitives.first,
       'second': primitives.second,
       'rest': primitives.rest,
       'last': primitives.last,
       'append': primitives.append,
       '<': primitives.less_than,
       '>': primitives.greater_than,
       '=': primitives.equal,
       'and': primitives.and_op,
       'or': primitives.or_op,
       'mat-transpose': primitives.mat_transpose,
       'mat-tanh': primitives.mat_tanh,
       'mat-mul': primitives.mat_mul,
       'mat-add': primitives.mat_add,
       'mat-repmat': primitives.mat_repmat,
       'if': primitives.conditional
       }


def deterministic_eval(exp):
    "Evaluation function for the deterministic target language of the graph based representation."
    if type(exp) is list:
        op = exp[0]
        args = exp[1:]
        return env[op](*map(deterministic_eval, args))
    elif type(exp) in [int, float, bool]:
        # We use torch for all numerical objects in our evaluator
        return torch.tensor(float(exp))
    elif type(exp) is torch.Tensor:
        return exp
    elif exp is None:
        return exp
    else:
        print(exp)
        raise "Expression type unknown."


def sample_from_joint(graph, sampling_order=None):
    "This function does ancestral sampling starting from the prior."

    if sampling_order is None:
        sampling_order = topological_sort(graph)

    # ignore observed variables - do not sample them
    # regex = re.compile(r'observe*')
    # sampling_order = [vertex for vertex in sampling_order if not regex.match(vertex)]

    for vertex in sampling_order:
        # substitute parent nodes with their sampled values
        raw_expression = graph[1]['P'][vertex]
        variable_bindings = graph[1]['Y']
        expression = substitute_sampled_vertices(raw_expression, variable_bindings)

        graph[1]['Y'][vertex] = deterministic_eval(expression)


    # substitute return nodes with sampled values
    raw_expression = graph[2]
    variable_bindings = graph[1]['Y']
    expression = substitute_sampled_vertices(raw_expression, variable_bindings)
    return deterministic_eval(expression)


def get_stream(graph):
    """Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    while True:
        yield sample_from_joint(graph)




#Testing:

def run_deterministic_tests():

    debug_start = 1
    for i in range(debug_start,13):
        #note: this path should be with respect to the daphne path!
        # graph = daphne(['graph','-i','../CPSC532W-HW/Kevin/FOPPL/programs/tests/deterministic/test_{}.daphne'.format(i)])
        graph = load_ast('programs/saved_asts/hw2/graph_deterministic{}.pkl'.format(i))
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = deterministic_eval(graph[-1])
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret,truth,graph))

        # print('Test passed', graph, 'test', i)
        print('Test passed')

    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    #TODO: 
    num_samples=1e4
    max_p_value = 1e-4

    debug_start = 1
    for i in range(debug_start,7):
        #note: this path should be with respect to the daphne path!        
        # graph = daphne(['graph', '-i', '../CPSC532W-HW/Kevin/FOPPL/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        graph = load_ast('programs/saved_asts/hw2/graph_prob{}.pkl'.format(i))
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))

        stream = get_stream(graph)

        p_val = run_prob_test(stream, truth, num_samples)

        print('p value', p_val)
        assert(p_val > max_p_value)
        # print('Test passed', graph, 'test', i)
        print ('Test passed')

    print('All probabilistic tests passed')    


if __name__ == '__main__':
    
    run_deterministic_tests()
    run_probabilistic_tests()

    debug_start = 1
    for i in range(debug_start,5):
        # graph = daphne(['graph','-i','../CPSC532W-HW/Kevin/FOPPL/programs/{}.daphne'.format(i)])
        graph = load_ast('programs/saved_asts/hw2/daphne_graph{}.pkl'.format(i))
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(sample_from_joint(graph))

    