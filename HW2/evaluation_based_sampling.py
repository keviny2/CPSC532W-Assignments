from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from distributions import my_distributions, Distribution
from primitives import math_operations, data_structure_operations, matrix_operations,\
    complex_operations, evaluate_math_operation, evaluate_data_structure_operation, evaluate_complex_operation,\
    evaluate_matrix_operation
import torch
from utils import load_ast


functions = {}

def evaluate_program(orig_ast):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """

    variable_bindings = {}

    if type(orig_ast[0]) is list and orig_ast[0][0] == 'defn':
        function_expression = orig_ast[0][2:]
        functions[orig_ast[0][1]] = function_expression
        ast = orig_ast[1]
    else:
        ast = orig_ast[0]
    return evaluate_program_helper(ast, variable_bindings)


def evaluate_program_helper(ast, variable_bindings):

    # print(ast)

    if type(ast) is not list:
        if ast in math_operations:
            return ast
        if ast in data_structure_operations:
            return ast
        if ast in matrix_operations:
            return ast
        if ast in complex_operations:
            return ast
        if ast in my_distributions:
            return ast
        if type(ast) is torch.Tensor:
            return ast
        if type(ast) in [int, float]:
            return torch.tensor(ast)
        if ast in list(variable_bindings.keys()):
            return variable_bindings[ast]
        if ast in list(functions.keys()):
            return functions[ast]
        if ast is None:
            return None
        else:
            raise RuntimeError('Invalid Function', ast)

    if type(ast) is list:
        if ast[0] == 'let':
            # evaluate the expression that the variable will be bound to
            binding_obj = evaluate_program_helper(ast[1][1], variable_bindings)

            # the variable name is found in let_ast[1][0]
            # update variable_bindings dictionary
            variable_bindings[ast[1][0]] = binding_obj

            # evaluate the return expression
            return evaluate_program_helper(ast[2], variable_bindings)
        if ast[0] in my_distributions:
            curr = [evaluate_program_helper(elem, variable_bindings) for elem in ast]
            return Distribution(dist_type=curr[0], params=curr[1:])
        if ast[0] in math_operations:
            curr = [evaluate_program_helper(elem, variable_bindings) for elem in ast]
            return evaluate_math_operation(curr)
        if ast[0] in data_structure_operations:
            curr = [evaluate_program_helper(elem, variable_bindings) for elem in ast]
            return evaluate_data_structure_operation(curr)
        if ast[0] in complex_operations:
            curr = [evaluate_program_helper(elem, variable_bindings) for elem in ast]
            return evaluate_complex_operation(curr)
        if ast[0] in matrix_operations:
            curr = [evaluate_program_helper(elem, variable_bindings) for elem in ast]
            return evaluate_matrix_operation(curr)
        if ast[0] in list(functions.keys()):
            inputs = [evaluate_program_helper(elem, variable_bindings) for elem in ast[1:]]
            body = functions[ast[0]]

            for idx, param in enumerate(body[0]):
                variable_bindings[param] = inputs[idx]

            return evaluate_program_helper(body[1], variable_bindings)


def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)
    

def run_deterministic_tests():

    debug_start = 1
    for i in range(debug_start,14):
        # note: this path should be with respect to the daphne path!
        # ast = daphne(['desugar', '-i', '../CPSC532W-HW/Kevin/HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        ast = load_ast('programs/saved_asts/det{}_ast.pkl'.format(i))
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret, sig = evaluate_program(ast), '0'
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))
        
        print('Test passed', ast, 'test', i)
        
    print('All deterministic tests passed')
    

def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-4

    debug_start = 1
    for i in range(debug_start,7):
        #note: this path should be with respect to the daphne path!        
        # ast = daphne(['desugar', '-i', '../CPSC532W-HW/Kevin/HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        ast = load_ast('programs/saved_asts/prob{}_ast.pkl'.format(i))
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))

        stream = get_stream(ast)

        p_val = run_prob_test(stream, truth, num_samples)

        print('p value', p_val)
        assert(p_val > max_p_value)
        print('Test passed', ast, 'test', i)
    
    print('All probabilistic tests passed')



if __name__ == '__main__':

    run_deterministic_tests()
    
    run_probabilistic_tests()

    debug_start = 1
    for i in range(debug_start,5):
        # ast = daphne(['desugar', '-i', '../CPSC532W-HW/Kevin/HW2/programs/{}.daphne'.format(i)])

        ast = load_ast('programs/saved_asts/daphne{}_ast.pkl'.format(i))
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate_program(ast))