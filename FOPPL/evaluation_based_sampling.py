from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from primitives import primitives_list, evaluate_primitive
import torch
from utils import load_ast, get_distribution, distributions


functions = {}

def evaluate_program(orig_ast, sig={}, sampler=''):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """

    variable_bindings = {'sampler': sampler}

    # first define procedures and bind them to the function dictionary
    if type(orig_ast[0]) is list and orig_ast[0][0] == 'defn':
        # bind procedure definition
        function_expression = orig_ast[0][2:]
        functions[orig_ast[0][1]] = function_expression
        ast = orig_ast[1]
    else:
        # extract first element because orig_ast is a list of a list - e.g. [[ast]]
        ast = orig_ast[0]
    return evaluate_program_helper(ast, sig, variable_bindings)


def evaluate_program_helper(ast, sig, variable_bindings):

    # simply return constants and variables
    if ast in primitives_list:
        return ast, sig
    if ast in distributions:
        return ast, sig
    if type(ast) is torch.Tensor:
        return ast, sig
    if type(ast) in [int, float]:
        return torch.tensor(ast), sig
    if type(ast) is bool:
        return torch.FloatTensor([ast]), sig
    if ast in list(variable_bindings.keys()):
        return variable_bindings[ast], sig
    if ast in list(functions.keys()):
        return functions[ast], sig
    if ast is None:
        return None, sig

    if type(ast) is list:
        if ast[0] == 'sample':
            if variable_bindings['sampler'] == 'IS':
                d, sig = evaluate_program_helper(ast[1], sig, variable_bindings)
                return d.sample(), sig
            if variable_bindings['sampler'] == 'BBVI' and len(ast) == 3:
                # form is ['sample', v, e]
                v = ast[1]

                d, sig = evaluate_program_helper(ast[2], sig, variable_bindings)

                # NOTE: I think v \in dom(sig(Q)) means on line 6 of Alg 11 on pg. 134 is equivalent to checking
                #  if the key exists
                if v not in list(sig['Q'].keys()):
                    dist, sig = evaluate_program_helper(ast[2], sig, variable_bindings)
                    sig['Q'][v] = dist.make_copy_with_grads()
                    sig['O'][v] = sig['optimizer'](sig['Q'][v].Parameters(), lr=sig['lr'])

                c = sig['Q'][v].sample()

                # calculating the function: G(v) = \grad_{\lambda_v} \log{q(X_v | \lambda_v}
                # see line 9 of Alg. 11 on pg. 134 of text

                log_prob = sig['Q'][v].log_prob(c)

                # backpropagate then store gradients for each parameter into sig['G']
                log_prob.backward()

                # store gradients
                if len(sig['Q'][v].Parameters()[0]) > 1:  # check if the parameter is already a list itself
                    sig['G'][v] = sig['Q'][v].Parameters()[0].grad.clone().detach()
                else:
                    sig['G'][v] = torch.FloatTensor([p.grad.clone().detach() for p in sig['Q'][v].Parameters()])
                sig['O'][v].zero_grad()

                logW_v = d.log_prob(c) - sig['Q'][v].log_prob(c)
                sig['logW'] += logW_v
                return c, sig
            else:
                d, sig = evaluate_program_helper(ast[1], sig, variable_bindings)
                return d.sample(), sig
        if ast[0] == 'observe':
            if variable_bindings['sampler'] == 'IS':
                d1, sig = evaluate_program_helper(ast[1], sig, variable_bindings)
                c2, sig = evaluate_program_helper(ast[2], sig, variable_bindings)
                try:
                    sig['logW'] += d1.log_prob(c2)
                except KeyError:
                    raise KeyError('There is no key "logW" in sig')
                return c2, sig
            if variable_bindings['sampler'] == 'BBVI' and len(ast) == 4:
                # form is ['observe', v, e1, e2]
                d1, sig = evaluate_program_helper(ast[2], sig, variable_bindings)
                c2, sig = evaluate_program_helper(ast[3], sig, variable_bindings)
                try:
                    sig['logW'] += d1.log_prob(c2)
                except KeyError:
                    raise KeyError('There is no key "logW" in sig')
                return c2, sig
            else:
                # just sample from the prior like in hw2
                d, sig = evaluate_program_helper(ast[1], sig, variable_bindings)
                return d.sample(), sig
        if ast[0] == 'let':

            if ast[1][0] == 'm':
                flag = True
                while flag:
                    # evaluate the expression that the variable will be bound to
                    binding_obj, sig = evaluate_program_helper(ast[1][1], sig, variable_bindings)

                    flag = ~(torch.abs(binding_obj) > 0.01)

                variable_bindings[ast[1][0]] = binding_obj
                return evaluate_program_helper(ast[2], sig, variable_bindings)

            # evaluate the expression that the variable will be bound to
            binding_obj, sig = evaluate_program_helper(ast[1][1], sig, variable_bindings)

            variable_bindings[ast[1][0]] = binding_obj

            # evaluate the return expression
            return evaluate_program_helper(ast[2], sig, variable_bindings)
        if ast[0] == 'if':
            e1, sig = evaluate_program_helper(ast[1], sig, variable_bindings)
            if e1:
                return evaluate_program_helper(ast[2], sig, variable_bindings)
            else:
                return evaluate_program_helper(ast[3], sig, variable_bindings)

        # evaluate a distribution object
        if ast[0] in distributions:
            curr = [evaluate_program_helper(elem, sig, variable_bindings) for elem in ast]

            # only keep the first element of each sublist (don't need sig)
            curr = list(list(zip(*curr))[0])
            return get_distribution(dist_type=curr[0], parameters=curr[1:]), sig

        # this is the case where we have a list of expressions
        # evaluate each sub-expression
        c = []
        for i in range(1, len(ast)):
            c_i, sig = evaluate_program_helper(ast[i], sig, variable_bindings)
            c.append(c_i)

        # evaluate a procedure
        if ast[0] in list(functions.keys()):
            variables, e = functions[ast[0]]

            for idx, param in enumerate(variables):
                variable_bindings[param] = c[idx]

            return evaluate_program_helper(e, sig, variable_bindings)

        # evaluate a primitive function
        if ast[0] in primitives_list:
            # insert the primitive function at the beginning
            c.insert(0, ast[0])
            return evaluate_primitive(c), sig

    # raise error because there is no valid function in the ast
    raise RuntimeError('Invalid Function', ast)

def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)[0]
    

def run_deterministic_tests():

    debug_start = 1
    for i in range(debug_start,14):
        # note: this path should be with respect to the daphne path!
        # ast = daphne(['desugar', '-i', '../CPSC532W-HW/Kevin/FOPPL/programs/tests/deterministic/test_{}.daphne'.format(i)])
        ast = load_ast('programs/saved_asts/hw2/det{}_ast.pkl'.format(i))
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret, sig = evaluate_program(ast)
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))
        
        print('Test passed', ast, 'test', i)
        # print('Test passed')
        
    print('All deterministic tests passed')
    

def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-4

    debug_start = 1
    for i in range(debug_start,7):
        #note: this path should be with respect to the daphne path!        
        # ast = daphne(['desugar', '-i', '../CPSC532W-HW/Kevin/FOPPL/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        ast = load_ast('programs/saved_asts/hw2/prob{}_ast.pkl'.format(i))
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))

        stream = get_stream(ast)

        p_val = run_prob_test(stream, truth, num_samples)

        print('p value', p_val)
        assert(p_val > max_p_value)
        print('Test passed', ast, 'test', i)
        # print('Test passed')
    
    print('All probabilistic tests passed')



if __name__ == '__main__':

    # run_deterministic_tests()

    # run_probabilistic_tests()

    debug_start = 4
    for i in range(debug_start,5):
        # ast = daphne(['desugar', '-i', '../CPSC532W-HW/Kevin/FOPPL/programs/{}.daphne'.format(i)])

        ast = load_ast('programs/saved_asts/hw2/daphne{}_ast.pkl'.format(i))
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate_program(ast))