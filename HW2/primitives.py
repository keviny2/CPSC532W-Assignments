import torch

math_operations = ['+', '-', '*', '/', 'sqrt', '<', '<=', '>', '>=', '==']

data_structure_operations = ['vector', 'hash-map', 'get', 'put', 'first',
                             'second', 'rest', 'last', 'append']

matrix_operations = ['mat-transpose', 'mat-tanh', 'mat-mul', 'mat-add', 'mat-repmat']

complex_operations = ['sample', 'if', 'defn', 'observe']


def conditional(*args):
    return evaluate_complex_operation(['if', *args])


def vector(*args):
    return evaluate_data_structure_operation(['vector', *args])


def sample(*args):
    return args[0].sample()


def observe(*args):
    return args[0].sample()


def hashmap(*args):
    return evaluate_data_structure_operation(['hash-map', *args])


def get(*args):
    return evaluate_data_structure_operation(['get', *args])


def put(*args):
    return evaluate_data_structure_operation(['put', *args])


def first(*args):
    return evaluate_data_structure_operation(['first', *args])


def second(*args):
    return evaluate_data_structure_operation(['second', *args])


def rest(*args):
    return evaluate_data_structure_operation(['rest', *args])


def last(*args):
    return evaluate_data_structure_operation(['last', *args])


def append(*args):
    return evaluate_data_structure_operation(['append', *args])


def less_than(*args):
    return evaluate_math_operation(['<', *args])


def greater_than(*args):
    return evaluate_math_operation(['>', *args])


def add(*args):
    return evaluate_math_operation(['+', *args])


def minus(*args):
    return evaluate_math_operation(['-', *args])


def multiply(*args):
    return evaluate_math_operation(['*', *args])


def divide(*args):
    return evaluate_math_operation(['/', *args])


def mat_transpose(*args):
    return evaluate_matrix_operation(['mat-transpose', *args])


def mat_tanh(*args):
    return evaluate_matrix_operation(['mat-tanh', *args])


def mat_mul(*args):
    return evaluate_matrix_operation(['mat-mul', *args])


def mat_add(*args):
    return evaluate_matrix_operation(['mat-add', *args])


def mat_repmat(*args):
    return evaluate_matrix_operation(['mat-repmat', *args])


def evaluate_matrix_operation(ast):
    if ast[0] == 'mat-mul':
        return torch.matmul(ast[1].float(), ast[2].float())
    if ast[0] == 'mat-add':
        return ast[1] + ast[2]
    if ast[0] == 'mat-repmat':
        return torch.tensor(ast[1]).repeat(int(ast[2]), int(ast[3]))
    if ast[0] == 'mat-tanh':
        return torch.tanh(ast[1])
    if ast[0] == 'mat-transpose':
        return ast[1].T


def evaluate_complex_operation(ast):
    if ast[0] == 'if':
        if ast[1]:
            return ast[2]
        else:
            return ast[3]

    if ast[0] == 'sample':
        return ast[1].sample()

    if ast[0] == 'observe':
        return ast[1].sample()


def evaluate_math_operation(ast):
    if ast[0] == '+':
        return torch.sum(torch.tensor(ast[1:]))
    elif ast[0] == '-':
        return ast[1] - torch.sum(torch.tensor(ast[2:]))
    elif ast[0] == '*':
        return torch.prod(torch.tensor(ast[1:]))
    elif ast[0] == '/':
        return ast[1] / torch.prod(torch.tensor(ast[2:]))
    elif ast[0] == 'sqrt':
        return torch.sqrt(ast[1])
    elif ast[0] == '<':
        return ast[1] < ast[2]
    elif ast[0] == '>':
        return ast[1] > ast[2]


def evaluate_data_structure_operation(ast):
    if ast[0] == 'vector':
        try:
            for i in range(1, len(ast)):
                ast[i] = torch.tensor(ast[i])
            return torch.stack(ast[1:])
        except:
            return ast[1:]
    elif ast[0] == 'hash-map':
        ret = {}
        for idx, elem in enumerate(ast[1:]):
            if idx % 2 == 0:
                if type(elem) is torch.Tensor:
                    ret[elem.numpy().item()] = ast[1:][idx + 1]
        return ret
    elif ast[0] == 'get':
        if type(ast[2]) is torch.Tensor:
            index = int(ast[2].numpy().item())
        else:
            index = ast[2]
        return ast[1][index]
    elif ast[0] == 'put':
        if type(ast[1]) is dict:
            if type(ast[2]) is torch.Tensor:
                ast[1][ast[2].numpy().item()] = ast[3]
        else:
            ast[1][ast[2]] = ast[3]
        return ast[1]
    elif ast[0] == 'remove':
        del ast[1][ast[2]]
        return ast[1]
    elif ast[0] == 'first':
        return ast[1][0]
    elif ast[0] == 'second':
        return ast[1][1]
    elif ast[0] == 'last':
        return ast[1][-1]
    elif ast[0] == 'rest':
        return ast[1][1:]
    elif ast[0] == 'append':
        return torch.cat((ast[1], torch.tensor([ast[2]])), dim=0)