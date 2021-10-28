import torch

primitives_list = ['+', '-', '*', '/', 'sqrt', '<', '<=', '>', '>=', '=', 'and', 'or',
                   'vector', 'hash-map', 'get', 'put', 'first', 'second', 'rest', 'last', 'append',
                   'mat-transpose', 'mat-tanh', 'mat-mul', 'mat-add', 'mat-repmat'
                   ]


def conditional(*args):
    if args[0]:
        return args[1]
    else:
        return args[2]


def vector(*args):
    return evaluate_primitive(['vector', *args])


def sample(*args):
    return args[0].sample()


def observe(*args):
    return args[0].log_prob(args[1])


def hashmap(*args):
    return evaluate_primitive(['hash-map', *args])


def get(*args):
    return evaluate_primitive(['get', *args])


def put(*args):
    return evaluate_primitive(['put', *args])


def first(*args):
    return evaluate_primitive(['first', *args])


def second(*args):
    return evaluate_primitive(['second', *args])


def rest(*args):
    return evaluate_primitive(['rest', *args])


def last(*args):
    return evaluate_primitive(['last', *args])


def append(*args):
    return evaluate_primitive(['append', *args])


def less_than(*args):
    return evaluate_primitive(['<', *args])


def greater_than(*args):
    return evaluate_primitive(['>', *args])


def equal(*args):
    return evaluate_primitive(['=', *args])

def and_op(*args):
    return evaluate_primitive(['and', *args])

def or_op(*args):
    return evaluate_primitive(['or', *args])

def add(*args):
    return evaluate_primitive(['+', *args])


def minus(*args):
    return evaluate_primitive(['-', *args])


def multiply(*args):
    return evaluate_primitive(['*', *args])


def divide(*args):
    return evaluate_primitive(['/', *args])


def mat_transpose(*args):
    return evaluate_primitive(['mat-transpose', *args])


def mat_tanh(*args):
    return evaluate_primitive(['mat-tanh', *args])


def mat_mul(*args):
    return evaluate_primitive(['mat-mul', *args])


def mat_add(*args):
    return evaluate_primitive(['mat-add', *args])


def mat_repmat(*args):
    return evaluate_primitive(['mat-repmat', *args])


def evaluate_primitive(ast):
    if ast[0] == 'mat-mul':
        return torch.matmul(ast[1].float(), ast[2].float())
    if ast[0] == 'mat-add':
        return ast[1] + ast[2]
    if ast[0] == 'mat-repmat':
        return ast[1].repeat(int(ast[2]), int(ast[3]))
    if ast[0] == 'mat-tanh':
        return torch.tanh(ast[1])
    if ast[0] == 'mat-transpose':
        return ast[1].T
    if ast[0] == '+':
        # return torch.sum(torch.tensor(ast[1:]))
        return ast[1] + ast[2]
    elif ast[0] == '-':
        return ast[1] - ast[2]
    elif ast[0] == '*':
        return ast[1] * ast[2]
    elif ast[0] == '/':
        return ast[1] / ast[2]
    elif ast[0] == 'sqrt':
        return torch.sqrt(ast[1])
    elif ast[0] == '<':
        return ast[1] < ast[2]
    elif ast[0] == '>':
        return ast[1] > ast[2]
    elif ast[0] == '=':
        return ast[1] == ast[2]
    elif ast[0] == 'and':
        return ast[1] and ast[2]
    elif ast[0] == 'or':
        return ast[1] or ast[2]
    if ast[0] == 'vector':
        try:
            for i in range(1, len(ast)):
                ast[i] = ast[i].clone().detach()
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

