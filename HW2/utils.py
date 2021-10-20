import pickle


def save_ast(file_name, my_ast):
    # faster load

    # saving the list
    open_file = open(file_name, "wb")
    pickle.dump(my_ast, open_file)
    open_file.close()


def load_ast(file_name):

    # loading the list
    open_file = open(file_name, 'rb')
    ret = pickle.load(open_file)
    open_file.close()
    return ret


def substitute_sampled_vertices(expression, variable_bindings):
    if type(expression) is not list:
        if isinstance(expression, str):
            if expression in list(variable_bindings.keys()):
                return variable_bindings[expression]
        return expression

    return [substitute_sampled_vertices(sub_expression, variable_bindings) for sub_expression in expression]


