import pickle

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