import torch


class Dirac:
    def __init__(self, param):
        """
        implementation of dirac function

        :param param: x + y
        """
        self.param = param
        self.log_norm_const = torch.log(torch.tensor(2)) + torch.lgamma(torch.tensor(5/4))

    def log_prob(self, obs):
        log_prob = -(self.param - obs)**4 - self.log_norm_const
        return log_prob

