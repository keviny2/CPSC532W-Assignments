import torch
from torch.distributions import normal, beta, exponential, uniform, categorical, bernoulli

my_distributions = ['normal', 'beta', 'exponential', 'uniform', 'discrete', 'bernoulli']

class Distribution:
    def __init__(self, dist_type, params):
        self.dist_type = dist_type

        self.params = []
        for param in params:
            if type(param) is torch.Tensor:
                try:
                    self.params.append(param.numpy().item())
                except:
                    self.params.append(param)
            else:
                self.params.append(param)

    def sample(self):
        if self.dist_type == 'normal':
            return torch.tensor(normal.Normal(self.params[0], self.params[1]).sample())
        if self.dist_type == 'beta':
            return torch.tensor(beta.Beta(self.params[0], self.params[1]).sample())
        if self.dist_type == 'exponential':
            return torch.tensor(exponential.Exponential(self.params[0]).sample())
        if self.dist_type == 'uniform':
            return torch.tensor(uniform.Uniform(self.params[0], self.params[1]).sample())
        if self.dist_type == 'discrete':
            return categorical.Categorical(probs=self.params[0]).sample()
        if self.dist_type == 'bernoulli':
            return torch.tensor(bernoulli.Bernoulli(self.params[0]))

