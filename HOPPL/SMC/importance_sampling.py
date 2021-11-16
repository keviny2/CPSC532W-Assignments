from evaluator import evaluate, sample_from_prior
import torch
import numpy as np
import json
from daphne import daphne
import sys



def get_IS_sample(exp):
    #init calc:
    output = lambda x: x
    res =  evaluate(exp, env=None)('addr_start', output)
    #TODO : hint, "get_sample_from_prior" as a basis for your solution
    # logWs = []
    while type(res) is tuple:
        cont, args, sigma = res
        res = cont(*args)
    return logW, res

if __name__ == '__main__':
    use_cache = True
    for i in range(1,5):
        if not use_cache:
            exp = daphne(['desugar-hoppl-cps', '-i',
                          '../CPSC532W-HW/Kevin/HOPPL/SMC/programs/{}.daphne'.format(i)])
            with open('programs/{}.json'.format(i), 'w') as f:
                json.dump(exp, f)
        else:
            with open('programs/{}.json'.format(i),'r') as f:
                exp = json.load(f)
        print('\n\n\nSample of prior of program {}:'.format(i))
        log_weights = []
        values = []
        for i in range(10000):
            logW, sample = get_IS_sample(exp)
            log_weights.append(logW)
            values.append(sample)

        log_weights = torch.tensor(log_weights)

        values = torch.stack(values)
        values = values.reshape((values.shape[0],values.size().numel()//values.shape[0]))
        log_Z = torch.logsumexp(log_weights,0) - torch.log(torch.tensor(log_weights.shape[0],dtype=float))

        log_norm_weights = log_weights - log_Z
        weights = torch.exp(log_norm_weights).detach().numpy()
        weighted_samples = (torch.exp(log_norm_weights).reshape((-1,1))*values.float()).detach().numpy()

        print('covariance: ', np.cov(values.float().detach().numpy(),rowvar=False, aweights=weights))
        print('posterior mean:', weighted_samples.mean(axis=0))
