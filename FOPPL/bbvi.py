from evaluation_based_sampling import evaluate_program
from sampler import Sampler
from utils import load_ast, create_fresh_variables, clone, save_ast
import numpy as np
import matplotlib.pyplot as plt
from daphne import daphne

import torch


class BBVI(Sampler):
    """
    class for black-box variational inference
    """

    def __init__(self,
                 lr=1e-2,
                 optimizer=torch.optim.Adam):

        super().__init__('BBVI')
        self.lr = lr
        self.optimizer = optimizer

    def plot_elbo(self, elbo_trace, program_num):
        """
        plot the elbo
        :param elbo_trace:
        :return:
        """
        fig = plt.figure()
        plt.plot(elbo_trace)
        fig.suptitle('ELBO plot for program {0} \n max: {1:.2f}'.format(program_num, np.max(elbo_trace)))
        plt.xlabel('Iterations')
        plt.ylabel('ELBO')
        fig.savefig('report/HW4/figures/elbo_program_{}'.format(program_num))

    def optimizer_step(self, sig, g_hat):
        """

        :param sig: map containing state of bbvi
        :param g_hat:
        :return:
        """
        for v in list(g_hat.keys()):
            parameters = sig['Q'][v].Parameters()
            for idx, param in enumerate(parameters):
                if len(param) > 1:
                    param.grad = torch.FloatTensor(-g_hat[v])
                else:
                    param.grad = torch.FloatTensor([-g_hat[v][idx]])

            sig['O'][v].step()
            sig['O'][v].zero_grad()
        return sig['Q']

    def elbo_gradients(self, G, logW):
        """
        compute estimate for ELBO gradient

        :param G: list of gradients of log q -> \grad_{\lambda_{v,d}} \log{q(X_v^l ; \lambda_v}
        :param logW: list of logW of importance weight
        :return: dictionary g_hat that contains gradient components for each variable v
        """

        L = len(G)  # get number of parameters

        # obtain the union of all gradient maps
        union = []
        for G_i in G:
            union += list(G_i.keys())
        union = list(set(union))

        # dictionary containing our gradient estimates for each variable v
        g_hat = {}
        for v in union:
            num_params = len(G[0][v])
            # tensors for computing b_hat afterwards
            F_one_to_L_v = torch.empty(0)
            G_one_to_L_v = torch.empty(0)
            for l in range(L):
                if v in list(G[l].keys()):
                    F_l_v = G[l][v] * logW[l]
                else:
                    F_l_v, G[l][v] = torch.zeros(num_params), torch.zeros(num_params)
                # saving each F_l_v and G[l][v] for future computations
                F_one_to_L_v = torch.cat((F_one_to_L_v, F_l_v), 0)
                G_one_to_L_v = torch.cat((G_one_to_L_v, G[l][v]), 0)

            # reshape the tensors
            F_one_to_L_v = torch.reshape(F_one_to_L_v, (L, num_params))
            G_one_to_L_v = torch.reshape(G_one_to_L_v, (L, num_params))

            # line 16 & 17 in Alg. 12 seem to be incorrect; this is following equations 4.41-4.44 instead
            b_hat = torch.empty(0)
            for d in range(F_one_to_L_v.size()[1]):
                F_v_d = F_one_to_L_v[:, d]
                G_v_d = G_one_to_L_v[:, d]
                cov_F_G = np.cov(F_v_d.numpy(), G_v_d.numpy())
                b_hat = torch.nan_to_num(torch.cat((b_hat, torch.FloatTensor([cov_F_G[0, 1] / cov_F_G[1, 1]])), 0), 1)

            g_hat[v] = torch.sum(F_one_to_L_v - b_hat * G_one_to_L_v, dim=0) / L
        return g_hat

    def sample(self, T, L, num, print_progress=True):
        """
        perform black-box variational inference

        :param T: number of iterations
        :param L: number of samples to draw when estimating the gradient
        :param num: specify which program we are evaluating
        :return: a weighted set of samples
        """

        print('=' * 10, 'Black-Box Variational Inference', '=' * 10)

        ast = load_ast('programs/saved_asts/hw3/program{}.pkl'.format(num))
        ast = create_fresh_variables(ast)

        sig = {
            'O': {},   # map to optimizer for each variable
            'Q': {},
            'optimizer': self.optimizer,  # optimizer type (e.g. Adam)
            'lr': self.lr
        }

        samples = []
        bbvi_loss = []
        for t in range(T):

            # ========== PRINTING PURPOSES =============
            if print_progress:
                if t % 100 == 0:
                    print('=' * 5, 'Iteration {}'.format(t), '=' * 5)

                    for key in list(sig['Q'].keys()):
                        print('parameter estimates: {}'.format(sig['Q'][key].Parameters()))

            G_t = []  # each element of G_t will contain the gradient FOR EACH parameter
                         # (so if family is normal, there will be 2 parameters; one for loc and one for scale)

            sig['logW_list'] = []  # reset sig['logW_list'] after updating the parameter

            for l in range(L):
                # reset sig['logW'] at each iteration
                sig['logW'] = 0
                sig['G'] = {}

                # r_tl is the return value of the expression (won't have a value for each parameter)
                r_tl, sig_tl = evaluate_program(ast, sig, self.method)

                G_tl = clone(sig_tl['G'])

                # cause all of a sudden we want to keep track of sigma now too in hw4...
                if num == 1:
                    r_tl = torch.FloatTensor([r_tl, next(iter(sig['Q'].values())).scale.clone().detach()])

                # add to return list
                samples.append([r_tl, sig_tl['logW'].clone().detach()])

                # add to list for self.elbo_gradient and self.optimizer_step functions
                G_t.append(G_tl)
                sig['logW_list'].append(sig_tl['logW'].clone().detach())

            g_hat = self.elbo_gradients(G_t, sig['logW_list'])
            bbvi_loss.append(torch.mean(torch.tensor(sig['logW_list'])))  # make a copy of the ELBO

            sig['Q'] = self.optimizer_step(sig, g_hat)

        print('Variational distribution: {}'.format(sig['Q']))

        return samples, bbvi_loss

