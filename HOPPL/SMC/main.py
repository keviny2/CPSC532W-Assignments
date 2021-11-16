import torch
import matplotlib.pyplot as plt
import numpy as np
import json

from smc import SMC


if __name__ == '__main__':

    plot = True
    n_particles_list = [1, 10, 100, 1000, 10000, 100000]
    for i in range(3,5):
        with open('programs/{}.json'.format(i), 'r') as f:
            exp = json.load(f)

        for n_particles in n_particles_list:
            logZ, particles = SMC(n_particles, exp)
            print('logZ: ', logZ)
            values = torch.stack(particles)

            if plot:
                values = values.numpy()

                if i == 4:
                    plt.hist(values)
                    plt.title('Program 4 - Particle Count: {0:.2f}\n'
                              'Posterior Expectation: {1:.2f} - Posterior Variance: {2:.2f}\n'
                              'Evidence Estimate: {3:.2f}'.format(n_particles,
                                                                  np.mean(values),
                                                                  np.var(values),
                                                                  logZ))
                    plt.xlabel('values')
                    plt.ylabel('frequency')
                    plt.savefig('report/figures/program4/{}_particles'.format(n_particles))

                if i == 3:
                    num_dim = values.shape[1]
                    fig, ax = plt.subplots(6, 3, figsize=(10, 15))
                    for dim in range(num_dim):
                        plt.subplot(6, 3, dim+1)
                        ax = plt.gca()
                        title = 'Dim. {0}\n ' \
                                'Exp: {1:.2f}' \
                                ' - Var: {2:.2f}'.format(dim+1,
                                                         np.mean(values[:, dim]),
                                                         np.var(values[:, dim]))
                        ax.set(ylabel='frequency', xlabel='state', title=title)
                        ax.hist(values[:, dim])

                    plt.suptitle('Program 3 - Particle Count: {0} - Evidence Estimate: {1:.2f}'.format(n_particles,
                                                                                                       logZ))

                    plt.tight_layout()

                    plt.savefig('report/figures/program3/{}_particles'.format(n_particles))

                if i == 2:
                    plt.hist(values)
                    plt.title('Program 2 - Particle Count: {0:.2f}\n'
                              'Posterior Expectation: {1:.2f} - Posterior Variance: {2:.2f}\n'
                              'Evidence Estimate: {3:.2f}'.format(n_particles,
                                                                  np.mean(values),
                                                                  np.var(values),
                                                                  logZ))
                    plt.xlabel('values')
                    plt.ylabel('frequency')
                    plt.savefig('report/figures/program2/{}_particles'.format(n_particles))

                if i == 1:
                    plt.hist(values)
                    plt.title('Program 1 - Particle Count: {0:.2f}\n'
                              'Posterior Expectation: {1:.2f}\n'
                              'Posterior Variance: {2:.2f}'.format(n_particles,
                                                                   np.mean(values),
                                                                   np.var(values)))
                    plt.xlabel('values')
                    plt.ylabel('frequency')
                    plt.savefig('report/figures/program1/{}_particles'.format(n_particles))

                plt.clf()
