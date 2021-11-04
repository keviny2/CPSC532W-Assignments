from importance_sampler import ImportanceSampler
from mh_gibbs_sampler import MHGibbsSampler
from hmc import HMCSampler
from bbvi import BBVI
import time
from utils import nth


if __name__ == "__main__":
    daphne_input_nums = [1, 2, 5, 4, 8]
    num_samples = int(4000)
    num_points = 10000  # number of points to plot

    debug_start = 2
    importance_sampler = ImportanceSampler()
    mh_gibbs_sampler = MHGibbsSampler()
    hmc_sampler = HMCSampler(T=10, epsilon=0.1)
    bbvi = BBVI(lr=1e-1)
    for idx, num in enumerate(daphne_input_nums[debug_start:], 1 + debug_start):
        print()

        start = time.time()
        samples, bbvi_loss = bbvi.sample(T=num_samples, L=int(1e2), num=num)
        end = time.time()
        print('Took {0:.2f} seconds to finish Program {1}'.format(end - start, idx))

        bbvi.summary(num, samples)
        bbvi.plot(num, samples, num_points, save_plot=True, program_num=idx, trace=False)
        bbvi.plot_elbo(bbvi_loss, idx)

        # ================================HW3=========================================

        # # ============ IS ===============
        # start = time.time()
        # samples = importance_sampler.sample(num_samples, num)
        # end = time.time()
        # print('Took {0:.2f} seconds to finish Program {1}'.format(end - start, num))
        #
        # importance_sampler.summary(num, samples)
        # importance_sampler.plot(num, samples, num_points, save_plot=True)
        #
        #
        # # ============ MH-Gibbs ===============
        # if num == 2:
        #     num_samples = 25000
        #
        # if num == 5:
        #     num_samples = 3000
        #
        # if num == 6:
        #     num_samples = 6000
        #
        # if num == 7:
        #     num_samples = 20000
        #
        # start = time.time()
        # samples = mh_gibbs_sampler.sample(num_samples, num)
        # end = time.time()
        # print('Took {0:.2f} seconds to finish Program {1}'.format(end - start, num))
        #
        # mh_gibbs_sampler.summary(num, samples)
        # mh_gibbs_sampler.plot(num, samples, num_points, save_plot=True)
        #
        # # ============ HMC ===============
        # if num in [5, 6]:
        #     num_samples = 100000
        #     continue
        #
        # if num == 1:
        #     num_samples = 20000
        #
        # if num == 2:
        #     num_samples = 10000
        #
        # if num == 7:
        #     num_samples = 20000
        #
        # # HMC
        # attempt = 1
        # while True:
        #     try:
        #         print(nth[attempt], 'attempt')
        #         start = time.time()
        #         samples = hmc_sampler.sample(num_samples, num)
        #         end = time.time()
        #     except ValueError:
        #         attempt += 1
        #         continue
        #     break
        #
        # print('Took {0} seconds to finish Program {1}'.format(end - start, num))
        #
        # # start = time.time()
        # # samples = hmc_sampler.sample(num_samples, num)
        # # end = time.time()
        # # print('Took {0:.2f} seconds to finish Program {1}'.format(end - start, num))
        #
        # hmc_sampler.summary(num, samples)
        # hmc_sampler.plot(num, samples, num_points, save_plot=True)
        #
        # num_samples = 100000
        #
