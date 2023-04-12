# ~~~
# This file is part of the paper:
#
#           "A Relaxed Localized Trust-Region Reduced Basis Approach for
#                      Optimization of Multiscale Problems"
#
# by: Tim Keil and Mario Ohlberger
#
#   https://github.com/TiKeil/Trust-region-TSRBLOD-code
#
# Copyright all developers. All rights reserved.
# License: Licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Author: Tim Keil 
# ~~~

import numpy as np

from pymor.algorithms.greedy import WeakGreedySurrogate, weak_greedy

def pdeopt_greedy(fom, reductor, training_set, error_norm=None,
                  J_atol=None, rtol=None, max_extensions=None, pool=None, compute_true_errors=False):
    surrogate = QuadraticPdeoptSurrogate(fom, reductor)

    print('Global Greedy for J target')
    result = weak_greedy(surrogate, training_set, atol=J_atol, rtol=rtol, max_extensions=max_extensions, pool=pool,
                         compute_true_errors=compute_true_errors)
    print(' ... finished after {} extensions'.format(result['extensions']))
    return result

class QuadraticPdeoptSurrogate(WeakGreedySurrogate):
    def __init__(self, fom, reductor):
        self.__auto_init(locals())
        self.rom = None

    def evaluate(self, mus, return_all_values=False, true_errors=False):
        if self.rom is None:
            with self.logger.block('Reducing ...'):
                self.rom = self.reductor.reduce()

        def estimate(mu):
            u_rom = self.rom.solve(mu)
            p_rom = self.rom.solve_dual(mu)
            Delta_J = self.rom.estimate_output_functional_hat(u_rom, p_rom, mu)
            J_abs = np.abs(self.rom.output_functional_hat(mu))
            return Delta_J/J_abs

        def error(mu):
            return np.abs(self.fom.output_functional_hat(mu) - self.rom.output_functional_hat(mu))

        if true_errors:
            result = [error(mu) for mu in mus]
        else:
            result = [estimate(mu) for mu in mus]

        if return_all_values:
            return np.hstack(result)
        else:
            errs, max_err_mus = result, mus
            max_err_ind = np.argmax(errs)
            if isinstance(errs[max_err_ind], float):
                return errs[max_err_ind], max_err_mus[max_err_ind]
            else:
                return errs[max_err_ind][0], max_err_mus[max_err_ind]

    def extend(self, mu):
        with self.logger.block('Extending basis with solution snapshot ...'):
            self.reductor.extend_bases(mu, printing=True)
        with self.logger.block('Reducing ...'):
            self.rom = self.reductor.reduce()
