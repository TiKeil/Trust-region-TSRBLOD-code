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
from pymor.parallel.dummy import dummy_pool
from pymor.parallel.interface import RemoteObject


def lrb_greedy(fom, reductor, training_set, use_error_estimator=True, error_norm=None,
               atol=None, rtol=None, max_extensions=None, extension_params=None, pool=None,
               enrichment_type='local',
               compute_true_errors=False):
    """Weak Greedy basis generation using the RB approximation error as surrogate.

    This algorithm generates a reduced basis using the :func:`weak greedy <weak_greedy>`
    algorithm :cite:`BCDDPW11`, where the approximation error is estimated from computing
    solutions of the reduced order model for the current reduced basis and then estimating
    the model reduction error.

    Parameters
    ----------
    fom
        The |Model| to reduce.
    reductor
        Reductor for reducing the given |Model|. This has to be
        an object with a `reduce` method, such that `reductor.reduce()`
        yields the reduced model, and an `exted_basis` method,
        such that `reductor.extend_basis(U, copy_U=False, **extension_params)`
        extends the current reduced basis by the vectors contained in `U`.
        For an example see :class:`~pymor.reductors.coercive.CoerciveRBReductor`.
    training_set
        The training set of |Parameters| on which to perform the greedy search.
    use_error_estimator
        If `False`, exactly compute the model reduction error by also computing
        the solution of `fom` for all |parameter values| of the training set.
        This is mainly useful when no estimator for the model reduction error
        is available.
    error_norm
        If `use_error_estimator` is `False`, use this function to calculate the
        norm of the error. If `None`, the Euclidean norm is used.
    atol
        See :func:`weak_greedy`.
    rtol
        See :func:`weak_greedy`.
    max_extensions
        See :func:`weak_greedy`.
    extension_params
        `dict` of parameters passed to the `reductor.extend_basis` method.
        If `None`, `'gram_schmidt'` basis extension will be used as a default
        for stationary problems (`fom.solve` returns `VectorArrays` of length 1)
        and `'pod'` basis extension (adding a single POD mode) for instationary
        problems.
    pool
        See :func:`weak_greedy`.

    Returns
    -------
    Dict with the following fields:

        :rom:                    The reduced |Model| obtained for the
                                 computed basis.
        :max_errs:               Sequence of maximum errors during the greedy run.
        :max_err_mus:            The parameters corresponding to `max_errs`.
        :extensions:             Number of performed basis extensions.
        :time:                   Total runtime of the algorithm.
    """
    surrogate = LRBSurrogate(fom, reductor, use_error_estimator, error_norm, extension_params,
                             pool or dummy_pool, enrichment_type=enrichment_type,
                             compute_true_errors=compute_true_errors)

    result = weak_greedy(surrogate, training_set, atol=atol, rtol=rtol, max_extensions=max_extensions, pool=None,
                         compute_true_errors=compute_true_errors)
    result['rom'] = surrogate.rom

    return result


class LRBSurrogate(WeakGreedySurrogate):
    """Surrogate for the :func:`weak_greedy` error used in :func:`rb_greedy`.

    Not intended to be used directly.
    """

    def __init__(self, fom, reductor, use_error_estimator, error_norm, extension_params, pool,
                 enrichment_type, compute_true_errors=False):
        self.__auto_init(locals())
        self.dummy_pool = dummy_pool
        if use_error_estimator and not compute_true_errors:
            self.remote_fom, self.remote_error_norm, self.remote_reductor = None, None, None
        else:
            self.remote_fom, self.remote_error_norm, self.remote_reductor = \
                fom, error_norm, reductor
            # TODO: not parallelizable at the moment because dune objects not pickable
            # pool.push(fom), pool.push(error_norm), pool.push(reductor)

        self.rom = None

    def evaluate(self, mus, return_all_values=False, true_errors=False):
        if self.rom is None:
            with self.logger.block('Reducing ...'):
                self.rom = self.reductor.reduce()

        if not isinstance(mus, RemoteObject):
            mus = self.dummy_pool.scatter_list(mus)

        result = self.dummy_pool.apply(_rb_surrogate_evaluate,
                                       rom=self.rom,
                                       fom=self.remote_fom,
                                       reductor=self.remote_reductor,
                                       mus=mus,
                                       error_norm=self.remote_error_norm,
                                       return_all_values=return_all_values,
                                       true_errors=true_errors,
                                       pool=self.pool)
        if return_all_values:
            return np.hstack(result)
        else:
            errs, max_err_mus = list(zip(*result))
            max_err_ind = np.argmax(errs)
            return errs[max_err_ind], max_err_mus[max_err_ind]

    def extend(self, mu):
        if self.enrichment_type == 'global':
            with self.logger.block('Extending basis with solution global snapshots ...'):
                u_global = self.fom.solve(mu)
                self.reductor.extend_bases_with_global_solution(u_global)
        elif self.enrichment_type == 'local':
            with self.logger.block('Extending basis with solution local snapshots ...'):
                self.reductor.enrich_all_locally(mu, pool=self.pool)
        elif self.enrichment_type == 'local_adaptive':
            with self.logger.block('Computing localized errors ...'):
                _, local_ests, _, _ = self.rom.estimate_error(mu)
            tol = self.reductor.local_enrichment_tolerance
            with self.logger.block(f'Enriching with local tolerance {tol} ...'):
                for I, local_est in enumerate(local_ests):
                    if local_est > tol:
                        print(f'Enrichment since {local_est} > {tol}')
                        self.reductor.enrich_locally(I, mu, use_global_matrix=False)
                    else:
                        print(f'skip enrichment since {local_est} < {tol}')
        else:
            assert 0, 'Enrichment type not known'
        # with self.logger.block('Extending basis with solution snapshot ...'):
            # extension_params = self.extension_params
            # if len(U) > 1:
            #     if extension_params is None:
            #         extension_params = {'method': 'pod'}
            #     else:
            #         extension_params.setdefault('method', 'pod')
            # self.reductor.extend_basis(U, copy_U=False, **(extension_params or {}))
            # if not self.use_error_estimator:
            #     self.remote_reductor = self.pool.push(self.reductor)
        with self.logger.block('Reducing ...'):
            self.rom = self.reductor.reduce()


def _rb_surrogate_evaluate(rom=None, fom=None, reductor=None, mus=None, error_norm=None, return_all_values=False,
                           true_errors=False, pool=None):
    if not mus:
        if return_all_values:
            return []
        else:
            return -1., None

    def estimate(mu):
        u_rom = rom.solve(mu)
        p_rom = rom.solve_dual(mu)
        Delta_J = rom.estimate_output_functional_hat(u_rom, p_rom, mu)
        J_abs = np.abs(rom.output_functional_hat(mu))
        return Delta_J / J_abs

    def error(mu):
        return np.abs(fom.output_functional_hat(mu, pool=pool) - rom.output_functional_hat(mu))

    if fom is None or true_errors is False:
        errors = [estimate(mu) for mu in mus]
    elif error_norm is not None:
        errors = [error_norm(fom.solve(mu) - reductor.reconstruct(rom.solve(mu))) for mu in mus]
    else:
        errors = [error(mu) for mu in mus]
    # most error_norms will return an array of length 1 instead of a number,
    # so we extract the numbers if necessary
    errors = [x[0] if hasattr(x, '__len__') else x for x in errors]
    if return_all_values:
        return errors
    else:
        max_err_ind = np.argmax(errors)
        return errors[max_err_ind], mus[max_err_ind]
