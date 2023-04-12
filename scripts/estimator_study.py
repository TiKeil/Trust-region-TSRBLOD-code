#!/usr/bin/env python
#    coding: utf-8

#
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

#
# ~~~
# This file is part of the paper:
#
#           "A relaxed localized trust-region reduced basis approach for
#                      optimization of multiscale problems"
#
# by: Tim Keil and Mario Ohlberger
#
#   https://github.com/TiKeil/Trust-region-TSRBLOD-code
#
# Copyright 2019-2022 all developers. All rights reserved.
# License: Licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Authors:
#   Tim Keil (2022)
# ~~~

import numpy as np

from pymor.tools.random import new_rng
from pymor.core.logger import set_log_levels
from pymor.core.defaults import set_defaults
from pymor.core.cache import disable_caching

set_log_levels({'pymor': 'ERROR',
                'notebook': 'INFO'})

def prepare_kernels():
    set_log_levels({'pymor': 'WARN'})
    set_defaults({"pymor.algorithms.gram_schmidt.gram_schmidt.rtol": 1e-4})# <-- very important for the estimator
    set_defaults({"pymor.algorithms.gram_schmidt.gram_schmidt.check": False})
    disable_caching()

use_pool = True
# use_pool = False
if use_pool:
    from pymor.parallel.mpi import MPIPool
    pool = MPIPool()
    store_in_tmp = '/scratch/tmp/t_keil02/tr_tsrblod/tmp3'  # <---- adjust this depending on your HPC system
else:
    from pymor.parallel.dummy import DummyPool
    pool = DummyPool()
    store_in_tmp = False
pool.apply(prepare_kernels)
print_on_ranks = False

'''
    Variables for the experiment and discretization
'''

coarse_elements = 20
n = 1200
diameter = np.sqrt(2)/n
max_extensions = 31

save_correctors = False

use_FEM = True
#use_FEM = False
use_fine_mesh = True
#use_fine_mesh = False
use_LOD = True
# use_LOD = False

skip_estimator = False
# skip_estimator = True

add_error_residual = True
# add_error_residual = False

from pdeopt.problems import large_thermal_block
from pdeopt.discretizer import discretize_quadratic_NCD_pdeopt_stationary_cg
from pdeopt.discretize_gridlod import (discretize_gridlod, discretize_quadratic_pdeopt_with_gridlod)

high_conductivity, low_conductivity, min_diffusivity, rhs_value = 4., 1.2, 1., 10.
first_factor, second_factor = 4, 8

print(f'\nVARIABLES: \n'
      f'Coarse elements:        {coarse_elements} x {coarse_elements}\n'
      f'Fine elements:          {n} x {n}\n'
      f'high_c/low_c/min_c:     {high_conductivity}/{low_conductivity}/{min_diffusivity}\n'
      f'rhs/f_1/f_2:            {rhs_value}/{first_factor}/{second_factor}\n')

global_problem, world, local_problem_constructer, f, aFines, f_fine = \
    large_thermal_block(diameter, coarse_elements, blocks=(4, 4), plot=False, return_fine=use_FEM,
                        high_conductivity=high_conductivity, low_conductivity=low_conductivity, rhs_value=rhs_value,
                        first_factor=first_factor, second_factor=second_factor, min_diffusivity=min_diffusivity)
domain_of_interest = None

problem = global_problem

new_rng(23).install()
mu_d = global_problem.parameter_space.sample_randomly(1)[0]
mu_d_array = mu_d.to_numpy()

for i in [3, 4, 6, 7, 8, 9, 11, 14]:
    mu_d_array[i] = high_conductivity
for i in [3, 4, 5, 6]:
    mu_d_array[i+25] = low_conductivity

mu_d = mu_d.parameters.parse(mu_d_array)
norm_mu_d = np.linalg.norm(mu_d_array)
mu_for_tikhonov = mu_d

sigma_u = 100
weights = {'sigma_u': sigma_u, 'diffusion': 0.001,
           'low_diffusion': 0.001}

optional_enrichment = False

coarse_J = True
N_coarse = coarse_elements
if coarse_J is False:
    assert use_fine_mesh

from pdeopt.tools import EvaluationCounter, LODEvaluationCounter
counter = EvaluationCounter()
lod_counter = LODEvaluationCounter()

print(f'Discretizing grid with diameter {diameter:.2e}')
if use_LOD and not use_FEM:
    gridlod_model, _, _, _, _ = discretize_gridlod(global_problem, diameter, coarse_elements, pool=pool,
                                                   counter=lod_counter, save_correctors=save_correctors,
                                                   store_in_tmp=store_in_tmp, use_fine_mesh=use_fine_mesh,
                                                   aFine_constructor=local_problem_constructer,
                                                   print_on_ranks=print_on_ranks)

if not use_fine_mesh:
    u_d = gridlod_model.solve(mu_d, pool=pool)
    mu_for_u_d = None
else:
    u_d = None
    mu_for_u_d = mu_d

if use_LOD:
    gridlod_opt_fom, data, mu_bar = discretize_quadratic_pdeopt_with_gridlod(
        global_problem, diameter, coarse_elements, weights.copy(), domain_of_interest,
        desired_temperature=None, mu_for_u_d=mu_for_u_d, mu_for_tikhonov=mu_for_tikhonov,
        pool=pool, counter=lod_counter, save_correctors=save_correctors, store_in_tmp=store_in_tmp,
        coarse_J=coarse_J, use_fine_mesh=use_fine_mesh, aFine_constructor=local_problem_constructer,
        u_d=u_d, print_on_ranks=print_on_ranks)
    gridlod_model = gridlod_opt_fom.optional_forward_model

if use_FEM:
    opt_fom, data, mu_bar = discretize_quadratic_NCD_pdeopt_stationary_cg(problem,
                                        diameter, weights.copy(),
                                        domain_of_interest=domain_of_interest,
                                        desired_temperature=None,
                                        mu_for_u_d=mu_for_u_d, mu_for_tikhonov=mu_for_tikhonov,
                                        coarse_functional_grid_size=N_coarse,
                                        u_d=u_d)

    ### counting evaluations in opt_fom
    opt_fom = opt_fom.with_(evaluation_counter=counter)
elif use_LOD:
    opt_fom = gridlod_opt_fom
else:
    assert 0, "not method was picked"

print('\nInformation on the grids:')
print(data['grid'])
print()
print(f"Coarse FEM mesh:            {coarse_elements} x {coarse_elements}")
print(f"Fine FEM mesh:              {n} x {n}")
if use_LOD:
    print(f"k:                          {gridlod_model.k}")
    print(f"|log H|:                    {np.abs(np.log(np.sqrt(2) * 1./coarse_elements)):.2f}")
    print(f"number of fine dofs         {gridlod_model.world.NpFine}\n")

reductor_type = 'simple_coercive'
LOD_reductor_type = 'coercive'

parameter_space = global_problem.parameter_space

# ### What methods do you want to test ?

optimization_methods = [
    #      'Method_RB',
          'Method_TSRBLOD',
 ]

#optimization_methods = ['All']

if not use_FEM:
    # filter the methods
    optimization_methods = [string for string in optimization_methods if (string not in ['BFGS','Method_RB'])]

if not add_error_residual:
    # filter the methods
    optimization_methods = [string for string in optimization_methods if (string not in ['Method_TSRBLOD'])]

if use_LOD:
    parameters = gridlod_opt_fom.parameters
else:
    parameters = opt_fom.parameters

from pdeopt.model import build_initial_basis
from pdeopt.reductor import QuadraticPdeoptStationaryCoerciveReductor
from pymor.parameters.functionals import MinThetaParameterFunctional

set_defaults({'pymor.operators.constructions.induced_norm.raise_negative': False})
set_defaults({'pymor.operators.constructions.induced_norm.tol': 1e-20})

if use_LOD:
    ce = MinThetaParameterFunctional(gridlod_opt_fom.primal_model.operator.coefficients, mu_bar)
else:
    ce = MinThetaParameterFunctional(opt_fom.primal_model.operator.coefficients, mu_bar)


counter.reset_counters()
params = []

from pdeopt.RB_greedy import pdeopt_greedy

if 'Method_RB' in optimization_methods or 'All' in optimization_methods:
    print("\n_________________Method_RB_____________________\n")
    # make sure to use the correct config
    opt_fom = opt_fom.with_(use_corrected_functional=True)
    opt_fom = opt_fom.with_(adjoint_approach=True)

    RBbasis, dual_RBbasis = build_initial_basis(opt_fom, params, build_sensitivities=False)

    pdeopt_reductor = QuadraticPdeoptStationaryCoerciveReductor(opt_fom,
                                                RBbasis, dual_RBbasis,
                                                opt_product=opt_fom.opt_product,
                                                coercivity_estimator=ce,
                                                reductor_type=reductor_type, mu_bar=mu_bar)

    training_set = parameter_space.sample_randomly(100)
    data = pdeopt_greedy(opt_fom, pdeopt_reductor, training_set, max_extensions=max_extensions, compute_true_errors=True)
    max_errs = data['max_errs']
    max_err_mus = data['max_err_mus']
    max_true_errs = data['max_true_errs']
    eff = list(np.divide(max_errs, max_true_errs))
    print(f"Max estimates:                  {max_errs}")
    print(f"Corresponding errors:           {max_true_errs}")
    print(f"Corresponding effectivities:    {eff}")

lod_counter.reset_counters()

from pdeopt.RBLOD_reductor import QuadraticPdeoptStationaryCoerciveLODReductor
from pdeopt.LRB_greedy import lrb_greedy

if 'Method_TSRBLOD' in optimization_methods or 'All' in optimization_methods:
    print("\n________________Method_TSRBLOD_____________________\n")
    pdeopt_reductor = QuadraticPdeoptStationaryCoerciveLODReductor(gridlod_opt_fom, f,
                                                opt_product=gridlod_opt_fom.opt_product,
                                                coercivity_estimator=ce,
                                                reductor_type=LOD_reductor_type,
                                                mu_bar=mu_bar,
                                                parameter_space=parameter_space,
                                                two_scale=True, pool=pool,
                                                store_in_tmp=store_in_tmp,
                                                optional_enrichment=optional_enrichment,
                                                use_fine_mesh=use_fine_mesh, print_on_ranks=print_on_ranks,
                                                add_error_residual=add_error_residual)

    training_set = parameter_space.sample_randomly(100)
    data = lrb_greedy(gridlod_opt_fom, pdeopt_reductor, training_set,
                      max_extensions=max_extensions, pool=pool, compute_true_errors=True)
    max_errs = data['max_errs']
    max_err_mus = data['max_err_mus']
    max_true_errs = data['max_true_errs']
    eff = list(np.divide(max_errs, max_true_errs))
    print(f"Max estimates:                  {max_errs}")
    print(f"Corresponding errors:           {max_true_errs}")
    print(f"Corresponding effectivities:    {eff}")
