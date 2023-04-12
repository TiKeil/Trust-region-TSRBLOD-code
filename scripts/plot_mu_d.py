#!/usr/bin/env python
#   coding: utf-8

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
from matplotlib import pyplot as plt

from pymor.core.logger import set_log_levels
from pymor.core.defaults import set_defaults
from pymor.core.cache import disable_caching
from pdeopt.tools import print_iterations_and_walltime
set_log_levels({'pymor': 'ERROR',
                'notebook': 'INFO'})

def prepare_kernels():
    set_log_levels({'pymor': 'WARN'})
    set_defaults({"pymor.algorithms.gram_schmidt.gram_schmidt.rtol": 1e-4})# <-- very important for the estimator
    set_defaults({"pymor.algorithms.gram_schmidt.gram_schmidt.check": False})
    disable_caching()

use_pool = True
if use_pool:
    from pymor.parallel.mpi import MPIPool
    pool = MPIPool()
    # store_in_tmp = '/scratch/tmp/t_keil02/lrblod/tmp'
    store_in_tmp = 'tmp'
else:
    from pymor.parallel.dummy import DummyPool
    pool = DummyPool()
    store_in_tmp = False
pool.apply(prepare_kernels)
print_on_ranks = True

'''
    Variables for the experiment and discretization
'''

coarse_elements = 20
n = 1200
diameter = np.sqrt(2)/n

two_scale_estimator_for_RBLOD = False
save_correctors = False

use_FEM = True
#use_FEM = False
use_fine_mesh = True
#use_fine_mesh = False

# skip_estimator = False
skip_estimator = True

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

mu_d = global_problem.parameter_space.sample_randomly(1, seed=23)[0]
mu_d_array = mu_d.to_numpy()

for i in [3,4,6,7,8,9,11,14]:
    mu_d_array[i] = high_conductivity
for i in [3,4,5,6]:
    mu_d_array[i+25] = low_conductivity

mu_d = mu_d.parameters.parse(mu_d_array)
norm_mu_d = np.linalg.norm(mu_d_array)
# mu_d = None

'''
    Some plotting
'''

#### plotting
from pdeopt.gridlod_model import construct_aFine_from_mu
from perturbations_for_2d_data import visualize

vis_mu_block_1_array = mu_d_array.copy()
vis_mu_block_2_array = mu_d_array.copy()
for i in range(0,len(mu_d_array),2):
    vis_mu_block_1_array[i] = 0
    vis_mu_block_2_array[i+1] = 0
vis_mu_block_1 = mu_d.parameters.parse(vis_mu_block_1_array)
vis_mu_block_2 = mu_d.parameters.parse(vis_mu_block_2_array)

plt.figure()
aFine = construct_aFine_from_mu(aFines, global_problem.diffusion.coefficients, mu_d)
visualize.drawCoefficient_origin(np.array([n, n]), aFine, colorbar_font_size=10, logNorm=False)

plt.figure()
aFine = construct_aFine_from_mu(aFines, global_problem.diffusion.coefficients, vis_mu_block_1)
visualize.drawCoefficient_origin(np.array([n, n]), aFine, colorbar_font_size=10, logNorm=False)

plt.figure()
aFine = construct_aFine_from_mu(aFines, global_problem.diffusion.coefficients, vis_mu_block_2)
visualize.drawCoefficient_origin(np.array([n, n]), aFine, colorbar_font_size=10, logNorm=False)

plt.show()

