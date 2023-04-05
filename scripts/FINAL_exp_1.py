#!/usr/bin/env python
#   coding: utf-8

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

EXPERIMENT_1 = True
EXPERIMENT_2 = False
MINIMAL = False

if EXPERIMENT_1:
    coarse_elements = 20
    n = 1200
    use_FEM = True
    use_fine_mesh = True
    use_LOD = True
    ### FOC TOLERANCES
    FEM_FOC_tolerance = 1e-4  # tau_FOC for FEM
    LOD_FOC_tolerance = 1e-6  # tau_FOC for LOD
elif EXPERIMENT_2:
    coarse_elements = 40
    n = 4000
    use_FEM = False
    use_fine_mesh = False
    use_LOD = True
    ### FOC TOLERANCES
    FEM_FOC_tolerance = 1e-4  # tau_FOC for FEM
    LOD_FOC_tolerance = 1e-6  # tau_FOC for LOD
elif MINIMAL:
    coarse_elements = 2
    n = 40
    use_FEM = True
    use_fine_mesh = True
    use_LOD = True
    ### FOC TOLERANCES
    FEM_FOC_tolerance = 1e-4  # tau_FOC for FEM
    LOD_FOC_tolerance = 1e-3  # tau_FOC for LOD
else:
    # define your own experiment config!
    assert 0

"""
######################################################################
    # NOTE: THE REST OF THE SCRIPT IS THE SAME FOR ALL EXPERIMENTS !!!
######################################################################
"""

import numpy as np
from matplotlib import pyplot as plt

from pymor.tools.random import new_rng
from pymor.core.logger import set_log_levels
from pymor.core.defaults import set_defaults
from pymor.core.cache import disable_caching
from pdeopt.tools import print_iterations_and_walltime, extract_further_timings, print_further_timings

def prepare_kernels():
    set_log_levels({'pymor': 'WARN'})
    set_defaults({"pymor.algorithms.gram_schmidt.gram_schmidt.rtol": 1e-4})# <-- very important for the estimator
    set_defaults({"pymor.algorithms.gram_schmidt.gram_schmidt.check": False})
    disable_caching()

use_pool = True
if use_pool:
    from pymor.parallel.mpi import MPIPool
    pool = MPIPool()
    # store_in_tmp = '/scratch/tmp/t_keil02/tr_tsrblod/tmp'  # <---- adjust this depending on your HPC system
    # test_outputs_file = '/scratch/tmp/t_keil02/tr_tsrblod/final/test_outputs' # <---- adjust this depending on your HPC system
    store_in_tmp = 'tmp'
    test_outputs_file = 'test_outputs'
else:
    from pymor.parallel.dummy import DummyPool
    pool = DummyPool()
    store_in_tmp = False
pool.apply(prepare_kernels)
print_on_ranks = False

"""
    fixed variables
"""
save_correctors = False
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

diameter = np.sqrt(2)/n

global_problem, world, local_problem_constructer, f, aFines, f_fine = \
    large_thermal_block(diameter, coarse_elements, blocks=(4, 4), plot=False, return_fine=use_FEM,
                        high_conductivity=high_conductivity, low_conductivity=low_conductivity, rhs_value=rhs_value,
                        first_factor=first_factor, second_factor=second_factor, min_diffusivity=min_diffusivity)
domain_of_interest = None

problem = global_problem

new_rng(23).install()
mu_d = global_problem.parameter_space.sample_randomly(1)[0]
mu_d_array = mu_d.to_numpy()

for i in [3,4,6,7,8,9,11,14]:
    mu_d_array[i] = high_conductivity
for i in [3,4,5,6]:
    mu_d_array[i+25] = low_conductivity

mu_d = mu_d.parameters.parse(mu_d_array)
norm_mu_d = np.linalg.norm(mu_d_array)
# mu_d = None
mu_for_tikhonov = mu_d

sigma_u = 100
weights = {'sigma_u': sigma_u, 'diffusion': 0.001,
           'low_diffusion': 0.001}

optional_enrichment = True

coarse_J = True
N_coarse = coarse_elements
if coarse_J is False:
    assert use_fine_mesh

from pdeopt.tools import EvaluationCounter, LODEvaluationCounter
counter = EvaluationCounter()
lod_counter = LODEvaluationCounter()

print(f'Discretizing grid with diameter {diameter:.2e}')
if use_LOD:
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

if use_FEM:
    opt_fom, data, mu_bar = discretize_quadratic_NCD_pdeopt_stationary_cg(
        problem, diameter, weights.copy(), domain_of_interest=domain_of_interest,
        desired_temperature=None, mu_for_u_d=mu_for_u_d, mu_for_tikhonov=mu_for_tikhonov,
        coarse_functional_grid_size=N_coarse, u_d=u_d)

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
    print(f"number of fine dofs         {gridlod_model.world.NpFine}")

'''
    Variables for the optimization algorithms
'''

seed = 1                   # random seed for the starting value
radius = 0.1               # TR radius 
sub_tolerance = 1e-8       # tau_sub
safety_tol = 1e-14         # Safeguard, to avoid running the optimizer for really small difference in digits
max_it = 60                # Maximum number of iteration for the TR algorithm
max_it_sub = 100           # Maximum number of iteration for the TR optimization subproblem
max_it_arm = 50            # Maximum number of iteration for the Armijo rule
init_step_armijo = 0.5     # Initial step for the Armijo rule
armijo_alpha = 1e-4        # kappa_arm
beta = 0.95                # beta_2
epsilon_i = 1e-8           # Treshold for the epsilon active set (Kelley '99)
control_mu = False

# some implementational variables
Qian_Grepl_subproblem = True

reductor_type = 'simple_coercive'
LOD_reductor_type = 'coercive'
relaxed_reductor_type = 'simple_coercive'
relaxed_LOD_reductor_type = 'coercive'
relaxed_add_error_residual = True

# starting with 
parameter_space = global_problem.parameter_space
new_rng(seed).install()
mu = parameter_space.sample_randomly(1)[0]

# ### What methods do you want to test ?

optimization_methods = [
    # FOM Method
     'BFGS',
     'BFGS_LOD',
    # TR-RB
        # NCD-corrected from KMSOV'20
          'Method_RB', # TR-RB
        # localized BFGS
         'Method_TSRBLOD',
    # R TR Methods
      'Method_R_TR',
      'Method_R_TR_STAR'
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

if mu_d is not None:
    mu_opt = mu_d
else:
    assert 0, 'compute mu_opt in advance'

mu_opt_as_array = mu_opt.to_numpy()
print('Optimal parameter: ', mu_opt_as_array)
print('Norm mu_d: ', norm_mu_d)
if use_FEM:
    J_opt = opt_fom.output_functional_hat(mu_opt)
    J_start = opt_fom.output_functional_hat(mu)
    print('Optimal J FEM: ', J_opt)
if use_LOD:
    J_opt_LOD = gridlod_opt_fom.output_functional_hat(mu_opt, pool=pool)
    print('Optimal J LOD: ', J_opt_LOD)
    if use_FEM:
        print('FEM Error in J LOD: ', J_opt_LOD - J_opt)

print()
print('Starting parameter: ', mu.to_numpy())
if use_LOD:
    J_start_LOD = gridlod_opt_fom.output_functional_hat(mu, pool=pool)
if use_FEM:
    print('Starting J FEM: ', J_start)
else:
    print('Starting J LOD: ', J_start_LOD)
    J_start = J_start_LOD
    J_opt = J_opt_LOD

print('\nParameter space: ', mu_opt.parameters)

'''
    Some plotting
'''

if use_FEM and 0:
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

if use_FEM and 0:
    #### plotting
    from pymor.discretizers.builtin.cg import InterpolationOperator
    vis_mu = mu_d
    diff = InterpolationOperator(data['grid'], problem.diffusion).as_vector(vis_mu)
    # opt_fom.visualize(diff)

    from pdeopt.gridlod_model import construct_aFine_from_mu
    aFine = construct_aFine_from_mu(aFines, global_problem.diffusion.coefficients, mu_d)
    from perturbations_for_2d_data import visualize
    plt.figure()
    visualize.drawCoefficient_origin(np.array([n, n]), aFine, colorbar_font_size=10, logNorm=False)
    # plt.show()

'''
    FOM OPTIMIZATION ALGORITHMS
'''

from pdeopt.tools import compute_errors
from pdeopt.TR import solve_optimization_subproblem_BFGS

counter.reset_counters()
TR_parameters = {'radius': 1.e18, 'sub_tolerance': FEM_FOC_tolerance,
                 'max_iterations_subproblem': 500,
                 'starting_parameter': mu,
                 'epsilon_i': epsilon_i,
                 'max_iterations_armijo': max_it_arm,
                 'initial_step_armijo': init_step_armijo,
                 'armijo_alpha': armijo_alpha,
                 'full_order_model': True}

if 'BFGS' in optimization_methods or 'All' in optimization_methods:
    print("\n________________ FOM BFGS________________________\n")
    muoptfom,_,_,_, times_FOM, mus_FOM, Js_FOM, FOC_FOM = solve_optimization_subproblem_BFGS(
        opt_fom, parameter_space, mu, TR_parameters, timing=True, FOM=True)
    times_full_FOM, J_error_FOM, mu_error_FOM, FOC = compute_errors(
        opt_fom, parameter_space, J_start, J_opt, mu, mu_opt_as_array, mus_FOM, Js_FOM, times_FOM, 0, FOC_FOM)
    times_full_FOM = times_full_FOM[1:]

if 'BFGS' in optimization_methods or 'All' in optimization_methods:
    print("\n________________ FOM BFGS________________________\n")
    BFGS_dict = counter.print_result(True)
    # print_RB_result(BFGS_dict)
    print_iterations_and_walltime(len(times_full_FOM), times_full_FOM[-1])
    print('mu_error: ', mu_error_FOM[-1])
    reference_time = times_full_FOM[-1]
else:
    reference_time = None

counter.reset_counters()

TR_parameters = {'radius': 1.e18, 'sub_tolerance': LOD_FOC_tolerance,
                 'max_iterations_subproblem': 500,
                 'starting_parameter': mu,
                 'epsilon_i': epsilon_i,
                 'max_iterations_armijo': max_it_arm,
                 'initial_step_armijo': init_step_armijo,
                 'armijo_alpha': armijo_alpha,
                 'full_order_model': True}

if 'BFGS_LOD' in optimization_methods or 'All' in optimization_methods:
    print("\n________________FOM LOD BFGS_____________________\n")
    muoptfom,_,_,_, times_FOM_LOD, mus_FOM_LOD, Js_FOM_LOD, FOC_FOM_LOD = solve_optimization_subproblem_BFGS(
        gridlod_opt_fom, parameter_space, mu, TR_parameters, timing=True, FOM=True, pool=pool)
    times_full_FOM_LOD, J_error_FOM_LOD, mu_error_FOM_LOD, FOC_FOM_LOD_ = compute_errors(
        opt_fom, parameter_space, J_start, J_opt, mu, mu_opt_as_array,
        mus_FOM_LOD, Js_FOM_LOD, times_FOM_LOD, 0, FOC_FOM_LOD, pool=pool)
    times_full_FOM_LOD = times_full_FOM_LOD[1:]

if 'BFGS_LOD' in optimization_methods or 'All' in optimization_methods:
    plt.semilogy(times_full_FOM_LOD, FOC_FOM_LOD, label='BFGS LOD')
    plt.legend()

if 'BFGS_LOD' in optimization_methods or 'All' in optimization_methods:
    print("\n________________FOM LOD BFGS_____________________\n")
    BFGSLOD_dict = lod_counter.print_result(True)
    print_iterations_and_walltime(len(times_full_FOM_LOD), times_full_FOM_LOD[-1])
    print('mu_error: ', mu_error_FOM_LOD[-1])
    lod_counter.reset_counters()
    if not use_FEM:
        reference_time = times_full_FOM_LOD[-1]

'''
    ROM OPTIMIZATION ALGORITHMS
'''

from time import perf_counter
from pdeopt.model import build_initial_basis
from pdeopt.reductor import QuadraticPdeoptStationaryCoerciveReductor
from pdeopt.TR import TR_algorithm
from pdeopt.relaxed_TR import Relaxed_TR_algorithm
from pymor.parameters.functionals import MinThetaParameterFunctional

set_defaults({'pymor.operators.constructions.induced_norm.raise_negative': False})
set_defaults({'pymor.operators.constructions.induced_norm.tol': 1e-20})

if use_LOD:
    ce = MinThetaParameterFunctional(gridlod_opt_fom.primal_model.operator.coefficients, mu_bar)
else:
    ce = MinThetaParameterFunctional(opt_fom.primal_model.operator.coefficients, mu_bar)

params = [mu]

# ## NCD corrected BFGS Method (KMSOV'20)
counter.reset_counters()

tic = perf_counter()
if ('Method_R_TR' in optimization_methods and 'Method_RB' in optimization_methods) or 'All' in optimization_methods:
    print("\n_________________Relaxed TR NCD BFGS_____________________\n")
    # make sure to use the correct config
    opt_fom = opt_fom.with_(use_corrected_functional=True)
    opt_fom = opt_fom.with_(adjoint_approach=True)

    RBbasis, dual_RBbasis = build_initial_basis(opt_fom, params, build_sensitivities=False)

    pdeopt_reductor = QuadraticPdeoptStationaryCoerciveReductor(
        opt_fom, RBbasis, dual_RBbasis, opt_product=opt_fom.opt_product,
        coercivity_estimator=ce, reductor_type=relaxed_reductor_type, mu_bar=mu_bar)

    opt_rom = pdeopt_reductor.reduce()

    tictoc = perf_counter() - tic

    TR_parameters = {'Qian-Grepl_subproblem': Qian_Grepl_subproblem, 'beta': beta,
                 'safety_tolerance': safety_tol,
                 'radius': 0.1, 'FOC_tolerance': FEM_FOC_tolerance,
                 'sub_tolerance': sub_tolerance,
                 'max_iterations': max_it, 'max_iterations_subproblem': max_it_sub,
                 'max_iterations_armijo': max_it_arm,
                 'initial_step_armijo': init_step_armijo,
                 'armijo_alpha': armijo_alpha,
                 'epsilon_i': epsilon_i,
                 'control_mu': control_mu,
                 'starting_parameter': mu,
                 'opt_method': 'BFGSMethod'}

    extension_params = {'Enlarge_radius': True, 'timings': True,
                        'opt_fom': opt_fom, 'return_data_dict': True}

    mus_ntr8, times_ntr8, Js_ntr8, FOC_ntr8, data_ntr8 = Relaxed_TR_algorithm(
        opt_rom, pdeopt_reductor, parameter_space, TR_parameters, extension_params, skip_estimator=False)

    times_full_ntr8_actual, J_error_ntr8_actual, mu_error_ntr8_actual, FOC_ntr8_actual = compute_errors(
        opt_fom, parameter_space, J_start, J_opt, mu, mu_opt_as_array, mus_ntr8, Js_ntr8, times_ntr8, tictoc, FOC_ntr8,
        pool=pool)


if ('Method_R_TR' in optimization_methods and 'Method_RB' in optimization_methods) or 'All' in optimization_methods:
    print("\n_________________Relaxed TR NCD BFGS_____________________\n")
    print(f'mu_error: {mu_error_ntr8_actual[-1]:.2e}')
    R_TRNCDRB_dict = counter.print_result(True)
    print_iterations_and_walltime(len(times_full_ntr8_actual), times_full_ntr8_actual[-1])
    R_TRNCDRB_dict_timings = extract_further_timings(times_full_ntr8_actual[-1], data_ntr8, pdeopt_reductor,
                                                     reference_time=reference_time)
    print_further_timings(R_TRNCDRB_dict_timings)

counter.reset_counters()

tic = perf_counter()
if ('Method_R_TR_STAR' in optimization_methods and 'Method_RB' in optimization_methods) or 'All' in optimization_methods:
    print("\n_________________Relaxed TR NCD BFGS without estimation_____________________\n")
    # make sure to use the correct config
    opt_fom = opt_fom.with_(use_corrected_functional=True)
    opt_fom = opt_fom.with_(adjoint_approach=True)
    relaxed_reductor_type = 'non_assembled'

    RBbasis, dual_RBbasis = build_initial_basis(opt_fom, params, build_sensitivities=False)

    pdeopt_reductor = QuadraticPdeoptStationaryCoerciveReductor(opt_fom,
                                                                RBbasis, dual_RBbasis,
                                                                opt_product=opt_fom.opt_product,
                                                                coercivity_estimator=ce,
                                                                reductor_type=relaxed_reductor_type, mu_bar=mu_bar)

    opt_rom = pdeopt_reductor.reduce()

    tictoc = perf_counter() - tic

    TR_parameters = {'Qian-Grepl_subproblem': Qian_Grepl_subproblem, 'beta': beta,
                 'safety_tolerance': safety_tol,
                 'radius': 0.1, 'FOC_tolerance': FEM_FOC_tolerance,
                 'sub_tolerance': sub_tolerance,
                 'max_iterations': max_it, 'max_iterations_subproblem': max_it_sub,
                 'max_iterations_armijo': max_it_arm,
                 'initial_step_armijo': init_step_armijo,
                 'armijo_alpha': armijo_alpha,
                 'epsilon_i': epsilon_i,
                 'control_mu': control_mu,
                 'starting_parameter': mu,
                 'opt_method': 'BFGSMethod'}

    extension_params = {'Enlarge_radius': True, 'timings': True,
                        'opt_fom': opt_fom, 'return_data_dict': True}

    mus_ntr8s, times_ntr8s, Js_ntr8s, FOC_ntr8s, data_ntr8s = Relaxed_TR_algorithm(opt_rom, pdeopt_reductor,
                                                                                   parameter_space, TR_parameters,
                                                                                   extension_params,
                                                                                   skip_estimator=True)

    times_full_ntr8s_actual, J_error_ntr8s_actual, mu_error_ntr8s_actual, FOC_ntr8s_actual = compute_errors(
        opt_fom, parameter_space, J_start, J_opt, mu, mu_opt_as_array, mus_ntr8s, Js_ntr8s, times_ntr8s,
        tictoc, FOC_ntr8s, pool=pool)


if ('Method_R_TR_STAR' in optimization_methods and 'Method_RB' in optimization_methods) or 'All' in optimization_methods:
    print("\n_________________Relaxed TR NCD BFGS without estimation_____________________\n")
    print(f'mu_error: {mu_error_ntr8s_actual[-1]:.2e}')
    R_STAR_TRNCDRB_dict = counter.print_result(True)
    print_iterations_and_walltime(len(times_full_ntr8s_actual), times_full_ntr8s_actual[-1])
    R_STAR_TRNCDRB_dict_timings = extract_further_timings(times_full_ntr8s_actual[-1], data_ntr8s, pdeopt_reductor,
                                                          reference_time=reference_time)
    print_further_timings(R_STAR_TRNCDRB_dict_timings)


counter.reset_counters()

tic = perf_counter()
if 'Method_RB' in optimization_methods or 'All' in optimization_methods:
    print("\n_________________TR NCD BFGS_____________________\n")
    # make sure to use the correct config
    opt_fom = opt_fom.with_(use_corrected_functional=True)
    opt_fom = opt_fom.with_(adjoint_approach=True)

    RBbasis, dual_RBbasis = build_initial_basis(opt_fom, params, build_sensitivities=False)

    pdeopt_reductor = QuadraticPdeoptStationaryCoerciveReductor(opt_fom,
                                                                RBbasis, dual_RBbasis,
                                                                opt_product=opt_fom.opt_product,
                                                                coercivity_estimator=ce,
                                                                reductor_type=reductor_type, mu_bar=mu_bar)

    opt_rom = pdeopt_reductor.reduce()

    tictoc = perf_counter() - tic

    TR_parameters = {'Qian-Grepl_subproblem': Qian_Grepl_subproblem, 'beta': beta,
                 'safety_tolerance': safety_tol,
                 'radius': radius, 'FOC_tolerance': FEM_FOC_tolerance,
                 'sub_tolerance': sub_tolerance,
                 'max_iterations': max_it, 'max_iterations_subproblem': max_it_sub,
                 'max_iterations_armijo': max_it_arm,
                 'initial_step_armijo': init_step_armijo,
                 'armijo_alpha': armijo_alpha,
                 'epsilon_i': epsilon_i,
                 'control_mu': control_mu,
                 'starting_parameter': mu,
                 'opt_method': 'BFGSMethod'}

    extension_params = {'Enlarge_radius': True, 'timings': True,
                        'opt_fom': opt_fom, 'return_data_dict': True}

    mus_8, times_8, Js_8, FOC_8, data_8 = TR_algorithm(
        opt_rom, pdeopt_reductor, parameter_space, TR_parameters, extension_params)

    times_full_8_actual, J_error_8_actual, mu_error_8_actual, FOC_8_actual = compute_errors(
        opt_fom, parameter_space, J_start, J_opt, mu, mu_opt_as_array, mus_8, Js_8, times_8, tictoc, FOC_8, pool=pool)

if 'Method_RB' in optimization_methods or 'All' in optimization_methods:
    plt.semilogy(times_full_8_actual, FOC_8_actual)

if 'Method_RB' in optimization_methods or 'All' in optimization_methods:
    print("\n_________________TR NCD BFGS_____________________\n")
    print(f'mu_error: {mu_error_8_actual[-1]:.2e}')
    TRNCDRB_dict = counter.print_result(True)
    print_iterations_and_walltime(len(times_full_8_actual), times_full_8_actual[-1])
    TRNCDRB_dict_timings = extract_further_timings(times_full_8_actual[-1], data_8, pdeopt_reductor,
                                                     reference_time=reference_time)
    print_further_timings(TRNCDRB_dict_timings)

from pdeopt.RBLOD_reductor import QuadraticPdeoptStationaryCoerciveLODReductor
lod_counter.reset_counters()

# R TR TSRBLOD !!
tic = perf_counter()
if ('Method_R_TR' in optimization_methods and 'Method_TSRBLOD' in optimization_methods) or 'All' in optimization_methods:
    print("\n________________Relaxed TR TSRBLOD BFGS_____________________\n")

    pdeopt_reductor = QuadraticPdeoptStationaryCoerciveLODReductor(gridlod_opt_fom, f,
                                                                   opt_product=gridlod_opt_fom.opt_product,
                                                                   coercivity_estimator=ce,
                                                                   reductor_type=relaxed_LOD_reductor_type,
                                                                   mu_bar=mu_bar,
                                                                   parameter_space=parameter_space,
                                                                   two_scale=True, pool=pool,
                                                                   store_in_tmp=store_in_tmp,
                                                                   optional_enrichment=False,
                                                                   use_fine_mesh=use_fine_mesh,
                                                                   print_on_ranks=print_on_ranks,
                                                                   add_error_residual=relaxed_add_error_residual)

    pdeopt_reductor.extend_bases(mu, pool=pool)
    opt_rom = pdeopt_reductor.reduce()

    tictoc = perf_counter() - tic

    TR_parameters = {'Qian-Grepl_subproblem': Qian_Grepl_subproblem, 'beta': beta,
                     'safety_tolerance': safety_tol,
                     'radius': radius, 'FOC_tolerance': LOD_FOC_tolerance,
                     'sub_tolerance': sub_tolerance,
                     'max_iterations': max_it, 'max_iterations_subproblem': max_it_sub,
                     'max_iterations_armijo': max_it_arm,
                     'initial_step_armijo': init_step_armijo,
                     'armijo_alpha': armijo_alpha,
                     'epsilon_i': epsilon_i,
                     'control_mu': control_mu,
                     'starting_parameter': mu,
                     'opt_method': 'BFGSMethod'}

    extension_params = {'Enlarge_radius': True, 'timings': True,
                        'opt_fom': opt_fom, 'return_data_dict': True}

    mus_ntr, times_ntr, Js_ntr, FOC_ntr, data_ntr = Relaxed_TR_algorithm(opt_rom, pdeopt_reductor, parameter_space,
                                                                         TR_parameters, extension_params, pool=pool,
                                                                         skip_estimator=False)
    #
    times_full_ntr_actual, J_error_ntr_actual, mu_error_ntr_actual, FOC_ntr_actual = compute_errors(
        opt_fom, parameter_space, J_start, J_opt, mu, mu_opt_as_array, mus_ntr, Js_ntr, times_ntr, tictoc, FOC_ntr,
        pool=pool)

if ('Method_R_TR' in optimization_methods and 'Method_TSRBLOD' in optimization_methods) or 'All' in optimization_methods:
    print("\n________________Relaxed TR TSRBLOD BFGS_____________________\n")
    print(f'mu_error: {mu_error_ntr_actual[-1]:.2e}')
    R_TR_TSTRRBLOD_dict = lod_counter.print_result(True)
    print_iterations_and_walltime(len(times_full_ntr_actual), times_full_ntr_actual[-1])
    R_TR_TSTRRBLOD_dict_timings = extract_further_timings(times_full_ntr_actual[-1], data_ntr, pdeopt_reductor,
                                                          reference_time=reference_time)
    print_further_timings(R_TR_TSTRRBLOD_dict_timings)

lod_counter.reset_counters()

# R TR TSRBLOD !!
tic = perf_counter()
if ('Method_R_TR_STAR' in optimization_methods and 'Method_TSRBLOD' in optimization_methods) or 'All' in optimization_methods:
    print("\n________________Relaxed TR TSRBLOD BFGS without estimation_____________________\n")
    relaxed_LOD_reductor_type = 'non_assembled'
    relaxed_add_error_residual = False

    pdeopt_reductor = QuadraticPdeoptStationaryCoerciveLODReductor(gridlod_opt_fom, f,
                                                                   opt_product=gridlod_opt_fom.opt_product,
                                                                   coercivity_estimator=ce,
                                                                   reductor_type=relaxed_LOD_reductor_type,
                                                                   mu_bar=mu_bar,
                                                                   parameter_space=parameter_space,
                                                                   two_scale=True, pool=pool,
                                                                   store_in_tmp=store_in_tmp,
                                                                   optional_enrichment=False,
                                                                   use_fine_mesh=use_fine_mesh,
                                                                   print_on_ranks=print_on_ranks,
                                                                   add_error_residual=relaxed_add_error_residual)

    pdeopt_reductor.extend_bases(mu, pool=pool)
    opt_rom = pdeopt_reductor.reduce()

    tictoc = perf_counter() - tic

    TR_parameters = {'Qian-Grepl_subproblem': Qian_Grepl_subproblem, 'beta': beta,
                     'safety_tolerance': safety_tol,
                     'radius': radius, 'FOC_tolerance': LOD_FOC_tolerance,
                     'sub_tolerance': sub_tolerance,
                     'max_iterations': max_it, 'max_iterations_subproblem': max_it_sub,
                     'max_iterations_armijo': max_it_arm,
                     'initial_step_armijo': init_step_armijo,
                     'armijo_alpha': armijo_alpha,
                     'epsilon_i': epsilon_i,
                     'control_mu': control_mu,
                     'starting_parameter': mu,
                     'opt_method': 'BFGSMethod'}

    extension_params = {'Enlarge_radius': True, 'timings': True,
                        'opt_fom': opt_fom, 'return_data_dict': True}

    mus_ntrs, times_ntrs, Js_ntrs, FOC_ntrs, data_ntrs = Relaxed_TR_algorithm(opt_rom, pdeopt_reductor, parameter_space,
                                                                         TR_parameters, extension_params, pool=pool,
                                                                         skip_estimator=True)
    #
    times_full_ntrs_actual, J_error_ntrs_actual, mu_error_ntrs_actual, FOC_ntrs_actual = compute_errors(
        opt_fom, parameter_space, J_start, J_opt, mu, mu_opt_as_array, mus_ntrs, Js_ntrs, times_ntrs, tictoc, FOC_ntrs,
        pool=pool)

if ('Method_R_TR_STAR' in optimization_methods and 'Method_TSRBLOD' in optimization_methods) or 'All' in optimization_methods:
    print("\n________________Relaxed TR TSRBLOD BFGS without estimation_____________________\n")
    print(f'mu_error: {mu_error_ntrs_actual[-1]:.2e}')
    R_STAR_TR_TSTRRBLOD_dict = lod_counter.print_result(True)
    print_iterations_and_walltime(len(times_full_ntrs_actual), times_full_ntrs_actual[-1])
    R_STAR_TR_TSTRRBLOD_dict_timings = extract_further_timings(times_full_ntrs_actual[-1], data_ntrs, pdeopt_reductor,
                                                          reference_time=reference_time)
    print_further_timings(R_STAR_TR_TSTRRBLOD_dict_timings)

lod_counter.reset_counters()

tic = perf_counter()
if 'Method_TSRBLOD' in optimization_methods or 'All' in optimization_methods:
    print("\n________________TR TSRBLOD BFGS_____________________\n")
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

    pdeopt_reductor.extend_bases(mu, pool=pool)

    opt_rom = pdeopt_reductor.reduce()

    tictoc = perf_counter() - tic

    TR_parameters = {'Qian-Grepl_subproblem': Qian_Grepl_subproblem, 'beta': beta,
                 'safety_tolerance': safety_tol,
                 'radius': radius, 'FOC_tolerance': LOD_FOC_tolerance,
                 'sub_tolerance': sub_tolerance,
                 'max_iterations': max_it, 'max_iterations_subproblem': max_it_sub,
                 'max_iterations_armijo': max_it_arm,
                 'initial_step_armijo': init_step_armijo,
                 'armijo_alpha': armijo_alpha,
                 'epsilon_i': epsilon_i,
                 'control_mu': control_mu,
                 'starting_parameter': mu,
                 'opt_method': 'BFGSMethod'}

    extension_params = {'Enlarge_radius': True, 'timings': True,
                        'opt_fom': opt_fom, 'return_data_dict': True}

    mus_8_tsloc, times_8_tsloc, Js_8_tsloc, FOC_8_tsloc, data_8_tsloc = TR_algorithm(
        opt_rom, pdeopt_reductor, parameter_space, TR_parameters, extension_params, pool=pool)
#     
    times_full_8_tsloc_actual, J_error_8_tsloc_actual, mu_error_8_tsloc_actual, FOC_8_tsloc_actual = compute_errors(
        opt_fom, parameter_space, J_start, J_opt, mu, mu_opt_as_array, mus_8_tsloc, Js_8_tsloc, times_8_tsloc, tictoc,
        FOC_8_tsloc, pool=pool)

if 'Method_TSRBLOD' in optimization_methods or 'All' in optimization_methods:
    print("\n________________TR TSRBLOD BFGS_____________________\n")
    print(f'mu_error: {mu_error_8_tsloc_actual[-1]:.2e}')
    TSTRRBLOD_dict = lod_counter.print_result(True)
    print_iterations_and_walltime(len(times_full_8_tsloc_actual), times_full_8_tsloc_actual[-1])
    TSTRRBLOD_dict_timings = extract_further_timings(times_full_8_tsloc_actual[-1], data_8_tsloc, pdeopt_reductor,
                                                     reference_time=reference_time)
    print_further_timings(TSTRRBLOD_dict_timings)

# # Results

u = opt_fom.solve(mu_opt, pool=pool)
opt_fom.visualize(u) if not use_pool else 1

import tikzplotlib

print("\n#####################################################################################")
print("#################################### SUMMARY ########################################")
print("#####################################################################################")
# ## Plot results

# ### J error

timings_figure = plt.figure()
if 'BFGS' in optimization_methods or 'All' in optimization_methods:
    plt.semilogy(times_full_FOM,J_error_FOM,'o-', label='BFGS Method')
if 'BFGS_LOD' in optimization_methods or 'All' in optimization_methods:
    plt.semilogy(times_full_FOM_LOD,J_error_FOM_LOD,'o-', label='BFGS LOD Method')
if 'Method_RB' in optimization_methods or 'All' in optimization_methods:
    plt.semilogy(times_full_8_actual,J_error_8_actual,'o-', label='TR-RB BFGS')
if ('Method_R_TR' in optimization_methods and 'Method_RB' in optimization_methods) or 'All' in optimization_methods:
    plt.semilogy(times_full_ntr8_actual,J_error_ntr8_actual,'o-', label='R TR-RB BFGS')
if ('Method_R_TR_STAR' in optimization_methods and 'Method_RB' in optimization_methods) or 'All' in optimization_methods:
    plt.semilogy(times_full_ntr8s_actual,J_error_ntr8s_actual,'o-', label='R* TR-RB BFGS')
if 'Method_TSRBLOD' in optimization_methods or 'All' in optimization_methods:
    plt.semilogy(times_full_8_tsloc_actual,J_error_8_tsloc_actual,'o-', label='TR-TSRBLOD BFGS')
if ('Method_R_TR' in optimization_methods and 'Method_TSRBLOD' in optimization_methods) in optimization_methods or 'All' in optimization_methods:
    plt.semilogy(times_full_ntr_actual,J_error_ntr_actual,'o-', label='R TR-TSRBLOD BFGS')
if ('Method_R_TR_STAR' in optimization_methods and 'Method_TSRBLOD' in optimization_methods) in optimization_methods or 'All' in optimization_methods:
    plt.semilogy(times_full_ntrs_actual,J_error_ntrs_actual,'o-', label='R* TR-TSRBLOD BFGS')

plt.xlabel('time in seconds [s]')
plt.ylabel('True optimization error of the output functional')
plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
tikzplotlib.save(f'{test_outputs_file}/exp_1_J_error.tex')

# ### Plot FOC

timings_figure = plt.figure()

if 'BFGS' in optimization_methods or 'All' in optimization_methods:
    plt.semilogy(times_full_FOM,FOC_FOM,'o-', label='BFGS Method')
if 'BFGS_LOD' in optimization_methods or 'All' in optimization_methods:
    plt.semilogy(times_full_FOM_LOD,FOC_FOM_LOD,'o-', label='BFGS LOD Method')
if 'Method_RB' in optimization_methods or 'All' in optimization_methods:
    plt.semilogy(times_full_8_actual,FOC_8_actual,'o-', label='TR-RB BFGS')
if ('Method_R_TR' in optimization_methods and 'Method_RB' in optimization_methods) or 'All' in optimization_methods:
    plt.semilogy(times_full_ntr8_actual,FOC_ntr8_actual,'o-', label='R TR-RB BFGS')
if ('Method_R_TR_STAR' in optimization_methods and 'Method_RB' in optimization_methods) or 'All' in optimization_methods:
    plt.semilogy(times_full_ntr8s_actual,FOC_ntr8s_actual,'o-', label='R* TR-RB BFGS')
if 'Method_TSRBLOD' in optimization_methods or 'All' in optimization_methods:
    plt.semilogy(times_full_8_tsloc_actual,FOC_8_tsloc_actual,'o-', label='TR-TSRBLOD BFGS')
if ('Method_R_TR' in optimization_methods and 'Method_TSRBLOD' in optimization_methods) or 'All' in optimization_methods:
    plt.semilogy(times_full_ntr_actual,FOC_ntr_actual,'o-', label='R TR-TSRBLOD BFGS')
if ('Method_R_TR_STAR' in optimization_methods and 'Method_TSRBLOD' in optimization_methods) or 'All' in optimization_methods:
    plt.semilogy(times_full_ntrs_actual,FOC_ntrs_actual,'o-', label='R* TR-TSRBLOD BFGS')

plt.xlabel('time in seconds [s]')
plt.ylabel('First-order critical condition')
# plt.xlim([-1,30])
plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
tikzplotlib.save(f'{test_outputs_file}/exp_1_FOC.tex')

# ### Plot Mu error

# In[ ]:

if 'BFGS' in optimization_methods or 'All' in optimization_methods:
    plt.semilogy(times_full_FOM,mu_error_FOM,'o-', label='BFGS Method')
if 'BFGS_LOD' in optimization_methods or 'All' in optimization_methods:
    plt.semilogy(times_full_FOM_LOD,mu_error_FOM_LOD,'o-', label='BFGS LOD Method')
if 'Method_RB' in optimization_methods or 'All' in optimization_methods:
    plt.semilogy(times_full_8_actual,mu_error_8_actual,'o-', label='TR-RB BFGS')
if ('Method_R_TR' in optimization_methods and 'Method_RB' in optimization_methods) or 'All' in optimization_methods:
    plt.semilogy(times_full_ntr8_actual,mu_error_ntr8_actual,'o-', label='R TR-RB BFGS')
if ('Method_R_TR_STAR' in optimization_methods and 'Method_RB' in optimization_methods) or 'All' in optimization_methods:
    plt.semilogy(times_full_ntr8s_actual, mu_error_ntr8s_actual, 'o-', label='R* TR-RB BFGS')
if 'Method_TSRBLOD' in optimization_methods or 'All' in optimization_methods:
    plt.semilogy(times_full_8_tsloc_actual,mu_error_8_tsloc_actual,'o-', label='TR-TSRBLOD BFGS')
if ('Method_R_TR' in optimization_methods and 'Method_TSRBLOD' in optimization_methods) or 'All' in optimization_methods:
    plt.semilogy(times_full_ntr_actual,mu_error_ntr_actual,'o-', label='R TR-TSRBLOD BFGS')
if ('Method_R_TR_STAR' in optimization_methods and 'Method_TSRBLOD' in optimization_methods) or 'All' in optimization_methods:
    plt.semilogy(times_full_ntrs_actual,mu_error_ntrs_actual,'o-', label='R* TR-TSRBLOD BFGS')

plt.xlabel('time in seconds [s]')
plt.ylabel('Mu error')
#plt.xlim([-1,100])
plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
tikzplotlib.save(f'{test_outputs_file}/exp_1_mu_error.tex')


if pool is not None:
   del pool
   if use_LOD:
       del gridlod_model, pdeopt_reductor, gridlod_opt_fom

# ## Evaluations of the methods

from pdeopt.tools import print_RB_result, print_RBLOD_result, print_iterations_and_walltime

if 'BFGS' in optimization_methods or 'All' in optimization_methods:
    print("\n________________ FOM BFGS________________________\n")
    print_RB_result(BFGS_dict)
    print()
    print(f'mu_error :             {mu_error_FOM[-1]:.2e}')
    print(f'J_error  :             {J_error_FOM[-1]:.2e}')
    print(f'FOC      :             {FOC_FOM[-1]:.2e}')
    print()
    print_iterations_and_walltime(len(times_full_FOM), times_full_FOM[-1])

if 'BFGS_LOD' in optimization_methods or 'All' in optimization_methods:
    print("\n________________FOM LOD BFGS_____________________\n")
    print_RBLOD_result(BFGSLOD_dict)
    print()
    print(f'mu_error :             {mu_error_FOM_LOD[-1]:.2e}')
    print(f'J_error  :             {J_error_FOM_LOD[-1]:.2e}')
    print(f'FOC      :             {FOC_FOM_LOD[-1]:.2e}')
    print()
    print_iterations_and_walltime(len(times_full_FOM_LOD), times_full_FOM_LOD[-1])
    if reference_time:
        print(f'Speedup:               {reference_time/times_full_FOM_LOD[-1]:.2f}')
if 'Method_RB' in optimization_methods or 'All' in optimization_methods:
    print("\n_________________TR-RB NCD BFGS_____________________\n")
    print_RB_result(TRNCDRB_dict)
    print()
    print(f'mu_error :             {mu_error_8_actual[-1]:.2e}')
    print(f'J_error  :             {J_error_8_actual[-1]:.2e}')
    print(f'FOC      :             {FOC_8_actual[-1]:.2e}')
    print()
    print_iterations_and_walltime(len(times_full_8_actual), times_full_8_actual[-1])
    print_further_timings(TRNCDRB_dict_timings)
if ('Method_R_TR' in optimization_methods and 'Method_RB' in optimization_methods) or 'All' in optimization_methods:
    print("\n_________________Relaxed TR-RB BFGS_____________________\n")
    print_RB_result(R_TRNCDRB_dict)
    print()
    print(f'mu_error :             {mu_error_ntr8_actual[-1]:.2e}')
    print(f'J_error  :             {J_error_ntr8_actual[-1]:.2e}')
    print(f'FOC      :             {FOC_ntr8_actual[-1]:.2e}')
    print()
    print_iterations_and_walltime(len(times_full_ntr8_actual), times_full_ntr8_actual[-1])
    print_further_timings(R_TRNCDRB_dict_timings)
if ('Method_R_TR_STAR' in optimization_methods and 'Method_RB' in optimization_methods) or 'All' in optimization_methods:
    print("\n_________________Relaxed TR-RB BFGS without estimation_____________________\n")
    print_RB_result(R_STAR_TRNCDRB_dict)
    print()
    print_iterations_and_walltime(len(times_full_ntr8s_actual), times_full_ntr8s_actual[-1])
    print(f'mu_error :             {mu_error_ntr8s_actual[-1]:.2e}')
    print(f'J_error  :             {J_error_ntr8s_actual[-1]:.2e}')
    print(f'FOC      :             {FOC_ntr8s_actual[-1]:.2e}')
    print()
    print_further_timings(R_STAR_TRNCDRB_dict_timings)
if 'Method_TSRBLOD' in optimization_methods or 'All' in optimization_methods:
    print("\n________________TR-TSRBLOD BFGS_____________________\n")
    print_RBLOD_result(TSTRRBLOD_dict)
    print()
    print(f'mu_error:              {mu_error_8_tsloc_actual[-1]:.2e}')
    print(f'J_error :              {J_error_8_tsloc_actual[-1]:.2e}')
    print(f'FOC     :              {FOC_8_tsloc_actual[-1]:.2e}')
    print()
    print_iterations_and_walltime(len(times_full_8_tsloc_actual), times_full_8_tsloc_actual[-1])
    print_further_timings(TSTRRBLOD_dict_timings)
if ('Method_R_TR' in optimization_methods and 'Method_TSRBLOD' in optimization_methods) or 'All' in optimization_methods:
    print("\n________________Relaxed TR-TSRBLOD BFGS_____________________\n")
    print_RBLOD_result(R_TR_TSTRRBLOD_dict)
    print()
    print(f'mu_error:              {mu_error_ntr_actual[-1]:.2e}')
    print(f'J_error :              {J_error_ntr_actual[-1]:.2e}')
    print(f'FOC     :              {FOC_ntr_actual[-1]:.2e}')
    print()
    print_iterations_and_walltime(len(times_full_ntr_actual), times_full_ntr_actual[-1])
    print_further_timings(R_TR_TSTRRBLOD_dict_timings)
if ('Method_R_TR_STAR' in optimization_methods and 'Method_TSRBLOD' in optimization_methods) or 'All' in optimization_methods:
    print("\n________________Relaxed TR-TSRBLOD BFGS without estimation_____________________\n")
    print_RBLOD_result(R_STAR_TR_TSTRRBLOD_dict)
    print()
    print(f'mu_error:              {mu_error_ntrs_actual[-1]:.2e}')
    print(f'J_error :              {J_error_ntrs_actual[-1]:.2e}')
    print(f'FOC     :              {FOC_ntrs_actual[-1]:.2e}')
    print()
    print_iterations_and_walltime(len(times_full_ntrs_actual), times_full_ntrs_actual[-1])
    print_further_timings(R_STAR_TR_TSTRRBLOD_dict_timings)
print()