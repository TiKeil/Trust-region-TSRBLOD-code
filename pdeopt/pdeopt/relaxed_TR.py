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

from pdeopt.TR import solve_optimization_subproblem_NewtonMethod
from pdeopt.TR import solve_optimization_subproblem_BFGS
from pdeopt.TR import enrichment_step, projection_onto_range
from pdeopt.RBLOD_reductor import QuadraticPdeoptStationaryCoerciveLODReductor

import time
import numpy as np

def Relaxed_TR_algorithm(opt_rom, reductor, parameter_space, TR_parameters=None, extension_params=None, opt_fom=None,
                         return_opt_rom=False, pool=None, mesh_adaptive=False, skip_estimator=True):
    if TR_parameters is None:
        assert 0
    else:
        if 'radius' not in TR_parameters:
            TR_parameters['radius'] = 0.1
        if 'sub_tolerance' not in TR_parameters:
            TR_parameters['sub_tolerance'] = 1e-8
        if 'max_iterations' not in TR_parameters:
            TR_parameters['max_iterations'] = 40
        if 'max_iterations_subproblem' not in TR_parameters:
            TR_parameters['max_iterations_subproblem'] = 400
        if 'starting_parameter' not in TR_parameters:
            TR_parameters['starting_parameter'] = parameter_space.sample_randomly(1)[0]
        if 'safety_tolerance' not in TR_parameters:
            TR_parameters['safety_tolerance'] = 1e-16
        if 'max_iterations_armijo' not in TR_parameters:
            TR_parameters['max_iterations_armijo'] = 50
        if 'initial_step_armijo' not in TR_parameters:
            TR_parameters['initial_step_armijo'] = 0.5
        if 'armijo_alpha' not in TR_parameters:
            TR_parameters['armijo_alpha'] = 1.e-4
        if 'opt_method' not in TR_parameters:
            TR_parameters['opt_method'] = 'BFGSMethod'
        if 'epsilon_i' not in TR_parameters:
            TR_parameters['epsilon_i'] = 1e-8
        if 'store_estimator' not in TR_parameters:
            TR_parameters['store_estimator'] = False
        if 'beta' not in TR_parameters:
            TR_parameters['beta'] = 0.95
        mu_k = TR_parameters['starting_parameter']

        TR_parameters['full_order_model'] = False # tbc

    if extension_params is None:
        extension_params={'Enlarge_radius': True, 'opt_fom': None,
                          'store_subproblem_iterations': True, 'return_data_dict': True}
    else:
        if 'Enlarge_radius' not in extension_params:
            extension_params['Enlarge_radius'] = True
        if 'opt_fom' not in extension_params:
            extension_params['opt_fom'] = None
        if 'store_subproblem_iterations' not in extension_params:
            extension_params['store_subproblem_iterations'] = True
        if 'return_data_dict' not in extension_params:
            extension_params['return_data_dict'] = False

    if 'FOC_tolerance' not in TR_parameters:
        TR_parameters['FOC_tolerance'] = TR_parameters['sub_tolerance']

    print('starting parameter {}'.format(mu_k))

    if 'eps_TR' not in TR_parameters:
        generic_sequence_to_zero = [10**k for k in range(20, -10, -1)]
        for i in range(TR_parameters['max_iterations']-len(generic_sequence_to_zero)):
            generic_sequence_to_zero.append(0)
        TR_parameters['eps_TR'] = generic_sequence_to_zero
    if 'eps_cond' not in TR_parameters:
        TR_parameters['eps_cond'] = generic_sequence_to_zero

    if (TR_parameters['eps_TR'][0] > 100) and (TR_parameters['eps_cond'][0] > 100) and skip_estimator:
        assert reductor.reductor_type == 'non_assembled'

    ###### RELAXING RADIUS

    eps_TR_ks = TR_parameters['eps_TR']
    eps_cond_ks = TR_parameters['eps_cond']

    original_radius = TR_parameters['radius']
    TR_parameters['radius'] = eps_TR_ks[0] + original_radius

    # timings
    tic = time.time()
    Js = []
    FOCs = []
    times = []
    j_list = []
    times_est_evaluations = []
    J_estimator = []
    mu_est = []
    all_mus = []
    total_subproblem_time = 0

    k = 0
    mu_list = []
    mu_list.append(mu_k)
    JFE_list = []
    # the next part is tbc
    if TR_parameters['opt_method']!= 'BFGSMethod': #Keil et al. '20 method does not have this feature
        if 'JFE_start' in TR_parameters:
            JFE_list.append(TR_parameters['JFE_start'])
        else:
            print("******")
            print("!!!!!!!!!!!!!!!!Please, to improve the method, give me the starting value of the cost for FOM model "
                  "(which you have accessible due to the initialization of the ROM model)!!!!!!!!!!!")
            print("******")
    #----- up to here
    normgrad = 1e6
    point_rejected = False
    additional_criteria = 0
    model_has_been_enriched = False
    J_k = opt_rom.output_functional_hat(mu_k, pool=pool)
    print("Starting value of the cost: {}".format(J_k))
    if mesh_adaptive:
        gradient = opt_rom.output_functional_hat_gradient(mu_k)
        mu_box = opt_rom.primal_model.parameters.parse(mu_k.to_numpy() - gradient)
        first_order_criticity = mu_k.to_numpy() - projection_onto_range(parameter_space, mu_box).to_numpy()
        normgrad = np.linalg.norm(first_order_criticity)
        print("Starting First order critical condition: {}".format(normgrad))
    print("******************************* \n")
    while k < TR_parameters['max_iterations']:
        if k > 2 and Js[-1] == Js[-2]:
            print('\n THE METHOD IS STAGNATING, STOP')
            break
        if normgrad <= TR_parameters['FOC_tolerance']:
            print('\nStopping criteria fulfilled: FOC condition: {}'.format(normgrad))
            additional_criteria = 1
        if 'control_mu' in TR_parameters and additional_criteria:
            if TR_parameters['control_mu']:
                print(' ... additional mu control requested.')
                assert 0, 'not available in relaxed algorithm'
        if additional_criteria:
            break
        print(f'Relaxing parameter for TR: {eps_TR_ks[k]:.0e} and SC: {eps_cond_ks[k]:.0e}')
        if (TR_parameters['eps_TR'][0] > 100) and (TR_parameters['eps_cond'][0] > 100) and skip_estimator:
            print('skipping estimations entirely')
            no_estimation = True
        else:
            no_estimation = False
        
        tic_ = time.perf_counter()
        if TR_parameters['opt_method'] == "BFGSMethod":
            mu_kp1, Jcp, j, J_kp1, _, mus = solve_optimization_subproblem_BFGS(opt_rom, parameter_space, mu_k,
                                                                               TR_parameters, pool=pool,
                                                                               skip_estimator=skip_estimator)
        else:
            mu_kp1, Jcp, j, J_kp1, _, mus = solve_optimization_subproblem_NewtonMethod(opt_rom, parameter_space,
                                                                                       mu_k, TR_parameters,
                                                                                       skip_estimator=skip_estimator)

        print(f'sub-problem took {time.perf_counter()-tic_}')
        total_subproblem_time += time.perf_counter()-tic_
        # verifying whether TR is fulfilled
        u_rom = opt_rom.solve(mu_kp1)
        p_rom = opt_rom.solve_dual(mu_kp1, U=u_rom)
        if not no_estimation:
            estimate_J = opt_rom.estimate_output_functional_hat(u_rom, p_rom, mu_kp1)
            TR_criterion = abs(estimate_J / J_kp1)
        else:
            estimate_J = 0
            TR_criterion = np.inf

        if J_kp1 + estimate_J < Jcp + eps_cond_ks[k]:
            print('starting enrichment')
            if isinstance(reductor, QuadraticPdeoptStationaryCoerciveLODReductor):
                if (eps_cond_ks[k] < 100) or (eps_TR_ks[k] < 100):
                    # to be implemented ! 
                    reductor = reductor.with_(reductor_type='coercive')
                if not skip_estimator:
                    print('checking global termination before expensive local enrichment')
                    u = reductor.fom.optional_forward_model.solve(mu_kp1, pool=pool)
                    p = reductor.fom.solve_dual(mu_kp1, U=u, pool=pool)
                    gradient = reductor.fom.output_functional_hat_gradient(mu_kp1, U=u, P=p)
                    mu_box = opt_rom.primal_model.parameters.parse(mu_kp1.to_numpy() - gradient)
                    first_order_criticity = mu_kp1.to_numpy() - projection_onto_range(parameter_space, mu_box).to_numpy()
                    normgrad = np.linalg.norm(first_order_criticity)
                else:
                    normgrad = np.inf
                if normgrad <= TR_parameters['FOC_tolerance']:
                    pass
                else:
                    opt_rom, reductor, KmsijT, corT = enrichment_step(mu_kp1, reductor, pool=pool)
                    u = reductor.fom.solve(mu_kp1, KmsijT=KmsijT, correctorsListT=corT, pool=pool)
                    p = reductor.fom.solve_dual(mu_kp1, U=u, KmsijT=KmsijT, correctorsListT=corT, pool=pool)
            else:
                if (eps_cond_ks[k] < 100) or (eps_TR_ks[k] < 100):
                    reductor = reductor.with_(reductor_type='simple_coercive')
                opt_rom, reductor, out_1, out_2 = enrichment_step(mu_kp1, reductor, pool=pool)
                u, p = out_1, out_2

            model_has_been_enriched = True
            JFE_list.append(reductor.fom.output_functional_hat(mu_kp1, u))

            print("k: {} - j {} - Cost Functional: {} - mu: {}".format(k, j, J_kp1, mu_kp1))
            mu_list.append(mu_kp1)
            times.append(time.time() -tic)
            Js.append(J_kp1)
            all_mus.append(mus)
            mu_k = mu_kp1

        elif J_kp1 - estimate_J > Jcp + eps_cond_ks[k]:
            print('necessary condition failed')
            TR_parameters['radius'] = TR_parameters['radius'] * 0.5
            print(f"Shrinking the TR radius to: {TR_parameters['radius']} because Jcp {Jcp} and J_kp1 {J_kp1}")
            point_rejected = True
            if point_rejected and (TR_parameters['opt_method'] =="BFGS_Newton"
                                   or TR_parameters['opt_method'] =='AdaptiveTaylor_BFGS_Newton'):
                # TODO: have a look up here
                assert 0, "check this line"
                reductor = starting_reductor
                opt_rom = starting_rom

        else:
            print('enriching to check the sufficient decrease condition')
            if TR_parameters['opt_method'] == 'AdaptiveTaylor_Newton':
                new_rom, new_reductor, u, p = enrichment_step(mu_kp1, reductor, adaptive_taylor=adaptive_taylor)
            else:
                new_rom, new_reductor, out_1, out_2 = enrichment_step(mu_kp1, reductor, pool=pool)
                if isinstance(reductor, QuadraticPdeoptStationaryCoerciveLODReductor):
                    u = reductor.fom.optional_forward_model.solve(mu_kp1, KmsijT=out_1, correctorsListT=out_2)
                    p = reductor.fom.solve_dual(mu_kp1, U=u, KmsijT=out_1, correctorsListT=out_2)
                else:
                    u, p = out_1, out_2
            model_has_been_enriched = True
            JFE_list.append(reductor.fom.output_functional_hat(mu_kp1, u, p))


            J_kp1 = new_rom.output_functional_hat(mu_kp1)
            print("k: {} - j {} - Cost Functional: {} - mu: {}".format(k, j, J_kp1, mu_kp1))
            if J_kp1 > Jcp + 1e-8 + eps_cond_ks[k]:    # add a safety tolerance of 1e-8 for avoiding numerical stability effects
                TR_parameters['radius'] = TR_parameters['radius'] * 0.5
                print(
                    "Shrinking the TR radius to: {} because Jcp {} and J_kp1 {}".format(TR_parameters['radius'],
                                                                                        Jcp,
                                                                                        J_kp1))
                JFE_list.pop(-1) #We need to remove the value from the list, because we reject the parameter
                point_rejected = True

            else:
                opt_rom = new_rom
                reductor = new_reductor
                mu_list.append(mu_kp1)
                times.append(time.time() -tic)
                Js.append(J_kp1)
                J_estimator.append(estimate_J)
                mu_k = mu_kp1
                # if extension_params['Enlarge_radius']:
                #     if len(JFE_list) >= 2:
                #         if (JFE_list[-2] - JFE_list[-1]) / (J_k - J_kp1) > 0.75:
                #             if 'JFE_start' not in TR_parameters:  # This is need to have method of Paper_1
                #                 if k - 1 != 0:
                #                     TR_parameters['radius'] *= 2
                #                     print('enlarging the TR radius to {}'.format(TR_parameters['radius']))
                #             else:
                #                 TR_parameters['radius'] *= 2
                #                 print('enlarging the TR radius to {}'.format(TR_parameters['radius']))
                J_k = J_kp1


        if not point_rejected:
            if model_has_been_enriched:
                gradient = reductor.fom.output_functional_hat_gradient(mu_k, U=u, P=p)
                mu_box = opt_rom.primal_model.parameters.parse(mu_k.to_numpy() - gradient)
                first_order_criticity = mu_k.to_numpy() - projection_onto_range(parameter_space, mu_box).to_numpy()
                normgrad = np.linalg.norm(first_order_criticity)
                model_has_been_enriched = False
            else:
                assert 0

            FOCs.append(normgrad)
            j_list.append(j)

            print(f'TR criterion: {TR_criterion}')
            if J_kp1 < Jcp + 1e-8:   # add a safety tolerance of 1e-8 for avoiding numerical stability effectsa
                print('strong sufficient decrease condition fulfilled')
            else:
                print('strong sufficient decrease condition NOT fulfilled')
            print("First order critical condition: {}".format(normgrad))

            k = k + 1
            # update radius
            TR_parameters['radius'] = eps_TR_ks[k] + original_radius
        print("******************************* \n")


    if extension_params['Enlarge_radius']:
        Js = JFE_list  # This is for speeding-up the computation of the error for the TR method,
        # of course it does not apply to Qian-Grepl
        # This procedure does not give additional speed-up to our method,
        # but improves only the time of the post-processing step,
        # to have the plot of the error, which is not counted in the computational time of the method.

    if k >= TR_parameters['max_iterations']:
        print (" WARNING: Maximum number of iteration for the TR algorithm reached")

    data = {}
    data['all_mus'] = all_mus
    print(f'Sub-problems took {total_subproblem_time:.5f}s')
    if extension_params['timings']:
        data['total_subproblem_time'] = total_subproblem_time
        if TR_parameters['control_mu']:
            data['times_est_evaluations'] = times_est_evaluations
            data['mu_est'] = mu_est
        if return_opt_rom:
            data['opt_rom'] = opt_rom
        if TR_parameters['store_estimator']:
            data['J_estimator'] = J_estimator
        if extension_params['store_subproblem_iterations']:
            data['j_list'] = j_list
        if extension_params['return_data_dict']:
            return mu_list, times, Js, FOCs, data
        else:
            # this is only here to be compatible with the old state
            if return_opt_rom:
                return mu_list, times, Js, FOCs, opt_rom
            else:
                return mu_list, times, Js, FOCs

    return mu_list
