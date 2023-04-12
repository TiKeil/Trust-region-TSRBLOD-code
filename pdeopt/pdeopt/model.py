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
import time
import scipy
from numbers import Number

from pymor.models.basic import StationaryModel
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.operators.constructions import VectorOperator, LincombOperator
from pymor.operators.constructions import IdentityOperator
from pymor.operators.interface import Operator

class QuadraticPdeoptStationaryModel(StationaryModel):

    def __init__(self, primal_model, output_functional_dict, opt_product=None,
                 estimators=None, dual_model=None,
                 projected_hessian=None, name=None,
                 use_corrected_functional=True,
                 adjoint_approach=False, separated_bases=True,
                 optional_forward_model=None, fom=None, is_rom=False,
                 evaluation_counter=None,
                 coarse_projection=None,
                 fine_prolongation=None):
        super().__init__(primal_model.operator, primal_model.rhs, primal_model.output_functional,
                         primal_model.products, primal_model.error_estimator, primal_model.visualizer,  name)
        self.__auto_init(locals())
        if self.optional_forward_model is not None and hasattr(self.optional_forward_model, 'store_in_tmp'):
            self.store_in_tmp = self.optional_forward_model.store_in_tmp
        else:
            self.store_in_tmp = False
        if self.opt_product is None:
            self.opt_product = primal_model.h1_product
        if self.estimators is None:
            self.estimators = {'primal': None, 'dual': None, 'output_functional_hat': None,
                               'output_functional_hat_d_mus': None}
        if coarse_projection is None:
            self.coarse_projection = IdentityOperator(self.solution_space)

        if fine_prolongation is None:
            self.fine_prolongation = IdentityOperator(self.solution_space)

        self.hessian_parts = {}
        self.local_index_to_global_index = {}
        k = 0
        for (key, size) in sorted(self.primal_model.parameters.items()):
            array_ = np.empty(size, dtype=object)
            for l in range(size):
                array_[l] = k
                k += 1
            self.local_index_to_global_index[key] = array_
        self.number_of_parameters = k

    def solve(self, mu, **kwargs):
        if self.optional_forward_model is not None:
            if isinstance(self.optional_forward_model, StationaryModel):
                self.evaluation_counter.count(two_scale=True)
                return self.optional_forward_model.solve(mu)
            else:
                return self.optional_forward_model.solve(mu, **kwargs)
        else:
            if self.evaluation_counter:
                self.evaluation_counter.count(self.is_rom)
            return self.primal_model.solve(mu)

    def solve_dual(self, mu, U=None, **kwargs):
        if self.evaluation_counter and not self.optional_forward_model:
            self.evaluation_counter.count(self.is_rom)
        if U is None:
            U = self.solve(mu, **kwargs)
            if isinstance(U, tuple):
                U = U[0] #<-- for the verbose case
        if self.dual_model is not None:
            mu = self._add_primal_to_parameter(mu, U)
            if self.optional_forward_model:
                assert self.evaluation_counter
                self.evaluation_counter.count(two_scale=True)
                if isinstance(self.dual_model, StationaryModel):
                    return self.dual_model.solve(mu)
                else:
                    return self.dual_model.solve(mu, **kwargs)
            return self.dual_model.solve(mu)
        else:
            if self.optional_forward_model:
                from pdeopt.gridlod_model import FEMGridlodModel, GridlodModel
                if isinstance(self.optional_forward_model, FEMGridlodModel):
                    dual_model = self._build_dual_model(U, mu)
                    F = dual_model.rhs.as_range_array(mu).to_numpy()[0]
                elif isinstance(self.optional_forward_model, GridlodModel):
                    coarse_dual_model_rhs = self._build_coarse_dual_model(U, mu)
                    F = coarse_dual_model_rhs.as_range_array(mu).to_numpy()[0]
                else:
                    assert 0, "not implemented"
                return self.optional_forward_model.solve(mu, F=F, **kwargs)
            if not isinstance(self.fine_prolongation, IdentityOperator):
                dual_fom = self._build_dual_model(U, mu, fine_coarse_fine=True)
            else:
                dual_fom = self._build_dual_model(U, mu)
            return dual_fom.solve(mu)

    def solve_for_u_d_mu(self, component, index, mu, U=None):
        if U is None:
            U = self.solve(mu)
        residual_dmu_lhs = self.primal_model.operator.d_mu(component, index).apply(U, mu=mu)
        residual_dmu_rhs = self.primal_model.rhs.d_mu(component, index).as_range_array(mu)
        rhs_operator = residual_dmu_rhs-residual_dmu_lhs
        u_d_mu = self.primal_model.operator.apply_inverse(rhs_operator, mu=mu)
        return u_d_mu

    def solve_for_u_d_eta(self, mu, eta, U=None):
        if not sum(eta):
            return self.solution_space.zeros()
        # differentiate in arbitrary direction
        if U is None:
            U = self.solve(mu)
        new_rhs = self.primal_model.rhs.range.zeros()
        k = 0
        for (key, size) in sorted(self.primal_model.parameters.items()):
            for l in range(size):
                if eta[k]:
                    new_rhs -= self.primal_model.operator.d_mu(key, l).apply(U, mu=mu) * eta[k]
                    new_rhs += self.primal_model.rhs.d_mu(key, l).as_range_array(mu) * eta[k]
                k +=1
        u_d_mu = self.primal_model.operator.apply_inverse(new_rhs, mu=mu)
        return u_d_mu

    def solve_for_p_d_mu(self, component, index, mu, U=None, P=None, u_d_mu=None):
        if U is None:
            U = self.solve(mu)
        if P is None:
            P = self.solve_dual(mu, U)
        if u_d_mu is None:
            u_d_mu = self.solve_for_u_d_mu(component, index, mu=mu, U=U)
        if self.dual_model is not None:
            mu = self._add_primal_to_parameter(mu, U)
            residual_dmu_lhs = self.dual_model.operator.d_mu(component, index).apply(P, mu=mu)
            residual_dmu_rhs = self.dual_model.rhs.d_mu(component, index).as_range_array(mu)
            k_term = self.output_functional_dict['dual_projected_d_u_bilinear_part'].apply_adjoint(u_d_mu, mu)
        else:
            dual_fom = self._build_dual_model(U, mu, forward_model=False)
            residual_dmu_lhs = dual_fom.operator.d_mu(component, index).apply(P, mu=mu)
            residual_dmu_rhs = dual_fom.rhs.d_mu(component, index).as_range_array(mu)
            k_term = self.output_functional_dict['d_u_bilinear_part'].apply(u_d_mu, mu)
        rhs_operator = residual_dmu_rhs-residual_dmu_lhs+k_term
        if self.dual_model is not None:
            p_d_mu = self.dual_model.operator.apply_inverse(rhs_operator, mu=mu)
        else:
            p_d_mu = self.primal_model.operator.apply_inverse(rhs_operator, mu=mu)
        return p_d_mu

    def solve_for_p_d_eta(self, mu, eta, U=None, P=None, u_d_eta=None):
        if not sum(eta):
            return self.solution_space.zeros()
        if U is None:
            U = self.solve(mu)
        if P is None:
            P = self.solve_dual(mu, U)
        if self.dual_model is not None:
            mu = self._add_primal_to_parameter(mu, U)
        if self.dual_model is None:
            dual_model = self._build_dual_model(U, mu, forward_model=False)
            new_rhs = self.primal_model.rhs.range.zeros()
        else:
            dual_model = self.dual_model
            new_rhs = self.dual_model.rhs.range.zeros()
        if u_d_eta is None:
            u_d_eta = self.solve_for_u_d_eta(mu=mu, eta=eta, U=U)
        k = 0
        for (key, size) in sorted(self.primal_model.parameters.items()):
            for l in range(size):
                if eta[k]:
                    new_rhs -= dual_model.operator.d_mu(key, l).apply(P, mu=mu) * eta[k]
                    new_rhs += dual_model.rhs.d_mu(key, l).as_range_array(mu) * eta[k]
                k +=1
        if self.dual_model is not None:
            k_term = self.output_functional_dict['dual_projected_d_u_bilinear_part'].apply_adjoint(u_d_eta, mu)
        else:
            k_term = self.output_functional_dict['d_u_bilinear_part'].apply(u_d_eta, mu)
        new_rhs += k_term
        p_d_eta = dual_model.operator.apply_inverse(new_rhs, mu=mu)
        return p_d_eta

    def solve_auxiliary_dual_problem(self, mu, U=None):
        assert self.dual_model is not None, 'this is only a ROM method'
        if U is None:
            U = self.solve(mu)
        mu = self._add_primal_to_parameter(mu, U)
        rhs_operator_1 = self.output_functional_dict['dual_primal_projected_op'].apply(U, mu=mu)
        rhs_operator_2 = self.output_functional_dict['dual_projected_rhs'].as_range_array(mu)

        rhs_operator = rhs_operator_1 - rhs_operator_2
        Z = self.dual_model.operator.apply_inverse(rhs_operator, mu=mu)
        return Z

    def solve_auxiliary_primal_problem(self, mu, Z=None, U=None, P=None):
        assert self.dual_model is not None, 'this is only a ROM method'
        if U is None:
            U = self.solve(mu)
        if P is None:
            P = self.solve_dual(mu, U)
        if Z is None:
            Z = self.solve_auxiliary_dual_problem(mu, U)
        mu = self._add_primal_to_parameter(mu, U)
        mu = self._add_dual_to_parameter(mu, P)

        rhs_operator_1 = self.output_functional_dict['dual_primal_projected_op'].apply_adjoint(P, mu=mu)
        rhs_operator_2 = self.output_functional_dict['primal_projected_dual_rhs'].as_range_array(mu)
        rhs_operator_3 = self.output_functional_dict['dual_projected_d_u_bilinear_part'].apply_adjoint(Z, mu=mu)

        rhs_operator = rhs_operator_2 - rhs_operator_1 - rhs_operator_3
        W = self.primal_model.operator.apply_inverse(rhs_operator, mu=mu)
        return W

    def output_functional_hat(self, mu, U=None, P=None, **kwargs):
        if U is None:
            U = self.solve(mu=mu, **kwargs)
        constant_part = self.output_functional_dict['output_coefficient']
        linear_part = self.output_functional_dict['linear_part'].apply_adjoint(self.coarse_projection.apply(U), mu).to_numpy()[0,0]
        bilinear_part = self.output_functional_dict['bilinear_part'].apply2(U, U, mu)[0,0]
        correction_term = 0
        if self.use_corrected_functional and self.dual_model is not None:
            if P is None:
               P = self.solve_dual(mu=mu, U=U)
            residual_lhs = self.output_functional_dict['dual_primal_projected_op'].apply2(P, U, mu=mu)[0, 0]
            residual_rhs = self.output_functional_dict['dual_projected_rhs'].apply_adjoint(P, mu=mu).to_numpy()[0, 0]
            correction_term = residual_rhs - residual_lhs
        return constant_part(mu) + linear_part + bilinear_part + correction_term

    def corrected_output_functional_hat(self, mu, u=None, p=None, **kwargs):
        if u is None:
            u = self.solve(mu=mu, **kwargs)
        if p is None:
            p = self.solve_dual(mu=mu, U=u, **kwargs)
        constant_part = self.output_functional_dict['output_coefficient']
        linear_part = self.output_functional_dict['linear_part'].apply_adjoint(u, mu).to_numpy()[0,0]
        bilinear_part = self.output_functional_dict['bilinear_part'].apply2(u, u, mu)[0,0]
        if self.dual_model is not None:
            residual_lhs = self.output_functional_dict['dual_primal_projected_op'].apply2(p, u, mu=mu)[0,0]
            residual_rhs = self.output_functional_dict['dual_projected_rhs'].apply_adjoint(p, mu=mu).to_numpy()[0,0]
        else:
            residual_lhs = self.primal_model.operator.apply2(u, p, mu=mu)[0,0]
            residual_rhs = self.primal_model.rhs.apply_adjoint(p, mu=mu).to_numpy()[0,0]
        correction_term = residual_rhs - residual_lhs
        return constant_part(mu) + linear_part + bilinear_part + correction_term

    def output_functional_hat_d_mu(self, component, index, mu, U=None, P=None, Z=None, W=None, **kwargs):
        if self.dual_model is not None:
            if self.adjoint_approach:
                return self.adjoint_corrected_output_functional_hat_d_mu(component, index, mu, U, P, Z, W, **kwargs)
        return self.uncorrected_output_functional_hat_d_mu(component, index, mu, U, P)

    def uncorrected_output_functional_hat_d_mu(self, component, index, mu, U=None, P=None, **kwargs):
        if U is None:
            U = self.solve(mu=mu, **kwargs)
        if P is None:
            P = self.solve_dual(mu=mu, U=U, **kwargs)
        output_coefficient = self.output_functional_dict['output_coefficient']
        J_dmu = output_coefficient.d_mu(component, index).evaluate(mu)
        if self.dual_model is not None:  # This is a cheat for detecting if it's a rom
            projected_op = self.output_functional_dict['dual_primal_projected_op']
            projected_rhs = self.output_functional_dict['dual_projected_rhs']
            residual_dmu_lhs = projected_op.d_mu(component, index).apply2(P, U, mu=mu)
            residual_dmu_rhs = projected_rhs.d_mu(component, index).apply_adjoint(P, mu=mu).to_numpy()[0,0]
        else:
            residual_dmu_lhs = self.primal_model.operator.d_mu(component, index).apply2(P, U, mu=mu)
            residual_dmu_rhs = self.primal_model.rhs.d_mu(component, index).apply_adjoint(P, mu=mu).to_numpy()[0,0]
        return (J_dmu - residual_dmu_lhs + residual_dmu_rhs)[0,0]

    def adjoint_corrected_output_functional_hat_d_mu(self, component, index, mu, U=None, P=None,
                                                     Z=None, W=None, **kwargs):
        assert self.dual_model is not None, 'not a FOM method'
        if U is None:
            U = self.solve(mu=mu, **kwargs)
        if P is None:
            P = self.solve_dual(mu=mu, U=U, **kwargs)
        if Z is None:
            Z = self.solve_auxiliary_dual_problem(mu, U=U)
        if W is None:
            W = self.solve_auxiliary_primal_problem(mu, Z=Z, U=U, P=P)

        output_coefficient = self.output_functional_dict['output_coefficient']
        J_dmu = output_coefficient.d_mu(component, index).evaluate(mu)
        projected_op = self.output_functional_dict['dual_primal_projected_op']
        projected_rhs = self.output_functional_dict['dual_projected_rhs']
        residual_dmu_lhs = projected_op.d_mu(component, index).apply2(P, U, mu=mu)
        residual_dmu_rhs = projected_rhs.d_mu(component, index).apply_adjoint(P, mu=mu).to_numpy()[0,0]
        # auxilialy problems
        mu = self._add_primal_to_parameter(mu, U)
        mu = self._add_dual_to_parameter(mu, P)

        w_term_1 = self.primal_model.operator.d_mu(component, index).apply2(W, U, mu=mu)
        w_term_2 = self.primal_model.rhs.d_mu(component, index).apply_adjoint(W, mu=mu).to_numpy()[0,0]
        w_term = w_term_2 - w_term_1
        z_term_1 = self.dual_model.operator.d_mu(component, index).apply2(Z, P, mu=mu)
        z_term_2 = self.dual_model.rhs.d_mu(component, index).apply_adjoint(Z, mu=mu).to_numpy()[0,0]
        z_term = z_term_1 - z_term_2

        return (J_dmu - residual_dmu_lhs + residual_dmu_rhs + w_term + z_term)[0,0]

    def output_functional_hat_gradient(self, mu, adjoint_approach=None, U=None, P=None, fom=False, **kwargs):
        if fom and self.optional_forward_model is not None:
            assert 0
            print('Using FEM model instead of LOD model for the gradient')
            # do not use optional_forward_model to get the best gradient information possible
            U = self.primal_model.solve(mu, **kwargs)
            dual_fom = self._build_dual_model(U, mu, forward_model=False)
            P = dual_fom.solve(mu, **kwargs)
        if adjoint_approach is None:
            if self.dual_model is not None:
                adjoint_approach = self.adjoint_approach
        gradient = []
        if U is None:
            U = self.solve(mu=mu, **kwargs)
        if P is None:
            P = self.solve_dual(mu=mu, U=U, **kwargs)
        if adjoint_approach:
            Z = self.solve_auxiliary_dual_problem(mu, U=U)
            W = self.solve_auxiliary_primal_problem(mu, Z=Z, U=U, P=P)
        for (key, size) in sorted(self.primal_model.parameters.items()):
            for l in range(size):
                if adjoint_approach:
                    gradient.append(self.adjoint_corrected_output_functional_hat_d_mu(key, l, mu, U, P, Z, W))
                else:
                    gradient.append(self.output_functional_hat_d_mu(key, l, mu, U, P))
        gradient = np.array(gradient)
        return gradient

    def output_functional_hat_gradient_adjoint(self, mu, **kwargs):
        return self.output_functional_hat_gradient(mu, adjoint_approach=True, **kwargs)

    def output_functional_hessian_operator(self, mu, eta, U=None, P=None, printing=False):
        if self.dual_model is not None:
            if self.adjoint_approach:
                assert self.projected_hessian is not None, 'you must set prepare_for_hessian=True in the reductor'
                return self.adjoint_corrected_output_functional_hessian_operator(mu, eta, U=U, P=P, printing=printing)
        return self.uncorrected_output_functional_hessian_operator(mu, eta, U=U, P=P, printing=printing)

    def extract_hessian_parts(self, mu, U=None, P=None, extract_sensitivities=True):
        mu_tuple = tuple(mu.to_numpy())
        if mu_tuple not in self.hessian_parts:
            output_coefficient = self.output_functional_dict['output_coefficient']
            parts_dict = {}
            if U is None:
                U = self.solve(mu=mu)
            if P is None:
                P = self.solve_dual(mu=mu, U=U)
            U_d_mu_dict = {}
            P_d_mu_dict = {}
            k = 0
            if extract_sensitivities:
                for (key, size) in sorted(self.primal_model.parameters.items()):
                    U_d_mu = np.empty(size, dtype=object)
                    P_d_mu = np.empty(size, dtype=object)
                    for l in range(size):
                        U_d_mu_ = self.solve_for_u_d_mu(key, l, mu, U)
                        P_d_mu_ = self.solve_for_p_d_mu(key, l, mu, U, P, U_d_mu_)
                        U_d_mu[l] = U_d_mu_
                        P_d_mu[l] = P_d_mu_
                    U_d_mu_dict[key] = U_d_mu
                    P_d_mu_dict[key] = P_d_mu
                gradient_operator_1, gradient_operator_2 = [], []
                gradient_rhs, J_vector  = [], []
                for (key, size) in sorted(self.primal_model.parameters.items()):
                    for l in range(size):
                        J_vector.append(output_coefficient.d_mu(key, l).d_mu(key, l).evaluate(mu))
                        go_1, go_2, rhs = [], [], []
                        k_ = 0
                        proj_ops = self.projected_hessian[key][l] # be careful ! this is key and not key_
                        for (key_, size_) in sorted(self.primal_model.parameters.items()):
                            for l_ in range(size_):
                                go_1.append(proj_ops['PS_D_op'][key_][l_].apply2(U_d_mu_dict[key_][l_],P,mu=mu)[0,0])
                                go_2.append(proj_ops['P_DS_op'][key_][l_].apply2(U, P_d_mu_dict[key_][l_],mu=mu)[0,0])
                                rhs.append(proj_ops['DS_rhs'][key_][l_].apply_adjoint(
                                    P_d_mu_dict[key_][l_], mu=mu).to_numpy()[0,0])
                                k_ += 1
                        gradient_operator_1.append(go_1)
                        gradient_operator_2.append(go_2)
                        gradient_rhs.append(rhs)
                        k +=1
                gradient_vector = []
                for l in range(k):
                    gradient_vector.append([gradient_operator_1[l]] + [gradient_operator_2[l]] + [gradient_rhs[l]])
                    gradient_vector[-1] = np.einsum('jk -> k', gradient_vector[-1])
                second_gradient_vector = [J_vector]
                second_gradient_vector = np.einsum('jk -> k', second_gradient_vector)
                parts_dict['gradient_vector'] = gradient_vector
                parts_dict['second_gradient_vector'] = second_gradient_vector
            else:
                parts_dict['U'], parts_dict['P'] = U, P
            self.hessian_parts[mu_tuple] = parts_dict
        else:
            parts_dict = self.hessian_parts[mu_tuple]
        if extract_sensitivities:
            gradient_vector = parts_dict['gradient_vector']
            second_gradient_vector = parts_dict['second_gradient_vector']
        else:
            U, P = parts_dict['U'], parts_dict['P']
        if extract_sensitivities:
            return gradient_vector, second_gradient_vector
        else:
            return U, P

    def uncorrected_output_functional_hessian_operator(self, mu, eta, U=None, P=None, printing=False):
        output_coefficient = self.output_functional_dict['output_coefficient']
        gradient_operator_1, gradient_operator_2 = [], []
        gradient_rhs, J_vector = [], []
        k = 0
        if printing:
            print('this is my current mu {}'.format(mu))
            print('this is my current eta {}'.format(eta))

        U, P = self.extract_hessian_parts(mu, U=U, P=P, extract_sensitivities=False)
        U_d_eta = self.solve_for_u_d_eta(mu, eta, U)
        P_d_eta = self.solve_for_p_d_eta(mu, eta, U, P, U_d_eta)

        for (key, size) in sorted(self.primal_model.parameters.items()):
            for l in range(size):
                if self.dual_model is not None:
                    projected_hessian = self.projected_hessian[key][l]
                    gradient_rhs.append(projected_hessian['D_rhs'].apply_adjoint(P_d_eta, mu=mu).to_numpy()[0,0])
                    gradient_operator_1.append(projected_hessian['D_P_op'].apply2(P, U_d_eta, mu=mu)[0,0])
                    gradient_operator_2.append(projected_hessian['D_P_op'].apply2(P_d_eta, U, mu=mu)[0,0])
                else:
                    gradient_rhs.append(self.primal_model.rhs.d_mu(key, l).apply_adjoint(
                        P_d_eta, mu=mu).to_numpy()[0,0])
                    gradient_operator_1.append(self.primal_model.operator.d_mu(key, l).apply2(P, U_d_eta, mu=mu)[0,0])
                    gradient_operator_2.append(self.primal_model.operator.d_mu(key, l).apply2(P_d_eta, U, mu=mu)[0,0])

                J_vector.append(output_coefficient.d_mu(key, l).d_mu(key, l).evaluate(mu)* eta[k])
                k +=1
        gradient_operator_1 = np.array(gradient_operator_1)
        gradient_operator_2 = np.array(gradient_operator_2)
        gradient_rhs = np.array(gradient_rhs)
        J_vector = np.array(J_vector)
        hessian_application = gradient_rhs - gradient_operator_1 - gradient_operator_2 + J_vector
        return hessian_application

    def estimate_error(self, U, mu):
        estimator = self.estimators['primal']
        if estimator is not None:
            if self.store_in_tmp:
                return estimator.estimate_error(U, mu=mu, store_in_tmp=self.store_in_tmp)
            else:
                return estimator.estimate_error(U, mu=mu)
        else:
            raise NotImplementedError('Model has no primal estimator.')

    def estimate_dual(self, U, P, mu):
        estimator = self.estimators['dual']
        if estimator is not None:
            if self.dual_model is not None:
                mu = self._add_primal_to_parameter(mu, U)
            if self.store_in_tmp:
                return estimator.estimate_error(U, P, mu=mu, store_in_tmp=self.store_in_tmp)
            else:
                return estimator.estimate_error(U, P, mu=mu)
        else:
            raise NotImplementedError('Model has no estimator for the dual problem.')

    def estimate_output_functional_hat(self, U, P, mu):
        estimator = self.estimators['output_functional_hat']
        if estimator is not None:
            if self.dual_model is not None:
                mu = self._add_primal_to_parameter(mu, U)
            if self.store_in_tmp:
                est = estimator.estimate_error(U, P, mu=mu, store_in_tmp=self.store_in_tmp)
            else:
                est = estimator.estimate_error(U, P, mu=mu)
            return est
        else:
            raise NotImplementedError('Model has no estimator for the output functional hat.')

    def estimate_output_functional_hat_d_mu(self, component, index, U, P, mu,
            U_d_mu=None, P_d_mu=None):
        assert not self.use_corrected_functional, "only the non corrected estimate is available in this publication"
        estimator = self.estimators['output_functional_hat_d_mus'][component][index]
        if estimator is not None:
            if self._check_input(component, index):
                if self.dual_model is not None:
                    mu = self._add_primal_to_parameter(mu, U)
                U_d_mu = self.solve_for_u_d_mu(component=component, index=index, mu=mu, U=U)
                P_d_mu = self.solve_for_p_d_mu(component=component, index=index, mu=mu, U=U, u_d_mu=U_d_mu, P=P)
                return estimator.estimate_error(U, P, mu=mu, U_d_mu=U_d_mu, P_d_mu=P_d_mu)
            else:
                return 0
        else:
            raise NotImplementedError('Model has no estimator for d_mu of the output functional hat.' +
                                      '\n If you need it, set prepare_for_gradient_estimate=True in the reductor')

    def estimate_output_functional_hat_gradient_norm(self, mu, U=None, P=None):
        gradient = []
        if U is None:
            U = self.solve(mu)
        if P is None:
            P = self.solve_dual(mu, U)
        for (key, size) in sorted(self.primal_model.parameters.items()):
            for l in range(size):
                gradient.append(self.estimate_output_functional_hat_d_mu(key, l, U, P, mu))
        gradient = np.array(gradient)
        return np.linalg.norm(gradient)

    def compute_coercivity(self, operator, mu, product):
        A = operator.assemble(mu).matrix
        K = product.matrix
        # source: A Tutorial on RB-Methods
        # see documentation of shift invert mode for smallest eigenvalue
        return scipy.sparse.linalg.eigsh(A, k=1, M=K, sigma=0, which='LM', return_eigenvectors=False)

    def compute_continuity_bilinear(self, operator, product, mu=None):
        # exclude zero case:
        if isinstance(operator, LincombOperator):
            if np.sum(operator.evaluate_coefficients(mu)) == 0:
                return 0
        elif not isinstance(operator, Operator):
            return 1
        A = operator.assemble(mu).matrix
        K = product.assemble(mu).matrix
        return scipy.sparse.linalg.eigsh(A, k=1, M=K, which='LM', return_eigenvectors=False)[0]

    def compute_continuity_linear(self, operator, product, mu=None):
        riesz_rep = product.apply_inverse(operator.as_vector(mu))
        output = np.sqrt(product.apply2(riesz_rep, riesz_rep))[0,0]
        return output

    def _build_dual_model(self, U, mu=None, coarse_model=False, coarse_input=False, fine_coarse_fine=False):
        suffix = '' if not coarse_model else '_coarse'
        ts_suffix = '' if not coarse_input else '_full'
        if fine_coarse_fine:
            suffix = '_coarse'
            ts_suffix = '_fine'
        if isinstance(self.output_functional_dict[f'd_u_linear_part{suffix}{ts_suffix}'], LincombOperator):
            operators = list(self.output_functional_dict[f'd_u_linear_part{suffix}{ts_suffix}'].operators)
            coefficients = list(self.output_functional_dict[f'd_u_linear_part{suffix}{ts_suffix}'].coefficients)
        else:
            operators = [self.output_functional_dict[f'd_u_linear_part{suffix}{ts_suffix}']]
            coefficients = [1]
        operators = [op.with_(name=op.name + '_linear_part') for op in operators]
        if isinstance(self.output_functional_dict[f'd_u_bilinear_part{suffix}{ts_suffix}'], LincombOperator):
            operators.extend([VectorOperator(op.apply(U)) for op in
                          self.output_functional_dict[f'd_u_bilinear_part{suffix}{ts_suffix}'].operators])
            coefficients.extend(self.output_functional_dict[f'd_u_bilinear_part{suffix}{ts_suffix}'].coefficients)
        else:
            operators.append(VectorOperator(self.output_functional_dict[f'd_u_bilinear_part{suffix}{ts_suffix}'].apply(U, mu)))
            coefficients.append(1)
        if coarse_model or (self.optional_forward_model is not None and not self.optional_forward_model.use_fine_mesh):
            dual_rhs_operator = LincombOperator(operators, coefficients)
            return dual_rhs_operator
        else:
            dual_rhs_operator = LincombOperator(operators, coefficients)
            return self.primal_model.with_(rhs = dual_rhs_operator)

    def _build_coarse_dual_model(self, U, mu=None):
        if self.optional_forward_model.use_fine_mesh:
            return self._build_dual_model(U, mu, coarse_model=True)
        else:
            return self._build_dual_model(U, mu)

    def _add_primal_to_parameter(self, mu, U):
        assert mu is not None
        return mu.with_(basis_coefficients=U.to_numpy()[0])

    def _add_dual_to_parameter(self, mu, P):
        assert mu is not None
        return mu.with_(basis_coefficients_dual=P.to_numpy()[0])

    def _add_eta_to_parameter(self, mu, eta):
        assert mu is not None
        return mu.with_(eta=eta)

    def _check_input(self, component, index):
        # check whether component is in parameter_type
        if component not in self.parameters:
            return False
        # check whether index is a number
        assert isinstance(index, Number)
        return True

    def extract_eta_from_component(self, component, index):
        eta = np.zeros(self.number_of_parameters)
        eta[self.local_index_to_global_index[component][index]] = 1
        return eta

def build_initial_basis(opt_fom, mus=None, build_sensitivities=False):
    primal_basis = opt_fom.solution_space.empty()
    dual_basis = opt_fom.solution_space.empty()
    if build_sensitivities:
        primal_sens_basis = {}
        dual_sens_basis = {}
        for (key, size) in opt_fom.parameters.items():
            sens_pr = np.empty(size, dtype=object)
            sens_du = np.empty(size, dtype=object)
            for l in range(size):
                sens_pr_ = opt_fom.solution_space.empty()
                sens_du_ = opt_fom.solution_space.empty()
                sens_pr[l] = sens_pr_
                sens_du[l] = sens_du_
            primal_sens_basis[key] = sens_pr
            dual_sens_basis[key] = sens_du
    for (i, mu) in enumerate(mus):
        dont_debug = 1 # set 0 for debugging
        u = opt_fom.solve(mu)
        primal_basis.append(u)
        if i != 1 or dont_debug: #< -- for debuginng
            dual_basis.append(opt_fom.solve_dual(mu, U=u))
        dual_basis = gram_schmidt(dual_basis, product=opt_fom.opt_product)
        primal_basis = gram_schmidt(primal_basis, product=opt_fom.opt_product)
        if build_sensitivities:
            for (key, size) in opt_fom.parameters.items():
                for l in range(size):
                    if key == 'biot' or dont_debug: #< -- for debuginng
                        if i != 2 and i!=3 or dont_debug: #< -- for debuginng
                            primal_sens_basis[key][l].append(opt_fom.solve_for_u_d_mu(key, l, mu))
                    else: #< -- for debuginng
                        if i!=3 or dont_debug: #< -- for debuginng
                            primal_sens_basis[key][l].append(opt_fom.solve_for_u_d_mu(key, l, mu))
                    dual_sens_basis[key][l].append(opt_fom.solve_for_p_d_mu(key, l, mu))
                    primal_sens_basis[key][l] = gram_schmidt(primal_sens_basis[key][l], product=opt_fom.opt_product)
                    dual_sens_basis[key][l] = gram_schmidt(dual_sens_basis[key][l], product=opt_fom.opt_product)
    if build_sensitivities:
        return primal_basis, dual_basis, primal_sens_basis, dual_sens_basis
    else:
        return primal_basis, dual_basis
