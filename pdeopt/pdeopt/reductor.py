# ~~~
# This file is part of the paper:
#
#           "tba"
#
#   https://github.com/TiKeil/tba
#
# Copyright 2019-2021 all developers. All rights reserved.
# License: Licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Authors:
#   Luca Mechelli (2019 - 2020)
#   Tim Keil      (2019 - 2021)
# ~~~

import numpy as np
import time

from pymor.core.base import ImmutableObject
from pymor.algorithms.projection import project
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.operators.constructions import VectorOperator, LincombOperator
from pymor.reductors.basic import ProjectionBasedReductor
from pymor.reductors.coercive import CoerciveRBReductor, SimpleCoerciveRBReductor
from pymor.reductors.basic import StationaryRBReductor
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.parameters.functionals import BaseMaxThetaParameterFunctional
from pymor.parameters.functionals import MaxThetaParameterFunctional
from pymor.operators.constructions import IdentityOperator


class QuadraticPdeoptStationaryCoerciveReductor(CoerciveRBReductor):
    def __init__(self, fom, RBPrimal=None, RBDual=None,
                 opt_product=None, coercivity_estimator=None,
                 check_orthonormality=None, check_tol=None, unique_basis=False,
                 reductor_type='simple_coercive', mu_bar=None,
                 prepare_for_hessian=False,
                 prepare_for_gradient_estimate=False, adjoint_estimate=False):
        assert not (prepare_for_hessian or prepare_for_gradient_estimate), 'not part of this publication'
        self.__auto_init(locals())
        if self.opt_product is None:
            self.opt_product = fom.opt_product
        super().__init__(fom, RBPrimal, product=opt_product, check_orthonormality=check_orthonormality,
                         check_tol=check_tol, coercivity_estimator=coercivity_estimator)

        self.non_assembled_primal_reductor = None
        self.non_assembled_primal_rom = None
        self.non_assembled_dual_rom = None

        self.adjoint_approach = self.fom.adjoint_approach
        self.separated_bases = False if self.unique_basis else True

        if unique_basis is True:
            self._build_unique_basis()
            self.bases = {'RB' : self.RBPrimal}
            print('Starting with only one basis with length {}'.format(len(self.RBPrimal)))
        else:
            self.bases = {'RB' : RBPrimal, 'DU': RBDual}
            print('Starting with two bases. ', end='')
            print('Primal and dual have length {} and {}'.format(
                len(RBPrimal), len(RBDual))) if RBPrimal is not None and RBDual is not None else print(
                'The Primal and/or the dual bases are empty')

        # primal model
        self.primal_fom = self.fom.primal_model
        self.primal_rom, self.primal_reductor = self._build_primal_rom()
        self.primal = self.primal_reductor

        # dual model
        if self.RBPrimal is not None:
            self.dual_intermediate_fom, self.dual_rom, self.dual_reductor = self._build_dual_models()
            self.dual = self.dual_reductor

        # pre compute constants for estimators
        try:
            k_form = self.fom.output_functional_dict['bilinear_part']
            if isinstance(k_form, LincombOperator):
                alpha_mu_bar = self.fom.compute_continuity_bilinear(k_form, self.fom.opt_product, mu_bar)
                self.cont_k = MaxThetaParameterFunctional(k_form.coefficients, mu_bar, gamma_mu_bar=alpha_mu_bar)
            else:
                self.cont_k = lambda mu: self.fom.compute_continuity_bilinear(k_form, self.fom.opt_product)
        except:
            self.cont_k = None

        if self.cont_k is None:
            try:
                k_form = self.fom.output_functional_dict['bilinear_part_coarse_full']
                coarse_product = self.fom.output_functional_dict['coarse_opt_product']
                if isinstance(k_form, LincombOperator):
                    alpha_mu_bar = self.fom.compute_continuity_bilinear(k_form, coarse_product, mu_bar)
                    self.cont_k = MaxThetaParameterFunctional(k_form.coefficients, mu_bar, gamma_mu_bar=alpha_mu_bar)
                else:
                    self.cont_k = lambda mu: self.fom.compute_continuity_bilinear(k_form, coarse_product)
            except:
                self.cont_k = None

        try:
            j_form = self.fom.output_functional_dict['linear_part']
            if isinstance(j_form, LincombOperator):
                conts_j = []
                for op in j_form.operators:
                    conts_j.append(self.fom.compute_continuity_linear(op, self.fom.opt_product))
                self.cont_j = lambda mu: np.dot(conts_j,np.abs(j_form.evaluate_coefficients(mu)))
            else:
                self.cont_j = lambda mu: self.fom.compute_continuity_linear(j_form, self.fom.opt_product)
        except:
            self.cont_j = None

        if self.cont_j is not None:
            try:
                j_form = self.fom.output_functional_dict['linear_part_coarse_full']
                coarse_product = self.fom.output_functional_dict['coarse_opt_product']
                if isinstance(j_form, LincombOperator):
                    conts_j = []
                    for op in j_form.operators:
                        conts_j.append(self.fom.compute_continuity_linear(op, coarse_product))
                    self.cont_j = lambda mu: np.dot(conts_j, np.abs(j_form.evaluate_coefficients(mu)))
                else:
                    self.cont_j = lambda mu: self.fom.compute_continuity_linear(j_form, coarse_product)
            except:
                self.cont_j = None

        if self.coercivity_estimator is None:
            print('WARNING: coercivity_estimator is None ... setting it to constant 1.')
            self.coercivity_estimator = lambda mu: 1.

        self.cont_a = MaxThetaParameterFunctional(self.primal_fom.operator.coefficients, mu_bar)
        self.time_for_enrichment = 0

    def reduce(self):
        assert self.RBPrimal is not None, 'I can not reduce without a RB basis'
        return super().reduce()

    def _reduce(self):
        # ensure that no logging output is generated for error_estimator assembly in case there is
        # no error estimator to assemble
        if self.assemble_error_estimator.__func__ is not ProjectionBasedReductor.assemble_error_estimator:
            with self.logger.block('Assembling error estimator ...'):
                error_estimator = self.assemble_error_estimator()
        else:
            error_estimator = None

        with self.logger.block('Building ROM ...'):
            rom = self.build_rom(error_estimator)
            rom = rom.with_(name=f'{self.fom.name}_reduced')
            rom.disable_logging()

        return rom

    def build_rom(self, estimator):
        if (not self.fom.adjoint_approach or not self.separated_bases) and self.prepare_for_hessian:
            projected_hessian = self.project_hessian()
        elif self.fom.adjoint_approach and self.prepare_for_hessian and self.separated_bases:
            projected_hessian = self.project_adjoint_hessian()
        else:
            projected_hessian = None
        projected_product = self.project_product()
        return self.fom.with_(primal_model=self.primal_rom, dual_model=self.dual_rom,
                              opt_product=projected_product,
                              estimators=estimator, output_functional_dict=self.projected_output,
                              projected_hessian=projected_hessian,
                              separated_bases=self.separated_bases, fom=self.fom,
                              is_rom=True,
                              coarse_projection=None)

    def extend_bases(self, mu, printing=True, U = None, P = None, **kwargs):
        tic = time.perf_counter()
        if self.unique_basis:
            U, P = self.extend_unique_basis(mu, U, P)
            return U, P

        if U is None:
            U = self.fom.solve(mu)
        if P is None:
            P = self.fom.solve_dual(mu,U=U)
        try:
            self.primal_reductor.extend_basis(U)
            # self.non_assembled_primal_reductor.extend_basis(U)
        except:
            pass
        self.primal_rom = self.primal_reductor.reduce()
        if self.non_assembled_primal_rom is not None:
            self.non_assembled_primal_rom = self.non_assembled_primal_reductor.reduce()
        self.bases['RB'] = self.primal_reductor.bases['RB']
        self.RBPrimal = self.bases['RB']
        self.RBDual.append(P)
        self.RBDual = gram_schmidt(self.RBDual, offset=len(self.RBDual)-1, product=self.opt_product)
        an, bn = len(self.RBPrimal), len(self.RBDual)
        self.dual_intermediate_fom, self.dual_rom, self.dual_reductor = self._build_dual_models()
        self.dual = self.dual_reductor
        self.bases['DU'] = self.dual_reductor.bases['RB']

        if printing:
            print('Enrichment completed... length of Bases are {} and {}'.format(an,bn))
        print(f'... enrichment took {time.perf_counter()-tic:.5f}s ...', end='')
        self.time_for_enrichment += time.perf_counter()-tic
        print(f'Total enrichment time is {self.time_for_enrichment:.5f}s')
        return U, P

    def extend_unique_basis(self,mu, U = None, P = None):
        assert self.unique_basis
        if U is None:
            U = self.fom.solve(mu=mu)
        if P is None:
            P = self.fom.solve_dual(mu=mu, U=U)
        try:
            self.primal_reductor.extend_basis(U)
        except:
            pass
        try:
            self.primal_reductor.extend_basis(P)
        except:
            pass

        self.primal_rom = self.primal_reductor.reduce()
        if self.non_assembled_primal_rom is not None:
            self.non_assembled_primal_rom = self.non_assembled_primal_reductor.reduce()
        self.bases['RB'] = self.primal_reductor.bases['RB']
        self.RBPrimal = self.bases['RB']

        self.RBDual = self.RBPrimal
        self.dual_intermediate_fom, self.dual_rom, self.dual_reductor = self._build_dual_models()
        self.dual = self.primal_reductor

        an = len(self.RBPrimal)
        print('Length of Basis is {}'.format(an))
        return U, P

    def extend_adaptive_taylor(self, mu, U = None, P = None):
        assert 0, 'not part of this publication'

    def _build_unique_basis(self):
        self.RBPrimal.append(self.RBDual)
        self.RBPrimal = gram_schmidt(self.RBPrimal, product=self.opt_product)
        self.RBDual = self.RBPrimal

    def _build_primal_rom(self):
        if self.reductor_type == 'simple_coercive':
            print('building simple coercive primal reductor...')
            primal_reductor = SimpleCoerciveRBReductor(self.fom.primal_model, RB=self.RBPrimal,
                                                       product=self.opt_product,
                                                       coercivity_estimator=self.coercivity_estimator)
        elif self.reductor_type == 'non_assembled':
            print('building non assembled for primal reductor...')
            primal_reductor = NonAssembledRBReductor(self.fom.primal_model, RB=self.RBPrimal,
                                                             product=self.opt_product,
                                                             coercivity_estimator=self.coercivity_estimator)
        else:
            print('building coercive primal reductor...')
            primal_reductor = CoerciveRBReductor(self.fom.primal_model, RB=self.RBPrimal,
                                                 product=self.opt_product,
                                                 coercivity_estimator=self.coercivity_estimator)

        primal_rom = primal_reductor.reduce()
        return primal_rom, primal_reductor

    def _build_dual_models(self):
        assert self.primal_rom is not None
        assert self.RBPrimal is not None
        RBbasis = self.RBPrimal

        if not isinstance(self.fom.fine_prolongation, IdentityOperator):
            suffix = '_coarse_fine'
        else:
            suffix = ''
        rhs_operators = list(self.fom.output_functional_dict[f'd_u_linear_part{suffix}'].operators)
        rhs_coefficients = list(self.fom.output_functional_dict[f'd_u_linear_part{suffix}'].coefficients)

        bilinear_part = self.fom.output_functional_dict[f'd_u_bilinear_part{suffix}']

        for i in range(len(RBbasis)):
            u = RBbasis[i]
            if isinstance(bilinear_part, LincombOperator):
                for j, op in enumerate(bilinear_part.operators):
                    rhs_operators.append(VectorOperator(op.apply(u)))
                    rhs_coefficients.append(ExpressionParameterFunctional('basis_coefficients[{}]'.format(i),
                                                                  {'basis_coefficients': len(RBbasis)})
                                        * bilinear_part.coefficients[j])
            else:
                rhs_operators.append(VectorOperator(bilinear_part.apply(u, None)))
                rhs_coefficients.append(1. * ExpressionParameterFunctional('basis_coefficients[{}]'.format(i),
                                                                  {'basis_coefficients': len(RBbasis)}))

        dual_rhs_operator = LincombOperator(rhs_operators, rhs_coefficients)

        dual_intermediate_fom = self.fom.primal_model.with_(rhs = dual_rhs_operator)

        if self.reductor_type == 'simple_coercive':
            print('building simple coercive dual reductor...')
            dual_reductor = SimpleCoerciveRBReductor(dual_intermediate_fom, RB=self.RBDual,
                                           product=self.opt_product,
                                           coercivity_estimator=self.coercivity_estimator)
        elif self.reductor_type == 'non_assembled':
            print('building non assembled dual reductor...')
            dual_reductor = NonAssembledRBReductor(dual_intermediate_fom, RB=self.RBDual,
                                            product=self.opt_product, coercivity_estimator=self.coercivity_estimator)
        else:
            print('building coercive dual reductor...')
            dual_reductor = CoerciveRBReductor(dual_intermediate_fom, RB=self.RBDual,
                                          product=self.opt_product,
                                          coercivity_estimator=self.coercivity_estimator)


        dual_rom = dual_reductor.reduce()
        return dual_intermediate_fom, dual_rom, dual_reductor

    def _construct_zero_dict(self, parameters):
        #prepare dict
        zero_dict = {}
        for key, size in parameters.items():
            zero_ = np.empty(size, dtype=object)
            zero_dict[key] = zero_
        return zero_dict

    #prepare dict
    def _construct_zero_dict_dict(self, parameters):
        zero_dict = {}
        for key, size in parameters.items():
            zero_ = np.empty(size, dtype=dict)
            zero_dict[key] = zero_
            for l in range(size):
                zero_dict[key][l] = self._construct_zero_dict(parameters)
        return zero_dict

    def assemble_error_estimator(self):
        self.projected_output = self.project_output()

        # print_pieces 
        print_pieces = 0

        estimators = {}

        # primal
        class PrimalCoerciveRBEstimator(ImmutableObject):
            def __init__(self, primal_rom, non_assembled_rom=None):
                self.__auto_init(locals())
            def estimate_error(self, U, mu, non_assembled=False):
                if non_assembled and self.non_assembled_rom is not None:
                    return self.non_assembled_rom.estimate_error(U, mu)
                else:
                    return self.primal_rom.estimate_error(U, mu)

        estimators['primal'] = PrimalCoerciveRBEstimator(self.primal_rom, self.non_assembled_primal_rom)

        ##########################################

        # dual
        class DualCoerciveRBEstimator(ImmutableObject):
            def __init__(self, coercivity_estimator, cont_k, primal_estimator, dual_rom, non_assembled_rom=None):
                self.__auto_init(locals())

            def estimate_error(self, U, P, mu, non_assembled=False):
                primal_estimate = self.primal_estimator.estimate_error(U, mu, non_assembled=non_assembled)[0]
                if non_assembled and self.non_assembled_rom is not None:
                    dual_intermediate_estimate = self.non_assembled_rom.estimate_error(P, mu)[0]
                else:
                    dual_intermediate_estimate = self.dual_rom.estimate_error(P, mu)
                if print_pieces or 0:
                    print(self.cont_k(mu), self.coercivity_estimator(mu), primal_estimate, dual_intermediate_estimate)
                return 2* self.cont_k(mu) /self.coercivity_estimator(mu) * primal_estimate + dual_intermediate_estimate

        estimators['dual'] = DualCoerciveRBEstimator(self.coercivity_estimator,
                                                     self.cont_k, estimators['primal'],
                                                     self.dual_rom, self.non_assembled_dual_rom)
        ##########################################

        # output hat
        class output_hat_RBEstimator(ImmutableObject):
            def __init__(self, coercivity_estimator, cont_k, cont_j, primal_estimator, dual_estimator,
                         projected_output, dual_rom, P_product, U_product, corrected_output):
                self.__auto_init(locals())

            def estimate_error(self, U, P, mu):
                primal_estimate = self.primal_estimator.estimate_error(U, mu)[0]
                dual_estimate = self.dual_estimator.estimate_error(U, P, mu)

                residual_lhs = self.projected_output['dual_primal_projected_op'].apply2(P, U, mu=mu)[0,0]
                residual_rhs = self.projected_output['dual_projected_rhs'].apply_adjoint(P, mu=mu).to_numpy()[0,0]

                if print_pieces or 0:
                    print(self.coercivity_estimator(mu), primal_estimate, dual_estimate, primal_estimate**2,
                          self.cont_k(mu), primal_estimate, self.coercivity_estimator(mu))

                if self.corrected_output:
                    est1 = self.coercivity_estimator(mu) * primal_estimate * dual_estimate + \
                       primal_estimate**2 * self.cont_k(mu)
                    return est1
                else:
                    est2 = self.coercivity_estimator(mu) * primal_estimate * dual_estimate + \
                       primal_estimate**2 * self.cont_k(mu) + \
                       + np.abs(residual_rhs - residual_lhs)
                    return est2

        estimators['output_functional_hat'] = output_hat_RBEstimator(self.coercivity_estimator,
                                                                     self.cont_k, self.cont_j,
                                                                     estimators['primal'], estimators['dual'],
                                                                     self.projected_output, self.dual_rom,
                                                                     self.dual_rom.opt_product,
                                                                     self.primal_rom.opt_product,
                                                                     self.fom.use_corrected_functional)


        ##########################################
        estimators['u_d_mu'] = None
        estimators['p_d_mu'] = None
        estimators['output_functional_hat_d_mus'] = None
        estimators['hessian_d_mu_il'] = None

        ##########################################
        return estimators

    def project_output(self):
        output_functional = self.fom.output_functional_dict
        li_part = output_functional['linear_part']
        bi_part = output_functional['bilinear_part']
        d_u_li_part = output_functional['d_u_linear_part']
        d_u_bi_part = output_functional['d_u_bilinear_part']

        RB_coarse = li_part.range.empty()
        for rb in self.RBPrimal:
             RB_coarse.append(self.fom.coarse_projection.apply(rb))
        RB = self.RBPrimal
        projected_functionals = {
            'output_coefficient' : output_functional['output_coefficient'],
            'linear_part' : project(li_part, RB_coarse, None),
            'bilinear_part' : project(bi_part, RB, RB),
            'd_u_linear_part' : project(d_u_li_part, RB_coarse, None),
            'd_u_bilinear_part' : project(d_u_bi_part, RB, RB),
            'dual_projected_d_u_bilinear_part' : project(d_u_bi_part, self.RBDual,  RB),
            'dual_primal_projected_op': project(self.fom.primal_model.operator, self.RBDual, RB),
            'dual_projected_rhs': project(self.fom.primal_model.rhs, self.RBDual, None),
            'primal_projected_dual_rhs': project(self.dual_intermediate_fom.rhs, RB, None),
        }
        return projected_functionals

    def project_product(self):
        projected_product = project(self.opt_product, self.RBPrimal, self.RBPrimal)
        return projected_product

    def assemble_estimator_for_subbasis(self, dims):
        raise NotImplementedError

    def _reduce_to_subbasis(self, dims):
        raise NotImplementedError

    def _reduce_to_primal_subbasis(self, dim):
        raise NotImplementedError

class NonAssembledRBReductor(StationaryRBReductor):
    def __init__(self, fom, RB=None, product=None, coercivity_estimator=None,
                 check_orthonormality=None, check_tol=None):
        assert fom.operator.linear and fom.rhs.linear
        assert isinstance(fom.operator, LincombOperator)
        assert all(not op.parametric for op in fom.operator.operators)
        if fom.rhs.parametric:
            assert isinstance(fom.rhs, LincombOperator)
            assert all(not op.parametric for op in fom.rhs.operators)

        super().__init__(fom, RB, product=product, check_orthonormality=check_orthonormality,
                         check_tol=check_tol)
        self.coercivity_estimator = coercivity_estimator

    def assemble_error_estimator(self):
        # compute the Riesz representative of (U, .)_L2 with respect to product
        return non_assembled_estimator(self.fom, self.products['RB'], self)

    def assemble_estimator_for_subbasis(self, dims):
        return self._last_rom.error_estimator.restricted_to_subbasis(dims['RB'], m=self._last_rom)

class non_assembled_estimator(ImmutableObject):
    def __init__(self, fom, product, reductor):
        self.__auto_init(locals())
        self.residual = None

    def estimate_error(self, U, mu, m):
        U = self.reductor.reconstruct(U)
        riesz = self.product.apply_inverse(self.fom.operator.apply(U, mu) - self.fom.rhs.as_vector(mu))
        sqrt = self.product.apply2(riesz,riesz)
        output = np.sqrt(sqrt)
        return output