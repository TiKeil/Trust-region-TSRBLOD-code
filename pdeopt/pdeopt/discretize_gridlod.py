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

from pymor.discretizers.builtin import discretize_stationary_cg
from pymor.analyticalproblems.functions import ConstantFunction, LincombFunction
from pymor.discretizers.builtin.grids.referenceelements import square
from pymor.operators.constructions import VectorOperator, ComponentProjectionOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.discretizers.builtin.cg import (BoundaryDirichletFunctional, L2ProductFunctionalQ1,
                                           L2ProductP1, L2ProductQ1, InterpolationOperator,
                                           BoundaryL2ProductFunctional)
from pymor.operators.constructions import LincombOperator, ZeroOperator
from pymor.parameters.functionals import ConstantParameterFunctional
from pymor.parameters.base import ParametricObject
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.discretizers.builtin.grids.rect import RectGrid

from pdeopt.model import QuadraticPdeoptStationaryModel
from pdeopt.gridlod_model import FEMGridlodModel, GridlodModel

from gridlod import util
from gridlod.world import World


def _construct_mu_bar(problem):
    mu_bar = []
    for key, size in sorted(problem.parameter_space.parameters.items()):
        range_ = problem.parameter_space.ranges[key]
        if range_[0] == 0:
            value = 10**(np.log10(range_[1])/2)
        else:
            value = 10**((np.log10(range_[0]) + np.log10(range_[1]))/2)
        for i in range(size):
            mu_bar.append(value)
    return problem.parameters.parse(mu_bar)

def discretize_gridlod_fem(problem, fine_diameter):
    n = int(1/fine_diameter * np.sqrt(2))
    assert n % 2 == 0
    N = 2

    NFine = np.array([n, n])
    NWorldCoarse = np.array([N, N])

    g = problem.dirichlet_data

    dom = problem.domain
    assert not dom.has_robin
    a = 0 if dom.left == "dirichlet" else 1
    b = 0 if dom.right == "dirichlet" else 1
    c = 0 if dom.top == "dirichlet" else 1
    d = 0 if dom.bottom == "dirichlet" else 1
    boundaryConditions = np.array([[a, b], [c, d]])

    NCoarseElement = NFine // NWorldCoarse
    world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

    xt = util.tCoordinates(NFine)
    NtFine = xt.shape[0]
    xp = util.pCoordinates(NFine)
    NpFine = xp.shape[0]

    # simplify the data structure
    if isinstance(problem.diffusion, LincombFunction):
        data = [ZeroOperator(NumpyVectorSpace(1), NumpyVectorSpace(NtFine))]
        coefficients = [1.]
        for (func, coef) in zip(problem.diffusion.functions, problem.diffusion.coefficients):
            if isinstance(coef, ParametricObject):
                data.append(NumpyMatrixOperator(func(xt)))
                coefficients.append(coef)
            else:
                data[0] += coef * NumpyMatrixOperator(func(xt))
    else:
        data = [NumpyMatrixOperator(problem.diffusion(xt))]
        coefficients = [1.]
    data[0] = data[0].assemble()
    lhs_data = LincombOperator(data, coefficients)

    if isinstance(problem.rhs, LincombFunction):
        data = [ZeroOperator(NumpyVectorSpace(1), NumpyVectorSpace(NpFine))]
        coefficients = [1.]
        for (func, coef) in zip(problem.rhs.functions, problem.rhs.coefficients):
            if isinstance(coef, ParametricObject):
                data.append(NumpyMatrixOperator(func(xp)))
                coefficients.append(coef)
            else:
                data[0] += coef * NumpyMatrixOperator(func(xp))
    else:
        data = [NumpyMatrixOperator(problem.rhs(xp))]
        coefficients = [1.]
    data[0] = data[0].assemble()
    rhs_data = LincombOperator(data, coefficients)

    fem_with_gridlod = FEMGridlodModel(lhs_data, rhs_data, boundaryConditions, world, g)

    return fem_with_gridlod


def discretize_gridlod(problem, fine_diameter, coarse_elements, pool=None, counter=None, save_correctors=True,
                       store_in_tmp=False, mu_energy_product=None, use_fine_mesh=True, aFine_constructor=None,
                       print_on_ranks=True, construct_aFine_globally=False):
    n = int(1/fine_diameter * np.sqrt(2))
    assert n % coarse_elements == 0
    N = coarse_elements

    coarse_diameter = 1./N * np.sqrt(2) + 1e-8
    coarse_pymor_model, coarse_data = discretize_stationary_cg(problem, diameter=coarse_diameter, grid_type=RectGrid,
                                                               preassemble=False)
    coarse_grid = coarse_data['grid']
    assert coarse_grid.num_intervals[0] == N
    coarse_pymor_rhs = coarse_pymor_model.rhs
    ops, coefs = [], []
    for op, coef in zip(coarse_pymor_rhs.operators, coarse_pymor_rhs.coefficients):
        if isinstance(op, L2ProductFunctionalQ1):
            ops.append(op.with_(dirichlet_clear_dofs=False))
            coefs.append(coef)
        elif isinstance(op, BoundaryDirichletFunctional):
            pass
        elif isinstance(op, BoundaryL2ProductFunctional):
            pass
        else:
            assert 0, "this should not happen!"

    filtered_coarse_pymor_rhs = LincombOperator(ops, coefs)

    NFine = np.array([n, n])
    NWorldCoarse = np.array([N, N])

    # g = problem.dirichlet_data
    g = None

    dom = problem.domain
    assert not dom.has_robin
    a = 0 if dom.left == "dirichlet" else 1
    b = 0 if dom.right == "dirichlet" else 1
    c = 0 if dom.top == "dirichlet" else 1
    d = 0 if dom.bottom == "dirichlet" else 1
    boundaryConditions = np.array([[a, b], [c, d]])
    assert np.sum(boundaryConditions) == 0, 'The other cases are not tested at the moment!!'

    NCoarseElement = NFine // NWorldCoarse
    world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

    middle_coarse_index = np.prod(world.NWorldCoarse) // 2 + world.NWorldCoarse[0] // 2
    from gridlod.world import Patch
    k = int(np.ceil(np.abs(np.log(np.sqrt(2 * (1.0 / world.NWorldCoarse[0] ** 2))))))
    # print(f"Gridlod discretizer: Max DoFs per patch with k={k}:  {Patch(world, k, middle_coarse_index).NpFine}\n")
    # assert 0
    if use_fine_mesh:
        xt = util.tCoordinates(NFine)
        NtFine = xt.shape[0]
        xp = util.pCoordinates(NFine)
        NpFine = xp.shape[0]

        if construct_aFine_globally:
            # extract data for gridlod model
            if isinstance(problem.diffusion, LincombFunction):
                data = []
                for func in problem.diffusion.functions:
                   op = NumpyMatrixOperator(func(xt))
                   data.append(op)
                coefs = problem.diffusion.coefficients
            else:
                data = [NumpyMatrixOperator(problem.diffusion(xt))]
                coefs = [1.]

            lhs_data = LincombOperator(data, coefs)
        else:
            lhs_data = None

        if isinstance(problem.rhs, LincombFunction):
            data = []
            for func in problem.rhs.functions:
                data.append(NumpyMatrixOperator(func(xp)))
            coefs = problem.rhs.coefficients
        else:
            data = [NumpyMatrixOperator(problem.rhs(xp))]
            coefs = [1.]
        rhs_data = LincombOperator(data, coefs)
    else:
        lhs_data, rhs_data = None, None

    gridlod_model = GridlodModel(lhs_data, rhs_data, boundaryConditions, world,
                                 g, pool, counter, save_correctors=save_correctors,
                                 coarse_pymor_rhs=filtered_coarse_pymor_rhs,
                                 store_in_tmp=store_in_tmp, use_fine_mesh=use_fine_mesh,
                                 aFine_local_constructor=aFine_constructor,
                                 parameters=problem.parameters,
                                 aFineCoefficients=problem.diffusion.coefficients,
                                 print_on_ranks=print_on_ranks)

    if mu_energy_product:
        # we have to do this one more time with preassemble=True. it is not too expensive since it is a coarse discretizer
        coarse_pymor_model, _ = discretize_stationary_cg(problem, diameter=coarse_diameter,
                                                                   grid_type=RectGrid,
                                                                   preassemble=True,
                                                                   mu_energy_product=mu_energy_product)
        coarse_product = coarse_pymor_model.products['energy']
    else:
        coarse_product = None

    return gridlod_model, coarse_grid, coarse_pymor_model, coarse_product, coarse_data['boundary_info']


def discretize_quadratic_pdeopt_with_gridlod(problem, diameter=np.sqrt(2)/200., coarse_elements=2, weights=None,
                                             domain_of_interest=None, desired_temperature=None, mu_for_u_d=None,
                                             mu_for_tikhonov=None, pool=None, counter=None, save_correctors=True,
                                             store_in_tmp=False, coarse_J=False, use_fine_mesh=True,
                                             aFine_constructor=None, u_d=None, print_on_ranks=True):

    mu_bar = _construct_mu_bar(problem)
    if use_fine_mesh:
        primal_fom, data = discretize_stationary_cg(problem, diameter=diameter,
                                                    grid_type=RectGrid, mu_energy_product=mu_bar)
    gridlod_fom, coarse_grid, coarse_model, coarse_opt_product, coarse_bi = discretize_gridlod(
        problem, diameter, coarse_elements, pool, counter, save_correctors,
        store_in_tmp=store_in_tmp, mu_energy_product=mu_bar, use_fine_mesh=use_fine_mesh,
        aFine_constructor=aFine_constructor, print_on_ranks=print_on_ranks)
    coarse_space = coarse_model.solution_space

    if use_fine_mesh:
        grid = data['grid']
    else:
        grid = coarse_grid
        data = {'grid': coarse_grid}

    d = grid.dim

    # prepare data functions
    domain_of_interest = domain_of_interest or ConstantFunction(1., d)

    if u_d is None:
        u_desired = ConstantFunction(desired_temperature, d) if desired_temperature is not None else None
        if mu_for_u_d is not None:
            modifified_mu = mu_for_u_d.copy()
            for key in mu_for_u_d.keys():
                if len(mu_for_u_d[key]) == 0:
                    modifified_mu.pop(key)
            if use_fine_mesh:
                u_d = primal_fom.solve(modifified_mu)
            else:
                u_d = gridlod_fom.solve(modifified_mu)
        else:
            assert desired_temperature is not None
            u_d = InterpolationOperator(grid, u_desired).as_vector()

    if grid.reference_element is square:
        L2_OP = L2ProductQ1
    else:
        L2_OP = L2ProductP1

    Restricted_L2_OP_on_coarse = L2_OP(coarse_grid, coarse_bi, dirichlet_clear_rows=False,
                                       coefficient_function=domain_of_interest)
    if use_fine_mesh:
        coarse_proj = ComponentProjectionOperator(gridlod_fom.CoarseDofsInFine, primal_fom.solution_space,
                                                  range_id=coarse_space.id)
        Restricted_L2_OP_coarse = ComponentProjectionFromBothSides(Restricted_L2_OP_on_coarse, coarse_proj)
        u_d_on_coarse = coarse_proj.apply(u_d)
    else:
        coarse_proj = None

    if coarse_J:
        if use_fine_mesh:
            Restricted_L2_OP = Restricted_L2_OP_coarse
        else:
            Restricted_L2_OP = Restricted_L2_OP_on_coarse
    else:
        Restricted_L2_OP = L2_OP(grid, data['boundary_info'], dirichlet_clear_rows=False,
                                 coefficient_function=domain_of_interest)
        coarse_proj = None

    l2_u_d_squared = Restricted_L2_OP.apply2(u_d, u_d)[0][0]
    constant_part = 0.5 * l2_u_d_squared

    # assemble output functional
    from pdeopt.theta import build_output_coefficient
    if weights is not None:
        weight_for_J = weights.pop('sigma_u')
    else:
        weight_for_J = 1.
    state_functional = ConstantParameterFunctional(weight_for_J)

    if mu_for_tikhonov:
        if mu_for_u_d is not None:
            mu_for_tikhonov = mu_for_u_d
        else:
            assert isinstance(mu_for_tikhonov, dict)
    output_coefficient = build_output_coefficient(gridlod_fom.parameters, weights, mu_for_tikhonov,
                                                  None, state_functional, constant_part)

    output_functional = {}

    output_functional['output_coefficient'] = output_coefficient
    output_functional['linear_part'] = LincombOperator(
        [VectorOperator(Restricted_L2_OP.apply(u_d))],[-state_functional])      # j(.)
    output_functional['bilinear_part'] = LincombOperator(
        [Restricted_L2_OP],[0.5*state_functional])                              # k(.,.)
    output_functional['d_u_linear_part'] = LincombOperator(
        [VectorOperator(Restricted_L2_OP.apply(u_d))],[-state_functional])      # j(.)
    output_functional['d_u_bilinear_part'] = LincombOperator(
        [Restricted_L2_OP], [state_functional])                                 # 2k(.,.)

    if use_fine_mesh:
        output_functional['linear_part_coarse'] = LincombOperator(
            [VectorOperator(Restricted_L2_OP_coarse.apply(u_d))],[-state_functional])      # j(.)
        output_functional['bilinear_part_coarse'] = LincombOperator(
            [Restricted_L2_OP_coarse],[0.5*state_functional])                              # k(.,.)
        output_functional['d_u_linear_part_coarse'] = LincombOperator(
            [VectorOperator(Restricted_L2_OP_coarse.apply(u_d))],[-state_functional])      # j(.)
        output_functional['d_u_bilinear_part_coarse'] = LincombOperator(
            [Restricted_L2_OP_coarse], [state_functional])                                 # 2k(.,.)

        output_functional['linear_part_coarse_full'] = LincombOperator(
            [VectorOperator(Restricted_L2_OP_on_coarse.apply(u_d_on_coarse))],[-state_functional])      # j(.)
        output_functional['bilinear_part_coarse_full'] = LincombOperator(
            [Restricted_L2_OP_on_coarse],[0.5*state_functional])                                        # k(.,.)
        output_functional['d_u_linear_part_coarse_full'] = LincombOperator(
            [VectorOperator(Restricted_L2_OP_on_coarse.apply(u_d_on_coarse))],[-state_functional])      # j(.)
        output_functional['d_u_bilinear_part_coarse_full'] = LincombOperator(
            [Restricted_L2_OP_on_coarse], [state_functional])                                           # 2k(.,.)

    output_functional['coarse_opt_product'] = coarse_opt_product

    C = domain_of_interest(grid.centers(2))  # <== these are the vertices!
    C = np.nonzero(C)[0]
    doI = ComponentProjectionOperator(C, Restricted_L2_OP.source)

    output_functional['sigma_u'] = state_functional
    output_functional['u_d'] = u_d
    output_functional['DoI'] = doI

    if use_fine_mesh:
        opt_product = primal_fom.energy_product                                # energy w.r.t. mu_bar (see above)
        primal_fom = primal_fom.with_(products=dict(opt=opt_product, **primal_fom.products))
    else:
        primal_fom = None
        opt_product = coarse_opt_product

    fom = primal_fom or coarse_model

    pde_opt_fom = QuadraticPdeoptStationaryModel(fom, output_functional, opt_product=opt_product,
                                                 use_corrected_functional=False, adjoint_approach=False,
                                                 optional_forward_model=gridlod_fom,
                                                 coarse_projection=coarse_proj,
                                                 fine_prolongation=None
                                                 )
    return pde_opt_fom, data, mu_bar

from pymor.operators.constructions import ConcatenationOperator
class ComponentProjectionFromBothSides(ConcatenationOperator):
    def __init__(self, operator, comp_proj):
        super().__init__([operator, comp_proj])
        self.range = comp_proj.source
        self.__auto_init(locals())

    def apply2(self, V, U, mu=None):
        return super().apply2(self.comp_proj.apply(V), U, mu)

    def apply_adjoint(self, V, mu=None):
        return super().apply_adjoint(self.comp_proj.apply(V), mu)
