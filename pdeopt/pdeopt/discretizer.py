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
#   Luca Mechelli (2020)
#   Tim Keil      (2019 - 2021)
# ~~~

import numpy as np

from pymor.discretizers.builtin import discretize_stationary_cg
from pymor.analyticalproblems.functions import ConstantFunction
from pymor.discretizers.builtin.grids.referenceelements import square
from pymor.operators.constructions import VectorOperator
from pymor.discretizers.builtin.cg import L2ProductP1, L2ProductQ1, InterpolationOperator
from pymor.operators.constructions import LincombOperator
from pymor.parameters.functionals import ConstantParameterFunctional

from pdeopt.model import QuadraticPdeoptStationaryModel
from pymor.discretizers.builtin.grids.rect import RectGrid

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

def discretize_quadratic_NCD_pdeopt_stationary_cg(problem, diameter=np.sqrt(2)/200., weights=None,
                                                  domain_of_interest=None, desired_temperature=None, mu_for_u_d=None,
                                                  mu_for_tikhonov=None, coarse_functional_grid_size=None, u_d=None):
    mu_bar = _construct_mu_bar(problem)

    # fine grid
    primal_fom, data = discretize_stationary_cg(problem, diameter=diameter,
                                                grid_type=RectGrid, mu_energy_product=mu_bar)
    grid = data['grid']
    d = grid.dim
    domain_of_interest = domain_of_interest or ConstantFunction(1., d)

    if grid.reference_element is square:
        L2_OP = L2ProductQ1
    else:
        L2_OP = L2ProductP1

    # prepare data functions
    if u_d is None:
        u_desired = ConstantFunction(desired_temperature, d) if desired_temperature is not None else None
        if mu_for_u_d is not None:
            modifified_mu = mu_for_u_d.copy()
            for key in mu_for_u_d.keys():
                if len(mu_for_u_d[key]) == 0:
                    modifified_mu.pop(key)
            u_d = primal_fom.solve(modifified_mu)
        else:
            assert desired_temperature is not None
            u_d = InterpolationOperator(grid, u_desired).as_vector()

    Restricted_L2_OP_fine = L2_OP(grid, data['boundary_info'], dirichlet_clear_rows=False,
                                  coefficient_function=domain_of_interest)

    if coarse_functional_grid_size is not None:
        assert grid.reference_element is square
        # J is defined on coarse grid
        # we need gridlod code !!!
        from gridlod import fem, util
        from gridlod.world import World
        from pymor.operators.constructions import ComponentProjectionOperator
        from pymor.operators.numpy import NumpyMatrixOperator
        from pdeopt.discretize_gridlod import ComponentProjectionFromBothSides

        assert isinstance(coarse_functional_grid_size, int)
        N = coarse_functional_grid_size
        coarse_diameter = 1. / N * np.sqrt(2)
        coarse_pymor_model, coarse_data = discretize_stationary_cg(problem, diameter=coarse_diameter,
                                                                   grid_type=RectGrid,
                                                                   preassemble=True,
                                                                   mu_energy_product=mu_bar)
        coarse_grid = coarse_data['grid']
        coarse_bi = coarse_data['boundary_info']
        coarse_product = coarse_pymor_model.products['energy']
        coarse_space = coarse_pymor_model.solution_space
        Restricted_L2_OP_on_coarse = L2_OP(coarse_grid, coarse_bi, dirichlet_clear_rows=False,
                                           coefficient_function=domain_of_interest)

        n = int(1 / diameter * np.sqrt(2))
        NFine = np.array([n, n])
        NWorldCoarse = np.array([N, N])

        dom = problem.domain
        a = 0 if dom.left == "dirichlet" else 1
        b = 0 if dom.right == "dirichlet" else 1
        c = 0 if dom.top == "dirichlet" else 1
        d = 0 if dom.bottom == "dirichlet" else 1
        boundaryConditions = np.array([[a, b], [c, d]])

        NCoarseElement = NFine // NWorldCoarse
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)
        CoarseDofsInFine = util.fillpIndexMap(world.NWorldCoarse, world.NWorldFine)
        coarse_proj = ComponentProjectionOperator(CoarseDofsInFine, primal_fom.solution_space,
                                                  range_id=coarse_space.id)
        fine_basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)
        fine_prolongation = NumpyMatrixOperator(fine_basis, source_id=primal_fom.solution_space.id,
                                                            range_id=primal_fom.solution_space.id)
        Restricted_L2_OP = ComponentProjectionFromBothSides(Restricted_L2_OP_on_coarse, coarse_proj)
        if not u_d in coarse_proj.source:
            u_d = fine_prolongation.apply(u_d)
        u_d_on_coarse = coarse_proj.apply(u_d)
    else:
        # J is defined on fine grid
        Restricted_L2_OP = Restricted_L2_OP_fine
        coarse_proj = None
        fine_prolongation = None

    Restricted_L2_OP_fine_coarse_fine = FineCoarseFineOperator(grid, data['boundary_info'], coarse_proj,
                                                               fine_prolongation, dirichlet_clear_rows=False,
                                                               coefficient_function=domain_of_interest)

    l2_u_d_squared = Restricted_L2_OP.apply2(u_d,u_d)[0][0]
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
    output_coefficient = build_output_coefficient(primal_fom.parameters, weights, mu_for_tikhonov,
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

    if coarse_functional_grid_size is not None:
        output_functional['linear_part_coarse_full'] = LincombOperator(
            [VectorOperator(Restricted_L2_OP_on_coarse.apply(u_d_on_coarse))],[-state_functional])      # j(.)
        output_functional['bilinear_part_coarse_full'] = LincombOperator(
            [Restricted_L2_OP_on_coarse],[0.5*state_functional])                                        # k(.,.)
        output_functional['d_u_linear_part_coarse_full'] = LincombOperator(
            [VectorOperator(Restricted_L2_OP_on_coarse.apply(u_d_on_coarse))],[-state_functional])      # j(.)
        output_functional['d_u_bilinear_part_coarse_full'] = LincombOperator(
            [Restricted_L2_OP_on_coarse], [state_functional])
        output_functional['coarse_opt_product'] = coarse_product

        output_functional['linear_part_coarse_fine'] = LincombOperator(
            [VectorOperator(Restricted_L2_OP_fine_coarse_fine.apply(u_d))],[-state_functional])      # j(.)
        output_functional['bilinear_part_coarse_fine'] = LincombOperator(
            [Restricted_L2_OP_fine_coarse_fine],[0.5*state_functional])                                        # k(.,.)
        output_functional['d_u_linear_part_coarse_fine'] = LincombOperator(
            [VectorOperator(Restricted_L2_OP_fine_coarse_fine.apply(u_d))],[-state_functional])      # j(.)
        output_functional['d_u_bilinear_part_coarse_fine'] = LincombOperator(
            [Restricted_L2_OP_fine_coarse_fine], [state_functional])

    opt_product = primal_fom.energy_product                                     # energy w.r.t. mu_bar (see above)

    primal_fom = primal_fom.with_(products=dict(opt=opt_product, **primal_fom.products))
    pde_opt_fom = QuadraticPdeoptStationaryModel(primal_fom, output_functional, opt_product=opt_product,
                                                 use_corrected_functional=True, adjoint_approach=True,
                                                 coarse_projection=coarse_proj,
                                                 fine_prolongation=fine_prolongation)
    return pde_opt_fom, data, mu_bar

class FineCoarseFineOperator(L2ProductQ1):
    def __init__(self, grid, bi, coarse_projection, fine_prolongation,
                 dirichlet_clear_rows=False, coefficient_function=None):
        super().__init__(grid, bi, dirichlet_clear_rows=False, coefficient_function=coefficient_function)
        self.__auto_init(locals())

    def to_coarse_and_back(self, U):
        return self.fine_prolongation.apply(self.coarse_projection.apply(U))

    def apply(self, U, mu=None):
        return self.assemble(mu).apply(self.to_coarse_and_back(U))

    def apply_adjoint(self, V, mu=None):
        return self.assemble(mu).apply_adjoint(self.to_coarse_and_back(V))

    def apply2(self, V, U, mu=None):
        return super().apply2(self.to_coarse_and_back(V), self.to_coarse_and_back(U), mu)