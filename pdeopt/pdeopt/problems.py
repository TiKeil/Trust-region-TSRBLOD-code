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
from pymor.basic import *
from numbers import Number
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.analyticalproblems.functions import Function

class RandomFieldFunction(Function):
    """Define a 2D |Function| via a random distribution.

    Parameters
    ----------
    bounding_box
        Lower left and upper right coordinates of the domain of the function.
    seed
        random seed for the distribution
    range
        defines the range of the random field
    """

    dim_domain = 2
    shape_range = ()

    def __init__(self, bounding_box=None, range=None, shape=(100,100), seed=0):
        bounding_box = bounding_box or [[0., 0.], [1., 1.]]
        range = range or [0., 1.]
        assert isinstance(range, list) and len(range) == 2
        a_val, b_val = range[0], range[1]
        np.random.seed(seed)
        self.diffusion_field = np.random.uniform(a_val, b_val, np.prod(shape)).reshape(shape).T[:, ::-1]
        self.__auto_init(locals())
        self.lower_left = np.array(bounding_box[0])
        self.size = np.array(bounding_box[1] - self.lower_left)

    def evaluate(self, x, mu=None):
        indices = np.maximum(np.floor((x - self.lower_left) * np.array(self.diffusion_field.shape) / self.size).astype(int), 0)
        F = self.diffusion_field[np.minimum(indices[..., 0], self.diffusion_field.shape[0] - 1),
                                 np.minimum(indices[..., 1], self.diffusion_field.shape[1] - 1)]
        return F

class DiffusionFieldFunction(Function):
    """Define a 2D |Function| via a diffusion field.

    Parameters
    ----------
    bounding_box
        Lower left and upper right coordinates of the domain of the function.
    diffusion_field
        diffusion field as a flattened array
    """

    dim_domain = 2
    shape_range = ()

    def __init__(self, diffusion_field=None, bounding_box=None, shape=None):
        self.shape = shape or (int(np.sqrt(len(diffusion_field))), int(np.sqrt(len(diffusion_field))))
        bounding_box = bounding_box or [[0., 0.], [1., 1.]]
        self.diffusion_field = diffusion_field.reshape(self.shape).T
        self.__auto_init(locals())
        self.lower_left = np.array(bounding_box[0])
        self.size = np.array(bounding_box[1] - self.lower_left)

    def evaluate(self, x, mu=None):
        indices = np.maximum(np.floor((x - self.lower_left) * np.array(self.diffusion_field.shape) /
                                      self.size).astype(int), 0)
        F = self.diffusion_field[np.minimum(indices[..., 0], self.diffusion_field.shape[0] - 1),
                         np.minimum(indices[..., 1], self.diffusion_field.shape[1] - 1)]
        return F

def standard_thermal_blocks_with_multiscale_function(type="expression"):
    """
        THIS FUNCTION HAS BEEN USED FOR GEOSCIENCE 2021 TALK !
    """
    problem = thermal_block_problem((6,6))

    # small multiscale Add-on for the model problem
    funcs, coefs = problem.diffusion.functions, problem.diffusion.coefficients

    if type == "expression":
        # a_2 from henning RBLOD
        const_function = ExpressionFunction("1/100 * (10 + 9 * sin(2 * pi * sqrt(2 * x[..., 0]) / 0.1) * "
                                            "sin(4.5 * pi * x[..., 1] ** 2 / 0.1))", 2, ())

        funcs = [func + const_function for func in funcs]
    elif type == "random_field":
        rf = RandomFieldFunction(range=[0.8, 1], shape=(100, 100))

        funcs = [func * rf for func in funcs]

    problem = problem.with_(diffusion=LincombFunction(funcs, coefs))
    domain_of_interest = ConstantFunction(1, 2)

    return problem, domain_of_interest

def local_thermal_block_multiscale_problem(bounding_box=[[0.25,0.75],[0.125,0.25]], blocks=[16,2]):
    '''
                 a
        __________________
        |                 | D
        |                 |
      b |                 |
        |                 |
        |_________________| C
        [ A              B ]

    '''

    A, B = bounding_box[0][0], bounding_box[0][1]
    C, D = bounding_box[1][0], bounding_box[1][1]

    a = B - A
    b = D - C
    blocks_a = int(blocks[0]/a)
    blocks_b = int(blocks[1]/b)
    problem = thermal_block_problem((blocks_a, blocks_b))

    funcs, coefs = problem.diffusion.functions, problem.diffusion.coefficients
    pre_local_funcs, pre_local_coefs = [], []
    for func, coef in zip(funcs, coefs):
        values = func.values
        ix, iy, dx, dy = values['ix'], values['iy'], values['dx'], values['dy']
        if (A <= ix * dx) and ((ix + 1) * dx <= B) and (C <= iy * dy) and ((iy + 1) * dy <= D):
            pre_local_funcs.append(func)
            pre_local_coefs.append(coef)

    local_coefs, local_funcs = [], []
    parameter_space_size = len(pre_local_funcs)
    for (i, coef) in enumerate(pre_local_coefs):
        coef = ProjectionParameterFunctional('diffusion', size=parameter_space_size, index=i, name=coef.name)
        local_coefs.append(coef)

    const_ms_function = ExpressionFunction("1/1000 * (10 + 9 * sin(2 * pi * sqrt(2 * x[..., 0]) / 0.1) * "
                                            "sin(4.5 * pi * x[..., 1] ** 2 / 0.1))", 2, ())

    local_funcs = [func + const_ms_function for func in pre_local_funcs]
    problem = problem.with_(diffusion=LincombFunction(local_funcs, local_coefs))

    X = '(x[..., 0] >= A) * (x[..., 0] < B)'
    Y = '(x[..., 1] >= C) * (x[..., 1] < D)'
    domain_of_interest = ExpressionFunction(f'{X} * {Y} * 1.', 2, (), values={'A': A, 'B': B, 'C': C, 'D': D},
                                               name='domain_of_interest')

    return problem, domain_of_interest

from gridlod.world import Patch

def construct_coefficients_on_T(Tpatch, first_factor, second_factor):
    from gridlod import util
    T = Tpatch.TInd # use this as seed !
    NFine = Tpatch.NPatchFine
    xt = util.computePatchtCoords(Tpatch)
    bounding_box = [xt[0], xt[-1]]
    NCoarseElement = Tpatch.world.NCoarseElement
    size = NCoarseElement[0]
    # range = [1., 1.]
    range = [0.9, 1.1]
    rf_1 = RandomFieldFunction(bounding_box, range, shape=(size//first_factor, size//first_factor), seed=T)
    rf_2 = RandomFieldFunction(bounding_box, range, shape=(size//second_factor, size//second_factor), seed=T)
    array_1 = rf_1.evaluate(xt).reshape(NFine)
    array_2 = rf_2.evaluate(xt).reshape(NFine)
    return [array_1, array_2]

def constructer_thermal_block(aFineFunctions, first_factor, second_factor):
    def construct_coefficients_model_problem(patch):
        functions = aFineFunctions
        from gridlod import util
        xt = util.computePatchtCoords(patch)
        coarse_indices = patch.coarseIndices
        coarse_indices_mod = coarse_indices % patch.world.NWorldCoarse[0]
        mod_old = -1
        j, l = 0, -1
        blocklists = [[], []]
        for i, (T, Tmod) in enumerate(zip(coarse_indices, coarse_indices_mod)):
            if Tmod < mod_old:
                j += 1
                l = 0
            else:
                l += 1
            Tpatch = Patch(patch.world, 0, T)
            a = construct_coefficients_on_T(Tpatch, first_factor, second_factor)
            for k, a_q in enumerate(a):
                if l==0:
                    blocklists[k].append(([a_q]))
                else:
                    blocklists[k][j].append(a_q)
            mod_old = Tmod
        aPatchMSblock = [np.block(blocklist).ravel() for blocklist in blocklists]
        aPatchblock = []
        for i, func in enumerate(functions[:-1]):
            if i % 2 == 0:
                aPatchblock.append(np.multiply(aPatchMSblock[0], func(xt)))
            else:
                aPatchblock.append(np.multiply(aPatchMSblock[1], func(xt)))
        aPatchblock.append(functions[-1](xt))
        return aPatchblock
    return construct_coefficients_model_problem

from gridlod.world import World

def large_thermal_block(fine_diameter, coarse_elements, blocks=(6,6), plot=False, return_fine=False,
                        rhs_value=10, high_conductivity=6, low_conductivity = 2, min_diffusivity=1,
                        first_factor=4, second_factor=8):
    n = int(1 / fine_diameter * np.sqrt(2))
    N = coarse_elements
    assert n % N == 0
    NFine = np.array([n,n])
    NWorldCoarse = np.array([N, N])

    boundaryConditions = np.array([[0, 0], [0, 0]])
    NCoarseElement = NFine // NWorldCoarse
    world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

    bounding_box = [[0, 0], [1, 1]]
    global_problem = thermal_block_problem(blocks)

    # manipulate thermal block
    funcs, coefs = global_problem.diffusion.functions, global_problem.diffusion.coefficients

    local_coefs, local_funcs = [], []
    low_diffusions = 0
    diffusions = 0
    for i, (coef, func) in enumerate(zip(coefs, funcs)):
        values = func.values
        dx, ix, dy, iy = values['dx'], values['ix'], values['dy'], values['iy']
        if ((dx * ix) > 0.15 and (dx * ix) < 0.7) and ((dy * iy) > 0.15 and (dy * iy) < 0.7):
            low_diffusions += 2
        else:
            diffusions += 2

    j = 0
    for i, (coef, func) in enumerate(zip(coefs, funcs)):
        values = func.values
        dx, ix, dy, iy = values['dx'], values['ix'], values['dy'], values['iy']
        if ((dx * ix) > 0.15 and (dx * ix) < 0.7) and ((dy * iy) > 0.15 and (dy * iy) < 0.7):
            parameter_type = 'low_diffusion'
            size = low_diffusions
            j += 1
            index = 2*(j - 1)
        else:
            parameter_type = 'diffusion'
            size = diffusions
            index = 2*(i - j)

        coef_1 = ProjectionParameterFunctional(parameter_type, size=size, index=index, name=coef.name)
        coef_2 = ProjectionParameterFunctional(parameter_type, size=size, index=index+1, name=coef.name)
        local_coefs.extend([coef_1, coef_2])

        local_func_1 = func
        local_func_2 = func

        local_funcs.append(local_func_1)
        local_funcs.append(local_func_2)

    diffusion_function = LincombFunction(local_funcs, local_coefs) + ConstantFunction(min_diffusivity, 2)

    global_problem = StationaryProblem(diffusion=diffusion_function,
                                       rhs=global_problem.rhs * rhs_value,
                                       parameter_ranges={'diffusion': (min_diffusivity, high_conductivity),
                                                         'low_diffusion': (min_diffusivity, low_conductivity)},
                                       domain = global_problem.domain)

    if return_fine:
        fullpatch = Patch(world, np.inf, 0)
        construct_function = constructer_thermal_block(diffusion_function.functions, first_factor, second_factor)
        aFines = construct_function(fullpatch)
        f_fine = np.ones(world.NpFine) * rhs_value

        funcs = []
        for aFine in aFines:
            funcs.append(DiffusionFieldFunction(aFine))
        ms_feature_diffusion_function = diffusion_function.with_(functions=funcs)
        global_problem_with_ms_features = global_problem.with_(diffusion=ms_feature_diffusion_function)
        global_problem = global_problem_with_ms_features
    else:
        f_fine = None
        aFines = None

    # right hand side
    f = np.ones(world.NpCoarse) * rhs_value

    return global_problem, world,\
           constructer_thermal_block(diffusion_function.functions, first_factor, second_factor), f, aFines, f_fine
