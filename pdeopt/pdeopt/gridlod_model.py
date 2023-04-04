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

import dill
import os
import numpy as np
import scipy.sparse as sparse

from pymor.core.base import ImmutableObject, BasicObject
from pymor.operators.constructions import ZeroOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace, NumpyVectorArray
from pymor.parallel.dummy import DummyPool

from gridlod import pglod, util, fem, linalg, lod, interp, coef
from gridlod.world import Patch


class FEMGridlodModel(ImmutableObject):
    def __init__(self, operator, rhs, boundaryConditions, world, g=None):
        self.__auto_init(locals())
        boundaryMap = boundaryConditions == 0
        self.solution_space = NumpyVectorSpace(world.NpFine, id='STATE')
        fixedFine = util.boundarypIndexMap(world.NWorldFine, boundaryMap=boundaryMap)
        self.freeFine = np.setdiff1d(np.arange(world.NpFine), fixedFine)
        self.ALocFine = world.ALocFine
        self.MFine = fem.assemblePatchMatrix(self.world.NWorldFine, self.world.MLocFine)

        #dirichlet data
        if g is not None:
            coords_fine = util.pCoordinates(world.NWorldFine)
            self.gV_h = g.evaluate(coords_fine)
        else:
            self.gV_h = np.zeros(world.NpFine)

    def solve(self, mu, A=None, F=None, pool=None):
        print("WARNING: gridlod call of FEM solve")
        aFine_mu = self.operator.assemble(mu).matrix[0]
        if A is None:
            AFine = fem.assemblePatchMatrix(self.world.NWorldFine, self.world.ALocFine, aFine_mu)
        else:
            AFine = A
        if F is not None:
            # fFine_mu = var_rhs.as_vector(mu).to_numpy()[0]
            g = np.zeros(self.world.NpFine)
        else:
            fFine_mu = self.rhs.assemble(mu).matrix[0]
            g = self.gV_h
            F = self.MFine * fFine_mu - AFine * g

        # print(AFine)
        AFineFree = AFine[self.freeFine][:, self.freeFine]
        FFineFree = F[self.freeFine]

        uFineFree = linalg.linSolve(AFineFree, FFineFree)
        uFineFull = np.zeros(self.world.NpFine)
        uFineFull[self.freeFine] += uFineFree
        uFineFull += g
        return self.solution_space.from_numpy(uFineFull)


class GridlodModel(BasicObject):
    def __init__(self, operator, rhs, boundaryConditions, world, g, pool=None, evaluation_counter=None,
                 romT=None, reductorT=None, is_rom=False, save_correctors=True,
                 coarse_pymor_rhs=None, store_in_tmp=False, use_fine_mesh=True,
                 aFine_local_constructor=None, parameters=None,
                 aFineCoefficients=None, print_on_ranks=True, construct_patches=True, patchT=None):
        self.__auto_init(locals())
        if use_fine_mesh:
            self.solution_space = NumpyVectorSpace(world.NpFine, id='STATE')
        else:
            self.solution_space = NumpyVectorSpace(world.NpCoarse, id='STATE')
            assert aFine_local_constructor is not None
        if pool is None:
            if construct_patches:
                print('WARNING: You are not using a parallel pool in GridlodModel')

        pool = pool or DummyPool()

        boundaryMap = boundaryConditions == 0
        fixed = util.boundarypIndexMap(world.NWorldCoarse, boundaryMap)
        self.free = np.setdiff1d(np.arange(0, world.NpCoarse), fixed)

        if romT:
            if self.save_correctors:
                assert reductorT is not None # we need the reductor for reconstruction
            else:
                reductorT is None  # it does not make sense to store this !

        self.k = int(np.ceil(np.abs(np.log(np.sqrt(2 * (1.0 / world.NWorldCoarse[0] ** 2))))))
        self.coarse_indices = range(world.NtCoarse)

        # rhs
        self.MCoarse = fem.assemblePatchMatrix(world.NWorldCoarse, world.MLocCoarse)
        # from fine DoFs to coarse DoFs
        self.CoarseDofsInFine = util.fillpIndexMap(world.NWorldCoarse, world.NWorldFine)
        alternative_bCoarseFull = coarse_pymor_rhs.assemble().matrix.T[0]

        if use_fine_mesh:
            if not rhs.parametric:
                # if isinstance(rhs_data, LincombOperator):
                #     for (op, coef) in zip(rhs_data.operators, rhs_data.coefficients):
                #         f_fine = coef * op.matrix[0]
                #         fCoarse = f_fine[CoarseDofsInFine]
                #         try:
                #             bCoarse += self.MCoarse * fCoarse
                #         except:
                #             bCoarse = self.MCoarse * fCoarse
                self.f_fine = rhs.assemble().matrix[0]
                self.fCoarse = self.f_fine[self.CoarseDofsInFine]
                self.bCoarseFull = self.MCoarse * self.fCoarse
            else:
                assert 0
        else:
            self.fCoarse = coarse_pymor_rhs.assemble().matrix[:, 0]
            self.bCoarseFull = alternative_bCoarseFull

        # make sure pymor and gridlod are doing the same thing
        assert np.allclose(alternative_bCoarseFull, self.bCoarseFull)

        if use_fine_mesh:
            # for fine data
            self.basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)

        # prepare patches
        if patchT is None:
            self.patchT = self.pool.map(prepare_patches, list(self.coarse_indices),
                                        world=world, k=self.k)
        else:
            self.patchT = patchT

        if use_fine_mesh or store_in_tmp:
            if construct_patches:
                # this can exceed communicated memory so this should be done serialized: resolved by store_in_tmp variable
                self.aPatchT = self.pool.map(construct_aPatches, self.patchT, operator=operator, store_in_tmp=store_in_tmp,
                                             aFine_constructor=aFine_local_constructor)

        # do not store the pool here
        del self.pool

        if operator is not None:
            self.mu_for_storage = operator.parameters.parse(np.zeros(operator.parameters.dim))
        else:
            self.mu_for_storage = parameters.parse(np.zeros(parameters.dim))

        if evaluation_counter:
            evaluation_counter.set_world(world)

        if g is not None:
            assert 0
            coords_coarse = util.pCoordinates(world.NWorldCoarse)
            # self.gV_H = g.evaluate(coords_coarse)
        # else:
            # self.gV_H = np.zeros(world.NpCoarse)

    def solve_for_correctors(self, mu, compute_correctors=False, pool=None):
        if self.evaluation_counter:
            self.evaluation_counter.count(is_rom=self.is_rom, coarse=False)
        save_correctors = compute_correctors or self.save_correctors
        if pool is None and not self.romT:
            print(f'WARNING: You are not using a parallel pool in corrector solve')
        pool = pool or DummyPool()
        if not self.romT:
            if self.operator is not None:
                aFine_mu = self.operator.assemble(mu).matrix[0]
            else:
                aFine_mu = None
            KmsijT, correctorsListT = zip(*pool.map(self.compute_FOM_corrector, self.patchT,
                                                    aFine_mu=aFine_mu,
                                                    save_correctors=save_correctors,
                                                    mu = mu))
        else:
            def compute_ROM_correctors_(rom):
                return compute_ROM_correctors(rom, mu=mu)
            KmsijT, correctorsListT = zip(*pool.map(compute_ROM_correctors_, list(self.romT)))
        return KmsijT, correctorsListT

    def compute_FOM_corrector(self, patch, aFine_mu, save_correctors, mu):
        Kmsij, correctorsList = compute_FOM_correctors(patch,
                                                aFine_mu=aFine_mu,
                                                boundaryConditions=self.boundaryConditions,
                                                save_correctors=save_correctors,
                                                aFineCoefficients=self.aFineCoefficients,
                                                mu=mu,
                                                aFine_constructor=self.aFine_local_constructor,
                                                store_in_tmp=self.store_in_tmp,
                                                print_on_ranks=self.print_on_ranks)
        return Kmsij, correctorsList

    def solve(self, mu, F=None, verbose=False, KmsijT=None, correctorsListT=None, pool=None, rhs_cor=False,
              return_coarse=False):
        # note: rhs is variable. Makes sense because we may save the computation of the correctors
        #       for the same mu. To reuse these, we store it for only the last instance of mu
        if F is None:
            if self.rhs is not None:
                rhs = self.rhs
                if rhs.parametric:
                    assert 0, "not implemented"

        # boundary conditions only apply if self.rhs is used as rhs
        if F is not None:
            # g = np.zeros(self.world.NpCoarse)
            bCoarseFull = F.copy()
        else:
            # g = self.gV_H
            f_fine = None
            bCoarseFull = self.bCoarseFull.copy()

        if mu == self.mu_for_storage:
            KmsijT, correctorsListT, KFull = self.KmsijT, self.correctorsListT, self.KFull
        else:
            if not KmsijT:
                KmsijT, correctorsListT = self.solve_for_correctors(mu, pool=pool)
            KFull = pglod.assembleMsStiffnessMatrix(self.world, self.patchT, KmsijT)
            # store this
            self.mu_for_storage = mu
            self.KmsijT, self.correctorsListT, self.KFull = KmsijT, correctorsListT, KFull

        if rhs_cor:
            aFine_mu = self.operator.assemble(mu).matrix[0]
            f_fine = f_fine or self.f_fine
            pool = pool or DummyPool()
            correctorRhsT, RmsiT = zip(*pool.map(compute_rhs_correctors, self.patchT,
                                                 aFine_mu=aFine_mu,
                                                 boundaryConditions=self.boundaryConditions,
                                                 f_fine=f_fine, print_on_ranks=self.print_on_ranks))
            Rf = pglod.assemblePatchFunction(self.world, self.patchT, correctorRhsT)
            RFull = pglod.assemblePatchFunction(self.world, self.patchT, RmsiT)
            bCoarseFull -= RFull
        else:
            Rf = 0

        if self.evaluation_counter:
            self.evaluation_counter.count(is_rom=self.is_rom, coarse=True)

        KFree = KFull[self.free][:, self.free]
        # bCoarseFull -= KFull * g
        bCoarseFree = bCoarseFull[self.free]
        xFree = sparse.linalg.spsolve(KFree, bCoarseFree)

        xFull = np.zeros(self.world.NpCoarse)
        xFull[self.free] = xFree

        # store correctors
        if self.save_correctors:
            if self.romT:
                # we have to reconstruct the correctors for the reduced case
                # no parallel version here
                correctorsListT_ = list(map(reconstruct_correctors, list(correctorsListT),
                                        list(self.reductorT), list(self.romT), self.patchT))
            else:
                correctorsListT_ = correctorsListT
            basisCorrectors = pglod.assembleBasisCorrectors(self.world, self.patchT, correctorsListT_)
            modifiedBasis = self.basis - basisCorrectors
            # uLodFine = modifiedBasis * (xFull + g) + Rf
            uLodFine = modifiedBasis * (xFull) + Rf
            u_H_ms = self.solution_space.from_numpy(uLodFine)
        if self.save_correctors and not verbose:
            return u_H_ms
        else:
            if self.use_fine_mesh:
                if return_coarse and F is not None:
                    # return xFull + g
                    return xFull
                # uLodCoarse = self.basis * (xFull + g)
                uLodCoarse = self.basis * (xFull)
            else:
                # uLodCoarse = xFull + g
                uLodCoarse = xFull
            u_H = self.solution_space.from_numpy(uLodCoarse)
            if verbose:
                return u_H, u_H_ms
            else:
                return u_H

def prepare_patches(T, world, k):
    return Patch(world, k, T)

def construct_aPatches(patch, operator, store_in_tmp=False, aFine_constructor=None):
    if operator is None:
        aPatches = aFine_constructor(patch)
    else:
        aFines = [op.matrix[0] for op in operator.operators if not isinstance(op, ZeroOperator)]
        aPatches = [coef.localizeCoefficient(patch, aFine) for aFine in aFines]
    if store_in_tmp is not False:
        dir = store_in_tmp if isinstance(store_in_tmp, str) else 'tmp'
        assert os.path.exists(f'{dir}/')
        if os.path.exists(f'{dir}/apatch_{patch.TInd}'):
            os.remove(f'{dir}/apatch_{patch.TInd}')
        dbfile = open(f'{dir}/apatch_{patch.TInd}', "ab")
        dill.dump(aPatches, dbfile)
        return None
    return aPatches

def construct_aFine_from_mu(aFines, aFinesCoefficients, mu):
    coefs = [c(mu) if not isinstance(c, float) else c for c in aFinesCoefficients]
    dim_array = aFines[0].ndim
    if dim_array == 3:
        a = np.einsum('ijkl,i', aFines, coefs)
    elif dim_array == 1:
        a = np.einsum('ij,i', aFines, coefs)
    return a

def compute_FOM_correctors(patch, aFine_mu, boundaryConditions, save_correctors,
                           aFineCoefficients, mu, aFine_constructor,
                           store_in_tmp=False, print_on_ranks=True):
    """
    classic LOD patch computation on an element
    """
    if print_on_ranks:
        print('s', end='', flush=True)
    IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)

    if aFine_mu is None:
        if store_in_tmp:
            dbfile = open(f'{store_in_tmp}/apatch_{patch.TInd}', "rb")
            aPatches = dill.load(dbfile)
            dbfile.close()
        else:
            aPatches = aFine_constructor(patch)
        aPatch = construct_aFine_from_mu(aPatches, aFineCoefficients, mu)
    else:
        aPatch = lambda: coef.localizeCoefficient(patch, aFine_mu)
    correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
    if not save_correctors:
        correctorsList = None
    return csi.Kmsij, correctorsList

from rblod.parameterized_stage_1 import _build_directional_mus
def compute_ROM_correctors(rom, mu):
    Kmsij, u_roms = [], []
    for dof, mu_ in enumerate(_build_directional_mus(mu)):
        # the output is globalized which is bad for us, so we need to but it back
        Kmsij_from_rom_opt, u_rom = rom.output(mu_, return_solution=True)
        Kmsij_from_rom_opt = sparse.csc_matrix(Kmsij_from_rom_opt).data
        Kmsij.append(Kmsij_from_rom_opt)
        u_roms.append(u_rom.to_numpy())
    Kij_constant = rom.Kij_constant(mu)
    full_Kmsij = np.column_stack(Kmsij).flatten() + Kij_constant
    return full_Kmsij, u_roms

def reconstruct_correctors(u_roms, reductor, rom, patch):
    u_foms = []
    for u_rom in u_roms:
        if not isinstance(u_rom, NumpyVectorArray):
            u_rom = rom.solution_space.from_numpy(u_rom)
        if reductor is None:
            # reductor is locally in storage
            T = patch.TInd
            dbfile = open(f'tmp/red_{T}', "rb")
            reductor = dill.load(dbfile)
        u_foms.append(reductor.reconstruct(u_rom).to_numpy()[0])
    return u_foms

def compute_rhs_correctors(patch, aFine_mu, boundaryConditions, f_fine):
    print('r', end='', flush=True)
    world = patch.world
    IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
    aPatch = lambda: coef.localizeCoefficient(patch, aFine_mu)

    MRhsList = [f_fine[util.extractElementFine(world.NWorldCoarse,
                                          world.NCoarseElement,
                                          patch.iElementWorldCoarse,
                                          extractElements=False)]];

    correctorRhs = lod.computeElementCorrector(patch, IPatch, aPatch, None, MRhsList)[0]
    Rmsi, _ = lod.computeRhsCoarseQuantities(patch, correctorRhs, aPatch, True)

    return correctorRhs, Rmsi
