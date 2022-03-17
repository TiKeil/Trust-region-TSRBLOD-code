import numpy as np
import os
import dill
import time

from pymor.core.base import ImmutableObject
from pymor.algorithms.projection import project
from pymor.operators.constructions import LincombOperator, VectorOperator, IdentityOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.reductors.basic import ProjectionBasedReductor
from pymor.reductors.coercive import CoerciveRBReductor, SimpleCoerciveRBReductor
from pymor.reductors.residual import ResidualOperator
from pymor.parameters.functionals import ExpressionParameterFunctional, MaxThetaParameterFunctional
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.parallel.dummy import DummyPool

from gridlod import fem

from pdeopt.discretize_gridlod import GridlodModel
from pdeopt.reductor import NonAssembledRBReductor

from rblod.parameterized_stage_1 import CorrectorProblem_for_all_rhs
from rblod.optimized_rom import OptimizedNumpyModelStage1
from rblod.two_scale_model import Two_Scale_Problem
from rblod.two_scale_reductor import CoerciveRBReductorForTwoScale, TrueDiagonalBlockOperator


class QuadraticPdeoptStationaryCoerciveLODReductor(CoerciveRBReductor):
    def __init__(self, fom, f, opt_product=None, coercivity_estimator=None,
                 check_orthonormality=None, check_tol=None, unique_basis=False,
                 reductor_type='simple_coercive', mu_bar=None,
                 prepare_for_hessian=False, parameter_space=None, two_scale=False,
                 greedy_for_two_scale=False, pool=None, optional_enrichment=False,
                 store_in_tmp=False, use_fine_mesh=True,
                 two_scale_estimator_for_RBLOD=True,
                 print_on_ranks=True, add_error_residual=True):
        tic = time.perf_counter()
        # lod setting
        self.gridlod_model = fom.optional_forward_model
        assert isinstance(self.gridlod_model, GridlodModel)
        self.__auto_init(locals())
        self.patchT = self.gridlod_model.patchT
        self.aFineCoefficients = self.gridlod_model.aFineCoefficients
        self.aPatchT = self.gridlod_model.aPatchT

        if self.opt_product is None:
            self.opt_product = fom.opt_product
        super().__init__(fom, product=opt_product, check_orthonormality=check_orthonormality,
                         check_tol=check_tol, coercivity_estimator=coercivity_estimator)

        # this is for the two scale model. Also needed for RBLOD estimator !!
        self.pool = pool or DummyPool()
        self.precompute_constants()
        self.mus_for_enrichment = [] #parameter_space.sample_randomly(5)
        self.initialize_roms()

        del self.pool
        # do not store the pool in this class

        print('preparing constants ... ', end='', flush=True)
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
                print('constant k finished late...', end='', flush=True)
            except:
                self.cont_k = None
                # self.cont_k = lambda mu: 1
                print('constant k failed...', end='', flush=True)
        else:
            print('constant k finished early...', end='', flush=True)


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
                print('constant j finished late...', end='', flush=True)
            except:
                self.cont_j = None
                print('constant j substituted...', end='', flush=True)
        else:
            print('constant j finished early...', end='', flush=True)
        print(f'Initialization took {time.perf_counter() -tic:.4f}s')

        self.total_stage_1_time = time.perf_counter() -tic
        self.total_stage_2_time = 0

    def initialize_roms(self):
        print('initializing roms ...')
        min_alpha = self.min_alpha
        coercivity_estimator = lambda mu: min_alpha
        self.romT, self.reductorT, self.KijT, self.rom_sizeT = \
            zip(*self.pool.map(build_reduced_patch_model, self.patchT, self.aPatchT,
                               aFineCoefficients=self.aFineCoefficients,
                               coercivity_estimator=coercivity_estimator,
                               store_in_tmp=self.store_in_tmp,
                               reductor_type=self.reductor_type,
                               print_on_ranks=self.print_on_ranks)) # aFine_Constructor=None)
        print('') if self.print_on_ranks else 0

    def reduce(self):
        # NOTE: never use super().reduce() for this since the dims are not correctly computed here !
        return self._reduce()

    def _reduce(self):
        tic = time.perf_counter()
        if self.two_scale or self.two_scale_estimator_for_RBLOD:
            self.two_scale_residual, self.m_two_scale, self.two_scale_reductor = self.build_two_scale_model(self.f)
        if self.assemble_error_estimator.__func__ is not ProjectionBasedReductor.assemble_error_estimator:
            with self.logger.block('Assembling error estimator ...'):
                error_estimator = self.assemble_error_estimator()
        else:
            error_estimator = None

        with self.logger.block('Building ROM ...'):
            rom = self.build_rom(error_estimator)
            rom = rom.with_(name=f'{self.fom.name}_reduced')
            rom.disable_logging()
        print(f' ... construction took {time.perf_counter() - tic:.4f}s')
        self.total_stage_2_time += time.perf_counter() - tic
        print(f' ... total stage 2 time is currently {self.total_stage_2_time:.5f}')
        return rom

    def build_rom(self, estimator):
        print('constructing ROM ...', end='', flush=True)
        evaluation_counter = self.gridlod_model.evaluation_counter
        parsed_reductors = self.reductorT if self.gridlod_model.save_correctors else None
        if self.two_scale:
            ##############################
            # forward model
            ##############################
            from pymor.basic import set_defaults
            #set_defaults({"pymor.algorithms.gram_schmidt.gram_schmidt.rtol": 1e-8})

            mus_for_primal_basis = []
            for mu in self.mus_for_enrichment:
            # if 1:
            #     mu = self.mus_for_enrichment[-1]
                mus_for_primal_basis.append(mu)
                print('extending primal model')
                evaluation_counter.count(is_rom=True)
                evaluation_counter.count(is_rom=True, coarse=True)
                try:
                    self.two_scale_reductor.extend_basis(self.m_two_scale.solve(mu))
                except:
                    pass
            tic = time.perf_counter()
            reduced_optional_forward_model = self.two_scale_reductor.reduce()
            print(f'residual reduction took {time.perf_counter()-tic:.5f}s')
            U_mu = reduced_optional_forward_model.solve(mu)
            if self.two_scale_residual is not None:
                print("last estimate is: ", reduced_optional_forward_model.estimate_error(U_mu, mu))

            self.rom_two_scale = reduced_optional_forward_model

            ##############################
            # dual model
            ##############################
            coarse_space = NumpyVectorSpace(len(self.gridlod_model.CoarseDofsInFine), id='STATE')

            len_RB_basis = len(self.two_scale_reductor.partial_basis)
            rhs_operators = []
            rhs_coefficients = []
            for i, basis in enumerate(self.two_scale_reductor.partial_basis):
                full_basis = np.zeros(self.gridlod_model.world.NpCoarse)
                full_basis[self.gridlod_model.free] = basis.to_numpy()[0]
                full_basis = coarse_space.from_numpy(full_basis)
                if self.gridlod_model.use_fine_mesh:
                    coarse_dual_model_rhs = self.fom._build_dual_model(full_basis, mu=None, coarse_model=True,
                                                                       coarse_input=True)
                else:
                    coarse_dual_model_rhs = self.fom._build_dual_model(full_basis, mu=None)
                basis_projection = ExpressionParameterFunctional(f'basis_coefficients[{i}]',
                                                                 {'basis_coefficients': len_RB_basis})
                for c, op in zip(coarse_dual_model_rhs.coefficients, coarse_dual_model_rhs.operators):
                    if not op.name == 'VectorOperator_linear_part':
                        rhs_coefficients.append(basis_projection * c)
                        rhs_operators.append(op)
            for c, op in zip(coarse_dual_model_rhs.coefficients, coarse_dual_model_rhs.operators):
                if op.name == 'VectorOperator_linear_part':
                    rhs_coefficients.append(c)
                    rhs_operators.append(op)

            rhs = LincombOperator(rhs_operators, rhs_coefficients)

            # use the matrices from before !!
            As, BTss, CTss, DTss, source_spaces = self.m_two_scale.extract_ABCD_and_sp()
            res_op, res_prod = self.two_scale_reductor.extract_op_prod()
            dual_two_scale_residual, dual_m_two_scale, self.dual_two_scale_reductor = self.build_two_scale_model(
                rhs, As, BTss, CTss, DTss, source_spaces, res_op, res_prod)
            mus_for_dual_basis = []
            for mu in self.mus_for_enrichment:
            # if 1:
            #     mu = mu
                mus_for_dual_basis.append(mu)
                print('extending dual model')
                evaluation_counter.count(is_rom=True)
                evaluation_counter.count(is_rom=True, coarse=True)
                U = self.rom_two_scale.solve(mu)
                mu_with_U = mu.with_(basis_coefficients=U.to_numpy()[0])
                P = dual_m_two_scale.solve(mu_with_U)
                try:
                    self.dual_two_scale_reductor.extend_basis(P)
                except:
                    pass
            
            tic = time.perf_counter()
            self.dual_reduced_optional_forward_model = self.dual_two_scale_reductor.reduce()
            print(f'residual reduction took {time.perf_counter()-tic:.5f}s')
            P_mu = self.dual_reduced_optional_forward_model.solve(mu_with_U)
            if dual_two_scale_residual is not None:
                print("last dual estimate is: ", self.dual_reduced_optional_forward_model.estimate_error(P_mu, mu_with_U))
            #set_defaults({"pymor.algorithms.gram_schmidt.gram_schmidt.rtol": 1e-4})

            ##############################
            # gather based for reduced output
            ##############################
            primal_lod_basis = self.gridlod_model.solution_space.empty()
            dual_lod_basis = self.gridlod_model.solution_space.empty()
            for mu, xFree_primal in zip(mus_for_primal_basis, self.two_scale_reductor.partial_basis):
                xFull_primal = np.zeros(self.gridlod_model.world.NpCoarse)
                xFull_primal[self.gridlod_model.free] = xFree_primal.to_numpy()[0]
                if self.gridlod_model.use_fine_mesh:
                    if self.gridlod_model.save_correctors:
                        assert 0, 'this part of the code is not available'
                    else:
                        basis = self.gridlod_model.basis
                    primalBasis = basis * xFull_primal
                    primal_lod_basis.append(self.gridlod_model.solution_space.from_numpy(primalBasis))
                else:
                    primalBasis = xFull_primal
                    primal_lod_basis.append(self.gridlod_model.solution_space.from_numpy(primalBasis))

            for mu, xFree_dual in zip(mus_for_dual_basis, self.dual_two_scale_reductor.partial_basis):
                xFull_dual = np.zeros(self.gridlod_model.world.NpCoarse)
                xFull_dual[self.gridlod_model.free] = xFree_dual.to_numpy()[0]
                if self.gridlod_model.use_fine_mesh:
                    if self.gridlod_model.save_correctors:
                        assert 0, 'this part of the code is not available'
                    else:
                        basis = self.gridlod_model.basis
                    dualBasis = basis * xFull_dual
                    dual_lod_basis.append(self.gridlod_model.solution_space.from_numpy(dualBasis))
                else:
                    dualBasis = xFull_dual
                    dual_lod_basis.append(self.gridlod_model.solution_space.from_numpy(dualBasis))

            two_scale_estimator = self.assemble_error_estimator(primal_lod_basis, dual_lod_basis)
            projected_product = self.project_product(primal_lod_basis)
            print(f' ... enrichment completed... length of two scale bases are '
                  f'{len(primal_lod_basis), len(dual_lod_basis)}')
            print(f' length of corrector bases are {self.rom_sizeT}')
            return self.fom.with_(estimators=two_scale_estimator, optional_forward_model=self.rom_two_scale,
                                  dual_model=self.dual_reduced_optional_forward_model, fom=self.fom,
                                  evaluation_counter=evaluation_counter,
                                  opt_product=projected_product,
                                  output_functional_dict=self.projected_output,
                                  coarse_projection=IdentityOperator(self.rom_two_scale.solution_space))
        else:
            print('constructing reduced gridlod model ...')
            reduced_optional_forward_model = GridlodModel(operator = self.gridlod_model.operator,
                                                  rhs = self.gridlod_model.rhs,
                                                  boundaryConditions = self.gridlod_model.boundaryConditions,
                                                  world = self.gridlod_model.world,
                                                  g = self.gridlod_model.g,
                                                  evaluation_counter = evaluation_counter,
                                                  romT = self.romT,
                                                  reductorT = parsed_reductors,
                                                  is_rom = True,
                                                  save_correctors= self.gridlod_model.save_correctors,
                                                  coarse_pymor_rhs=self.gridlod_model.coarse_pymor_rhs,
                                                  aFine_local_constructor=self.gridlod_model.aFine_local_constructor,
                                                  use_fine_mesh=self.gridlod_model.use_fine_mesh,
                                                  parameters=self.gridlod_model.parameters,
                                                  aFineCoefficients=self.gridlod_model.aFineCoefficients,
                                                  store_in_tmp=self.gridlod_model.store_in_tmp,
                                                  patchT=self.patchT,
                                                  construct_patches=False
                                                  )
            print(f' ... enrichment completed... length of bases are {self.rom_sizeT}')
            return self.fom.with_(estimators=estimator, optional_forward_model=reduced_optional_forward_model,
                                  fom=self.fom)

    def extend_bases(self, mu, U = None, P = None, corT = None, pool=None):
        print('extending bases...', end='', flush=True)
        tic = time.perf_counter()
        if self.two_scale:
            self.mus_for_enrichment.append(mu)
        if pool is None:
            print('WARNING: You are not using a parallel pool')
        pool = pool or DummyPool()
        self.gridlod_model.evaluation_counter.count(is_rom=False)
        self.gridlod_model.evaluation_counter.count(is_rom=False, coarse=True)
        corT = [None for i in range(len(self.romT))]

        if self.store_in_tmp:
            if self.optional_enrichment:
                KmsijT, self.rom_sizeT = zip(*pool.map(
                    extend_patch_adaptively_cached_reductor, self.patchT, list(self.romT), list(corT), mu=mu,
                    store_in_tmp=self.store_in_tmp, gridlod_model=self.gridlod_model,
                    print_on_ranks=self.print_on_ranks, add_error_residual=self.add_error_residual
                ))
            else:
                KmsijT, self.rom_sizeT = zip(*pool.map(
                    extend_patch_cached_reductor, self.patchT, list(corT), mu=mu, store_in_tmp=self.store_in_tmp,
                    gridlod_model=self.gridlod_model, print_on_ranks=self.print_on_ranks,
                    add_error_residual=self.add_error_residual
                ))
            self.romT = []
            dir = self.store_in_tmp if isinstance(self.store_in_tmp, str) else 'tmp'
            for T in range(len(self.patchT)):
                dbfile = open(f'{dir}/rom_{T}', "rb")
                rom = dill.load(dbfile)
                self.romT.append(rom)
        else:
            if self.optional_enrichment:
                KmsijT, self.romT, self.reductorT, self.rom_sizeT = zip(*pool.map(
                    extend_patch_adaptively, list(self.romT), list(self.reductorT), list(corT), mu=mu,
                    print_on_ranks=self.print_on_ranks, add_error_residual=self.add_error_residual
                ))
            else:
                KmsijT, self.romT, self.reductorT, self.rom_sizeT = zip(*pool.map(
                    extend_patch, list(self.reductorT), list(corT), mu=mu, print_on_ranks=self.print_on_ranks,
                    add_error_residual=self.add_error_residual
                ))
        print(f' ... Stage 1 enrichment took {time.perf_counter() - tic:.4f}s')
        self.total_stage_1_time += time.perf_counter() - tic
        print(f' ... total stage 1 time is currently {self.total_stage_1_time:.5f}')
        return KmsijT, corT

    def build_two_scale_model(self, f, As=None, BTss=None, CTss=None, DTss=None, source_spaces=None,
                              res_op=None, res_prod=None):
        print('building_two_scale_model')
        m_two_scale = Two_Scale_Problem(self.romT, self.KijT, f,
                                        self.patchT, self.aFineCoefficients,
                                        self.contrast, self.min_alpha,
                                        As=As, BTss=BTss, CTss=CTss, DTss=DTss, source_spaces=source_spaces)
        world = self.gridlod_model.world
        H1Coarse = fem.assemblePatchMatrix(world.NWorldCoarse, world.ALocCoarse)
        H1Coarse = H1Coarse[m_two_scale.free][:, m_two_scale.free]
        blocks = [NumpyMatrixOperator(H1Coarse)]
        blocks.extend([None for _ in range(m_two_scale.NCoarse)])
        full_source_spaces = [blocks[0].source]
        full_source_spaces.extend(m_two_scale.source_spaces)
        two_scale_product = TrueDiagonalBlockOperator(blocks, only_first=True, source_spaces=full_source_spaces)
        print('building_two_scale_reductor')
        reductor_two_scale = CoerciveRBReductorForTwoScale(world, self.romT, m_two_scale,
                                                           coercivity_estimator=self.constants,
                                                           product=two_scale_product,
                                                           check_orthonormality=False,
                                                           residual_operator=res_op,
                                                           residual_product=res_prod)
        residual_reductor = reductor_two_scale.residual_reductor
        if residual_reductor is not None:
            residual = ResidualOperator(residual_reductor.operator, residual_reductor.rhs)
        else:
            residual = None
        return residual, m_two_scale, reductor_two_scale

    def precompute_constants(self):
        assert self.parameter_space
        training_set = self.parameter_space.sample_randomly(50)

        contrasts, min_alphas = zip(*self.pool.map(compute_contrast_for_all_patches, list(self.aPatchT),
                                                   list(np.arange(len(self.aPatchT))),
                                                   aFineCoefficients=self.aFineCoefficients,
                                                   training_set=training_set, store_in_tmp=self.store_in_tmp))
        self.contrast, self.min_alpha = np.max(contrasts), np.min(min_alphas)

        # C = sqrt(5) * gamma_k^(-1) = sqrt(5) * C_IH * 1/sqrt(alpha)
        # gamma_k^(-1) = C_IH * 1/sqrt(alpha)
        C_IH = 1 # always C_IH > 1. and  C_IH /approx 1 for quadrilateral meshes
        self.gamma_k = np.sqrt(self.min_alpha) * 1/C_IH
        C = np.sqrt(5) * 1/self.gamma_k
        self.constants = lambda mu: 1 / C   # needs to be inverted for pymor

    def assemble_error_estimator(self, RB_primal=None, RB_dual=None):
        if self.two_scale and RB_primal is None:
            return None
        if self.two_scale:
            assert RB_primal is not None
            assert RB_dual is not None
            self.projected_output = self.project_output(RB_primal, RB_dual)

        estimators = {}

        # primal
        class PrimalCoerciveRBEstimator(ImmutableObject):
            # THIS IS A NAIVE ESTIMATOR !!
            def __init__(self, romT):
                self.__auto_init(locals())
                self.pool = DummyPool()
            def estimate_error(self, U, mu, **kwargs):
                errorsT = self.pool.map(compute_patch_errors, list(self.romT), mu=mu, **kwargs)
                ret = np.linalg.norm(errorsT)
                return ret

        # primal two-scale for RBLOD
        class PrimalCoerciveRBEstimatorTS(ImmutableObject):
            def __init__(self, m_two_scale, residual):
                self.__auto_init(locals())
            def estimate_error(self, U, mu):
                U = self.m_two_scale.solve(mu)
                ret = self.residual.apply(U, mu).norm()
                return ret

        # primal two-scale rblod
        class PrimalCoerciveRBEstimatorTSRBLOD(ImmutableObject):
            def __init__(self, rom_two_scale):
                self.__auto_init(locals())
            def estimate_error(self, U, mu):
                U = self.rom_two_scale.solve(mu)
                return self.rom_two_scale.estimate(U, mu)

        if self.two_scale:
            estimators['primal'] = PrimalCoerciveRBEstimatorTSRBLOD(self.rom_two_scale)
        else:
            # using the two scale formulation for the error estimation
            if self.two_scale_estimator_for_RBLOD:
                estimators['primal'] = PrimalCoerciveRBEstimatorTS(self.m_two_scale, self.two_scale_residual)
            else:
                # using a naive error estimation
                estimators['primal'] = PrimalCoerciveRBEstimator(self.romT)

        ##########################################

        # dual
        class DualCoerciveRBEstimatorTSRBLOD(ImmutableObject):
            def __init__(self, gamma_k, cont_k, primal_estimator, dual_rom_two_scale):
                self.__auto_init(locals())

            def estimate_error(self, U, P, mu):
                primal_estimate = self.primal_estimator.estimate_error(U, mu)
                dual_estimate = self.dual_rom_two_scale.estimate_error(P, mu)
                return np.sqrt(5)/self.gamma_k * (2* self.cont_k(mu) * primal_estimate + dual_estimate)

        class DualCoerciveRBEstimatorTS(ImmutableObject):
            def __init__(self, gamma_k, cont_k, primal_estimator, primal_residual, build_dual_rhs, coarse_space,
                         free_coarse_dofs, use_fine_mesh):
                self.__auto_init(locals())

            def estimate_error(self, U, P, mu):
                primal_estimate = self.primal_estimator.estimate_error(U, mu)

                full_U = np.zeros(self.coarse_space.dim)
                full_U[self.free_coarse_dofs] = U.block(0).to_numpy()[0]
                full_U = self.coarse_space.from_numpy(full_U)
                primal_rhs = self.primal_residual.rhs

                rhs = self.build_dual_rhs(full_U, mu, coarse_model=self.use_fine_mesh,
                                          coarse_input=self.use_fine_mesh)
                rhsFull = rhs.as_vector().to_numpy()[0]
                rhsFree = rhsFull[self.free_coarse_dofs]
                rhs_ = VectorOperator(primal_rhs.blocks[0][0].source.from_numpy(rhsFree))

                dual_rhs_blocks = primal_rhs.blocks
                dual_rhs_blocks[0] = [rhs_]
                dual_rhs = primal_rhs.with_(blocks=dual_rhs_blocks)

                dual_residual = self.primal_residual.with_(rhs = dual_rhs)

                # the corrector problems remain the same for the symmetric operator
                P_ = U.copy().to_numpy()
                if not isinstance(P, np.ndarray):
                    # then P is a pymor object
                    P = P.to_numpy()[0]
                np.put(P_, np.arange(len(self.free_coarse_dofs)), P[self.free_coarse_dofs])
                P = U.space.from_numpy(P_)
                dual_estimate = dual_residual.apply(P, mu).norm()
                return np.sqrt(5)/self.gamma_k * (2* self.cont_k(mu) * primal_estimate + dual_estimate)


        if self.two_scale:
            estimators['dual'] = DualCoerciveRBEstimatorTSRBLOD(self.gamma_k, self.cont_k, estimators['primal'],
                                                                self.dual_reduced_optional_forward_model)
        else:
            if self.two_scale_estimator_for_RBLOD:
                coarse_space = NumpyVectorSpace(len(self.gridlod_model.CoarseDofsInFine), id='STATE')
                estimators['dual'] = DualCoerciveRBEstimatorTS(self.gamma_k, self.cont_k, estimators['primal'],
                                                               self.two_scale_residual, self.fom._build_dual_model,
                                                               coarse_space, self.gridlod_model.free,
                                                               self.gridlod_model.use_fine_mesh)
            else:
                estimators['dual'] = estimators['primal']

        ##########################################

        # truncation homogenization term
        class TruncationHomogenizationTerm(ImmutableObject):
            def __init__(self, primal, dual, testing=0):
                self.__auto_init(locals())

            def estimate(self, U, P, mu):
                if not self.testing:
                    est_1 = self.primal.estimate_error(U, mu)
                    est_2 = self.dual.estimate_error(U, P, mu)
                    norm_P = P.block(0).norm()
                    a_priori_part = 0
                    return est_1 *(a_priori_part +  est_2  * norm_P) + est_1 * norm_P
                else:
                    return 0

        # output hat
        class output_hat_RBEstimator_TSRBLOD(ImmutableObject):
            def __init__(self, cont_k, primal_estimator, dual_estimator, two_scale, hom_term):
                self.__auto_init(locals())

            def estimate_error(self, U, P, mu, **kwargs):
                if self.two_scale:
                    U = self.two_scale.solve(mu)
                    primal_estimate = self.primal_estimator.estimate_error(U, mu, **kwargs)
                    dual_estimate = self.dual_estimator.estimate_error(U, P, mu, **kwargs)
                else:
                    primal_estimate = self.primal_estimator.estimate_error(U, mu, **kwargs)
                    dual_estimate = primal_estimate

                est = primal_estimate * dual_estimate + primal_estimate**2 * self.cont_k(mu)\
                      + self.hom_term.estimate(U, P, mu)

                return est

        hom_term = TruncationHomogenizationTerm(estimators['primal'], estimators['dual'], 1)
        if self.two_scale:
            estimators['output_functional_hat'] = output_hat_RBEstimator_TSRBLOD(
                self.cont_k, estimators['primal'], estimators['dual'], self.rom_two_scale, hom_term)
        else:
            if self.two_scale_estimator_for_RBLOD:
                estimators['output_functional_hat'] = output_hat_RBEstimator_TSRBLOD(
                    self.cont_k, estimators['primal'], estimators['dual'], self.m_two_scale, hom_term)
            else:
                estimators['output_functional_hat'] = output_hat_RBEstimator_TSRBLOD(
                    self.cont_k, estimators['primal'], estimators['dual'], None, hom_term)

        ##########################################
        estimators['u_d_mu'] = None
        estimators['p_d_mu'] = None
        estimators['output_functional_hat_d_mus'] = None
        estimators['hessian_d_mu_il'] = None

        return estimators

    def project_output(self, RB_primal, RB_dual):
        output_functional = self.fom.output_functional_dict
        li_part = output_functional['linear_part']
        bi_part = output_functional['bilinear_part']
        d_u_li_part = output_functional['d_u_linear_part']
        d_u_bi_part = output_functional['d_u_bilinear_part']
        RB_coarse = li_part.range.empty()
        for rb in RB_primal:
             RB_coarse.append(self.fom.coarse_projection.apply(rb))
        projected_functionals = {
           'output_coefficient' : output_functional['output_coefficient'],
           'linear_part' : project(li_part, RB_coarse, None),
           'bilinear_part' : project(bi_part, RB_primal, RB_primal),
           'd_u_linear_part' : project(d_u_li_part, RB_coarse, None),
           'd_u_bilinear_part' : project(d_u_bi_part, RB_primal, RB_primal),
           'dual_projected_d_u_bilinear_part' : project(d_u_bi_part, RB_primal, RB_dual),
           'dual_primal_projected_op': project(self.fom.primal_model.operator, RB_dual, RB_primal),
           'dual_projected_rhs': project(self.fom.primal_model.rhs, RB_dual, None),
        }
        return projected_functionals

    def project_product(self, RB_primal):
        projected_product = project(self.opt_product, RB_primal, RB_primal)
        return projected_product

    def assemble_estimator_for_subbasis(self, dims):
        raise NotImplementedError

    def _reduce_to_subbasis(self, dims):
        raise NotImplementedError

    def _reduce_to_primal_subbasis(self, dim):
        raise NotImplementedError


def build_reduced_patch_model(patch, aPatch, aFineCoefficients, coercivity_estimator=None,
                              aFine_Constructor=None, store_in_tmp=False,
                              reductor_type='coercive', print_on_ranks=True):
    """
    prepare patch models with parameterized right hand side for TSRBLOD
    """
    if print_on_ranks:
        print("c", end="", flush=True)
    if store_in_tmp is not False:
        dir = store_in_tmp if isinstance(store_in_tmp, str) else 'tmp'

    if aPatch is None:
        if store_in_tmp:
            dbfile = open(f'{dir}/apatch_{patch.TInd}', "rb")
            aPatch = dill.load(dbfile)
            dbfile.close()
        else:
            assert aFine_Constructor is not None
            aPatch = aFine_Constructor(patch)

    m = CorrectorProblem_for_all_rhs(patch, aPatch, aFineCoefficients)
    if reductor_type == 'simple_coercive':
        reductor = SimpleCoerciveRBReductor(m, product=m.products["h1"], coercivity_estimator=coercivity_estimator)
    elif reductor_type == 'coercive':
        reductor = CoerciveRBReductor(m, product=m.products["h1"], coercivity_estimator=coercivity_estimator)
    elif reductor_type == 'non_assembled':
        reductor = NonAssembledRBReductor(m, product=m.products["h1"], coercivity_estimator=coercivity_estimator)
    else:
        assert 0, 'reductor type not known'

    rom = reductor.reduce()
    optimized_rom, _ = OptimizedNumpyModelStage1(rom, m.Kij, patch.TInd).minimal_object()
    rom_size = rom.solution_space.dim

    if store_in_tmp is not False:
        dir = store_in_tmp if isinstance(store_in_tmp, str) else 'tmp'
        if os.path.exists(f'{dir}/red_{patch.TInd}'):
            os.remove(f'{dir}/red_{patch.TInd}')
        dbfile = open(f'{dir}/red_{patch.TInd}', "ab")
        dill.dump(reductor, dbfile)
        if os.path.exists(f'{dir}/rom_{patch.TInd}'):
            os.remove(f'{dir}/rom_{patch.TInd}')
        dbfile = open(f'{dir}/rom_{patch.TInd}', "ab")
        dill.dump(optimized_rom, dbfile)
        del reductor, rom
        return None, None, m.Kij, rom_size
    else:
        return optimized_rom, reductor, m.Kij, rom_size


from rblod.parameterized_stage_1 import _build_directional_mus
def extend_patch(reductor, cors, mu, gridlod_model, print_on_ranks=True, add_error_residual=True):
    sol_space = reductor.fom.solution_space
    if cors is None:
        Kmsij, cors = gridlod_model.compute_FOM_corrector(reductor.fom.patch, None, True, mu)
    else:
        Kmsij = None

    cors = [sol_space.from_numpy(cor) for cor in cors]
    for cor, mu_dir in zip(cors, _build_directional_mus(mu)):
        try:
            reductor.extend_basis(cor)
            if print_on_ranks:
                print("e", end="", flush=True)
        except:
            if print_on_ranks:
                print("E", end="", flush=True)
            pass

    rom = reductor.reduce()
    optimized_rom = OptimizedNumpyModelStage1(rom, reductor.fom.Kij, reductor.fom.patch.TInd)
    optimized_rom, error_residual = optimized_rom.minimal_object(add_error_residual=add_error_residual)
    rom_size = rom.solution_space.dim
    return Kmsij, optimized_rom, reductor, rom_size, error_residual

def extend_patch_adaptively(rom, reductor, cors, mu, gridlod_model, print_on_ranks=True, add_error_residual=True):
    sol_space = reductor.fom.solution_space
    if cors is None:
        Kmsij, cors = gridlod_model.compute_FOM_corrector(reductor.fom.patch, None, True, mu)
    else:
        Kmsij = None

    cors = [sol_space.from_numpy(cor) for cor in cors]
    for cor, mu_dir in zip(cors, _build_directional_mus(mu)):
        if rom.estimate_error(mu_dir) > 0.001:
            try:
                reductor.extend_basis(cor)
                if print_on_ranks:
                    print("e", end="", flush=True)
            except:
                if print_on_ranks:
                    print("E", end="", flush=True)
                pass
        else:
            if print_on_ranks:
                print("E", end="", flush=True)

    rom = reductor.reduce()
    optimized_rom = OptimizedNumpyModelStage1(rom, reductor.fom.Kij, reductor.fom.patch.TInd)
    optimized_rom, error_residual = optimized_rom.minimal_object(add_error_residual=add_error_residual)
    rom_size = rom.solution_space.dim
    return Kmsij, optimized_rom, reductor, rom_size, error_residual


def extend_patch_cached_reductor(patch, cors, mu, store_in_tmp, gridlod_model, print_on_ranks=True,
                                 add_error_residual=True):
    dir = store_in_tmp if isinstance(store_in_tmp, str) else 'tmp'
    assert os.path.exists(f'{dir}/')
    T = patch.TInd
    dbfile = open(f'{dir}/red_{T}', "rb")
    reductor = dill.load(dbfile)
    dbfile.close()
    Kmsij, rom, reductor, rom_size, error_residual = extend_patch(
        reductor, cors, mu, gridlod_model, print_on_ranks, add_error_residual)
    if error_residual is not None:
        if os.path.exists(f'{dir}/err_res_{T}'):
            os.remove(f'{dir}/err_res_{T}')
        dbfile = open(f'{dir}/err_res_{T}', "ab")
        dill.dump(error_residual, dbfile)
    os.remove(f'{dir}/red_{patch.TInd}')
    dbfile = open(f'{dir}/red_{T}', "ab")
    dill.dump(reductor, dbfile)
    if os.path.exists(f'{dir}/rom_{T}'):
        os.remove(f'{dir}/rom_{T}')
    dbfile = open(f'{dir}/rom_{T}', "ab")
    dill.dump(rom, dbfile)
    return Kmsij, rom_size


def extend_patch_adaptively_cached_reductor(patch, rom, cors, mu, store_in_tmp, gridlod_model, print_on_ranks=True,
                                            add_error_residual=True):
    dir = store_in_tmp if isinstance(store_in_tmp, str) else 'tmp'
    assert os.path.exists(f'{dir}/')
    T = patch.TInd
    dbfile = open(f'{dir}/red_{T}', "rb")
    reductor = dill.load(dbfile)
    dbfile.close()
    dbfile = open(f'{dir}/rom_{T}', "rb")
    rom = dill.load(dbfile)
    dbfile.close()
    Kmsij, rom, reductor, rom_size, error_residual = extend_patch_adaptively(
        rom, reductor, cors, mu, gridlod_model, print_on_ranks, add_error_residual)
    if error_residual is not None:
        if os.path.exists(f'{dir}/err_res_{T}'):
            os.remove(f'{dir}/err_res_{T}')
        dbfile = open(f'{dir}/err_res_{T}', "ab")
        dill.dump(error_residual, dbfile)
    os.remove(f'{dir}/red_{T}')
    dbfile = open(f'{dir}/red_{T}', "ab")
    dill.dump(reductor, dbfile)
    os.remove(f'{dir}/rom_{T}')
    dbfile = open(f'{dir}/rom_{T}', "ab")
    dill.dump(rom, dbfile)
    return Kmsij, rom_size


def compute_patch_errors(rom, mu, **kwargs):
    errors = [rom.estimate_error(mu_dir, **kwargs) for mu_dir in _build_directional_mus(mu)]
    return errors


def compute_contrast_for_all_patches(aPatches, T, aFineCoefficients, training_set, store_in_tmp=False):
    if aPatches is None:
        assert store_in_tmp
        dir = store_in_tmp if isinstance(store_in_tmp, str) else 'tmp'
        dbfile = open(f'{dir}/apatch_{T}', "rb")
        aPatches = dill.load(dbfile)
        dbfile.close()

    contrast, min_alpha = compute_constrast(aPatches, aFineCoefficients, training_set)
    return contrast, min_alpha


# NOTE: THIS IS COPIED FROM RBLOD.scripts
def compute_constrast(aFines, aFineCoefficients, training_set):
    max_contrast = 1
    min_alpha = 10000
    for mu in training_set:
        a = _construct_aFine_from_mu(aFines, aFineCoefficients, mu)
        if a.ndim == 3:
            a = np.linalg.norm(a, axis=(1, 2), ord=2)
        max_contrast = max(max_contrast, np.max(a) / np.min(a))
        min_alpha = min(min_alpha, np.min(a))
    return max_contrast, min_alpha

# NOTE: THIS IS COPIED FROM RBLOD.scripts
def _construct_aFine_from_mu(aFines, aFinesCoefficients, mu):
    coefs = [c.evaluate(mu) if hasattr(c, 'evaluate') else c for c in aFinesCoefficients]
    dim_array = aFines[0].ndim
    if dim_array == 3:
        a = np.einsum('ijkl,i', aFines, coefs)
    elif dim_array == 1:
        a = np.einsum('ij,i', aFines, coefs)
    return a

def build_box_parameter_space(parameter_space, center, radius):
    ranges = {}
    for (component, index) in parameter_space.parameters.items():
        ranges[component] = [center[component][()] - radius, center[component][()] + radius]
        range_ = parameter_space.ranges[component]
        if ranges[component][0] < range_[0]:
            ranges[component][0] = range_[0]
        if ranges[component][0] > range_[1]:
            ranges[component][1] = range_[1]
    box_parameter_space = parameter_space.with_(ranges=ranges)
    return box_parameter_space