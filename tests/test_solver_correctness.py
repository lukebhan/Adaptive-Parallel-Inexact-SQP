"""Independent derivative checks and adversarial residual/gate regressions.

Run: PYTHONPATH=src python -m unittest discover -s tests -v
"""

import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import importlib
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

import newton as N
from newton.AOTDsolver import solve
from newton.baselines.common import initial_guess, pack_traj
from newton.ComputeKKT import hessian_blocks, modify_hessian_blocks

AOTD = importlib.import_module('newton.AOTD')


def initial_case():
    problem = N.make_swing_problem(8, mode=1, pm_scale=0.0)
    x, u = initial_guess(problem)
    return problem, pack_traj(problem, x, u), np.zeros(N.n_lam(problem))


class MeritDerivativeTests(unittest.TestCase):
    def test_exact_merit_gradient_with_nonzero_dynamics_curvature(self):
        problem, z, lam = initial_case()
        rng = np.random.default_rng(18)
        z += rng.normal(0, .2, len(z))
        lam = rng.normal(0, 2, len(lam))
        d = rng.normal(size=len(z) + len(lam))
        d /= np.linalg.norm(d)
        eta1, eta2 = 2.0, .5
        gradient = N.grad_merit_flat(problem, z, lam, eta1, eta2)
        eps = 1e-5
        nz = len(z)
        fd = (N.merit(problem, z+eps*d[:nz], lam+eps*d[nz:], eta1, eta2)
              - N.merit(problem, z-eps*d[:nz], lam-eps*d[nz:], eta1, eta2)) / (2*eps)
        self.assertAlmostEqual(float(gradient @ d), fd, delta=2e-6*max(1, abs(fd)))
        gn = sp.block_diag(hessian_blocks(problem, z, lam, gauss_newton=True))
        wrong = N.grad_merit_flat(problem, z, lam, eta1, eta2, H=gn)
        self.assertGreater(abs(float((wrong-gradient) @ d)), 1e-4)

    def test_algorithm_uses_true_merit_derivative_for_both_hessian_models(self):
        for gauss_newton in (False, True):
            with self.subTest(gauss_newton=gauss_newton):
                problem, z, lam = initial_case()
                lam = np.random.default_rng(12).normal(0, 2, len(lam))
                original_armijo = AOTD.armijo
                calls = []

                def checked_armijo(p, x, multipliers, direction, eta1, eta2, beta, **kw):
                    eps = 1e-5 / max(1, np.linalg.norm(direction))
                    nz = len(x)
                    fd = (N.merit(p, x+eps*direction[:nz], multipliers+eps*direction[nz:], eta1, eta2)
                          - N.merit(p, x-eps*direction[:nz], multipliers-eps*direction[nz:], eta1, eta2)) / (2*eps)
                    self.assertAlmostEqual(kw['current_grad_dot_dir'], fd,
                                           delta=2e-5*max(1, abs(fd)))
                    calls.append(fd)
                    return original_armijo(p, x, multipliers, direction, eta1, eta2, beta, **kw)

                cfg = N.AlgorithmConfig(M=1, b0=8, adaptive=False, inner_solver='direct',
                    use_preconditioner=False, eps_i_0=1e-9, gauss_newton=gauss_newton,
                    xi_H=10.0, max_outer_iters=1)
                with patch.object(AOTD, 'armijo', side_effect=checked_armijo):
                    result = N.run_algorithm(problem, cfg, z, lam)
                self.assertEqual(len(calls), 1)
                self.assertGreater(result['trajectory'][0]['hessian_shift_max'], 0)

    def test_stagewise_shift_preserves_input_and_enforces_floor(self):
        block = np.array([[-2.0, .3], [.3, 1.0]])
        original = block.copy()
        modified, shifts, count, _ = modify_hessian_blocks([block], .1)
        np.testing.assert_array_equal(block, original)
        self.assertEqual(count, 1)
        self.assertGreaterEqual(np.linalg.eigvalsh(modified[0]).min(), .1-1e-14)
        np.testing.assert_allclose(modified[0]-block, shifts[0]*np.eye(2))


class LocalResidualTests(unittest.TestCase):
    def test_preconditioned_warm_start_false_positive(self):
        matrix = sp.diags([1.0, 2.0], format='csr')
        rhs = np.ones(2)
        warm = np.array([1.0, 0.0])
        precondition = lambda v: np.array([1.0, 1e-8])*v
        # This start satisfies the former preconditioned criterion, but violates (13).
        self.assertLess(np.linalg.norm(precondition(matrix@warm-rhs)) /
                        np.linalg.norm(precondition(rhs)), 1e-3)
        for method in ('gmres_qlp', 'sketch'):
            with self.subTest(method=method):
                d, info = solve(method, matrix, rhs, tol=1e-3, max_iters=50,
                                M=precondition, x0=warm, restart=2)
                self.assertTrue(info['converged'], info)
                self.assertGreater(info['iters'], 0)
                self.assertLessEqual(np.linalg.norm(matrix@d-rhs), 1e-3*np.linalg.norm(rhs))

    def test_cold_start_cannot_stop_on_preconditioned_residual(self):
        matrix = sp.diags(np.linspace(1, 2, 20), format='csr')
        rhs = np.ones(20)
        scale = np.ones(20)
        scale[0] = 1e-8
        d, info = solve('gmres_qlp', matrix, rhs, tol=1e-3, max_iters=30,
                        M=lambda v: scale*v, rank_tol=1e-14)
        self.assertTrue(info['converged'], info)
        self.assertGreater(info['iters'], 5)
        self.assertLessEqual(np.linalg.norm(matrix@d-rhs), 1e-3*np.linalg.norm(rhs))

    def test_exhaustion_is_not_success(self):
        matrix = sp.diags(np.arange(1.0, 11), format='csr')
        rhs = np.ones(10)
        for method in ('gmres_qlp', 'sketch'):
            with self.subTest(method=method):
                d, info = solve(method, matrix, rhs, tol=1e-12, max_iters=1)
                self.assertFalse(info['converged'])
                self.assertTrue(info['cap_hit'])
                self.assertGreater(np.linalg.norm(matrix@d-rhs), 1e-12*np.linalg.norm(rhs))

    def test_zero_rhs_discards_inaccurate_warm_start(self):
        for method in ('gmres_qlp', 'sketch'):
            d, info = solve(method, sp.eye(3, format='csr'), np.zeros(3),
                            tol=1e-3, max_iters=10, x0=np.ones(3))
            np.testing.assert_array_equal(d, np.zeros(3))
            self.assertTrue(info['converged'])
            self.assertEqual(info['residual_norm'], 0)

    def test_residual_certification_matvecs_are_counted(self):
        class CountedMatrix:
            def __init__(self):
                self.matrix = sp.diags([1., 2., 3.], format='csr')
                self.nnz = self.matrix.nnz
                self.calls = 0

            def __matmul__(self, value):
                self.calls += 1
                return self.matrix @ value

        for method in ('gmres_qlp', 'sketch'):
            matrix = CountedMatrix()
            _, info = solve(method, matrix, np.ones(3), tol=1e-5, max_iters=30, restart=3)
            self.assertEqual(matrix.calls, info['matvecs'])


class AcceptanceTests(unittest.TestCase):
    def test_local_failure_does_not_reach_line_search(self):
        problem, z, lam = initial_case()
        cfg = N.AlgorithmConfig(M=2, b0=1, inner_solver='gmres_qlp',
            use_preconditioner=False, eps_i_0=1e-12, max_inner_iters=1, max_outer_iters=2)
        with patch.object(AOTD, 'armijo') as line_search:
            result = N.run_algorithm(problem, cfg, z, lam)
        line_search.assert_not_called()
        self.assertEqual(result['stop_reason'], 'local_accuracy')
        np.testing.assert_array_equal(result['z'], z)
        np.testing.assert_array_equal(result['lam'], lam)
        self.assertFalse(result['trajectory'][0]['step_applied'])

    def test_global_failure_does_not_reach_line_search(self):
        problem, z, lam = initial_case()
        cfg = N.AlgorithmConfig(M=2, b0=0, max_overlap=1, inner_solver='direct',
            use_preconditioner=False, eps0=1e-20, max_inner_passes=1, max_outer_iters=2)
        with patch.object(AOTD, 'armijo') as line_search:
            result = N.run_algorithm(problem, cfg, z, lam)
        line_search.assert_not_called()
        self.assertEqual(result['stop_reason'], 'accuracy_budget')
        np.testing.assert_array_equal(result['z'], z)
        np.testing.assert_array_equal(result['lam'], lam)

    def test_line_search_failure_does_not_apply_last_trial(self):
        problem, z, lam = initial_case()
        cfg = N.AlgorithmConfig(M=1, b0=8, adaptive=False, inner_solver='direct',
            use_preconditioner=False, eps_i_0=1e-9, max_outer_iters=1)
        with patch.object(AOTD, 'armijo', return_value=(1e-6, 30, False)):
            result = N.run_algorithm(problem, cfg, z, lam)
        self.assertEqual(result['stop_reason'], 'line_search')
        np.testing.assert_array_equal(result['z'], z)
        np.testing.assert_array_equal(result['lam'], lam)


if __name__ == '__main__':
    unittest.main()
