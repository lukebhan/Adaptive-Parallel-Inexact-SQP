"""NE39 ten-generator Swing dynamics with optional stage-wise inertia modes.

States are angles and frequencies; controls act on frequencies.
Dynamics and analytic derivatives read schedule[k], falling back to params.mode."""

import numpy as np
from . import burgers_setting as bs
from .eval_stats import bump

# Kron-reduced NE39 parameters (round-trip-exact, 10 generator buses)
_M_BASE = [
    1.4000000000000001,
    1.01,
    1.1766666666666667,
    0.9533333333333335,
    0.8666666666666667,
    1.16,
    0.88,
    0.81,
    1.1500000000000001,
    16.666666666666668,
]
_D_DAMP = [
    0.1966666666666667,
    0.2883333333333334,
    0.2883333333333334,
    0.2883333333333334,
    0.2883333333333334,
    0.2883333333333334,
    0.2883333333333334,
    0.2883333333333334,
    0.30366666666666664,
    0.30366666666666664,
]
_PM = [
    -0.19983394,
    -0.25653884,
    -0.25191885,
    -0.10242008,
    -0.34510365,
    0.23206371,
    0.4404325,
    0.5896664,
    0.26257738,
    -0.36892462,
]
_COEF_COST = [
    0.94162537,
    1.15464676,
    0.55487802,
    0.71176656,
    1.27476119,
    1.02529959,
    1.2946042,
    1.01800112,
    0.99465694,
    0.73467357,
]
_K = [
    [
        0.0,
        1.138716399958298,
        1.3606040523384149,
        1.2791332133302582,
        0.5725319952852771,
        1.291387170652045,
        1.0516770291704949,
        3.175070187048039,
        1.7979613231347757,
        5.194975542217739,
    ],
    [
        1.138716399958298,
        0.0,
        2.2804016761212016,
        0.7847815847175774,
        0.3512633100908221,
        0.7922997071037642,
        0.6452312839369623,
        0.6032208817937575,
        0.5339053141746971,
        3.185137357587494,
    ],
    [
        1.3606040523384144,
        2.2804016761212016,
        0.0,
        1.037937440292858,
        0.46457428161450903,
        1.0478807682930118,
        0.8533708235361006,
        0.7284145883664191,
        0.6661286587302977,
        3.2071328685623444,
    ],
    [
        1.279133213330258,
        0.7847815847175768,
        1.0379374402928578,
        0.0,
        2.746518496384744,
        2.0251245471482795,
        1.6492164517708374,
        0.7610195437562253,
        0.9034713378882271,
        1.620980584471724,
    ],
    [
        0.5725319952852769,
        0.35126331009082196,
        0.46457428161450887,
        2.7465184963847458,
        0.0,
        0.9064330326169404,
        0.7381789292542378,
        0.34062756974581654,
        0.4043880984197797,
        0.7255407323292409,
    ],
    [
        1.291387170652045,
        0.7922997071037641,
        1.0478807682930118,
        2.025124547148281,
        0.9064330326169407,
        0.0,
        4.124239303551531,
        0.768310028369595,
        0.9121264952249022,
        1.6365094024983056,
    ],
    [
        1.0516770291704955,
        0.6452312839369625,
        0.8533708235361013,
        1.6492164517708392,
        0.7381789292542382,
        4.124239303551531,
        0.0,
        0.6256946224033295,
        0.7428155587463924,
        1.3327369093809385,
    ],
    [
        3.1750701870480396,
        0.6032208817937577,
        0.7284145883664191,
        0.7610195437562258,
        0.34062756974581676,
        0.7683100283695955,
        0.6256946224033293,
        0.0,
        1.4515272867928857,
        2.5534966066757634,
    ],
    [
        1.7979613231347755,
        0.5339053141746972,
        0.6661286587302977,
        0.9034713378882274,
        0.4043880984197798,
        0.9121264952249024,
        0.742815558746392,
        1.4515272867928852,
        0.0,
        1.6531544899612127,
    ],
    [
        5.19497554221774,
        3.1851373575874957,
        3.207132868562346,
        1.620980584471725,
        0.7255407323292412,
        1.6365094024983065,
        1.332736909380938,
        2.5534966066757634,
        1.6531544899612136,
        0.0,
    ],
]
N_BUS = 10
INERTIA_SCALES = (0.3, 1.0, 5.0)  # mode q in {0,1,2}
TWO_PI = 2.0 * np.pi


def make_schedule(N, switch_every, seq=(2, 1, 1, 0)):
    """Build a length-N inertia schedule by cycling modes every switch_every stages."""
    q = np.empty(N, dtype=int)
    for k in range(N):
        q[k] = seq[(k // switch_every) % len(seq)]
    return q


class SwingParams:
    """Problem data for the NE39 swing OCP, with a (possibly switching) inertia
    schedule. `mode` is the fallback mode when no schedule / no stage index."""

    def __init__(self, mode=1, dt=0.01, schedule=None, pm_scale=1.0):
        self.n = N_BUS
        self.dt = float(dt)
        self.mode = int(mode)
        self.M_base = np.asarray(_M_BASE, float)
        self.D = np.asarray(_D_DAMP, float)
        # pm_scale=0 makes the origin an equilibrium; Jacobians do not depend on Pm.
        self.Pm = pm_scale * np.asarray(_PM, float)
        self.K = np.asarray(_K, float)
        # per-mode inverse inertia (length-n each); schedule selects per stage.
        self.Minv_modes = [1.0 / (s * self.M_base) for s in INERTIA_SCALES]
        self.Minv = self.Minv_modes[self.mode]  # single-mode fallback
        self.schedule = None if schedule is None else np.asarray(schedule, int)
        # Laplacian of the linearized model (== dg/dtheta at theta*=0).
        self.L = -self.K.copy()
        np.fill_diagonal(self.L, self.K.sum(axis=1))


def make_swing_step_fns(dt, p: SwingParams):
    """Return stage-indexed (F, A, B, Hdyn) closures. Each takes an optional `k`
    and uses inertia mode `p.schedule[k]` (or `p.mode` when k is None / no schedule)."""
    n = p.n
    dt = float(dt)
    K = p.K
    D = p.D
    Pm = p.Pm
    Minv_modes = p.Minv_modes
    sched = p.schedule
    default_mode = p.mode
    # per-mode control Jacobian (control enters omega only; -u convention)
    Bconst_modes = []
    for Minv in Minv_modes:
        Bm = np.zeros((2 * n, n))
        Bm[n:, :] = -dt * np.diag(Minv)
        Bconst_modes.append(Bm)

    def _mode(k):
        if sched is None or k is None:
            return default_mode
        return int(sched[k])

    def _split(x):
        return x[:n], x[n:]

    def F(x, w, k=None):
        bump("F")
        Minv = Minv_modes[_mode(k)]
        theta, omega = _split(x)
        S = theta[:, None] - theta[None, :]
        g = np.einsum("ij,ij->i", K, np.sin(S))  # g_i = sum_j K_ij sin(theta_i-theta_j)
        theta_n = theta + dt * TWO_PI * omega
        omega_n = omega + dt * Minv * (Pm - D * omega - w - g)
        return np.concatenate([theta_n, omega_n])

    def A(x, w, k=None):
        bump("jac_x")
        Minv = Minv_modes[_mode(k)]
        theta, _ = _split(x)
        C = K * np.cos(theta[:, None] - theta[None, :])
        H = -C.copy()
        np.fill_diagonal(H, C.sum(axis=1))  # dg/dtheta (weighted Laplacian)
        Ax = np.zeros((2 * n, 2 * n))
        Ax[:n, :n] = np.eye(n)
        Ax[:n, n:] = dt * TWO_PI * np.eye(n)
        Ax[n:, :n] = -dt * (Minv[:, None] * H)
        Ax[n:, n:] = np.eye(n) - dt * np.diag(Minv * D)
        return Ax

    def B(x, w, k=None):
        bump("jac_u")
        return Bconst_modes[_mode(k)]

    def Hdyn(x, u, lam, k=None):
        # Dynamics curvature affects only angle blocks; close diagonal entries by row sums.
        Minv = Minv_modes[_mode(k)]
        theta, _ = _split(x)
        wt = Minv * lam[n:]  # per-generator weight
        S = np.sin(theta[:, None] - theta[None, :])  # antisymmetric
        Wt = K * S * (wt[:, None] - wt[None, :])  # off-diagonal part
        np.fill_diagonal(Wt, 0.0)
        W = Wt + np.diag(-Wt.sum(axis=1))
        Hb = np.zeros((2 * n + n, 2 * n + n))
        Hb[:n, :n] = dt * W
        return Hb

    return F, A, B, Hdyn


def swing_x0(p: SwingParams, kick=0.25):
    """A deterministic post-disturbance IC: spread angles + a frequency kick."""
    n = p.n
    idx = np.arange(n)
    theta0 = kick * np.sin(2 * np.pi * idx / n)
    omega0 = 0.4 * kick * np.cos(2 * np.pi * idx / n)
    return np.concatenate([theta0, omega0])


def make_swing_problem(
    N,
    dt=0.1,
    mode=1,
    q_theta=10.0,
    q_omega=10.0,
    r_scale=0.05,
    x0=None,
    kick=0.25,
    schedule=None,
    switch_every=None,
    mode_seq=(2, 1, 1, 0),
    pm_scale=1.0,
):
    """Build a Swing OCP with zero target and optional inertia switching.

    Sets global state/control dimensions to 2n/n. Terminal weight equals Q."""
    from .ComputeKKT import Problem  # local import: avoid import cycle

    bs.set_dims(2 * N_BUS, nu=N_BUS)
    if schedule is None and switch_every is not None:
        schedule = make_schedule(N, switch_every, mode_seq)
    p = SwingParams(mode=mode, dt=dt, schedule=schedule, pm_scale=pm_scale)
    n = p.n
    Q = np.diag(np.concatenate([q_theta * np.ones(n), q_omega * np.ones(n)]))
    R = r_scale * np.diag(np.asarray(_COEF_COST, float))
    QN = Q.copy()
    if x0 is None:
        x0 = swing_x0(p, kick=kick)
    x_des = np.zeros((N + 1, 2 * n))  # regulate to the origin
    step_fns = make_swing_step_fns(dt, p)
    return Problem(N, dt, Q, R, QN, x_des, x0, p, step_fns=step_fns)
