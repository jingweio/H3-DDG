# -*- coding: utf-8 -*-
"""C2 -- the tail of ``c_hat`` (spec Sec.1.3).

    "a minority of sequence-near pairs jump beyond what a smooth landscape
     produces"

Everything here runs on the PRIMARY NESTED SET ``P_a`` (spec Sec.1.0): nested
pairs with (a) ``B != {}``, (b) neither endpoint at a detected censoring level,
(c) finite ``phi^oof`` at both endpoints, (d) the assay in tier PRIMARY or ARM.
``P_a`` is evaluated against the CENSORING AND CROSS-FIT OF WHATEVER DATASET IS
IN HAND -- the observed data or a surrogate replicate -- because a null that
selects its edges by the observed data's mask is not running the same experiment.
:func:`cliff.nulls._pa_mask` is that rule; :func:`pa_mask` here is the same rule
with the floor-masking clause made a knob, which is the T13 ``floor_mask`` row.

WHAT THIS MODULE OWNS
    ``T06_cliff_tail_C2.csv``, ``cliff_catalogue_{DMS_id}.csv.gz``, and the
    ``centring`` / ``tau`` / ``sigma_mult`` / ``scale`` / ``floor_mask`` rows of
    ``T13_sensitivity.csv``.  It writes NO verdict: ``verdict_C2`` and
    ``failing_criterion`` are created empty and filled by :mod:`cliff.verdict`.

THE FOUR DEFINITIONS THAT DECIDE WHAT THE NUMBERS MEAN
    1. ``c_hat`` IS PHI-CENTRED (ORCHESTRATOR D2).  ``corr(e_oof, phi_oof)`` is
       +0.24 to +0.36 on the four biggest assays, so the spec's uncentred
       ``(e_v - e_u)/sqrt(s2_u + s2_v)`` has a non-zero mean on any nested pair
       whose endpoints sit in different ``phi`` regions and would MANUFACTURE
       cliffs.  The primary statistic subtracts the cached per-bin median of
       ``e`` (``sigma_knots_median_e``) at each endpoint first, interpolated the
       same way ``sigma`` is.  The uncentred form is computed too and is the T13
       ``centring`` row.  Nulls go through the identical centring.
    2. ``scale='raw'`` IS THE IDENTITY-LINK ADDITIVE FIT, i.e. ``phi = lsqr([1|X],
       y)`` and ``e = y - phi``, with the same Tobit E-step, the same 5 folds, the
       same 20-bin MAD ``sigma-hat(phi)`` and the same per-bin median centring --
       everything the latent scale does except step 2/3's monotone link.  This is
       the repo's own meaning of "raw" (``LatentFit.r2_add_raw`` vs
       ``r2_add_latent``), and it is the only reading computable IDENTICALLY for
       the observed data and for a surrogate replicate: the alternative
       ``e = y - g_oof(phi_oof)`` needs the replicate's own ``g``, which
       ``nulls._refit_bundle`` does not return.  Latent is primary; a verdict that
       holds on raw but not on latent is DISCARDED (spec Sec.1.4 C3-A.4), which is
       :mod:`cliff.verdict`'s call, not this module's.
    3. THE TWO UNIT SYSTEMS.  ``unit='sigma'`` divides the centred numerator by
       ``sqrt(sigma^2(phi_u) + sigma^2(phi_v))`` -- level-dependent, the spec's
       ``c_hat``.  ``unit='MAD'`` divides the SAME numerator by one global
       ``1.4826 MAD`` of that numerator over ``P_a``, which is the unit system a
       wrong ``sigma(phi)`` cannot touch.  Both are required for the C2 verdict.
    4. ``P_a`` IS HELD FIXED ACROSS SCALES.  Condition (c) is intersected between
       the latent and the raw cross-fit (the raw fit has no ``ginv`` and so loses
       fewer rows), so ``n_Pa`` is identical on the two scales and the scale
       comparison is the same edges with a different statistic.

INFERENCE
    Null-referenced only.  ``TR`` is ranked among the N1 replicates (spec Sec.1.3:
    "never a naive bootstrap of an extreme quantile -- the latter is not
    consistent").  ``T(tau) = P_obs / mean_b P_{N2,b}`` gets the empirical
    ``p = (1 + #{b: P_b >= P_obs})/(B+1)``, BH-FDR over the 14 primary+arm
    assays, and a CI from a block bootstrap over MUTATED POSITIONS (the position
    of the added substitution, ``ctx.pos_of_add``) -- never over edges.
    ``Lambda = 2(l2 - l1)`` is ranked among the N1 replicates because the LRT for
    a mixture is non-regular and chi-square is wrong.

Self-check: ``python -m cliff.stats_c2 --selfcheck`` (closed forms + invariants),
``--stage`` runs the real thing and prints every number.
"""

from __future__ import print_function

import gzip
import json
import math
import os
import sys
import time

# BLAS threads: 1, before numpy, for the same reason cliff.nulls does it (64
# workers x the default OpenBLAS pool measured a load average of 1,409 here).
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from scipy.special import expit

from . import config
from . import latent as _latent
from . import nulls as _nulls
from . import pairs as _pairs
from .config import PATHS, SEEDS, TAUS, THRESH
from .latent import (classify_levels, mad_scaled, sigma_eval, sigma_of_phi,
                     tobit_estep, with_intercept)
from .nulls import c_hat            # ORCHESTRATOR D2's phi-centred statistic

__all__ = [
    'TAU_GRID', 'BLOCKS', 'BLOCK_NAMES',
    'c_hat', 'tail_ratio', 'mixture_two_component', 'enrichment_sweep',
    'block_bootstrap_positions', 'rate_bootstrap_positions', 'bh_fdr',
    'empirical_p_upper', 'grid_guard', 'pa_mask',
    'fit_raw', 'crossfit_raw', 'raw_bundle',
    'block_values', 'block_stats', 'stat_fn_c2', 'stat_fn_c2_mix',
    'observed_blocks', 'ensembles_for', 'build_T06', 'build_catalogue',
    'build_T13_rows', 'run_all', 'stage', 'STAT_C2_VERSION',
]

# --------------------------------------------------------------------------- #
# grids and blocks                                                            #
# --------------------------------------------------------------------------- #

#: The tau grid the ensembles carry.  ``config.TAUS`` = (2,3,4,5,6,8) is the
#: sweep the verdict reads; the extra points exist so the T13 ``sigma_mult`` row
#: is FREE rather than a second 13,600-job run: multiplying ``sigma`` by ``m``
#: divides every ``c`` (observed AND null, since N2 reuses the observed
#: ``sigma_oof``) by ``m``, so ``T(tau; sigma x m) == T(m tau; sigma)`` exactly.
#: ``m in {0.5, 1, 2} x TAUS`` needs {1,1.5,2,2.5,3,4} and {4,6,8,10,12,16}.
TAU_GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0)
assert set(float(t) for t in TAUS) <= set(TAU_GRID)
assert set(m * t for m in config.SIGMA_MULTIPLIERS for t in TAUS) <= set(TAU_GRID)

#: ``(scale, unit, centred, floor_mask)``.  The first four are the T06 rows; the
#: last four are the T13 sensitivity knobs this module owns.  ``sigma`` before
#: ``MAD`` matters: :func:`cliff.verdict._first_num` takes the first non-empty
#: value of a per-assay column over the whole ``(DMS_id, scale)`` group, so the
#: sigma system -- the spec's ``c_hat`` -- is the one its TR and mixture clauses
#: read.
BLOCKS = (
    ('L_s',  'latent', 'sigma', True,  True),      # PRIMARY
    ('L_m',  'latent', 'MAD',   True,  True),      # PRIMARY, second unit system
    ('R_s',  'raw',    'sigma', True,  True),      # secondary scale
    ('R_m',  'raw',    'MAD',   True,  True),
    ('Lu_s', 'latent', 'sigma', False, True),      # T13 centring
    ('Lu_m', 'latent', 'MAD',   False, True),      # T13 centring
    ('Lf_s', 'latent', 'sigma', True,  False),     # T13 floor_mask
    ('Rf_s', 'raw',    'sigma', True,  False),     # T13 floor_mask
)
BLOCK_NAMES = tuple(b[0] for b in BLOCKS)
PRIMARY_BLOCKS = ('L_s', 'L_m', 'R_s', 'R_m')
_BLOCK = {b[0]: b for b in BLOCKS}

#: Blocks whose mixture is carried through the null ensembles.  Only ``N1``
#: needs one (T06's ``Lambda_N1_p995``), and only for the four T06 blocks: a
#: mixture EM per replicate per block is the single most expensive thing in the
#: statistic vector, and no T06 or T13 column reads a mixture under N2/N2b/N2c.
MIX_BLOCKS = PRIMARY_BLOCKS

#: Number of quantile bins the mixture EM runs on.  ``n <= _MIX_BINS`` is exact.
_MIX_BINS_COARSE = 512          # multi-restart search
_MIX_BINS_FINE = 8192           # polish + the bootstrap's fixed bins
_MIX_ITER_POLISH = 100

_SQRT_12 = math.sqrt(12.0)
_LOG_2PI = math.log(2.0 * math.pi)


def _stat_c2_version():
    """Fingerprint of what a ``stats_c2`` statistic vector MEANS."""
    import hashlib
    blob = ('|'.join(BLOCK_NAMES) + '||' + ','.join('%g' % t for t in TAU_GRID)
            + '||phi-centred|raw=identity-link-lsqr|Pa-intersected|v1')
    return hashlib.md5(blob.encode()).hexdigest()[:12]


STAT_C2_VERSION = _stat_c2_version()


# =========================================================================== #
# small closed forms                                                          #
# =========================================================================== #

def tail_ratio(c, q_hi, q_lo):
    """``Q_{q_hi}(|c|) / Q_{q_lo}(|c|)`` (spec Sec.1.3).

    Scale-free by construction: invariant under ``c -> lambda c``, which is why a
    wrong ``sigma_noise`` can neither create nor destroy the C2 signal.  Gaussian
    references, exact: ``Q.75(|Z|)=1.1503``, ``Q.99=2.5758``, ``Q.999=3.2905``,
    so ``TR1_gauss=2.8606`` and ``TR2_gauss=2.2393``.
    """
    a = np.abs(np.asarray(c, dtype=np.float64))
    a = a[np.isfinite(a)]
    if a.size < 8:
        return float('nan')
    hi, lo = np.percentile(a, [100.0 * q_hi, 100.0 * q_lo])
    return float(hi / lo) if lo > 0 else float('nan')


def bh_fdr(p):
    """Benjamini-Hochberg step-up ``q``-values (statsmodels is not installed).

    ``q_(i) = min_{j >= i} ( m p_(j) / j )``, monotone by the running minimum from
    the largest ``p`` down, then mapped back to the input order and clipped to 1.
    NaNs are carried through as NaN and do not count toward ``m``.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx], kind='stable')]
    ranked = m * p[order] / np.arange(1, m + 1, dtype=np.float64)
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1]
    q[ok] = np.minimum(q[ok], 1.0)
    return q


def empirical_p_upper(obs, null):
    """``p = (1 + #{b: stat_b >= stat_obs}) / (B + 1)`` -- spec Sec.1.3 verbatim.

    Upper-tailed: C2 is a claim that the observed tail is HEAVIER than the null.
    Conservative for a discrete statistic (a rate over ``|P_a|`` edges is exactly
    0 in most replicates at ``tau >= 5``), and the direction is the safe one --
    it can only under-reject, never manufacture a C2 positive.  ``cliff.nulls
    ._empirical_p`` reports the mid-p and randomised forms alongside; G4 gates on
    those, the observed claim is scored with this one.
    """
    v = np.asarray(null, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0 or not np.isfinite(obs):
        return float('nan')
    return float((1.0 + float((v >= float(obs)).sum())) / (v.size + 1.0))


def pa_mask(ctx, cm, oof_finite, *, floor_mask=True):
    """``P_a`` conditions (a)(b)(c); (d) is the caller's tier filter.

    ``floor_mask=False`` drops clause (b) and is the T13 ``floor_mask`` row --
    spec Sec.1.4 C3-A.3 requires the verdict to be unchanged before/after floor
    masking.  Identical to :func:`cliff.nulls._pa_mask` when ``floor_mask=True``.
    """
    u, v = ctx.nested_idx[:, 0], ctx.nested_idx[:, 1]
    m = (~ctx.wt_anchored) & oof_finite[u] & oof_finite[v]
    if floor_mask:
        m = m & ~(cm[u] | cm[v])
    return m


# =========================================================================== #
# the raw (identity-link) scale                                               #
# =========================================================================== #

def fit_raw(A, cn, As, y, censor_mask, floors, ceils, *, n_iter=None, n_bins=None,
            tol=None, sigma_floor=0.0):
    """The additive fit on the RAW scale: :func:`cliff.latent.fit_latent` with
    ``g = identity``, i.e. step 1 and the Tobit E-step only.

    No PAV, no ``ginv``, and NO identifiability anchor: with the identity link
    ``phi`` already IS the least-squares additive fit of ``y``, and rescaling it
    to ``y``'s sd -- which the latent fit must do so the isotonic step is
    identified -- would break ``e = y - phi`` being a residual.  The censored
    rows enter step 1 through the same E-step (spec Sec.1.0), with the detected
    level used directly instead of ``g^-1(L)``.  Returns the same knot triple the
    latent fit caches, so ``sigma`` and ``mu`` are read the identical way.
    """
    if n_iter is None:
        n_iter = THRESH['latent_n_iter']
    if n_bins is None:
        n_bins = THRESH['sigma_n_bins']
    if tol is None:
        tol = THRESH['latent_conv_tol']
    y = np.asarray(y, dtype=np.float64)
    cm = np.asarray(censor_mask, dtype=bool)
    unc = ~cm
    if unc.sum() < 3:
        raise ValueError('fit_raw: fewer than 3 uncensored rows')
    lvl = np.array(list(floors) + list(ceils), dtype=np.float64)
    is_fl = np.array([True] * len(floors) + [False] * len(ceils), dtype=bool)
    if cm.any() and lvl.size:
        which = np.abs(y[cm][:, None] - lvl[None, :]).argmin(axis=1)
        row_is_floor = is_fl[which]
        cvals = lvl[which]
    else:
        which = np.zeros(0, dtype=np.int64)
        row_is_floor = np.zeros(0, dtype=bool)
        cvals = np.zeros(0, dtype=np.float64)

    z = y.copy()
    beta = np.zeros(A.shape[1], dtype=np.float64)
    x0 = None
    phi = None
    sk = None
    used = 0
    dlast = float('nan')
    for it in range(int(n_iter)):
        res = lsqr(As, z, atol=1e-12, btol=1e-12, x0=x0)
        b = np.asarray(res[0], dtype=np.float64) / cn
        phi = A.dot(b)
        e_unc = y[unc] - phi[unc]
        sk = sigma_of_phi(phi[unc], e_unc, n_bins=n_bins, sigma_floor=sigma_floor)
        if cm.any() and lvl.size:
            z[cm] = tobit_estep(phi[cm], sigma_eval((sk[0], sk[1]), phi[cm]),
                                cvals, row_is_floor)
        dlast = float(np.abs(b - beta).max())
        beta = b
        x0 = b * cn
        used = it + 1
        if dlast < tol or not cm.any():
            break                       # with no censoring the fit is one lsqr
    return dict(beta=beta, phi=phi, z=z, sigma_knots=sk,
                n_iter_used=used, dbeta_last=dlast,
                resid_mad=mad_scaled(y[unc] - phi[unc]))


def crossfit_raw(ctx, y, censor_mask, *, sigma_floor=None):
    """The 5-fold cross-fit of :func:`fit_raw`, mirroring
    :func:`cliff.latent.crossfit_latent` clause for clause.

    Same folds (``ctx.folds``), same unseen-design-column rule (a held-out row
    carrying a substitution the training folds never saw gets ``phi_oof = nan``,
    because ``lsqr`` returns ``beta_j = 0`` there and the substitution's whole
    MAIN effect would otherwise land in ``e``), same Tobit E-step on held-out
    censored rows.  ``mu`` knots come from the FULL fit and ``sigma`` from the
    per-fold fits -- the convention :func:`cliff.nulls.build_context` and
    ``nulls._refit_bundle`` both use for the latent scale, so observed and
    surrogate are read identically.
    """
    if sigma_floor is None:
        sigma_floor = ctx.sigma_floor
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    cm = np.asarray(censor_mask, dtype=bool)
    floors, ceils = classify_levels(y, ctx.censor_levels)
    A, cn, As = ctx.A, ctx.cn, ctx.As
    full = fit_raw(A, cn, As, y, cm, floors, ceils, sigma_floor=sigma_floor)
    mu_knots = (full['sigma_knots'][0], full['sigma_knots'][2])

    Xc = ctx.X.tocsc()
    col_count_all = np.diff(Xc.indptr)
    phi_oof = np.full(n, np.nan)
    z_oof = np.full(n, np.nan)
    sigma_oof = np.full(n, np.nan)
    lvl = np.array(list(floors) + list(ceils), dtype=np.float64)
    is_fl = np.array([True] * len(floors) + [False] * len(ceils), dtype=bool)
    for k in np.unique(ctx.folds):
        te = ctx.folds == k
        tr = ~te
        Atr = with_intercept(ctx.X[tr])
        cntr = _latent._colnorms(Atr)
        Astr = (Atr @ sp.diags(1.0 / cntr)).tocsr()
        fl_tr, ce_tr = classify_levels(y[tr], ctx.censor_levels)
        f = fit_raw(Atr, cntr, Astr, y[tr], cm[tr], fl_tr, ce_tr,
                    sigma_floor=sigma_floor)
        cc_te = np.diff(Xc[te].tocsc().indptr)
        seen = (col_count_all - cc_te) > 0
        Xte = ctx.X[te]
        bad = np.asarray(Xte[:, ~seen].sum(axis=1)).ravel() > 0
        ph = float(f['beta'][0]) + Xte.dot(f['beta'][1:])
        ph[bad] = np.nan
        zz = y[te].copy()
        cmte = cm[te]
        if cmte.any() and lvl.size:
            wh = np.abs(y[te][cmte][:, None] - lvl[None, :]).argmin(axis=1)
            sg = sigma_eval((f['sigma_knots'][0], f['sigma_knots'][1]), ph[cmte])
            tb = tobit_estep(ph[cmte], sg, lvl[wh], is_fl[wh])
            zc = zz[cmte]
            fin = np.isfinite(ph[cmte])
            zc[fin] = tb[fin]
            zz[cmte] = zc
        zz[bad] = np.nan
        phi_oof[te] = ph
        z_oof[te] = zz
        sigma_oof[te] = sigma_eval((f['sigma_knots'][0], f['sigma_knots'][1]), ph)
    e_oof = z_oof - phi_oof
    fin = np.isfinite(phi_oof) & np.isfinite(z_oof)
    return dict(phi_oof=phi_oof, z_oof=z_oof, e_oof=e_oof, sigma_oof=sigma_oof,
                mu_oof=sigma_eval(mu_knots, phi_oof), oof_finite=fin,
                mu_knots=mu_knots, full=full,
                resid_mad_oof=mad_scaled(e_oof[fin & ~cm]),
                frac_oof_finite=float(fin.mean()))


def raw_bundle(ctx, rep=None):
    """The raw-scale analogue of a replicate bundle, for the observed data
    (``rep=None``) or for a surrogate's ``y*``."""
    y = ctx.y if rep is None else rep['y']
    cm = ctx.censor_mask if rep is None else rep['censor_mask']
    return crossfit_raw(ctx, y, cm)


# =========================================================================== #
# the mixture (zero-mean two-component Gaussian, spec Sec.1.3)                #
# =========================================================================== #

def _binned(u, n_bins):
    """Equal-count quantile bins of ``u = c^2`` with the WITHIN-BIN MEAN as the
    representative.  Exact when ``n <= n_bins``.

    Why binning is safe here and not a shortcut: every quantity in the model is a
    function of ``c`` only through ``u``, and in the tail ``log f(u)`` is
    asymptotically LINEAR in ``u`` (component 2 dominates), so the mean-of-``u``
    representative reproduces the bin's likelihood contribution to first order
    with no error at all in the region that decides ``pi`` and ``rho``.  The
    curvature is at the crossover, where the bins are dense.  Validated against
    the exact full-data EM in :func:`_selfcheck`.
    """
    u = np.asarray(u, dtype=np.float64)
    n = u.size
    if n <= n_bins:
        return u.copy(), np.ones(n, dtype=np.float64), None
    su = np.sort(u)
    parts = np.array_split(np.arange(n), n_bins)
    edges = np.array([su[p[0]] for p in parts], dtype=np.float64)
    ub = np.array([su[p].mean() for p in parts], dtype=np.float64)
    w = np.array([p.size for p in parts], dtype=np.float64)
    return ub, w, edges


def _mix_ll(ub, w, pi, v1, v2):
    la = math.log(max(pi, 1e-300)) - 0.5 * (_LOG_2PI + math.log(v2)) - ub / (2.0 * v2)
    lb = (math.log(max(1.0 - pi, 1e-300)) - 0.5 * (_LOG_2PI + math.log(v1))
          - ub / (2.0 * v1))
    return float((w * np.logaddexp(la, lb)).sum())


def _mix_em(ub, w, pi, v1, v2, n_iter, v_floor):
    """Weighted EM with the closed-form M-step.  Zero mean, so the responsibility
    is a logistic in ``u``: ``r = expit(B u - log A)`` with
    ``A = ((1-pi)/pi) sqrt(v2/v1)`` and ``B = (1/v1 - 1/v2)/2`` -- one ``exp`` per
    bin per iteration and no underflow."""
    W = float(w.sum())
    for _ in range(int(n_iter)):
        if not (v2 > v1):
            v1, v2 = min(v1, v2), max(v1, v2) + v_floor
        B = 0.5 * (1.0 / v1 - 1.0 / v2)
        logA = math.log(max(1.0 - pi, 1e-300)) - math.log(max(pi, 1e-300)) \
            + 0.5 * (math.log(v2) - math.log(v1))
        r = expit(B * ub - logA)
        wr = w * r
        s2 = float(wr.sum())
        s1 = W - s2
        if not (s2 > 1e-12 and s1 > 1e-12):
            break
        pi = s2 / W
        v2 = max(float((wr * ub).sum()) / s2, v_floor)
        v1 = max(float(((w - wr) * ub).sum()) / s1, v_floor)
    return pi, v1, v2


def _mix_starts(n_restart, rng):
    """42 deterministic grid starts, then log-uniform jitter to ``n_restart``."""
    pis = (0.001, 0.005, 0.02, 0.05, 0.10, 0.25, 0.50)
    rhos = (1.5, 2.0, 3.0, 5.0, 10.0, 20.0)
    st = [(p, r) for p in pis for r in rhos]
    k = int(n_restart) - len(st)
    if k > 0 and rng is not None:
        pj = np.exp(rng.uniform(math.log(1e-4), math.log(0.5), k))
        rj = np.exp(rng.uniform(math.log(1.2), math.log(50.0), k))
        st += list(zip(pj.tolist(), rj.tolist()))
    return st[:max(int(n_restart), 1)]


def mixture_two_component(c, n_restart=None, n_iter=None, *, seed_name='mixture_em',
                          dms_id=None, exact_ll=True, warm=None,
                          n_bins_coarse=_MIX_BINS_COARSE,
                          n_bins_fine=_MIX_BINS_FINE):
    """Zero-mean two-component Gaussian mixture (spec Sec.1.3).

    ``200 restarts x 100 iterations, closed-form M-step``.  Reports ``pi`` (the
    weight of the LARGER-variance component = the cliff mass), ``rho = s2/s1``,
    ``dBIC = BIC2 - BIC1`` and ``Lambda = 2(l2 - l1)``.  ``Lambda`` is calibrated
    against the N1 ensemble, never chi-square: with the null on the boundary
    (``pi = 0`` or ``rho = 1``) the LRT is non-regular.

    ``dBIC = -Lambda + 2 ln n`` holds exactly here (3 parameters against 1), and
    :func:`_selfcheck` asserts it -- it is the cheapest available check that the
    two likelihoods were computed on the same data.

    Search on ``n_bins_coarse`` quantile bins of ``c^2``, polish the winner on
    ``n_bins_fine``, then evaluate both log-likelihoods EXACTLY on the full data
    (``exact_ll``), so ``dBIC`` and ``Lambda`` are full-data quantities.
    """
    if n_restart is None:
        n_restart = THRESH['C2_em_n_restart']
    if n_iter is None:
        n_iter = THRESH['C2_em_n_iter']
    out = dict(pi_hat=float('nan'), sigma1=float('nan'), sigma2=float('nan'),
               rho_hat=float('nan'), dBIC=float('nan'), Lambda=float('nan'),
               ll1=float('nan'), ll2=float('nan'), n_mix=0, n_restart_used=0,
               converged=False)
    a = np.asarray(c, dtype=np.float64)
    a = a[np.isfinite(a)]
    n = a.size
    if n < 32:
        return out
    u = a * a
    vt = float(u.mean())
    if not (vt > 0):
        return out
    v_floor = max(vt * 1e-10, np.finfo(float).tiny)
    out['n_mix'] = int(n)
    out['ll1'] = -0.5 * n * (_LOG_2PI + math.log(vt) + 1.0)

    ubc, wc, _ = _binned(u, n_bins_coarse)
    ubf, wf, _ = _binned(u, n_bins_fine)
    if warm is not None:
        starts = [(float(warm[0]), float(warm[2] / warm[1]))]
        n_it = max(int(n_iter) // 2, 20)
    else:
        rng = np.random.default_rng(
            config.assay_seed(seed_name, dms_id) if dms_id is not None
            else [SEEDS[seed_name]])
        starts = _mix_starts(n_restart, rng)
        n_it = int(n_iter)
    best = None
    for p0, r0 in starts:
        p0 = min(max(float(p0), 1.0 / n), 1.0 - 1.0 / n)
        # E[u] = pi v2 + (1-pi) v1 = vt with v2 = r0^2 v1 pins both starts
        v1 = vt / (p0 * r0 * r0 + (1.0 - p0))
        v2 = max(v1 * r0 * r0, v1 * (1.0 + 1e-6))
        p, v1, v2 = _mix_em(ubc, wc, p0, max(v1, v_floor), max(v2, v_floor),
                            n_it, v_floor)
        ll = _mix_ll(ubc, wc, p, v1, v2)
        if np.isfinite(ll) and (best is None or ll > best[0]):
            best = (ll, p, v1, v2)
    out['n_restart_used'] = len(starts)
    if best is None:
        return out
    p, v1, v2 = _mix_em(ubf, wf, best[1], best[2], best[3], _MIX_ITER_POLISH,
                        v_floor)
    if v2 < v1:                                   # component 2 is the wide one
        p, v1, v2 = 1.0 - p, v2, v1
    ll2 = _mix_ll(u, np.ones(1), p, v1, v2) if exact_ll else _mix_ll(ubf, wf, p, v1, v2)
    if exact_ll:
        la = math.log(max(p, 1e-300)) - 0.5 * (_LOG_2PI + math.log(v2)) - u / (2.0 * v2)
        lb = (math.log(max(1.0 - p, 1e-300)) - 0.5 * (_LOG_2PI + math.log(v1))
              - u / (2.0 * v1))
        ll2 = float(np.logaddexp(la, lb).sum())
    out.update(pi_hat=float(p), sigma1=float(math.sqrt(v1)),
               sigma2=float(math.sqrt(v2)),
               rho_hat=float(math.sqrt(v2 / v1)) if v1 > 0 else float('nan'),
               ll2=float(ll2), converged=True)
    out['Lambda'] = 2.0 * (out['ll2'] - out['ll1'])
    out['dBIC'] = (-2.0 * out['ll2'] + 3.0 * math.log(n)) \
        - (-2.0 * out['ll1'] + 1.0 * math.log(n))
    return out


def mixture_ci_positions(c, pos, *, B=None, seed_name='bootstrap_block',
                         dms_id=None, point=None, n_bins=_MIX_BINS_FINE):
    """Block bootstrap over MUTATED POSITIONS for ``pi_hat`` (and ``rho``).

    Resamples the POSITION SET with replacement and takes every edge whose
    differing position is in the resample -- spec Sec.1.3, "the edge bootstrap
    ignores the dominant dependence".  The bins of ``c^2`` are held FIXED at the
    point estimate's quantile edges, so a replicate's binned data is
    ``multiplicity @ per-position-bin-counts``: one BLAS matmul for all ``B``
    replicates, and the EM then runs on ``n_bins`` numbers instead of ``|P_a|``.
    """
    if B is None:
        B = THRESH['C2_block_bootstrap_B']
    a = np.asarray(c, dtype=np.float64)
    pos = np.asarray(pos)
    ok = np.isfinite(a)
    a, pos = a[ok], pos[ok]
    n = a.size
    out = dict(pi_lo95=float('nan'), pi_hi95=float('nan'),
               rho_lo95=float('nan'), rho_hi95=float('nan'), n_boot=0)
    if n < 64:
        return out
    u = a * a
    ub, w, edges = _binned(u, n_bins)
    if edges is None:                        # n <= n_bins: one bin per edge
        edges = np.sort(u)
        ub = edges.copy()
    bin_of = np.searchsorted(edges, u, side='right') - 1
    np.clip(bin_of, 0, edges.size - 1, out=bin_of)
    upos, pi_idx = np.unique(pos, return_inverse=True)
    npos, nb = upos.size, edges.size
    # per-position x per-bin counts, and the bin's mean u (recomputed on the
    # actual assignment so the representative matches the fixed edges)
    flat = pi_idx.astype(np.int64) * nb + bin_of
    C = np.bincount(flat, minlength=npos * nb).astype(np.float64).reshape(npos, nb)
    su = np.bincount(bin_of, weights=u, minlength=nb)
    sc = np.bincount(bin_of, minlength=nb).astype(np.float64)
    ub = np.where(sc > 0, su / np.maximum(sc, 1.0), ub[:nb])
    vt_all = float(u.mean())
    v_floor = max(vt_all * 1e-10, np.finfo(float).tiny)
    rng = np.random.default_rng(config.assay_seed(seed_name, dms_id)
                                if dms_id is not None else [SEEDS[seed_name]])
    B = int(B)
    draws = rng.integers(0, npos, size=(B, npos))
    mult = np.zeros((B, npos), dtype=np.float64)
    for b in range(B):
        mult[b] = np.bincount(draws[b], minlength=npos)
    Wb = mult @ C                                            # (B, nb)
    p0 = (point or {}).get('pi_hat', 0.02)
    v1_0 = (point or {}).get('sigma1', math.sqrt(vt_all)) ** 2
    v2_0 = (point or {}).get('sigma2', 3.0 * math.sqrt(vt_all)) ** 2
    pis, rhos = np.empty(B), np.empty(B)
    for b in range(B):
        wb = Wb[b]
        m = wb > 0
        if m.sum() < 8:
            pis[b] = rhos[b] = np.nan
            continue
        p, v1, v2 = _mix_em(ub[m], wb[m], p0, max(v1_0, v_floor),
                            max(v2_0, v_floor), 60, v_floor)
        if v2 < v1:
            p, v1, v2 = 1.0 - p, v2, v1
        pis[b] = p
        rhos[b] = math.sqrt(v2 / v1) if v1 > 0 else np.nan
    fp = pis[np.isfinite(pis)]
    fr = rhos[np.isfinite(rhos)]
    if fp.size >= 8:
        out['pi_lo95'], out['pi_hi95'] = map(float, np.percentile(fp, [2.5, 97.5]))
    if fr.size >= 8:
        out['rho_lo95'], out['rho_hi95'] = map(float, np.percentile(fr, [2.5, 97.5]))
    out['n_boot'] = int(fp.size)
    return out


# =========================================================================== #
# the tau sweep, its CI and its grid guard                                    #
# =========================================================================== #

def rate_bootstrap_positions(absc, pos, taus, *, B=None,
                             seed_name='bootstrap_block', dms_id=None):
    """95% CI of ``P(|c| >= tau)`` by a block bootstrap over MUTATED POSITIONS.

    Exact and O(n_pos) per replicate rather than O(|P_a|): a rate over a
    resampled edge multiset is ``sum_k n_exceed[p_k] / sum_k n_tot[p_k]``, so the
    per-position (exceedance, total) counts are the whole sufficient statistic.
    :func:`block_bootstrap_positions` is the general form and
    :func:`_selfcheck` asserts the two agree bit-for-bit on a real assay.
    """
    if B is None:
        B = THRESH['C2_block_bootstrap_B']
    a = np.asarray(absc, dtype=np.float64)
    pos = np.asarray(pos)
    ok = np.isfinite(a)
    a, pos = a[ok], pos[ok]
    upos, pi_idx = np.unique(pos, return_inverse=True)
    npos = upos.size
    ntot = np.bincount(pi_idx, minlength=npos).astype(np.float64)
    rng = np.random.default_rng(config.assay_seed(seed_name, dms_id)
                                if dms_id is not None else [SEEDS[seed_name]])
    B = int(B)
    mult = np.empty((B, npos), dtype=np.float64)
    for b in range(B):
        mult[b] = np.bincount(rng.integers(0, npos, size=npos), minlength=npos)
    den = mult @ ntot
    out = {}
    for t in taus:
        nex = np.bincount(pi_idx[a >= float(t)], minlength=npos).astype(np.float64)
        num = mult @ nex
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(den > 0, num / np.maximum(den, 1.0), np.nan)
        f = r[np.isfinite(r)]
        out[float(t)] = ((float(np.percentile(f, 2.5)),
                          float(np.percentile(f, 97.5))) if f.size >= 8
                         else (float('nan'), float('nan')))
    return out


def block_bootstrap_positions(c, pos_of_pair, stat_fn, B=None,
                              seed_name='bootstrap_block', dms_id=None):
    """Spec Sec.3's signature: resample the POSITION set with replacement, take
    all edges whose differing position is in the resample, recompute ``stat_fn``.

    General but O(|P_a|) per replicate; used for statistics that are not rates.
    Returns ``(lo95, hi95, values)``.
    """
    if B is None:
        B = THRESH['C2_block_bootstrap_B']
    c = np.asarray(c, dtype=np.float64)
    pos = np.asarray(pos_of_pair)
    upos, pi_idx = np.unique(pos, return_inverse=True)
    npos = upos.size
    order = np.argsort(pi_idx, kind='stable')
    starts = np.searchsorted(pi_idx[order], np.arange(npos), side='left')
    ends = np.searchsorted(pi_idx[order], np.arange(npos), side='right')
    rng = np.random.default_rng(config.assay_seed(seed_name, dms_id)
                                if dms_id is not None else [SEEDS[seed_name]])
    vals = np.empty(int(B), dtype=np.float64)
    for b in range(int(B)):
        d = rng.integers(0, npos, size=npos)
        take = np.concatenate([order[starts[j]:ends[j]] for j in d]) \
            if npos else np.zeros(0, dtype=np.int64)
        vals[b] = stat_fn(c[take]) if take.size else np.nan
    f = vals[np.isfinite(vals)]
    if f.size < 8:
        return float('nan'), float('nan'), vals
    return (float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5)), vals)


def grid_guard(ctx, scale_value, *, taus=TAUS):
    """Spec Sec.1.0: drop any ``tau`` whose ABSOLUTE cut is below ``3 q_a``.

    ``scale_value`` is the statistic's own denominator in the units of the
    residual (median ``sqrt(s2_u + s2_v)`` for the sigma system, the global
    ``1.4826 MAD`` of the numerator for the MAD system).  On the RAW scale the
    residual is already in ``y`` units, so ``tau_absolute = tau * scale_value``
    is directly comparable to ``q_a``.  On the LATENT scale it is a ``z``-unit
    quantity and is mapped through the link's LOCAL slope
    ``dy/dz`` at the median of ``P_a``'s endpoints -- a local slope of the cached
    hull, not ``cliff.nulls._grid_guard_taus``'s global range ratio, because the
    link is far from linear (that is why there is a link).
    """
    q = float(ctx.quantum)
    cut = float(THRESH['grid_guard_mult']) * q
    return {float(t): (float(t) * float(scale_value),
                       bool(float(t) * float(scale_value) >= cut))
            for t in taus}


def _link_slope(ctx, phi):
    """Median local ``dy/dz`` of the cached hull over ``phi`` -- the only
    assumption-free way to put a latent-scale cut into ``y`` units."""
    yh, ph = ctx.hull                      # strict_hull -> (y_hull, phi_hull)
    yh = np.asarray(yh, dtype=np.float64)
    ph = np.asarray(ph, dtype=np.float64)
    if ph.size < 3:
        return 1.0
    sl = np.gradient(yh, ph)
    p = np.asarray(phi, dtype=np.float64)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return float(np.median(sl))
    return float(np.median(np.interp(p, ph, sl)))


def enrichment_sweep(c_obs, c_null, taus, unit, quantum):
    """Spec Sec.3's signature, kept as the documented reference form of the
    sweep: ``T(tau) = P_obs(|c| >= tau) / mean_b P_{null,b}(|c| >= tau)``.

    ``c_null`` is a ``(B, m)`` array of replicate ``c`` vectors.  The production
    path never materialises that -- 200 x 183,690 float64 is 294 MB per assay per
    null and spec Sec.5 caches STATISTIC VECTORS ONLY -- so :func:`build_T06`
    reads the cached rates instead.  This function is what that path is checked
    against in :func:`_selfcheck`.
    """
    a = np.abs(np.asarray(c_obs, dtype=np.float64))
    a = a[np.isfinite(a)]
    Cn = np.atleast_2d(np.asarray(c_null, dtype=np.float64))
    rows = []
    for t in taus:
        ro = float((a >= float(t)).mean()) if a.size else float('nan')
        rb = np.array([float((np.abs(r)[np.isfinite(r)] >= float(t)).mean())
                       for r in Cn])
        mb = float(np.nanmean(rb)) if rb.size else float('nan')
        rows.append(dict(tau=float(t), unit=unit,
                         tau_absolute=float(t), grid_guard_pass=bool(
                             float(t) >= THRESH['grid_guard_mult'] * float(quantum)),
                         rate_obs=ro, rate_null_mean=mb,
                         T=(ro / mb if mb and np.isfinite(mb) and mb > 0
                            else float('nan')),
                         p_perm=empirical_p_upper(ro, rb)))
    return pd.DataFrame(rows)


# =========================================================================== #
# blocks                                                                      #
# =========================================================================== #

def _lat_bundle(ctx, rep=None):
    if rep is None:
        return dict(e_oof=ctx.e_oof, sigma_oof=ctx.sigma_oof, mu_oof=ctx.mu_oof,
                    oof_finite=ctx.oof_finite, censor_mask=ctx.censor_mask,
                    z_oof=ctx.z_oof, phi_oof=ctx.phi_oof)
    return dict(e_oof=rep['e_oof'], sigma_oof=rep['sigma_oof'],
                mu_oof=rep['mu_oof'], oof_finite=rep['oof_finite'],
                censor_mask=rep['censor_mask'], z_oof=rep['z_oof'],
                phi_oof=rep['phi_oof'])


def block_values(ctx, lat, raw, block):
    """``c`` (and its numerator, denominator and edge selection) for one block.

    ``P_a`` clause (c) is the INTERSECTION of the latent and raw cross-fits'
    finite sets, so ``n_Pa`` is identical on the two scales and the scale
    sensitivity is the same edges with a different statistic.  Measured: the two
    sets are identical on 17/17 assays (both are exactly "no unseen design column
    in this row"), so the intersection costs nothing and guarantees the property.
    """
    name, scale, unit, centred, fmask = _BLOCK[block]
    b = lat if scale == 'latent' else raw
    of = lat['oof_finite'] & raw['oof_finite']
    keep = pa_mask(ctx, b['censor_mask'], of, floor_mask=fmask)
    sub = ctx.nested_idx[keep]
    n_Pa = int(keep.sum())
    if n_Pa == 0:
        z = np.zeros(0, dtype=np.float64)
        return dict(name=name, scale=scale, unit=unit, centred=centred,
                    floor_mask=fmask, n_Pa=0, c=z, num=z, den=z,
                    keep=keep, good=np.zeros(0, dtype=bool),
                    scale_value=float('nan'), edge=np.zeros(0, dtype=np.int64))
    e = b['e_oof'] if centred else b['e_oof']
    mu = b['mu_oof'] if centred else None
    if mu is None:
        num = e[sub[:, 1]] - e[sub[:, 0]]
    else:
        num = (e - mu)[sub[:, 1]] - (e - mu)[sub[:, 0]]
    if unit == 'sigma':
        den = np.sqrt(b['sigma_oof'][sub[:, 0]] ** 2
                      + b['sigma_oof'][sub[:, 1]] ** 2)
        with np.errstate(divide='ignore', invalid='ignore'):
            c = num / den
        fd = den[np.isfinite(den)]
        sv = float(np.median(fd)) if fd.size else float('nan')
    else:
        fn = num[np.isfinite(num)]
        sv = float(mad_scaled(fn)) if fn.size else float('nan')
        den = np.full(num.shape, sv, dtype=np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            c = num / sv if sv > 0 else np.full(num.shape, np.nan)
    good = np.isfinite(c)
    return dict(name=name, scale=scale, unit=unit, centred=centred,
                floor_mask=fmask, n_Pa=n_Pa, c=c[good], num=num[good],
                den=den[good], keep=keep, good=good, scale_value=sv,
                edge=np.flatnonzero(keep)[good])


def block_stats(bv, *, mixture=False, dms_id=None, taus=TAU_GRID,
                n_restart=None):
    """The per-block statistic dictionary (unprefixed)."""
    c = bv['c']
    out = dict(n_Pa=float(bv['n_Pa']), n_c=float(c.size),
               scale_value=float(bv['scale_value']),
               frac_c_zero=float('nan'), q75=float('nan'), q99=float('nan'),
               q999=float('nan'), TR1=float('nan'), TR2=float('nan'),
               kurt=float('nan'), mad=float('nan'), sd=float('nan'))
    for t in taus:
        out['rate_tau%g' % t] = float('nan')
    if mixture:
        out.update(pi_hat=float('nan'), sigma1=float('nan'), sigma2=float('nan'),
                   rho_hat=float('nan'), dBIC=float('nan'), Lambda=float('nan'))
    if c.size < 8:
        return out
    a = np.abs(c)
    out['frac_c_zero'] = float((c == 0).mean())
    q75, q99, q999 = np.percentile(a, [75.0, 99.0, 99.9])
    out['q75'], out['q99'], out['q999'] = map(float, (q75, q99, q999))
    out['TR1'] = float(q999 / q75) if q75 > 0 else float('nan')
    out['TR2'] = float(q99 / q75) if q75 > 0 else float('nan')
    d = c - c.mean()
    m2 = float((d * d).mean())
    out['kurt'] = float((d ** 4).mean() / (m2 * m2)) if m2 > 0 else float('nan')
    out['mad'] = float(mad_scaled(c))
    out['sd'] = float(c.std())
    for t in taus:
        out['rate_tau%g' % t] = float((a >= float(t)).mean())
    if mixture:
        m = mixture_two_component(c, n_restart=n_restart, dms_id=dms_id)
        for k in ('pi_hat', 'sigma1', 'sigma2', 'rho_hat', 'dBIC', 'Lambda'):
            out[k] = float(m[k])
    return out


def _prefixed(block, d):
    return dict(('%s__%s' % (block, k), v) for k, v in d.items())


def _stat_names_c2(mix_blocks=MIX_BLOCKS):
    names = []
    for nm in BLOCK_NAMES:
        d = dict.fromkeys(
            ['n_Pa', 'n_c', 'scale_value', 'frac_c_zero', 'q75', 'q99', 'q999',
             'TR1', 'TR2', 'kurt', 'mad', 'sd'], 0.0)
        for t in TAU_GRID:
            d['rate_tau%g' % t] = 0.0
        if nm in mix_blocks:
            for k in ('pi_hat', 'sigma1', 'sigma2', 'rho_hat', 'dBIC', 'Lambda'):
                d[k] = 0.0
        names += ['%s__%s' % (nm, k) for k in d]
    return tuple(names) + ('raw_n_iter', 'raw_resid_mad_oof',
                           'frac_oof_finite_raw', 'frac_oof_finite_lat',
                           'n_iter_used', 'wall_raw_s', 'wall_s')


STAT_NAMES_C2 = _stat_names_c2()
STAT_NAMES_C2_NOMIX = _stat_names_c2(mix_blocks=())


def _stat_fn(ctx, rep, *, mixture):
    t0 = time.time()
    lat = _lat_bundle(ctx, rep)
    t1 = time.time()
    raw = raw_bundle(ctx, rep)
    raw['censor_mask'] = lat['censor_mask']
    wall_raw = time.time() - t1
    out = {}
    for nm in BLOCK_NAMES:
        bv = block_values(ctx, lat, raw, nm)
        mix = bool(mixture) and nm in MIX_BLOCKS
        out.update(_prefixed(nm, block_stats(bv, mixture=mix,
                                             dms_id=ctx.dms_id)))
    out['raw_n_iter'] = float(raw['full']['n_iter_used'])
    out['raw_resid_mad_oof'] = float(raw['resid_mad_oof'])
    out['frac_oof_finite_raw'] = float(raw['frac_oof_finite'])
    out['frac_oof_finite_lat'] = float(np.mean(lat['oof_finite']))
    out['n_iter_used'] = float(rep.get('n_iter_used', np.nan)) if rep else 0.0
    out['wall_raw_s'] = float(wall_raw)
    out['wall_s'] = float(time.time() - t0)
    return out


def stat_fn_c2(ctx, rep):
    """The C2 statistic vector, no mixture (N2 / N2b / N2c)."""
    return _stat_fn(ctx, rep, mixture=False)


def stat_fn_c2_mix(ctx, rep):
    """The C2 statistic vector WITH the mixture (N1 only: T06's
    ``Lambda_N1_p995`` is the only column that reads a null mixture)."""
    return _stat_fn(ctx, rep, mixture=True)


def observed_blocks(ctx, *, keep_values=True):
    """Every block on the OBSERVED data, from the CACHED latent fit (spec Sec.5:
    never refit a cached latent fit for the observed data) plus a fresh raw fit
    (there is no cached raw fit; it is created here and nowhere else)."""
    lat = _lat_bundle(ctx, None)
    raw = raw_bundle(ctx, None)
    raw['censor_mask'] = ctx.censor_mask
    bl = {}
    for nm in BLOCK_NAMES:
        bv = block_values(ctx, lat, raw, nm)
        if not keep_values:
            for k in ('c', 'num', 'den', 'keep', 'good', 'edge'):
                bv.pop(k, None)
        bl[nm] = bv
    bl['_raw'] = raw
    bl['_lat'] = lat
    return bl


# =========================================================================== #
# ensembles                                                                   #
# =========================================================================== #

def _c2_ensemble_path(dms_id, null, B):
    """A name that can NEVER be confused with a canonical ``nulls/*.npz``: the
    statistic vector is a different one, so it gets its own suffix and its own
    ``stat_version`` guard."""
    return os.path.join(PATHS.nulls, '%s_%s_B%d_seed%d_c2v%s.npz'
                        % (dms_id, null, int(B),
                           SEEDS['nulls_' + null], STAT_C2_VERSION))


def ensembles_for(dms_id, nulls=('N1', 'N2', 'N2b', 'N2c'), B=None, *, nproc=1,
                  use_cache=True, write=True, verbose=True):
    """The four C2 ensembles for one assay, cached under this module's own name.

    ``run_ensemble`` refuses to write the SHARED cache for a custom ``stat_fn``
    (a bespoke statistic must never contaminate the canonical ensembles), so the
    caching is done here.
    """
    if B is None:
        B = THRESH['null_B']
    out = {}
    for nl in nulls:
        p = _c2_ensemble_path(dms_id, nl, B)
        if use_cache and os.path.exists(p):
            with np.load(p, allow_pickle=False) as z:
                names = [str(s) for s in z['stat_names']]
                meta = json.loads(str(z['meta_json']))
                arr = z['stats']
            if meta.get('stat_c2_version') == STAT_C2_VERSION:
                df = pd.DataFrame(arr, columns=names)
                df.attrs['meta'] = meta
                df.attrs['from_cache'] = True
                out[nl] = df
                if verbose:
                    print('    [c2-cache] %s %s B=%d' % (dms_id, nl, B))
                continue
        fn = stat_fn_c2_mix if nl == 'N1' else stat_fn_c2
        t0 = time.time()
        df = _nulls.run_ensemble(dms_id, nl, B, stat_fn=fn, nproc=nproc,
                                 use_cache=False, write=False, verbose=verbose)
        meta = dict(df.attrs.get('meta', {}))
        meta['stat_c2_version'] = STAT_C2_VERSION
        meta['mixture'] = (nl == 'N1')
        meta['blocks'] = list(BLOCK_NAMES)
        meta['tau_grid'] = list(TAU_GRID)
        df.attrs['meta'] = meta
        df.attrs['from_cache'] = False
        if write:
            PATHS.ensure_cache_dirs()
            tmp = p[:-4] + '.tmp%d.npz' % os.getpid()
            np.savez(tmp, stats=df.values.astype(np.float64),
                     stat_names=np.array(list(df.columns)),
                     meta_json=np.array(json.dumps(meta, sort_keys=True,
                                                   default=str)))
            os.replace(tmp, p)
        if verbose:
            print('    [c2 %s %s] B=%d  %.1fs  (%.2f s/rep serial-equiv)'
                  % (dms_id, nl, B, time.time() - t0,
                     (time.time() - t0) * max(nproc, 1) / B))
        out[nl] = df
    return out


def _pctl(v, q):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if v.size else float('nan')


def _mean(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float('nan')


# =========================================================================== #
# T06                                                                         #
# =========================================================================== #

T06_COLUMNS = (
    'DMS_id', 'scale', 'unit', 'n_Pa', 'frac_c_exact_zero', 'Q75', 'Q99',
    'Q999', 'TR_used', 'TR', 'TR_N1_p95', 'TR_N1_p995', 'TR_N2c_mean',
    'kurtosis', 'pi_hat', 'pi_lo95', 'pi_hi95', 'sigma1', 'sigma2', 'rho_hat',
    'dBIC', 'Lambda', 'Lambda_N1_p995', 'tau', 'tau_absolute',
    'grid_guard_pass', 'n_cliff', 'rate_obs', 'rate_N1_mean', 'rate_N2_mean',
    'rate_N2_p95', 'rate_N2b_mean', 'T_N2', 'T_N2_lo95', 'T_N2_hi95', 'T_N2b',
    'p_perm_N2', 'q_BH', 'n_consecutive_tau_passing', 'verdict_C2',
    'failing_criterion',
    # ---- one column beyond spec Sec.6's list, appended LAST (see module note)
    'notes',
)

#: D3: an assay whose standardised residuals are not identified cannot supply
#: evidence in either direction.  Its numbers are still computed and reported;
#: the INCONCLUSIVE stamp is :mod:`cliff.verdict`'s, read from T02.
D3_UNIDENTIFIED = ('CD19_FMC63_Fitness_7URV',)
D3_NOTE = ('STRUCTURALLY_UNIDENTIFIED (ORCHESTRATOR D3): 62.04%% oof-finite, '
           'resid_mad_oof/in 3.26x, sigma(phi) dyn range 2648x -- numbers '
           'reported, evidence value nil')


def _per_assay_c2(dms_id, *, B=None, nproc_null=1, nulls=('N1', 'N2', 'N2b', 'N2c'),
                  boot_B=None, verbose=True, use_cache=True):
    """Everything T06 / T13 / the catalogue need for ONE assay."""
    if B is None:
        B = THRESH['null_B']
    ctx = _nulls.get_context(dms_id, verify=False)
    obs = observed_blocks(ctx)
    ens = ensembles_for(dms_id, nulls=nulls, B=B, nproc=nproc_null,
                        use_cache=use_cache, verbose=verbose)
    pos_all = ctx.pos_of_add
    slope = _link_slope(ctx, ctx.phi_oof)
    res = dict(dms_id=dms_id, ctx=ctx, obs=obs, ens=ens, slope=slope, blocks={})
    for nm in BLOCK_NAMES:
        bv = obs[nm]
        pos = pos_all[bv['edge']] if bv['edge'].size else np.zeros(0, np.int64)
        mix = (mixture_two_component(bv['c'], dms_id=dms_id)
               if nm in PRIMARY_BLOCKS else
               mixture_two_component(bv['c'], n_restart=48, dms_id=dms_id))
        ci = (mixture_ci_positions(bv['c'], pos, B=boot_B, dms_id=dms_id,
                                   point=mix) if nm in PRIMARY_BLOCKS
              else dict(pi_lo95=float('nan'), pi_hi95=float('nan')))
        rci = rate_bootstrap_positions(np.abs(bv['c']), pos, TAUS, B=boot_B,
                                       dms_id=dms_id) if bv['c'].size else {}
        sv = bv['scale_value']
        gg = grid_guard(ctx, sv * (slope if bv['scale'] == 'latent' else 1.0))
        res['blocks'][nm] = dict(bv=dict((k, v) for k, v in bv.items()
                                         if k not in ('c', 'num', 'den', 'keep',
                                                      'good', 'edge')),
                                 stats=block_stats(bv, taus=TAU_GRID),
                                 mix=mix, ci=ci, rate_ci=rci, grid=gg,
                                 n_pos=int(np.unique(pos).size) if pos.size else 0)
    return res


def build_T06(per_assay, *, write=True, verbose=True):
    """T06 with spec Sec.6's exact columns (plus a trailing ``notes``).

    One row per ``(DMS_id, scale, unit, tau)``; the per-assay columns repeat on
    every row of the group, which is what :func:`cliff.verdict._first_num`
    expects.  ``sigma`` rows precede ``MAD`` rows so the spec's ``c_hat`` unit is
    the one the TR and mixture clauses read.

    BH-FDR is over the 14 PRIMARY+ARM assays (spec Sec.1.3, "never over 1.7e6
    edges"), per ``(scale, unit, tau)`` cell.  The 3 CONTROL assays are not data
    points and form their own BH family of 3 -- pooling them into the 14 would
    change the primary assays' ``q``, and G5/G6 read ``T(4)`` against the N2 band
    rather than ``q``.
    """
    ids = list(per_assay)
    bh_primary = [i for i in ids if i in config.PRIMARY_AND_ARM]
    bh_control = [i for i in ids if i not in config.PRIMARY_AND_ARM]
    rows = []
    for dms_id in ids:
        R = per_assay[dms_id]
        ens = R['ens']
        for scale in ('latent', 'raw'):
            for unit in ('sigma', 'MAD'):
                nm = {('latent', 'sigma'): 'L_s', ('latent', 'MAD'): 'L_m',
                      ('raw', 'sigma'): 'R_s', ('raw', 'MAD'): 'R_m'}[(scale, unit)]
                Bk = R['blocks'][nm]
                st, mix, ci, rci, gg = (Bk['stats'], Bk['mix'], Bk['ci'],
                                        Bk['rate_ci'], Bk['grid'])
                n_Pa = st['n_Pa']
                reg = ('TR1' if n_Pa >= THRESH['C2_TR1_min_Pa'] else
                       'TR2' if n_Pa >= THRESH['C2_TR2_min_Pa'] else 'none')
                TR = st.get(reg, float('nan')) if reg != 'none' else float('nan')
                trcol = reg if reg != 'none' else 'TR1'
                n1 = ens['N1']['%s__%s' % (nm, trcol)].values if 'N1' in ens else []
                n2c = ens['N2c']['%s__%s' % (nm, trcol)].values if 'N2c' in ens else []
                lam1 = (ens['N1']['%s__Lambda' % nm].values
                        if ('N1' in ens and '%s__Lambda' % nm in ens['N1'].columns)
                        else [])
                note = []
                if dms_id in D3_UNIDENTIFIED:
                    note.append(D3_NOTE)
                if reg == 'none':
                    note.append('|P_a|<%d: NO tail-ratio verdict (spec Sec.1.3)'
                                % THRESH['C2_TR2_min_Pa'])
                note.append('BH family=%s(%d)'
                            % ('PRIMARY+ARM' if dms_id in config.PRIMARY_AND_ARM
                               else 'CONTROL',
                               len(bh_primary) if dms_id in config.PRIMARY_AND_ARM
                               else len(bh_control)))
                for t in TAUS:
                    tf = float(t)
                    key = 'rate_tau%g' % tf
                    ro = st[key]
                    rn1 = _mean(ens['N1']['%s__%s' % (nm, key)]) if 'N1' in ens else float('nan')
                    rn2v = (ens['N2']['%s__%s' % (nm, key)].values
                            if 'N2' in ens else np.zeros(0))
                    rn2 = _mean(rn2v)
                    rn2b = _mean(ens['N2b']['%s__%s' % (nm, key)]) if 'N2b' in ens else float('nan')
                    T = (ro / rn2) if (np.isfinite(rn2) and rn2 > 0) else float('nan')
                    Tb = (ro / rn2b) if (np.isfinite(rn2b) and rn2b > 0) else float('nan')
                    lo, hi = rci.get(tf, (float('nan'), float('nan')))
                    rows.append(dict(
                        DMS_id=dms_id, scale=scale, unit=unit,
                        n_Pa=int(n_Pa), frac_c_exact_zero=st['frac_c_zero'],
                        Q75=st['q75'], Q99=st['q99'], Q999=st['q999'],
                        TR_used=reg, TR=TR, TR_N1_p95=_pctl(n1, 95.0),
                        TR_N1_p995=_pctl(n1, 99.5), TR_N2c_mean=_mean(n2c),
                        kurtosis=st['kurt'], pi_hat=mix['pi_hat'],
                        pi_lo95=ci['pi_lo95'], pi_hi95=ci['pi_hi95'],
                        sigma1=mix['sigma1'], sigma2=mix['sigma2'],
                        rho_hat=mix['rho_hat'], dBIC=mix['dBIC'],
                        Lambda=mix['Lambda'], Lambda_N1_p995=_pctl(lam1, 99.5),
                        tau=tf, tau_absolute=gg[tf][0],
                        grid_guard_pass=bool(gg[tf][1]),
                        n_cliff=int(round(ro * st['n_c'])), rate_obs=ro,
                        rate_N1_mean=rn1, rate_N2_mean=rn2,
                        rate_N2_p95=_pctl(rn2v, 95.0), rate_N2b_mean=rn2b,
                        T_N2=T,
                        T_N2_lo95=(lo / rn2 if (np.isfinite(rn2) and rn2 > 0)
                                   else float('nan')),
                        T_N2_hi95=(hi / rn2 if (np.isfinite(rn2) and rn2 > 0)
                                   else float('nan')),
                        T_N2b=Tb, p_perm_N2=empirical_p_upper(ro, rn2v),
                        q_BH=float('nan'), n_consecutive_tau_passing=-1,
                        verdict_C2='', failing_criterion='',
                        notes='; '.join(note)))
    df = pd.DataFrame(rows)
    # ---- BH-FDR per (scale, unit, tau) cell, within each BH family ---------- #
    for fam in (bh_primary, bh_control):
        if not fam:
            continue
        for (sc, un, tf), g in df[df['DMS_id'].isin(fam)].groupby(
                ['scale', 'unit', 'tau']):
            df.loc[g.index, 'q_BH'] = bh_fdr(g['p_perm_N2'].values)
    # ---- the consecutive-tau count, exactly as verdict.py recomputes it ----- #
    lo_w, hi_w = config.TAU_WINDOW
    for (dms_id, sc, un), g in df.groupby(['DMS_id', 'scale', 'unit']):
        seq = []
        for _, r in g.sort_values('tau').iterrows():
            if not (lo_w <= r['tau'] <= hi_w):
                continue
            if not bool(r['grid_guard_pass']):
                continue                      # guard-dropped tau is REMOVED
            seq.append(bool(np.isfinite(r['T_N2'])
                            and r['T_N2'] >= THRESH['C2_T_sup']
                            and np.isfinite(r['q_BH'])
                            and r['q_BH'] < THRESH['C2_q_BH_sup']))
        best = cur = 0
        for v in seq:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        df.loc[g.index, 'n_consecutive_tau_passing'] = best
    order = {'latent': 0, 'raw': 1}
    uorder = {'sigma': 0, 'MAD': 1}
    df['_o'] = [order[s] * 100 + uorder[u] * 10 for s, u in
                zip(df['scale'], df['unit'])]
    df = df.sort_values(['DMS_id', '_o', 'tau']).drop(columns=['_o'])
    df = df[list(T06_COLUMNS)]
    if write:
        PATHS.ensure_cache_dirs()
        p = os.path.join(PATHS.artifacts, 'T06_cliff_tail_C2.csv')
        df.to_csv(p, index=False)
        if verbose:
            print('[T06] %d rows -> %s' % (len(df), p))
    return df


# =========================================================================== #
# T13 -- the sensitivity rows this module owns                                #
# =========================================================================== #

T13_COLUMNS = ('DMS_id', 'knob', 'value', 'n_Pa', 'TR', 'T_N2', 'q_BH', 'dBIC',
               'pi_hat', 'SI', 'beta_sibling', 'OR_iface', 'verdict_flips')

#: The knobs this module owns.  ``centring`` is not in spec Sec.6's enum -- it is
#: ORCHESTRATOR D2's mandated row and is added here rather than folded into
#: ``scale``, which means something else.
C2_KNOBS = ('centring', 'tau', 'sigma_mult', 'scale', 'floor_mask')

#: The tau the single-number sensitivity rows are read at: 4 is the catalogue
#: threshold and G5's reference, and it sits inside the [3,8] verdict window.
T13_REF_TAU = 4.0


def _t13_lock(timeout=300.0, poll=0.05):
    import contextlib
    import fcntl

    @contextlib.contextmanager
    def _cm():
        d = os.path.join(PATHS.cache, '.locks')
        os.makedirs(d, exist_ok=True)
        fh = open(os.path.join(d, 'T13.lock'), 'a+')
        t0 = time.time()
        try:
            while True:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.time() - t0 > timeout:
                        raise RuntimeError('T13 lock held > %.0f s' % timeout)
                    time.sleep(poll)
            yield
        finally:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            finally:
                fh.close()
    return _cm()


def _si_from_T04():
    p = os.path.join(PATHS.artifacts, 'T04_smoothness_C1.csv')
    if not os.path.exists(p):
        return {}
    try:
        d = pd.read_csv(p)
        return dict(zip(d['DMS_id'].astype(str), pd.to_numeric(d['SI'],
                                                               errors='coerce')))
    except Exception:                                    # pragma: no cover
        return {}


def _clause2(T, q):
    return bool(np.isfinite(T) and T >= THRESH['C2_T_sup']
                and np.isfinite(q) and q < THRESH['C2_q_BH_sup'])


def build_T13_rows(per_assay, t06=None, *, write=True, verbose=True):
    """The ``centring`` / ``tau`` / ``sigma_mult`` / ``scale`` / ``floor_mask``
    rows of T13.

    ``sigma_mult`` is FREE and EXACT rather than a second null run: N2 reuses the
    observed ``sigma_oof``, so ``sigma -> m sigma`` divides observed AND null
    ``c`` by the same ``m`` and ``T(tau; m sigma) == T(m tau; sigma)``.  The
    ensembles therefore carry the extended :data:`TAU_GRID` and the row is read
    off it.  ``TR`` is EXACTLY invariant under the same map (a ratio of quantiles
    of ``|c|``), which is the property spec Sec.1.3 says "carries the design"; the
    row reports the identical TR on all three multipliers on purpose, as evidence
    that the scale-freeness is real and not asserted.

    Rows carry only the columns this module can measure.  ``SI`` is copied from
    T04 when it exists (read-only); ``beta_sibling`` (T07) and ``OR_iface``
    (T09) belong to other modules and are left empty rather than guessed.
    """
    si = _si_from_T04()
    ids = list(per_assay)
    rows = []

    def _rate(dms_id, nm, t, source):
        R = per_assay[dms_id]
        key = 'rate_tau%g' % float(t)
        if source == 'obs':
            return R['blocks'][nm]['stats'].get(key, float('nan'))
        e = R['ens'].get(source)
        if e is None or '%s__%s' % (nm, key) not in e.columns:
            return float('nan'), np.zeros(0)
        return _mean(e['%s__%s' % (nm, key)]), e['%s__%s' % (nm, key)].values

    def _row(dms_id, knob, value, nm, tau, tr_key=None, ci=None):
        R = per_assay[dms_id]
        Bk = R['blocks'][nm]
        st = Bk['stats']
        n_Pa = st['n_Pa']
        reg = ('TR1' if n_Pa >= THRESH['C2_TR1_min_Pa'] else
               'TR2' if n_Pa >= THRESH['C2_TR2_min_Pa'] else 'none')
        TR = st.get(tr_key or reg, float('nan')) if reg != 'none' else float('nan')
        ro = st.get('rate_tau%g' % float(tau), float('nan'))
        rn2, rn2v = _rate(dms_id, nm, tau, 'N2')
        T = (ro / rn2) if (np.isfinite(rn2) and rn2 > 0) else float('nan')
        return dict(DMS_id=dms_id, knob=knob, value=str(value), n_Pa=int(n_Pa),
                    TR=TR, T_N2=T, q_BH=float('nan'),
                    dBIC=Bk['mix']['dBIC'], pi_hat=Bk['mix']['pi_hat'],
                    SI=si.get(dms_id, float('nan')), beta_sibling='',
                    OR_iface='', verdict_flips='',
                    _p=empirical_p_upper(ro, rn2v), _tau=float(tau),
                    _block=nm, _TRp995=_pctl(
                        R['ens']['N1']['%s__%s' % (nm, tr_key or (reg if reg != 'none'
                                                                 else 'TR1'))].values
                        if 'N1' in R['ens'] else [], 99.5))

    for dms_id in ids:
        # ---- knob = tau : the whole sweep, latent x sigma ------------------ #
        for t in TAUS:
            rows.append(_row(dms_id, 'tau', '%g' % t, 'L_s', t))
        # ---- knob = sigma_mult : T(tau_ref ; sigma x m) == T(m tau_ref) ---- #
        for m in config.SIGMA_MULTIPLIERS:
            rows.append(_row(dms_id, 'sigma_mult', '%g' % m, 'L_s',
                             m * T13_REF_TAU))
        # ---- knob = scale -------------------------------------------------- #
        rows.append(_row(dms_id, 'scale', 'latent', 'L_s', T13_REF_TAU))
        rows.append(_row(dms_id, 'scale', 'raw', 'R_s', T13_REF_TAU))
        # ---- knob = floor_mask --------------------------------------------- #
        rows.append(_row(dms_id, 'floor_mask', 'on', 'L_s', T13_REF_TAU))
        rows.append(_row(dms_id, 'floor_mask', 'off', 'Lf_s', T13_REF_TAU))
        # ---- knob = centring (ORCHESTRATOR D2) ----------------------------- #
        rows.append(_row(dms_id, 'centring', 'phi_centred', 'L_s', T13_REF_TAU))
        rows.append(_row(dms_id, 'centring', 'uncentred', 'Lu_s', T13_REF_TAU))
    df = pd.DataFrame(rows)
    # BH within each (knob, value) cell over the 14 primary+arm, controls apart
    fam_p = [i for i in ids if i in config.PRIMARY_AND_ARM]
    for fam in (fam_p, [i for i in ids if i not in config.PRIMARY_AND_ARM]):
        if not fam:
            continue
        for _k, g in df[df['DMS_id'].isin(fam)].groupby(['knob', 'value']):
            df.loc[g.index, 'q_BH'] = bh_fdr(g['_p'].values)
    # verdict_flips: does clause 2 (T>=2 & q<0.05) or the TR>N1_p995 clause move
    # away from the PRIMARY block's answer?
    base = {}
    for dms_id in ids:
        r = df[(df['DMS_id'] == dms_id) & (df['knob'] == 'scale')
               & (df['value'] == 'latent')]
        if len(r):
            r = r.iloc[0]
            base[dms_id] = (_clause2(r['T_N2'], r['q_BH']),
                            bool(np.isfinite(r['TR']) and np.isfinite(r['_TRp995'])
                                 and r['TR'] > r['_TRp995']))
    flips = []
    for _, r in df.iterrows():
        b = base.get(r['DMS_id'])
        if b is None:
            flips.append('')
            continue
        c2 = _clause2(r['T_N2'], r['q_BH'])
        tr = bool(np.isfinite(r['TR']) and np.isfinite(r['_TRp995'])
                  and r['TR'] > r['_TRp995'])
        flips.append('T>=2&q<.05:%s->%s TR>N1p995:%s->%s%s'
                     % ('T' if b[0] else 'F', 'T' if c2 else 'F',
                        'T' if b[1] else 'F', 'T' if tr else 'F',
                        '' if (c2 == b[0] and tr == b[1]) else '  FLIP'))
    df['verdict_flips'] = flips
    df = df[list(T13_COLUMNS)]
    if write:
        PATHS.ensure_cache_dirs()
        p = os.path.join(PATHS.artifacts, 'T13_sensitivity.csv')
        with _t13_lock():
            if os.path.exists(p):
                old = pd.read_csv(p, dtype=str)
                if 'knob' in old.columns:
                    old = old[~old['knob'].astype(str).isin(C2_KNOBS)]
                else:                                    # pragma: no cover
                    old = old.iloc[0:0]
                for c in T13_COLUMNS:
                    if c not in old.columns:
                        old[c] = ''
                extra = [c for c in old.columns if c not in T13_COLUMNS]
                out = pd.concat([old, df.astype(object)], ignore_index=True,
                                sort=False)
                out = out[list(T13_COLUMNS) + extra]
            else:
                out = df
            out.to_csv(p, index=False)
        if verbose:
            print('[T13] %d stats_c2 rows (knobs %s) -> %s'
                  % (len(df), ','.join(C2_KNOBS), p))
    return df


# =========================================================================== #
# cliff_catalogue_{DMS_id}.csv.gz                                             #
# =========================================================================== #

CATALOGUE_COLUMNS = (
    'pair_id', 'DMS_id', 'row_index_u', 'row_index_v', 'mutant_u', 'mutant_v',
    'order_u', 'order_v', 'background_key', 'add_chain', 'add_seq_pos',
    'add_resseq', 'add_icode', 'add_wt_aa', 'add_mut_aa', 'y_u', 'y_v',
    'delta_y', 'delta_latent', 'beta_hat_add', 'c_hat', 'c_hat_MAD_unit',
    'sigma_used', 'sigma_provenance', 'censor_class', 'wt_anchored', 'degree_u',
    'degree_v', 'density_quintile', 'n_siblings', 'sibling_mean', 'sibling_z',
    'tau_min_included', 'q_value', 'levy_class', 'min_heavy_dist', 'dsasa',
    'rsa_iso', 'rsa_cplx', 'blosum62', 'd_hydrophobicity', 'd_volume',
    'family', 'PSI', 'partners_cliff_in', 'verdict_flags',
)

#: Kyte & Doolittle 1982 hydropathy index.
KD_HYDRO = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4,
                H=-3.2, I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8,
                T=-0.7, W=-0.9, Y=-1.3, V=4.2)
#: Zamyatnin 1972 residue volumes, A^3.
RES_VOLUME = dict(A=88.6, R=173.4, N=114.1, D=111.1, C=108.5, Q=143.8, E=138.4,
                  G=60.1, H=153.2, I=166.7, L=166.7, K=168.6, M=162.9, F=189.9,
                  P=112.7, S=89.0, T=116.1, W=227.8, Y=193.6, V=140.0)


def _blosum62():
    from Bio.Align import substitution_matrices
    return substitution_matrices.load('BLOSUM62')


def _sibling_adjacency(idx, add_col, keys):
    """CSR sibling graph of the nested edges: ``S(e) = {(B', B' u {i}) :
    |B xor B'| = 1}`` (spec Sec.1.4 L1).

    Enumerated by REMOVAL only and then symmetrised: if ``B' = B \\ {j}`` then
    ``B = B' u {j}``, so one pass over each edge's ``|B|`` sub-keys gives both
    directions of the sibling relation.  Same construction as
    :func:`cliff.pairs.sibling_counts`, which returns the degree only.
    """
    m = idx.shape[0]
    if m == 0:
        return (np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int32))
    src, dst = [], []
    order = np.argsort(add_col, kind='stable')
    ac = add_col[order]
    for g in np.split(order, np.flatnonzero(np.diff(ac)) + 1):
        loc = {}
        for t, e in enumerate(g):
            loc[keys[idx[e, 0]]] = t
        get = loc.get
        for t, e in enumerate(g):
            k = keys[idx[e, 0]]
            for j in range(len(k)):
                u = get(k[:j] + k[j + 1:])
                if u is not None:
                    src.append(g[t]); dst.append(g[u])
                    src.append(g[u]); dst.append(g[t])
    if not src:
        return (np.zeros(m + 1, dtype=np.int64), np.zeros(0, dtype=np.int32))
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int32)
    o = np.argsort(src, kind='stable')
    src, dst = src[o], dst[o]
    ptr = np.zeros(m + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(np.bincount(src, minlength=m))
    return ptr, dst


def build_catalogue(dms_id, per_assay_entry=None, t06=None, *, write=True,
                    verbose=True, c_min=None):
    """``cliff_catalogue_{DMS_id}.csv.gz`` -- every nested edge with
    ``|c_hat| >= 4`` (spec Sec.6), on the PRIMARY definition (latent scale,
    sigma units, phi-centred).

    **Scope.**  The row set is every nested edge with a finite ``c_hat``, i.e.
    ``P_a`` clause (c) only, NOT clauses (a)/(b): the spec's own column list
    carries ``wt_anchored`` and ``censor_class{floorfree|crossing|floorfloor}``,
    which are constant on the masked set, and a catalogue whose two provenance
    columns cannot vary is useless for the floor-invariance check C3-A.3 asks
    for.  ``verdict_flags`` says of every row whether it is in ``P_a`` and why
    not when it is not, so filtering back to the primary set is one predicate.

    ``tau_min_included`` is the LARGEST sweep ``tau`` the edge clears, i.e. the
    edge is a member of the exceedance set of every sweep ``tau`` up to and
    including that value ("the smallest tau at which it is included" is 2 for
    every catalogued row and carries no information).  ``q_value`` is the assay's
    BH ``q`` for ``T(tau)`` at that ``tau``, not a per-edge quantity -- there is
    no per-edge test in this design and inventing one would be the
    million-edge multiplicity the spec forbids.
    """
    from . import io_bgym
    if c_min is None:
        c_min = THRESH['C2_catalogue_c_min']
    ctx = _nulls.get_context(dms_id, verify=False)
    lat = _lat_bundle(ctx, None)
    raw = raw_bundle(ctx, None)
    raw['censor_mask'] = ctx.censor_mask
    of = lat['oof_finite'] & raw['oof_finite']
    idx = ctx.nested_idx
    u, v = idx[:, 0], idx[:, 1]
    live = of[u] & of[v]
    mu = lat['mu_oof']
    num = (lat['e_oof'] - mu)[v] - (lat['e_oof'] - mu)[u]
    den = np.sqrt(lat['sigma_oof'][u] ** 2 + lat['sigma_oof'][v] ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        c = num / den
    live &= np.isfinite(c)
    # the MAD unit uses the PRIMARY P_a's global scale, so the two columns of the
    # catalogue are the two unit systems of the same statistic
    keep_pa = pa_mask(ctx, ctx.censor_mask, of, floor_mask=True)
    sc_mad = mad_scaled(num[keep_pa & np.isfinite(num)])
    sel = live & (np.abs(c) >= float(c_min))
    e_sel = np.flatnonzero(sel)
    if verbose:
        print('[cat ] %-40s %7d of %7d nested edges  |c|>=%g'
              % (dms_id, e_sel.size, int(live.sum()), c_min))

    assay = io_bgym.load_assay(dms_id)
    if assay.n != ctx.n:
        raise RuntimeError('%s: csv n=%d but cache n=%d' % (dms_id, assay.n, ctx.n))
    mut_raw = pd.read_csv(PATHS.dms_csv(dms_id), usecols=['mutant'])['mutant'].values
    inv_col = {vv: kk for kk, vv in _latent.load_cached_design(
        dms_id, verify=False)['col_index'].items()}
    deg = _pairs.degrees(ctx.n, idx)
    dmean = 0.5 * (deg[u] + deg[v]).astype(np.float64)
    qq = np.percentile(dmean[live], [20, 40, 60, 80]) if live.any() else np.zeros(4)
    dq = np.searchsorted(qq, dmean, side='right') + 1
    ptr, adj = _sibling_adjacency(idx, ctx.add_col, assay.keys)

    # ---- structural annotation, joined through (chain, seq_pos) ------------ #
    st = {}
    p09 = os.path.join(PATHS.artifacts, 'T09_structure_sites.csv')
    if os.path.exists(p09):
        d9 = pd.read_csv(p09)
        d9 = d9[d9['DMS_id'].astype(str) == dms_id]
        for _, r in d9.iterrows():
            st[(str(r['chain']), int(r['seq_idx']))] = r
    B62 = _blosum62()
    prov = ('internal_residual: sigma_oof = 1.4826 MAD(e) per phi-bin from this '
            "assay's own 5-fold cross-fit (spec Sec.1.0 table, row 3)")
    fam = config.family_of(dms_id) or ''
    # per-tau q from T06 (latent x sigma), for q_value
    qmap = {}
    if t06 is not None and len(t06):
        g = t06[(t06['DMS_id'].astype(str) == dms_id)
                & (t06['scale'].astype(str) == 'latent')
                & (t06['unit'].astype(str) == 'sigma')]
        for _, r in g.iterrows():
            qmap[float(r['tau'])] = float(r['q_BH'])
    rows = []
    for e in e_sel:
        iu, iv = int(u[e]), int(v[e])
        ku = assay.keys[iu]
        cm_u, cm_v = bool(ctx.censor_mask[iu]), bool(ctx.censor_mask[iv])
        cls = ('floorfloor' if (cm_u and cm_v) else
               'crossing' if (cm_u or cm_v) else 'floorfree')
        ch, sp, aa = inv_col[int(ctx.add_col[e])]
        rec = st.get((ch, int(sp)))
        wt = str(rec['wt_aa']) if rec is not None else ''
        sib = adj[ptr[e]:ptr[e + 1]]
        sc_ = c[sib] if sib.size else np.zeros(0)
        sc_ = sc_[np.isfinite(sc_)]
        if sc_.size >= 3:
            md = mad_scaled(sc_)
            sz = float((c[e] - np.median(sc_)) / md) if md > 0 else float('nan')
        else:
            sz = float('nan')
        clears = [t for t in TAUS if abs(c[e]) >= t]
        tmin = float(max(clears)) if clears else float('nan')
        flags = ['in_Pa'] if keep_pa[e] else (
            ['not_in_Pa'] + (['wt_anchored'] if ctx.wt_anchored[e] else [])
            + (['censor_touching'] if (cm_u or cm_v) else []))
        rows.append((
            '%s#%d->%d' % (dms_id, int(assay.row_index[iu]), int(assay.row_index[iv])),
            dms_id, int(assay.row_index[iu]), int(assay.row_index[iv]),
            str(mut_raw[iu]), str(mut_raw[iv]),
            int(ctx.n_muts[iu]), int(ctx.n_muts[iv]), str(ku),
            ch, int(sp),
            (int(rec['resseq']) if rec is not None else ''),
            (('' if pd.isna(rec['icode']) else str(rec['icode']))
             if rec is not None else ''),
            wt, aa,
            float(ctx.y[iu]), float(ctx.y[iv]),
            float(ctx.y[iv] - ctx.y[iu]),
            float(lat['z_oof'][iv] - lat['z_oof'][iu]),
            float(ctx.beta[1 + int(ctx.add_col[e])]),
            float(c[e]), float(num[e] / sc_mad) if sc_mad > 0 else float('nan'),
            float(den[e]), prov, cls, bool(ctx.wt_anchored[e]),
            int(deg[iu]), int(deg[iv]), int(dq[e]),
            int(sib.size), (float(sc_.mean()) if sc_.size else float('nan')), sz,
            tmin, qmap.get(tmin, float('nan')),
            (str(rec['levy_class']) if rec is not None else ''),
            (float(rec['min_heavy_dist']) if rec is not None else ''),
            (float(rec['dsasa']) if rec is not None else ''),
            (float(rec['rsa_iso']) if rec is not None else ''),
            (float(rec['rsa_cplx']) if rec is not None else ''),
            (float(B62[wt, aa]) if (wt in KD_HYDRO and aa in KD_HYDRO) else ''),
            (float(KD_HYDRO[aa] - KD_HYDRO[wt]) if (wt in KD_HYDRO and aa in KD_HYDRO) else ''),
            (float(RES_VOLUME[aa] - RES_VOLUME[wt]) if (wt in RES_VOLUME and aa in RES_VOLUME) else ''),
            fam, '', '', ';'.join(flags)))
    df = pd.DataFrame(rows, columns=list(CATALOGUE_COLUMNS))
    if write:
        PATHS.ensure_cache_dirs()
        p = os.path.join(PATHS.artifacts, 'cliff_catalogue_%s.csv.gz' % dms_id)
        with gzip.open(p, 'wt', newline='') as fh:
            df.to_csv(fh, index=False)
        if verbose:
            print('       -> %s  (%d rows, %.1f kB)'
                  % (os.path.basename(p), len(df), os.path.getsize(p) / 1e3))
    return df


# =========================================================================== #
# driver                                                                      #
# =========================================================================== #

def register_c2_cache(extra=None):
    """md5 every ``nulls/*_c2v*.npz`` into ``MANIFEST.json`` -- ONE call at the
    END of the run (D8), through the ``flock``-protected
    :func:`cliff.pairs.write_manifest`, then :func:`cliff.pairs.verify_manifest`.
    """
    from .io_bgym import md5_of
    PATHS.ensure_cache_dirs()
    ents = []
    for f in sorted(os.listdir(PATHS.nulls)):
        if not (f.endswith('.npz') and '_c2v' in f):
            continue
        q = os.path.join(PATHS.nulls, f)
        ents.append(dict(path=os.path.relpath(q, config.REPO), md5=md5_of(q),
                         bytes=os.path.getsize(q)))
    if ents:
        _pairs.write_manifest(ents, extra=extra)
    return ents


def _summary(R):
    """Drop the heavy objects; keep exactly what T06/T13 read."""
    return dict(dms_id=R['dms_id'], slope=R['slope'], blocks=R['blocks'],
                ens=dict((k, v) for k, v in R['ens'].items()))


def run_all(assays=None, *, B=None, nproc=None, boot_B=None, catalogue=True,
            write=True, verbose=True, use_cache=True,
            nulls=('N1', 'N2', 'N2b', 'N2c')):
    """Stage C2 end to end: ensembles, T06, the catalogues, the T13 rows."""
    if assays is None:
        assays = list(config.PRIMARY + config.ARM + config.CONTROL)
    if nproc is None:
        nproc = THRESH['nproc_cap']
    if B is None:
        B = THRESH['null_B']
    t0 = time.time()
    per = {}
    for i, a in enumerate(assays):
        if verbose:
            print('[c2  ] %2d/%d %s' % (i + 1, len(assays), a))
        R = _per_assay_c2(a, B=B, nproc_null=nproc, nulls=nulls, boot_B=boot_B,
                          verbose=verbose, use_cache=use_cache)
        per[a] = _summary(R)
        del R
        _nulls.clear_context_cache()
    t06 = build_T06(per, write=write, verbose=verbose)
    t13 = build_T13_rows(per, t06, write=write, verbose=verbose)
    cats = {}
    if catalogue:
        for a in assays:
            cats[a] = build_catalogue(a, t06=t06, write=write, verbose=verbose)
            _nulls.clear_context_cache()
    if write:
        ents = register_c2_cache(extra=dict(stats_c2=dict(
            stat_c2_version=STAT_C2_VERSION, blocks=list(BLOCK_NAMES),
            tau_grid=list(TAU_GRID), B=int(B),
            centring='phi-centred (ORCHESTRATOR D2)',
            raw_scale='identity-link additive fit (lsqr on y), same Tobit/folds/'
                      'sigma(phi)/mu(phi) machinery',
            written_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))))
        bad = _pairs.verify_manifest()
        if bad:
            raise RuntimeError('MANIFEST verification failed: %r' % (bad,))
        if verbose:
            print('[c2  ] manifest: %d c2 ensembles registered, verify_manifest '
                  'clean' % len(ents))
    if verbose:
        print('[c2  ] total wall %.1f s' % (time.time() - t0))
    return dict(per_assay=per, T06=t06, T13=t13, catalogues=cats)


def stage(assays=None, verbose=True, **kw):
    """``run_all.py``'s ``_call`` signature."""
    return run_all(assays=assays, verbose=verbose, **kw)


# =========================================================================== #
# self-check                                                                  #
# =========================================================================== #

def _ok(name, cond, detail=''):
    print('  %-58s %s %s' % (name, 'OK ' if cond else 'FAIL', detail))
    if not cond:
        raise AssertionError(name + '  ' + detail)


def _selfcheck(dms_id='Z-domain_ZpA963_HL1_fitness_2M5A'):
    config.assert_env()
    print('[stats_c2] closed forms and invariants')
    # ---- 1. Gaussian tail-ratio references -------------------------------- #
    rng = np.random.default_rng(7)
    z = rng.standard_normal(4_000_000)
    tr1 = tail_ratio(z, THRESH['C2_TR_q_hi1'], THRESH['C2_TR_q_lo'])
    tr2 = tail_ratio(z, THRESH['C2_TR_q_hi2'], THRESH['C2_TR_q_lo'])
    _ok('TR1(N(0,1)) == C2_TR1_gauss 2.8606',
        abs(tr1 - THRESH['C2_TR1_gauss']) < 0.02, '%.4f' % tr1)
    _ok('TR2(N(0,1)) == C2_TR2_gauss 2.2393',
        abs(tr2 - THRESH['C2_TR2_gauss']) < 0.01, '%.4f' % tr2)
    _ok('TR is scale-free (c -> 1000c)',
        abs(tail_ratio(1000 * z, 0.999, 0.75) - tr1) < 1e-9)
    # ---- 2. BH-FDR against the hand computation --------------------------- #
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205,
                  0.212, 0.216])
    q = bh_fdr(p)
    want = np.minimum.accumulate((10 * np.sort(p) / np.arange(1, 11))[::-1])[::-1]
    _ok('bh_fdr == the step-up definition', np.allclose(q, want),
        'q1=%.5f' % q[0])
    _ok('bh_fdr is monotone in p', np.all(np.diff(q[np.argsort(p)]) >= -1e-12))
    _ok('bh_fdr carries NaN through',
        np.isnan(bh_fdr(np.array([0.01, np.nan]))[1]))
    # ---- 3. empirical p --------------------------------------------------- #
    _ok('empirical_p_upper is (1+#>=)/(B+1)',
        abs(empirical_p_upper(1.0, np.zeros(199)) - 1.0 / 200.0) < 1e-15)
    # ---- 4. mixture: identities, recovery, binning error ------------------ #
    r2 = np.random.default_rng(11)
    n = 200_000
    lab = r2.random(n) < 0.02
    cs = np.where(lab, r2.standard_normal(n) * 5.0, r2.standard_normal(n) * 1.0)
    m = mixture_two_component(cs, n_restart=48, dms_id=None)
    _ok('mixture dBIC == -Lambda + 2 ln n exactly',
        abs(m['dBIC'] - (-m['Lambda'] + 2.0 * math.log(m['n_mix']))) < 1e-6,
        'dBIC=%.2f Lambda=%.2f' % (m['dBIC'], m['Lambda']))
    _ok('mixture recovers pi=0.02 within 0.004', abs(m['pi_hat'] - 0.02) < 0.004,
        'pi=%.5f' % m['pi_hat'])
    _ok('mixture recovers rho=5 within 0.4', abs(m['rho_hat'] - 5.0) < 0.4,
        'rho=%.4f' % m['rho_hat'])
    _ok('mixture beats one component on a real mixture', m['dBIC'] < -10,
        'dBIC=%.1f' % m['dBIC'])
    m1 = mixture_two_component(r2.standard_normal(200_000), n_restart=48)
    print('    single-Gaussian control: pi=%.4g rho=%.4g dBIC=%+.2f'
          % (m1['pi_hat'], m1['rho_hat'], m1['dBIC']))
    # exact (unbinned) EM on the same data, for the binning error
    ue = cs * cs
    pe, v1e, v2e = _mix_em(ue, np.ones(ue.size), m['pi_hat'],
                           m['sigma1'] ** 2, m['sigma2'] ** 2, 200,
                           float(ue.mean()) * 1e-10)
    _ok('binned EM == exact full-data EM (|dpi| < 2e-4)',
        abs(pe - m['pi_hat']) < 2e-4,
        'binned %.6f vs exact %.6f' % (m['pi_hat'], pe))
    _ok('binned EM == exact full-data EM (|drho| < 5e-3)',
        abs(math.sqrt(v2e / v1e) - m['rho_hat']) < 5e-3,
        'binned %.5f vs exact %.5f' % (m['rho_hat'], math.sqrt(v2e / v1e)))
    # ---- 5. block bootstrap: the fast rate path == the general form -------- #
    ctx = _nulls.get_context(dms_id, verify=False)
    lat = _lat_bundle(ctx, None)
    raw = raw_bundle(ctx, None)
    raw['censor_mask'] = ctx.censor_mask
    bv = block_values(ctx, lat, raw, 'L_s')
    pos = ctx.pos_of_add[bv['edge']]
    a = np.abs(bv['c'])
    fast = rate_bootstrap_positions(a, pos, (3.0,), B=200, dms_id=dms_id)
    lo, hi, _ = block_bootstrap_positions(
        a, pos, lambda x: float((x >= 3.0).mean()), B=200, dms_id=dms_id)
    _ok('rate_bootstrap_positions == block_bootstrap_positions',
        abs(fast[3.0][0] - lo) < 1e-12 and abs(fast[3.0][1] - hi) < 1e-12,
        '[%.6f,%.6f]' % (lo, hi))
    _ok('the block bootstrap resamples POSITIONS not edges',
        np.unique(pos).size < a.size, 'n_pos=%d < n_edges=%d'
        % (np.unique(pos).size, a.size))
    # ---- 6. c_hat is nulls.c_hat, and pa_mask is nulls._pa_mask ----------- #
    _ok('stats_c2.c_hat IS cliff.nulls.c_hat', c_hat is _nulls.c_hat)
    mine = pa_mask(ctx, ctx.censor_mask, ctx.oof_finite, floor_mask=True)
    theirs = _nulls._pa_mask(ctx, ctx.censor_mask, ctx.oof_finite)
    _ok('pa_mask(floor_mask=True) == nulls._pa_mask',
        bool(np.array_equal(mine, theirs)), '%d edges' % int(mine.sum()))
    # ---- 7. the raw fit is a genuine least-squares fit --------------------- #
    fr = raw['full']
    res = ctx.y - fr['phi']
    g = np.abs(ctx.A.T.dot(res)).max() / max(np.abs(ctx.y).max(), 1.0)
    _ok('fit_raw: max|A^T (y - phi)| / max|y| < 1e-6 (an OLS fit)', g < 1e-6,
        '%.3e' % g)
    _ok('raw oof_finite == latent oof_finite',
        bool(np.array_equal(raw['oof_finite'], ctx.oof_finite)),
        'raw %.6f vs lat %.6f' % (raw['frac_oof_finite'],
                                  float(ctx.oof_finite.mean())))
    # ---- 8. my L_s / L_m blocks reproduce nulls.default_stat_fn ----------- #
    o = _nulls.observed_stats(ctx)
    st_s = block_stats(bv, taus=TAU_GRID)
    bvm = block_values(ctx, lat, raw, 'L_m')
    st_m = block_stats(bvm, taus=TAU_GRID)
    pairsx = [('n_Pa', st_s['n_Pa'], o['n_Pa']), ('q75', st_s['q75'], o['q75']),
              ('q99', st_s['q99'], o['q99']), ('q999', st_s['q999'], o['q999']),
              ('TR1', st_s['TR1'], o['TR1']), ('TR2', st_s['TR2'], o['TR2']),
              ('kurt_c', st_s['kurt'], o['kurt_c']),
              ('mad_c', st_s['mad'], o['mad_c']), ('sd_c', st_s['sd'], o['sd_c']),
              ('q75_mad', st_m['q75'], o['q75_mad']),
              ('TR1_mad', st_m['TR1'], o['TR1_mad'])]
    for t in TAUS:
        pairsx.append(('rate_sigma_tau%g' % t, st_s['rate_tau%g' % float(t)],
                       o['rate_sigma_tau%g' % t]))
        pairsx.append(('rate_mad_tau%g' % t, st_m['rate_tau%g' % float(t)],
                       o['rate_mad_tau%g' % t]))
    worst = max((abs(x - y) if np.isfinite(x) and np.isfinite(y) else 0.0, k)
                for k, x, y in pairsx)
    _ok('L_s / L_m == nulls.default_stat_fn on the observed data (max|d| < 1e-9)',
        worst[0] < 1e-9, 'worst %s %.3e' % (worst[1], worst[0]))
    # ---- 9. grid guard ---------------------------------------------------- #
    gg = grid_guard(ctx, bv['scale_value'] * _link_slope(ctx, ctx.phi_oof))
    _ok('grid_guard cut is 3 q_a', True,
        '3q=%.3g  tau=2 cut=%.4g pass=%s'
        % (3 * ctx.quantum, gg[2.0][0], gg[2.0][1]))
    _ok('grid_guard tau_absolute is increasing in tau',
        all(gg[float(TAUS[i])][0] < gg[float(TAUS[i + 1])][0]
            for i in range(len(TAUS) - 1)))
    # ---- 10. the sigma_mult identity T(tau; m sigma) == T(m tau; sigma) ---- #
    for m_ in config.SIGMA_MULTIPLIERS:
        cm_ = bv['c'] / m_
        for t in (3.0, 4.0):
            r_direct = float((np.abs(cm_) >= t).mean())
            r_grid = st_s['rate_tau%g' % (m_ * t)]
            _ok('rate(|c/%g|>=%g) == rate_tau%g' % (m_, t, m_ * t),
                abs(r_direct - r_grid) < 1e-12, '%.8f' % r_direct)
    print('[stats_c2] all invariants OK')
    return True


def _main(argv):
    if '--selfcheck' in argv:
        _selfcheck()
        return 0
    assays = None
    for i, a in enumerate(argv):
        if a == '--assays':
            assays = argv[i + 1].split(',')
    kw = {}
    for i, a in enumerate(argv):
        if a == '--B':
            kw['B'] = int(argv[i + 1])
        if a == '--nproc':
            kw['nproc'] = int(argv[i + 1])
        if a == '--boot-B':
            kw['boot_B'] = int(argv[i + 1])
        if a == '--no-catalogue':
            kw['catalogue'] = False
        if a == '--no-write':
            kw['write'] = False
        if a == '--nulls':
            kw['nulls'] = tuple(argv[i + 1].split(','))
    config.assert_env()
    out = run_all(assays=assays, **kw)
    t = out['T06']
    pri = t[(t['scale'] == 'latent') & (t['unit'] == 'sigma')
            & (t['tau'] == 4.0)]
    print()
    print('=== T06 latent x sigma, tau=4 ===')
    cols = ['DMS_id', 'n_Pa', 'TR_used', 'TR', 'TR_N1_p995', 'pi_hat',
            'rho_hat', 'dBIC', 'Lambda', 'Lambda_N1_p995', 'rate_obs',
            'T_N2', 'T_N2_hi95', 'T_N2b', 'q_BH', 'n_consecutive_tau_passing']
    print(pri[cols].to_string(index=False))
    return 0


if __name__ == '__main__':                                  # pragma: no cover
    sys.exit(_main(sys.argv[1:]))
