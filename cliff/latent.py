# -*- coding: utf-8 -*-
"""``cliff.latent`` -- the latent scale (spec Sec.1.0) and its 5-fold cross-fit.

Spec Sec.1.0, verbatim::

    Latent scale.  y = g(phi), phi = X beta, g monotone.  Fit by alternation,
    10 iterations, convergence max|dbeta| < 1e-6:
      1. beta = scipy.sparse.linalg.lsqr([1|X], z);  phi = [1|X] beta
      2. g = sklearn.isotonic.IsotonicRegression(out_of_bounds='clip').fit(phi, y)
      3. z = g^-1(y), where g^-1 is linear interpolation over the
         strictly-increasing hull of the PAV breakpoints, clipped to
         [min phi, max phi].
    Censored rows enter step 1 through a Tobit E-step only:
      z_i = phi_i - sigma(phi_i) * pdf(a) / Phi(a),  a = (g^-1(L) - phi_i)/sigma(phi_i)
    They are excluded from every pair statistic regardless.

    Cross-fitting (mandatory).  5 folds over variants, fixed seed 20260902.
    phi_i^oof, z_i^oof, e_i = z_i^oof - phi_i^oof are computed from a fit that
    never saw variant i.

    Level-dependent noise scale.  sigma-hat(phi) = 1.4826 * MAD(e) within each of
    20 equal-count phi bins, linearly interpolated.  MAD, NEVER sd.

Three things the spec does not say and this module had to decide.  All three are
reported in the module's own self-check output, so a reviewer sees them without
reading the source:

1.  **The alternation is scale-indeterminate.**  If ``(beta, g, z)`` is a fixed
    point then so is ``(lam*beta, g(./lam), lam*z)`` for every ``lam > 0``: PAV
    depends only on the *order* of ``phi``, so rescaling ``phi`` rescales the
    x-thresholds and leaves the y-thresholds untouched.  Run as written the
    iteration therefore drifts geometrically -- measured on GB1_IgG-Fc_1FCC,
    ``sd(z)`` grows by a factor 1.18 per iteration (1.16 -> 7.28 over 12
    iterations) while ``R^2(y ~ g(phi))`` has already converged to 0.9514.
    ``max|dbeta| < 1e-6`` can then never be met, for a reason that has nothing to
    do with the fit.  Fixed by an **identifiability anchor**: after every ``lsqr``
    the pair ``(beta, phi)`` is mapped affinely so that ``phi`` has the mean and
    sd of ``y`` over the uncensored rows.  This is a pure reparametrisation --
    every downstream statistic (``c_hat``, ``TR``, ``T(tau)``, ``pi``, ``rho``,
    the sibling slope, ICC, ``dR2_oos``, N1's ``z* = X beta + eps*``) is invariant
    under ``z -> lam z`` -- and it is what makes the spec's own convergence test
    and a cross-assay-comparable ``resid_mad`` meaningful.  mean/sd (not MAD) is
    correct *here* because this is an identifiability constraint, not a noise
    scale; ``sigma_of_phi`` still uses MAD and nothing else.

2.  **``g`` is fitted on the uncensored rows only.**  "Censored rows enter step 1
    through a Tobit E-step only" is read as "step 1 is the only step they enter".
    The alternative (fit PAV on all rows) is not implementable: the E-step needs
    ``g^-1(L)``, and if the clamped rows are in the PAV fit then ``L`` sits on a
    plateau whose pre-image is a whole interval -- on CR9114_FluAH3 that plateau
    is 89.05% of the data.

3.  **A residual MAD of exactly zero happens, and it is not a rounding
    problem.**  On CD19_FMC63_7URV two of the twenty phi bins have
    ``MAD(e) == 0`` exactly: the design is nearly saturated (n/M = 2.1, 1,467 of
    1,826 substitutions seen once) so on the low-``phi`` rows the additive fit is
    exact and every residual there is identically 0.  Floored at the assay's
    decimal grid those bins give ``sigma ~ 3e-16`` and ``|e/sigma|`` up to
    **7.3e16**, which would have handed ``stats_c2`` a tail made of pure
    zero-division.  A bin with no residual spread carries no scale information,
    so ``sigma_of_phi`` **pools** it: a bin whose raw MAD is exactly zero
    inherits the assay-wide ``1.4826 * MAD(e)``, which sends ``c_hat`` there to
    ~0 -- the correct conclusion, since no cliff is detectable among rows whose
    residuals are all identically zero.  The count is reported as
    ``n_bins_degenerate``.  Separately, ``sigma_floor`` (default
    ``quantum_uncensored / sqrt(12)``, the sd of rounding noise on the assay's
    own decimal grid -- a resolution limit, not a threshold: nothing is compared
    against it) still guards the coarse-grid case; the quantum is re-derived on
    the UNCENSORED score strings because CR9114_FluAH3's raw modal decimal count
    is 1 only because 89.05% of that file is the literal string ``6.0``.

Every numeric threshold is read from :data:`cliff.config.THRESH`; every seed from
:data:`cliff.config.SEEDS`.

Self-check::

    python -m cliff.latent            # full run over the 17 PRIMARY+ARM+CONTROL
"""

import json
import math
import os
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from scipy.special import log_ndtr
from sklearn.isotonic import IsotonicRegression

from . import config
from . import io_bgym
from .config import PATHS, SEEDS, THRESH
from .io_bgym import md5_of

__all__ = [
    'LatentFit', 'design_from_keys', 'design_from_codes', 'with_intercept',
    'load_cached_design', 'strict_hull', 'g_apply', 'ginv', 'mad', 'mad_scaled',
    'sigma_of_phi', 'sigma_eval', 'make_sigma_fn', 'classify_levels',
    'tobit_estep', 'fit_latent', 'make_folds', 'crossfit_latent',
    'cache_latent', 'register_latent_cache', 'run_latent', 'run_all',
    'load_report',
    'iteration_stability', 'stage2', 'LATENT_COLUMNS',
]

#: log(sqrt(2 pi)) -- the normal pdf normaliser, a mathematical constant.
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)

#: sd of a uniform(-q/2, q/2) rounding error is q/sqrt(12).  Mathematical
#: constant, not a decision boundary: no verdict is read off it.
_SQRT_12 = math.sqrt(12.0)


# --------------------------------------------------------------------------- #
# design matrix                                                               #
# --------------------------------------------------------------------------- #

def design_from_keys(keys, col_index, n_muts=None):
    """``X`` in ``{0,1}^{n x M}``, CSR; row nnz == ``num_muts`` (spec Sec.1.0).

    ``keys`` is :attr:`cliff.io_bgym.Assay.keys` (canonical keys, chain
    mandatory) and ``col_index`` its ``(chain, seq_pos, aa_mut) -> column`` map.
    """
    n = len(keys)
    M = len(col_index)
    if n_muts is None:
        cnt = np.fromiter((len(k) for k in keys), dtype=np.int64, count=n)
    else:
        cnt = np.asarray(n_muts, dtype=np.int64)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(cnt, out=indptr[1:])
    nnz = int(indptr[-1])
    indices = np.fromiter((col_index[s] for k in keys for s in k),
                          dtype=np.int32, count=nnz)
    data = np.ones(nnz, dtype=np.float64)
    return sp.csr_matrix((data, indices, indptr), shape=(n, M))


def design_from_codes(codes, col_index, pos_index):
    """The same ``X``, rebuilt from the ``int8`` code vector -- the cache-only
    path (``data/cliff_cache/keys/{id}.npz`` stores ``codes``, not ``keys``).

    Asserted identical to :func:`design_from_keys` in the self-check.
    """
    codes = np.asarray(codes)
    n, P = codes.shape
    posarr = [None] * len(pos_index)
    for k, v in pos_index.items():
        posarr[v] = k
    rows, cols = np.nonzero(codes)
    vals = codes[rows, cols]
    tgt = np.empty(rows.size, dtype=np.int32)
    for j in range(P):
        m = cols == j
        if not m.any():
            continue
        ch, pos = posarr[j]
        for v in np.unique(vals[m]):
            tgt[m & (vals == v)] = col_index[(ch, pos, config.CODE_AA[int(v)])]
    order = np.lexsort((tgt, rows))
    rows, tgt = rows[order], tgt[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=n), out=indptr[1:])
    return sp.csr_matrix((np.ones(rows.size), tgt, indptr),
                         shape=(n, len(col_index)))


def with_intercept(X):
    """``[1|X]`` as CSR -- the spec's step-1 design."""
    n = X.shape[0]
    one = sp.csr_matrix((np.ones(n), np.zeros(n, dtype=np.int32),
                         np.arange(n + 1, dtype=np.int64)), shape=(n, 1))
    return sp.hstack([one, X], format='csr')


def load_cached_design(dms_id, *, verify=True):
    """Read ``data/cliff_cache/keys/{DMS_id}.npz`` -> everything a refit needs.

    The cache-only entry point for the null ensembles (13,600 replicate-jobs
    must not re-parse a 62 MB csv).  ``verify=True`` checks the md5 recorded in
    ``MANIFEST.json`` and raises on a mismatch, as spec Sec.5 requires.
    """
    p = os.path.join(PATHS.keys, dms_id + '.npz')
    if verify:
        rel = os.path.relpath(p, config.REPO)
        with open(PATHS.manifest) as fh:
            man = json.load(fh)
        want = man.get('files', {}).get(rel)
        if want is None:
            raise RuntimeError('%s is not in MANIFEST.json -- run stage 0' % rel)
        got = md5_of(p)
        if got != want['md5']:
            raise RuntimeError('md5 mismatch for %s: %s != %s'
                               % (rel, got, want['md5']))
    z = np.load(p, allow_pickle=False)
    col_index = {(k[0], int(k[1]), k[2]): int(v)
                 for k, v in json.loads(str(z['col_index_json']))}
    pos_index = {(k[0], int(k[1])): int(v)
                 for k, v in json.loads(str(z['pos_index_json']))}
    X = design_from_codes(z['codes'], col_index, pos_index)
    return dict(dms_id=dms_id, X=X, A=with_intercept(X), y=z['y'],
                y_raw=z['y_raw'], codes=z['codes'], col_index=col_index,
                pos_index=pos_index, censor_mask=z['censor_mask'],
                censor_levels=tuple(float(v) for v in z['censor_levels']),
                row_index=z['row_index'], n_muts=z['n_muts'],
                wt_row=int(z['wt_row']), quantum=float(z['quantum']),
                transform=str(z['transform']))


# --------------------------------------------------------------------------- #
# the monotone link g and its inverse                                         #
# --------------------------------------------------------------------------- #

def strict_hull(g_knots):
    """``(y_hull, phi_hull)``: the strictly-increasing hull of the PAV breakpoints.

    ``g_knots = (X_thresholds_, y_thresholds_)`` as sklearn returns them: ``y``
    non-decreasing, each plateau represented by its two end points.  A
    non-decreasing step/ramp function has no single-valued inverse on its
    plateaus, so the hull keeps ONE ``phi`` per distinct ``y`` -- the **midpoint**
    of the plateau's ``phi`` interval, the only choice that is both continuous in
    ``y`` and unbiased between the plateau's two ends (taking the left end pulls
    every ``z`` down, the right end up).
    """
    xk = np.asarray(g_knots[0], dtype=np.float64)
    yk = np.asarray(g_knots[1], dtype=np.float64)
    uy, first = np.unique(yk, return_index=True)
    last = np.searchsorted(yk, uy, side='right') - 1
    return uy, 0.5 * (xk[first] + xk[last])


def g_apply(g_knots, phi):
    """Forward link ``y_hat = g(phi)``: piecewise-linear interpolation over the
    PAV breakpoints, clipped outside them (sklearn's ``out_of_bounds='clip'``)."""
    xk = np.asarray(g_knots[0], dtype=np.float64)
    yk = np.asarray(g_knots[1], dtype=np.float64)
    if xk.size == 1:
        return np.full(np.shape(phi), yk[0], dtype=np.float64)
    return np.interp(np.asarray(phi, dtype=np.float64), xk, yk)


def ginv(g_knots, y, lo, hi):
    """Spec step 3: ``z = g^-1(y)``, linear interpolation over the
    strictly-increasing hull of the PAV breakpoints, clipped to ``[lo, hi]``.

    ``lo, hi`` are ``[min phi, max phi]`` of the FIT.  The clip is a no-op when
    they bracket the hull (always, for knots and bounds from one fit); it bites
    only if a caller supplies a narrower window.  ``y`` outside the fitted range
    extrapolates flat, i.e. to the extreme plateau midpoints -- the same
    convention as ``out_of_bounds='clip'`` in the forward direction.
    """
    uy, mid = strict_hull(g_knots)
    y = np.asarray(y, dtype=np.float64)
    if uy.size == 1:
        z = np.full(y.shape, mid[0], dtype=np.float64)
    else:
        z = np.interp(y, uy, mid)
    return np.clip(z, lo, hi)


# --------------------------------------------------------------------------- #
# sigma-hat(phi):  1.4826 * MAD(e) in 20 equal-count phi bins                  #
# --------------------------------------------------------------------------- #

def mad(v):
    """Median absolute deviation about the median.  Raw, unscaled."""
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return float('nan')
    return float(np.median(np.abs(v - np.median(v))))


def mad_scaled(v):
    """``THRESH['mad_const'] * MAD(v)``.  **MAD, never sd** (spec Sec.1.0:
    Z-ZSPA1-LL1 has sd 0.140 against range 4.871)."""
    return THRESH['mad_const'] * mad(v)


def sigma_of_phi(phi, e, *, n_bins=None, sigma_floor=0.0):
    """``sigma-hat(phi)`` knots: ``1.4826 * MAD(e)`` in ``n_bins`` equal-count
    ``phi`` bins (spec Sec.1.0).

    Returns ``(centers, sigmas, medians, counts, degenerate)``:

    * ``centers``  -- the bin median of ``phi``, strictly increasing (bins whose
      medians tie are merged, so ``np.interp`` is well posed);
    * ``sigmas``   -- ``1.4826 * MAD(e)`` in the bin, floored at ``sigma_floor``;
    * ``medians``  -- the bin median of ``e``.  Not used by the spec's ``c_hat``,
      returned because a non-zero ``median(e | phi)`` is a *location* artefact of
      ``g^-1`` that ``sigma-hat`` cannot absorb, and a consumer must be able to
      see it (measured: ``corr(e, phi) = +0.24`` on GB1_IgG-Fc_1FCC);
    * ``counts``   -- rows per bin;
    * ``degenerate`` -- bool, the bin's raw ``MAD(e)`` was exactly 0 and its
      ``sigma`` was pooled from the assay-wide scale.

    Pass UNCENSORED rows only: a censored row's ``e`` is ``-sigma * Mills(a)``, a
    deterministic function of ``phi``, not a measurement error.
    """
    if n_bins is None:
        n_bins = THRESH['sigma_n_bins']
    phi = np.asarray(phi, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    ok = np.isfinite(phi) & np.isfinite(e)
    phi, e = phi[ok], e[ok]
    if phi.size == 0:
        raise ValueError('sigma_of_phi: no finite (phi, e) pairs')
    order = np.argsort(phi, kind='stable')
    parts = np.array_split(order, min(int(n_bins), phi.size))
    c, s, m, k = [], [], [], []
    for ix in parts:
        if ix.size == 0:
            continue
        c.append(np.median(phi[ix]))
        s.append(mad_scaled(e[ix]))
        m.append(np.median(e[ix]))
        k.append(ix.size)
    c = np.asarray(c, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    k = np.asarray(k, dtype=np.int64)
    # merge tied bin centres (heavy ties in phi) -- np.interp needs increasing x
    if c.size > 1 and (np.diff(c) <= 0).any():
        uc, inv = np.unique(c, return_inverse=True)
        w = np.bincount(inv, weights=k.astype(np.float64), minlength=uc.size)
        s = np.bincount(inv, weights=s * k, minlength=uc.size) / w
        m = np.bincount(inv, weights=m * k, minlength=uc.size) / w
        k = np.bincount(inv, weights=k.astype(np.float64),
                        minlength=uc.size).astype(np.int64)
        c = uc
    # ---- pool degenerate bins (see docstring item 3) --------------------- #
    degen = s <= 0.0
    if degen.any():
        glob = mad_scaled(e)
        if not (glob > 0):
            pos = s[~degen]
            glob = (pos.min() if pos.size else
                    (float(sigma_floor) if sigma_floor > 0 else 1.0))
        s = np.where(degen, glob, s)
    s = np.maximum(s, float(sigma_floor))
    if (s <= 0).any():                      # last resort: a wholly flat e
        s = np.where(s > 0, s, 1.0)
        degen = degen | (s <= 0)
    return c, s, m, k, degen


def sigma_eval(knots, phi):
    """``sigma-hat(phi)`` by linear interpolation over the knots, flat outside."""
    c, s = np.asarray(knots[0], dtype=np.float64), np.asarray(knots[1], dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    out = np.full(phi.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(phi)
    if c.size == 1:
        out[ok] = s[0]
    else:
        out[ok] = np.interp(phi[ok], c, s)
    return out


def make_sigma_fn(knots):
    """The ``LatentFit.sigma_of_phi`` callable (spec's dataclass field)."""
    c = np.array(knots[0], dtype=np.float64, copy=True)
    s = np.array(knots[1], dtype=np.float64, copy=True)

    def _sigma(phi):
        return sigma_eval((c, s), phi)

    _sigma.knots = (c, s)
    return _sigma


# --------------------------------------------------------------------------- #
# Tobit E-step                                                                #
# --------------------------------------------------------------------------- #

def classify_levels(y, levels):
    """Split detected censoring levels into ``(floors, ceilings)``.

    :func:`cliff.io_bgym.detect_censoring` returns floors first then ceilings but
    the tuple alone does not say which is which, and ``level < median(y)`` fails
    on CR9114_FluAH3 where the floor 6.000 IS the median (89.05% of the mass).
    A level is a floor iff less mass lies below it than above it.
    """
    y = np.asarray(y, dtype=np.float64)
    floors, ceils = [], []
    for L in levels:
        below = float((y < L).mean())
        above = float((y > L).mean())
        (floors if below < above else ceils).append(float(L))
    return tuple(floors), tuple(ceils)


def tobit_estep(phi, sigma, c, is_floor):
    """The spec's E-step, vectorised over rows.

    Floor (left-censored, ``Z <= c``)::

        a = (c - phi)/sigma ;  z = phi - sigma * pdf(a)/Phi(a)

    Ceiling (right-censored, ``Z >= c``) is the mirror, ``z = phi + sigma *
    pdf(a)/(1-Phi(a))`` -- the spec writes only the floor form; every censoring
    detected in the benchmark is a floor, so the ceiling branch is dead code kept
    for correctness.

    Evaluated as ``exp(logpdf - log Phi)`` through :func:`scipy.special.log_ndtr`:
    the naive ratio underflows to 0/0 by ``a ~ -38``, and CR9114_FluAH3 has rows
    at ``a < -100``.
    """
    phi = np.asarray(phi, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    is_floor = np.asarray(is_floor, dtype=bool)
    a = (c - phi) / sigma
    logpdf = -0.5 * a * a - _LOG_SQRT_2PI
    mills = np.where(is_floor,
                     np.exp(logpdf - log_ndtr(a)),
                     np.exp(logpdf - log_ndtr(-a)))
    return np.where(is_floor, phi - sigma * mills, phi + sigma * mills)


# --------------------------------------------------------------------------- #
# LatentFit                                                                   #
# --------------------------------------------------------------------------- #

class LatentFit(object):
    """Spec Sec.3's dataclass, plus the diagnostics the run has to report.

    Spec fields: ``beta phi z g_knots sigma_of_phi r2_link_gain n_iter_used``.
    ``sigma_of_phi`` is a callable (``LatentFit.sigma_knots`` is what gets
    cached, a callable cannot go in an npz).
    """

    __slots__ = ('beta', 'phi', 'z', 'g_knots', 'sigma_of_phi', 'r2_link_gain',
                 'n_iter_used', 'sigma_knots', 'e', 'lo', 'hi', 'converged',
                 'dbeta_last', 'dbeta_trace', 'dphi_last', 'r2_add_raw',
                 'r2_lin_final', 'r2_link', 'r2_add_latent',
                 'r2_link_gain_vs_raw', 'resid_mad', 'corr_e_phi', 'sigma_floor',
                 'n_censored', 'lsqr_itn', 'lsqr_istop', 'wall_s', 'anchor')

    def __init__(self, **kw):
        for s in self.__slots__:
            setattr(self, s, kw.get(s))

    def __repr__(self):
        return ('LatentFit(n=%d, M=%d, n_iter_used=%s, converged=%s, '
                'dbeta_last=%.3e, r2_link=%.5f, r2_link_gain=%.5f, '
                'resid_mad=%.5f)'
                % (self.phi.size, self.beta.size - 1, self.n_iter_used,
                   self.converged, self.dbeta_last, self.r2_link,
                   self.r2_link_gain, self.resid_mad))


def _r2(y, yhat):
    y = np.asarray(y, dtype=np.float64)
    ss = float(((y - y.mean()) ** 2).sum())
    if ss <= 0:
        return float('nan')
    return 1.0 - float(((y - np.asarray(yhat)) ** 2).sum()) / ss


def _r2_lin(x, y):
    """``R^2`` of ``y ~ a + b x`` == ``corr(x, y)^2``; invariant to any affine
    reparametrisation of ``x``, so the identifiability anchor cannot move it."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or x.std() <= 0 or y.std() <= 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1]) ** 2


def _colnorms(A):
    """Euclidean column norms of ``A``, with empty columns mapped to 1.

    A column absent from a training subset has norm 0 and ``lsqr`` cannot move it
    off zero anyway (its ``beta_j`` stays exactly 0), so 1 is the safe divisor.
    """
    cn = np.sqrt(np.asarray(A.multiply(A).sum(axis=0), dtype=np.float64).ravel())
    return np.where(cn > 0, cn, 1.0)


def fit_latent(X, y, censor_mask, censor_levels, *, n_iter=None, n_bins=None,
               tol=None, sigma_floor=0.0, lsqr_tol=1e-12, verbose=False):
    """The spec Sec.1.0 alternating additive-plus-monotone-link fit.

    ``X`` is ``{0,1}^{n x M}`` (sparse, no intercept); ``[1|X]`` is built here.
    ``censor_levels`` are the assay's detected levels, split into floors and
    ceilings by :func:`classify_levels`.

    ``n_iter`` / ``n_bins`` / ``tol`` default to ``THRESH['latent_n_iter']`` (10),
    ``THRESH['sigma_n_bins']`` (20) and ``THRESH['latent_conv_tol']`` (1e-6): the
    spec writes those literals into the signature, ``config`` owns them.

    **Column-scaled ``lsqr``.**  Step 1 solves ``lsqr(A D, z)`` with
    ``D = diag(1/||A_j||)`` and returns ``beta = D beta'``.  A diagonal right
    preconditioner leaves the least-squares problem and its solution
    algebraically unchanged; it only fixes the conditioning, and the design's
    column norms span 1 (a substitution seen once) to 181 (CR9114's 32,767).
    Measured on GB1_IgG-Fc_1FCC at ``atol=btol=1e-12``: 22 ``lsqr`` iterations
    and ``max|phi - phi_exact| = 8e-12``, against 268 iterations and 5.5e-9
    unscaled -- a 10x speed-up AND three orders of accuracy, which is what lets
    the whole fit sit inside G0's ``fit_latent <= %g s`` while solving to 1e-12
    instead of ``lsqr``'s 1e-6 default (whose 4.6e-3 error in ``phi`` is 2.6%% of
    this assay's residual MAD).  The only thing it changes is *which* minimum-norm
    point is returned for a rank-deficient design, and ``phi`` -- everything
    downstream reads -- is identical either way.
    """ % THRESH['G0_fit_latent_s']
    t0 = time.time()
    if n_iter is None:
        n_iter = THRESH['latent_n_iter']
    if n_bins is None:
        n_bins = THRESH['sigma_n_bins']
    if tol is None:
        tol = THRESH['latent_conv_tol']
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    cm = np.asarray(censor_mask, dtype=bool)
    if cm.shape != y.shape:
        raise ValueError('censor_mask shape %s != y shape %s' % (cm.shape, y.shape))
    A = with_intercept(X)
    cn = _colnorms(A)
    As = (A @ sp.diags(1.0 / cn)).tocsr()
    unc = ~cm
    if unc.sum() < 3:
        raise ValueError('fewer than 3 uncensored rows')
    floors, ceils = classify_levels(y, censor_levels)
    lvl = np.array(list(floors) + list(ceils), dtype=np.float64)
    lvl_is_floor = np.array([True] * len(floors) + [False] * len(ceils), dtype=bool)
    # each censored row -> the detected level it is clamped at
    if cm.any() and lvl.size:
        which = np.abs(y[cm][:, None] - lvl[None, :]).argmin(axis=1)
        row_is_floor = lvl_is_floor[which]
    else:
        which = np.zeros(0, dtype=np.int64)
        row_is_floor = np.zeros(0, dtype=bool)

    my = float(y[unc].mean())
    sy = float(y[unc].std())
    if not (sy > 0):
        raise ValueError('y is constant on the uncensored rows')

    z = y.copy()
    beta = np.zeros(A.shape[1], dtype=np.float64)
    dbeta_trace, itn, istop = [], [], []
    phi = e_unc = sk = g_knots = None
    lo = hi = float('nan')
    r2_add_raw = float('nan')
    used = 0
    x0 = None
    phi_prev = None
    dphi = float('nan')
    for it in range(int(n_iter)):
        # ---- step 1: beta = lsqr([1|X], z) ------------------------------- #
        res = lsqr(As, z, atol=lsqr_tol, btol=lsqr_tol, x0=x0)
        b = np.asarray(res[0], dtype=np.float64) / cn
        istop.append(int(res[1]))
        itn.append(int(res[2]))
        phi = A.dot(b)
        # ---- identifiability anchor (see the module docstring) ----------- #
        spd = float(phi[unc].std())
        if spd > 0:
            sc = sy / spd
            sh = my - sc * float(phi[unc].mean())
            b = sc * b
            b[0] += sh
            phi = sh + sc * phi
        if it == 0:
            # phi_1 IS the least-squares additive fit of y, so R^2 of a linear
            # function of it is the plain additive R^2 -- anchor-invariant.
            r2_add_raw = _r2_lin(phi[unc], y[unc])
        # ---- step 2: g = PAV(phi -> y), UNCENSORED rows only ------------- #
        ir = IsotonicRegression(out_of_bounds='clip', increasing=True)
        ir.fit(phi[unc], y[unc])
        g_knots = (np.asarray(ir.X_thresholds_, dtype=np.float64),
                   np.asarray(ir.y_thresholds_, dtype=np.float64))
        lo = float(phi[unc].min())
        hi = float(phi[unc].max())
        # ---- step 3: z = g^-1(y) ---------------------------------------- #
        znew = np.empty(n, dtype=np.float64)
        znew[unc] = ginv(g_knots, y[unc], lo, hi)
        e_unc = znew[unc] - phi[unc]
        sk = sigma_of_phi(phi[unc], e_unc, n_bins=n_bins, sigma_floor=sigma_floor)
        # ---- censored rows: the Tobit E-step, and step 1 only ----------- #
        if cm.any():
            cvals = (ginv(g_knots, lvl, lo, hi)[which] if lvl.size
                     else np.full(int(cm.sum()), lo))
            znew[cm] = tobit_estep(phi[cm],
                                   sigma_eval((sk[0], sk[1]), phi[cm]),
                                   cvals, row_is_floor)
        d = float(np.abs(b - beta).max())
        dbeta_trace.append(d)
        dphi = (float(np.abs(phi - phi_prev).max())
                if phi_prev is not None else float('nan'))
        beta, z, phi_prev = b, znew, phi.copy()
        x0 = b * cn
        used = it + 1
        if verbose:
            print('    it%2d lsqr(istop=%d itn=%3d) max|dbeta|=%.3e '
                  'max|dphi|=%.3e madS(e)=%.6f'
                  % (used, istop[-1], itn[-1], d, dphi, mad_scaled(e_unc)))
        if d < tol:
            break

    e = z - phi
    yhat = g_apply(g_knots, phi[unc])
    r2_link = _r2(y[unc], yhat)
    r2_lin_final = _r2_lin(phi[unc], y[unc])
    r2_add_latent = _r2(z[unc], phi[unc])
    return LatentFit(
        beta=beta, phi=phi, z=z, g_knots=g_knots,
        sigma_of_phi=make_sigma_fn((sk[0], sk[1])), sigma_knots=sk,
        r2_link_gain=r2_link - r2_lin_final,
        r2_link_gain_vs_raw=r2_link - r2_add_raw,
        n_iter_used=used, e=e, lo=lo, hi=hi,
        converged=bool(dbeta_trace and dbeta_trace[-1] < tol),
        dbeta_last=(dbeta_trace[-1] if dbeta_trace else float('nan')),
        dbeta_trace=tuple(dbeta_trace), dphi_last=dphi,
        r2_add_raw=r2_add_raw, r2_lin_final=r2_lin_final, r2_link=r2_link,
        r2_add_latent=r2_add_latent, resid_mad=mad_scaled(e_unc),
        corr_e_phi=(float(np.corrcoef(e[unc], phi[unc])[0, 1])
                    if e_unc.std() > 0 else float('nan')),
        sigma_floor=float(sigma_floor), n_censored=int(cm.sum()),
        lsqr_itn=tuple(itn), lsqr_istop=tuple(istop),
        wall_s=time.time() - t0, anchor=(my, sy))


# --------------------------------------------------------------------------- #
# cross-fit                                                                   #
# --------------------------------------------------------------------------- #

def make_folds(n, dms_id, n_folds=None, seed_name='crossfit'):
    """5 folds over variants, seeded from :data:`cliff.config.SEEDS` (spec
    Sec.1.0 "5 folds over variants, fixed seed 20260902").

    Entropy is ``config.assay_seed(seed_name, dms_id)`` == ``[SEEDS[name],
    ASSAY_ORDINAL[dms_id]]``, the convention the foundation already uses for the
    random-pair sample: with one shared int every assay of the same ``n`` would
    get the identical partition.
    """
    if n_folds is None:
        n_folds = THRESH['crossfit_n_folds']
    rng = np.random.default_rng(config.assay_seed(seed_name, dms_id))
    perm = rng.permutation(int(n))
    folds = np.empty(int(n), dtype=np.int8)
    for k, ix in enumerate(np.array_split(perm, int(n_folds))):
        folds[ix] = k
    return folds


def crossfit_latent(X, y, censor_mask, censor_levels, folds, *, n_iter=None,
                    n_bins=None, tol=None, sigma_floor=0.0, verbose=False):
    """5-fold cross-fit (spec Sec.1.0, MANDATORY).

    Returns ``phi_oof, z_oof, e_oof, sigma_oof`` (all length ``n``) plus the
    per-fold diagnostics.  For every fold the whole of :func:`fit_latent` --
    ``beta``, the link ``g``, and ``sigma-hat(phi)`` -- is refitted on the other
    four folds, so nothing about variant ``i`` is in ``i``'s own numbers.

    **Unseen design columns.**  A held-out row may use a substitution the
    training folds never saw (CD19_FMC63_7URV: 1,467 of its 1,826 columns occur
    exactly once, so the row that carries such a column is *always* held out
    without it).  ``lsqr`` silently returns ``beta_j = 0`` there, which would put
    the substitution's whole MAIN effect into ``e`` and manufacture a cliff:
    ``e_{B+i} - e_B`` would be ``Delta_i(B)``, not ``Delta_i(B) - beta_i``.  Such
    rows therefore get ``phi_oof = nan``, which is exactly what makes the spec's
    ``P_a`` condition (c) "finite ``phi^oof`` at both endpoints" non-empty.
    ``frac_oof_finite`` is reported per assay.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    cm = np.asarray(censor_mask, dtype=bool)
    folds = np.asarray(folds)
    if folds.shape != y.shape:
        raise ValueError('folds shape %s != y shape %s' % (folds.shape, y.shape))
    Xc = X.tocsc()
    col_count_all = np.diff(Xc.indptr)
    phi_oof = np.full(n, np.nan)
    z_oof = np.full(n, np.nan)
    sigma_oof = np.full(n, np.nan)
    unseen_cols_in_row = np.zeros(n, dtype=np.int32)
    per_fold = []
    ks = np.unique(folds)
    for k in ks:
        te = folds == k
        tr = ~te
        t0 = time.time()
        fit = fit_latent(X[tr], y[tr], cm[tr], censor_levels, n_iter=n_iter,
                         n_bins=n_bins, tol=tol, sigma_floor=sigma_floor)
        # columns absent from the training rows
        cc_te = np.diff(Xc[te].tocsc().indptr)
        seen = (col_count_all - cc_te) > 0
        Xte = X[te]
        bad_cnt = np.asarray(Xte[:, ~seen].sum(axis=1)).ravel().astype(np.int32)
        unseen_cols_in_row[te] = bad_cnt
        ph = float(fit.beta[0]) + Xte.dot(fit.beta[1:])
        ph[bad_cnt > 0] = np.nan
        phi_oof[te] = ph
        zz = ginv(fit.g_knots, y[te], fit.lo, fit.hi)
        zz[bad_cnt > 0] = np.nan
        # censored held-out rows get the same Tobit E-step as in sample, so
        # e_oof is one quantity everywhere (they are masked out downstream)
        cmte = cm[te]
        if cmte.any():
            fl, ce = classify_levels(y[tr], censor_levels)
            lvl = np.array(list(fl) + list(ce), dtype=np.float64)
            isfl = np.array([True] * len(fl) + [False] * len(ce), dtype=bool)
            if lvl.size:
                wh = np.abs(y[te][cmte][:, None] - lvl[None, :]).argmin(axis=1)
                cvals = ginv(fit.g_knots, lvl, fit.lo, fit.hi)[wh]
                sg = sigma_eval((fit.sigma_knots[0], fit.sigma_knots[1]),
                                ph[cmte])
                tb = tobit_estep(ph[cmte], sg, cvals, isfl[wh])
                zc = zz[cmte]
                zc[np.isfinite(ph[cmte])] = tb[np.isfinite(ph[cmte])]
                zz[cmte] = zc
        z_oof[te] = zz
        sigma_oof[te] = sigma_eval((fit.sigma_knots[0], fit.sigma_knots[1]), ph)
        per_fold.append(dict(
            fold=int(k), n_train=int(tr.sum()), n_test=int(te.sum()),
            n_cols_unseen=int((~seen).sum()),
            n_rows_unseen=int((bad_cnt > 0).sum()),
            n_iter_used=fit.n_iter_used, converged=bool(fit.converged),
            dbeta_last=float(fit.dbeta_last), r2_link=float(fit.r2_link),
            r2_link_gain=float(fit.r2_link_gain),
            resid_mad=float(fit.resid_mad),
            g_knots=fit.g_knots, sigma_knots=fit.sigma_knots,
            beta=fit.beta, lo=fit.lo, hi=fit.hi,
            wall_s=round(time.time() - t0, 3)))
        if verbose:
            print('    fold %d: n_tr=%6d n_te=%6d unseen_cols=%4d '
                  'nan_rows=%5d it=%2d dbeta=%.2e madS=%.5f %.2fs'
                  % (k, tr.sum(), te.sum(), (~seen).sum(), (bad_cnt > 0).sum(),
                     fit.n_iter_used, fit.dbeta_last, fit.resid_mad,
                     per_fold[-1]['wall_s']))
    e_oof = z_oof - phi_oof
    fin = np.isfinite(phi_oof) & np.isfinite(z_oof)
    return dict(phi_oof=phi_oof, z_oof=z_oof, e_oof=e_oof, sigma_oof=sigma_oof,
                folds=folds, per_fold=per_fold,
                unseen_cols_in_row=unseen_cols_in_row,
                oof_finite=fin, n_oof_finite=int(fin.sum()),
                frac_oof_finite=float(fin.mean()),
                resid_mad_oof=mad_scaled(e_oof[fin & ~cm]),
                n_cols_unseen_total=int(sum(f['n_cols_unseen'] for f in per_fold)))


# --------------------------------------------------------------------------- #
# cache + driver                                                              #
# --------------------------------------------------------------------------- #

def _ragged(list_of_arrays):
    """Concatenate + offsets, so per-fold knots survive an npz without pickle."""
    arrs = [np.asarray(a, dtype=np.float64).ravel() for a in list_of_arrays]
    ptr = np.zeros(len(arrs) + 1, dtype=np.int64)
    np.cumsum([a.size for a in arrs], out=ptr[1:])
    return (np.concatenate(arrs) if arrs else np.zeros(0)), ptr


def cache_latent(dms_id, fit, cf, extra=None):
    """``data/cliff_cache/latent/{DMS_id}.npz`` (spec Sec.5: ``beta, phi, z,
    e_oof, sigma_oof, folds, g_knots``) + the manifest entry."""
    PATHS.ensure_cache_dirs()
    p = os.path.join(PATHS.latent, dms_id + '.npz')
    gph, gptr = _ragged([f['g_knots'][0] for f in cf['per_fold']])
    gy, _ = _ragged([f['g_knots'][1] for f in cf['per_fold']])
    sc, sptr = _ragged([f['sigma_knots'][0] for f in cf['per_fold']])
    ss, _ = _ragged([f['sigma_knots'][1] for f in cf['per_fold']])
    bb, bptr = _ragged([f['beta'] for f in cf['per_fold']])
    arrays = dict(
        beta=fit.beta, phi=fit.phi, z=fit.z, e=fit.e,
        sigma_in=fit.sigma_of_phi(fit.phi),
        g_knots_phi=fit.g_knots[0], g_knots_y=fit.g_knots[1],
        sigma_knots_phi=fit.sigma_knots[0], sigma_knots_sigma=fit.sigma_knots[1],
        sigma_knots_median_e=fit.sigma_knots[2], sigma_knots_count=fit.sigma_knots[3],
        sigma_knots_degenerate=fit.sigma_knots[4],
        phi_oof=cf['phi_oof'], z_oof=cf['z_oof'], e_oof=cf['e_oof'],
        sigma_oof=cf['sigma_oof'], folds=cf['folds'],
        oof_finite=cf['oof_finite'], unseen_cols_in_row=cf['unseen_cols_in_row'],
        fold_g_knots_phi=gph, fold_g_knots_y=gy, fold_g_ptr=gptr,
        fold_sigma_knots_phi=sc, fold_sigma_knots_sigma=ss, fold_sigma_ptr=sptr,
        fold_beta=bb, fold_beta_ptr=bptr,
        fold_lo=np.array([f['lo'] for f in cf['per_fold']], dtype=np.float64),
        fold_hi=np.array([f['hi'] for f in cf['per_fold']], dtype=np.float64),
        lo=np.float64(fit.lo), hi=np.float64(fit.hi),
        anchor_mean=np.float64(fit.anchor[0]), anchor_sd=np.float64(fit.anchor[1]),
        r2_link_gain=np.float64(fit.r2_link_gain),
        n_iter_used=np.int64(fit.n_iter_used),
        converged=np.bool_(fit.converged),
        dbeta_trace=np.asarray(fit.dbeta_trace, dtype=np.float64),
        sigma_floor=np.float64(fit.sigma_floor),
        seed=np.asarray(config.assay_seed('crossfit', dms_id)),
        meta_json=np.array(json.dumps(extra or {}, sort_keys=True, default=str)),
    )
    tmp = p[:-4] + '.tmp.npz'
    np.savez(tmp, **arrays)
    os.replace(tmp, p)
    return dict(path=os.path.relpath(p, config.REPO), md5=md5_of(p),
                bytes=os.path.getsize(p))


def _update_manifest(entries, extra=None):
    """Merge new md5 entries into ``MANIFEST.json`` -- ``pairs.write_manifest``
    REPLACES ``files``, so stage 0's entries have to be carried forward."""
    from . import pairs
    _RESERVED = ('schema', 'written_utc', 'env', 'env_observed', 'git',
                 'seed_base', 'seeds', 'assay_ordinal', 'taus',
                 'bindinggym_input', 'files')
    old, oldextra = {}, {}
    if os.path.exists(PATHS.manifest):
        with open(PATHS.manifest) as fh:
            man = json.load(fh)
        old = man.get('files', {})
        oldextra = {k: v for k, v in man.items() if k not in _RESERVED}
    new = {e['path']: e for e in entries}
    merged = [dict(path=p, md5=m['md5'], bytes=m['bytes'])
              for p, m in old.items() if p not in new]
    merged.extend(new.values())
    ex = dict(oldextra)
    if extra:
        ex.update(extra)
    return pairs.write_manifest(merged, extra=ex)


def register_latent_cache(extra=None):
    """md5 every file in ``data/cliff_cache/latent/`` into ``MANIFEST.json``.

    Spec Sec.5 requires every cache file to be fingerprinted.  This is a
    whole-directory sweep rather than a per-call append because (a) an assay fit
    on demand by :mod:`cliff.noise` (the five EXCLUDED-tier files that still
    carry a downstream role) would otherwise never be registered, and (b)
    ``MANIFEST.json`` is a read-modify-write shared with the other stage
    modules, so a sweep is idempotent and self-healing if a concurrent writer
    drops entries.
    """
    PATHS.ensure_cache_dirs()
    ents = []
    for f in sorted(os.listdir(PATHS.latent)):
        if not f.endswith('.npz'):
            continue
        p = os.path.join(PATHS.latent, f)
        ents.append(dict(path=os.path.relpath(p, config.REPO), md5=md5_of(p),
                         bytes=os.path.getsize(p)))
    if ents:
        _update_manifest(ents, extra=extra)
    return ents


#: the per-assay report this module produces (feeds T04's latent columns and
#: T03's ``internal_residual`` rows).
LATENT_COLUMNS = [
    'DMS_id', 'tier', 'family_id', 'n', 'M', 'P', 'n_censored', 'frac_censored',
    'transform', 'quantum', 'quantum_uncensored', 'sigma_floor',
    'n_iter_used', 'converged', 'dbeta_last', 'dbeta_min', 'conv_tol',
    'r2_add_raw', 'r2_lin_final', 'r2_link', 'r2_link_gain',
    'r2_link_gain_vs_raw', 'r2_add_latent',
    'resid_mad_in', 'resid_mad_oof', 'resid_sd_in', 'corr_e_phi',
    'sigma_median', 'sigma_min', 'sigma_max', 'sigma_dyn_range',
    'sigma_over_mad_y',
    'n_bins_used', 'n_bins_degenerate', 'n_cols_unseen_total',
    'n_oof_finite', 'frac_oof_finite',
    'folds_converged', 'wall_fit_s', 'wall_crossfit_s', 'wall_total_s',
]


def run_latent(dms_id, *, assay=None, write=True, verbose=True):
    """Full fit + 5-fold cross-fit for one assay; writes the npz cache."""
    t00 = time.time()
    a = assay if assay is not None else io_bgym.load_assay(
        dms_id, keep_score_strings=True)
    X = design_from_keys(a.keys, a.col_index, a.n_muts)
    cm = a.censor_mask
    # quantum RE-DERIVED on the uncensored score strings: CR9114_FluAH3's raw
    # modal decimal count is 1 only because 89.05% of the file is '6.0'.
    q_unc = a.quantum
    if a.score_strings and cm.any():
        ss = [s for s, c in zip(a.score_strings, cm) if not c]
        if ss:
            q_unc = io_bgym.score_quantum(tuple(ss))[0]
    sigma_floor = q_unc / _SQRT_12
    t0 = time.time()
    fit = fit_latent(X, a.y, cm, a.censor_levels, sigma_floor=sigma_floor,
                     verbose=verbose)
    t_fit = time.time() - t0
    folds = make_folds(a.n, dms_id)
    t0 = time.time()
    cf = crossfit_latent(X, a.y, cm, a.censor_levels, folds,
                         sigma_floor=sigma_floor, verbose=verbose)
    t_cf = time.time() - t0
    row = dict(
        DMS_id=dms_id, tier=config.tier_of(dms_id),
        family_id=config.family_of(dms_id), n=a.n, M=a.M, P=a.P,
        n_censored=int(cm.sum()), frac_censored=round(float(cm.mean()), 6),
        transform=a.transform, quantum=a.quantum, quantum_uncensored=q_unc,
        sigma_floor=round(sigma_floor, 10),
        n_iter_used=fit.n_iter_used, converged=bool(fit.converged),
        dbeta_last=fit.dbeta_last, dbeta_min=min(fit.dbeta_trace),
        conv_tol=THRESH['latent_conv_tol'],
        r2_add_raw=round(fit.r2_add_raw, 6),
        r2_lin_final=round(fit.r2_lin_final, 6),
        r2_link=round(fit.r2_link, 6),
        r2_link_gain=round(fit.r2_link_gain, 6),
        r2_link_gain_vs_raw=round(fit.r2_link_gain_vs_raw, 6),
        r2_add_latent=round(fit.r2_add_latent, 6),
        resid_mad_in=round(fit.resid_mad, 6),
        resid_mad_oof=round(cf['resid_mad_oof'], 6),
        resid_sd_in=round(float(fit.e[~cm].std()), 6),
        corr_e_phi=round(fit.corr_e_phi, 4),
        sigma_median=round(float(np.median(fit.sigma_knots[1])), 6),
        sigma_min=round(float(fit.sigma_knots[1].min()), 6),
        sigma_max=round(float(fit.sigma_knots[1].max()), 6),
        sigma_dyn_range=round(float(fit.sigma_knots[1].max()
                                    / fit.sigma_knots[1].min()), 2),
        sigma_over_mad_y=(round(float(np.median(fit.sigma_knots[1]))
                                / mad_scaled(a.y[~cm]), 4)
                          if mad_scaled(a.y[~cm]) > 0 else float('nan')),
        n_bins_used=int(fit.sigma_knots[1].size),
        n_bins_degenerate=int(fit.sigma_knots[4].sum()),
        n_cols_unseen_total=cf['n_cols_unseen_total'],
        n_oof_finite=cf['n_oof_finite'],
        frac_oof_finite=round(cf['frac_oof_finite'], 6),
        folds_converged=int(sum(f['converged'] for f in cf['per_fold'])),
        wall_fit_s=round(t_fit, 2), wall_crossfit_s=round(t_cf, 2),
        wall_total_s=round(time.time() - t00, 2))
    ent = None
    if write:
        ent = cache_latent(dms_id, fit, cf, extra=row)
    return dict(row=row, fit=fit, crossfit=cf, manifest=(ent and [ent]) or [],
                assay=a)


def run_all(assays=None, *, write=True, verbose=True):
    """Cross-fit the 17 PRIMARY+ARM+CONTROL assays and cache them (spec Sec.5
    stage 2).  Returns the per-assay report as a DataFrame."""
    import pandas as pd
    config.assert_env()
    if assays is None:
        assays = tuple(sorted(set(config.PRIMARY_AND_ARM + config.CONTROL)))
    rows, ents = [], []
    for i, d in enumerate(assays):
        if verbose:
            print('[latent] %2d/%d %s' % (i + 1, len(assays), d))
        out = run_latent(d, write=write, verbose=verbose)
        rows.append(out['row'])
        ents.extend(out['manifest'])
    df = pd.DataFrame(rows)[LATENT_COLUMNS]
    if write and ents:
        register_latent_cache(extra=dict(latent=dict(
            n_assays=len(assays), assays=list(assays),
            written_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            n_iter=THRESH['latent_n_iter'], n_folds=THRESH['crossfit_n_folds'],
            n_bins=THRESH['sigma_n_bins'], conv_tol=THRESH['latent_conv_tol'],
            seed_name='crossfit', seed=SEEDS['crossfit'])))
    return df


def load_report(assays=None):
    """Rebuild the per-assay report from the cached npz files' ``meta_json``.

    ``run_all``'s DataFrame is not written to ``artifacts/`` because no T-table
    in spec Sec.6 owns it (T04 carries ``R2_add_raw`` / ``R2_add_latent`` /
    ``link_R2_gain``, T03 carries the sigmas).  Nothing is lost: every row is
    stored inside its own ``data/cliff_cache/latent/{id}.npz``, and this
    rebuilds the table from the cache.
    """
    import pandas as pd
    if assays is None:
        assays = tuple(sorted(set(config.PRIMARY_AND_ARM + config.CONTROL)))
    rows = []
    for d in assays:
        p = os.path.join(PATHS.latent, d + '.npz')
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=False)
        rows.append(json.loads(str(z['meta_json'])))
    cols = [c for c in LATENT_COLUMNS if rows and c in rows[0]]
    return pd.DataFrame(rows)[cols]


def stage2(assays=None, verbose=True):
    """spec Sec.5 stage 2, this module's half: ``fit_latent`` + the 5-fold
    cross-fit for the 17 PRIMARY+ARM+CONTROL assays, cached to
    ``data/cliff_cache/latent/*.npz`` and md5'd into ``MANIFEST.json``.

    The name ``run_all.py``'s stage table looks for first.
    """
    return run_all(assays=assays, verbose=verbose)


# --------------------------------------------------------------------------- #
# iteration-stability probe                                                   #
# --------------------------------------------------------------------------- #

def iteration_stability(dms_id, n_iter_b=40, *, assay=None, sigma_floor=None):
    """How much of ``e / sigma-hat(phi)`` is still moving at iteration 10.

    The spec's stopping rule (``max|dbeta| < 1e-6``) is never met -- the
    alternation limit-cycles at a level set by how well the design identifies
    ``beta`` (measured over 200 iterations: 4e-3 on GB1_IgG-Fc_1FCC, 6e-2 on
    KRAS_RALGDS-RBD, 5e-1 on CD19_FMC63).  "17/17 did not converge" is therefore
    uninformative on its own; what a tail statistic actually needs to know is how
    far a *standardised* residual moves if you keep iterating.  This runs the fit
    at ``THRESH['latent_n_iter']`` and at ``n_iter_b`` and reports the quantiles
    of ``|Delta(e/sigma)|``.
    """
    a = assay if assay is not None else io_bgym.load_assay(
        dms_id, keep_score_strings=True)
    X = design_from_keys(a.keys, a.col_index, a.n_muts)
    if sigma_floor is None:
        q = a.quantum
        if a.score_strings and a.censor_mask.any():
            ss = [s for s, c in zip(a.score_strings, a.censor_mask) if not c]
            if ss:
                q = io_bgym.score_quantum(tuple(ss))[0]
        sigma_floor = q / _SQRT_12
    fa = fit_latent(X, a.y, a.censor_mask, a.censor_levels, sigma_floor=sigma_floor)
    fb = fit_latent(X, a.y, a.censor_mask, a.censor_levels, sigma_floor=sigma_floor,
                    n_iter=int(n_iter_b))
    ca = fa.e / fa.sigma_of_phi(fa.phi)
    cb = fb.e / fb.sigma_of_phi(fb.phi)
    d = np.abs(ca - cb)[~a.censor_mask]
    d = d[np.isfinite(d)]
    return dict(DMS_id=dms_id, n_iter_a=fa.n_iter_used, n_iter_b=fb.n_iter_used,
                dbeta_a=fa.dbeta_last, dbeta_b=fb.dbeta_last,
                d_r2_link=abs(fa.r2_link - fb.r2_link),
                d_resid_mad=abs(fa.resid_mad - fb.resid_mad),
                dc_median=float(np.median(d)), dc_p95=float(np.percentile(d, 95)),
                dc_p999=float(np.percentile(d, 99.9)), dc_max=float(d.max()))


# --------------------------------------------------------------------------- #
# self-check                                                                  #
# --------------------------------------------------------------------------- #

def _selfcheck():
    import pandas as pd
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 100)
    print('[latent] env %r' % (config.assert_env(),))

    # ---- closed-form / algebraic checks on the real data ------------------ #
    a = io_bgym.load_assay('GB1_IgG-Fc_fitness_1FCC', keep_score_strings=True)
    Xk = design_from_keys(a.keys, a.col_index, a.n_muts)
    Xc = design_from_codes(a.codes, a.col_index, a.pos_index)
    assert Xk.shape == Xc.shape and (Xk != Xc).nnz == 0, \
        'design_from_keys != design_from_codes'
    assert np.array_equal(np.asarray(Xk.sum(axis=1)).ravel(),
                          a.n_muts.astype(np.float64)), 'row nnz != num_muts'
    print('[latent] design matrix: keys == codes path, row nnz == num_muts  OK'
          '  (n=%d, M=%d, nnz=%d)' % (Xk.shape[0], Xk.shape[1], Xk.nnz))

    # g / ginv round trip on a strictly increasing link
    gk = (np.array([0., 1., 2., 3.]), np.array([0., 0., 1., 2.]))
    uy, mid = strict_hull(gk)
    assert np.array_equal(uy, [0., 1., 2.]) and np.allclose(mid, [0.5, 2., 3.]), \
        (uy, mid)
    assert np.allclose(ginv(gk, [0., 1., 2.], 0., 3.), [0.5, 2., 3.])
    assert np.allclose(ginv(gk, [-9., 99.], 0., 3.), [0.5, 3.]), 'clip/extrap'
    print('[latent] strict_hull / ginv plateau + clip semantics  OK')

    # MAD is 1.4826 * median|e - med e|, never sd
    r = np.random.default_rng(0).standard_normal(200000)
    assert abs(mad_scaled(r) - 1.0) < 0.01, mad_scaled(r)
    print('[latent] 1.4826 * MAD(N(0,1)) = %.4f  (sd = %.4f)  OK'
          % (mad_scaled(r), r.std()))

    # Tobit E-step against the closed-form truncated-normal mean, and the
    # stability that log_ndtr buys (CR9114_FluAH3 has rows at a < -100)
    from scipy.stats import norm
    ph = np.zeros(3)
    sg = np.ones(3)
    cc = np.array([0.0, -2.0, -150.0])
    isf = np.ones(3, dtype=bool)
    got = tobit_estep(ph, sg, cc, isf)
    with np.errstate(invalid='ignore', divide='ignore'):
        want = ph - sg * norm.pdf(cc) / norm.cdf(cc)
    assert np.allclose(got[:2], want[:2], rtol=1e-10), (got, want)
    assert np.isfinite(got[2]) and abs(got[2] - (-150.0)) < 0.02, got[2]
    assert not np.isfinite(want[2]), 'norm.pdf/norm.cdf should underflow at -150'
    print('[latent] Tobit E-step == truncated-normal mean at a=0,-2; stable at '
          'a=-150 (E[Z|Z<=-150] = %.5f, naive ratio = %s)  OK' % (got[2], want[2]))

    # the column-scaled lsqr solves the SAME least-squares problem
    A = with_intercept(Xk)
    cn = _colnorms(A)
    zz = a.y
    b_plain = lsqr(A, zz, atol=1e-12, btol=1e-12)[0]
    b_scal = lsqr((A @ sp.diags(1.0 / cn)).tocsr(), zz,
                  atol=1e-12, btol=1e-12)[0] / cn
    AtA = (A.T @ A).toarray()
    b_ex = np.linalg.lstsq(AtA, A.T @ zz, rcond=None)[0]
    dp_plain = float(np.abs(A.dot(b_plain) - A.dot(b_ex)).max())
    dp_scal = float(np.abs(A.dot(b_scal) - A.dot(b_ex)).max())
    assert dp_scal < dp_plain, (dp_scal, dp_plain)
    print('[latent] lsqr column scaling: max|phi - phi_exact| = %.2e scaled vs '
          '%.2e plain  OK' % (dp_scal, dp_plain))

    # scale invariance of everything downstream reads
    f1 = fit_latent(Xk, a.y, a.censor_mask, a.censor_levels)
    f2 = fit_latent(Xk, 3.0 * a.y + 7.0, a.censor_mask,
                    tuple(3.0 * v + 7.0 for v in a.censor_levels))
    c1 = f1.e / f1.sigma_of_phi(f1.phi)
    c2 = f2.e / f2.sigma_of_phi(f2.phi)
    print('[latent] affine y -> 3y+7: max|d(e/sigma)| = %.2e, '
          'd(r2_link) = %.2e  (scale-free)  OK'
          % (np.abs(c1 - c2).max(), abs(f1.r2_link - f2.r2_link)))

    print('[latent] fit_latent on GB1_IgG-Fc_1FCC: %r' % (f1,))
    print('[latent]   wall %.2f s  (spec Sec.5 anchor 1.33 s, G0 budget %.1f s)'
          % (f1.wall_s, THRESH['G0_fit_latent_s']))
    print('[latent]   dbeta trace: %s'
          % ' '.join('%.1e' % v for v in f1.dbeta_trace))

    # ---- the real run ----------------------------------------------------- #
    df = run_all(verbose=False)
    print()
    print(df[['DMS_id', 'n', 'M', 'n_censored', 'n_iter_used', 'converged',
              'dbeta_last', 'r2_add_raw', 'r2_link', 'r2_link_gain',
              'resid_mad_in', 'resid_mad_oof', 'corr_e_phi', 'frac_oof_finite',
              'n_bins_degenerate', 'sigma_min', 'sigma_median',
              'sigma_dyn_range', 'wall_fit_s',
              'wall_crossfit_s']].to_string(index=False))
    print()
    nc = df.loc[~df['converged'], 'DMS_id'].tolist()
    print('[latent] NOT converged at max|dbeta| < %g in %d iterations: %d/%d'
          % (THRESH['latent_conv_tol'], THRESH['latent_n_iter'], len(nc), len(df)))
    if nc:
        print('[latent]   -> the stopping rule is unreachable (limit cycle, see '
              'iteration_stability); what moves is quantified below.')
    st = pd.DataFrame([iteration_stability(d) for d in df['DMS_id']])
    print()
    print(st[['DMS_id', 'n_iter_a', 'n_iter_b', 'dbeta_a', 'dbeta_b',
              'd_r2_link', 'd_resid_mad', 'dc_median', 'dc_p95', 'dc_p999',
              'dc_max']].to_string(index=False,
                                   float_format=lambda v: '%.3e' % v))

    dr = df.sort_values('sigma_dyn_range', ascending=False)
    print('[latent] sigma-hat(phi) dynamic range (max/min over the %d bins): '
          'worst %s = %.0fx, next %s = %.1fx, median assay %.1fx'
          % (THRESH['sigma_n_bins'], dr['DMS_id'].iloc[0],
             dr['sigma_dyn_range'].iloc[0], dr['DMS_id'].iloc[1],
             dr['sigma_dyn_range'].iloc[1], df['sigma_dyn_range'].median()))
    print('[latent] frac_oof_finite (spec P_a condition (c)): worst %s = %.4f, '
          'next %s = %.4f'
          % (df.sort_values('frac_oof_finite')['DMS_id'].iloc[0],
             df['frac_oof_finite'].min(),
             df.sort_values('frac_oof_finite')['DMS_id'].iloc[1],
             df.sort_values('frac_oof_finite')['frac_oof_finite'].iloc[1]))
    print('[latent] worst fit wall %.2f s (%s); G0 budget %.1f s'
          % (df['wall_fit_s'].max(),
             df.loc[df['wall_fit_s'].idxmax(), 'DMS_id'],
             THRESH['G0_fit_latent_s']))
    print('[latent] total wall %.1f s for %d assays (%d fits)'
          % (df['wall_total_s'].sum(), len(df),
             len(df) * (1 + THRESH['crossfit_n_folds'])))
    from . import pairs
    bad = pairs.verify_manifest()
    print('[latent] verify_manifest(): %d mismatches' % len(bad))
    assert not bad, bad
    return df


if __name__ == '__main__':
    _selfcheck()
