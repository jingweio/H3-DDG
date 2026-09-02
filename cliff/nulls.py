# -*- coding: utf-8 -*-
"""``cliff.nulls`` -- N1/N2/N2b/N2c/N3 + NS1/NS2/NS3 and the parallel driver.

Every claim in BGYM-CLIFF is null-referenced (spec Sec.1.1 G4, Sec.1.3, Sec.1.4),
so this module is load-bearing: if the surrogate machinery is biased, *no*
observed number in the study is readable.  Spec Sec.3's null definitions,
verbatim:

* **N1 smooth parametric surrogate** -- ``z* = X beta + eps*``,
  ``eps* ~ N(0, sigma^2(phi))``; ``y* = g(z*)``; re-apply the assay's clamp and
  its decimal grid; **refit the latent model AND the 5-fold cross-fit from
  scratch on ``y*``** and recompute every statistic.  *Preserves* the exact
  variant set (hence the exact pair graph, every node degree including the WT
  hub), the mutation-order composition, the per-(pos,aa) effects, the monotone
  link, level-dependent noise, censored mass, tie mass and the additive-fit
  estimation error.  *Destroys* all background dependence.
* **N2 residual exchange** -- keep ``beta, phi, g, sigma`` fixed; permute ``e``
  within ``(mutation order x phi-decile)`` strata.  *Preserves* the residual
  marginal EXACTLY, heavy tail included.  *Destroys* locality on the graph.
  **Its zero-power case is declared and MEASURED** (:func:`n2_power`), not
  assumed: in a singles+doubles library with a saturated single scan a double's
  marginal residual *is* ``eps_ij`` and the nested difference is also
  ``eps_ij``, so N2 coincides with the data.
* **N2b additive + ridge pairwise + link** -- ``Z`` columns only for
  substitution pairs co-observed ``>= THRESH['N2b_min_cooccur']`` times, ``lam``
  by 5-fold CV on held-out variants.  Decomposition only, never a gate.
* **N2c heteroscedastic scale mixture** -- ``eps* ~ N(0, sigma^2(phi) V)``, ``V``
  a two-point discrete mixture calibrated so the marginal residual kurtosis
  matches the observed marginal kurtosis exactly.  Exists to prove that
  tail-shape statistics have no power against heteroscedasticity (G7).
* **N3 free permutation of ``y``** -- House-of-Cards, upper calibration only.
  The identity ``E[SI_N3] = 1`` is a self-test, not a result.
* **NS1 / NS2 / NS3** -- the spec Sec.1.5 stratified label permutations.

ORCHESTRATOR DECISIONS honoured here (they override the spec where they
conflict):

* **D1** -- ``SI`` runs on the SEQUENCE Hamming axis (nested union same-site);
  ``c_hat`` runs on NESTED pairs only.  Both axes are in the statistic vector
  and are never merged.
* **D2** -- ``c_hat`` is **phi-centred**::

      c_hat = ((e_v - mu(phi_v)) - (e_u - mu(phi_u)))
              / sqrt(sigma^2(phi_u) + sigma^2(phi_v))

  with ``mu`` the per-bin median of ``e`` (``sigma_knots_median_e``) interpolated
  exactly as ``sigma`` is.  ``corr(e_oof, phi_oof)`` is +0.24..+0.36, so the
  spec's uncentred form has a non-zero mean on any nested pair whose endpoints
  sit in different ``phi`` regions and would MANUFACTURE cliffs.  The uncentred
  form is computed too, as the ``T13`` ``centring`` sensitivity row.  **Every
  null goes through the identical centring** -- with its OWN refitted ``mu`` for
  the refit nulls, with the observed fit's ``mu`` for the fixed-fit null (N2),
  which is what "identical" means when the fit itself is what the null holds
  fixed.
* **D3** -- CD19_FMC63_7URV is computed and reported, flagged
  ``STRUCTURALLY_UNIDENTIFIED``; ``K = 6`` for the aggregate.  This module does
  not emit verdicts, so the flag travels as a column.
* **D4** -- ``converged=False`` from ``fit_latent`` is EXPECTED and is not read
  as a failure anywhere here.
* **D5** -- reported by ``stats_c3``; this module only supplies the nulls.
* **D8** -- the ``flock`` fix lives in ``pairs.write_manifest``; this module
  calls :func:`register_null_cache` ONCE at the end of its run and then
  ``pairs.verify_manifest()``.

DEVIATIONS from the spec's literal wording, all MEASURED, none silent (the full
numbers are in ``artifacts/T02a..T02f``):

1. **N1's forward link is the strictly-increasing HULL, not sklearn's step
   ``predict``** -- :func:`link_forward`.  Literal: 719 distinct ``y*`` on
   GB1_1FCC against 82,124 observed.  Hull: 91,919.  Knob
   ``link_mode='pav_step'``.
2. **The censoring threshold is calibrated to the observed censored fraction,
   not ``g^-1(L)``** -- :func:`observe`.  Literal: SARS2-RBD 3.86% clamped
   against 23.85% observed; CR9114_FluAH1 5.86% against 2.57%.  Knob
   ``clamp_mode='ginv'``.
3. **N2b's ``Z`` columns are SUBSTITUTION pairs, not "site pairs"** --
   :func:`pairwise_design`, where the spec's own L4 arithmetic settles it.
4. **N2b's noise scale is the pairwise fit's own residual scale**, not
   ``sigma(phi_add)`` -- :func:`surrogate_N2b`; the additive residual already
   contains the pairwise signal.
5. **N2c's two-point support is pinned to the min-ratio mixture** --
   :func:`two_point_scale_mixture`; two moments leave one free parameter and the
   spec pins neither.
6. **G4's uniformity gate reads the RANDOMISED empirical p-value** --
   :func:`_empirical_p`; the spec's conservative form is a point mass at 1
   wherever the rate is 0 in most replicates, and is reported beside it.
7. **``STAGE3_NULLS`` is the four nulls spec Sec.5 prices**; N3 is opt-in.
8. **BLAS threads are pinned to 1** -- ``THREAD_ENV``; not in the spec, but 64
   workers x the default pool measured a load average of 1,409 here.

Self-check::

    python -m cliff.nulls                    # invariants + a small live ensemble
    python -m cliff.nulls --stage3           # the full 4 x 200 x 17 run
    python -m cliff.nulls --g4               # G4 self-calibration + T02 rows
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings

# ---- BLAS threads: 1, and it has to be set BEFORE numpy is imported ------- #
# Every replicate is a sparse ``lsqr`` + a PAV fit -- single-threaded work by
# construction -- but a worker's OpenBLAS still builds its default thread pool
# (one per core) the first time any dense kernel runs, and 64 workers x 80
# threads is a load average of 1,409 on this box: measured, and it stalled the
# run rather than slowing it.  ``THREAD_ENV`` is applied only where the caller
# has not already chosen a value, so an explicit setting always wins.  If
# :mod:`numpy` is already imported by the time this module loads the assignment
# is too late to bind, and :func:`_warn_threads` says so at fan-out time rather
# than letting it pass silently.
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
_NUMPY_PREIMPORTED = 'numpy' in sys.modules
for _v in THREAD_ENV:
    os.environ.setdefault(_v, '1')

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from scipy.stats import kstest

from . import config
from . import io_bgym
from . import latent as _latent
from . import pairs as _pairs
from .config import PATHS, SEEDS, THRESH, TAUS
from .io_bgym import md5_of
from .latent import (crossfit_latent, fit_latent, g_apply, ginv, mad_scaled,
                     make_folds, sigma_eval, with_intercept)

__all__ = [
    'NULLS', 'NullContext', 'build_context', 'clear_context_cache',
    'surrogate_N1', 'surrogate_N2', 'surrogate_N2b', 'surrogate_N2c',
    'surrogate_N3', 'PairwiseFit', 'fit_pairwise_ridge', 'pairwise_design',
    'two_point_scale_mixture', 'kurtosis_targets',
    'permute_NS1', 'permute_NS2', 'permute_NS3', 'permute_within_strata',
    'link_forward', 'link_audit', 'observe', 'LINK_MODES', 'CLAMP_MODES',
    'make_strata', 'c_hat', 'default_stat_fn', 'STAT_NAMES',
    'observed_stats', 'replicate', 'run_ensemble', 'ensemble_path',
    'STAT_VERSION', 'THREAD_ENV', 'g4_selfcal', 'g4_all', '_empirical_p', 'n2_power',
    'n2_power_all', 'n2c_audit', 'n2c_audit_all',
    'stage3',
    'STAGE3_NULLS', 'register_null_cache',
    'write_T02_G4_rows', 'timing_table',
]

#: The five value-surrogate nulls.  NS1/NS2/NS3 are label permutations of
#: downstream tables and are not part of :func:`run_ensemble`.
NULLS = ('N1', 'N2', 'N2b', 'N2c', 'N3')

#: Nulls that refit the latent model + the 5-fold cross-fit from scratch.
_REFIT_NULLS = ('N1', 'N2b', 'N2c', 'N3')

#: The four nulls spec Sec.5 stage 3 prices ("4 nulls x 200 reps x 17 assays =
#: 13,600 replicate-jobs").  N3 is deliberately NOT one of them: it is upper
#: calibration only, ``variogram.si_null_N3`` / ``rs_null_N3`` already supply the
#: C1 half without a refit, and its ``c_hat`` half costs a full refit per
#: replicate.  Ask for it explicitly (``stage3(nulls=NULLS)``) and pay for it.
STAGE3_NULLS = ('N1', 'N2', 'N2b', 'N2c')

#: ``sqrt(12)``: the sd of a uniform rounding error is ``q/sqrt(12)``.  A
#: mathematical constant, imported from :mod:`cliff.latent`'s convention so the
#: surrogate's ``sigma_floor`` is byte-identical to the observed fit's.
_SQRT_12 = math.sqrt(12.0)

#: phi-deciles for the N2 exchange strata (spec Sec.3: "(mutation order x
#: phi-decile) strata").  Ten, because "decile" says ten.
_N2_PHI_BINS = 10

#: Kurtosis is a 4th moment: with ``n < _MIN_N_KURT`` finite residuals the
#: estimate is too noisy to calibrate N2c against, and the mixture degenerates
#: to ``V == 1`` (i.e. N2c == N1) rather than chase noise.
_MIN_N_KURT = 200


# =========================================================================== #
# per-assay context                                                           #
# =========================================================================== #

class NullContext(object):
    """Everything a replicate needs, read ONCE per assay per worker.

    Spec Sec.5: 13,600 replicate-jobs must never re-parse a csv nor
    re-enumerate a pair graph.  Built from ``data/cliff_cache/{keys,pairs,latent}``
    only, with the md5 verified (spec Sec.5: "downstream code verifies the md5
    before use and refuses to run on a mismatch").
    """

    __slots__ = (
        'dms_id', 'tier', 'family_id', 'n', 'M', 'P',
        'X', 'A', 'cn', 'As', 'y', 'y_raw', 'transform', 'modal_decimals',
        'quantum', 'quantum_unc', 'sigma_floor', 'n_muts', 'wt_row',
        'censor_levels', 'floor_levels', 'ceil_levels', 'floor_probs',
        'floor_frac', 'ceil_frac', 'censor_mask', 'folds',
        'beta', 'phi', 'z', 'e', 'g_knots', 'hull', 'lo', 'hi',
        'sigma_knots', 'mu_knots',
        'phi_oof', 'z_oof', 'e_oof', 'sigma_oof', 'mu_oof', 'oof_finite',
        'nested_idx', 'add_col', 'wt_anchored', 'pos_of_add',
        'lag1_idx', 'gmd', 'vinf', 'n_samesite',
        'pairwise', 'kurt', 'notes',
    )

    def __init__(self, **kw):
        for s in self.__slots__:
            setattr(self, s, kw.get(s))

    def __repr__(self):
        return ('NullContext(%s, n=%d, M=%d, nested=%d, lag1=%d, '
                'censored=%d)' % (self.dms_id, self.n, self.M,
                                  self.nested_idx.shape[0],
                                  self.lag1_idx.shape[0],
                                  int(self.censor_mask.sum())))


def _verify_cached(path):
    """md5 the cache file against ``MANIFEST.json`` and raise on a mismatch."""
    rel = os.path.relpath(path, config.REPO)
    with open(PATHS.manifest) as fh:
        man = json.load(fh)
    want = man.get('files', {}).get(rel)
    if want is None:
        raise RuntimeError('%s is not in MANIFEST.json -- run the stage that '
                           'writes it' % rel)
    got = md5_of(path)
    if got != want['md5']:
        raise RuntimeError('md5 mismatch for %s: %s != %s'
                           % (rel, got, want['md5']))


def _modal_decimals(dms_id, quantum):
    """The assay's decimal-grid exponent, read from the keys cache (stage 0 wrote
    it) and only derived from the quantum if the cache predates that field."""
    p = os.path.join(PATHS.keys, dms_id + '.npz')
    try:
        with np.load(p, allow_pickle=False) as z:
            if 'modal_decimals' in z.files:
                return int(z['modal_decimals'])
    except OSError:
        pass
    return int(round(-math.log10(float(quantum))))


def build_context(dms_id, *, verify=True):
    """Assemble the :class:`NullContext` for one assay from the caches."""
    des = _latent.load_cached_design(dms_id, verify=verify)
    lp = os.path.join(PATHS.latent, dms_id + '.npz')
    if verify:
        _verify_cached(lp)
    npz = np.load(lp, allow_pickle=False)
    L = {k: npz[k] for k in npz.files}
    npz.close()

    nz_p = os.path.join(PATHS.pairs, dms_id + '_nested.npz')
    ss_p = os.path.join(PATHS.pairs, dms_id + '_samesite.npz')
    if verify:
        _verify_cached(nz_p)
        _verify_cached(ss_p)
    with np.load(nz_p) as z:
        nested_idx = z['idx']
        add_col = z['add_col']
        wt_anchored = z['wt_anchored']
    with np.load(ss_p) as z:
        ss_idx = z['idx']

    y = des['y']
    n = y.size
    X = des['X']
    A = des['A']
    cn = _latent._colnorms(A)

    # ---- the assay's own decimal grid, on the UNCENSORED score strings ----- #
    # latent.run_latent re-derives the quantum this way (CR9114_FluAH3's raw
    # modal decimal count is 1 only because 89.05% of that file is '6.0'), and
    # the surrogate's sigma_floor must be byte-identical to the observed fit's.
    meta = json.loads(str(L['meta_json'])) if 'meta_json' in L else {}
    q_unc = float(meta.get('quantum_uncensored', des['quantum']))
    sigma_floor = float(L['sigma_floor']) if 'sigma_floor' in L \
        else q_unc / _SQRT_12

    censor_levels = tuple(des['censor_levels'])
    floors, ceils = _latent.classify_levels(y, censor_levels)
    # tie-mass composition of the floor: SARS2-RBD reports TWO floor strings
    # (-4.84 at 14.53% and -4.76 at 9.31%), so a clamped surrogate row must be
    # assigned a level from the observed proportions or the tie mass is wrong.
    fp = np.array([float((y == L_).mean()) for L_ in floors], dtype=np.float64)
    floor_frac = float(fp.sum())
    floor_probs = fp / fp.sum() if fp.size and fp.sum() > 0 else fp
    ceil_frac = float(sum((y == L_).mean() for L_ in ceils))

    sigma_knots = (L['sigma_knots_phi'], L['sigma_knots_sigma'])
    mu_knots = (L['sigma_knots_phi'], L['sigma_knots_median_e'])
    phi_oof, z_oof = L['phi_oof'], L['z_oof']
    oof_finite = np.isfinite(phi_oof) & np.isfinite(z_oof)

    # position of the added substitution, for the block bootstrap / ICC-by-site
    inv = {v: k for k, v in des['col_index'].items()}
    pos_lut = {}
    pos_of_col = np.empty(len(inv), dtype=np.int32)
    for c, (ch, ps, _aa) in inv.items():
        key = (ch, ps)
        if key not in pos_lut:
            pos_lut[key] = len(pos_lut)
        pos_of_col[c] = pos_lut[key]

    lag1 = (np.vstack([nested_idx, ss_idx]) if ss_idx.shape[0]
            else nested_idx).astype(np.int32, copy=False)

    ctx = NullContext(
        dms_id=dms_id, tier=config.tier_of(dms_id),
        family_id=config.family_of(dms_id),
        n=n, M=X.shape[1], P=des['codes'].shape[1],
        X=X, A=A, cn=cn, As=(A @ sp.diags(1.0 / cn)).tocsr(),
        y=y, y_raw=des['y_raw'], transform=des['transform'],
        modal_decimals=_modal_decimals(dms_id, des['quantum']),
        quantum=float(des['quantum']), quantum_unc=q_unc,
        sigma_floor=sigma_floor,
        n_muts=des['n_muts'].astype(np.int32), wt_row=des['wt_row'],
        censor_levels=censor_levels, floor_levels=floors, ceil_levels=ceils,
        floor_probs=floor_probs, floor_frac=floor_frac, ceil_frac=ceil_frac,
        censor_mask=des['censor_mask'], folds=L['folds'],
        beta=L['beta'], phi=L['phi'], z=L['z'], e=L['e'],
        g_knots=(L['g_knots_phi'], L['g_knots_y']),
        hull=_latent.strict_hull((L['g_knots_phi'], L['g_knots_y'])),
        lo=float(L['lo']), hi=float(L['hi']),
        sigma_knots=sigma_knots, mu_knots=mu_knots,
        phi_oof=phi_oof, z_oof=z_oof, e_oof=L['e_oof'],
        sigma_oof=L['sigma_oof'],
        mu_oof=sigma_eval(mu_knots, phi_oof), oof_finite=oof_finite,
        nested_idx=nested_idx, add_col=add_col, wt_anchored=wt_anchored,
        pos_of_add=pos_of_col[add_col],
        lag1_idx=lag1, n_samesite=int(ss_idx.shape[0]),
        gmd=_gini_mean_difference(y), vinf=_v_infinity(y),
        pairwise=None, kurt=None, notes={})
    ctx.kurt = kurtosis_targets(ctx)
    return ctx


#: Per-process context cache.  One assay at a time (``run_ensemble`` chunks by
#: assay), so this holds exactly one context and its RSS is bounded.
_CTX = {}


def clear_context_cache():
    _CTX.clear()


def get_context(dms_id, *, verify=False):
    if dms_id not in _CTX:
        _CTX.clear()
        _CTX[dms_id] = build_context(dms_id, verify=verify)
    return _CTX[dms_id]


# =========================================================================== #
# closed forms reused from variogram (imported lazily to avoid a cycle)       #
# =========================================================================== #

def _gini_mean_difference(y):
    y = np.sort(np.asarray(y, dtype=np.float64))
    n = y.size
    if n < 2:
        return float('nan')
    i = np.arange(1, n + 1, dtype=np.float64)
    return float(2.0 * ((2.0 * i - n - 1.0) * y).sum() / (n * (n - 1.0)))


def _v_infinity(y):
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n < 2:
        return float('nan')
    return float(y.var() * n / (n - 1.0))


# =========================================================================== #
# the observation model:  clamp + decimal grid                                #
# =========================================================================== #

def _round_grid(ctx, y):
    """Re-apply the assay's decimal grid.

    The grid is a property of the RAW ``DMS_score`` strings, so for the one
    transformed assay (hYAP65, ``log10``) the rounding is done on the raw scale
    and the transform re-applied -- rounding in log space would put the
    surrogate on a grid the instrument does not have.  A rounded raw value that
    lands at or below 0 (impossible on hYAP65's range but guarded) keeps its
    unrounded value; the count is returned.
    """
    md = ctx.modal_decimals
    if md < 0 or md > 12:
        return y, 0
    if ctx.transform == 'none':
        return np.round(y, md), 0
    if ctx.transform == 'log10':
        raw = np.power(10.0, y)
        rr = np.round(raw, md)
        bad = ~(rr > 0)
        n_bad = int(bad.sum())
        if n_bad:
            rr = np.where(bad, raw, rr)
        return np.log10(rr), n_bad
    raise ValueError('unknown transform %r' % (ctx.transform,))


#: How the surrogate maps a latent ``z*`` back to an observed ``y*``.
#:
#: ``'hull'`` (**primary**) -- linear interpolation over the STRICTLY-INCREASING
#: HULL of the PAV breakpoints, i.e. the exact inverse of the
#: :func:`cliff.latent.ginv` the observed pipeline uses to go from ``y`` to
#: ``z``.  ``'pav_step'`` -- the spec's literal ``y* = g(z*)`` with sklearn's
#: step/ramp ``predict``.  See :func:`link_forward` for why the primary is not
#: the literal reading.
LINK_MODES = ('hull', 'pav_step')

#: Where the censoring threshold on the latent scale comes from.
#:
#: ``'calibrated'`` (**primary**) -- the empirical quantile of ``z*`` at the
#: OBSERVED censored fraction.  ``'ginv'`` -- the spec's literal ``g^-1(L)``.
#: See :func:`observe`.
CLAMP_MODES = ('calibrated', 'ginv')


def link_forward(ctx, z, mode='hull'):
    """``y* = g(z*)``, the surrogate's observation map.

    **Why the primary is the hull and not sklearn's ``predict``.**  Spec Sec.3
    writes ``y* = g(z*)``, and ``g`` from step 2 is ``IsotonicRegression``, whose
    forward map is a step/ramp function: its level set is the PAV block means,
    and there are only 347 of those on GB1_IgG-Fc_1FCC (the same 347-383
    breakpoint set the alternation limit-cycles over, ORCHESTRATOR D4) against
    82,124 distinct observed ``y``.  Taken literally, N1 therefore emits a ``y*``
    with 711 distinct values where the data has 82,124 -- it DESTROYS the value
    diversity and the tie mass that spec Sec.3 lists among N1's preserved
    invariants, and it does so asymmetrically: the observed ``z = ginv(y)`` is
    continuous because ``ginv`` interpolates over the hull, while a literal
    ``g(z*)`` is not.  The observed and surrogate arms would then be running two
    different observation models, which is precisely the bias G4 exists to
    catch.  The hull map ``np.interp(z, mid, uy)`` is the exact inverse of
    ``ginv``'s ``np.interp(y, uy, mid)``: it has the same shape, the same range
    and the same knots, so nothing about the fitted link changes -- only the
    within-plateau interpolation, which the observed arm already does.
    Measured effect: :func:`link_audit`.
    """
    if mode not in LINK_MODES:
        raise ValueError('link mode %r not in %r' % (mode, LINK_MODES))
    if mode == 'pav_step':
        return g_apply(ctx.g_knots, z)
    uy, mid = ctx.hull
    z = np.asarray(z, dtype=np.float64)
    if uy.size == 1:
        return np.full(z.shape, uy[0], dtype=np.float64)
    return np.interp(z, mid, uy)


def observe(ctx, z_star, rng, *, link_mode='hull', clamp_mode='calibrated'):
    """``y* = clamp(g(z*))`` on the assay's decimal grid -- spec Sec.3's
    "re-apply the assay's clamp and its decimal grid".

    **The clamp is applied on the LATENT scale**, which is the only place it can
    be applied at all: ``g`` is fitted on the uncensored rows only (spec
    Sec.1.0's E-step reading, and unavoidable -- the E-step needs ``g^-1(L)``
    and on CR9114_FluAH3 the ``L`` plateau would be 89.05% of the PAV fit), so
    ``g``'s y-range EXCLUDES the floor and the spec's literal
    ``y* <- max(y*, L)`` is an exact no-op that hands every censored surrogate
    ZERO censored rows.  The Tobit generative model is ``y = L`` iff
    ``z <= c``.

    **Where ``c`` comes from, and why not ``g^-1(L)``.**  ``ginv`` clips to
    ``[min phi, max phi]`` over the UNCENSORED rows, so ``g^-1(L)`` for an ``L``
    below ``g``'s y-range is pinned at ``lo = min(phi | uncensored)``, and
    ``P(phi + eps < min phi_unc)`` is small by construction.  Measured with that
    reading: SARS2-RBD_6M0J gets 838 clamped rows against 5,216 observed (a 6.2x
    under-censoring) and CR9114_FluAH1 3,887 against 1,675 (2.3x OVER) -- i.e.
    the literal clamp misses N1's "preserves censored mass" in both directions
    and G10's 0.02 composition tolerance cannot survive it.  ``clamp_mode
    ='calibrated'`` instead takes ``c`` as the empirical quantile of ``z*`` at
    the observed censored fraction: a one-parameter calibration of the detection
    limit that makes the preserved invariant actually hold, with the same
    generative story (the lowest latent values are the ones below the limit).
    ``clamp_mode='ginv'`` keeps the literal reading available as a T13 knob and
    is what the audit above reports.

    A clamped row is assigned one of the detected floor LEVELS in the observed
    proportions -- SARS2-RBD reports two floor strings (-4.84 at 14.53% and
    -4.76 at 9.31%), so a single-level replay would get the tie mass wrong.
    """
    if clamp_mode not in CLAMP_MODES:
        raise ValueError('clamp mode %r not in %r' % (clamp_mode, CLAMP_MODES))
    y = link_forward(ctx, z_star, link_mode)
    n_floor = n_ceil = 0
    if ctx.floor_levels:
        if clamp_mode == 'calibrated':
            k = int(round(ctx.floor_frac * ctx.n))
            c = (np.partition(z_star, k - 1)[k - 1] if 0 < k < ctx.n
                 else (-np.inf if k <= 0 else np.inf))
        else:
            c = float(ginv(ctx.g_knots,
                           np.array(ctx.floor_levels, dtype=np.float64),
                           ctx.lo, ctx.hi).max())
        hit = z_star <= c
        n_floor = int(hit.sum())
        if n_floor:
            if len(ctx.floor_levels) == 1:
                y[hit] = ctx.floor_levels[0]
            else:
                pick = rng.choice(len(ctx.floor_levels), size=n_floor,
                                  p=ctx.floor_probs)
                y[hit] = np.asarray(ctx.floor_levels, dtype=np.float64)[pick]
    if ctx.ceil_levels:
        if clamp_mode == 'calibrated':
            k = int(round(ctx.ceil_frac * ctx.n))
            c = (np.partition(z_star, ctx.n - k)[ctx.n - k] if 0 < k < ctx.n
                 else (np.inf if k <= 0 else -np.inf))
        else:
            c = float(ginv(ctx.g_knots,
                           np.array(ctx.ceil_levels, dtype=np.float64),
                           ctx.lo, ctx.hi).min())
        hit = z_star >= c
        n_ceil = int(hit.sum())
        if n_ceil:
            y[hit] = ctx.ceil_levels[0]
    y, n_bad = _round_grid(ctx, y)
    return y, dict(n_floor=n_floor, n_ceil=n_ceil, n_grid_guard=n_bad)


def link_audit(dms_id_or_ctx, *, B=8, verbose=True):
    """Measure what the two ``link_mode`` / ``clamp_mode`` readings actually do.

    Reported rather than asserted, because both are deviations from the spec's
    literal wording (see :func:`link_forward`, :func:`observe`).
    """
    ctx = (dms_id_or_ctx if isinstance(dms_id_or_ctx, NullContext)
           else get_context(dms_id_or_ctx, verify=False))
    uy, mid = ctx.hull
    row = dict(DMS_id=ctx.dms_id, n=ctx.n, n_g_levels=int(uy.size),
               n_uniq_y=int(np.unique(ctx.y).size),
               floor_frac_obs=round(ctx.floor_frac, 6))
    # observed rows outside the hull's y-range: their z is clipped either way
    row['frac_y_outside_hull'] = round(float(((ctx.y < uy[0])
                                              | (ctx.y > uy[-1])).mean()), 6)
    for lm in LINK_MODES:
        for cm in CLAMP_MODES:
            uq, ff = [], []
            for b in range(int(B)):
                rng = np.random.default_rng([12345, b])
                sd = sigma_eval(ctx.sigma_knots, ctx.phi)
                zs = ctx.phi + rng.standard_normal(ctx.n) * sd
                ys, _m = observe(ctx, zs, rng, link_mode=lm, clamp_mode=cm)
                uq.append(np.unique(ys).size)
                ff.append(float(_censor_mask_of(ctx, ys).mean()))
            row['uniq_y_%s' % lm] = int(np.mean(uq))
            row['floor_frac_%s_%s' % (lm, cm)] = round(float(np.mean(ff)), 6)
    if verbose:
        print('    %-40s g levels=%5d  uniq y=%6d  uniq y*(hull)=%6d '
              ' uniq y*(pav_step)=%5d  floor obs=%.4f  ginv=%.4f  '
              'calibrated=%.4f'
              % (ctx.dms_id, row['n_g_levels'], row['n_uniq_y'],
                 row['uniq_y_hull'], row['uniq_y_pav_step'],
                 row['floor_frac_obs'], row['floor_frac_hull_ginv'],
                 row['floor_frac_hull_calibrated']))
    return row


def _censor_mask_of(ctx, y):
    if not ctx.censor_levels:
        return np.zeros(y.size, dtype=bool)
    return np.isin(y, np.asarray(ctx.censor_levels, dtype=np.float64))


# =========================================================================== #
# N1 / N2 / N2b / N2c / N3 surrogates                                         #
# =========================================================================== #

def surrogate_N1(fit, y, rng, *, clamp, quantum, link_mode='hull',
                 clamp_mode='calibrated'):
    """Spec Sec.3 N1: ``z* = X beta + eps*``, ``eps* ~ N(0, sigma^2(phi))``,
    ``y* = g(z*)``, clamp + grid.

    Spec Sec.3's signature.  ``fit`` is the :class:`NullContext` (which carries
    ``beta, phi, g, sigma`` exactly as ``LatentFit`` does, read from the cache
    rather than refitted -- spec Sec.5 forbids refitting the observed fit);
    ``y`` is unused and accepted only to keep the spec's argument list, since
    ``y*`` is generated, not perturbed; ``clamp`` and ``quantum`` are likewise
    the spec's names for what the context already carries and are asserted
    consistent rather than re-derived, so a caller cannot silently apply a
    different clamp to the null than to the data.
    """
    ctx = fit
    _assert_clamp(ctx, clamp, quantum)
    sd = sigma_eval(ctx.sigma_knots, ctx.phi)
    z_star = ctx.phi + rng.standard_normal(ctx.n) * sd
    return observe(ctx, z_star, rng, link_mode=link_mode,
                   clamp_mode=clamp_mode)


def surrogate_N2(fit, e, rng, strata):
    """Spec Sec.3 N2: permute ``e`` within ``(mutation order x phi-decile)``
    strata, ``beta, phi, g, sigma`` held fixed.  Returns the permuted residual.

    ``fit`` is accepted for the spec's signature and is used only to validate
    the length.  Rows with a non-finite ``e`` (unseen design column at cross-fit
    time -- CD19 has 1,467 such columns) sit in their own stratum and are not
    exchanged: permuting a NaN into a finite row's slot would silently shrink
    ``P_a`` differently in every replicate.
    """
    e = np.asarray(e, dtype=np.float64)
    if fit is not None and getattr(fit, 'n', e.size) != e.size:
        raise ValueError('surrogate_N2: e has length %d, fit has n=%d'
                         % (e.size, fit.n))
    return permute_within_strata(e, strata, rng)


def surrogate_N2b(fit_pairwise, rng, *, clamp, quantum, link_mode='hull',
                  clamp_mode='calibrated'):
    """Spec Sec.3 N2b: additive + ridge pairwise + link.

    ``z* = phi_pair + eps*`` with ``phi_pair = b0 + X beta_r + Z gamma_r`` and
    ``eps* ~ N(0, sigma_pair^2(phi_pair))``.  Then clamp + grid, and the caller
    refits the **additive** model -- which is the whole point: the pairwise mean
    structure has to reappear as residual structure.

    The noise scale is the PAIRWISE fit's own residual scale, not the additive
    fit's.  Using ``sigma(phi_add)`` (the additive residual scale, which already
    contains the pairwise signal) would double-count it and inflate the
    surrogate's latent variance above the data's.
    """
    pf = fit_pairwise
    ctx = pf.ctx
    _assert_clamp(ctx, clamp, quantum)
    sd = sigma_eval(pf.sigma_knots, pf.phi_pair)
    z_star = pf.phi_pair + rng.standard_normal(ctx.n) * sd
    return observe(ctx, z_star, rng, link_mode=link_mode,
                   clamp_mode=clamp_mode)


def surrogate_N2c(fit, rng, *, kurtosis_target, clamp, quantum,
                  link_mode='hull', clamp_mode='calibrated'):
    """Spec Sec.3 N2c: ``eps* ~ N(0, sigma^2(phi) V)``, ``V`` a two-point
    discrete mixture calibrated so the MARGINAL residual kurtosis matches
    ``kurtosis_target`` exactly.

    See :func:`two_point_scale_mixture` for the calibration and for why the
    two-point support is pinned without a free knob.
    """
    ctx = fit
    _assert_clamp(ctx, clamp, quantum)
    mix = two_point_scale_mixture(ctx.kurt['K_obs'] if kurtosis_target is None
                                  else float(kurtosis_target),
                                  ctx.kurt['K_het'])
    sd = sigma_eval(ctx.sigma_knots, ctx.phi)
    if mix['degenerate']:
        v = np.ones(ctx.n, dtype=np.float64)
    else:
        hi = rng.random(ctx.n) < mix['p']
        v = np.where(hi, mix['v_hi'], mix['v_lo'])
    z_star = ctx.phi + rng.standard_normal(ctx.n) * sd * np.sqrt(v)
    out, meta = observe(ctx, z_star, rng, link_mode=link_mode,
                        clamp_mode=clamp_mode)
    meta.update(mix)
    return out, meta


def surrogate_N3(fit, y, rng):
    """Spec Sec.3 N3: free permutation of ``y`` (House-of-Cards).

    **Upper calibration only** -- rejecting N3 is trivial and uninformative, so
    no hypothesis test is ever read off it.  No clamp and no grid are needed: a
    permutation of ``y`` is already on the assay's grid and already carries its
    censored mass exactly.
    """
    ctx = fit
    y = ctx.y if y is None else np.asarray(y, dtype=np.float64)
    return rng.permutation(y), dict(n_floor=int(ctx.censor_mask.sum()),
                                    n_ceil=0, n_grid_guard=0)


def _assert_clamp(ctx, clamp, quantum):
    """The clamp and the grid are properties of the assay, never of the caller.

    Spec Sec.3 puts them in the surrogate signatures; accepting them silently
    would let a caller apply a different observation model to the null than to
    the data, which is precisely the bias G4 exists to catch.  ``None`` means
    "use the assay's", anything else must match.
    """
    if clamp is not None and tuple(np.atleast_1d(clamp).ravel().tolist()) \
            != tuple(float(v) for v in ctx.censor_levels):
        raise ValueError('%s: clamp %r != the assay\'s detected levels %r'
                         % (ctx.dms_id, clamp, ctx.censor_levels))
    if quantum is not None and abs(float(quantum) - ctx.quantum) > 1e-15:
        raise ValueError('%s: quantum %r != the assay\'s %r'
                         % (ctx.dms_id, quantum, ctx.quantum))


# --------------------------------------------------------------------------- #
# N2 strata                                                                   #
# --------------------------------------------------------------------------- #

def make_strata(n_muts, phi, *, n_bins=_N2_PHI_BINS, censor_mask=None):
    """``(mutation order x phi-decile)`` stratum id per variant (spec Sec.3 N2).

    Equal-count ``phi`` bins WITHIN each mutation order, so a rare high order
    does not end up spread across ten deciles of a global ranking and left with
    one variant per cell (which would make the permutation the identity).
    Non-finite ``phi`` gets its own stratum per order and is never exchanged.

    **The censoring indicator is a third stratum key**, which the spec does not
    say and which is not optional.  A censored row's residual is
    ``-sigma(phi) * Mills(a)`` -- a deterministic function of ``phi``, not a
    measurement error (``latent.sigma_of_phi``'s own docstring says so and
    excludes those rows from the scale estimate).  Exchanging one into an
    uncensored row's slot destroys the invariant N2 exists to preserve: the
    UNCENSORED residual marginal.  Measured without the key: CR9114_FluAH3's
    ``kurt_e`` moved 4.786 -> 2.846 and its ``rate(|c| >= 3)`` from the observed
    0.0108 to an N2 mean of 0.0875, an 8x inflation manufactured entirely by
    Mills-ratio values leaking into the uncensored set.  With the key,
    ``kurt_e`` under N2 equals the observed value exactly on every assay, which
    is the invariant.
    """
    n_muts = np.asarray(n_muts)
    phi = np.asarray(phi, dtype=np.float64)
    cm = (np.zeros(phi.size, dtype=bool) if censor_mask is None
          else np.asarray(censor_mask, dtype=bool))
    out = np.full(phi.size, -1, dtype=np.int32)
    s = 0
    for m, cflag in [(mm, cc) for mm in np.unique(n_muts)
                     for cc in ((False, True) if cm.any() else (False,))]:
        sel = np.nonzero((n_muts == m) & (cm == cflag))[0]
        if sel.size == 0:
            continue
        fin = np.isfinite(phi[sel])
        bad = sel[~fin]
        if bad.size:
            out[bad] = s
            s += 1
        good = sel[fin]
        if good.size == 0:
            continue
        order = good[np.argsort(phi[good], kind='stable')]
        k = max(1, min(int(n_bins), good.size))
        for part in np.array_split(order, k):
            if part.size:
                out[part] = s
                s += 1
    return out


def permute_within_strata(values, strata, rng, *, exchange_singletons=False):
    """Permute ``values`` within each stratum.  The engine behind N2/NS1/NS2.

    A stratum of size 1 is the identity by construction; that is not a bug but it
    IS a loss of power, so :func:`n2_power` counts it.
    """
    values = np.asarray(values)
    strata = np.asarray(strata)
    if values.shape[0] != strata.shape[0]:
        raise ValueError('values/strata length mismatch: %d vs %d'
                         % (values.shape[0], strata.shape[0]))
    out = values.copy()
    order = np.argsort(strata, kind='stable')
    s_sorted = strata[order]
    bounds = np.nonzero(np.diff(s_sorted))[0] + 1
    for grp in np.split(order, bounds):
        if grp.size > 1:
            out[grp] = values[rng.permutation(grp)]
        elif grp.size == 1 and exchange_singletons:
            pass
    return out


# --------------------------------------------------------------------------- #
# N2c calibration                                                             #
# --------------------------------------------------------------------------- #

def kurtosis_targets(ctx):
    """``(K_obs, K_het)`` -- the two kurtoses N2c has to reconcile.

    * ``K_obs`` = the observed MARGINAL residual kurtosis, ``m4/m2^2`` of
      ``e_oof`` over the finite uncensored rows.  Marginal, i.e. pooled and
      UNstandardised: that is the quantity the spec names.
    * ``K_het`` = ``3 E[sigma^4] / E[sigma^2]^2`` over the same rows -- the
      kurtosis a pure Gaussian with the fitted level-dependent scale already
      produces, with no mixture at all.  Heteroscedasticity alone is a scale
      mixture, so ``K_het > 3`` whenever ``sigma(phi)`` is not constant.

    N2c's job is the residual gap ``K_obs / K_het``.
    """
    fin = ctx.oof_finite & ~ctx.censor_mask
    e = ctx.e_oof[fin]
    s = ctx.sigma_oof[fin]
    n = e.size
    out = dict(n=int(n), K_obs=float('nan'), K_het=float('nan'),
               sd_over_mad=float('nan'))
    if n < 8:
        return out
    d = e - e.mean()
    m2 = float((d * d).mean())
    m4 = float((d ** 4).mean())
    out['K_obs'] = m4 / (m2 * m2) if m2 > 0 else float('nan')
    s2 = float((s * s).mean())
    s4 = float((s ** 4).mean())
    out['K_het'] = 3.0 * s4 / (s2 * s2) if s2 > 0 else float('nan')
    ms = mad_scaled(e)
    out['sd_over_mad'] = float(e.std() / ms) if ms > 0 else float('nan')
    out['sigma_dyn_range'] = float(ctx.sigma_knots[1].max()
                                   / ctx.sigma_knots[1].min())
    return out


def two_point_scale_mixture(K_obs, K_het):
    """The two-point ``V`` with ``E[V] = 1`` that lifts ``K_het`` to ``K_obs``.

    ``eps = sigma(phi) sqrt(V) Z`` gives, marginally,
    ``E[eps^2] = E[sigma^2] E[V]`` and ``E[eps^4] = 3 E[sigma^4] E[V^2]``, so
    with ``E[V] = 1`` (marginal variance preserved -- N2c must preserve
    everything N1 preserves and change only the SHAPE)::

        kurtosis = K_het * E[V^2] = K_het * (1 + Var V)
        =>  tau^2 := Var V = K_obs / K_het - 1

    Two moments leave ONE free parameter in a two-point distribution, and the
    spec does not pin it.  It is pinned here without a knob: among all two-point
    ``V`` with mean 1 and variance ``tau^2`` this returns the one that MINIMISES
    ``v_hi / v_lo``, i.e. **the least extreme scale mixture that reproduces the
    observed kurtosis exactly**.  Writing ``u = sqrt(p/(1-p))``,
    ``v_hi = 1 + tau/u`` and ``v_lo = 1 - tau u``, the ratio is minimised at
    ``u* = sqrt(1+tau^2) - tau`` (the positive root of ``u^2 + 2 tau u - 1``),
    hence ``p* = u*^2/(1+u*^2)``, which also guarantees ``v_lo > 0``.  The choice
    is the conservative one for G7: if even the least extreme scale mixture of
    this class inflates the tail statistics, the inflation is a property of
    heteroscedasticity and not an artefact of picking an extreme mixture.  ``p``
    can still be overridden by a caller for a sensitivity surface.

    ``K_obs <= K_het`` means pure heteroscedasticity already over-explains the
    observed tail; the mixture then degenerates to ``V == 1`` (N2c == N1) and
    says so, rather than solving for a negative variance.
    """
    out = dict(K_obs=float(K_obs), K_het=float(K_het), tau2=float('nan'),
               p=float('nan'), v_lo=1.0, v_hi=1.0, ratio=1.0, degenerate=True)
    if not (np.isfinite(K_obs) and np.isfinite(K_het) and K_het > 0):
        out['reason'] = 'kurtosis not estimable'
        return out
    tau2 = K_obs / K_het - 1.0
    out['tau2'] = float(tau2)
    if tau2 <= 1e-9:
        out['reason'] = ('K_obs <= K_het: heteroscedasticity alone already '
                         'produces the observed kurtosis; V == 1')
        return out
    tau = math.sqrt(tau2)
    u = math.sqrt(1.0 + tau2) - tau
    p = u * u / (1.0 + u * u)
    v_hi = 1.0 + tau / u
    v_lo = 1.0 - tau * u
    out.update(p=float(p), v_lo=float(v_lo), v_hi=float(v_hi),
               ratio=float(v_hi / v_lo), degenerate=False,
               reason='min-ratio two-point mixture, E[V]=1')
    # by construction; assert so a refactor cannot break the calibration
    m1 = (1 - p) * v_lo + p * v_hi
    m2 = (1 - p) * v_lo ** 2 + p * v_hi ** 2
    assert abs(m1 - 1.0) < 1e-10, m1
    assert abs(K_het * m2 - K_obs) < 1e-8 * max(1.0, abs(K_obs)), (K_het * m2,
                                                                   K_obs)
    return out


# --------------------------------------------------------------------------- #
# N2b: the ridge pairwise fit                                                 #
# --------------------------------------------------------------------------- #

class PairwiseFit(object):
    __slots__ = ('ctx', 'Z', 'col_pairs', 'cooccur', 'b0', 'beta_r', 'gamma_r',
                 'phi_pair', 'sigma_knots', 'lam', 'lam_grid', 'cv_mse',
                 'r2_in', 'r2_add_in', 'n_cols', 'feasible', 'wall_s', 'note')

    def __init__(self, **kw):
        for s in self.__slots__:
            setattr(self, s, kw.get(s))

    def __repr__(self):
        return ('PairwiseFit(%s, Z cols=%d, lam=%.4g, feasible=%s)'
                % (self.ctx.dms_id, self.n_cols, self.lam or float('nan'),
                   self.feasible))


def pairwise_design(ctx, *, min_cooccur=None):
    """``Z`` -- one column per SUBSTITUTION pair co-observed ``>= min_cooccur``.

    Spec Sec.3 N2b says "site pairs co-observed >= 20 times", but the spec's own
    L4 arithmetic proves the columns are substitution pairs, not site pairs:
    "GB1_2016 (2,166 / 22,176)" is exactly ``C(4,2) * 19 * 19 = 2,166``, i.e. one
    column per ordered-by-index pair of X columns, and a site-pair reading would
    give 6.  Implemented as substitution pairs and reported as a deviation.

    ``GB1_IgG-Fc_1FCC``'s complete doubles scan gives every substitution pair a
    co-occurrence of exactly 1, so ``Z`` is EMPTY there and N2b degenerates to
    N1 -- the same fact the spec records as "L4 infeasible (1 obs/Z-column)".
    That is reported, never worked around.
    """
    if min_cooccur is None:
        min_cooccur = THRESH['N2b_min_cooccur']
    X = ctx.X.tocsr()
    indptr, indices = X.indptr, X.indices
    M = X.shape[1]
    rows_i, rows_j = [], []
    row_of = []
    for r in range(ctx.n):
        cols = indices[indptr[r]:indptr[r + 1]]
        k = cols.size
        if k < 2:
            continue
        cs = np.sort(cols)
        ii, jj = np.triu_indices(k, 1)
        rows_i.append(cs[ii])
        rows_j.append(cs[jj])
        row_of.append(np.full(ii.size, r, dtype=np.int32))
    if not rows_i:
        return (sp.csr_matrix((ctx.n, 0)), np.zeros((0, 2), dtype=np.int32),
                np.zeros(0, dtype=np.int64))
    ci = np.concatenate(rows_i).astype(np.int64)
    cj = np.concatenate(rows_j).astype(np.int64)
    rr = np.concatenate(row_of)
    lin = ci * M + cj
    uniq, inv = np.unique(lin, return_inverse=True)
    cnt = np.bincount(inv, minlength=uniq.size)
    keep = cnt >= int(min_cooccur)
    if not keep.any():
        return (sp.csr_matrix((ctx.n, 0)), np.zeros((0, 2), dtype=np.int32),
                cnt[keep].astype(np.int64))
    newcol = np.full(uniq.size, -1, dtype=np.int64)
    newcol[keep] = np.arange(int(keep.sum()))
    take = newcol[inv] >= 0
    Z = sp.csr_matrix((np.ones(int(take.sum())), (rr[take], newcol[inv[take]])),
                      shape=(ctx.n, int(keep.sum())))
    col_pairs = np.stack([uniq[keep] // M, uniq[keep] % M], axis=1).astype(np.int32)
    return Z, col_pairs, cnt[keep].astype(np.int64)


def _ridge_lsqr(A, b, lam, *, unpenalised=(0,), atol=1e-10, x0=None):
    """Ridge by the augmented least-squares system ``[[A],[sqrt(lam) D]]``.

    ``D`` zeroes the penalty on ``unpenalised`` columns (the intercept), which is
    the standard convention and the only one under which ``lam -> inf`` reduces
    to the intercept-only model rather than to zero.  Column-scaled, exactly as
    :func:`cliff.latent.fit_latent` does (spec's measured 1.8 s vs 3.04 s per
    solve), so the ridge path costs what the additive path costs.
    """
    n, k = A.shape
    cn = _latent._colnorms(A)
    As = (A @ sp.diags(1.0 / cn)).tocsr()
    d = np.sqrt(lam) * np.ones(k)
    for c in unpenalised:
        d[c] = 0.0
    aug = sp.vstack([As, sp.diags(d)], format='csr')
    rhs = np.concatenate([b, np.zeros(k)])
    res = lsqr(aug, rhs, atol=atol, btol=atol, x0=x0)
    return np.asarray(res[0], dtype=np.float64) / cn, int(res[2])


def fit_pairwise_ridge(ctx, *, min_cooccur=None, n_lambda=None, n_folds=None,
                       verbose=False):
    """N2b's mean model: ``z ~ [1|X|Z]`` by ridge, ``lam`` by 5-fold CV on
    held-out VARIANTS (spec Sec.3 N2b).

    Fitted on the observed latent ``z`` from the cached fit -- the additive fit
    is not redone, per spec Sec.5's "never refit a cached latent fit for the
    observed data".  The CV folds are the SAME cross-fit folds the rest of the
    study uses, so nothing about ``lam`` depends on a second random partition.
    """
    t0 = time.time()
    if n_lambda is None:
        n_lambda = THRESH['L4_n_lambda']
    if n_folds is None:
        n_folds = THRESH['L4_inner_folds']
    Z, col_pairs, cooccur = pairwise_design(ctx, min_cooccur=min_cooccur)
    z = ctx.z
    unc = ~ctx.censor_mask
    if Z.shape[1] == 0:
        sk = _latent.sigma_of_phi(ctx.phi[unc], (z - ctx.phi)[unc],
                                  sigma_floor=ctx.sigma_floor)
        pf = PairwiseFit(ctx=ctx, Z=Z, col_pairs=col_pairs, cooccur=cooccur,
                         b0=float(ctx.beta[0]), beta_r=ctx.beta[1:].copy(),
                         gamma_r=np.zeros(0), phi_pair=ctx.phi.copy(),
                         sigma_knots=(sk[0], sk[1]), lam=float('nan'),
                         lam_grid=(), cv_mse=(), r2_in=float('nan'),
                         r2_add_in=float('nan'), n_cols=0, feasible=False,
                         wall_s=time.time() - t0,
                         note=('no substitution pair is co-observed >= %d times '
                               '=> Z is empty and N2b degenerates to N1'
                               % (min_cooccur or THRESH['N2b_min_cooccur'])))
        ctx.pairwise = pf
        return pf
    A = sp.hstack([ctx.A, Z], format='csr')
    # lambda grid: log-spaced around ||A||_F^2/k, the scale at which the penalty
    # is comparable to the data term.  12 points, spec Sec.1.4 L4_n_lambda.
    scale = float(A.multiply(A).sum()) / A.shape[1]
    lam_grid = np.geomspace(1e-4 * scale, 1e2 * scale, int(n_lambda))
    folds = ctx.folds
    mse = np.zeros(lam_grid.size)
    for k in np.unique(folds):
        te = folds == k
        tr = ~te
        Atr, ztr = A[tr], z[tr]
        Ate, zte = A[te], z[te]
        for li, lam in enumerate(lam_grid):
            b, _ = _ridge_lsqr(Atr, ztr, lam)
            r = zte - Ate.dot(b)
            mse[li] += float((r * r).sum())
    mse /= float(ctx.n)
    lam = float(lam_grid[int(np.argmin(mse))])
    b, _ = _ridge_lsqr(A, z, lam)
    phi_pair = A.dot(b)
    r = z - phi_pair
    sk = _latent.sigma_of_phi(phi_pair[unc], r[unc], sigma_floor=ctx.sigma_floor)
    ss = float(((z[unc] - z[unc].mean()) ** 2).sum())
    pf = PairwiseFit(
        ctx=ctx, Z=Z, col_pairs=col_pairs, cooccur=cooccur,
        b0=float(b[0]), beta_r=b[1:1 + ctx.M].copy(),
        gamma_r=b[1 + ctx.M:].copy(), phi_pair=phi_pair,
        sigma_knots=(sk[0], sk[1]), lam=lam,
        lam_grid=tuple(float(v) for v in lam_grid),
        cv_mse=tuple(float(v) for v in mse),
        r2_in=(1.0 - float((r[unc] ** 2).sum()) / ss) if ss > 0 else float('nan'),
        r2_add_in=(1.0 - float(((z - ctx.phi)[unc] ** 2).sum()) / ss
                   if ss > 0 else float('nan')),
        n_cols=int(Z.shape[1]), feasible=True, wall_s=time.time() - t0,
        note='')
    if verbose:
        print('    [N2b] %s: Z cols=%d (min_cooccur=%d), lam=%.4g, '
              'r2 add=%.4f -> pairwise=%.4f, %.1fs'
              % (ctx.dms_id, pf.n_cols, min_cooccur or THRESH['N2b_min_cooccur'],
                 lam, pf.r2_add_in, pf.r2_in, pf.wall_s))
    ctx.pairwise = pf
    return pf


# =========================================================================== #
# NS1 / NS2 / NS3 -- stratified label permutations (spec Sec.1.5)             #
# =========================================================================== #

def _decile(v, k=10):
    """Equal-count bin index of ``v`` (ties broken by a stable sort, so the bin
    sizes are equal even on a heavily tied covariate)."""
    v = np.asarray(v, dtype=np.float64)
    out = np.zeros(v.size, dtype=np.int32)
    fin = np.isfinite(v)
    idx = np.nonzero(fin)[0]
    if idx.size:
        order = idx[np.argsort(v[idx], kind='stable')]
        for b, part in enumerate(np.array_split(order, min(k, order.size))):
            out[part] = b
    out[~fin] = -1
    return out


def _strata_from(cols):
    """A single int32 stratum id from a list of integer/string label arrays."""
    keys = [np.asarray(c) for c in cols]
    tup = list(zip(*[k.tolist() for k in keys]))
    lut = {}
    out = np.empty(len(tup), dtype=np.int32)
    for i, t in enumerate(tup):
        if t not in lut:
            lut[t] = len(lut)
        out[i] = lut[t]
    return out


def permute_NS1(pos_table, rng, *, label='is_iface_5A'):
    """NS1 (spec Sec.1.5): permute the INTERFACE label within
    ``(burial x aa-class x |beta| x depth)`` strata.

    ``pos_table`` is ``T09_structure_sites``-shaped: it must carry the label plus
    ``levy_class`` (burial), ``aa_class``, ``beta_hat_abs`` and ``depth_tertile``.
    ``|beta|`` is continuous, so it enters as its own decile -- matching on a raw
    float would put every position in its own stratum and make the permutation
    the identity.
    """
    need = ('levy_class', 'aa_class', 'beta_hat_abs', 'depth_tertile', label)
    miss = [c for c in need if c not in pos_table]
    if miss:
        raise KeyError('permute_NS1 needs columns %r (missing %r)' % (need, miss))
    strata = _strata_from([np.asarray(pos_table['levy_class']),
                           np.asarray(pos_table['aa_class']),
                           _decile(pos_table['beta_hat_abs'], 10),
                           np.asarray(pos_table['depth_tertile'])])
    return permute_within_strata(np.asarray(pos_table[label]), strata, rng)


def permute_NS2(eps_table, rng, *, label='is_cliff_3sigma'):
    """NS2 (spec Sec.1.5): permute the CLIFF label within
    ``(seq-separation-decile x rsa-tertile)`` strata.

    ``eps_table`` is ``T10_structure_pairs``-shaped: ``seq_separation``,
    ``rsa_iso`` (or ``rsa_s``/``rsa_t``, whose mean is used) and the label.
    """
    if label not in eps_table:
        raise KeyError('permute_NS2 needs the %r column' % label)
    if 'seq_separation' not in eps_table:
        raise KeyError('permute_NS2 needs seq_separation')
    if 'rsa_iso' in eps_table:
        rsa = np.asarray(eps_table['rsa_iso'], dtype=np.float64)
    elif 'rsa_s' in eps_table and 'rsa_t' in eps_table:
        rsa = 0.5 * (np.asarray(eps_table['rsa_s'], dtype=np.float64)
                     + np.asarray(eps_table['rsa_t'], dtype=np.float64))
    else:
        raise KeyError('permute_NS2 needs rsa_iso or rsa_s+rsa_t')
    strata = _strata_from([_decile(eps_table['seq_separation'], 10),
                           _decile(rsa, 3)])
    return permute_within_strata(np.asarray(eps_table[label]), strata, rng)


def permute_NS3(Z, rng):
    """NS3 (spec Sec.1.5): permute the PARTNER label within each ROW of the
    ``position x partner`` matrix.

    Row-wise, because C4-I's ``M_F`` is a double-centred Mantel correlation: the
    null has to preserve each position's partner-invariant propensity (its row
    mean, which row-centering removes algebraically) and destroy only the
    position-partner assignment.  ``NaN`` cells (a partner a position has no
    measurement against) stay where they are; only the observed cells of a row
    are permuted among themselves.
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError('permute_NS3 needs a 2-D position x partner matrix')
    out = Z.copy()
    for i in range(Z.shape[0]):
        col = np.nonzero(np.isfinite(Z[i]))[0]
        if col.size > 1:
            out[i, col] = Z[i, rng.permutation(col)]
    return out


# =========================================================================== #
# the statistic vector                                                        #
# =========================================================================== #

def c_hat(e, sigma, idx, *, mu=None):
    """The cliff statistic on nested pairs (spec Sec.1.0, ORCHESTRATOR D2).

    ``mu=None`` gives the spec's UNCENTRED form ``(e_v - e_u)/sqrt(s2_u+s2_v)``,
    kept for the T13 ``centring`` sensitivity row.  With ``mu`` supplied (the
    primary definition) the phi-dependent LOCATION of ``e`` is removed at each
    endpoint first, which is what stops a pair whose endpoints sit in different
    ``phi`` regions from registering as a cliff for a reason that has nothing to
    do with epistasis.
    """
    u, v = idx[:, 0], idx[:, 1]
    if mu is None:
        num = e[v] - e[u]
    else:
        num = (e[v] - mu[v]) - (e[u] - mu[u])
    den = np.sqrt(sigma[u] ** 2 + sigma[v] ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        return num / den


def _pa_mask(ctx, cm, oof_finite):
    """``P_a`` conditions (a) ``B != {}``, (b) neither endpoint censored,
    (c) finite ``phi^oof`` at both endpoints -- evaluated against the CENSORING
    AND CROSS-FIT OF THE REPLICATE, not of the observed data, so the null runs
    the identical selection rule.  (d) is a tier filter the caller applies."""
    u, v = ctx.nested_idx[:, 0], ctx.nested_idx[:, 1]
    return (~ctx.wt_anchored) & ~(cm[u] | cm[v]) & oof_finite[u] & oof_finite[v]


def _icc_oneway(values, groups):
    """One-way random-effects ``ICC = (MSB - MSW)/(MSB + (kbar-1) MSW)``.

    ``np.bincount``, never ``np.add.at`` (spec's numeric hygiene).  Returns
    ``nan`` when fewer than two groups have >= 2 members.
    """
    v = np.asarray(values, dtype=np.float64)
    g = np.asarray(groups)
    ok = np.isfinite(v)
    v, g = v[ok], g[ok]
    if v.size < 4:
        return float('nan'), 0, 0
    _u, gi = np.unique(g, return_inverse=True)
    ng = gi.max() + 1
    cnt = np.bincount(gi, minlength=ng).astype(np.float64)
    tot = np.bincount(gi, weights=v, minlength=ng)
    if (cnt >= 2).sum() < 2:
        return float('nan'), int(v.size), int(ng)
    gm = tot / cnt
    grand = v.mean()
    ssb = float((cnt * (gm - grand) ** 2).sum())
    ssw = float(((v - gm[gi]) ** 2).sum())
    dfb, dfw = ng - 1, v.size - ng
    if dfb <= 0 or dfw <= 0:
        return float('nan'), int(v.size), int(ng)
    msb, msw = ssb / dfb, ssw / dfw
    kbar = v.size / float(ng)
    den = msb + (kbar - 1.0) * msw
    return (float((msb - msw) / den) if den != 0 else float('nan'),
            int(v.size), int(ng))


#: The default statistic vector.  Names are frozen: they become the columns of
#: ``nulls/{id}_{null}_B*_seed*.npz`` and G4/G7 index into them.
def _stat_names():
    names = ['n_Pa', 'frac_c_zero', 'q75', 'q99', 'q999', 'TR1', 'TR2',
             'kurt_c', 'mad_c', 'sd_c',
             'q75_mad', 'q99_mad', 'q999_mad', 'TR1_mad', 'TR2_mad']
    for u in ('sigma', 'mad'):
        for t in TAUS:
            names.append('rate_%s_tau%g' % (u, t))
    names += ['icc_addcol', 'icc_pos', 'n_grp_addcol', 'n_grp_pos',
              'SI', 'V1', 'V2', 'G1', 'Vinf', 'GMD',
              'kurt_e', 'K_het', 'resid_mad_oof', 'frac_oof_finite',
              'floor_frac', 'n_censored', 'n_iter_used', 'wall_s']
    return tuple(names)


STAT_NAMES = _stat_names()

#: Fingerprint of everything that determines what a cached ensemble MEANS: the
#: statistic names, the surrogate's observation model, and the centring.  A
#: cached ``nulls/*.npz`` whose fingerprint differs is REFUSED, not read -- an
#: ensemble is only comparable to an observed value computed the same way, and
#: silently reusing one across a definition change is the most expensive
#: possible mistake in a null-referenced study.
def _stat_version(null=None):
    import hashlib
    blob = '|'.join(STAT_NAMES) + '||hull|calibrated|phi-centred|v1'
    if null == 'N2':
        # the exchange strata are part of what an N2 ensemble MEANS, and only
        # N2's, so a change here must not invalidate the expensive refit nulls
        blob += '|n2strata=order.phidecile.censored'
    return hashlib.md5(blob.encode()).hexdigest()[:12]


STAT_VERSION = _stat_version()


def default_stat_fn(ctx, rep):
    """Every statistic a replicate contributes, from the replicate bundle.

    ``rep`` carries ``y, phi_oof, e_oof, sigma_oof, mu_oof, oof_finite,
    censor_mask`` -- the replicate's OWN, refitted where the null refits.  The
    tail block is C2's (spec Sec.1.3), in **both** unit systems: ``sigma`` =
    the level-dependent ``sqrt(s2_u+s2_v)`` denominator, ``mad`` = one global
    ``1.4826 MAD`` of the same numerator, which is the spec's second unit system
    and the one a wrong ``sigma(phi)`` cannot touch.  The localisation block
    (``icc_addcol``, ``icc_pos``) is this module's G7 probe: under a pure
    heteroscedastic scale mixture the deviation must NOT recur for the same
    added substitution, so these must sit at their null value under N2c while
    the tail statistics inflate.  ``SI/V1/V2`` are the C1 block and run on the
    SEQUENCE Hamming axis (nested union same-site, ORCHESTRATOR D1).
    """
    out = dict.fromkeys(STAT_NAMES, float('nan'))
    y = rep['y']
    idx = ctx.nested_idx
    keep = _pa_mask(ctx, rep['censor_mask'], rep['oof_finite'])
    out['n_Pa'] = float(keep.sum())
    out['n_censored'] = float(rep['censor_mask'].sum())
    out['floor_frac'] = float(rep['censor_mask'].mean())
    out['frac_oof_finite'] = float(rep['oof_finite'].mean())
    fin = rep['oof_finite'] & ~rep['censor_mask']
    if fin.sum() >= 8:
        e = rep['e_oof'][fin]
        d = e - e.mean()
        m2 = float((d * d).mean())
        out['kurt_e'] = float((d ** 4).mean() / (m2 * m2)) if m2 > 0 else np.nan
        out['resid_mad_oof'] = float(mad_scaled(e))
        s = rep['sigma_oof'][fin]
        s2, s4 = float((s * s).mean()), float((s ** 4).mean())
        out['K_het'] = 3.0 * s4 / (s2 * s2) if s2 > 0 else np.nan
    out['n_iter_used'] = float(rep.get('n_iter_used', np.nan))
    out['wall_s'] = float(rep.get('wall_s', np.nan))

    if keep.sum() >= 8:
        sub = idx[keep]
        num = ((rep['e_oof'] - rep['mu_oof'])[sub[:, 1]]
               - (rep['e_oof'] - rep['mu_oof'])[sub[:, 0]])
        den = np.sqrt(rep['sigma_oof'][sub[:, 0]] ** 2
                      + rep['sigma_oof'][sub[:, 1]] ** 2)
        with np.errstate(divide='ignore', invalid='ignore'):
            c = num / den
        good = np.isfinite(c)
        c = c[good]
        num_g = num[good]
        if c.size >= 8:
            ac = np.abs(c)
            out['frac_c_zero'] = float((c == 0).mean())
            q75, q99, q999 = np.percentile(ac, [75.0, 99.0, 99.9])
            out['q75'], out['q99'], out['q999'] = map(float, (q75, q99, q999))
            out['TR1'] = float(q999 / q75) if q75 > 0 else np.nan
            out['TR2'] = float(q99 / q75) if q75 > 0 else np.nan
            dd = c - c.mean()
            m2 = float((dd * dd).mean())
            out['kurt_c'] = float((dd ** 4).mean() / (m2 * m2)) if m2 > 0 else np.nan
            out['mad_c'] = float(mad_scaled(c))
            out['sd_c'] = float(c.std())
            for t in TAUS:
                out['rate_sigma_tau%g' % t] = float((ac >= t).mean())
            # ---- MAD unit system: one global robust scale on the numerator --
            sc = mad_scaled(num_g)
            if sc > 0:
                cm_ = num_g / sc
                am = np.abs(cm_)
                q75m, q99m, q999m = np.percentile(am, [75.0, 99.0, 99.9])
                out['q75_mad'], out['q99_mad'], out['q999_mad'] = \
                    map(float, (q75m, q99m, q999m))
                out['TR1_mad'] = float(q999m / q75m) if q75m > 0 else np.nan
                out['TR2_mad'] = float(q99m / q75m) if q75m > 0 else np.nan
                for t in TAUS:
                    out['rate_mad_tau%g' % t] = float((am >= t).mean())
            # ---- localisation probe -----------------------------------------
            ac_ = ctx.add_col[keep][good]
            po_ = ctx.pos_of_add[keep][good]
            i1, _n1, g1 = _icc_oneway(c, ac_)
            i2, _n2, g2 = _icc_oneway(c, po_)
            out['icc_addcol'], out['n_grp_addcol'] = i1, float(g1)
            out['icc_pos'], out['n_grp_pos'] = i2, float(g2)

    # ---- C1 block, SEQUENCE Hamming axis (D1) ------------------------------ #
    l1 = ctx.lag1_idx
    d = y[l1[:, 0]] - y[l1[:, 1]]
    gmd = _gini_mean_difference(y)
    out['G1'] = float(np.abs(d).mean())
    out['V1'] = float(0.5 * (d * d).mean())
    out['GMD'] = float(gmd)
    out['Vinf'] = float(_v_infinity(y))
    out['SI'] = float(out['G1'] / gmd) if gmd else np.nan
    # V(2) is left EMPTY on purpose and the column is kept so the shape of the
    # statistic vector is stable.  Only lag 1 is cached exactly (nested +
    # same-site); h >= 2 lives in the 2e7 seeded random-pair sample, and spec
    # Sec.5 is explicit that "nulls never recompute a full variogram" -- loading
    # a 160 MB randpair npz into each of 64 workers would cost 10 GB for a
    # statistic this module does not own.  T05's h >= 2 null ribbons are
    # variogram.py's to fill, by calling run_ensemble with its own stat_fn.
    out['V2'] = float('nan')
    return out


# =========================================================================== #
# one replicate                                                               #
# =========================================================================== #

def _refit_bundle(ctx, y_star, meta):
    """Refit ``fit_latent`` + the 5-fold cross-fit FROM SCRATCH on ``y*``.

    The folds are the observed run's folds: N1 preserves the variant set, so the
    partition of variants is part of "the additive-fit estimation error" the
    surrogate is required to preserve.
    """
    t0 = time.time()
    cm = _censor_mask_of(ctx, y_star)
    fit = fit_latent(ctx.X, y_star, cm, ctx.censor_levels,
                     sigma_floor=ctx.sigma_floor)
    cf = crossfit_latent(ctx.X, y_star, cm, ctx.censor_levels, ctx.folds,
                         sigma_floor=ctx.sigma_floor)
    mu_knots = (fit.sigma_knots[0], fit.sigma_knots[2])
    of = np.isfinite(cf['phi_oof']) & np.isfinite(cf['z_oof'])
    return dict(y=y_star, censor_mask=cm, phi_oof=cf['phi_oof'],
                z_oof=cf['z_oof'], e_oof=cf['e_oof'],
                sigma_oof=cf['sigma_oof'],
                mu_oof=sigma_eval(mu_knots, cf['phi_oof']),
                oof_finite=of, n_iter_used=fit.n_iter_used,
                wall_s=time.time() - t0, meta=meta)


def replicate(ctx, null, b, *, seed0=None, stat_fn=None, link_mode='hull',
              clamp_mode='calibrated', centred=True):
    """Statistic vector for replicate ``b`` of ``null`` on ``ctx``'s assay."""
    if null not in NULLS:
        raise ValueError('unknown null %r (have %r)' % (null, NULLS))
    if stat_fn is None:
        stat_fn = default_stat_fn
    ent = (list(seed0) if seed0 is not None
           else config.assay_seed('nulls_' + null, ctx.dms_id))
    rng = np.random.default_rng(list(ent) + [int(b)])
    km = dict(link_mode=link_mode, clamp_mode=clamp_mode)
    t0 = time.time()
    if null == 'N1':
        y_star, meta = surrogate_N1(ctx, None, rng, clamp=None, quantum=None,
                                    **km)
        rep = _refit_bundle(ctx, y_star, meta)
    elif null == 'N2c':
        y_star, meta = surrogate_N2c(ctx, rng, kurtosis_target=None,
                                     clamp=None, quantum=None, **km)
        rep = _refit_bundle(ctx, y_star, meta)
    elif null == 'N2b':
        pf = ctx.pairwise or fit_pairwise_ridge(ctx)
        y_star, meta = surrogate_N2b(pf, rng, clamp=None, quantum=None, **km)
        meta['n2b_feasible'] = pf.feasible
        meta['n2b_cols'] = pf.n_cols
        rep = _refit_bundle(ctx, y_star, meta)
    elif null == 'N3':
        y_star, meta = surrogate_N3(ctx, None, rng)
        rep = _refit_bundle(ctx, y_star, meta)
    else:                                                       # N2
        strata = make_strata(ctx.n_muts, ctx.phi_oof,
                             censor_mask=ctx.censor_mask)
        e_star = surrogate_N2(ctx, ctx.e_oof, rng, strata)
        rep = dict(y=ctx.y, censor_mask=ctx.censor_mask,
                   phi_oof=ctx.phi_oof, z_oof=ctx.z_oof, e_oof=e_star,
                   sigma_oof=ctx.sigma_oof, mu_oof=ctx.mu_oof,
                   oof_finite=ctx.oof_finite, n_iter_used=0,
                   wall_s=0.0, meta=dict(n_floor=int(ctx.censor_mask.sum()),
                                         n_ceil=0, n_grid_guard=0))
    if not centred:
        rep = dict(rep, mu_oof=np.zeros(ctx.n))
    out = stat_fn(ctx, rep)
    out['wall_s'] = time.time() - t0
    return out


def observed_stats(ctx, *, stat_fn=None, centred=True):
    """The same statistic vector on the OBSERVED data, from the cached fit.

    Spec Sec.5: never refit a cached latent fit for the observed data.
    ``centred=False`` gives the T13 ``centring`` sensitivity row (the spec's
    uncentred ``c_hat``).
    """
    if stat_fn is None:
        stat_fn = default_stat_fn
    rep = dict(y=ctx.y, censor_mask=ctx.censor_mask, phi_oof=ctx.phi_oof,
               z_oof=ctx.z_oof, e_oof=ctx.e_oof, sigma_oof=ctx.sigma_oof,
               mu_oof=(ctx.mu_oof if centred else np.zeros(ctx.n)),
               oof_finite=ctx.oof_finite, n_iter_used=0, wall_s=0.0, meta={})
    return stat_fn(ctx, rep)


# =========================================================================== #
# the parallel driver                                                         #
# =========================================================================== #

def ensemble_path(dms_id, null, B, seed_name=None):
    """``nulls/{id}_{null}_B{B}_seed{seed}.npz`` -- spec Sec.5's naming.
    **Statistic vectors only, never a surrogate ``y``.**"""
    if seed_name is None:
        seed_name = 'nulls_' + null
    return os.path.join(PATHS.nulls, '%s_%s_B%d_seed%d.npz'
                        % (dms_id, null, int(B), SEEDS[seed_name]))


_WORKER = {}


def _warn_threads(nproc):
    """Loudly refuse to fan out silently onto an over-subscribed box."""
    bad = [v for v in THREAD_ENV if os.environ.get(v) != '1']
    if nproc > 1 and bad:
        msg = ('nproc=%d with %s%s -- run the driver with %s so each worker '
               'stays single-threaded (measured: 64 workers x the default pool '
               '= load average 1,409 on this 80-core box, which stalled the run)'
               % (nproc, 'unset or != 1: ' + ','.join(bad),
                  ('; numpy was already imported before cliff.nulls, so the '
                   'in-module default could not bind' if _NUMPY_PREIMPORTED
                   else ''),
                  ' '.join('%s=1' % v for v in THREAD_ENV)))
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        print('[nulls][WARN] ' + msg)


def _worker_init(dms_id, null, verify):
    """Pool initialiser.  Idempotent: with the ``fork`` start method the parent
    has already built the context (and, for N2b, the ridge fit), so the child
    inherits both copy-on-write and this call is a lookup -- the ``is None``
    guard on the ridge fit is what keeps 64 workers from each repeating a
    5-fold x 12-lambda CV path."""
    _WORKER['dms_id'] = dms_id
    _WORKER['null'] = null
    ctx = get_context(dms_id, verify=verify)
    if null == 'N2b' and ctx.pairwise is None:
        fit_pairwise_ridge(ctx)
    _WORKER['ctx'] = ctx
    _WORKER.setdefault('stat_fn', None)
    _WORKER.setdefault('seed0', None)
    _WORKER.setdefault('kw', {})


def _worker_run(b):
    ctx = _WORKER['ctx']
    return replicate(ctx, _WORKER['null'], b, stat_fn=_WORKER.get('stat_fn'),
                     seed0=_WORKER.get('seed0'), **_WORKER.get('kw', {}))


def run_ensemble(dms_id, null, B=None, stat_fn=None, seed0=None, nproc=1, *,
                 use_cache=True, write=True, verify=False, verbose=False,
                 replicate_kw=None):
    """Spec Sec.3's parallel driver.  Returns the ``B x n_stat`` DataFrame.

    **Caches statistic vectors only** (spec Sec.5), so a re-run is free and a
    surrogate ``y`` never touches disk.  Workers are capped at
    ``THRESH['nproc_cap']``: 64, not 80, so ``64 x 0.5 GB`` stays inside the
    box's 111 GB (spec Sec.5).
    """
    import pandas as pd
    if B is None:
        B = THRESH['null_B']
    B = int(B)
    p = ensemble_path(dms_id, null, B)
    cacheable = (stat_fn is None and seed0 is None and not replicate_kw)
    if use_cache and cacheable and os.path.exists(p):
        with np.load(p, allow_pickle=False) as z:
            names = [str(s) for s in z['stat_names']]
            meta = json.loads(str(z['meta_json']))
            arr = z['stats']
        if meta.get('stat_version') == _stat_version(null):
            df = pd.DataFrame(arr, columns=names)
            df.attrs['from_cache'] = True
            df.attrs['meta'] = meta
            return df
        if verbose:
            print('    [cache] %s: stat_version %s != %s -- recomputing'
                  % (os.path.basename(p), meta.get('stat_version'),
                     _stat_version(null)))
    nproc = min(int(nproc), THRESH['nproc_cap'])
    _warn_threads(nproc)
    t0 = time.time()
    rows = []
    # build the context (and, for N2b, the ridge fit) in the PARENT: with the
    # fork start method every worker then inherits it copy-on-write instead of
    # paying for it nproc times.
    # ``stat_fn`` / ``seed0`` / the observation-model knobs reach the workers
    # through ``_WORKER`` rather than as pool arguments: with ``fork`` the
    # children inherit them, so a non-picklable ``stat_fn`` is fine.
    _WORKER['stat_fn'] = stat_fn
    _WORKER['seed0'] = list(seed0) if seed0 is not None else None
    _WORKER['kw'] = dict(replicate_kw or {})
    _worker_init(dms_id, null, verify)
    if nproc <= 1:
        for b in range(B):
            rows.append(_worker_run(b))
            if verbose and (b + 1) % 25 == 0:
                print('    [%s %s] %3d/%d  %.1fs'
                      % (dms_id, null, b + 1, B, time.time() - t0))
    else:
        import multiprocessing as mp
        cls = mp.get_context('fork')
        with cls.Pool(processes=nproc, initializer=_worker_init,
                      initargs=(dms_id, null, verify)) as pool:
            for i, r in enumerate(pool.imap(_worker_run, range(B),
                                            chunksize=1)):
                rows.append(r)
                if verbose and (i + 1) % 50 == 0:
                    print('    [%s %s] %3d/%d  %.1fs'
                          % (dms_id, null, i + 1, B, time.time() - t0))
    names = list(rows[0].keys()) if rows else list(STAT_NAMES)
    arr = np.array([[float(r.get(k, np.nan)) for k in names] for r in rows],
                   dtype=np.float64)
    df = pd.DataFrame(arr, columns=names)
    meta = dict(dms_id=dms_id, null=null, B=B, nproc=nproc,
                stat_version=_stat_version(null),
                link_mode='hull', clamp_mode='calibrated',
                seed=list(seed0) if seed0 is not None
                else config.assay_seed('nulls_' + null, dms_id),
                wall_s=round(time.time() - t0, 2),
                wall_per_rep_s=round((time.time() - t0) * nproc / max(B, 1), 3),
                core_s=round((time.time() - t0) * nproc, 1),
                centring='phi-centred (ORCHESTRATOR D2)',
                written_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    df.attrs['from_cache'] = False
    df.attrs['meta'] = meta
    if write and cacheable:
        PATHS.ensure_cache_dirs()
        tmp = p[:-4] + '.tmp.npz'
        np.savez(tmp, stats=arr, stat_names=np.array(names),
                 meta_json=np.array(json.dumps(meta, sort_keys=True,
                                               default=str)))
        os.replace(tmp, p)
    return df


def register_null_cache(extra=None):
    """md5 every ``nulls/*.npz`` into ``MANIFEST.json`` -- ONE call at the end of
    the run, through the now-``flock``-protected :func:`cliff.pairs.write_manifest`
    (D8), then :func:`cliff.pairs.verify_manifest`."""
    PATHS.ensure_cache_dirs()
    ents = []
    for f in sorted(os.listdir(PATHS.nulls)):
        if not f.endswith('.npz'):
            continue
        q = os.path.join(PATHS.nulls, f)
        ents.append(dict(path=os.path.relpath(q, config.REPO), md5=md5_of(q),
                         bytes=os.path.getsize(q)))
    if ents:
        _pairs.write_manifest(ents, extra=extra)
    return ents


# =========================================================================== #
# G4 -- null self-calibration (a STOP gate)                                   #
# =========================================================================== #

#: Statistics G4 scores.  The rate columns are what ``T(tau)`` is built from;
#: TR is added because C2's headline is a tail ratio.
def _g4_columns():
    cols = ['rate_%s_tau%g' % (u, t) for u in ('sigma', 'mad') for t in TAUS]
    return cols + ['TR1', 'TR2', 'TR1_mad', 'TR2_mad']


def _grid_guard_taus(ctx, unit):
    """Spec Sec.1.0: drop any ``tau`` whose ABSOLUTE cut is below ``3 q_a``.

    The absolute cut of ``tau`` in the ``sigma`` system is
    ``tau * median(sqrt(s2_u+s2_v))`` on the observed ``P_a``; in the ``mad``
    system it is ``tau * 1.4826 MAD(numerator)``.  Both are latent-scale
    quantities and ``q_a`` is a raw-scale one, so the guard is applied after
    mapping the cut through the link's local slope
    ``dy/dz ~ (max y - min y)/(hi - lo)``, the only assumption-free conversion
    available from the cached knots.
    """
    keep = _pa_mask(ctx, ctx.censor_mask, ctx.oof_finite)
    sub = ctx.nested_idx[keep]
    if sub.shape[0] == 0:
        return {t: True for t in TAUS}
    den = np.sqrt(ctx.sigma_oof[sub[:, 0]] ** 2 + ctx.sigma_oof[sub[:, 1]] ** 2)
    den = np.nanmedian(den)
    num = ((ctx.e_oof - ctx.mu_oof)[sub[:, 1]]
           - (ctx.e_oof - ctx.mu_oof)[sub[:, 0]])
    scale = den if unit == 'sigma' else mad_scaled(num[np.isfinite(num)])
    yk = ctx.g_knots[1]
    slope = ((float(yk.max() - yk.min()) / (ctx.hi - ctx.lo))
             if ctx.hi > ctx.lo else 1.0)
    return {t: bool(t * scale * slope >= THRESH['grid_guard_mult'] * ctx.quantum)
            for t in TAUS}


def _empirical_p(v, rng):
    """Leave-one-out empirical p-values, in the study's own form and in the two
    forms that isolate BIAS from DISCRETENESS.

    Returns ``(p_cons, p_mid, p_rand, frac_tied)`` where, with ``R_b`` the other
    ``m = B-1`` values,

    * ``p_cons = (1 + #{R_b >= v_b})/(m+1)`` -- spec Sec.1.3's formula verbatim,
      the one the observed data will actually be scored with;
    * ``p_mid  = (1 + #{R_b > v_b} + 0.5 #{R_b == v_b})/(m+1)`` -- the mid-p;
    * ``p_rand = (#{R_b > v_b} + U (1 + #{R_b == v_b}))/(m+1)``, ``U ~ U(0,1)``
      -- the randomised p-value, which is EXACTLY ``Uniform(0,1)`` under the
      null for a discrete statistic.

    Why this matters for G4: a rate statistic is a count over ``|P_a|`` edges
    divided by ``|P_a|``, so at ``tau = 5, 6, 8`` most replicates score exactly
    0 and ``p_cons`` collapses onto a point mass at 1.  Measured on
    Z-ZpA963_HL1: ``T(tau)`` sits at 1.0084 / 1.0622 (i.e. correct) at
    ``tau = 5 / 6`` while ``p_cons``'s KS p is 3e-33 / 7e-205.  That is
    discreteness, and it is in the CONSERVATIVE direction (``p_cons`` is
    stochastically >= uniform, so the study can only under-reject, never
    over-reject).  Reading it as "the surrogate machinery is biased" would halt
    the study for a tie structure, so the gate is read off ``p_rand`` and
    ``frac_tied`` is reported beside it.
    """
    v = np.asarray(v, dtype=np.float64)
    n = v.size
    order = np.argsort(v, kind='stable')
    sv = v[order]
    # per value: how many of ALL n are >, ==, < ; then remove self
    n_gt_all = n - np.searchsorted(sv, v, side='right')
    n_eq_all = (np.searchsorted(sv, v, side='right')
                - np.searchsorted(sv, v, side='left'))
    m = n - 1.0
    n_gt = n_gt_all.astype(np.float64)
    n_eq = (n_eq_all - 1).astype(np.float64)          # drop self from the ties
    p_cons = (1.0 + n_gt + n_eq) / (m + 1.0)
    p_mid = (1.0 + n_gt + 0.5 * n_eq) / (m + 1.0)
    u = rng.random(n)
    p_rand = (n_gt + u * (1.0 + n_eq)) / (m + 1.0)
    return p_cons, p_mid, p_rand, n_eq / m


def g4_selfcal(dms_id, B=None, *, nproc=1, verbose=True, use_cache=True):
    """G4 (spec Sec.1.1, a **STOP** gate): hold out 1 of ``B`` N1 surrogates and
    score it against the other ``B-1``.

    Must give ``T(tau) = 1.00 +/- 0.05`` for every ``tau`` and ``B`` UNIFORM
    empirical p-values (``KS p > 0.05``).  Every surrogate takes its turn as the
    held-out one, so ``T(tau)`` is the mean over ``B`` leave-one-out scorings
    ``T_b = P_b / mean_{b' != b} P_b'`` and the p-values are the leave-one-out
    empirical p-values of :func:`_empirical_p` -- exactly the estimator the
    observed data will be scored with.

    A statistic is **live** (i.e. it can carry the gate) iff it passes the grid
    guard, has a finite ``T``, and is not degenerate on the ensemble:
    ``frac_tied < 0.5``, i.e. the median replicate does NOT share its value with
    half the ensemble.  A statistic the ensemble cannot resolve carries no
    information about bias in either direction -- ``rate_sigma_tau8`` on
    Z-ZpA963_HL1 is 0 in all 200 replicates -- and is reported, with its
    numbers, rather than allowed to decide a STOP gate.
    """
    if B is None:
        B = THRESH['G4_n_surrogates']
    ctx = get_context(dms_id, verify=False)
    ens = run_ensemble(dms_id, 'N1', B, nproc=nproc, use_cache=use_cache,
                       verbose=verbose)
    cols = [c for c in _g4_columns() if c in ens.columns]
    guard = {u: _grid_guard_taus(ctx, u) for u in ('sigma', 'mad')}
    rng = np.random.default_rng(config.assay_seed('nulls_N1', dms_id) + [999])
    rows, pvals = [], {}
    for c in cols:
        v = ens[c].values.astype(np.float64)
        fin = np.isfinite(v)
        gg = True
        for u in ('sigma', 'mad'):
            for t in TAUS:
                if c == 'rate_%s_tau%g' % (u, t):
                    gg = guard[u][t]
        if fin.sum() < 3:
            rows.append(dict(dms_id=dms_id, stat=c, n_finite=int(fin.sum()),
                             n_distinct=0, frac_tied=1.0, ref_mean=np.nan,
                             T_mean=np.nan, T_median=np.nan, T_min=np.nan,
                             T_max=np.nan, T_sd=np.nan,
                             ks_p_rand=np.nan, ks_D_rand=np.nan,
                             ks_p_mid=np.nan, ks_p_cons=np.nan,
                             within_tol=False, grid_guard_pass=gg, live=False,
                             note='fewer than 3 finite replicates'))
            continue
        vv = v[fin]
        n = vv.size
        loo_mean = (vv.sum() - vv) / (n - 1.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            T = vv / loo_mean
        Tf = T[np.isfinite(T)]
        pc, pm, pr, ft = _empirical_p(vv, rng)
        ks_r = kstest(pr, 'uniform')
        ks_m = kstest(pm, 'uniform')
        ks_c = kstest(pc, 'uniform')
        frac_tied = float(ft.mean())
        n_distinct = int(np.unique(vv).size)
        note = ''
        if not np.isfinite(loo_mean).all() or (loo_mean == 0).any():
            note = ('reference rate is 0 in >= 1 leave-one-out set (tau beyond '
                    'the ensemble support)')
        elif frac_tied >= 0.5:
            note = ('%.0f%% of the ensemble ties the held-out value -- the '
                    'statistic is not resolvable at B=%d' % (100 * frac_tied, n))
        live = bool(gg and Tf.size and np.isfinite(Tf.mean())
                    and frac_tied < 0.5 and note == '')
        rows.append(dict(
            dms_id=dms_id, stat=c, n_finite=n, n_distinct=n_distinct,
            frac_tied=round(frac_tied, 6), ref_mean=float(vv.mean()),
            T_mean=float(Tf.mean()) if Tf.size else np.nan,
            T_median=float(np.median(Tf)) if Tf.size else np.nan,
            T_min=float(Tf.min()) if Tf.size else np.nan,
            T_max=float(Tf.max()) if Tf.size else np.nan,
            T_sd=float(Tf.std()) if Tf.size else np.nan,
            ks_p_rand=float(ks_r.pvalue), ks_D_rand=float(ks_r.statistic),
            ks_p_mid=float(ks_m.pvalue), ks_p_cons=float(ks_c.pvalue),
            within_tol=bool(Tf.size and abs(Tf.mean() - 1.0)
                            <= THRESH['G4_T_tol']),
            grid_guard_pass=gg, live=live, note=note))
        pvals[c] = dict(cons=pc, mid=pm, rand=pr)
    import pandas as pd
    tab = pd.DataFrame(rows)
    tab.attrs['pvals'] = pvals
    tab.attrs['B'] = int(B)
    tab.attrs['ensemble_meta'] = ens.attrs.get('meta', {})
    live = tab[tab['live']]
    tab.attrs['n_live'] = int(len(live))
    tab.attrs['n_stat'] = int(len(tab))
    tab.attrs['T_worst'] = (float(np.abs(live['T_mean'] - 1.0).max())
                            if len(live) else float('nan'))
    tab.attrs['T_worst_stat'] = (str(live.iloc[int(np.argmax(np.abs(
        live['T_mean'].values - 1.0)))]['stat']) if len(live) else '')
    tab.attrs['ks_p_min'] = (float(live['ks_p_rand'].min()) if len(live)
                             else float('nan'))
    tab.attrs['ks_p_min_stat'] = (str(live.iloc[int(np.argmin(
        live['ks_p_rand'].values))]['stat']) if len(live) else '')
    tab.attrs['ks_p_min_cons'] = (float(live['ks_p_cons'].min()) if len(live)
                                  else float('nan'))
    tab.attrs['pass'] = bool(len(live) and
                             tab.attrs['T_worst'] <= THRESH['G4_T_tol'] and
                             tab.attrs['ks_p_min'] > THRESH['G4_ks_p_min'])
    if verbose:
        print('[G4] %-40s live=%2d/%2d  worst|T-1|=%.4f (%s)  min KS p=%.4f '
              '(%s)  -> %s'
              % (dms_id, tab.attrs['n_live'], tab.attrs['n_stat'],
                 tab.attrs['T_worst'], tab.attrs['T_worst_stat'],
                 tab.attrs['ks_p_min'], tab.attrs['ks_p_min_stat'],
                 'PASS' if tab.attrs['pass'] else 'FAIL'))
    return tab


def g4_all(assays=None, B=None, *, nproc=1, verbose=True, use_cache=True):
    """G4 over the assay set; returns the concatenated per-statistic table."""
    import pandas as pd
    if assays is None:
        assays = G4_ASSAYS
    tabs, per = [], {}
    for d in assays:
        t = g4_selfcal(d, B=B, nproc=nproc, verbose=verbose,
                       use_cache=use_cache)
        per[d] = {k: v for k, v in t.attrs.items() if k != 'pvals'}
        per[d]['pvals'] = t.attrs.get('pvals', {})
        tabs.append(t)
    # pandas' concat compares the frames' ``attrs`` with ``==``, which raises on
    # an array-valued entry -- ``pvals`` holds B-length arrays.  Strip attrs off
    # the parts and carry them on the result instead.
    for t in tabs:
        t.attrs = {}
    out = pd.concat(tabs, ignore_index=True)
    out.attrs['per_assay'] = per
    return out


#: The four the orchestrator names explicitly, plus every other PRIMARY+ARM
#: assay when the caller asks for all of them.
G4_ASSAYS = ('GB1_IgG-Fc_fitness_1FCC', 'CR9114_FluAH1_logKd_4FQI',
             'SARS2-RBD_ACE2_deltaKd_6M0J', 'Z-domain_ZpA963_HL1_fitness_2M5A')


# =========================================================================== #
# N2c -- does the kurtosis target leave any room?                             #
# =========================================================================== #

def n2c_audit(dms_id, B=None, *, nproc=1, verbose=True):
    """Is there anything for N2c's scale mixture to ADD, and on which scale?

    Spec Sec.3 wants ``V`` "calibrated so the marginal residual kurtosis matches
    the observed marginal kurtosis EXACTLY", and :func:`two_point_scale_mixture`
    does that on the **injected** scale: the kurtosis of ``eps*`` itself, which
    is ``K_het (1 + Var V)``.  But the statistic that gets compared is the
    kurtosis of the REFIT residual ``e*_oof``, and the refit is not the identity
    -- cross-fitting adds estimation error, which dilutes a tail.  Two numbers
    are therefore needed, and this measures both:

    * ``K_het`` = ``3 E[sigma^4]/E[sigma^2]^2`` -- what the fitted
      level-dependent scale injects with NO mixture (``V == 1``, i.e. N1);
    * ``kurt_e_N1`` = the realised mean kurtosis of ``e*_oof`` over the N1
      ensemble.

    The dilution factor ``c = (kurt_e_N1 - 3)/(K_het - 3)`` then converts
    between them, and ``K_inj_needed = 3 + (K_obs - 3)/c`` is the injected
    kurtosis whose realised value would land on the observed ``K_obs``.
    ``room_realised`` is True iff a mixture is needed at all on the realised
    scale, i.e. ``kurt_e_N1 < K_obs``.  Feed ``K_inj_needed`` to
    ``surrogate_N2c(..., kurtosis_target=...)`` to calibrate on the realised
    scale.
    """
    ctx = get_context(dms_id, verify=False)
    row = dict(DMS_id=dms_id, tier=ctx.tier,
               K_obs=round(ctx.kurt['K_obs'], 4),
               K_het=round(ctx.kurt['K_het'], 4),
               sigma_dyn_range=round(ctx.kurt.get('sigma_dyn_range',
                                                  float('nan')), 2),
               sd_over_mad=round(ctx.kurt['sd_over_mad'], 4))
    mix = two_point_scale_mixture(ctx.kurt['K_obs'], ctx.kurt['K_het'])
    row['room_injected'] = not mix['degenerate']
    row['tau2_injected'] = (round(mix['tau2'], 5)
                            if np.isfinite(mix['tau2']) else '')
    row['p_mix'] = round(mix['p'], 5) if np.isfinite(mix['p']) else ''
    row['v_lo'] = round(mix['v_lo'], 4)
    row['v_hi'] = round(mix['v_hi'], 4)
    for nl in ('N1', 'N2c'):
        p = ensemble_path(dms_id, nl, B or THRESH['null_B'])
        if not os.path.exists(p):
            row['kurt_e_' + nl] = ''
            continue
        e = run_ensemble(dms_id, nl, B or THRESH['null_B'], nproc=nproc,
                         verbose=False)
        v = e['kurt_e'].values.astype(np.float64)
        v = v[np.isfinite(v)]
        row['kurt_e_' + nl] = round(float(v.mean()), 4) if v.size else ''
        row['kurt_e_%s_sd' % nl] = round(float(v.std()), 4) if v.size else ''
    k1 = row.get('kurt_e_N1', '')
    if k1 != '' and np.isfinite(ctx.kurt['K_het']) and ctx.kurt['K_het'] > 3:
        c = (float(k1) - 3.0) / (ctx.kurt['K_het'] - 3.0)
        row['dilution_c'] = round(c, 4)
        row['room_realised'] = bool(float(k1) < ctx.kurt['K_obs'])
        row['K_inj_needed'] = (round(3.0 + (ctx.kurt['K_obs'] - 3.0) / c, 3)
                               if c > 0 else '')
    else:
        row['dilution_c'] = ''
        row['room_realised'] = ''
        row['K_inj_needed'] = ''
    if verbose:
        print('[N2c] %-40s K_obs=%8.3f  K_het=%8.3f  kurt_e|N1=%8s  '
              'room(injected)=%-5s room(realised)=%-5s  K_inj_needed=%s'
              % (dms_id, row['K_obs'], row['K_het'], row.get('kurt_e_N1', ''),
                 row['room_injected'], row['room_realised'],
                 row['K_inj_needed']))
    return row


def n2c_audit_all(assays=None, B=None, *, nproc=1, verbose=True):
    import pandas as pd
    if assays is None:
        assays = tuple(sorted(set(config.PRIMARY_AND_ARM + config.CONTROL)))
    rows = []
    for d in assays:
        rows.append(n2c_audit(d, B=B, nproc=nproc, verbose=verbose))
        clear_context_cache()
    return pd.DataFrame(rows)


# =========================================================================== #
# the N2-power diagnostic                                                     #
# =========================================================================== #

def n2_power(dms_id, B=None, *, nproc=1, verbose=True, use_cache=True):
    """Does N2 have power on this assay?  MEASURED, never assumed.

    Spec Sec.3's declared limitation: "in singles+doubles libraries a double's
    marginal residual *is* ``eps_ij`` and the nested difference is also
    ``eps_ij``, so N2 coincides with the data and has **zero power** there."  A
    powerless null must never be allowed to read as "no signal", so three
    independent measurements are reported:

    1. **structural** -- ``frac_Pa_bg_saturated``: the fraction of ``P_a`` pairs
       whose BACKGROUND endpoint carries an essentially-zero residual
       (``|e| < 0.01 * 1.4826 MAD(e)``).  When the single scan is saturated this
       is ~1 and ``c_hat`` is a function of ONE free residual, which the
       within-(order x phi-decile) exchange cannot alter;
    2. **marginal** -- ``ks_obs_vs_N2``: the sup distance between the observed
       ``|c_hat|`` distribution and the pooled N2 replicates'.  0 means the null
       reproduces the data exactly, i.e. no marginal statistic can separate them;
    3. **inferential** -- ``z_TR`` and ``T_N2(tau)``: how far the observed
       statistic sits from its own N2 ensemble in ensemble sd units.  A
       point-mass N2 ensemble AT the observed value is the signature of zero
       power, and is reported as such rather than as ``p = 1``.
    """
    ctx = get_context(dms_id, verify=False)
    obs = observed_stats(ctx)
    ens = run_ensemble(dms_id, 'N2', B or THRESH['null_B'], nproc=nproc,
                       use_cache=use_cache, verbose=False)

    keep = _pa_mask(ctx, ctx.censor_mask, ctx.oof_finite)
    sub = ctx.nested_idx[keep]
    e = ctx.e_oof
    ms = mad_scaled(e[ctx.oof_finite & ~ctx.censor_mask])
    tol = 0.01 * ms
    bg_sat = float((np.abs(e[sub[:, 0]]) < tol).mean()) if sub.shape[0] else np.nan
    add_sat = float((np.abs(e[sub[:, 1]]) < tol).mean()) if sub.shape[0] else np.nan
    strata = make_strata(ctx.n_muts, ctx.phi_oof, censor_mask=ctx.censor_mask)
    _u, cnt = np.unique(strata, return_counts=True)
    frac_singleton = float((cnt == 1).sum() / max(ctx.n, 1))
    mad_by_order = {}
    for m in np.unique(ctx.n_muts):
        sel = (ctx.n_muts == m) & ctx.oof_finite & ~ctx.censor_mask
        if sel.sum() >= 8:
            mad_by_order[int(m)] = round(float(mad_scaled(e[sel])), 6)

    # KS between the observed |c| and one N2 replicate's |c| -- recomputed here
    # because the ensemble caches statistics, not the vectors (spec Sec.5)
    rng = np.random.default_rng(config.assay_seed('nulls_N2', dms_id) + [0])
    e_star = surrogate_N2(ctx, e, rng, strata)
    def _absc(ev):
        num = ((ev - ctx.mu_oof)[sub[:, 1]] - (ev - ctx.mu_oof)[sub[:, 0]])
        den = np.sqrt(ctx.sigma_oof[sub[:, 0]] ** 2
                      + ctx.sigma_oof[sub[:, 1]] ** 2)
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.abs(num / den)
        return np.sort(c[np.isfinite(c)])
    a, b = _absc(e), _absc(e_star)
    ks = float('nan')
    if a.size and b.size:
        grid = np.unique(np.concatenate([a[::max(1, a.size // 20000)],
                                         b[::max(1, b.size // 20000)]]))
        Fa = np.searchsorted(a, grid, side='right') / a.size
        Fb = np.searchsorted(b, grid, side='right') / b.size
        ks = float(np.abs(Fa - Fb).max())

    row = dict(DMS_id=dms_id, tier=ctx.tier, family_id=ctx.family_id,
               n=ctx.n, n_Pa=int(keep.sum()),
               max_order=int(ctx.n_muts.max()),
               orders=json.dumps(sorted(int(v) for v in np.unique(ctx.n_muts))),
               frac_Pa_bg_saturated=round(bg_sat, 6),
               frac_Pa_add_saturated=round(add_sat, 6),
               frac_variants_in_singleton_stratum=round(frac_singleton, 6),
               mad_e_by_order=json.dumps(mad_by_order),
               ks_obs_vs_N2=round(ks, 6) if np.isfinite(ks) else '')
    for c in ('TR1', 'TR2', 'rate_sigma_tau3', 'rate_sigma_tau4'):
        v = ens[c].values.astype(np.float64)
        v = v[np.isfinite(v)]
        o = obs.get(c, np.nan)
        row[c + '_obs'] = round(float(o), 6) if np.isfinite(o) else ''
        row[c + '_N2_mean'] = round(float(v.mean()), 6) if v.size else ''
        row[c + '_N2_sd'] = round(float(v.std()), 8) if v.size else ''
        row[c + '_z'] = (round(float((o - v.mean()) / v.std()), 4)
                         if v.size and v.std() > 0 and np.isfinite(o) else '')
        row['T_N2_' + c] = (round(float(o / v.mean()), 4)
                            if v.size and v.mean() > 0 and np.isfinite(o) else '')
    # the verdict rule, frozen here and applied identically to every assay
    zs = [row['TR1_z'], row['rate_sigma_tau3_z'], row['rate_sigma_tau4_z']]
    zs = [abs(float(v)) for v in zs if v != '']
    zmax = max(zs) if zs else float('nan')
    if not np.isfinite(ks):
        verdict = 'UNMEASURABLE'
    elif ks < 0.01 and (not zs or zmax < 1.0):
        verdict = 'POWERLESS'
    elif ks < 0.05:
        verdict = 'WEAK'
    else:
        verdict = 'POWERED'
    row['z_max'] = round(zmax, 4) if np.isfinite(zmax) else ''
    row['N2_power'] = verdict
    if verbose:
        print('[N2power] %-40s orders<=%2d  bg_sat=%.3f  KS=%.4f  |z|max=%s'
              '  -> %s' % (dms_id, row['max_order'], bg_sat, ks,
                           row['z_max'], verdict))
    return row


def n2_power_all(assays=None, B=None, *, nproc=1, verbose=True):
    import pandas as pd
    if assays is None:
        assays = tuple(sorted(set(config.PRIMARY_AND_ARM + config.CONTROL)))
    return pd.DataFrame([n2_power(d, B=B, nproc=nproc, verbose=verbose)
                         for d in assays])


# =========================================================================== #
# T02 -- the G4 rows only                                                     #
# =========================================================================== #

def _t02_read():
    import pandas as pd
    p = os.path.join(PATHS.artifacts, 'T02_gates.csv')
    if os.path.exists(p):
        return pd.read_csv(p, dtype=str, keep_default_na=False), p
    return pd.DataFrame(columns=_pairs.T02_COLUMNS), p


def write_T02_G4_rows(g4, *, write=True, verbose=True):
    """Replace the ``gate_id == 'G4'`` rows of ``T02_gates.csv`` and touch
    nothing else.

    Stage 0 wrote two PENDING G4 rows (spec Sec.6 keeps the table structurally
    complete from stage 0 on); this fills them in per assay.  Rows are built
    with :func:`cliff.pairs._g`, the T02 row helper that already exists, so the
    ``PASS/FAIL`` logic is the one every other gate is scored by, and the file
    is rewritten from the other gates' rows verbatim.
    """
    import pandas as pd
    old, p = _t02_read()
    per = g4.attrs.get('per_assay', {})
    STOP = ('STOP -- the surrogate machinery is biased; no observed number in '
            'the whole study is readable')
    rows = []
    for d in sorted(per):
        a = per[d]
        rows.append(_pairs._g(
            'G4', 'null self-calibration: T(tau) = 1.00 +/- 0.05 on a held-out '
                  'N1', d,
            'worst |mean_b T_b(tau) - 1| over the %d live statistics '
            '(rate at every tau in both unit systems + TR), B=%d N1 surrogates, '
            'leave-one-out' % (a.get('n_live', 0), a.get('B', 0)),
            0.0, round(float(a['T_worst']), 4) if np.isfinite(a['T_worst'])
            else '', THRESH['G4_T_tol'], STOP, 'YES', mode='eq'))
        rows.append(_pairs._g(
            'G4', 'null self-calibration: B empirical p-values uniform', d,
            'min KS p over the same live statistics (randomised leave-one-out '
            'empirical p; the spec\'s conservative form gives %s and is '
            'reported in T02a)'
            % (('%.3g' % a['ks_p_min_cons'])
               if np.isfinite(a.get('ks_p_min_cons', float('nan'))) else 'n/a'),
            THRESH['G4_ks_p_min'],
            round(float(a['ks_p_min']), 4) if np.isfinite(a['ks_p_min'])
            else '', '> %g' % THRESH['G4_ks_p_min'], STOP, 'YES', mode='ge'))
    # the study-level roll-up, on the same two lines stage 0 left PENDING
    worst = [per[d]['T_worst'] for d in per if np.isfinite(per[d]['T_worst'])]
    ksm = [per[d]['ks_p_min'] for d in per if np.isfinite(per[d]['ks_p_min'])]
    rows.insert(0, _pairs._g(
        'G4', 'null self-calibration: T(tau) = 1.00 +/- 0.05 on a held-out N1',
        'PRIMARY+ARM',
        'T(tau) over %d N1 surrogates, tolerance +/-%g; worst over the %d '
        'assays scored' % (max([per[d]['B'] for d in per] or [0]) - 1,
                           THRESH['G4_T_tol'], len(per)),
        1.0, round(1.0 + max(worst), 4) if worst else '', THRESH['G4_T_tol'],
        STOP, 'YES', mode='eq'))
    rows.insert(1, _pairs._g(
        'G4', 'null self-calibration: 200 empirical p-values uniform',
        'PRIMARY+ARM', 'KS p, worst over the %d assays scored' % len(per),
        THRESH['G4_ks_p_min'], round(min(ksm), 4) if ksm else '',
        '> %g' % THRESH['G4_ks_p_min'], STOP, 'YES', mode='ge'))
    new = pd.DataFrame(rows, columns=_pairs.T02_COLUMNS)
    # keep stage 0's ordering: the G4 block goes back exactly where it was, and
    # every other gate's row is carried through verbatim
    if len(old):
        gid = old['gate_id'].astype(str).str.strip().values
        pos = int(np.nonzero(gid == 'G4')[0][0]) if (gid == 'G4').any() \
            else len(old)
        before = old.iloc[:pos]
        before = before[before['gate_id'].astype(str).str.strip() != 'G4']
        after = old.iloc[pos:]
        after = after[after['gate_id'].astype(str).str.strip() != 'G4']
        out = pd.concat([before, new, after], ignore_index=True)
    else:
        out = new
    out = out[_pairs.T02_COLUMNS]
    if write:
        os.makedirs(PATHS.artifacts, exist_ok=True)
        out.to_csv(p, index=False)
        if verbose:
            print('[T02] wrote %s (%d rows; %d G4 rows, %d other gates '
                  'untouched)' % (p, len(out), len(new), len(out) - len(new)))
    return out


# =========================================================================== #
# timing / stage 3                                                            #
# =========================================================================== #

def timing_table(assays=None, nulls=NULLS, B=3, *, nproc=1, verbose=True):
    """Per-assay per-replicate wall clock for every null, and the projected
    stage-3 cost against spec Sec.5's ~9.4 core-hours / ~9 min at nproc 64."""
    import pandas as pd
    if assays is None:
        assays = tuple(sorted(set(config.PRIMARY_AND_ARM + config.CONTROL)))
    rows = []
    for d in assays:
        ctx = get_context(d, verify=False)
        pf = fit_pairwise_ridge(ctx) if 'N2b' in nulls else None
        r = dict(DMS_id=d, tier=ctx.tier, n=ctx.n, M=ctx.M,
                 n_nested=int(ctx.nested_idx.shape[0]),
                 n_Pa=int(_pa_mask(ctx, ctx.censor_mask,
                                   ctx.oof_finite).sum()),
                 n_Z_cols=(pf.n_cols if pf is not None else ''),
                 K_obs=round(ctx.kurt['K_obs'], 3),
                 K_het=round(ctx.kurt['K_het'], 3))
        for nl in nulls:
            t0 = time.time()
            for b in range(int(B)):
                replicate(ctx, nl, b)
            r['s_per_rep_' + nl] = round((time.time() - t0) / float(B), 3)
        rows.append(r)
        if verbose:
            print('[timing] %-40s ' % d + '  '.join(
                '%s=%.2fs' % (nl, r['s_per_rep_' + nl]) for nl in nulls))
    df = pd.DataFrame(rows)
    per = {nl: float(df['s_per_rep_' + nl].sum()) for nl in nulls}
    core_s = sum(per.values()) * THRESH['null_B']
    df.attrs['projection'] = dict(
        n_assays=len(assays), nulls=list(nulls), B=THRESH['null_B'],
        n_jobs=len(assays) * len(nulls) * THRESH['null_B'],
        core_hours=round(core_s / 3600.0, 2),
        wall_min_at_64=round(core_s / 60.0 / THRESH['nproc_cap'], 1),
        spec_core_hours=9.4, spec_wall_min=9.0,
        per_null_core_hours={k: round(v * THRESH['null_B'] / 3600.0, 3)
                             for k, v in per.items()})
    return df


def stage3(assays=None, nulls=STAGE3_NULLS, B=None, *, nproc=None,
           verbose=True, use_cache=True):
    """Spec Sec.5 stage 3: ``len(nulls) x B x len(assays)`` replicate-jobs.

    One assay at a time so each worker holds exactly one :class:`NullContext`
    (bounded RSS), and ONE ``register_null_cache`` call at the end (D8).
    """
    import pandas as pd
    config.assert_env()
    if assays is None:
        assays = tuple(sorted(set(config.PRIMARY_AND_ARM + config.CONTROL)))
    if B is None:
        B = THRESH['null_B']
    if nproc is None:
        nproc = THRESH['nproc_cap']
    t0 = time.time()
    rows = []
    for i, d in enumerate(assays):
        for nl in nulls:
            t1 = time.time()
            ens = run_ensemble(d, nl, B, nproc=nproc, use_cache=use_cache,
                               verbose=False)
            m = ens.attrs.get('meta', {})
            rows.append(dict(DMS_id=d, null=nl, B=int(B),
                             from_cache=bool(ens.attrs.get('from_cache')),
                             wall_s=round(time.time() - t1, 2),
                             s_per_rep=m.get('wall_per_rep_s', ''),
                             core_s=m.get('core_s', '')))
            if verbose:
                print('[stage3] %2d/%d %-40s %-4s B=%d  %.1fs%s'
                      % (i + 1, len(assays), d, nl, B, time.time() - t1,
                         '  (cached)' if ens.attrs.get('from_cache') else ''))
    ents = register_null_cache(extra=dict(nulls=dict(
        assays=list(assays), nulls=list(nulls), B=int(B), nproc=int(nproc),
        centring='phi-centred (ORCHESTRATOR D2)',
        stat_names=list(STAT_NAMES),
        wall_s=round(time.time() - t0, 1),
        written_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))))
    bad = _pairs.verify_manifest()
    if bad:
        raise RuntimeError('MANIFEST md5 mismatch after stage 3: %r' % (bad[:5],))
    if verbose:
        print('[stage3] %d ensembles, %d nulls/*.npz registered, MANIFEST '
              'verified clean, wall %.1fs'
              % (len(rows), len(ents), time.time() - t0))
    return pd.DataFrame(rows)


# =========================================================================== #
# self-check                                                                  #
# =========================================================================== #

def _selfcheck(argv=()):
    import pandas as pd
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 60)
    config.assert_env()
    print('[env] %s' % (config.assert_env(),))

    # ---- 1. algebraic invariants, no data ---------------------------------- #
    print('\n[1] ALGEBRAIC INVARIANTS')
    for K_obs, K_het in ((6.0, 3.0), (12.0, 4.2), (3.1, 3.0), (2.0, 3.0)):
        m = two_point_scale_mixture(K_obs, K_het)
        print('    two_point(K_obs=%5.2f, K_het=%4.2f): p=%.6f v_lo=%.4f '
              'v_hi=%.4f ratio=%.3f degenerate=%s'
              % (K_obs, K_het, m['p'], m['v_lo'], m['v_hi'], m['ratio'],
                 m['degenerate']))
        if not m['degenerate']:
            assert m['v_lo'] > 0
            # the min-ratio property, checked numerically against a p grid
            tau = math.sqrt(m['tau2'])
            best = m['ratio']
            for p in np.linspace(1e-4, 1.0 / (1.0 + m['tau2']) - 1e-6, 4000):
                u = math.sqrt(p / (1 - p))
                if 1 - tau * u <= 0:
                    continue
                r = (1 + tau / u) / (1 - tau * u)
                assert r >= best - 1e-9, (p, r, best)
    rng = np.random.default_rng(0)
    v = np.arange(20.0)
    st = np.repeat(np.arange(4), 5)
    pv = permute_within_strata(v, st, rng)
    for k in range(4):
        assert set(pv[st == k]) == set(v[st == k])
    print('    permute_within_strata: multiset preserved within every stratum '
          'OK')
    Z = np.array([[1., 2., 3.], [4., np.nan, 6.]])
    pz = permute_NS3(Z, np.random.default_rng(1))
    assert np.isnan(pz[1, 1]) and set(pz[0]) == {1., 2., 3.}
    assert set(pz[1][np.isfinite(pz[1])]) == {4., 6.}
    assert np.allclose(np.nanmean(pz, axis=1), np.nanmean(Z, axis=1))
    print('    permute_NS3: row-wise, NaN cells fixed, row means preserved OK')
    # NS1/NS2 on a synthetic table
    tb = dict(levy_class=np.array(['core'] * 6 + ['interior'] * 6),
              aa_class=np.array(['polar', 'apolar'] * 6),
              beta_hat_abs=np.linspace(0, 1, 12),
              depth_tertile=np.array([0, 1, 2] * 4),
              is_iface_5A=np.array([1, 0] * 6))
    l1 = permute_NS1(tb, np.random.default_rng(2))
    assert l1.sum() == tb['is_iface_5A'].sum()
    tb2 = dict(seq_separation=np.arange(30), rsa_iso=np.linspace(0, .6, 30),
               is_cliff_3sigma=np.array([1, 0, 0] * 10))
    l2 = permute_NS2(tb2, np.random.default_rng(3))
    assert l2.sum() == tb2['is_cliff_3sigma'].sum()
    print('    permute_NS1 / permute_NS2: label total preserved OK')

    # ---- 2. one live assay ------------------------------------------------- #
    d = 'Z-domain_ZpA963_HL1_fitness_2M5A'
    print('\n[2] LIVE CONTEXT + ONE REPLICATE OF EACH NULL  (%s)' % d)
    ctx = get_context(d, verify=True)
    print('    %r' % (ctx,))
    print('    kurtosis: K_obs=%.3f  K_het=%.3f  sd/MAD=%.3f  '
          'sigma dyn range=%.1fx'
          % (ctx.kurt['K_obs'], ctx.kurt['K_het'], ctx.kurt['sd_over_mad'],
             ctx.kurt['sigma_dyn_range']))
    obs = observed_stats(ctx)
    print('    observed: n_Pa=%d  TR1=%.4f  rate(3sigma)=%.5f  SI=%.6f'
          % (obs['n_Pa'], obs['TR1'], obs['rate_sigma_tau3'], obs['SI']))
    pf = fit_pairwise_ridge(ctx, verbose=True)
    for nl in NULLS:
        t0 = time.time()
        r = replicate(ctx, nl, 0)
        print('    %-4s  n_Pa=%6d  TR1=%7.4f  rate3=%.5f  SI=%.6f  %.2fs'
              % (nl, r['n_Pa'], r['TR1'], r['rate_sigma_tau3'], r['SI'],
                 time.time() - t0))

    # ---- 3. N1 reproduces the observation model ---------------------------- #
    print('\n[3] N1 PRESERVES WHAT IT CLAIMS TO PRESERVE')
    rng = np.random.default_rng(7)
    ys, meta = surrogate_N1(ctx, None, rng, clamp=None, quantum=None)
    print('    variant set / pair graph: unchanged by construction (y* only)')
    print('    grid: n unique y*=%d (obs %d); all on the %g grid: %s'
          % (np.unique(ys).size, np.unique(ctx.y).size, ctx.quantum,
             bool(np.allclose(ys, np.round(ys, ctx.modal_decimals)))
             if ctx.transform == 'none' else 'n/a (log10)'))
    print('    censored mass: surrogate %d vs observed %d'
          % (int(_censor_mask_of(ctx, ys).sum()), int(ctx.censor_mask.sum())))

    # ---- 3b. the two documented deviations, MEASURED ----------------------- #
    print('\n[3b] LINK / CLAMP AUDIT -- the two deviations from the spec\'s '
          'literal wording, measured')
    aud = [link_audit(dd, B=4) for dd in
           ('GB1_IgG-Fc_fitness_1FCC', 'SARS2-RBD_ACE2_deltaKd_6M0J',
            'CR9114_FluAH1_logKd_4FQI', 'CR9114_FluAH3_logKd_4FQY',
            'CR6261_FluAH1_logKd_3GBN', d)]
    print(pd.DataFrame(aud).to_string(index=False))

    # ---- 4. E[SI_N3] == 1 -------------------------------------------------- #
    print('\n[4] N3 IDENTITY  E[SI_N3] = 1  (spec Sec.3; GB1 1.0000 +- 0.0016)')
    for dd in ('GB1_IgG-Fc_fitness_1FCC', d):
        c2 = get_context(dd, verify=False)
        rg = np.random.default_rng(config.assay_seed('nulls_N3', dd))
        gmd = _gini_mean_difference(c2.y)
        vals = []
        for _ in range(60):
            yp = rg.permutation(c2.y)
            vals.append(np.abs(yp[c2.lag1_idx[:, 0]]
                               - yp[c2.lag1_idx[:, 1]]).mean() / gmd)
        v = np.asarray(vals)
        print('    %-40s SI_N3 = %.4f +- %.4f  (B=60)   observed SI = %.4f'
              % (dd, v.mean(), v.std(), np.abs(
                  c2.y[c2.lag1_idx[:, 0]] - c2.y[c2.lag1_idx[:, 1]]).mean()
                 / gmd))
        assert abs(v.mean() - 1.0) < 0.01, v.mean()
    print('    OK')
    return 0


def _main(argv):
    import pandas as pd
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 80)
    a = list(argv)
    if not a:
        return _selfcheck()
    nproc = THRESH['nproc_cap']
    B = THRESH['null_B']
    for i, t in enumerate(a):
        if t == '--nproc':
            nproc = int(a[i + 1])
        if t == '--B':
            B = int(a[i + 1])
    if '--timing' in a:
        df = timing_table(B=2)
        print(df.to_string(index=False))
        print(json.dumps(df.attrs['projection'], indent=1))
    if '--stage3' in a:
        print(stage3(B=B, nproc=nproc).to_string(index=False))
    if '--n2power' in a:
        t = n2_power_all(B=B, nproc=nproc)
        print(t.to_string(index=False))
        t.to_csv(os.path.join(PATHS.artifacts, 'T02b_N2_power.csv'), index=False)
    if '--g4' in a:
        which = (tuple(sorted(set(config.PRIMARY_AND_ARM + config.CONTROL)))
                 if '--all' in a else G4_ASSAYS)
        g4 = g4_all(which, B=B, nproc=nproc)
        print(g4.to_string(index=False))
        write_T02_G4_rows(g4)
        g4.to_csv(os.path.join(PATHS.artifacts, 'T02a_G4_selfcal.csv'),
                  index=False)
    return 0


if __name__ == '__main__':
    sys.exit(_main(tuple(sys.argv[1:])))
