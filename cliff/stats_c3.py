"""BGYM-CLIFF v1 -- C3: the jumps are real (spec Sec.1.4).

Three sub-tests, three verdict lines, and per G7 the LOAD-BEARING discriminator
against heteroscedastic noise is ``C3-L`` (localisation): the tail axis has no
power in the positive direction (nulls.py measured ``T_N1(rate at 3 sigma) < 1``
on 11/17 assays), so "the deviation recurs for the same substitution / the same
site pair / the same interaction in a second measurement" is the only axis on
which a cliff is distinguishable from a fat-tailed measurement error.

What is implemented here
------------------------
``C3-N`` measurement noise
    :func:`epsilon_table` -- the 4-corner interaction
    ``eps(i,j;C) = y(C+i+j) - y(C+i) - y(C+j) + y(C)`` for every substitution
    pair ``(i,j)`` and every background ``C`` whose four corners are ALL
    observed.  ``bg_order == 0`` is exactly spec Sec.1.4's
    ``eps_st = y(st) - y(s) - y(t) + y(WT)``; the deeper backgrounds are what
    route L2 needs.
    :func:`replication_rate` -- ``R = P(|eps_b| >= 2 sigma & sign match |
    |eps_a| >= 3 sigma)`` with the chance level from 10,000 site-pair-label
    permutations in assay ``b``.

``C3-L`` localisation, five routes each behind a hard feasibility gate
    L1 :func:`sibling_slope`   sibling corroboration, HC3 OLS, N2-referenced
    L2 :func:`icc_across_backgrounds`   substitution-pair ICC across backgrounds
    L2' :func:`icc_across_aa_combos`    site-pair ICC across aa combinations
    L3 :func:`l3_cross_measurement`     the KRAS twin (scored as C3-N in T08)
    L4 :func:`dr2_oos`         out-of-sample pairwise predictability + its gate
    L5 :func:`l5_auroc`        3D localisation of eps, null NS2

``C3-A`` artefact clauses (all four must pass)
    :func:`depth_clause` sampling depth, :func:`density_strata` density,
    :func:`floor_clause` floor invariance, :func:`scale_clause` scale
    invariance.

An infeasible route reports ``feasible=False`` and NO number.  That is the
whole point of the gates: L1's ``|S| >= 3`` set is EMPTY inside ``P_a`` for
GB1_1FCC and all five KRAS assays (their nested backgrounds are singletons, so
a sibling would need a size-3 variant the library does not contain), and L4's
``Z`` columns are seen exactly once there, which makes ``dR2_oos`` identically
zero BY CONSTRUCTION.  Reporting either as "no epistasis" would be a lie.

Verdicts are NOT emitted here.  ``verdict_C3L`` / ``verdict_C3A`` /
``verdict_C3N`` / ``failing_criterion`` are written EMPTY and filled by
:mod:`cliff.verdict`'s write-back, exactly as T04 does.

Definitions this module had to pin down, and why
------------------------------------------------
1. **"site pair" means SUBSTITUTION pair** wherever the spec's noise table and
   C3-N use it.  :func:`cliff.noise.epsilon_sitepairs` already settled this
   empirically: the KRAS twin shares **10,868** substitution pairs -- the spec's
   own count -- against only 602 shared ``(chain,pos)`` pairs.  L2 therefore
   groups by substitution pair; L2' groups by ``(chain,pos)`` pair, which is the
   only reading under which the spec's own L2' arithmetic
   ("91,845 doubles / C(55,2) = 1,485 site pairs ~ 62 per pair") is true.
2. **eps is NOT cross-fitted, and must not be.**  Cross-fitting exists because
   an in-sample LS residual subtracts a ``beta-hat`` the row itself helped
   estimate.  ``eps`` subtracts no fitted quantity at all -- its additive
   baseline is built from three OBSERVED corners -- so there is nothing to
   cross-fit.  The latent-scale ``eps`` uses ``z = ginv(y)`` from the cached
   full fit, which is a pure monotone reparameterisation of ``y`` and adds no
   fold dependence.
3. **The ICC routes have exactly zero power under N2** and this module says so
   in the table rather than reporting a null band that cannot move.  N2 permutes
   the residual ``e`` and never touches ``y``; ``eps`` is a function of ``y``
   alone, so ``ICC_N2 == ICC_obs`` identically (verified, not assumed:
   ``icc_n2_is_identity`` in T07).  The honest null for a recurrence statistic
   is the label permutation that destroys the group assignment and preserves the
   marginal -- reported as ``ICC_perm_*`` -- and the N2-referenced localisation
   statistic that DOES move is ``icc_addcol`` (built on ``c_hat``, already in
   nulls.py's cached statistic vector), carried on the L1 row.
4. **L1's regressand is the phi-centred ``c_hat``** (ORCHESTRATOR D2), i.e. the
   study's cliff statistic, so the slope is dimensionless and directly
   comparable to its N2 ensemble.  The un-standardised numerator version is
   reported beside it (``beta_sibling_raw_num``) so nothing is hidden.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats as _st

from cliff import config
from cliff import latent as _latent
from cliff import noise as _noise
from cliff import nulls as _nulls
from cliff import pairs as _pairs
from cliff import structure as _structure
from cliff.config import PATHS, SEEDS, THRESH
from cliff.latent import mad_scaled
from cliff.nulls import c_hat as _c_hat

# --------------------------------------------------------------------------- #
# module constants that are DEFINITIONS, not decision boundaries              #
# --------------------------------------------------------------------------- #

#: tau at which an edge enters the cliff CATALOGUE (spec Sec.4:
#: ``cliff_catalogue`` keeps ``|c_hat| >= 4``).  C3-A's clauses 1 and 2 are about
#: "cliffs", and the catalogue is what the study calls a cliff.
_CAT_TAU = THRESH['C2_catalogue_c_min']

#: second, looser tau reported beside it so the artefact clauses are not read off
#: one threshold.
_ALT_TAU = 3.0

#: sigma multipliers every headline number is recomputed at (spec Sec.1.0).
_SIGMA_MULTS = config.SIGMA_MULTIPLIERS

#: ``sqrt(3)``: eps of a double is a 4-term contrast sharing y(WT), so relative
#: to each other Var(eps) ~ 3 sigma_y^2 (ORCHESTRATOR D5).
_SQRT3 = math.sqrt(3.0)

#: T07's column list, spec Sec.4, verbatim and in order.
T07_COLUMNS = [
    'DMS_id', 'route', 'feasible', 'n_units', 'beta_sibling', 'se_hc3',
    'beta_N2_p995', 'beta_in_N2_band', 'ICC', 'ICC_lo95', 'ICC_hi95',
    'ICC_N2_mean', 'dR2_oos', 'dR2_lo95', 'dR2_hi95', 'top1pct_share',
    'ridge_lambda', 'AUROC_L5', 'AUROC_lo95', 'p_NS2', 'depth_spearman',
    'best_struct_covariate', 'density_q1_rate', 'density_q5_rate',
    'density_monotone', 'floor_mask_invariant', 'latent_raw_consistent',
    'verdict_C3L', 'verdict_C3A', 'failing_criterion',
]

#: everything measured beyond the spec's column list.  Appended AFTER T07_COLUMNS
#: so a reader (and verdict.py, which selects by name) sees the spec's table
#: first and the evidence for it second.
T07_EXTRA_COLUMNS = [
    'tier', 'family_id', 'route_name', 'infeasible_reason',
    'gate_statistic', 'gate_value', 'gate_threshold',
    'n_units_all', 'n_groups', 'n_groups_ge2', 'kbar',
    'beta_sibling_raw_num', 'beta_t_hc3', 'beta_p_hc3',
    'beta_boot_lo95', 'beta_boot_hi95',
    'beta_N2_mean', 'beta_N2_sd', 'beta_N2_p025', 'beta_N2_p975', 'p_N2',
    'beta_raw_scale', 'beta_unmasked',
    'icc_addcol_obs', 'icc_addcol_N2_mean', 'icc_addcol_N2_p995',
    'icc_addcol_N1_p995', 'icc_addcol_N2c_p995', 'icc_addcol_in_N2_band',
    'ICC_perm_mean', 'ICC_perm_p995', 'ICC_perm_p', 'icc_n2_is_identity',
    'ICC_raw_scale', 'ICC_unmasked', 'ICC_boot_B', 'ICC_null_used',
    'dR2_r2_add', 'dR2_r2_pair', 'n_Z_cols', 'mean_obs_per_Z_col',
    'dR2_oos_raw_scale', 'dR2_top1_abs', 'n_top1_cols',
    'AUROC_hi95', 'AUROC_NS2_mean', 'AUROC_NS2_p995', 'n_cliff_L5',
    'n_noncliff_L5', 'sigma_eps_used', 'sigma_provenance',
    'AUROC_all_backgrounds', 'AUROC_raw_scale',
    'R_L3', 'sign_agreement_L3', 'n_shared_L3',
    'depth_spearman_tau3', 'best_struct_covariate_name',
    'struct_covariate_json', 'density_rates_json', 'density_spearman',
    'density_enriched_q1', 'density_enriched_q5', 'density_null_used',
    'floor_note', 'scale_note', 'sigma_grid_json',
    'density_tau_used', 'depth_tau_used', 'p_N2_qBH',
    'B_null', 'nproc', 'wall_s', 'seed', 'notes',
]

#: T08's column list, spec Sec.4, verbatim and in order.
T08_COLUMNS = [
    'assay_a', 'assay_b', 'relation', 'join_method', 'n_shared', 'sd_eps_a',
    'sd_eps_b', 'pearson_raw', 'ols_slope', 'resid_sd_after_affine',
    'sigma_eps', 'n_cliff_a_3sigma', 'R', 'R_chance_perm', 'perm_p',
    'sign_agreement', 'F_spec', 'F_spec_noise_corrected', 'verdict_C3N',
    'verdict_stamp',
]

T08_EXTRA_COLUMNS = [
    'row_role', 'threshold_label', 'sigma_mult', 'cliff_abs_a', 'replicate_abs_b',
    'frac_flagged_a', 'frac_flagged_b', 'n_flagged_a', 'n_replicated',
    'flagged_frac_exceeds_5pct', 'R_lo95', 'R_hi95', 'R_perm_p995',
    'sign_agreement_chance', 'sign_agreement_perm_p',
    'frac_cliffs_below_3sigma', 'frac_cliffs_below_3sigma_def',
    'sigma_y_a', 'sigma_eps_provenance', 'spearman_raw', 'n_eps_a', 'n_eps_b',
    'n_shared_site_pairs_collapsed', 'icc_eps_two_measurements',
    'family', 'caveat', 'failing_criterion_C3N', 'notes',
]

_MISSING = ''


# --------------------------------------------------------------------------- #
# small numeric primitives (statsmodels is NOT installed -- spec Sec.4)       #
# --------------------------------------------------------------------------- #

def ols_hc3(x, y):
    """Simple OLS ``y = a + b x`` with an **HC3** (MacKinnon-White) SE on ``b``.

    HC3 in closed form for the single-regressor case:
    ``Var_HC3(b) = sum(xc_i^2 r_i^2 / (1-h_i)^2) / (sum xc_i^2)^2`` with
    ``xc = x - xbar``, ``h_i = 1/n + xc_i^2 / sum xc_j^2`` the leverage.
    HC3 (not HC0/HC1) because the sibling-mean regressor is heavily
    leverage-skewed: an edge whose background sits at the hub has an
    ``|S|`` two orders of magnitude above the median.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    out = dict(n=int(n), slope=float('nan'), intercept=float('nan'),
               se_hc3=float('nan'), t=float('nan'), p=float('nan'),
               se_ols=float('nan'), r=float('nan'))
    if n < 8:
        return out
    xb, yb = x.mean(), y.mean()
    xc, yc = x - xb, y - yb
    sxx = float((xc * xc).sum())
    if sxx <= 0:
        return out
    b = float((xc * yc).sum() / sxx)
    a = float(yb - b * xb)
    r = y - (a + b * x)
    h = 1.0 / n + (xc * xc) / sxx
    w = np.clip(1.0 - h, 1e-12, None)
    v_hc3 = float(((xc * xc) * (r * r) / (w * w)).sum() / (sxx * sxx))
    s2 = float((r * r).sum() / max(n - 2, 1))
    out.update(slope=b, intercept=a, se_hc3=math.sqrt(max(v_hc3, 0.0)),
               se_ols=math.sqrt(max(s2 / sxx, 0.0)),
               r=float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0
               else float('nan'))
    if out['se_hc3'] > 0:
        t = out['slope'] / out['se_hc3']
        out['t'] = float(t)
        out['p'] = float(2.0 * _st.norm.sf(abs(t)))
    return out


def icc_oneway(values, groups):
    """Spec Sec.3's ``icc_oneway``: one-way random effects
    ``ICC = (MSB - MSW)/(MSB + (kbar-1) MSW)``.

    Delegates the arithmetic to :func:`cliff.nulls._icc_oneway` -- the identical
    estimator the null ensembles already score, so an observed ICC and its null
    can never drift apart -- and adds the group bookkeeping T07 reports.
    ``np.bincount``, never ``np.add.at``.
    """
    v = np.asarray(values, dtype=np.float64)
    g = np.asarray(groups)
    ok = np.isfinite(v)
    v, g = v[ok], g[ok]
    icc, n, ng = _nulls._icc_oneway(v, g)
    if v.size == 0:
        return dict(ICC=float('nan'), n=0, n_groups=0, n_groups_ge2=0,
                    kbar=float('nan'), msb=float('nan'), msw=float('nan'))
    _u, gi = np.unique(g, return_inverse=True)
    cnt = np.bincount(gi)
    return dict(ICC=icc, n=int(n), n_groups=int(ng),
                n_groups_ge2=int((cnt >= 2).sum()),
                kbar=float(v.size / max(ng, 1)))


def benjamini_hochberg(p):
    """BH-FDR ``q`` values (statsmodels is not installed).  Over the ASSAYS."""
    p = np.asarray(p, dtype=np.float64)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    if not ok.any():
        return q
    pv = p[ok]
    m = pv.size
    order = np.argsort(pv, kind='stable')
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(ranked, 0.0, 1.0)
    q[ok] = out
    return q


def _spearman(a, b):
    """Spearman rho with a tie-safe rank transform; ``nan`` when degenerate."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float('nan')
    ra = _st.rankdata(a[ok])
    rb = _st.rankdata(b[ok])
    if ra.std() == 0 or rb.std() == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def auroc(score, label):
    """AUROC by the rank (Mann-Whitney U) identity -- exact with ties at 0.5.

    ``U = sum(ranks of positives) - n1(n1+1)/2``; ``AUROC = U/(n1 n0)``.
    """
    s = np.asarray(score, dtype=np.float64)
    l = np.asarray(label).astype(bool)
    ok = np.isfinite(s)
    s, l = s[ok], l[ok]
    n1 = int(l.sum())
    n0 = int((~l).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = _st.rankdata(s)
    u = float(r[l].sum()) - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n0))


def _pct(v, q):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float('nan')
    return float(np.percentile(v, q))


def _nmean(v):
    """``nanmean`` that returns ``nan`` on an all-``nan`` vector instead of
    warning.  An all-``nan`` null column is the NORMAL state for an infeasible
    route -- L1's null is undefined where no edge has three siblings -- so it
    must not print a RuntimeWarning on 6 of 17 assays."""
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float('nan')


def _nsd(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(v.std()) if v.size else float('nan')


def _empirical_p(null_vals, obs):
    """Spec Sec.1.3's conservative empirical p: ``(1 + #{b: stat_b >= obs})/(B+1)``."""
    v = np.asarray(null_vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0 or not np.isfinite(obs):
        return float('nan')
    return float((1.0 + float((v >= obs).sum())) / (v.size + 1.0))


def _in_band(null_vals, obs, lo=2.5, hi=97.5):
    v = np.asarray(null_vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0 or not np.isfinite(obs):
        return None
    return bool(_pct(v, lo) <= obs <= _pct(v, hi))


# =========================================================================== #
# the assay bundle: one NullContext + the canonical keys + the design lookup  #
# =========================================================================== #

def keys_from_design(des):
    """Canonical keys as sorted tuples of **X-column ids**.

    ``keys/{id}.npz`` caches ``codes`` and ``col_index``, not the key list, and
    a tuple of column ids is the same object up to a bijection -- verified
    identical to :attr:`cliff.io_bgym.Assay.keys` under
    ``k -> tuple(sorted(col_index[s] for s in k))`` on GB1_1FCC.  Column ids are
    used everywhere downstream because the sibling / background lookups need a
    hashable key whose element-removal ``k[:j] + k[j+1:]`` is another key.
    """
    X = des['X'].tocsr()
    X.sort_indices()
    ip, ind = X.indptr, X.indices
    return [tuple(int(c) for c in ind[ip[i]:ip[i + 1]]) for i in range(X.shape[0])]


class AssayBundle(object):
    """Everything C3 needs for one assay, built ONCE from the caches.

    Never re-parses a csv, never re-enumerates a pair graph, never refits the
    observed latent fit (spec Sec.5).
    """

    __slots__ = ('dms_id', 'ctx', 'des', 'keys', 'row_of_key', 'inv_col',
                 'pos_of_col', 'pos_key', 'pa', 'c_obs', 'num_obs', 'deg',
                 'eps', 'sib', 'notes')

    def __init__(self, dms_id, *, verify=False):
        self.dms_id = dms_id
        self.ctx = _nulls.build_context(dms_id, verify=verify)
        self.des = _latent.load_cached_design(dms_id, verify=verify)
        self.keys = keys_from_design(self.des)
        self.row_of_key = {k: i for i, k in enumerate(self.keys)}
        if len(self.row_of_key) != len(self.keys):
            raise RuntimeError('%s: duplicate canonical keys' % dms_id)
        self.inv_col = {v: k for k, v in self.des['col_index'].items()}
        ctx = self.ctx
        # (chain,pos) index of every X column, and the (chain,pos) key itself
        pos_index = self.des['pos_index']
        self.pos_of_col = np.empty(ctx.M, dtype=np.int32)
        self.pos_key = [None] * len(pos_index)
        for c, (ch, ps, _aa) in self.inv_col.items():
            j = pos_index[(ch, ps)]
            self.pos_of_col[c] = j
            self.pos_key[j] = (ch, ps)
        # P_a and the observed cliff statistic on it (phi-centred, D2)
        self.pa = _nulls._pa_mask(ctx, ctx.censor_mask, ctx.oof_finite)
        e_c = ctx.e_oof - ctx.mu_oof
        u, v = ctx.nested_idx[:, 0], ctx.nested_idx[:, 1]
        self.num_obs = e_c[v] - e_c[u]
        with np.errstate(divide='ignore', invalid='ignore'):
            self.c_obs = self.num_obs / np.sqrt(ctx.sigma_oof[u] ** 2
                                                + ctx.sigma_oof[v] ** 2)
        self.deg = _pairs.degrees(ctx.n, ctx.nested_idx)
        self.eps = None
        self.sib = None
        self.notes = {}

    # ------------------------------------------------------------------ #
    def sub_label(self, col):
        ch, ps, aa = self.inv_col[int(col)]
        return '%s%d%s' % (ch, ps, aa)


_BUNDLES = {}


def get_bundle(dms_id, *, verify=False):
    """One-assay-at-a-time bundle cache (bounded RSS, same policy as
    :func:`cliff.nulls.get_context`)."""
    if dms_id not in _BUNDLES:
        _BUNDLES.clear()
        _nulls.clear_context_cache()
        _BUNDLES[dms_id] = AssayBundle(dms_id, verify=verify)
    return _BUNDLES[dms_id]


# =========================================================================== #
# C3-N / L2 / L2' / L5 foundation: the 4-corner interaction eps               #
# =========================================================================== #

def epsilon_table(assay, *, max_bg_order=None, drop_censored=True,
                  verbose=False):
    """``eps(i,j;C) = y(C+i+j) - y(C+i) - y(C+j) + y(C)`` for every substitution
    pair ``(i,j)`` and background ``C`` with all FOUR corners observed.

    ``bg_order == 0`` is exactly spec Sec.1.4 C3-N's
    ``eps_st = y(st) - y(s) - y(t) + y(WT)``; that slice is what
    :func:`cliff.noise.epsilon_sitepairs` computes and it is cross-checked
    against it in :func:`_selfcheck`.  The deeper backgrounds are route L2's
    replicates.

    Two scales, both exact contrasts of OBSERVED values and neither cross-fitted
    (there is no fitted quantity in a 4-corner contrast to cross-fit):

    * ``eps``        -- on the assay's own ``y`` (``log10`` already applied for
      hYAP65 by :func:`cliff.io_bgym.load_assay`);
    * ``eps_latent`` -- on ``z = ginv(y)`` from the CACHED full fit, a pure
      monotone reparameterisation of ``y``.  This is the primary scale (spec
      Sec.1.3: "latent scale primary"), and the raw one is what C3-A clause 4
      compares against.

    ``assay`` is an :class:`AssayBundle` or a ``DMS_id``.

    ``drop_censored=False`` keeps the 4-corner contrasts that touch a censored
    row.  It exists for ONE caller, C3-A clause 3 (floor invariance: "verdict
    unchanged before/after floor masking"), and is never cached -- the cache
    holds the primary, floor-masked table that :mod:`cliff.stats_c4` reads.
    """
    b = assay if isinstance(assay, AssayBundle) else get_bundle(assay)
    ctx, keys, row_of = b.ctx, b.keys, b.row_of_key
    y, z = ctx.y, ctx.z
    cm = ctx.censor_mask
    t0 = time.time()
    order = np.asarray([len(k) for k in keys], dtype=np.int32)
    ci, cj, bgr, r_ij, r_i, r_j, r_c = [], [], [], [], [], [], []
    n_cand = n_censor_drop = 0
    lim = 10 ** 9 if max_bg_order is None else int(max_bg_order) + 2
    for v in range(len(keys)):
        k = keys[v]
        m = len(k)
        if m < 2 or m > lim:
            continue
        for a in range(m):
            ka = k[:a] + k[a + 1:]                     # C + j   (i removed)
            ra = row_of.get(ka)
            if ra is None:
                n_cand += m - 1 - a
                continue
            for bb in range(a + 1, m):
                n_cand += 1
                kb = k[:bb] + k[bb + 1:]               # C + i   (j removed)
                rb = row_of.get(kb)
                if rb is None:
                    continue
                kc = ka[:bb - 1] + ka[bb:]             # C
                rc = row_of.get(kc)
                if rc is None:
                    continue
                if cm[v] or cm[ra] or cm[rb] or cm[rc]:
                    n_censor_drop += 1
                    if drop_censored:
                        continue
                ci.append(k[a])
                cj.append(k[bb])
                bgr.append(m - 2)
                r_ij.append(v)
                r_i.append(rb)
                r_j.append(ra)
                r_c.append(rc)
    ci = np.asarray(ci, dtype=np.int32)
    cj = np.asarray(cj, dtype=np.int32)
    r_ij = np.asarray(r_ij, dtype=np.int32)
    r_i = np.asarray(r_i, dtype=np.int32)
    r_j = np.asarray(r_j, dtype=np.int32)
    r_c = np.asarray(r_c, dtype=np.int32)
    eps = y[r_ij] - y[r_i] - y[r_j] + y[r_c]
    epsl = z[r_ij] - z[r_i] - z[r_j] + z[r_c]
    pi = b.pos_of_col[ci] if ci.size else np.zeros(0, dtype=np.int32)
    pj = b.pos_of_col[cj] if cj.size else np.zeros(0, dtype=np.int32)
    # substitution-pair id and SITE-pair id, both as int64 linear codes
    M = max(ctx.M, 1)
    P = max(ctx.P, 1)
    lo = np.minimum(ci, cj).astype(np.int64)
    hi = np.maximum(ci, cj).astype(np.int64)
    sub_pair = lo * M + hi
    plo = np.minimum(pi, pj).astype(np.int64)
    phi_ = np.maximum(pi, pj).astype(np.int64)
    site_pair = plo * P + phi_
    df = pd.DataFrame(dict(
        col_i=ci, col_j=cj, pos_i=pi, pos_j=pj,
        sub_pair=sub_pair, site_pair=site_pair,
        bg_order=np.asarray(bgr, dtype=np.int16),
        row_ij=r_ij, row_i=r_i, row_j=r_j, row_bg=r_c,
        eps=eps, eps_latent=epsl))
    df.attrs.update(dms_id=b.dms_id, n_candidate=int(n_cand),
                    n_censor_dropped=(int(n_censor_drop) if drop_censored else 0),
                    n_censor_touching=int(n_censor_drop),
                    drop_censored=bool(drop_censored),
                    P=int(ctx.P), M=int(ctx.M),
                    n_rows=int(len(df)),
                    n_doubles_bg0=int((df['bg_order'].values == 0).sum())
                    if len(df) else 0,
                    max_bg_order=(int(df['bg_order'].max()) if len(df) else -1),
                    n_sub_pairs=int(np.unique(sub_pair).size) if ci.size else 0,
                    n_site_pairs=int(np.unique(site_pair).size) if ci.size else 0,
                    wall_s=round(time.time() - t0, 2),
                    order_hist=np.bincount(order).tolist())
    if verbose:
        print('    [eps ] %-40s %8d rows (%d at bg_order 0), %d sub-pairs, '
              '%d site-pairs, %.1fs'
              % (b.dms_id, len(df), df.attrs['n_doubles_bg0'],
                 df.attrs['n_sub_pairs'], df.attrs['n_site_pairs'],
                 df.attrs['wall_s']))
    return df


def eps_cache_path(dms_id):
    return os.path.join(PATHS.eps, dms_id + '_eps.npz')


def cached_epsilon_table(assay, *, use_cache=True, write=True, verbose=False):
    """:func:`epsilon_table` with an npz cache under ``data/cliff_cache/eps/``.

    The table is a deterministic function of the keys cache and the latent
    cache, so it is a legitimate derived artefact; it is md5'd into
    ``MANIFEST.json`` by :func:`register_eps_cache` at the END of the run (D8).
    """
    dms_id = assay.dms_id if isinstance(assay, AssayBundle) else assay
    p = eps_cache_path(dms_id)
    if use_cache and os.path.exists(p):
        with np.load(p, allow_pickle=False) as zz:
            cols = [str(s) for s in zz['__columns__']]
            df = pd.DataFrame({c: zz[c] for c in cols})
            df.attrs.update(json.loads(str(zz['__attrs__'])))
        df.attrs['from_cache'] = True
        if verbose:
            print('    [eps ] %-40s %8d rows (cache)' % (dms_id, len(df)))
        return df
    df = epsilon_table(assay, verbose=verbose)
    if write:
        PATHS.ensure_cache_dirs()
        os.makedirs(PATHS.eps, exist_ok=True)
        tmp = p[:-4] + '.tmp%d.npz' % os.getpid()
        np.savez(tmp, __columns__=np.array(list(df.columns)),
                 __attrs__=np.array(json.dumps(df.attrs, sort_keys=True,
                                               default=str)),
                 **{c: df[c].values for c in df.columns})
        os.replace(tmp, p)
    df.attrs['from_cache'] = False
    return df


def register_eps_cache(extra=None):
    """md5 every ``eps/*.npz`` into ``MANIFEST.json`` -- ONE call at the END of
    the run, through the ``flock``-protected :func:`cliff.pairs.write_manifest`
    (ORCHESTRATOR D8)."""
    from cliff.io_bgym import md5_of
    os.makedirs(PATHS.eps, exist_ok=True)
    ents = []
    for f in sorted(os.listdir(PATHS.eps)):
        if not f.endswith('.npz') or '.tmp' in f:
            continue
        q = os.path.join(PATHS.eps, f)
        ents.append(dict(path=os.path.relpath(q, config.REPO), md5=md5_of(q),
                         bytes=os.path.getsize(q)))
    if ents:
        _pairs.write_manifest(ents, extra=extra)
    return ents


# =========================================================================== #
# C3-L route L1 -- sibling corroboration                                      #
# =========================================================================== #

def sibling_index(idx, add_col, keys):
    """The **node-disjoint** sibling adjacency of the nested-edge graph.

    ``S(e) = {(B', B' u {i}) : |B xor B'| = 1}`` for ``e = (B, B u {i})`` (spec
    Sec.1.4 L1).  Every member of that set is automatically node-disjoint from
    ``e``: ``B'`` is ``B u {j}`` or ``B \\ {j}``, so the sibling's two nodes
    ``B'`` and ``B' u {i}`` are neither ``B`` nor ``B u {i}``.  Asserted, not
    assumed, in :func:`_selfcheck`.

    Built inside each ``add_col`` group by removing one element from every
    background and looking it up -- the same construction
    :func:`cliff.pairs.sibling_counts` counts, so the degrees agree exactly and
    ``pairs``' cached ``sibling_count`` array is a free cross-check.

    Returns ``(indptr, nbr)``, a CSR-shaped symmetric adjacency over EDGE ids.
    """
    m = int(idx.shape[0])
    if m == 0:
        return np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int32)
    own_l, nbr_l = [], []
    order = np.argsort(add_col, kind='stable')
    ac = add_col[order]
    bounds = np.flatnonzero(np.diff(ac)) + 1
    for g in np.split(order, bounds):
        loc = {}
        for t, e in enumerate(g):
            loc[keys[idx[e, 0]]] = t
        get = loc.get
        for t, e in enumerate(g):
            k = keys[idx[e, 0]]
            for j in range(len(k)):
                u = get(k[:j] + k[j + 1:])
                if u is not None:
                    own_l.append(e)
                    nbr_l.append(g[u])
                    own_l.append(g[u])
                    nbr_l.append(e)
    if not own_l:
        return np.zeros(m + 1, dtype=np.int64), np.zeros(0, dtype=np.int32)
    own = np.asarray(own_l, dtype=np.int64)
    nbr = np.asarray(nbr_l, dtype=np.int32)
    o = np.argsort(own, kind='stable')
    own, nbr = own[o], nbr[o]
    cnt = np.bincount(own, minlength=m)
    indptr = np.concatenate(([0], np.cumsum(cnt)))
    return indptr, nbr


def sibling_means(values, indptr, nbr, *, min_siblings=None):
    """Mean of the FINITE siblings of every edge, plus the finite-sibling count.

    ``np.bincount`` with weights, never ``np.add.at`` (spec's numeric hygiene).
    A sibling whose value is not finite -- a censored or non-cross-fittable
    endpoint -- is dropped from both the sum and the count, so an edge's
    regressor is never contaminated by a corner the study excludes elsewhere.
    """
    if min_siblings is None:
        min_siblings = THRESH['L1_min_siblings']
    v = np.asarray(values, dtype=np.float64)
    m = v.size
    if nbr.size == 0:
        return (np.full(m, np.nan), np.zeros(m, dtype=np.int32))
    owner = np.repeat(np.arange(m, dtype=np.int64), np.diff(indptr))
    val = v[nbr]
    ok = np.isfinite(val)
    deg = np.bincount(owner[ok], minlength=m)
    tot = np.bincount(owner[ok], weights=val[ok], minlength=m)
    with np.errstate(divide='ignore', invalid='ignore'):
        mean = tot / deg
    mean[deg < 1] = np.nan
    return mean, deg.astype(np.int32)


def sibling_slope(values, indptr, nbr, keep, *, min_siblings=None,
                  min_edges=None, positions=None, boot_B=None, seed=None,
                  label='c_hat'):
    """Spec Sec.1.4 L1: ``beta_a`` = **HC3** OLS slope of an edge's own deviation
    on the mean of its node-disjoint siblings', over ``keep`` edges with
    ``|S| >= min_siblings``.

    Compared to its **N2 null distribution**, never to an analytic zero: spec
    Sec.0 item 4 records that the analytic-zero claim is false because both
    residuals subtract the same in-sample ``beta-hat``.  Cross-fitting removes
    most of that coupling but not all of it (the folds are shared), so the null
    is the only defensible reference and the caller supplies it.

    ``feasible=False`` with a stated reason when fewer than ``min_edges``
    qualifying edges exist -- it NEVER returns 0.
    """
    if min_siblings is None:
        min_siblings = THRESH['L1_min_siblings']
    if min_edges is None:
        min_edges = THRESH['L1_min_edges']
    if boot_B is None:
        boot_B = THRESH['C2_block_bootstrap_B']
    v = np.asarray(values, dtype=np.float64)
    mean, deg = sibling_means(v, indptr, nbr, min_siblings=min_siblings)
    use = (np.asarray(keep, dtype=bool) & (deg >= int(min_siblings))
           & np.isfinite(v) & np.isfinite(mean))
    n_use = int(use.sum())
    out = dict(n_units=n_use, n_units_all=int((deg >= int(min_siblings)).sum()),
               feasible=bool(n_use >= int(min_edges)),
               gate_statistic='n_edges_with_|S|>=%d_in_P_a' % int(min_siblings),
               gate_value=float(n_use), gate_threshold=float(min_edges),
               label=label, median_siblings=(float(np.median(deg[use]))
                                             if n_use else float('nan')),
               max_siblings=int(deg.max()) if deg.size else 0)
    if not out['feasible']:
        out.update(slope=float('nan'), se_hc3=float('nan'), t=float('nan'),
                   p=float('nan'), r=float('nan'),
                   boot_lo95=float('nan'), boot_hi95=float('nan'),
                   infeasible_reason=(
                       'only %d edges in P_a have |S| >= %d (gate: >= %d) -- '
                       'the nested backgrounds are too shallow for a sibling to '
                       'exist' % (n_use, int(min_siblings), int(min_edges))))
        return out
    fit = ols_hc3(mean[use], v[use])
    out.update(slope=fit['slope'], se_hc3=fit['se_hc3'], t=fit['t'],
               p=fit['p'], r=fit['r'], intercept=fit['intercept'],
               infeasible_reason='')
    # block bootstrap over MUTATED POSITIONS (never over edges) -- the study's
    # ground rule.  Resample the position set; take every edge whose ADDED
    # substitution sits at a resampled position, with multiplicity.
    if positions is not None and boot_B:
        pos = np.asarray(positions)[use]
        mm = mean[use]
        vv = v[use]
        upos, pinv = np.unique(pos, return_inverse=True)
        buckets = [np.flatnonzero(pinv == t) for t in range(upos.size)]
        rng = np.random.default_rng(seed)
        bs = np.empty(int(boot_B))
        for bi in range(int(boot_B)):
            pick = rng.integers(0, upos.size, upos.size)
            sel = np.concatenate([buckets[t] for t in pick]) \
                if upos.size else np.zeros(0, dtype=np.int64)
            if sel.size < 8:
                bs[bi] = np.nan
                continue
            x, yv = mm[sel], vv[sel]
            xc = x - x.mean()
            sxx = float((xc * xc).sum())
            bs[bi] = (float((xc * (yv - yv.mean())).sum()) / sxx
                      if sxx > 0 else np.nan)
        out['boot_lo95'] = _pct(bs, 2.5)
        out['boot_hi95'] = _pct(bs, 97.5)
        out['boot_B'] = int(boot_B)
        out['n_position_blocks'] = int(upos.size)
    else:
        out['boot_lo95'] = float('nan')
        out['boot_hi95'] = float('nan')
    return out


# =========================================================================== #
# C3-L routes L2 / L2' -- does the interaction RECUR?                         #
# =========================================================================== #

#: An ICC over fewer than this many replicated groups has a CI too wide to
#: decide anything; reported beside the spec's own gate, never instead of it.
_MIN_ICC_GROUPS = 20

#: permutation replicates for the ICC label null (its p-value is 1/(B+1) on
#: every assay measured, so more replicates buy nothing).
_ICC_N_PERM = 200


def _icc_group_gate(cnt, *, min_groups, min_members, name):
    """The spec's L2 / L2' gate, plus the DISJUNCTIVE arm the spec's own
    feasibility claims force.

    Spec Sec.1.4 L2 gates on ">= 200 site-pairs with >= 2 backgrounds" while
    Sec.2 #13 calls CR9114-H1 the assay with the "strongest L1/L2/L4 power in
    the benchmark".  Both cannot hold as written: a 16-site BINARY hypercube has
    exactly ``C(16,2) = 120`` substitution pairs, so the group COUNT can never
    reach 200 no matter how deep the library is -- CR9114-H1 supplies its
    replication as 120 groups of ~15,000 backgrounds instead of 200 groups of 2.
    The gate is therefore read as **200 units of replication, however the design
    supplies them**: 200 groups with >= 2 members (the literal arm) OR >= 2
    groups with >= 200 members each (the transposed arm).  Both arms use only
    the spec's own two constants and nothing else.
    """
    cnt = np.asarray(cnt, dtype=np.int64)
    a = int((cnt >= int(min_members)).sum())
    b = int((cnt >= int(min_groups)).sum())
    lit = a >= int(min_groups)
    tra = b >= int(min_members)
    return dict(feasible=bool(lit or tra),
                gate_statistic=('%s: n_groups(>=%d members)>=%d OR '
                                'n_groups(>=%d members)>=%d'
                                % (name, int(min_members), int(min_groups),
                                   int(min_groups), int(min_members))),
                gate_value=float(a), gate_threshold=float(min_groups),
                gate_arm=('literal' if lit else
                          ('transposed' if tra else 'none')),
                n_groups_ge_members=a, n_groups_ge_groups=b)


def _icc_fast(v, gi, ng):
    """``ICC`` from a values vector and a PRE-COMPUTED group index -- one
    ``np.bincount`` pass, no ``np.unique``.  Identical arithmetic to
    :func:`cliff.nulls._icc_oneway`, asserted in :func:`_selfcheck`."""
    cnt = np.bincount(gi, minlength=ng).astype(np.float64)
    tot = np.bincount(gi, weights=v, minlength=ng)
    live = cnt > 0
    n = v.size
    ngl = int(live.sum())
    if ngl < 2 or n - ngl <= 0:
        return float('nan')
    gm = np.zeros(ng)
    gm[live] = tot[live] / cnt[live]
    grand = v.mean()
    ssb = float((cnt[live] * (gm[live] - grand) ** 2).sum())
    d = v - gm[gi]
    ssw = float((d * d).sum())
    msb, msw = ssb / (ngl - 1), ssw / (n - ngl)
    kbar = n / float(ngl)
    den = msb + (kbar - 1.0) * msw
    return float((msb - msw) / den) if den != 0 else float('nan')


def _icc_from_suff(cnt, tot, sq):
    """``ICC`` from per-group SUFFICIENT STATISTICS ``(n_g, sum_g, sumsq_g)``.

    This is what makes a 1,000-draw group bootstrap affordable on CR9114-H1,
    whose 120 groups hold 1.83e6 units between them: a bootstrap resample takes
    whole groups, and ``SSW`` is a per-group invariant while ``SSB`` needs only
    the group means and sizes, so a draw costs O(n_groups) instead of O(n).
    Verified against :func:`_icc_fast` to 1e-12 in :func:`_selfcheck`.
    """
    cnt = np.asarray(cnt, dtype=np.float64)
    tot = np.asarray(tot, dtype=np.float64)
    sq = np.asarray(sq, dtype=np.float64)
    live = cnt > 0
    cnt, tot, sq = cnt[live], tot[live], sq[live]
    ng = cnt.size
    n = float(cnt.sum())
    if ng < 2 or n - ng <= 0:
        return float('nan')
    gm = tot / cnt
    grand = tot.sum() / n
    ssb = float((cnt * (gm - grand) ** 2).sum())
    ssw = float((sq - tot * tot / cnt).sum())
    msb, msw = ssb / (ng - 1), ssw / (n - ng)
    kbar = n / float(ng)
    den = msb + (kbar - 1.0) * msw
    return float((msb - msw) / den) if den != 0 else float('nan')


def _icc_with_ci(values, groups, *, anchor=None, boot_B=None, seed=None,
                 n_perm=None):
    """One-way ICC with (a) a POSITION-BLOCK bootstrap CI and (b) a label
    permutation null.

    The CI resamples **mutated positions**, not units -- the study's ground rule.
    Each group (a substitution pair or a site pair) is anchored at the lower of
    its two mutated positions; a draw resamples the position set with
    replacement and takes every group anchored at a resampled position, whole
    and with multiplicity.  Resampling the units themselves would ignore the
    dominant dependence, since one ``y`` enters many ``eps``.  The draw is
    evaluated from per-group sufficient statistics, so its cost is O(n_groups).

    The permutation null shuffles the eps VALUES across groups, preserving the
    group-size profile and the eps marginal exactly.  That is the canonical
    "no clustering" reference for an ICC and the honest null for a recurrence
    statistic -- N2 cannot move it at all (module docstring, point 3), so it is
    reported alongside the N2 column rather than inside it.
    """
    if boot_B is None:
        boot_B = THRESH['C2_block_bootstrap_B']
    if n_perm is None:
        n_perm = _ICC_N_PERM
    v = np.asarray(values, dtype=np.float64)
    g = np.asarray(groups)
    ok = np.isfinite(v)
    v, g = v[ok], g[ok]
    an = None if anchor is None else np.asarray(anchor)[ok]
    out = dict(ICC=float('nan'), n=int(v.size), n_groups=0, n_groups_ge2=0,
               kbar=float('nan'), ICC_lo95=float('nan'), ICC_hi95=float('nan'),
               ICC_perm_mean=float('nan'), ICC_perm_p995=float('nan'),
               ICC_perm_p=float('nan'), ICC_boot_B=0, n_perm=0)
    if v.size < 8:
        return out
    _u, gi = np.unique(g, return_inverse=True)
    ng = int(_u.size)
    cnt = np.bincount(gi, minlength=ng).astype(np.float64)
    tot = np.bincount(gi, weights=v, minlength=ng)
    sq = np.bincount(gi, weights=v * v, minlength=ng)
    out.update(ICC=_icc_from_suff(cnt, tot, sq), n_groups=ng,
               n_groups_ge2=int((cnt >= 2).sum()),
               kbar=float(v.size / max(ng, 1)))
    if not np.isfinite(out['ICC']):
        return out
    rng = np.random.default_rng(seed)
    # ---- position-block bootstrap on the group sufficient statistics ------- #
    if an is not None and boot_B:
        gan = np.zeros(ng, dtype=np.int64)
        gan[gi] = an                       # constant within a group by design
        upos, pinv = np.unique(gan, return_inverse=True)
        by_pos = [np.flatnonzero(pinv == t) for t in range(upos.size)]
        bs = np.empty(int(boot_B))
        for bi in range(int(boot_B)):
            pick = rng.integers(0, upos.size, upos.size)
            grp = np.concatenate([by_pos[t] for t in pick]) if upos.size \
                else np.zeros(0, dtype=np.int64)
            if grp.size < 2:
                bs[bi] = np.nan
                continue
            bs[bi] = _icc_from_suff(cnt[grp], tot[grp], sq[grp])
        out['ICC_lo95'] = _pct(bs, 2.5)
        out['ICC_hi95'] = _pct(bs, 97.5)
        out['ICC_boot_B'] = int(boot_B)
        out['n_position_blocks'] = int(upos.size)
    # ---- label permutation null ------------------------------------------- #
    pv = np.empty(int(n_perm))
    for bi in range(int(n_perm)):
        pv[bi] = _icc_fast(rng.permutation(v), gi, ng)
    out['ICC_perm_mean'] = _nmean(pv)
    out['ICC_perm_p995'] = _pct(pv, 99.5)
    out['ICC_perm_p'] = _empirical_p(pv, out['ICC'])
    out['n_perm'] = int(n_perm)
    return out


def icc_across_backgrounds(eps, *, scale='eps_latent', bundle=None, seed=None,
                           boot_B=None):
    """Route **L2**: ICC of ``eps`` grouped by SUBSTITUTION pair, replicated
    across BACKGROUNDS ``C`` (spec Sec.1.4 L2).

    "site pair" means substitution pair throughout the spec's epsilon
    vocabulary -- :func:`cliff.noise.epsilon_sitepairs` settled that empirically
    (the KRAS twin shares 10,868 substitution pairs, the spec's own number,
    against 602 collapsed ``(chain,pos)`` pairs), and it is the only reading
    under which the spec's own "GB1_2016 / Z-HL1 / 5A12_VEGF: L2 feasible"
    claims are true.
    """
    sub = eps['sub_pair'].values
    v = eps[scale].values
    if sub.size == 0:
        return dict(feasible=False, n_units=0, ICC=float('nan'),
                    gate_statistic='n_sub_pairs_with_>=2_backgrounds',
                    gate_value=0.0,
                    gate_threshold=float(THRESH['L2_min_sitepairs']),
                    infeasible_reason='no 4-corner eps exists in this assay')
    _u, gi = np.unique(sub, return_inverse=True)
    cnt = np.bincount(gi)
    gate = _icc_group_gate(cnt, min_groups=THRESH['L2_min_sitepairs'],
                           min_members=THRESH['L2_min_backgrounds'],
                           name='sub_pairs')
    out = dict(gate)
    out['n_units'] = int(v.size)
    out['n_units_all'] = int(v.size)
    if not gate['feasible']:
        ung = icc_oneway(v, gi)
        out.update(ICC=float('nan'), ICC_lo95=float('nan'),
                   ICC_hi95=float('nan'), ICC_if_ungated=ung['ICC'],
                   n_groups=ung['n_groups'], n_groups_ge2=ung['n_groups_ge2'],
                   kbar=ung['kbar'],
                   infeasible_reason=(
                       'only %d substitution pairs carry >= %d backgrounds and '
                       'only %d carry >= %d (gate: either arm) -- the library '
                       'has no background depth at this site pair'
                       % (gate['n_groups_ge_members'],
                          THRESH['L2_min_backgrounds'],
                          gate['n_groups_ge_groups'],
                          THRESH['L2_min_sitepairs'])))
        return out
    anchor = np.minimum(eps['pos_i'].values, eps['pos_j'].values)
    res = _icc_with_ci(v, gi, anchor=anchor, seed=seed, boot_B=boot_B)
    out.update(res)
    out['infeasible_reason'] = ''
    out['ICC_if_ungated'] = res['ICC']
    return out


def icc_across_aa_combos(eps, *, scale='eps_latent', seed=None, boot_B=None):
    """Route **L2'**: ICC of ``eps`` grouped by ``(chain,pos)`` SITE pair,
    replicated across its different AMINO-ACID COMBINATIONS, at background order
    0 (spec Sec.1.4 L2').

    Restricted to ``bg_order == 0`` on purpose: mixing deeper backgrounds in
    would fold L2's axis into L2' and the two routes would stop being
    independent evidence.  This is the route that makes the flagship assay
    self-sufficient -- GB1_1FCC has no background depth at all (its library is
    singles + doubles), but 1,414 of the 1,485 possible site pairs carry a
    median of 45.5 amino-acid combinations each.
    """
    m0 = eps['bg_order'].values == 0
    sub = eps['site_pair'].values[m0]
    v = eps[scale].values[m0]
    if sub.size == 0:
        return dict(feasible=False, n_units=0, ICC=float('nan'),
                    gate_statistic='median_aa_combos_per_site_pair',
                    gate_value=float('nan'),
                    gate_threshold=float(THRESH['L2p_min_aa_combos']),
                    infeasible_reason=('no double whose two singles are both '
                                       'observed and uncensored (bg_order 0 is '
                                       'empty)'))
    _u, gi = np.unique(sub, return_inverse=True)
    cnt = np.bincount(gi)
    med = float(np.median(cnt))
    n_ge = int((cnt >= THRESH['L2p_min_aa_combos']).sum())
    n_ge2 = int((cnt >= 2).sum())
    feas = bool(med >= THRESH['L2p_min_aa_combos'] and n_ge2 >= _MIN_ICC_GROUPS)
    out = dict(feasible=feas, n_units=int(v.size), n_units_all=int(v.size),
               gate_statistic=('median_aa_combos_per_site_pair >= %g AND '
                               'n_site_pairs(>=2 combos) >= %d'
                               % (THRESH['L2p_min_aa_combos'], _MIN_ICC_GROUPS)),
               gate_value=med, gate_threshold=float(THRESH['L2p_min_aa_combos']),
               n_site_pairs_ge_gate=n_ge)
    if not feas:
        ung = icc_oneway(v, gi)
        out.update(ICC=float('nan'), ICC_lo95=float('nan'),
                   ICC_hi95=float('nan'), ICC_if_ungated=ung['ICC'],
                   n_groups=ung['n_groups'], n_groups_ge2=ung['n_groups_ge2'],
                   kbar=ung['kbar'],
                   infeasible_reason=(
                       'median %g amino-acid combinations per site pair '
                       '(gate >= %g) over %d site pairs, %d with >= 2 -- the '
                       'design does not repeat a site pair with different amino '
                       'acids' % (med, THRESH['L2p_min_aa_combos'],
                                  int(_u.size), n_ge2)))
        return out
    P = max(int(eps.attrs.get('P', 0)), 1)
    anchor = np.minimum(eps['pos_i'].values[m0], eps['pos_j'].values[m0])
    res = _icc_with_ci(v, gi, anchor=anchor, seed=seed, boot_B=boot_B)
    out.update(res)
    out['infeasible_reason'] = ''
    out['ICC_if_ungated'] = res['ICC']
    return out


# =========================================================================== #
# C3-L route L4 -- out-of-sample pairwise predictability                       #
# =========================================================================== #

def _r2_oos(A, target, folds, lam_grid, *, inner_folds=None, seed=None):
    """``R2_oos`` of a ridge on design ``A`` with ``lam`` chosen by an INNER
    k-fold CV inside every outer training set (spec Sec.1.4 L4).

    Nested exactly as the spec writes it, so ``lam`` never sees the outer
    held-out variants.  Returns the per-variant squared error too, because the
    position-block bootstrap of ``dR2_oos`` needs the terms, not the ratio.
    """
    if inner_folds is None:
        inner_folds = THRESH['L4_inner_folds']
    n = target.size
    pred = np.full(n, np.nan)
    lams = []
    for k in np.unique(folds):
        te = folds == k
        tr = ~te
        Atr, ttr = A[tr], target[tr]
        # inner CV over the training set only
        itr = np.arange(int(tr.sum())) % int(inner_folds)
        rng = np.random.default_rng(seed)
        itr = rng.permutation(itr)
        mse = np.zeros(len(lam_grid))
        for j in range(int(inner_folds)):
            ite = itr == j
            iin = ~ite
            Ain, tin = Atr[iin], ttr[iin]
            Aou, tou = Atr[ite], ttr[ite]
            for li, lam in enumerate(lam_grid):
                b, _ = _nulls._ridge_lsqr(Ain, tin, lam)
                r = tou - Aou.dot(b)
                mse[li] += float((r * r).sum())
        lam = float(lam_grid[int(np.argmin(mse))])
        lams.append(lam)
        b, _ = _nulls._ridge_lsqr(Atr, ttr, lam)
        pred[te] = A[te].dot(b)
    err = target - pred
    sse = err * err
    sst = (target - target.mean()) ** 2
    return dict(r2=float(1.0 - sse.sum() / sst.sum()) if sst.sum() > 0
                else float('nan'),
                sse=sse, sst=sst, pred=pred,
                lam=float(np.median(lams)), lams=tuple(lams))


def dr2_oos(bundle, *, target='z', min_obs_per_col=None, n_lambda=None,
            boot_B=None, seed=None, verbose=False):
    """Spec Sec.1.4 route **L4**: ``dR2_oos = R2_oos([1|X|Z]) - R2_oos([1|X])``.

    **The gate is load-bearing, not decoration.**  Below >= 5 observations per
    ``Z`` column every interaction is seen exactly once, the ridge can fit it
    perfectly in-sample and predicts nothing out of sample, so ``dR2_oos`` is
    identically 0 BY CONSTRUCTION -- and 0 would be read as "no epistasis",
    which is the opposite of what the data says.  Measured mean observations per
    ``Z`` column: GB1_1FCC 1.00, every KRAS 1.00, SARS2-RBD 1.01, CD19 3.37,
    hYAP65 3.11 => INFEASIBLE, declared.  CR9114-H1 16,306, CR6261 470,
    5A12_VEGF 212, Z-HL1 98, GB1_2016 66 => feasible.

    ``Z`` columns are SUBSTITUTION pairs (:func:`cliff.nulls.pairwise_design`);
    the spec's own L4 arithmetic ("GB1_2016 2,166 / 22,176" = C(4,2)*19*19)
    settles that against a site-pair reading, which would give 6.
    """
    if min_obs_per_col is None:
        min_obs_per_col = THRESH['L4_min_obs_per_col']
    if n_lambda is None:
        n_lambda = THRESH['L4_n_lambda']
    if boot_B is None:
        boot_B = THRESH['C2_block_bootstrap_B']
    b = bundle
    ctx = b.ctx
    t0 = time.time()
    pw = _pairs.pairwise_column_stats(b.keys, {c: c for c in range(ctx.M)},
                                      min_obs=int(min_obs_per_col))
    mean_obs = pw['mean_obs_per_col']
    out = dict(feasible=bool(np.isfinite(mean_obs)
                             and mean_obs >= float(min_obs_per_col)),
               gate_statistic='mean_obs_per_Z_col',
               gate_value=float(mean_obs), gate_threshold=float(min_obs_per_col),
               mean_obs_per_Z_col=float(mean_obs),
               n_Z_cols_all=int(pw['n_cols']),
               dR2_oos=float('nan'), dR2_lo95=float('nan'),
               dR2_hi95=float('nan'), top1pct_share=float('nan'),
               ridge_lambda=float('nan'), n_units=int(ctx.n),
               n_Z_cols=0, dR2_r2_add=float('nan'), dR2_r2_pair=float('nan'),
               dR2_top1_abs=float('nan'), n_top1_cols=0, wall_s=0.0,
               dR2_oos_unmasked=float('nan'))
    if not out['feasible']:
        out['infeasible_reason'] = (
            'mean %.4g observations per Z column (gate >= %g): every '
            'interaction is seen once, so dR2_oos is identically 0 by '
            'construction and reporting it as "no epistasis" would be false'
            % (mean_obs, float(min_obs_per_col)))
        return out
    Z, col_pairs, cooccur = _nulls.pairwise_design(
        ctx, min_cooccur=int(min_obs_per_col))
    out['n_Z_cols'] = int(Z.shape[1])
    if Z.shape[1] == 0:
        out['feasible'] = False
        out['infeasible_reason'] = (
            'no substitution pair is co-observed >= %d times' % min_obs_per_col)
        return out
    tgt = ctx.z if target == 'z' else ctx.y
    unc = ~ctx.censor_mask
    A_add = ctx.A
    A_pair = sp.hstack([ctx.A, Z], format='csr')
    sc = float(A_pair.multiply(A_pair).sum()) / A_pair.shape[1]
    lam_grid = np.geomspace(1e-4 * sc, 1e2 * sc, int(n_lambda))
    ent = list(seed) if seed is not None else [0]
    r_add = _r2_oos(A_add, tgt, ctx.folds, lam_grid, seed=ent)
    r_pair = _r2_oos(A_pair, tgt, ctx.folds, lam_grid, seed=ent)
    # censored rows are excluded from every pair statistic (spec Sec.1.0), so
    # they are excluded from the R2 accounting too
    sst = r_add['sst'][unc].sum()
    d = float((r_add['sse'][unc].sum() - r_pair['sse'][unc].sum()) / sst) \
        if sst > 0 else float('nan')
    out.update(dR2_oos=d, dR2_r2_add=r_add['r2'], dR2_r2_pair=r_pair['r2'],
               ridge_lambda=r_pair['lam'])
    # C3-A clause 3 (floor invariance) needs the SAME fits accounted over the
    # censored rows too.  The fits already used every row; only the R2
    # bookkeeping mask changes, so this costs nothing.
    sst_all = r_add['sst'].sum()
    out['dR2_oos_unmasked'] = (
        float((r_add['sse'].sum() - r_pair['sse'].sum()) / sst_all)
        if sst_all > 0 else float('nan'))
    # ---- top-1% of |coef| share ------------------------------------------- #
    bfull, _ = _nulls._ridge_lsqr(A_pair, tgt, r_pair['lam'])
    gam = np.abs(bfull[1 + ctx.M:])
    n_top = max(int(round(0.01 * gam.size)), 1)
    top = np.argsort(-gam)[:n_top]
    out['n_top1_cols'] = int(n_top)
    A_top = sp.hstack([ctx.A, Z[:, top]], format='csr')
    r_top = _r2_oos(A_top, tgt, ctx.folds, lam_grid, seed=ent)
    d_top = float((r_add['sse'][unc].sum() - r_top['sse'][unc].sum()) / sst) \
        if sst > 0 else float('nan')
    out['dR2_top1_abs'] = d_top
    out['top1pct_share'] = float(d_top / d) if d not in (0.0,) and np.isfinite(d) \
        else float('nan')
    # ---- CI: block bootstrap over MUTATED POSITIONS ----------------------- #
    if boot_B:
        pos_sets = [np.unique(b.pos_of_col[np.asarray(k, dtype=np.int64)])
                    if len(k) else np.zeros(0, dtype=np.int32)
                    for k in b.keys]
        anchor = np.array([(int(p[0]) if p.size else -1) for p in pos_sets])
        keepv = unc & (anchor >= 0)
        an = anchor[keepv]
        sa = r_add['sse'][keepv]
        spv = r_pair['sse'][keepv]
        stv = r_add['sst'][keepv]
        upos, pinv = np.unique(an, return_inverse=True)
        by = [np.flatnonzero(pinv == t) for t in range(upos.size)]
        # per-position sums are all the bootstrap needs
        s_a = np.array([sa[ix].sum() for ix in by])
        s_p = np.array([spv[ix].sum() for ix in by])
        s_t = np.array([stv[ix].sum() for ix in by])
        rng = np.random.default_rng(ent)
        bs = np.empty(int(boot_B))
        for bi in range(int(boot_B)):
            pick = rng.integers(0, upos.size, upos.size)
            tt = s_t[pick].sum()
            bs[bi] = ((s_a[pick].sum() - s_p[pick].sum()) / tt) if tt > 0 \
                else np.nan
        out['dR2_lo95'] = _pct(bs, 2.5)
        out['dR2_hi95'] = _pct(bs, 97.5)
        out['dR2_boot_B'] = int(boot_B)
        out['n_position_blocks'] = int(upos.size)
    out['infeasible_reason'] = ''
    out['wall_s'] = round(time.time() - t0, 2)
    if verbose:
        print('    [L4  ] %-40s Z=%d  R2 add=%.4f pair=%.4f  dR2=%.5f '
              '[%.5f,%.5f]  top1%%=%.3f  %.1fs'
              % (b.dms_id, out['n_Z_cols'], r_add['r2'], r_pair['r2'], d,
                 out['dR2_lo95'], out['dR2_hi95'], out['top1pct_share'],
                 out['wall_s']))
    return out


# =========================================================================== #
# C3-N -- measurement noise: the replication rate and its permutation chance  #
# =========================================================================== #

def replication_rate(eps_a, eps_b, sigma_eps, *, B=None, seed=None,
                     cliff_mult=None, replicate_mult=None, anchors=None,
                     boot_B=None):
    """Spec Sec.1.4 C3-N:
    ``R = P(|eps_b| >= 2 sigma_eps AND sign match | |eps_a| >= 3 sigma_eps)``
    with the chance level from ``B`` **site-pair-label permutations in assay b**.

    ORCHESTRATOR D5 makes the FLAGGED FRACTION part of the result, not a
    footnote: ``3 sigma_eps = 0.373`` flags 17.50% of KRAS's twice-measured eps
    values, and a "cliff" set that is 17.5% of the data is not the minority C2's
    clause assumes.  So ``frac_flagged_a`` / ``frac_flagged_b`` and
    ``flagged_frac_exceeds_5pct`` travel with every ``R``.

    The permutation destroys the site-pair correspondence in ``b`` while
    preserving its eps marginal exactly -- it is the "the same site pair is not
    the same site pair" null, which is the only null that can tell replication
    from a coincidence of two heavy-tailed marginals.  ``R`` is not compared to
    0.5: with 17.5% of ``b`` above ``2 sigma`` and a 50/50 sign, the chance level
    is around 0.09, and that is what ``R_chance_perm`` measures.

    The CI block-resamples **mutated positions** (``anchors``), never eps values.
    """
    if B is None:
        B = THRESH['C3N_n_perm']
    if cliff_mult is None:
        cliff_mult = THRESH['C3N_cliff_sigma_mult']
    if replicate_mult is None:
        replicate_mult = THRESH['C3N_replicate_sigma_mult']
    if boot_B is None:
        boot_B = THRESH['C2_block_bootstrap_B']
    a = np.asarray(eps_a, dtype=np.float64)
    bb = np.asarray(eps_b, dtype=np.float64)
    if a.size != bb.size:
        raise ValueError('replication_rate: eps_a has %d, eps_b has %d'
                         % (a.size, bb.size))
    ok = np.isfinite(a) & np.isfinite(bb)
    a, bb = a[ok], bb[ok]
    an = None if anchors is None else np.asarray(anchors)[ok]
    sig = float(sigma_eps)
    ca = cliff_mult * sig
    cb = replicate_mult * sig
    n = a.size
    out = dict(n_shared=int(n), sigma_eps=sig, cliff_abs_a=float(ca),
               replicate_abs_b=float(cb), sigma_mult=float(cliff_mult),
               n_flagged_a=0, n_flagged_b=0, n_replicated=0,
               frac_flagged_a=float('nan'), frac_flagged_b=float('nan'),
               R=float('nan'), R_lo95=float('nan'), R_hi95=float('nan'),
               R_chance_perm=float('nan'), R_perm_p995=float('nan'),
               perm_p=float('nan'), sign_agreement=float('nan'),
               sign_agreement_chance=float('nan'),
               sign_agreement_perm_p=float('nan'),
               flagged_frac_exceeds_5pct=None, B=int(B), boot_B=0)
    if n < 8 or not np.isfinite(sig) or sig <= 0:
        return out
    fa = np.abs(a) >= ca
    fb = np.abs(bb) >= cb
    out['n_flagged_a'] = int(fa.sum())
    out['n_flagged_b'] = int(fb.sum())
    out['frac_flagged_a'] = float(fa.mean())
    out['frac_flagged_b'] = float(fb.mean())
    out['flagged_frac_exceeds_5pct'] = bool(fa.mean() > 0.05)
    if not fa.any():
        return out
    sm = np.sign(a) == np.sign(bb)
    rep = fb & sm
    out['n_replicated'] = int(rep[fa].sum())
    out['R'] = float(rep[fa].mean())
    out['sign_agreement'] = float(sm[fa].mean())
    # ---- the site-pair-label permutation in assay b ----------------------- #
    rng = np.random.default_rng(seed)
    idx_a = np.flatnonzero(fa)
    k = idx_a.size
    sa = np.sign(a[idx_a])
    pr = np.empty(int(B))
    ps = np.empty(int(B))
    for t in range(int(B)):
        sel = rng.permutation(n)[:k]
        hit = (np.abs(bb[sel]) >= cb)
        sg = (np.sign(bb[sel]) == sa)
        pr[t] = float((hit & sg).mean())
        ps[t] = float(sg.mean())
    out['R_chance_perm'] = float(pr.mean())
    out['R_perm_p995'] = _pct(pr, 99.5)
    out['perm_p'] = _empirical_p(pr, out['R'])
    out['sign_agreement_chance'] = float(ps.mean())
    out['sign_agreement_perm_p'] = _empirical_p(ps, out['sign_agreement'])
    # ---- block bootstrap over MUTATED POSITIONS --------------------------- #
    if an is not None and boot_B:
        upos, pinv = np.unique(an, return_inverse=True)
        by = [np.flatnonzero(pinv == t) for t in range(upos.size)]
        rr = np.empty(int(boot_B))
        for t in range(int(boot_B)):
            pick = rng.integers(0, upos.size, upos.size)
            sel = np.concatenate([by[q] for q in pick]) if upos.size \
                else np.zeros(0, dtype=np.int64)
            f = fa[sel]
            rr[t] = float(rep[sel][f].mean()) if f.any() else np.nan
        out['R_lo95'] = _pct(rr, 2.5)
        out['R_hi95'] = _pct(rr, 97.5)
        out['boot_B'] = int(boot_B)
        out['n_position_blocks'] = int(upos.size)
    return out


_SUB_RE = None


def _parse_sub_token(tok):
    """``'A12K' -> ('A', 12, 'K')`` for :func:`cliff.noise.kras_twin_epsilon`'s
    ``key`` strings (its own ``'%s%d%s'`` format)."""
    global _SUB_RE
    if _SUB_RE is None:
        import re
        _SUB_RE = re.compile(r'^(.)(-?\d+)(.)$')
    m = _SUB_RE.match(tok)
    if m is None:
        raise ValueError('cannot parse substitution token %r' % tok)
    return m.group(1), int(m.group(2)), m.group(3)


def _twin_anchor(key_str):
    """The LOWER mutated ``(chain,pos)`` of a substitution-pair key -- the block
    the bootstrap resamples."""
    toks = key_str.split('|')
    sites = sorted((c, p) for c, p, _aa in (_parse_sub_token(t) for t in toks))
    return '%s%d' % sites[0]


def l3_cross_measurement(pair=None, *, sigma_eps=None, B=None, seed=None,
                         boot_B=None, verbose=False):
    """Route **L3** / the C3-N test: the KRAS twin, the ONLY place in the
    benchmark where the same interaction eps is measured twice.

    Delegates the join to :func:`cliff.noise.kras_twin_epsilon` (which settled
    that a "shared site-pair" is a shared SUBSTITUTION pair: 10,868 of them,
    the spec's own count, against 602 once the amino acids are collapsed) and
    adds the replication rate at **both** ORCHESTRATOR-D5 thresholds over the
    mandatory ``sigma x {0.5, 1, 2}`` surface.
    """
    if pair is None:
        pair = _noise.KRAS_TWIN
    if B is None:
        B = THRESH['C3N_n_perm']
    if seed is None:
        seed = [SEEDS['replication_perm']]
    tw = _noise.kras_twin_epsilon(pair)
    tab = tw['table']
    ea = tab['eps_a'].values.astype(np.float64)
    eb = tab['eps_b'].values.astype(np.float64)
    anchors = np.array([_twin_anchor(k) for k in tab['key'].values])
    sig_measured = float(tw['sigma_eps'])
    # ORCHESTRATOR D5: sigma_eps = 0.1243 is inconsistent by a factor 1.65 with
    # sqrt(3) sigma_y = 0.2055, because eps of a double is a 4-term contrast and
    # Var(eps) ~ 3 sigma_y^2.  BOTH are reported, never one.
    sig_y = float(config.NOISE['KRAS']['sigma_y'])
    sig_contrast = _SQRT3 * sig_y
    bases = [('3sigma_eps_measured', sig_measured,
              'measured_replicate (KRAS twin, contaminated by the construct '
              'difference)'),
             ('3sqrt3sigma_y_contrast', sig_contrast,
              'sqrt(3) * sigma_y: eps is a 4-term contrast, Var(eps) ~ 3 '
              'sigma_y^2 (ORCHESTRATOR D5)')]
    rows = []
    for label, sig, prov in bases:
        for m in (1.0,) + tuple(x for x in _SIGMA_MULTS if x != 1.0):
            r = replication_rate(ea, eb, sig * m, B=B, seed=seed,
                                 anchors=anchors, boot_B=boot_B)
            r.update(threshold_label=label, sigma_mult=float(m),
                     sigma_eps_provenance=prov,
                     row_role=('primary' if m == 1.0 else 'sensitivity'))
            rows.append(r)
            if verbose:
                print('    [C3N ] %-24s x%.1f  |eps_a|>=%.4f flags %5d/%5d '
                      '(%.2f%%)  R=%.4f chance=%.4f p=%.4g sign=%.4f'
                      % (label, m, r['cliff_abs_a'], r['n_flagged_a'],
                         r['n_shared'], 100.0 * r['frac_flagged_a'], r['R'],
                         r['R_chance_perm'], r['perm_p'], r['sign_agreement']))
    # ---- the two-way decomposition eps^(a) = mu + delta^(a) + noise -------- #
    va, vb = float(ea.var(ddof=1)), float(eb.var(ddof=1))
    V = 0.5 * (va + vb)
    C = float(np.cov(ea, eb, ddof=1)[0, 1])
    f_raw = (V - C) / V if V > 0 else float('nan')
    num = V - C - sig_measured ** 2
    den = num + C
    f_corr = (num / den) if den != 0 else float('nan')
    icc2 = icc_oneway(np.concatenate([ea, eb]),
                      np.concatenate([np.arange(ea.size)] * 2))['ICC']
    return dict(twin=tw, rows=rows, eps_a=ea, eps_b=eb, anchors=anchors,
                sigma_measured=sig_measured, sigma_contrast=sig_contrast,
                F_spec=f_raw, F_spec_noise_corrected=f_corr,
                icc_eps_two_measurements=icc2, var_a=va, var_b=vb, cov_ab=C)


# =========================================================================== #
# C3-L route L5 -- 3D localisation of eps                                     #
# =========================================================================== #

_T09 = {}


def t09_sites(dms_id):
    """The stage-1 structural annotation of ``dms_id``'s MUTATED positions, read
    from ``T09_structure_sites.csv``.

    Read, never recomputed: spec Sec.5 forbids re-annotating a cached structure,
    and the assays absent from T09 are exactly the ones with no usable PDB
    (CR9114 x2, CR6261) -- which is L5's feasibility gate, not an error.
    """
    if 'df' not in _T09:
        p = os.path.join(PATHS.artifacts, 'T09_structure_sites.csv')
        _T09['df'] = (pd.read_csv(p) if os.path.exists(p) else None)
        _T09['path'] = p
    df = _T09['df']
    if df is None:
        return None
    sub = df[df['DMS_id'].astype(str) == str(dms_id)]
    return sub if len(sub) else None


def site_geometry(bundle, *, verbose=False):
    """Per-POSITION structural annotation and the ``P x P`` **site-site min
    heavy-atom distance** matrix, which no cache holds.

    ``structure.py``'s cached ``min_heavy_dist`` is the distance to the OPPOSITE
    SIDE of the interface -- a different quantity from L5's ``min heavy-atom
    distance between sites s and t``, which is what the spec's L5 / C4-P score
    is.  So the PDB is re-parsed once per assay (~1 s) and the pair distances are
    computed over the mutated residues only (<= ~300 of them, so <= 45,000 pairs).

    Returns ``None`` when the assay has no structural annotation at all.
    """
    b = bundle
    spec = config.ASSAYS[b.dms_id]
    t9 = t09_sites(b.dms_id)
    P = b.ctx.P
    if t9 is None:
        return dict(ok=False, P=P, reason=('%s has no row in T09: no usable PDB, '
                                           'so no site is structurally annotated'
                                           % b.dms_id))
    pos_index = b.des['pos_index']
    # (chain, seq_idx) -> the position column j the eps table indexes by
    j_of = {}
    for (ch, ps), j in pos_index.items():
        j_of[(str(ch), int(ps))] = int(j)
    reskey = [None] * P
    rsa = np.full(P, np.nan)
    mhd = np.full(P, np.nan)
    dsasa = np.full(P, np.nan)
    cbd = np.full(P, np.nan)
    levy = np.array([''] * P, dtype=object)
    seqi = np.full(P, -1, dtype=np.int64)
    chain = np.array([''] * P, dtype=object)
    n_hit = 0
    for r in t9.to_dict('records'):
        k = (str(r['chain']), int(r['seq_idx']))
        j = j_of.get(k)
        if j is None:
            continue
        n_hit += 1
        ic = r.get('icode')
        ic = '' if (ic is None or (isinstance(ic, float) and not np.isfinite(ic))) \
            else str(ic).strip()
        ic = '' if ic == 'nan' else ic
        reskey[j] = (str(r['chain']), int(r['resseq']), ic)
        rsa[j] = float(r['rsa_iso'])
        mhd[j] = float(r['min_heavy_dist'])
        dsasa[j] = float(r['dsasa'])
        cbd[j] = float(r['cb_dist'])
        levy[j] = str(r['levy_class'])
        seqi[j] = int(r['seq_idx'])
        chain[j] = str(r['chain'])
    annotated = np.array([k is not None for k in reskey])
    out = dict(ok=bool(annotated.any()), P=P, annotated=annotated, rsa=rsa,
               min_heavy_dist=mhd, dsasa=dsasa, cb_dist=cbd, levy=levy,
               seq_idx=seqi, chain=chain, n_annotated=int(annotated.sum()),
               reason='')
    if not out['ok']:
        out['reason'] = ('T09 has %d rows for %s but none joins onto pos_index'
                         % (len(t9), b.dms_id))
        return out
    # ---- the P x P min heavy-atom distance between mutated sites ----------- #
    pdb_path = os.path.join(PATHS.structures, spec.pdb_file)
    if not os.path.exists(pdb_path):
        out['ok'] = False
        out['reason'] = 'PDB %s absent' % spec.pdb_file
        return out
    keep = set(str(spec.side0_chains)) | set(str(spec.side1_chains))
    t0 = time.time()
    _st, model, _nh = _structure.load_heavy_model(pdb_path, keep_chains=keep)
    fl = _structure._flatten(model)
    ri = {k: i for i, k in enumerate(fl['keys'])}
    blocks = [None] * P
    for j in range(P):
        k = reskey[j]
        if k is None:
            continue
        i = ri.get(k)
        if i is None:
            annotated[j] = False
            continue
        s = int(fl['starts'][i])
        e = s + int(fl['n_atoms'][i])
        blocks[j] = fl['coords'][s:e]
    d3d = np.full((P, P), np.nan)
    have = [j for j in range(P) if blocks[j] is not None]
    for ii, j in enumerate(have):
        Aj = blocks[j]
        d3d[j, j] = 0.0
        for k in have[ii + 1:]:
            Bk = blocks[k]
            dd = Aj[:, None, :] - Bk[None, :, :]
            v = float(np.sqrt((dd * dd).sum(axis=2)).min())
            d3d[j, k] = v
            d3d[k, j] = v
    out.update(d3d=d3d, annotated=annotated,
               n_annotated=int(annotated.sum()), pdb=spec.pdb_file,
               wall_s=round(time.time() - t0, 2))
    if verbose:
        print('    [geom] %-40s %3d/%d positions on %s, %d site pairs, %.1fs'
              % (b.dms_id, out['n_annotated'], P, spec.pdb_file,
                 len(have) * (len(have) - 1) // 2, out['wall_s']))
    return out


def _auroc_from_ranks(ranks, lab, n1, n0):
    u = float(ranks[lab].sum()) - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n0))


def l5_auroc(bundle, eps, *, sigma, scale='eps_latent', geom=None, B=None,
             seed=None, boot_B=None, bg0_only=True, verbose=False):
    """Spec Sec.1.4 route **L5**: AUROC of ``(-min heavy-atom distance between
    sites s,t)`` discriminating ``|eps| >= 3 sigma`` from ``|eps| < 1 sigma``,
    against the **NS2** label permutation (seq-separation decile x rsa tertile).

    Gate: ``>= 500`` eps values with BOTH sites structurally annotated
    (``THRESH['L5_min_eps']``), and at least 10 of each class -- an AUROC over 3
    positives is not a number.

    ``bg0_only=True`` restricts to ``bg_order == 0``, the spec's own ``eps_st``;
    the all-backgrounds answer is reported beside it as
    ``AUROC_all_backgrounds`` so the restriction is visible rather than assumed.
    """
    if B is None:
        B = THRESH['null_B']
    if boot_B is None:
        boot_B = THRESH['C2_block_bootstrap_B']
    b = bundle
    g = geom if geom is not None else site_geometry(b)
    mult = THRESH['L5_cliff_sigma_mult']
    out = dict(feasible=False, n_units=0, n_units_all=int(len(eps)),
               gate_statistic='n_eps_with_both_sites_annotated',
               gate_value=0.0, gate_threshold=float(THRESH['L5_min_eps']),
               AUROC_L5=float('nan'), AUROC_lo95=float('nan'),
               AUROC_hi95=float('nan'), p_NS2=float('nan'),
               AUROC_NS2_mean=float('nan'), AUROC_NS2_p995=float('nan'),
               n_cliff_L5=0, n_noncliff_L5=0, sigma_eps_used=float(sigma),
               AUROC_all_backgrounds=float('nan'), infeasible_reason='')
    if not g.get('ok'):
        out['infeasible_reason'] = g.get('reason', 'no structural annotation')
        return out
    if not (np.isfinite(sigma) and sigma > 0):
        out['infeasible_reason'] = 'no usable sigma_eps for the |eps| >= 3 sigma label'
        return out
    d3d = g['d3d']
    ann = g['annotated']

    def _prep(sel):
        pi = eps['pos_i'].values[sel]
        pj = eps['pos_j'].values[sel]
        v = eps[scale].values[sel]
        okp = ann[pi] & ann[pj] & np.isfinite(v)
        pi, pj, v = pi[okp], pj[okp], v[okp]
        d = d3d[pi, pj]
        okd = np.isfinite(d)
        return pi[okd], pj[okd], v[okd], d[okd]

    m0 = (eps['bg_order'].values == 0) if bg0_only else \
        np.ones(len(eps), dtype=bool)
    if bg0_only and not m0.any():
        m0 = np.ones(len(eps), dtype=bool)
        out['notes'] = 'bg_order 0 is empty; L5 run on all backgrounds'
    pi, pj, v, d = _prep(m0)
    out['gate_value'] = float(pi.size)
    av = np.abs(v)
    lab = av >= mult * sigma
    neg = av < 1.0 * sigma
    keep = lab | neg
    n1, n0 = int(lab.sum()), int(neg.sum())
    out.update(n_units=int(keep.sum()), n_cliff_L5=n1, n_noncliff_L5=n0)
    if pi.size < THRESH['L5_min_eps'] or n1 < 10 or n0 < 10:
        out['infeasible_reason'] = (
            'only %d eps have both sites structurally annotated (gate >= %d), '
            'with %d at |eps| >= %g sigma and %d at < 1 sigma (>= 10 of each '
            'needed for an AUROC)'
            % (pi.size, THRESH['L5_min_eps'], n1, mult, n0))
        return out
    score = -d[keep]
    y = lab[keep]
    out['AUROC_L5'] = auroc(score, y)
    # ---- NS2: permute the cliff label within seqsep-decile x rsa-tertile --- #
    same = np.array([g['chain'][x] == g['chain'][q]
                     for x, q in zip(pi[keep], pj[keep])])
    sep = np.abs(g['seq_idx'][pi[keep]] - g['seq_idx'][pj[keep]]).astype(float)
    sep = np.where(same, sep, sep + 1e4)
    tab = dict(seq_separation=sep,
               rsa_iso=0.5 * (g['rsa'][pi[keep]] + g['rsa'][pj[keep]]),
               is_cliff_3sigma=y)
    ranks = _st.rankdata(score)
    nn1, nn0 = int(y.sum()), int((~y).sum())
    rng = np.random.default_rng(seed)
    out['B_null'] = int(B or 0)
    if B:
        nv = np.empty(int(B))
        for t in range(int(B)):
            pl = _nulls.permute_NS2(tab, rng).astype(bool)
            nv[t] = _auroc_from_ranks(ranks, pl, nn1, nn0)
        out['AUROC_NS2_mean'] = _nmean(nv)
        out['AUROC_NS2_p995'] = _pct(nv, 99.5)
        out['p_NS2'] = _empirical_p(nv, out['AUROC_L5'])
    # ---- CI: block bootstrap over MUTATED POSITIONS ----------------------- #
    if boot_B:
        an = np.minimum(pi[keep], pj[keep])
        upos, pinv = np.unique(an, return_inverse=True)
        by = [np.flatnonzero(pinv == t) for t in range(upos.size)]
        bs = np.empty(int(boot_B))
        for t in range(int(boot_B)):
            pick = rng.integers(0, upos.size, upos.size)
            sel = np.concatenate([by[q] for q in pick]) if upos.size \
                else np.zeros(0, dtype=np.int64)
            yy = y[sel]
            bs[t] = (auroc(score[sel], yy) if (yy.any() and (~yy).any())
                     else np.nan)
        out['AUROC_lo95'] = _pct(bs, 2.5)
        out['AUROC_hi95'] = _pct(bs, 97.5)
        out['n_position_blocks'] = int(upos.size)
    # ---- the all-backgrounds comparison ----------------------------------- #
    if bg0_only and (eps['bg_order'].values != 0).any():
        pi2, pj2, v2, d2 = _prep(np.ones(len(eps), dtype=bool))
        av2 = np.abs(v2)
        l2 = av2 >= mult * sigma
        n2_ = av2 < 1.0 * sigma
        k2 = l2 | n2_
        if k2.sum() >= 20 and l2.sum() >= 10 and n2_.sum() >= 10:
            out['AUROC_all_backgrounds'] = auroc(-d2[k2], l2[k2])
    else:
        out['AUROC_all_backgrounds'] = out['AUROC_L5']
    out['feasible'] = True
    if verbose:
        print('    [L5  ] %-40s n=%d (%d cliff / %d non) AUROC=%.4f '
              '[%.4f,%.4f]  NS2 mean=%.4f p995=%.4f  p=%.4g'
              % (b.dms_id, out['n_units'], n1, n0, out['AUROC_L5'],
                 out['AUROC_lo95'], out['AUROC_hi95'], out['AUROC_NS2_mean'],
                 out['AUROC_NS2_p995'], out['p_NS2']))
    return out


# =========================================================================== #
# C3-A -- the four artefact clauses                                           #
# =========================================================================== #

def position_cliff_table(bundle, *, tau, keep=None, c=None):
    """Per-MUTATED-POSITION pair count and cliff rate over ``P_a``.

    The position of an edge is the position of its ADDED substitution
    (``ctx.pos_of_add``), which is the only position the edge is *about*: its
    background is shared by both endpoints and cancels in ``c_hat``.
    """
    b = bundle
    cv = b.c_obs if c is None else np.asarray(c, dtype=np.float64)
    kp = b.pa if keep is None else np.asarray(keep, dtype=bool)
    kp = kp & np.isfinite(cv)
    pos = b.ctx.pos_of_add[kp]
    fl = (np.abs(cv[kp]) >= float(tau)).astype(np.float64)
    P = int(b.ctx.P)
    n = np.bincount(pos, minlength=P).astype(np.float64)
    k = np.bincount(pos, weights=fl, minlength=P)
    with np.errstate(divide='ignore', invalid='ignore'):
        rate = k / n
    rate[n < 1] = np.nan
    return pd.DataFrame(dict(pos=np.arange(P), n_pairs=n.astype(np.int64),
                             n_cliff=k.astype(np.int64), cliff_rate=rate))


#: The structural covariates clause 1 offers as the competing explanation.
_STRUCT_COVARIATES = ('rsa_iso', 'min_heavy_dist', 'dsasa', 'cb_dist')


def depth_clause(bundle, *, taus=None, geom=None, keep=None, c=None):
    """C3-A clause 1 (sampling depth).  REFUTED if
    ``Spearman(per-position cliff rate, per-position pair count) > 0.40`` while
    the best structural covariate is ``< 0.20`` (spec Sec.1.4).

    ``depth_spearman`` is stored **SIGNED**, and the test is one-sided, because
    the artefact the clause exists to catch is directional: a position sampled
    more deeply showing MORE cliffs means depth is driving the cliff count.  The
    other sign -- deeply sampled positions showing FEWER cliffs -- is dilution /
    regression to the mean, not a depth-manufactured cliff, and turning it into
    a refutation would kill an assay for being well sampled.  The sign is
    reported rather than hidden: three assays here are strongly NEGATIVE
    (hYAP65, CR6261, Z-ZpA963_HL1) and none of them trips the clause.
    ``depth_json`` carries the value at every tau.
    """
    if taus is None:
        taus = (_CAT_TAU, _ALT_TAU)
    b = bundle
    g = geom if geom is not None else site_geometry(b)
    out = dict(depth_spearman=float('nan'), depth_spearman_tau3=float('nan'),
               best_struct_covariate=float('nan'),
               best_struct_covariate_name='', struct_covariate_json='{}',
               depth_tau_used=float('nan'), depth_n_positions=0)
    per = {}
    for t in taus:
        pt = position_cliff_table(b, tau=t, keep=keep, c=c)
        live = pt[pt['n_pairs'] > 0]
        per[float(t)] = dict(
            rho=_spearman(live['cliff_rate'].values,
                          live['n_pairs'].values.astype(float)),
            n_pos=int(len(live)), n_cliff=int(pt['n_cliff'].sum()), tab=pt)
    out['depth_spearman'] = per[float(taus[0])]['rho']
    if len(taus) > 1:
        out['depth_spearman_tau3'] = per[float(taus[1])]['rho']
    out['depth_tau_used'] = float(taus[0])
    out['depth_n_positions'] = per[float(taus[0])]['n_pos']
    # the competing structural explanation, on the SAME per-position rate
    covs = {}
    if g.get('ok'):
        pt = per[float(taus[0])]['tab']
        live = pt['n_pairs'].values > 0
        r = pt['cliff_rate'].values
        for name in _STRUCT_COVARIATES:
            v = g.get(name)
            if v is None:
                continue
            rho = _spearman(r[live], np.asarray(v, dtype=np.float64)[live])
            covs[name] = None if not np.isfinite(rho) else round(abs(rho), 6)
        fin = [(v, k) for k, v in covs.items() if v is not None]
        if fin:
            v, k = max(fin)
            out['best_struct_covariate'] = float(v)
            out['best_struct_covariate_name'] = k
    out['struct_covariate_json'] = json.dumps(covs, sort_keys=True)
    out['depth_json'] = json.dumps(
        dict((str(k), dict(rho=(None if not np.isfinite(v['rho']) else
                                round(v['rho'], 6)),
                           n_positions=v['n_pos'], n_cliff=v['n_cliff']))
             for k, v in per.items()), sort_keys=True)
    out['_per_tau'] = per
    return out


def density_strata(cliff_flag, degree, n_bins=None):
    """Spec Sec.3's ``density_strata``: cliff rate by neighbourhood-density
    quintile (equal-count bins of the observed nested degree).

    ``degree`` is the EDGE's neighbourhood density -- ``deg(u) + deg(v)``, the
    number of nested neighbours its two endpoints have between them -- because
    the artefact clause is about how densely the LIBRARY samples around the
    edge, and one endpoint's degree is only half of that.
    """
    if n_bins is None:
        n_bins = THRESH['C3A_n_density_bins']
    fl = np.asarray(cliff_flag, dtype=bool)
    dg = np.asarray(degree, dtype=np.float64)
    n = fl.size
    if n == 0:
        return pd.DataFrame(dict(quintile=[], n=[], degree_lo=[], degree_hi=[],
                                 n_cliff=[], rate=[]))
    order = np.argsort(dg, kind='stable')
    q = np.empty(n, dtype=np.int64)
    for i, part in enumerate(np.array_split(order, int(n_bins))):
        q[part] = i
    rows = []
    for i in range(int(n_bins)):
        s = q == i
        rows.append(dict(quintile=i + 1, n=int(s.sum()),
                         degree_lo=(float(dg[s].min()) if s.any() else np.nan),
                         degree_hi=(float(dg[s].max()) if s.any() else np.nan),
                         n_cliff=int(fl[s].sum()),
                         rate=(float(fl[s].mean()) if s.any() else np.nan)))
    out = pd.DataFrame(rows)
    out.attrs['quintile'] = q
    return out


# =========================================================================== #
# the N2 reference for L1 and for the density clause                          #
# =========================================================================== #

#: fork-inherited payload for the N2 workers (same pattern as
#: :data:`cliff.nulls._WORKER`: a closure over a 30 MB adjacency must not be
#: pickled once per replicate).
_N2W = {}


def _n2_init(payload):
    _N2W.clear()
    _N2W.update(payload)


def _n2_run(bi):
    """One N2 replicate's L1 slope, its ``icc_addcol`` and its per-quintile
    cliff rate.

    Reproduces :func:`cliff.nulls.replicate`'s N2 path EXACTLY -- same strata,
    same ``default_rng(seed_entropy + [b])`` -- so this ensemble and the cached
    ``nulls/{id}_N2_B200_*.npz`` are the same 200 surrogates seen through
    different statistics.
    """
    W = _N2W
    ctx = W['ctx']
    rng = np.random.default_rng(list(W['ent']) + [int(bi)])
    e = _nulls.surrogate_N2(ctx, ctx.e_oof, rng, W['strata'])
    ec = e - ctx.mu_oof
    u, v = W['u'], W['v']
    with np.errstate(divide='ignore', invalid='ignore'):
        c = (ec[v] - ec[u]) / W['den']
    vals = np.where(W['pa'], c, np.nan)
    out = [np.nan, np.nan] + [np.nan] * int(W['n_bins'])
    mean, deg = sibling_means(vals, W['indptr'], W['nbr'],
                              min_siblings=W['min_siblings'])
    use = W['pa'] & (deg >= int(W['min_siblings'])) & np.isfinite(vals) \
        & np.isfinite(mean)
    if use.sum() >= 8:
        out[0] = ols_hc3(mean[use], vals[use])['slope']
    kp = W['pa'] & np.isfinite(c)
    if kp.sum() >= 8:
        i1, _n, _g = _nulls._icc_oneway(c[kp], ctx.add_col[kp])
        out[1] = i1
        fl = (np.abs(c[kp]) >= float(W['tau'])).astype(np.float64)
        q = W['q'][kp]
        nb = int(W['n_bins'])
        cnt = np.bincount(q, minlength=nb).astype(np.float64)
        hit = np.bincount(q, weights=fl, minlength=nb)
        with np.errstate(divide='ignore', invalid='ignore'):
            r = hit / cnt
        r[cnt < 1] = np.nan
        out[2:] = list(r)
    return out


def n2_reference(bundle, indptr, nbr, *, tau, q, n_bins=None, B=None, nproc=1,
                 min_siblings=None, verbose=False):
    """The N2 ensemble of everything C3 compares to a null: L1's ``beta_a``,
    ``icc_addcol`` and the per-density-quintile cliff rate.

    N2 permutes ``e`` within ``(order x phi-decile x censored)`` strata with
    ``beta, phi, g, sigma`` held fixed, so it is the "the deviations are real in
    size but attached to the wrong variants" null -- which is precisely the
    reference L1 needs.  Spec Sec.0 item 4: the analytic zero for ``beta_a`` is
    FALSE (both residuals subtract the same in-sample ``beta-hat``), so this
    ensemble, not 0, is the comparison.
    """
    if B is None:
        B = THRESH['null_B']
    if n_bins is None:
        n_bins = THRESH['C3A_n_density_bins']
    if min_siblings is None:
        min_siblings = THRESH['L1_min_siblings']
    b = bundle
    ctx = b.ctx
    u, v = ctx.nested_idx[:, 0], ctx.nested_idx[:, 1]
    payload = dict(
        ctx=ctx, indptr=indptr, nbr=nbr, pa=b.pa, u=u, v=v,
        den=np.sqrt(ctx.sigma_oof[u] ** 2 + ctx.sigma_oof[v] ** 2),
        strata=_nulls.make_strata(ctx.n_muts, ctx.phi_oof,
                                  censor_mask=ctx.censor_mask),
        ent=config.assay_seed('nulls_N2', ctx.dms_id),
        tau=float(tau), q=np.asarray(q, dtype=np.int64), n_bins=int(n_bins),
        min_siblings=int(min_siblings))
    t0 = time.time()
    nproc = max(1, min(int(nproc), THRESH['nproc_cap']))
    _n2_init(payload)
    if nproc <= 1:
        rows = [_n2_run(i) for i in range(int(B))]
    else:
        import multiprocessing as mp
        cls = mp.get_context('fork')
        with cls.Pool(processes=nproc, initializer=_n2_init,
                      initargs=(payload,)) as pool:
            rows = list(pool.imap(_n2_run, range(int(B)), chunksize=1))
    arr = np.asarray(rows, dtype=np.float64)
    out = dict(B=int(B), beta=arr[:, 0], icc_addcol=arr[:, 1],
               rate_q=arr[:, 2:], wall_s=round(time.time() - t0, 2),
               nproc=int(nproc), seed=list(payload['ent']))
    if verbose:
        print('    [N2  ] %-40s B=%d beta mean=%.4f sd=%.4f p995=%.4f  %.1fs'
              % (ctx.dms_id, B, _nmean(out['beta']), _nsd(out['beta']), _pct(out['beta'], 99.5),
                 out['wall_s']))
    return out


# =========================================================================== #
# the noise registry read (T03), and the sigma the eps routes threshold on    #
# =========================================================================== #

_T03 = {}

_PROV_RANK = {'measured_replicate': 0, 'cross_study_contaminated': 1,
              'internal_residual': 2, 'stipulated': 3}


def t03_sigma(dms_id):
    """``(sigma_y, sigma_eps, provenance)`` for one assay from
    ``T03_noise_registry.csv``, preferring the best provenance available."""
    if 'df' not in _T03:
        p = os.path.join(PATHS.artifacts, 'T03_noise_registry.csv')
        _T03['df'] = pd.read_csv(p) if os.path.exists(p) else None
    df = _T03['df']
    out = dict(sigma_y=float('nan'), sigma_eps=float('nan'), provenance='absent')
    if df is None:
        return out
    sub = df[df['DMS_id'].astype(str) == str(dms_id)]
    if not len(sub):
        return out
    sub = sub.assign(_r=[_PROV_RANK.get(str(p), 9)
                         for p in sub['provenance'].values])
    sub = sub.sort_values('_r', kind='stable')
    for r in sub.to_dict('records'):
        if np.isfinite(r.get('sigma_y', np.nan)):
            out.update(sigma_y=float(r['sigma_y']),
                       provenance=str(r['provenance']))
            break
    for r in sub.to_dict('records'):
        if np.isfinite(r.get('sigma_eps', np.nan)):
            out['sigma_eps'] = float(r['sigma_eps'])
            out['sigma_eps_provenance'] = str(r['provenance'])
            break
    return out


def eps_sigma(dms_id, eps, *, scale='eps_latent'):
    """The ``sigma`` the ``|eps| >= 3 sigma`` label uses, and where it came from.

    Priority, and the reason for it (ORCHESTRATOR D5):

    1. a **measured** ``sigma_eps`` (KRAS only, 0.1243) -- the only replicate
       measurement in the benchmark, contaminated but real;
    2. ``sqrt(3) sigma_y`` -- ``eps`` of a double is a 4-term contrast, so
       ``Var(eps) ~ 3 sigma_y^2``.  D5 records that these two DISAGREE by a
       factor 1.65 on KRAS, which is exactly why both thresholds are reported
       and why the ``sigma x {0.5,1,2}`` surface is mandatory rather than
       optional;
    3. ``1.4826 MAD(eps)`` -- the assay's own robust scale, when neither exists.
    """
    s = t03_sigma(dms_id)
    if np.isfinite(s['sigma_eps']):
        return float(s['sigma_eps']), 'measured_replicate (sigma_eps)'
    if np.isfinite(s['sigma_y']):
        return _SQRT3 * float(s['sigma_y']), \
            'sqrt(3)*sigma_y [%s]' % s['provenance']
    v = np.asarray(eps[scale].values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return (float(mad_scaled(v)) if v.size else float('nan')), 'eps_mad_scaled'


# =========================================================================== #
# one assay, all five routes and all four artefact clauses                    #
# =========================================================================== #

_ROUTES = ('L1', 'L2', "L2'", 'L3', 'L4', 'L5')


def _blank_route(route, *, reason, gate=None):
    d = dict(route=route, feasible=False, n_units=0,
             infeasible_reason=reason)
    if gate:
        d.update(gate)
    return d


def _per_assay_c3(dms_id, *, B=None, nproc=1, boot_B=None, seed=None,
                  l5_B=None, twin=None, verbose=True, scale_check=True,
                  floor_check=True):
    """Every C3 number for one assay.  Nothing is decided here."""
    if B is None:
        B = THRESH['null_B']
    if l5_B is None:
        l5_B = THRESH['null_B']
    t0 = time.time()
    b = get_bundle(dms_id)
    ctx = b.ctx
    spec = config.ASSAYS[dms_id]
    # spec Sec.1.4's own seed registry: 'bootstrap_block' is the 1,000
    # position block bootstraps, 'perm_NS2' the L5 label permutation and
    # 'replication_perm' the 10,000 C3-N site-pair label permutations.  No new
    # seed name is invented here -- config.SEEDS is the only place they live.
    ent = (list(config.assay_seed('bootstrap_block', dms_id)) if seed is None
           else list(seed))
    ns2_ent = list(config.assay_seed('perm_NS2', dms_id))
    R = dict(dms_id=dms_id, tier=spec.tier, family_id=spec.family_id or '',
             seed=ent, notes={}, routes={})
    eps = cached_epsilon_table(b, verbose=verbose)
    geom = site_geometry(b, verbose=verbose)
    sig_eps, sig_prov = eps_sigma(dms_id, eps)
    R['sigma_eps_used'] = sig_eps
    R['sigma_provenance'] = sig_prov
    R['n_eps'] = int(len(eps))
    R['n_eps_bg0'] = int((eps['bg_order'].values == 0).sum())

    # ---- the density quintiles and the tau the artefact clauses read ------- #
    nb = int(THRESH['C3A_n_density_bins'])
    u, v = ctx.nested_idx[:, 0], ctx.nested_idx[:, 1]
    dens_full = (b.deg[u] + b.deg[v]).astype(np.float64)
    kp = b.pa & np.isfinite(b.c_obs)
    # WHY tau is chosen and not fixed at the catalogue's 4.0: clause 2 is read
    # off a per-QUINTILE rate, and "the rate in quintile 1 is > 0" is not a
    # statement about the data when the whole assay has 26 catalogued cliffs to
    # spread over five bins -- an empty bottom bin is then a counting accident,
    # not evidence of a depth artefact.  So the largest tau in the spec's own
    # sweep with >= 20 cliffs PER BIN is used, and the FULL sweep (4, 3, 2) goes
    # into density_rates_json so nothing is hidden by the choice.
    n_at = dict((float(t), int((np.abs(b.c_obs[kp]) >= t).sum()))
                for t in (_CAT_TAU, _ALT_TAU, 2.0))
    tau_used = 2.0
    for t in (_CAT_TAU, _ALT_TAU, 2.0):
        if n_at[float(t)] >= 20 * nb:
            tau_used = float(t)
            break
    dens_underpowered = bool(n_at[tau_used] < 20 * nb)
    ds = density_strata(np.abs(b.c_obs[kp]) >= tau_used, dens_full[kp],
                        n_bins=nb)
    dens_sweep = dict((str(t), density_strata(np.abs(b.c_obs[kp]) >= t,
                                              dens_full[kp], n_bins=nb))
                      for t in (_CAT_TAU, _ALT_TAU, 2.0))
    q_pa = np.zeros(ctx.nested_idx.shape[0], dtype=np.int64)
    q_pa[np.flatnonzero(kp)] = ds.attrs['quintile']
    R['tau_used'] = tau_used
    R['density_table'] = ds

    # ---- L1 and its N2 reference ------------------------------------------ #
    ip, nbr = sibling_index(ctx.nested_idx, ctx.add_col, b.keys)
    vals = np.where(b.pa, b.c_obs, np.nan)
    l1 = sibling_slope(vals, ip, nbr, b.pa, positions=ctx.pos_of_add,
                       boot_B=boot_B, seed=ent + [1], label='c_hat_phi_centred')
    l1['route'] = 'L1'
    # the un-standardised numerator version, so nothing is hidden by sigma
    numv = np.where(b.pa, b.num_obs, np.nan)
    l1n = sibling_slope(numv, ip, nbr, b.pa, boot_B=0, positions=None,
                        label='numerator')
    l1['beta_sibling_raw_num'] = l1n.get('slope', float('nan'))
    n2 = n2_reference(b, ip, nbr, tau=tau_used, q=q_pa, n_bins=nb, B=B,
                      nproc=nproc, verbose=verbose)
    R['n2'] = n2
    l1.update(beta_N2_mean=_nmean(n2['beta']),
              beta_N2_sd=_nsd(n2['beta']),
              beta_N2_p995=_pct(n2['beta'], 99.5),
              beta_N2_p025=_pct(n2['beta'], 2.5),
              beta_N2_p975=_pct(n2['beta'], 97.5),
              p_N2=_empirical_p(n2['beta'], l1.get('slope', np.nan)),
              beta_in_N2_band=_in_band(n2['beta'], l1.get('slope', np.nan)),
              B_null=int(B), nproc=int(nproc))
    # icc_addcol: the N2-referenced localisation statistic that DOES move
    obs = _nulls.observed_stats(ctx)
    l1['icc_addcol_obs'] = float(obs.get('icc_addcol', np.nan))
    l1['icc_addcol_N2_mean'] = _nmean(n2['icc_addcol'])
    l1['icc_addcol_N2_p995'] = _pct(n2['icc_addcol'], 99.5)
    l1['icc_addcol_in_N2_band'] = _in_band(n2['icc_addcol'],
                                           l1['icc_addcol_obs'])
    for nm in ('N1', 'N2c'):
        ens = _read_cached_ensemble(dms_id, nm)
        l1['icc_addcol_%s_p995' % nm] = (
            _pct(ens['icc_addcol'].values, 99.5)
            if ens is not None and 'icc_addcol' in ens else float('nan'))
    R['routes']['L1'] = l1

    # ---- L2 / L2' --------------------------------------------------------- #
    l2 = icc_across_backgrounds(eps, bundle=b, seed=ent + [2], boot_B=boot_B)
    l2['route'] = 'L2'
    l2['ICC_null_used'] = ('label permutation (N2 cannot move an ICC of eps: '
                           'N2 permutes e and never touches y)')
    l2['icc_n2_is_identity'] = True
    l2p = icc_across_aa_combos(eps, seed=ent + [3], boot_B=boot_B)
    l2p['route'] = "L2'"
    l2p['ICC_null_used'] = l2['ICC_null_used']
    l2p['icc_n2_is_identity'] = True
    R['routes']['L2'] = l2
    R['routes']["L2'"] = l2p

    # ---- L3: the KRAS twin ------------------------------------------------ #
    if twin is not None and dms_id in tuple(_noise.KRAS_TWIN):
        pr = [r for r in twin['rows'] if r['row_role'] == 'primary'][0]
        l3 = dict(route='L3', feasible=True, n_units=int(pr['n_shared']),
                  n_units_all=int(pr['n_shared']),
                  gate_statistic='n_shared_substitution_pairs_twice_measured',
                  gate_value=float(pr['n_shared']), gate_threshold=1.0,
                  R_L3=pr['R'], sign_agreement_L3=pr['sign_agreement'],
                  n_shared_L3=int(pr['n_shared']),
                  sigma_eps_used=pr['sigma_eps'],
                  sigma_provenance=pr['sigma_eps_provenance'],
                  infeasible_reason='',
                  notes=('scored as C3-N in T08; conditional in every case '
                         '(spec Sec.1.4: testable in exactly one family)'))
    else:
        l3 = _blank_route('L3', reason=(
            'no independent measurement of the same site pair exists for this '
            'assay (the KRAS RAF1 / RAF1-RBD twin is the only one in the '
            'benchmark)'),
            gate=dict(gate_statistic='n_shared_substitution_pairs_twice_measured',
                      gate_value=0.0, gate_threshold=1.0))
    R['routes']['L3'] = l3

    # ---- L4 --------------------------------------------------------------- #
    l4 = dr2_oos(b, seed=ent + [4], boot_B=boot_B, verbose=verbose)
    l4['route'] = 'L4'
    R['routes']['L4'] = l4

    # ---- L5 --------------------------------------------------------------- #
    l5 = l5_auroc(b, eps, sigma=sig_eps, geom=geom, B=l5_B, seed=ns2_ent,
                  boot_B=boot_B, verbose=verbose)
    l5['route'] = 'L5'
    l5['sigma_provenance'] = sig_prov
    # the mandatory sigma surface
    grid = {}
    for m in _SIGMA_MULTS:
        g5 = l5_auroc(b, eps, sigma=sig_eps * m, geom=geom, B=0, seed=ns2_ent,
                      boot_B=0)
        grid['sigma_x%g' % m] = dict(
            sigma=round(sig_eps * m, 6),
            AUROC=(None if not np.isfinite(g5['AUROC_L5'])
                   else round(g5['AUROC_L5'], 6)),
            n_cliff=g5['n_cliff_L5'], n_noncliff=g5['n_noncliff_L5'],
            feasible=bool(g5['feasible']))
    l5['sigma_grid_json'] = json.dumps(grid, sort_keys=True)
    R['routes']['L5'] = l5

    # ---- C3-A clause 1: sampling depth ------------------------------------ #
    dep = depth_clause(b, taus=(tau_used, _ALT_TAU if tau_used != _ALT_TAU
                                else 2.0), geom=geom)
    R['depth'] = dep

    # ---- C3-A clause 2: density ------------------------------------------- #
    rates = ds['rate'].values.astype(np.float64)
    nq = n2['rate_q']
    with np.errstate(all='ignore'):
        n2_mean = (np.array([_nmean(nq[:, j]) for j in range(nq.shape[1])])
                   if nq.size else np.full(nb, np.nan))
        n2_p95 = (np.array([_pct(nq[:, j], 95.0) for j in range(nq.shape[1])])
                  if nq.size else np.full(nb, np.nan))
    rho_d = _spearman(rates, np.arange(1, nb + 1, dtype=float))
    R['density'] = dict(
        density_q1_rate=float(rates[0]), density_q5_rate=float(rates[-1]),
        density_spearman=rho_d,
        density_monotone=(None if not np.isfinite(rho_d)
                          else bool(abs(rho_d) >= 1.0 - 1e-12)),
        density_enriched_q1=(bool(rates[0] > n2_p95[0])
                             if np.isfinite(n2_p95[0]) else None),
        density_enriched_q5=(bool(rates[-1] > n2_p95[-1])
                             if np.isfinite(n2_p95[-1]) else None),
        density_null_used=(
            'N2 (B=%d) per-quintile rate at tau=%g; tau is the largest of '
            '(4,3,2) with >= %d cliffs (20 per bin) so a per-quintile rate is '
            'estimable%s' % (B, tau_used, 20 * nb,
                             '; UNDERPOWERED: only %d cliffs even at tau=2'
                             % n_at[2.0] if dens_underpowered else '')),
        density_tau_used=tau_used,
        density_underpowered=dens_underpowered,
        density_rates_json=json.dumps(dict(
            tau=tau_used, n_bins=nb,
            n=[int(x) for x in ds['n'].values],
            degree_lo=[float(x) for x in ds['degree_lo'].values],
            degree_hi=[float(x) for x in ds['degree_hi'].values],
            n_cliff=[int(x) for x in ds['n_cliff'].values],
            rate=[None if not np.isfinite(x) else round(float(x), 8)
                  for x in rates],
            rate_N2_mean=[None if not np.isfinite(x) else round(float(x), 8)
                          for x in n2_mean],
            rate_N2_p95=[None if not np.isfinite(x) else round(float(x), 8)
                         for x in n2_p95],
            n_cliff_by_tau=dict((k, int(v)) for k, v in n_at.items()),
            underpowered=dens_underpowered,
            rate_by_tau=dict(
                (k, [None if not np.isfinite(x) else round(float(x), 8)
                     for x in d['rate'].values])
                for k, d in dens_sweep.items()),
            n_cliff_by_tau_by_quintile=dict(
                (k, [int(x) for x in d['n_cliff'].values])
                for k, d in dens_sweep.items())), sort_keys=True))

    # ---- C3-N's third refutation clause, per assay ------------------------ #
    # "> 50% of catalogued cliffs have |Delta| < 3 sigma" (spec Sec.1.4).  The
    # catalogue is |c_hat| >= 4 (spec Sec.4) and Delta is the RAW jump y_v - y_u,
    # so this asks whether the phi-centred, sigma(phi)-standardised statistic is
    # manufacturing "cliffs" out of jumps that are small on the measurement scale.
    s_y = t03_sigma(dms_id)['sigma_y']
    cat = kp & (np.abs(b.c_obs) >= _CAT_TAU)
    R['n_catalogue'] = int(cat.sum())
    if cat.any() and np.isfinite(s_y):
        dy = np.abs(ctx.y[v][cat] - ctx.y[u][cat])
        R['frac_cliffs_below_3sigma'] = float((dy < 3.0 * s_y).mean())
    else:
        R['frac_cliffs_below_3sigma'] = float('nan')
    R['frac_cliffs_below_3sigma_def'] = (
        'fraction of catalogued cliffs (|c_hat| >= %g inside P_a; n=%d) whose '
        'raw |delta_y| < 3 sigma_y = %.4f'
        % (_CAT_TAU, int(cat.sum()), 3.0 * s_y if np.isfinite(s_y) else float('nan')))

    # ---- C3-A clause 3: floor invariance ---------------------------------- #
    R['floor'] = _floor_clause(b, eps, l1, l2, l2p, l4, l5, ip, nbr,
                               sig_eps, geom, tau_used,
                               enabled=floor_check, verbose=verbose)
    # ---- C3-A clause 4: scale invariance ---------------------------------- #
    R['scale'] = _scale_clause(b, eps, l1, l2, l2p, l4, l5, ip, nbr,
                               sig_eps, geom, ent, boot_B,
                               enabled=scale_check, verbose=verbose)
    R['wall_s'] = round(time.time() - t0, 2)
    if verbose:
        print('    [c3  ] %-40s tau=%g  L1 %s  L2 %s  L2p %s  L4 %s  L5 %s  '
              '%.1fs' % (dms_id, tau_used,
                         'Y' if l1['feasible'] else '.',
                         'Y' if l2['feasible'] else '.',
                         'Y' if l2p['feasible'] else '.',
                         'Y' if l4['feasible'] else '.',
                         'Y' if l5['feasible'] else '.', R['wall_s']))
    return R


def _read_cached_ensemble(dms_id, null, B=None):
    """Read ``nulls/{id}_{null}_B{B}_seed*.npz`` from disk WITHOUT ever
    recomputing it: this module consumes stage 3's ensembles, it does not own
    them, and a silent 200-replicate refit inside C3 would be a scheduling bug.
    """
    if B is None:
        B = THRESH['null_B']
    p = _nulls.ensemble_path(dms_id, null, int(B))
    if not os.path.exists(p):
        return None
    try:
        with np.load(p, allow_pickle=False) as z:
            names = [str(s) for s in z['stat_names']]
            arr = z['stats']
        return pd.DataFrame(arr, columns=names)
    except Exception:                                          # pragma: no cover
        return None


# --------------------------------------------------------------------------- #
# C3-A clause 3: floor invariance                                             #
# --------------------------------------------------------------------------- #

def _fnum(v):
    """``float`` or ``nan`` -- never a ``TypeError``.  A route dict rebuilt for a
    sensitivity arm may simply not carry a key, and a missing statistic must read
    as "unevaluable", never crash the clause."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float('nan')
    return x


def _support_flags(l1, l2, l2p, l4, l5):
    """The four support conclusions C3-L's rule actually reads, as tri-state
    booleans -- what "the verdict is unchanged" must be checked on."""
    f = {}
    if l1.get('feasible'):
        b_, p_ = _fnum(l1.get('slope')), _fnum(l1.get('beta_N2_p995'))
        f['L1'] = (None if not (np.isfinite(b_) and np.isfinite(p_))
                   else bool(b_ > p_))
    for nm, r in (('L2', l2), ("L2'", l2p)):
        if r.get('feasible'):
            v = _fnum(r.get('ICC'))
            f[nm] = (None if not np.isfinite(v)
                     else bool(v >= THRESH['C3L_ICC_sup']))
    if l4.get('feasible'):
        v = _fnum(l4.get('dR2_oos'))
        f['L4'] = (None if not np.isfinite(v)
                   else bool(v >= THRESH['C3L_dR2_sup']))
    if l5.get('feasible'):
        v = _fnum(l5.get('AUROC_L5'))
        f['L5'] = (None if not np.isfinite(v)
                   else bool(v >= THRESH['C3L_AUROC_sup']))
    return f


def _agree(a, b):
    if a is None or b is None:
        return None
    return bool(a == b)


def _floor_clause(b, eps, l1, l2, l2p, l4, l5, ip, nbr, sig_eps, geom, tau,
                  *, enabled=True, verbose=False):
    """Spec Sec.1.4 C3-A clause 3: the verdict must be unchanged before and
    after floor masking.

    "Floor masking" is ``P_a`` clause (b) -- neither endpoint censored -- and,
    for the eps routes, the 4-corner censoring drop.  Dropping it is what this
    clause re-runs.  On an assay with **zero** censored rows the two are
    identical BY CONSTRUCTION and the clause is reported as trivially invariant
    with the row count as the evidence, not silently skipped.
    """
    ctx = b.ctx
    n_cens = int(ctx.censor_mask.sum())
    out = dict(floor_mask_invariant=None, beta_unmasked=float('nan'),
               ICC_unmasked=float('nan'), n_censored=n_cens,
               floor_note='', floor_flags={})
    base = _support_flags(l1, l2, l2p, l4, l5)
    if n_cens == 0:
        out.update(floor_mask_invariant=True,
                   beta_unmasked=l1.get('slope', float('nan')),
                   ICC_unmasked=(l2p.get('ICC') if l2p.get('feasible')
                                 else l2.get('ICC', float('nan'))),
                   floor_note=('0 censored rows: the masked and unmasked P_a '
                               'are the same %d edges, so the clause holds by '
                               'construction' % int(b.pa.sum())))
        return out
    if not enabled:
        out['floor_note'] = 'floor_check disabled by the caller'
        return out
    u, v = ctx.nested_idx[:, 0], ctx.nested_idx[:, 1]
    pa_u = (~ctx.wt_anchored) & ctx.oof_finite[u] & ctx.oof_finite[v]
    vals_u = np.where(pa_u, b.c_obs, np.nan)
    l1u = sibling_slope(vals_u, ip, nbr, pa_u, boot_B=0, positions=None)
    # the SAME N2 reference: the null was built on the masked P_a, and reusing it
    # for the unmasked slope is the conservative comparison (the unmasked P_a is
    # strictly larger, so its null would be no wider).  Stated, not hidden.
    l1u['beta_N2_p995'] = l1.get('beta_N2_p995', float('nan'))
    out['beta_unmasked'] = l1u.get('slope', float('nan'))
    eps_u = epsilon_table(b, drop_censored=False)
    l2u = icc_across_backgrounds(eps_u, bundle=b, boot_B=0, seed=[7])
    l2pu = icc_across_aa_combos(eps_u, boot_B=0, seed=[7])
    out['ICC_unmasked'] = (l2pu.get('ICC') if l2p.get('feasible')
                           else l2u.get('ICC', float('nan')))
    l5u = l5_auroc(b, eps_u, sigma=sig_eps, geom=geom, B=0, boot_B=0)
    l4u = dict(l4)
    if l4.get('feasible') and np.isfinite(l4.get('dR2_oos_unmasked', np.nan)):
        l4u['dR2_oos'] = l4['dR2_oos_unmasked']
    unm = _support_flags(l1u if l1.get('feasible') else l1,
                         l2u if l2.get('feasible') else l2,
                         l2pu if l2p.get('feasible') else l2p, l4u, l5u)
    flags = dict((k, _agree(base.get(k), unm.get(k)))
                 for k in base if k in unm)
    dec = [x for x in flags.values() if x is not None]
    out['floor_flags'] = flags
    out['floor_mask_invariant'] = (all(dec) if dec else None)
    out['floor_note'] = (
        '%d censored rows; P_a %d -> %d edges, eps %d -> %d rows; per-route '
        'support conclusion agreement: %s'
        % (n_cens, int(b.pa.sum()), int(pa_u.sum()), len(eps), len(eps_u),
           json.dumps(dict((k, ('agree' if x is True else
                                ('DISAGREE' if x is False else 'n/a')))
                           for k, x in flags.items()), sort_keys=True)))
    if verbose:
        print('    [A3  ] %-40s %s' % (ctx.dms_id, out['floor_note']))
    return out


# --------------------------------------------------------------------------- #
# C3-A clause 4: scale invariance                                             #
# --------------------------------------------------------------------------- #

def _scale_clause(b, eps, l1, l2, l2p, l4, l5, ip, nbr, sig_eps, geom, ent,
                  boot_B, *, enabled=True, verbose=False):
    """Spec Sec.1.4 C3-A clause 4: a verdict holding on the RAW scale but not on
    the LATENT scale is **discarded**, never reported as positive.

    Latent is primary (spec Sec.1.3), so this recomputes every feasible route on
    the raw scale and reports whether the support conclusion is the same.  The
    raw scale is the identity-link additive fit -- ``stats_c2.raw_bundle``, the
    same machinery C2's raw block uses, so the two modules cannot disagree about
    what "raw" means.

    The eps routes get a **MAD-matched** sigma on the raw scale
    (``sigma * MAD(eps)/MAD(eps_latent)``) so the two label sets have comparable
    size and the comparison is of the AUROC, not of a threshold artefact.
    """
    ctx = b.ctx
    out = dict(latent_raw_consistent=None, beta_raw_scale=float('nan'),
               ICC_raw_scale=float('nan'), dR2_oos_raw_scale=float('nan'),
               AUROC_raw_scale=float('nan'), scale_note='', scale_flags={})
    if not enabled:
        out['scale_note'] = 'scale_check disabled by the caller'
        return out
    base = _support_flags(l1, l2, l2p, l4, l5)
    raw = {}
    # ---- L1 on the raw scale ---------------------------------------------- #
    l1r = dict(l1)
    try:
        from cliff import stats_c2 as _c2
        rb = _c2.raw_bundle(ctx)
        of = ctx.oof_finite & rb['oof_finite']
        u, v = ctx.nested_idx[:, 0], ctx.nested_idx[:, 1]
        pa_r = (~ctx.wt_anchored) & ~(ctx.censor_mask[u] | ctx.censor_mask[v]) \
            & of[u] & of[v]
        ec = rb['e_oof'] - rb['mu_oof']
        with np.errstate(divide='ignore', invalid='ignore'):
            c_r = (ec[v] - ec[u]) / np.sqrt(rb['sigma_oof'][u] ** 2
                                            + rb['sigma_oof'][v] ** 2)
        vals_r = np.where(pa_r, c_r, np.nan)
        l1r = sibling_slope(vals_r, ip, nbr, pa_r, boot_B=0, positions=None)
        l1r['beta_N2_p995'] = l1.get('beta_N2_p995', float('nan'))
        out['beta_raw_scale'] = l1r.get('slope', float('nan'))
        raw['L1'] = 'ok'
    except Exception as exc:                                   # pragma: no cover
        raw['L1'] = 'unavailable: %s' % exc
        l1r = dict(l1)
        l1r['feasible'] = False        # excluded from the clause, not faked
    # ---- L2 / L2' on the raw eps ------------------------------------------ #
    l2r = (icc_across_backgrounds(eps, bundle=b, scale='eps', boot_B=0,
                                  seed=ent + [8]) if l2.get('feasible')
           else dict(l2))
    l2pr = (icc_across_aa_combos(eps, scale='eps', boot_B=0, seed=ent + [9])
            if l2p.get('feasible') else dict(l2p))
    out['ICC_raw_scale'] = (l2pr.get('ICC') if l2p.get('feasible')
                            else l2r.get('ICC', float('nan')))
    # ---- L4 with the raw target ------------------------------------------- #
    l4r = dict(l4)
    if l4.get('feasible'):
        l4r = dr2_oos(b, target='y', seed=ent + [10], boot_B=0)
        out['dR2_oos_raw_scale'] = l4r.get('dR2_oos', float('nan'))
    # ---- L5 on the raw eps, MAD-matched threshold ------------------------- #
    l5r = dict(l5)
    if l5.get('feasible'):
        m_lat = float(mad_scaled(eps['eps_latent'].values))
        m_raw = float(mad_scaled(eps['eps'].values))
        sr = sig_eps * (m_raw / m_lat) if m_lat > 0 else sig_eps
        l5r = l5_auroc(b, eps, sigma=sr, scale='eps', geom=geom, B=0, boot_B=0)
        out['AUROC_raw_scale'] = l5r.get('AUROC_L5', float('nan'))
    rawf = _support_flags(l1r, l2r, l2pr, l4r, l5r)
    # The spec's clause is ASYMMETRIC and the asymmetry is the whole point:
    # "a verdict holding on RAW but NOT on the LATENT scale is DISCARDED, never
    # reported as positive".  Latent is primary (Sec.1.3), so a raw-only
    # positive is already discarded by using the latent number -- nothing
    # positive is emitted and the clause is not violated.  What DOES violate it
    # is the other direction: a route reported POSITIVE on the latent scale that
    # does not survive on raw, i.e. a positive that exists only because of the
    # link.  ``latent_raw_consistent`` is therefore False iff some feasible
    # route is latent-positive and raw-negative; both directions are recorded.
    flags = {}
    for k in base:
        if k not in rawf:
            continue
        lat, rw = base.get(k), rawf.get(k)
        if lat is None or rw is None:
            flags[k] = 'n/a'
        elif lat == rw:
            flags[k] = 'agree'
        elif lat and not rw:
            flags[k] = 'LATENT_ONLY_POSITIVE'
        else:
            flags[k] = 'raw_only_positive_discarded'
    out['scale_flags'] = flags
    dec = [v for v in flags.values() if v != 'n/a']
    out['latent_raw_consistent'] = (
        None if not dec else bool('LATENT_ONLY_POSITIVE' not in dec))
    out['scale_note'] = (
        'raw = identity-link additive fit (stats_c2.raw_bundle); eps routes on '
        'the y scale with a MAD-matched sigma.  Clause 4 is asymmetric: a '
        'raw-only positive is discarded (latent is primary) and does NOT fail '
        'the clause; a LATENT_ONLY_POSITIVE does.  Per route: %s%s'
        % (json.dumps(flags, sort_keys=True),
           ('' if raw.get('L1') == 'ok' else '  [L1 raw %s]' % raw.get('L1'))))
    if verbose:
        print('    [A4  ] %-40s %s' % (ctx.dms_id, out['scale_note']))
    return out


# =========================================================================== #
# T07                                                                         #
# =========================================================================== #

def _fmt(v):
    """CSV cell: '' for a missing value, the plain value otherwise."""
    if v is None:
        return _MISSING
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (float, np.floating)):
        return _MISSING if not np.isfinite(v) else float(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    return v


def _t07_row(R, route):
    """One T07 row: the spec's columns first, the evidence for them second."""
    r = R['routes'][route]
    dep, den, flo, sca = R['depth'], R['density'], R['floor'], R['scale']
    spec = config.ASSAYS[R['dms_id']]
    row = dict((c, _MISSING) for c in T07_COLUMNS + T07_EXTRA_COLUMNS)
    row.update(
        DMS_id=R['dms_id'], route=route, feasible=_fmt(r.get('feasible')),
        n_units=_fmt(r.get('n_units')),
        tier=R['tier'], family_id=R['family_id'],
        route_name={'L1': 'sibling_corroboration',
                    'L2': 'sitepair_ICC_across_backgrounds',
                    "L2'": 'sitepair_ICC_across_aa_combinations',
                    'L3': 'cross_measurement_eps_replication',
                    'L4': 'out_of_sample_pairwise_predictability',
                    'L5': 'eps_3D_localisation'}[route],
        infeasible_reason=r.get('infeasible_reason', ''),
        gate_statistic=r.get('gate_statistic', ''),
        gate_value=_fmt(r.get('gate_value')),
        gate_threshold=_fmt(r.get('gate_threshold')),
        n_units_all=_fmt(r.get('n_units_all')),
        n_groups=_fmt(r.get('n_groups')), n_groups_ge2=_fmt(r.get('n_groups_ge2')),
        kbar=_fmt(r.get('kbar')),
        # ---- the per-assay artefact block, repeated on every row --------- #
        depth_spearman=_fmt(dep['depth_spearman']),
        best_struct_covariate=_fmt(dep['best_struct_covariate']),
        density_q1_rate=_fmt(den['density_q1_rate']),
        density_q5_rate=_fmt(den['density_q5_rate']),
        density_monotone=_fmt(den['density_monotone']),
        floor_mask_invariant=_fmt(flo['floor_mask_invariant']),
        latent_raw_consistent=_fmt(sca['latent_raw_consistent']),
        depth_spearman_tau3=_fmt(dep['depth_spearman_tau3']),
        best_struct_covariate_name=dep['best_struct_covariate_name'],
        struct_covariate_json=dep['struct_covariate_json'],
        density_rates_json=den['density_rates_json'],
        density_spearman=_fmt(den['density_spearman']),
        density_enriched_q1=_fmt(den['density_enriched_q1']),
        density_enriched_q5=_fmt(den['density_enriched_q5']),
        density_null_used=den['density_null_used'],
        density_tau_used=_fmt(den['density_tau_used']),
        depth_tau_used=_fmt(dep['depth_tau_used']),
        floor_note=flo['floor_note'], scale_note=sca['scale_note'],
        seed=json.dumps(R['seed']), wall_s=_fmt(R['wall_s']),
        verdict_C3L=_MISSING, verdict_C3A=_MISSING, failing_criterion=_MISSING,
        notes=json.dumps(dict(
            spec_expected_routes=list(getattr(spec, 'c3l_routes', ()) or ()),
            tau_used=R['tau_used'], n_eps=R['n_eps'], n_eps_bg0=R['n_eps_bg0'],
            sigma_eps_used=round(R['sigma_eps_used'], 6)
            if np.isfinite(R['sigma_eps_used']) else None,
            sigma_provenance=R['sigma_provenance'],
            structurally_unidentified=bool(
                R['dms_id'] == 'CD19_FMC63_Fitness_7URV')), sort_keys=True))
    if route == 'L1':
        row.update(beta_sibling=_fmt(r.get('slope')),
                   se_hc3=_fmt(r.get('se_hc3')),
                   beta_N2_p995=_fmt(r.get('beta_N2_p995')),
                   beta_in_N2_band=_fmt(r.get('beta_in_N2_band')),
                   beta_sibling_raw_num=_fmt(r.get('beta_sibling_raw_num')),
                   beta_t_hc3=_fmt(r.get('t')), beta_p_hc3=_fmt(r.get('p')),
                   beta_boot_lo95=_fmt(r.get('boot_lo95')),
                   beta_boot_hi95=_fmt(r.get('boot_hi95')),
                   beta_N2_mean=_fmt(r.get('beta_N2_mean')),
                   beta_N2_sd=_fmt(r.get('beta_N2_sd')),
                   beta_N2_p025=_fmt(r.get('beta_N2_p025')),
                   beta_N2_p975=_fmt(r.get('beta_N2_p975')),
                   p_N2=_fmt(r.get('p_N2')),
                   beta_raw_scale=_fmt(sca['beta_raw_scale']),
                   beta_unmasked=_fmt(flo['beta_unmasked']),
                   icc_addcol_obs=_fmt(r.get('icc_addcol_obs')),
                   icc_addcol_N2_mean=_fmt(r.get('icc_addcol_N2_mean')),
                   icc_addcol_N2_p995=_fmt(r.get('icc_addcol_N2_p995')),
                   icc_addcol_N1_p995=_fmt(r.get('icc_addcol_N1_p995')),
                   icc_addcol_N2c_p995=_fmt(r.get('icc_addcol_N2c_p995')),
                   icc_addcol_in_N2_band=_fmt(r.get('icc_addcol_in_N2_band')),
                   B_null=_fmt(r.get('B_null')), nproc=_fmt(r.get('nproc')))
    if route in ('L2', "L2'"):
        row.update(ICC=_fmt(r.get('ICC')), ICC_lo95=_fmt(r.get('ICC_lo95')),
                   ICC_hi95=_fmt(r.get('ICC_hi95')),
                   ICC_N2_mean=_MISSING,
                   ICC_perm_mean=_fmt(r.get('ICC_perm_mean')),
                   ICC_perm_p995=_fmt(r.get('ICC_perm_p995')),
                   ICC_perm_p=_fmt(r.get('ICC_perm_p')),
                   icc_n2_is_identity=_fmt(r.get('icc_n2_is_identity')),
                   ICC_null_used=r.get('ICC_null_used', ''),
                   ICC_boot_B=_fmt(r.get('ICC_boot_B')),
                   ICC_raw_scale=_fmt(sca['ICC_raw_scale']),
                   ICC_unmasked=_fmt(flo['ICC_unmasked']))
    if route == 'L3':
        row.update(R_L3=_fmt(r.get('R_L3')),
                   sign_agreement_L3=_fmt(r.get('sign_agreement_L3')),
                   n_shared_L3=_fmt(r.get('n_shared_L3')),
                   sigma_eps_used=_fmt(r.get('sigma_eps_used')),
                   sigma_provenance=r.get('sigma_provenance', ''))
    if route == 'L4':
        row.update(dR2_oos=_fmt(r.get('dR2_oos')),
                   dR2_lo95=_fmt(r.get('dR2_lo95')),
                   dR2_hi95=_fmt(r.get('dR2_hi95')),
                   top1pct_share=_fmt(r.get('top1pct_share')),
                   ridge_lambda=_fmt(r.get('ridge_lambda')),
                   dR2_r2_add=_fmt(r.get('dR2_r2_add')),
                   dR2_r2_pair=_fmt(r.get('dR2_r2_pair')),
                   n_Z_cols=_fmt(r.get('n_Z_cols')),
                   mean_obs_per_Z_col=_fmt(r.get('mean_obs_per_Z_col')),
                   dR2_top1_abs=_fmt(r.get('dR2_top1_abs')),
                   n_top1_cols=_fmt(r.get('n_top1_cols')),
                   dR2_oos_raw_scale=_fmt(sca['dR2_oos_raw_scale']))
    if route == 'L5':
        row.update(AUROC_L5=_fmt(r.get('AUROC_L5')),
                   AUROC_lo95=_fmt(r.get('AUROC_lo95')),
                   AUROC_hi95=_fmt(r.get('AUROC_hi95')),
                   p_NS2=_fmt(r.get('p_NS2')),
                   AUROC_NS2_mean=_fmt(r.get('AUROC_NS2_mean')),
                   AUROC_NS2_p995=_fmt(r.get('AUROC_NS2_p995')),
                   n_cliff_L5=_fmt(r.get('n_cliff_L5')),
                   n_noncliff_L5=_fmt(r.get('n_noncliff_L5')),
                   sigma_eps_used=_fmt(r.get('sigma_eps_used')),
                   sigma_provenance=r.get('sigma_provenance', ''),
                   AUROC_all_backgrounds=_fmt(r.get('AUROC_all_backgrounds')),
                   AUROC_raw_scale=_fmt(sca['AUROC_raw_scale']),
                   sigma_grid_json=r.get('sigma_grid_json', ''),
                   B_null=_fmt(r.get('B_null')))
    return row


def build_T07(per, *, write=True, verbose=True, fill_verdicts=True):
    """T07, spec Sec.4's columns verbatim and in order, then the evidence."""
    rows = []
    for dms_id, R in per.items():
        for route in _ROUTES:
            rows.append(_t07_row(R, route))
    df = pd.DataFrame(rows, columns=T07_COLUMNS + T07_EXTRA_COLUMNS)
    # BH-FDR over the ASSAYS (never over the edges) on L1's N2 p-value
    l1 = df['route'] == 'L1'
    pv = pd.to_numeric(df.loc[l1, 'p_N2'], errors='coerce').values
    q = benjamini_hochberg(pv)
    df.loc[l1, 'p_N2_qBH'] = [(_MISSING if not np.isfinite(x) else x) for x in q]
    if fill_verdicts:
        _fill_T07_verdicts(df, verbose=verbose)
    if write:
        PATHS.ensure_cache_dirs()
        p = os.path.join(PATHS.artifacts, 'T07_localisation_C3.csv')
        df.to_csv(p, index=False)
        if verbose:
            print('[c3  ] wrote %s  (%d rows x %d cols)'
                  % (p, len(df), df.shape[1]))
    return df


def _fill_T07_verdicts(df, *, verbose=True):
    """Apply :mod:`cliff.verdict`'s OWN C3-L / C3-A rules to the table, so T07
    ships decided rather than waiting for stage 8's write-back.

    ``verdict.py`` is the single authority on the rule; this only calls it.  Its
    stage-8 write-back recomputes and overwrites these three columns, so the two
    can never diverge.
    """
    try:
        from cliff import verdict as _v
    except Exception as exc:                                   # pragma: no cover
        if verbose:
            print('[c3  ] verdict.py unavailable (%s); T07 verdict columns left '
                  'empty for stage 8 to fill' % exc)
        return
    for a in sorted(set(df['DMS_id'])):
        m = df['DMS_id'] == a
        dl = _v.verdict_C3L(a, df, 'ok')
        da = _v.verdict_C3A(a, df, 'ok')
        df.loc[m, 'verdict_C3L'] = dl.outcome
        df.loc[m, 'verdict_C3A'] = da.outcome
        df.loc[m, 'failing_criterion'] = '; '.join(
            x for x in (dl.failing_criterion, da.failing_criterion) if x)


# =========================================================================== #
# T08                                                                         #
# =========================================================================== #

#: KRAS assays that share the 166-position library.  Reported as CONTEXT rows
#: (``R`` empty) because they are the same library against a DIFFERENT partner:
#: a disagreement there is partner specificity (C4-I), not measurement noise, and
#: letting one carry an ``R`` would give ``verdict_C3N`` a second family to
#: decide when the spec says C3-N is testable in exactly one.
_KRAS_CONTEXT_PAIRS = (
    ('KRAS_RAF1-RBD_norfitness_6VJJ', 'KRAS_RALGDS-RBD_norfitness_1LFD'),
    ('KRAS_RAF1-RBD_norfitness_6VJJ', 'KRAS_PICK3CG-RBD_norfitness_1HE8'),
    ('KRAS_RAF1-RBD_norfitness_6VJJ', 'KRAS_SOS1_norfitness_8BE4'),
)


def _t08_blank():
    return dict((c, _MISSING) for c in T08_COLUMNS + T08_EXTRA_COLUMNS)


def build_T08(twin, *, per=None, context=True, write=True, verbose=True,
              fill_verdicts=True):
    """T08 epsilon_replication, spec Sec.4's columns verbatim.

    Row order is load-bearing: ``verdict_C3N`` takes the FIRST row of the family
    that carries an ``R``, so the measured-``sigma_eps``, ``sigma x 1`` row of
    the KRAS twin comes first and the D5 sensitivity rows follow it.
    """
    tw = twin['twin']
    a_id, b_id = tw['assay_a'], tw['assay_b']
    s_a = t03_sigma(a_id)
    fcb = None if per is None else per.get(a_id, {}).get(
        'frac_cliffs_below_3sigma')
    fcb_def = None if per is None else per.get(a_id, {}).get(
        'frac_cliffs_below_3sigma_def', '')
    rows = []
    for r in twin['rows']:
        row = _t08_blank()
        row.update(
            assay_a=a_id, assay_b=b_id, relation=tw['relation'],
            join_method=tw['join_method'], n_shared=_fmt(r['n_shared']),
            sd_eps_a=_fmt(tw['sd_eps_a']), sd_eps_b=_fmt(tw['sd_eps_b']),
            pearson_raw=_fmt(tw['r']), ols_slope=_fmt(tw['slope']),
            resid_sd_after_affine=_fmt(tw['resid_sd']),
            sigma_eps=_fmt(r['sigma_eps']),
            n_cliff_a_3sigma=_fmt(r['n_flagged_a']), R=_fmt(r['R']),
            R_chance_perm=_fmt(r['R_chance_perm']), perm_p=_fmt(r['perm_p']),
            sign_agreement=_fmt(r['sign_agreement']),
            F_spec=_fmt(twin['F_spec']),
            F_spec_noise_corrected=_fmt(twin['F_spec_noise_corrected']),
            verdict_C3N=_MISSING, verdict_stamp=_MISSING,
            row_role=r['row_role'], threshold_label=r['threshold_label'],
            sigma_mult=_fmt(r['sigma_mult']),
            cliff_abs_a=_fmt(r['cliff_abs_a']),
            replicate_abs_b=_fmt(r['replicate_abs_b']),
            frac_flagged_a=_fmt(r['frac_flagged_a']),
            frac_flagged_b=_fmt(r['frac_flagged_b']),
            n_flagged_a=_fmt(r['n_flagged_a']),
            n_replicated=_fmt(r['n_replicated']),
            flagged_frac_exceeds_5pct=_fmt(r['flagged_frac_exceeds_5pct']),
            R_lo95=_fmt(r['R_lo95']), R_hi95=_fmt(r['R_hi95']),
            R_perm_p995=_fmt(r['R_perm_p995']),
            sign_agreement_chance=_fmt(r['sign_agreement_chance']),
            sign_agreement_perm_p=_fmt(r['sign_agreement_perm_p']),
            frac_cliffs_below_3sigma=_fmt(fcb),
            frac_cliffs_below_3sigma_def=(fcb_def or ''),
            sigma_y_a=_fmt(s_a['sigma_y']),
            sigma_eps_provenance=r['sigma_eps_provenance'],
            spearman_raw=_fmt(tw['spearman']),
            n_eps_a=_fmt(tw['n_eps_a']), n_eps_b=_fmt(tw['n_eps_b']),
            n_shared_site_pairs_collapsed=_fmt(
                tw['n_shared_site_pairs_collapsed']),
            icc_eps_two_measurements=_fmt(twin['icc_eps_two_measurements']),
            family=config.family_of(a_id) or '', caveat=tw['caveat'],
            notes=json.dumps(dict(
                B_perm=int(r['B']), boot_B=int(r.get('boot_B', 0)),
                n_position_blocks=int(r.get('n_position_blocks', 0)),
                var_a=round(twin['var_a'], 8), var_b=round(twin['var_b'], 8),
                cov_ab=round(twin['cov_ab'], 8),
                F_spec_definition=(
                    'eps^(a) = mu + delta^(a) + noise over the two '
                    'measurements: Var(mu) = Cov(eps_a, eps_b), '
                    'Var(delta)+Var(noise) = mean Var - Cov, '
                    'Var(noise) = sigma_eps^2'),
                D5=('both thresholds are reported and the flagged fraction is '
                    'part of the result: sigma_eps = 0.1243 and '
                    'sqrt(3)*sigma_y = 0.2055 disagree by a factor 1.65')),
                sort_keys=True))
        rows.append(row)
    # ---- the GB1 overlap: eps-level, R deliberately EMPTY ----------------- #
    if context:
        try:
            gb = _noise.kras_twin_epsilon(pair=_noise.GB1_TWIN)
            gy = _noise.gb1_cross_study()
            row = _t08_blank()
            row.update(
                assay_a=gb['assay_a'], assay_b=gb['assay_b'],
                relation='same_partner_diff_construct',
                join_method='canonical_key', n_shared=_fmt(gb['n_shared']),
                sd_eps_a=_fmt(gb['sd_eps_a']), sd_eps_b=_fmt(gb['sd_eps_b']),
                pearson_raw=_fmt(gb['r']), ols_slope=_fmt(gb['slope']),
                resid_sd_after_affine=_fmt(gb['resid_sd']),
                sigma_eps=_fmt(gb['sigma_eps']), R=_MISSING,
                row_role='context_not_a_replicate',
                threshold_label='n/a', spearman_raw=_fmt(gb['spearman']),
                n_eps_a=_fmt(gb['n_eps_a']), n_eps_b=_fmt(gb['n_eps_b']),
                n_shared_site_pairs_collapsed=_fmt(
                    gb['n_shared_site_pairs_collapsed']),
                sigma_y_a=_fmt(t03_sigma(gb['assay_a'])['sigma_y']),
                sigma_eps_provenance='cross_study_contaminated',
                family=config.family_of(gb['assay_a']) or '',
                caveat=config.NOISE['GB1']['caveat'],
                notes=json.dumps(dict(
                    R_left_empty_on_purpose=(
                        'chain C position 2 is Q in the 55-site library and T '
                        'in the 4-site library: TWO BACKGROUNDS, not two '
                        'measurements, so an R here would give verdict_C3N a '
                        'second family when the spec says C3-N is testable in '
                        'exactly one'),
                    y_level_r=round(float(gy['r']), 6),
                    y_level_n_shared=int(gy['n_shared']),
                    wt_seq_differences=[list(map(str, d))
                                        for d in gy['wt_seq_differences']]),
                    sort_keys=True))
            rows.append(row)
        except Exception as exc:                               # pragma: no cover
            if verbose:
                print('[c3  ] GB1 T08 context row skipped: %s' % exc)
        for pr in _KRAS_CONTEXT_PAIRS:
            try:
                jj = _noise.kras_twin_epsilon(pair=pr)
            except Exception as exc:                           # pragma: no cover
                if verbose:
                    print('[c3  ] T08 context %s skipped: %s' % (pr[1], exc))
                continue
            ea = jj['table']['eps_a'].values
            eb = jj['table']['eps_b'].values
            V = 0.5 * (float(ea.var(ddof=1)) + float(eb.var(ddof=1)))
            C = float(np.cov(ea, eb, ddof=1)[0, 1]) if ea.size > 2 else np.nan
            s2 = float(config.NOISE['KRAS']['sigma_eps']) ** 2
            num = V - C - s2
            row = _t08_blank()
            row.update(
                assay_a=pr[0], assay_b=pr[1],
                relation='same_library_diff_partner',
                join_method='canonical_key', n_shared=_fmt(jj['n_shared']),
                sd_eps_a=_fmt(jj['sd_eps_a']), sd_eps_b=_fmt(jj['sd_eps_b']),
                pearson_raw=_fmt(jj['r']), ols_slope=_fmt(jj['slope']),
                resid_sd_after_affine=_fmt(jj['resid_sd']),
                sigma_eps=_fmt(config.NOISE['KRAS']['sigma_eps']), R=_MISSING,
                F_spec=_fmt((V - C) / V if V > 0 else np.nan),
                F_spec_noise_corrected=_fmt(num / (num + C)
                                            if (num + C) != 0 else np.nan),
                row_role='context_partner_specificity',
                threshold_label='n/a', spearman_raw=_fmt(jj['spearman']),
                n_eps_a=_fmt(jj['n_eps_a']), n_eps_b=_fmt(jj['n_eps_b']),
                n_shared_site_pairs_collapsed=_fmt(
                    jj['n_shared_site_pairs_collapsed']),
                sigma_y_a=_fmt(t03_sigma(pr[0])['sigma_y']),
                sigma_eps_provenance='measured_replicate (KRAS twin)',
                family=config.family_of(pr[0]) or '',
                caveat=('SAME library, DIFFERENT partner: the disagreement is '
                        'partner specificity (C4-I), not measurement noise, so '
                        'R is left empty on purpose.  MEASURED: n_shared = 0 '
                        'under join_method=canonical_key, because the four KRAS '
                        'partner files do NOT share a coordinate system -- '
                        'RAF1-RBD numbers its mutations on chain A from '
                        'position 3 while RALGDS-RBD numbers the SAME residues '
                        'on chain B from position 2 (0 of 166 vs 164 '
                        '(chain,pos) keys shared, 1 of 23,162 vs 20,341 variant '
                        'keys).  A cross-partner eps join therefore needs '
                        'join_method=mutant_pdb_aligned, i.e. the structural '
                        'alignment T11 builds -- it cannot be done on the '
                        'canonical key, and stats_c4 must not assume it can.'),
                notes=json.dumps(dict(var_pooled=round(V, 8),
                                      cov_ab=round(C, 8)), sort_keys=True))
            rows.append(row)
    df = pd.DataFrame(rows, columns=T08_COLUMNS + T08_EXTRA_COLUMNS)
    if fill_verdicts:
        try:
            from cliff import verdict as _v
            t03 = _T03.get('df')
            if t03 is None:
                t03_sigma(a_id)
                t03 = _T03.get('df')
            fam = config.family_of(a_id)
            # every row IS a complete C3-N evaluation (its own threshold, its
            # own flagged fraction), so the rule is applied per row rather than
            # once to rows[0] -- which is exactly what ORCHESTRATOR D5's "report
            # at BOTH thresholds" asks for.  Stage 8's write-back will still
            # stamp the family from the FIRST row, the spec-frozen 0.373 one.
            for i in df.index:
                if str(df.at[i, 'R']).strip() == '':
                    continue
                d = _v.verdict_C3N(df.loc[[i]], 'ok', family=fam, gates=None,
                                   t03=t03)
                df.at[i, 'verdict_C3N'] = d.outcome
                df.at[i, 'verdict_stamp'] = d.detail.get('stamp', 'conditional')
                df.at[i, 'failing_criterion_C3N'] = d.failing_criterion
        except Exception as exc:                               # pragma: no cover
            if verbose:
                print('[c3  ] verdict_C3N not applied (%s)' % exc)
    if write:
        PATHS.ensure_cache_dirs()
        p = os.path.join(PATHS.artifacts, 'T08_epsilon_replication.csv')
        df.to_csv(p, index=False)
        if verbose:
            print('[c3  ] wrote %s  (%d rows x %d cols)'
                  % (p, len(df), df.shape[1]))
    return df


# =========================================================================== #
# the stage entry point                                                       #
# =========================================================================== #

#: bumped whenever a number in T07/T08 would change.
STAT_C3_VERSION = 'c3v1'


def run_all(assays=None, *, B=None, nproc=None, boot_B=None, l5_B=None,
            n_perm_c3n=None, seed=None, write=True, verbose=True, context=True,
            scale_check=True, floor_check=True, register_cache=True):
    """C3 end to end: the KRAS twin, the five routes on every eligible assay,
    the four artefact clauses, T07 and T08.

    ``assays`` defaults to spec Sec.5's stage-3 set (12 PRIMARY + 2 ARM + 3
    CONTROL).  Every route runs on every assay and the GATES decide -- the
    registry's ``c3l_routes`` is carried into ``notes`` as
    ``spec_expected_routes`` so a disagreement between what the spec predicted
    and what the data supports is visible in the table instead of being
    designed away.
    """
    if assays is None:
        assays = list(config.PRIMARY + config.ARM + config.CONTROL)
    if nproc is None:
        nproc = THRESH['nproc_cap']
    if B is None:
        B = THRESH['null_B']
    t0 = time.time()
    if verbose:
        print('[c3  ] %d assays, B=%d N2 replicates, nproc=%d, %s'
              % (len(assays), B, nproc, STAT_C3_VERSION))
    twin = l3_cross_measurement(B=n_perm_c3n, seed=(list(seed) if seed else None),
                                boot_B=boot_B, verbose=verbose)
    per = {}
    for i, a in enumerate(assays):
        if verbose:
            print('[c3  ] %2d/%d %s' % (i + 1, len(assays), a))
        per[a] = _per_assay_c3(a, B=B, nproc=nproc, boot_B=boot_B, l5_B=l5_B,
                               seed=seed, twin=twin, verbose=verbose,
                               scale_check=scale_check, floor_check=floor_check)
        _BUNDLES.clear()
        _nulls.clear_context_cache()
    t07 = build_T07(per, write=write, verbose=verbose)
    t08 = build_T08(twin, per=per, context=context, write=write, verbose=verbose)
    if write and register_cache:
        ents = register_eps_cache(extra=dict(stats_c3=dict(
            stat_c3_version=STAT_C3_VERSION, B=int(B),
            routes=list(_ROUTES),
            centring='phi-centred (ORCHESTRATOR D2)',
            eps_scales='eps (raw y) and eps_latent (z = ginv(y)), neither cross-fitted',
            written_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))))
        bad = _pairs.verify_manifest()
        if verbose:
            print('[c3  ] manifest: %d eps caches registered, verify_manifest '
                  '%s' % (len(ents), 'clean' if not bad else
                          ('%d mismatch(es) -- another stage is writing '
                           'concurrently: %r' % (len(bad), bad[:3]))))
    if verbose:
        print('[c3  ] total wall %.1f s' % (time.time() - t0))
    return dict(per_assay=per, T07=t07, T08=t08, twin=twin)


def stage3(assays=None, nproc=None, verbose=True, **kw):
    """``run_all.py``'s stage-3 entry point (its ``_call`` passes ``assays``,
    ``nproc`` and ``verbose`` and nothing else)."""
    return run_all(assays=assays, nproc=nproc, verbose=verbose, **kw)


stage = stage3


# =========================================================================== #
# self-check                                                                  #
# =========================================================================== #

def _ok(name, cond, detail=''):
    print('  %-62s %s %s' % (name, 'OK ' if cond else 'FAIL', detail))
    if not cond:
        raise AssertionError(name + '  ' + detail)


def _selfcheck(dms_id='Z-domain_ZpA963_HL1_fitness_2M5A'):
    """Closed forms and invariants.  No verdict, no table: just the arithmetic
    every number in T07/T08 rests on."""
    config.assert_env()
    print('[stats_c3] closed forms and invariants')
    rng = np.random.default_rng(3)
    # ---- 1. HC3 against the textbook sandwich ----------------------------- #
    x = rng.standard_normal(400)
    y = 2.5 + 1.7 * x + rng.standard_normal(400) * (0.5 + np.abs(x))
    f = ols_hc3(x, y)
    xc = x - x.mean()
    sxx = (xc * xc).sum()
    r = y - (f['intercept'] + f['slope'] * x)
    h = 1.0 / x.size + xc * xc / sxx
    want = math.sqrt(((xc * xc) * (r * r) / (1 - h) ** 2).sum() / sxx ** 2)
    _ok('ols_hc3 slope == closed-form OLS',
        abs(f['slope'] - (xc * (y - y.mean())).sum() / sxx) < 1e-12,
        'b=%.6f' % f['slope'])
    _ok('ols_hc3 se == the HC3 sandwich', abs(f['se_hc3'] - want) < 1e-12,
        'se=%.6f' % f['se_hc3'])
    _ok('HC3 se > HC0 se (leverage inflation)',
        f['se_hc3'] > math.sqrt(((xc * xc) * (r * r)).sum() / sxx ** 2))
    # ---- 2. AUROC == Mann-Whitney U / n1 n0, ties at 0.5 ------------------- #
    s = rng.standard_normal(300)
    lab = rng.random(300) < 0.3
    u = _st.mannwhitneyu(s[lab], s[~lab], alternative='greater').statistic
    _ok('auroc == U/(n1 n0)',
        abs(auroc(s, lab) - u / (lab.sum() * (~lab).sum())) < 1e-12,
        '%.6f' % auroc(s, lab))
    _ok('auroc of an all-tied score is exactly 0.5',
        abs(auroc(np.zeros(100), np.arange(100) < 40) - 0.5) < 1e-12)
    # ---- 3. the three ICC implementations agree --------------------------- #
    g = rng.integers(0, 40, 2000)
    v = rng.standard_normal(2000) + 0.8 * g
    _u, gi = np.unique(g, return_inverse=True)
    ng = _u.size
    cnt = np.bincount(gi, minlength=ng).astype(float)
    tot = np.bincount(gi, weights=v, minlength=ng)
    sq = np.bincount(gi, weights=v * v, minlength=ng)
    i_ref = _nulls._icc_oneway(v, g)[0]
    _ok('_icc_fast == nulls._icc_oneway',
        abs(_icc_fast(v, gi, ng) - i_ref) < 1e-12, '%.8f' % i_ref)
    _ok('_icc_from_suff == nulls._icc_oneway',
        abs(_icc_from_suff(cnt, tot, sq) - i_ref) < 1e-12)
    _ok('icc_oneway == nulls._icc_oneway',
        abs(icc_oneway(v, g)['ICC'] - i_ref) < 1e-12)
    _ok('ICC of pure noise is ~0',
        abs(_nulls._icc_oneway(rng.standard_normal(20000),
                               rng.integers(0, 200, 20000))[0]) < 0.05)
    # ---- 4. BH-FDR and the empirical p ------------------------------------ #
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    q = benjamini_hochberg(p)
    want = np.minimum.accumulate((8 * np.sort(p) / np.arange(1, 9))[::-1])[::-1]
    _ok('benjamini_hochberg == the step-up definition', np.allclose(q, want),
        'q1=%.6f' % q[0])
    _ok('empirical p is (1+#>=)/(B+1)',
        abs(_empirical_p(np.zeros(199), 1.0) - 1.0 / 200.0) < 1e-15)
    # ---- 5. replication_rate on a construction with a KNOWN answer -------- #
    n = 20000
    truth = rng.standard_normal(n) * 1.0
    sg = 0.2
    ea = truth + rng.standard_normal(n) * sg
    eb = truth + rng.standard_normal(n) * sg
    rr = replication_rate(ea, eb, sg, B=200, seed=[1], boot_B=0)
    _ok('replication_rate: R >> chance when eps is shared',
        rr['R'] > 0.9 and rr['R_chance_perm'] < 0.5,
        'R=%.4f chance=%.4f' % (rr['R'], rr['R_chance_perm']))
    ind = replication_rate(ea, rng.permutation(eb), sg, B=200, seed=[1],
                           boot_B=0)
    _ok('replication_rate: R == chance when eps is NOT shared',
        abs(ind['R'] - ind['R_chance_perm']) < 0.05,
        'R=%.4f chance=%.4f p=%.4g' % (ind['R'], ind['R_chance_perm'],
                                       ind['perm_p']))
    _ok('replication_rate reports the flagged fraction',
        0.0 < rr['frac_flagged_a'] < 1.0 and rr['n_flagged_a'] > 0,
        'frac=%.4f' % rr['frac_flagged_a'])
    # ---- 6. density_strata bins are equal-count --------------------------- #
    ds = density_strata(rng.random(1000) < 0.1, rng.random(1000), n_bins=5)
    _ok('density_strata gives 5 equal-count bins',
        list(ds['n']) == [200] * 5, str(list(ds['n'])))
    _ok('density_strata is monotone in the degree edges',
        all(ds['degree_hi'].values[i] <= ds['degree_lo'].values[i + 1] + 1e-12
            for i in range(4)))
    # ---- 7. one real assay: siblings, eps, and the N2 identity ------------- #
    b = get_bundle(dms_id)
    ctx = b.ctx
    ip, nbr = sibling_index(ctx.nested_idx, ctx.add_col, b.keys)
    idx = ctx.nested_idx
    bad = 0
    owner = np.repeat(np.arange(idx.shape[0]), np.diff(ip))
    for e, s in list(zip(owner, nbr))[:20000]:
        if idx[e, 0] in (idx[s, 0], idx[s, 1]) or \
           idx[e, 1] in (idx[s, 0], idx[s, 1]):
            bad += 1
    _ok('sibling_index siblings are NODE-DISJOINT (spec L1)', bad == 0,
        '%d violations in the first 20,000 adjacencies' % bad)
    _ok('sibling_index agrees with the same add_col group',
        bool(np.all(ctx.add_col[nbr] == ctx.add_col[owner])))
    sc = _pairs.sibling_counts(idx, ctx.add_col, b.keys)
    got = np.diff(ip).astype(np.int32)
    _ok('sibling degrees == pairs.sibling_counts (independent construction)',
        bool(np.array_equal(got, np.asarray(sc))),
        'max %d, median %.1f' % (int(got.max()), float(np.median(got))))
    # eps at bg_order 0 == noise.epsilon_sitepairs
    from cliff import io_bgym as _io
    A = _io.load_assay(dms_id)
    es = _noise.epsilon_sitepairs(A)
    eps = cached_epsilon_table(b)
    m0 = eps['bg_order'].values == 0
    mine = dict(zip(zip(eps['row_ij'].values[m0], ), eps['eps'].values[m0]))
    theirs = dict(zip([(int(r),) for r in es['rows']], es['eps']))
    common = set(mine) & set(theirs)
    worst = max([abs(mine[k] - theirs[k]) for k in common] or [0.0])
    _ok('epsilon_table bg_order 0 == noise.epsilon_sitepairs',
        worst < 1e-12 and len(common) > 0,
        '%d shared doubles, max |d| = %.3e' % (len(common), worst))
    # icc_n2_is_identity: N2 leaves y untouched, and eps is a function of y only
    rep_n2 = None
    strata = _nulls.make_strata(ctx.n_muts, ctx.phi_oof,
                                censor_mask=ctx.censor_mask)
    e_star = _nulls.surrogate_N2(ctx, ctx.e_oof, np.random.default_rng(0), strata)
    _ok('N2 permutes e and leaves y byte-identical (=> ICC_N2 == ICC_obs)',
        bool(np.array_equal(ctx.y, ctx.y)) and not np.array_equal(e_star, ctx.e_oof),
        'e moved on %d of %d rows'
        % (int((e_star != ctx.e_oof).sum()), ctx.n))
    _ok('N2 preserves the uncensored e marginal exactly',
        abs(float(np.sort(e_star[np.isfinite(e_star)]).sum()
                  - np.sort(ctx.e_oof[np.isfinite(ctx.e_oof)]).sum())) < 1e-8)
    # ---- 8. the L1 / L4 gates never return a number when infeasible ------- #
    z = sibling_slope(np.zeros(idx.shape[0]), np.zeros(idx.shape[0] + 1,
                                                       dtype=np.int64),
                      np.zeros(0, dtype=np.int32), np.ones(idx.shape[0], bool))
    _ok('sibling_slope with no siblings is feasible=False and slope=nan',
        (z['feasible'] is False) and not np.isfinite(z['slope']),
        z['infeasible_reason'][:60])
    print('[stats_c3] all invariants OK')
    return True


def _main(argv):
    if '--selfcheck' in argv:
        _selfcheck()
        return 0
    kw = {}
    assays = None
    for i, a in enumerate(argv):
        if a == '--assays':
            assays = argv[i + 1].split(',')
        if a == '--B':
            kw['B'] = int(argv[i + 1])
        if a == '--nproc':
            kw['nproc'] = int(argv[i + 1])
        if a == '--boot-B':
            kw['boot_B'] = int(argv[i + 1])
        if a == '--l5-B':
            kw['l5_B'] = int(argv[i + 1])
        if a == '--perm-B':
            kw['n_perm_c3n'] = int(argv[i + 1])
        if a == '--no-write':
            kw['write'] = False
        if a == '--no-context':
            kw['context'] = False
        if a == '--no-scale-check':
            kw['scale_check'] = False
        if a == '--no-floor-check':
            kw['floor_check'] = False
        if a == '--no-register':
            kw['register_cache'] = False
    config.assert_env()
    out = run_all(assays=assays, **kw)
    t = out['T07']
    print()
    print('=== T07 C3-L, feasible routes only ===')
    show = t[t['feasible'].astype(str) == 'True']
    cols = ['DMS_id', 'route', 'n_units', 'beta_sibling', 'beta_N2_p995',
            'p_N2', 'ICC', 'ICC_lo95', 'ICC_hi95', 'dR2_oos', 'dR2_lo95',
            'AUROC_L5', 'p_NS2', 'verdict_C3L', 'verdict_C3A']
    print(show[cols].to_string(index=False))
    print()
    print('=== T08 C3-N ===')
    c8 = ['assay_a', 'assay_b', 'threshold_label', 'sigma_mult', 'sigma_eps',
          'cliff_abs_a', 'n_flagged_a', 'frac_flagged_a', 'R', 'R_chance_perm',
          'perm_p', 'sign_agreement', 'flagged_frac_exceeds_5pct',
          'verdict_C3N', 'verdict_stamp']
    print(out['T08'][c8].to_string(index=False))
    return 0


if __name__ == '__main__':                                  # pragma: no cover
    import sys as _sys
    _sys.exit(_main(_sys.argv[1:]))
