"""BGYM-CLIFF v1 -- stage 6: C4, "is *interaction* cliff the right name?"
(spec Sec.1.5).  INTERPRETATION ONLY: nothing here gates C1-C3.

Three sub-tests, each with its own null, plus the G11 twin-structure control:

* **C4-S** site level, burial-matched.  Poisson/binomial GLM by IRLS (no
  statsmodels), null **NS1**.  ORCHESTRATOR **D6**: the interface definition runs
  BOTH ways, co-primary -- the pre-registered ``min heavy-atom distance < 5.0 A``
  flag AND ``dSASA > 1 A^2`` (which on this benchmark is *identical* to
  ``dSASA > 0`` and to Levy ``support+rim+core``, so that cut is NON-BINDING and
  is never sold as robustness).  Kill switch: REFUTED if ``beta_iface`` loses
  significance once ``rsa_iso`` enters.
* **C4-P** pair level = route L5's AUROC of ``-d3d_min_heavy`` separating
  ``|eps| >= 3 sigma`` from ``|eps| < 1 sigma``, null **NS2**.
* **C4-I** partner specificity, double-centred (the actual interaction test):
  ``Z_jk = logit(per-position cliff rate)``, ``W_jk = -min heavy-atom distance
  from j to partner k``, ``M_F = corr(Z~, W~)``, null **NS3**; plus ``F_spec``
  from ``eps^(a) = mu + delta^(a) + noise`` with ``Var(noise) = sigma_eps^2``
  subtracted, ``PSI_j``, and the REQUIRED fold-axis validation.
* **G11**: the KRAS score table is registered against two complexes (8BE4/SOS1
  and 5O2S/DARPinK27) with byte-identical scores, so at most ONE of the two
  interface localisations can be causal.

Writes ``T10_structure_pairs.csv``, ``T11_partner_specificity.csv`` and the
CLIFF-DERIVED columns of ``T09_structure_sites.csv`` (appended in place; the
structural columns stage 1 wrote are never touched).  Verdicts are emitted only
by ``cliff/verdict.py``; this module writes numbers.

Frozen definitions, all of them declared here rather than buried in the code
-------------------------------------------------------------------------
1. **cliff flag on nested pairs** = ``|c_hat| >= TAU_PRIMARY`` over the primary
   nested set ``P_a``, with ``c_hat`` phi-centred (ORCHESTRATOR D2) and taken
   from ``cliff.nulls`` -- never re-derived here.  ``TAU_PRIMARY = 3.0``, the
   study's cliff convention everywhere it is named (C3-N ``|eps| >= 3 sigma_eps``,
   L5 ``|eps| >= 3 sigma``) and the low end of C2's verdict window ``[3, 8]``.
   ``TAU_SECONDARY = 4.0`` (``THRESH['C2_catalogue_c_min']``, the C2 catalogue
   cut) is carried in its own columns so no conclusion rests on one cut.
2. **per-position exposure** ``n_p`` = the number of ``P_a`` pairs whose ADDED
   substitution sits at position ``p``; ``k_p`` = how many of those are cliffs.
   A pair's position is the position of the substitution that differs -- the
   only choice under which the count is an exposure.
3. **``beta_hat_abs``** = the mean of ``|beta_c|`` over the position's observed
   substitution columns (``max`` is carried as ``beta_hat_max`` for sensitivity).
4. **``OR_burial_matched``** = the Mantel-Haenszel odds ratio of (cliff x
   interface) stratified on the assay's own ``rsa_iso`` TERTILE -- the literal
   reading of "burial-matched".  Its CI is a POSITION block bootstrap
   (the study's ground rule: block-resample mutated positions, never edges);
   the analytic Robins-Breslow-Greenland CI is reported beside it and is
   anticonservative because the ``n_p`` pairs of one position are dependent.
5. **NS1 strata**: the spec's ``(levy x aa_class x |beta|-decile x depth-tertile)``
   is the identity permutation on a 55-position assay (900 cells, 55 positions).
   A declared COARSENING LADDER is used and the level reached is reported in
   ``NS1_strata_level`` / ``NS1_frac_exchangeable``.  See :data:`NS1_LADDER`.
6. **``sigma`` for L5** -- KRAS family: ``sigma_eps = 0.124313``
   (``measured_replicate``); every other assay: ``1.4826 x MAD(eps)`` of its own
   eps distribution (``internal_residual``, conservative -- it attributes all
   epistasis to noise).  The AUROC is reported at BOTH that scale and the
   ``sqrt(3) sigma_y`` alternative ORCHESTRATOR D5 demands, so no verdict rests
   on the choice.
7. **cross-assay position joins are WT-identity-verified integer offsets**,
   scanned over ``[-500, 500]`` and required to be unique with ZERO wt-letter
   mismatches.  A naive ``seq_idx`` join across the four KRAS partners
   disagrees on the WT letter at 156 of 159 shared positions and is banned
   (spec G1b's rule, applied to KRAS).

Python 3.9: no ``match``, no runtime ``X | Y`` unions.  ``statsmodels`` is not
installed: the IRLS GLM, the HC3 sandwich, BH-FDR and the exact Mantel /
sign-flip permutation tests are implemented here on scipy.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

from scipy import linalg as _splinalg
from scipy import special as _spec
from scipy import stats as _sps

from cliff import config
from cliff import io_bgym
from cliff import latent as _latent
from cliff import noise as _noise
from cliff import nulls as _nulls
from cliff import pairs as _pairs
from cliff import structure as _structure
from cliff.config import ASSAYS, PATHS, SEEDS, THRESH

# --------------------------------------------------------------------------- #
# frozen local constants (decision boundaries all come from THRESH)           #
# --------------------------------------------------------------------------- #

#: the cliff cut on ``|c_hat|`` used for every per-position count (item 1 above)
TAU_PRIMARY = 3.0
#: the C2 catalogue cut, carried as a second column set
TAU_SECONDARY = float(THRESH['C2_catalogue_c_min'])
#: every tau whose per-position counts are tabulated
TAU_GRID = tuple(float(t) for t in THRESH.get('C4_tau_grid', (2.0, 3.0, 4.0, 5.0, 6.0, 8.0)))

#: NS1 coarsening ladder (item 5).  Level 0 is the spec's literal stratification.
NS1_LADDER = (
    ('spec:levy x aa_class x |beta|-decile x depth-tertile',
     ('levy_class', 'aa_class', 'beta_decile', 'depth_tertile')),
    ('levy x |beta|-decile x depth-tertile',
     ('levy_class', 'beta_decile', 'depth_tertile')),
    ('levy x |beta|-tertile', ('levy_class', 'beta_tertile')),
    ('rsa-tertile x |beta|-tertile', ('rsa_tertile', 'beta_tertile')),
    ('rsa-tertile', ('rsa_tertile',)),
    ('unstratified', ()),
)
#: a stratification is USABLE when at least this fraction of positions sit in a
#: stratum that can actually exchange the label (size >= 2 and both labels present)
NS1_MIN_FRAC_EXCHANGEABLE = 0.50
NS1_MIN_N_EXCHANGEABLE = 10

#: the two interface definitions D6 makes co-primary
IFACE_DEFS = (('5A', 'is_iface_5A'), ('dsasa', 'is_iface_dsasa'))

#: bootstrap / permutation counts
B_NS1 = int(THRESH['C4_n_perm_NS1'])          # 10,000
B_NS3 = int(THRESH['C4_n_perm_NS3'])          # 10,000
B_NS2 = int(THRESH['C3N_n_perm'])             # 10,000 -- THRESH carries no NS2 count
B_BOOT = int(THRESH['C2_block_bootstrap_B'])  # 1,000 position/site-pair blocks

_SQRT3 = math.sqrt(3.0)


#: C4-I is the one test whose unit is a FAMILY, not an assay, so a family label
#: reaches :func:`_rng` where ``config.ASSAY_ORDINAL`` has no entry for it.  The
#: ordinals are frozen by the alphabetical order of :data:`C4I_FAMILIES` and
#: offset past every assay ordinal, so a family stream can never collide with an
#: assay stream and neither can drift.
FAMILY_ORDINAL = {'5A12': 1000, 'BH3': 1001, 'CR9114': 1002,
                  'KRAS': 1003, 'PSD95': 1004}


def _rng(name, dms_id=None):
    """``np.random.default_rng`` seeded ONLY from ``config.SEEDS``.

    ``dms_id`` is either a registered DMS_id (``config.assay_seed``) or one of
    the C4-I family labels in :data:`FAMILY_ORDINAL`.  Anything else raises
    rather than silently falling back to an unseeded stream.
    """
    if dms_id is None:
        return np.random.default_rng([SEEDS[name]])
    if dms_id in config.ASSAY_ORDINAL:
        return np.random.default_rng(config.assay_seed(name, dms_id))
    if dms_id in FAMILY_ORDINAL:
        if name not in SEEDS:
            raise KeyError('unknown seed name %r; add it to config.SEEDS'
                           % (name,))
        return np.random.default_rng([SEEDS[name], FAMILY_ORDINAL[dms_id]])
    raise KeyError('%r is neither a registered DMS_id nor a C4-I family label'
                   % (dms_id,))


# =========================================================================== #
# 1. numerical helpers (statsmodels is not installed)                          #
# =========================================================================== #

def bh_fdr(p):
    """Benjamini-Hochberg q-values.  NaNs pass through as NaN."""
    p = np.asarray(p, dtype=np.float64)
    q = np.full(p.shape, np.nan)
    fin = np.nonzero(np.isfinite(p))[0]
    if fin.size == 0:
        return q
    order = fin[np.argsort(p[fin], kind='stable')]
    m = order.size
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * m / (rank + 1.0))
        q[i] = min(1.0, prev)
    return q


def empirical_p(obs, null_vec, *, side='greater'):
    """``p = (1 + #{b: stat_b >= stat_obs}) / (B + 1)`` -- the study's only
    inference form (spec Sec.1.3; the ground rule repeats it)."""
    v = np.asarray(null_vec, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0 or not np.isfinite(obs):
        return float('nan'), 0
    if side == 'greater':
        c = int((v >= obs).sum())
    elif side == 'less':
        c = int((v <= obs).sum())
    elif side == 'two-sided':
        c = int((np.abs(v) >= abs(obs)).sum())
    else:
        raise ValueError('side must be greater|less|two-sided')
    return (1.0 + c) / (v.size + 1.0), v.size


def _rank_reduce(X, tol=1e-9):
    """Column-pivoted QR rank detection: which columns of ``X`` are ALIASED.

    Returned as ``(keep, dropped)`` with the first column (the intercept) always
    kept.  An exactly collinear covariate is dropped and NAMED rather than
    silently inverted through a pseudo-inverse -- with ``iface = dSASA > 0`` the
    Levy dummies are exactly collinear with ``iface`` on this benchmark (395 of
    395 mutated positions), and a study that did not say so would be reporting a
    coefficient that does not exist.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.shape[1] == 0:
        return [], []
    scale = np.sqrt((X * X).sum(axis=0))
    scale[scale == 0] = 1.0
    Q, R, piv = _splinalg.qr(X / scale, mode='economic', pivoting=True)
    d = np.abs(np.diag(R))
    rank = int((d > tol * max(d[0], 1.0)).sum()) if d.size else 0
    keep = sorted(int(c) for c in piv[:rank])
    if 0 not in keep and X.shape[1]:
        keep = sorted(set(keep) | {0})
    dropped = [c for c in range(X.shape[1]) if c not in keep]
    return keep, dropped


def irls_glm(y, X, *, family='poisson', offset=None, exposure=None,
             n_iter=100, tol=1e-11, names=None):
    """Poisson-log / binomial-logit GLM by IRLS, with model and HC3 SEs.

    ``family='poisson'``  : ``log E[y] = X b + offset``.
    ``family='binomial'`` : ``y`` successes out of ``exposure`` trials,
    ``logit(p) = X b`` (no offset -- the exposure is the binomial denominator).

    HC3: ``V = A^-1 [sum_i x_i x_i' r_i^2 / (1-h_i)^2] A^-1`` with
    ``A = X'WX``, ``r_i`` the score residual (``y_i - mu_i`` for Poisson,
    ``y_i - n_i p_i`` for binomial) and ``h_i`` the GLM hat value.  The
    model-based SE is reported beside it; the HC3 one is used for every Wald p
    because per-position pair counts are over-dispersed by construction (the
    ``n_p`` pairs of one position are dependent).
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    n, k0 = X.shape
    if names is None:
        names = ['x%d' % i for i in range(k0)]
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=np.float64)
    keep, dropped = _rank_reduce(X)
    Xk = X[:, keep]
    b = np.zeros(len(keep))
    if family == 'poisson':
        m = max(y.mean(), 1e-8)
        b[0] = math.log(m) - off.mean()
    elif family == 'binomial':
        N = np.asarray(exposure, dtype=np.float64)
        p0 = min(max(y.sum() / max(N.sum(), 1.0), 1e-6), 1 - 1e-6)
        b[0] = math.log(p0 / (1 - p0))
    else:
        raise ValueError('family must be poisson|binomial')
    conv, it = False, 0
    W = mu = None
    for it in range(1, n_iter + 1):
        eta = Xk @ b
        if family == 'poisson':
            mu = np.exp(np.clip(eta + off, -700, 700))
            W = mu
            r = y - mu
            zw = eta + np.where(mu > 0, r / np.maximum(mu, 1e-300), 0.0)
        else:
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -700, 700)))
            mu = N * p
            W = np.maximum(N * p * (1 - p), 1e-12)
            r = y - mu
            zw = eta + r / W
        A = Xk.T @ (Xk * W[:, None])
        rhs = Xk.T @ (W * zw)
        try:
            b_new = _splinalg.solve(A, rhs, assume_a='pos')
        except (np.linalg.LinAlgError, ValueError):
            b_new = np.linalg.lstsq(A, rhs, rcond=None)[0]
        step = float(np.max(np.abs(b_new - b))) if b.size else 0.0
        b = b_new
        if step < tol:
            conv = True
            break
    # ---- covariance -------------------------------------------------------- #
    eta = Xk @ b
    if family == 'poisson':
        mu = np.exp(np.clip(eta + off, -700, 700))
        W = mu
        r = y - mu
        with np.errstate(divide='ignore', invalid='ignore'):
            dev = 2.0 * np.nansum(np.where(y > 0, y * np.log(y / np.maximum(mu, 1e-300)), 0.0) - (y - mu))
        ll = float(np.sum(y * np.log(np.maximum(mu, 1e-300)) - mu
                          - _spec.gammaln(y + 1.0)))
    else:
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -700, 700)))
        mu = N * p
        W = np.maximum(N * p * (1 - p), 1e-12)
        r = y - mu
        with np.errstate(divide='ignore', invalid='ignore'):
            dev = 2.0 * np.nansum(
                np.where(y > 0, y * np.log(y / np.maximum(mu, 1e-300)), 0.0)
                + np.where(N - y > 0, (N - y) * np.log((N - y) / np.maximum(N - mu, 1e-300)), 0.0))
        ll = float(np.sum(y * np.log(np.maximum(p, 1e-300))
                          + (N - y) * np.log(np.maximum(1 - p, 1e-300))
                          + _spec.gammaln(N + 1) - _spec.gammaln(y + 1)
                          - _spec.gammaln(N - y + 1)))
    A = Xk.T @ (Xk * W[:, None])
    try:
        Ainv = _splinalg.inv(A)
    except (np.linalg.LinAlgError, ValueError):
        Ainv = np.linalg.pinv(A)
    se_m = np.sqrt(np.maximum(np.diag(Ainv), 0.0))
    h = np.einsum('ij,jk,ik->i', Xk, Ainv, Xk) * W
    h = np.clip(h, 0.0, 1.0 - 1e-8)
    u = Xk * (r / (1.0 - h))[:, None]
    V = Ainv @ (u.T @ u) @ Ainv
    se_r = np.sqrt(np.maximum(np.diag(V), 0.0))
    out = dict(coef={}, se_model={}, se_hc3={}, z={}, p={}, p_model={},
               dropped=[names[c] for c in dropped], rank=len(keep),
               converged=bool(conv), n_iter=int(it), loglik=float(ll),
               deviance=float(dev), n=int(n), family=family,
               dispersion=float(dev / max(n - len(keep), 1)))
    for j, c in enumerate(keep):
        nm = names[c]
        out['coef'][nm] = float(b[j])
        out['se_model'][nm] = float(se_m[j])
        out['se_hc3'][nm] = float(se_r[j])
        zz = b[j] / se_r[j] if se_r[j] > 0 else float('nan')
        out['z'][nm] = float(zz)
        out['p'][nm] = float(2.0 * _sps.norm.sf(abs(zz))) if np.isfinite(zz) else float('nan')
        zm = b[j] / se_m[j] if se_m[j] > 0 else float('nan')
        out['p_model'][nm] = float(2.0 * _sps.norm.sf(abs(zm))) if np.isfinite(zm) else float('nan')
    return out


def mh_or(k, n, iface, strata):
    """Mantel-Haenszel OR of (cliff x interface) stratified on ``strata``.

    ``k`` cliff pairs, ``n`` total pairs, per POSITION; ``iface`` the binary
    position label.  Returns the MH estimate, the Robins-Breslow-Greenland
    log-OR variance, the crude OR and the per-stratum table.
    """
    k = np.asarray(k, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    f = np.asarray(iface, dtype=bool)
    s = np.asarray(strata)
    R = S = 0.0
    vPR = vPSQR = vQS = 0.0
    tab = []
    for sv in pd.unique(s):
        m = (s == sv)
        a = float(k[m & f].sum())
        b = float((n - k)[m & f].sum())
        c = float(k[m & ~f].sum())
        d = float((n - k)[m & ~f].sum())
        N = a + b + c + d
        if N <= 0 or (a + b) == 0 or (c + d) == 0:
            tab.append(dict(stratum=str(sv), a=a, b=b, c=c, d=d, usable=False))
            continue
        r = a * d / N
        ss = b * c / N
        P = (a + d) / N
        Q = (b + c) / N
        R += r
        S += ss
        vPR += P * r
        vPSQR += P * ss + Q * r
        vQS += Q * ss
        tab.append(dict(stratum=str(sv), a=a, b=b, c=c, d=d, usable=True,
                        or_stratum=(a * d / (b * c)) if (b * c) > 0 else float('nan')))
    orr = (R / S) if S > 0 else float('nan')
    if R > 0 and S > 0:
        var = vPR / (2 * R * R) + vPSQR / (2 * R * S) + vQS / (2 * S * S)
    else:
        var = float('nan')
    A = float(k[f].sum()); Bb = float((n - k)[f].sum())
    C = float(k[~f].sum()); D = float((n - k)[~f].sum())
    crude = (A * D / (Bb * C)) if (Bb > 0 and C > 0) else float('nan')
    return dict(OR=orr, log_var=var, crude_OR=crude,
                n_strata_usable=int(sum(1 for t in tab if t['usable'])),
                a=A, b=Bb, c=C, d=D, table=tab,
                rate_iface=(A / (A + Bb)) if (A + Bb) > 0 else float('nan'),
                rate_noniface=(C / (C + D)) if (C + D) > 0 else float('nan'))


def auroc_from_ranks(rank, label):
    """AUROC via the rank-sum identity, MID-RANKS for ties.

    ``rank`` are mid-ranks (1-based) of the SCORE over the two-class subset,
    ``label`` the binary class.  Ties are unavoidable here: every eps of one
    site pair shares that site pair's ``d3d_min_heavy``.
    """
    rank = np.asarray(rank, dtype=np.float64)
    lab = np.asarray(label, dtype=bool)
    n1 = int(lab.sum())
    n0 = int(lab.size - n1)
    if n1 == 0 or n0 == 0:
        return float('nan')
    U = rank[lab].sum() - n1 * (n1 + 1.0) / 2.0
    return float(U / (n1 * n0))


def midranks(v):
    """1-based mid-ranks of ``v`` (average rank within a tie group)."""
    v = np.asarray(v, dtype=np.float64)
    order = np.argsort(v, kind='stable')
    sv = v[order]
    r = np.empty(v.size, dtype=np.float64)
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def group_permuter(strata):
    """Pre-sorted stratified permuter.

    ``cliff.nulls.permute_within_strata`` re-``argsort``s the stratum vector on
    every call, which is 10,000 sorts of a 91,845-element array for one NS2
    p-value.  The groups are fixed, so they are computed ONCE here and the rng
    call sequence is kept IDENTICAL (``rng.permutation(grp)`` per group, groups
    in ascending stratum order) so this returns exactly what
    ``permute_within_strata`` returns for the same rng state.  Asserted in
    :func:`_selfcheck`.
    """
    s = np.asarray(strata)
    order = np.argsort(s, kind='stable')
    bounds = np.nonzero(np.diff(s[order]))[0] + 1
    groups = [g for g in np.split(order, bounds) if g.size > 1]

    def permute(values, rng):
        out = np.asarray(values).copy()
        for g in groups:
            out[g] = np.asarray(values)[rng.permutation(g)]
        return out
    return permute, groups


def double_centre(M):
    """``M - rowmean - colmean + grandmean`` on a matrix with no missing cells."""
    M = np.asarray(M, dtype=np.float64)
    return M - M.mean(axis=1, keepdims=True) - M.mean(axis=0, keepdims=True) + M.mean()


def mantel_corr(Z, W):
    """Pearson correlation of the two double-centred matrices over all cells."""
    a = double_centre(Z).ravel()
    b = double_centre(W).ravel()
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float('nan')
    return float(_sps.spearmanr(a[m], b[m]).correlation)


def emp_logit(k, n, *, add=0.5):
    """Empirical logit with the Cox/Anscombe correction ``log((k+.5)/(n-k+.5))``.

    A per-position cliff rate of exactly 0 is common (57 of 166 KRAS positions at
    tau=3), so the raw logit does not exist and the correction is part of the
    definition, not a patch."""
    k = np.asarray(k, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    out = np.full(k.shape, np.nan)
    m = n > 0
    out[m] = np.log((k[m] + add) / (n[m] - k[m] + add))
    return out


# =========================================================================== #
# 2. the per-position cliff table (the exposure/count layer for C4-S)          #
# =========================================================================== #

def has_latent(dms_id):
    return os.path.exists(os.path.join(PATHS.latent, dms_id + '.npz'))


def position_key_of_columns(dms_id):
    """``(pos_index array over X columns, ordered list of (chain, seq_pos))``.

    Built from the CACHED design's ``col_index`` -- the same object
    ``cliff.nulls`` uses for ``pos_of_add`` -- but with the ``(chain, seq_pos)``
    label kept, which is what T09 is joined on.
    """
    des = _latent.load_cached_design(dms_id, verify=False)
    inv = {v: k for k, v in des['col_index'].items()}
    lut, keys = {}, []
    pos_of_col = np.empty(len(inv), dtype=np.int32)
    for c in range(len(inv)):
        ch, ps, _aa = inv[c]
        key = (ch, int(ps))
        if key not in lut:
            lut[key] = len(lut)
            keys.append(key)
        pos_of_col[c] = lut[key]
    return pos_of_col, keys, des


def cliff_counts(dms_id, *, centred=True, verify=False):
    """Per-position ``P_a`` exposure and cliff counts at every tau in TAU_GRID.

    Returns a DataFrame with one row per DISTINCT mutated ``(chain, seq_pos)``
    of the assay plus ``df.attrs`` carrying the assay-level totals.  The cliff
    statistic itself is ``cliff.nulls.c_hat`` (phi-centred, D2) on the cached
    fit -- never refitted here (spec Sec.5).
    """
    ctx = _nulls.get_context(dms_id, verify=verify)
    pos_of_col, keys, des = position_key_of_columns(dms_id)
    npos = len(keys)
    pa = _nulls._pa_mask(ctx, ctx.censor_mask, ctx.oof_finite)
    c = _nulls.c_hat(ctx.e_oof, ctx.sigma_oof, ctx.nested_idx,
                     mu=(ctx.mu_oof if centred else None))
    ac = np.abs(c[pa])
    pos = pos_of_col[ctx.add_col[pa]]
    n_p = np.bincount(pos, minlength=npos)
    beta = np.asarray(ctx.beta, dtype=np.float64)
    # beta carries the intercept in column 0 of [1|X] (latent.with_intercept)
    b_cols = beta[1:] if beta.size == pos_of_col.size + 1 else beta
    b_abs_mean = np.zeros(npos)
    b_abs_max = np.zeros(npos)
    cnt = np.bincount(pos_of_col, minlength=npos).astype(np.float64)
    tot = np.bincount(pos_of_col, weights=np.abs(b_cols), minlength=npos)
    with np.errstate(invalid='ignore', divide='ignore'):
        b_abs_mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    for j in range(npos):
        sel = np.abs(b_cols[pos_of_col == j])
        b_abs_max[j] = sel.max() if sel.size else np.nan
    d = pd.DataFrame(dict(
        DMS_id=dms_id,
        chain=[k[0] for k in keys], seq_idx=[k[1] for k in keys],
        n_pairs_at_site=n_p.astype(np.int64),
        n_aa_columns=cnt.astype(np.int64),
        beta_hat_abs=b_abs_mean, beta_hat_max=b_abs_max))
    for t in TAU_GRID:
        k_p = np.bincount(pos[ac >= t], minlength=npos)
        d['n_cliff_tau%g' % t] = k_p.astype(np.int64)
    d['n_cliff_pairs'] = d['n_cliff_tau%g' % TAU_PRIMARY]
    with np.errstate(invalid='ignore', divide='ignore'):
        d['cliff_rate'] = np.where(d['n_pairs_at_site'] > 0,
                                   d['n_cliff_pairs'] / d['n_pairs_at_site'].replace(0, np.nan),
                                   np.nan)
    d.attrs.update(dms_id=dms_id, n_Pa=int(pa.sum()),
                   n_nested=int(ctx.nested_idx.shape[0]),
                   n_positions=npos, centred=bool(centred),
                   rate_tau_primary=float((ac >= TAU_PRIMARY).mean()) if ac.size else float('nan'),
                   n_cliff_total=int((ac >= TAU_PRIMARY).sum()))
    return d


def site_table(dms_id, t09=None, *, centred=True):
    """T09's structural rows for ``dms_id`` joined to :func:`cliff_counts`.

    The join is ``(chain, seq_idx)`` WITHIN one assay, which is exact -- both
    sides come from the same ``pos_index``.  Cross-ASSAY joins never use
    ``seq_idx`` (see :func:`resolve_offsets`)."""
    if t09 is None:
        t09 = read_T09()
    st = t09[t09['DMS_id'].astype(str) == dms_id].copy()
    if not len(st) or not has_latent(dms_id):
        return None
    cc = cliff_counts(dms_id, centred=centred)
    st['chain'] = st['chain'].astype(str)
    st['seq_idx'] = st['seq_idx'].astype(np.int64)
    out = st.merge(cc.drop(columns=['DMS_id']), on=['chain', 'seq_idx'],
                   how='left', validate='one_to_one')
    if out['n_pairs_at_site'].isna().any():
        miss = out.loc[out['n_pairs_at_site'].isna(), ['chain', 'seq_idx']]
        raise AssertionError('%s: %d T09 positions absent from the design '
                             '(%r)' % (dms_id, len(miss), miss.values[:5].tolist()))
    out['rsa_tertile'] = _rank_bins(out['rsa_iso'].values, 3)
    out['beta_decile'] = _rank_bins(out['beta_hat_abs'].values, 10)
    out['beta_tertile'] = _rank_bins(out['beta_hat_abs'].values, 3)
    out.attrs.update(cc.attrs)
    return out


def _rank_bins(v, k):
    """Equal-count bin index -- the same construction as ``nulls._decile``."""
    v = np.asarray(v, dtype=np.float64)
    out = np.zeros(v.size, dtype=np.int32)
    fin = np.nonzero(np.isfinite(v))[0]
    if fin.size:
        order = fin[np.argsort(v[fin], kind='stable')]
        for b, part in enumerate(np.array_split(order, min(k, order.size))):
            out[part] = b
    out[np.setdiff1d(np.arange(v.size), fin)] = -1
    return out


# =========================================================================== #
# 3. C4-S -- the burial-matched site-level test                                #
# =========================================================================== #

LEVY_LEVELS = ('interior', 'surface', 'support', 'rim', 'core')


def _glm_design(tab, iface_col, *, with_levy=True):
    """``[1, iface, rsa_iso, levy dummies, log n_p, |beta|]`` and its names."""
    n = len(tab)
    cols = [np.ones(n), tab[iface_col].values.astype(float)]
    names = ['const', 'iface']
    cols.append(tab['rsa_iso'].values.astype(float)); names.append('rsa_iso')
    if with_levy:
        lv = tab['levy_class'].astype(str).values
        for lev in LEVY_LEVELS[1:]:                    # reference = interior
            if (lv == lev).any():
                cols.append((lv == lev).astype(float))
                names.append('levy_' + lev)
    cols.append(np.log(tab['n_pairs_at_site'].values.astype(float)))
    names.append('log_n_p')
    cols.append(np.nan_to_num(tab['beta_hat_abs'].values.astype(float), nan=0.0))
    names.append('beta_hat_abs')
    return np.column_stack(cols), names


def c4s_models(tab, iface_col, *, tau=TAU_PRIMARY):
    """The three nested GLMs the kill switch needs, Poisson AND binomial."""
    kcol = 'n_cliff_tau%g' % tau
    use = tab[tab['n_pairs_at_site'] > 0].copy()
    y = use[kcol].values.astype(float)
    n_p = use['n_pairs_at_site'].values.astype(float)
    off = np.log(n_p)
    out = dict(n_positions=int(len(use)),
               n_dropped_zero_exposure=int((tab['n_pairs_at_site'] <= 0).sum()),
               n_iface=int(use[iface_col].sum()),
               n_noniface=int((~use[iface_col].astype(bool)).sum()),
               n_cliff=int(y.sum()), n_pairs=int(n_p.sum()))
    if out['n_iface'] == 0 or out['n_noniface'] == 0 or len(use) < 4:
        out['note'] = ('NO CONTRAST: %d interface / %d non-interface positions '
                       '-- the interface coefficient is not identified'
                       % (out['n_iface'], out['n_noniface']))
        for k in ('beta_iface_unadj', 'beta_iface_after_rsa', 'beta_iface_adj',
                  'p_wald', 'p_wald_after_rsa', 'p_wald_unadj', 'OR_glm',
                  'beta_iface_binom', 'p_wald_binom'):
            out[k] = float('nan')
        out['glm_dropped'] = []
        return out, use
    X, names = _glm_design(use, iface_col)
    iu = names.index('iface')
    m_un = irls_glm(y, X[:, :2], family='poisson', offset=off, names=names[:2])
    m_rsa = irls_glm(y, X[:, :3], family='poisson', offset=off, names=names[:3])
    m_full = irls_glm(y, X, family='poisson', offset=off, names=names)
    m_bin = irls_glm(y, X, family='binomial', exposure=n_p, names=names)
    out.update(
        beta_iface_unadj=m_un['coef'].get('iface', float('nan')),
        p_wald_unadj=m_un['p'].get('iface', float('nan')),
        beta_iface_after_rsa=m_rsa['coef'].get('iface', float('nan')),
        p_wald_after_rsa=m_rsa['p'].get('iface', float('nan')),
        beta_iface_adj=m_full['coef'].get('iface', float('nan')),
        p_wald=m_full['p'].get('iface', float('nan')),
        p_wald_model_se=m_full['p_model'].get('iface', float('nan')),
        se_hc3_iface=m_full['se_hc3'].get('iface', float('nan')),
        se_model_iface=m_full['se_model'].get('iface', float('nan')),
        OR_glm=(math.exp(m_full['coef']['iface']) if 'iface' in m_full['coef']
                else float('nan')),
        RR_unadj=(math.exp(m_un['coef']['iface']) if 'iface' in m_un['coef']
                  else float('nan')),
        beta_iface_binom=m_bin['coef'].get('iface', float('nan')),
        p_wald_binom=m_bin['p'].get('iface', float('nan')),
        OR_binom=(math.exp(m_bin['coef']['iface']) if 'iface' in m_bin['coef']
                  else float('nan')),
        glm_dropped=m_full['dropped'],
        glm_dispersion=m_full['dispersion'],
        glm_converged=bool(m_full['converged']),
        beta_rsa_adj=m_full['coef'].get('rsa_iso', float('nan')),
        p_rsa_adj=m_full['p'].get('rsa_iso', float('nan')),
        beta_logn_adj=m_full['coef'].get('log_n_p', float('nan')),
        beta_babs_adj=m_full['coef'].get('beta_hat_abs', float('nan')),
        note='')
    out['_models'] = dict(unadj=m_un, rsa=m_rsa, full=m_full, binom=m_bin)
    out['_iface_idx'] = iu
    return out, use


def c4s_or(use, iface_col, *, tau=TAU_PRIMARY, dms_id=None, B=None):
    """``OR_burial_matched`` (MH on the rsa tertile) + the position block
    bootstrap CI + the analytic RBG CI."""
    kcol = 'n_cliff_tau%g' % tau
    k = use[kcol].values.astype(float)
    n = use['n_pairs_at_site'].values.astype(float)
    f = use[iface_col].values.astype(bool)
    st = use['rsa_tertile'].values
    res = mh_or(k, n, f, st)
    dec = mh_or(k, n, f, use['rsa_decile'].values)
    lev = mh_or(k, n, f, use['levy_class'].astype(str).values)
    out = dict(OR_burial_matched=res['OR'], crude_OR=res['crude_OR'],
               OR_rsa_decile=dec['OR'], OR_levy_stratified=lev['OR'],
               n_strata_usable=res['n_strata_usable'],
               cliff_rate_iface=res['rate_iface'],
               cliff_rate_noniface=res['rate_noniface'],
               n_cliff_iface=res['a'], n_pairs_iface=res['a'] + res['b'],
               n_cliff_noniface=res['c'], n_pairs_noniface=res['c'] + res['d'])
    v = res['log_var']
    if np.isfinite(v) and np.isfinite(res['OR']) and res['OR'] > 0:
        z = _sps.norm.isf(0.025)
        out['OR_lo95_rbg'] = float(math.exp(math.log(res['OR']) - z * math.sqrt(v)))
        out['OR_hi95_rbg'] = float(math.exp(math.log(res['OR']) + z * math.sqrt(v)))
        out['p_wald_MH'] = float(2 * _sps.norm.sf(abs(math.log(res['OR'])) / math.sqrt(v)))
    else:
        out['OR_lo95_rbg'] = out['OR_hi95_rbg'] = out['p_wald_MH'] = float('nan')
    # ---- position block bootstrap (the study's ground rule) ---------------- #
    B = B_BOOT if B is None else int(B)
    rng = _rng('bootstrap_block', dms_id) if dms_id else _rng('bootstrap_block')
    m = len(use)
    boot = np.full(B, np.nan)
    for b in range(B):
        idx = rng.integers(0, m, m)
        r = mh_or(k[idx], n[idx], f[idx], st[idx])
        boot[b] = r['OR']
    fin = boot[np.isfinite(boot) & (boot > 0)]
    if fin.size >= 20:
        out['OR_lo95'] = float(np.percentile(fin, 2.5))
        out['OR_hi95'] = float(np.percentile(fin, 97.5))
        out['OR_boot_n_finite'] = int(fin.size)
    else:
        out['OR_lo95'] = out['OR_hi95'] = float('nan')
        out['OR_boot_n_finite'] = int(fin.size)
    return out


def ns1_strata(use, iface_col):
    """Walk :data:`NS1_LADDER` and return the finest USABLE stratification."""
    f = use[iface_col].values.astype(bool)
    for level, (label, cols) in enumerate(NS1_LADDER):
        if cols:
            key = _nulls._strata_from([use[c].astype(str).values if use[c].dtype == object
                                       else use[c].values for c in cols])
        else:
            key = np.zeros(len(use), dtype=np.int32)
        n_exch = 0
        for sv in np.unique(key):
            m = key == sv
            if m.sum() > 1 and 0 < f[m].sum() < m.sum():
                n_exch += int(m.sum())
        frac = n_exch / float(len(use)) if len(use) else 0.0
        ok = (frac >= NS1_MIN_FRAC_EXCHANGEABLE and n_exch >= NS1_MIN_N_EXCHANGEABLE)
        if ok or level == len(NS1_LADDER) - 1:
            return dict(level=level, label=label, cols=cols, strata=key,
                        n_exchangeable=n_exch, frac_exchangeable=frac,
                        n_strata=int(np.unique(key).size), usable=ok)
    raise AssertionError('unreachable')


def ns1_pvalue(use, iface_col, *, tau=TAU_PRIMARY, dms_id=None, B=None,
               obs=None, with_glm=True):
    """NS1: permute the INTERFACE label within the strata, recompute log OR_MH
    (primary) and ``beta_iface_adj`` (GLM).  One-sided upward -- the hypothesis
    is enrichment AT the interface; the two-sided p is reported beside it."""
    B = B_NS1 if B is None else int(B)
    kcol = 'n_cliff_tau%g' % tau
    k = use[kcol].values.astype(float)
    n = use['n_pairs_at_site'].values.astype(float)
    f0 = use[iface_col].values.astype(bool)
    st = use['rsa_tertile'].values
    info = ns1_strata(use, iface_col)
    permute, groups = group_permuter(info['strata'])
    rng = _rng('perm_NS1', dms_id) if dms_id else _rng('perm_NS1')
    if obs is None:
        obs = mh_or(k, n, f0, st)['OR']
    log_obs = math.log(obs) if (np.isfinite(obs) and obs > 0) else float('nan')
    X = names = off = None
    if with_glm and np.isfinite(log_obs):
        X, names = _glm_design(use, iface_col)
        off = np.log(n)
        obs_g = irls_glm(k, X, family='poisson', offset=off,
                         names=names)['coef'].get('iface', float('nan'))
    else:
        obs_g = float('nan')
    lo = np.full(B, np.nan)
    lg = np.full(B, np.nan)
    n_identical = 0
    t0 = time.time()
    for b in range(B):
        fb = permute(f0, rng)
        if np.array_equal(fb, f0):
            n_identical += 1
        r = mh_or(k, n, fb, st)['OR']
        lo[b] = math.log(r) if (np.isfinite(r) and r > 0) else np.nan
        if X is not None:
            Xb = X.copy()
            Xb[:, names.index('iface')] = fb.astype(float)
            lg[b] = irls_glm(k, Xb, family='poisson', offset=off,
                             names=names)['coef'].get('iface', np.nan)
    p1, nb = empirical_p(log_obs, lo, side='greater')
    p2, _ = empirical_p(log_obs, lo, side='two-sided')
    pg, _ = empirical_p(obs_g, lg, side='greater')
    import warnings as _w
    with _w.catch_warnings():           # an all-NaN null is the NO-CONTRAST
        _w.simplefilter('ignore')       # case, reported as B = 0, not a crash
        null_mean = float(np.nanmean(lo)) if np.isfinite(lo).any() else float('nan')
        null_sd = float(np.nanstd(lo)) if np.isfinite(lo).any() else float('nan')
        perm_mean = (float(np.nanmean(lg)) if (with_glm and np.isfinite(lg).any())
                     else float('nan'))
    return dict(p_NS1=p1, p_NS1_two_sided=p2, p_NS1_glm=pg,
                NS1_strata_level=info['level'], NS1_strata_label=info['label'],
                NS1_n_strata=info['n_strata'],
                NS1_frac_exchangeable=info['frac_exchangeable'],
                NS1_n_exchangeable=info['n_exchangeable'],
                NS1_strata_usable=info['usable'],
                NS1_frac_identity_draws=n_identical / float(B),
                NS1_B=int(nb), NS1_null_mean_logOR=null_mean,
                NS1_null_sd_logOR=null_sd,
                NS1_wall_s=time.time() - t0,
                beta_iface_adj_perm_mean=perm_mean)


# =========================================================================== #
# 4. site-site geometry (C4-P needs residue-residue distances, which T09 --    #
#    a per-residue table of distance to the OPPOSITE side -- does not carry)  #
# =========================================================================== #

C4_CACHE = os.path.join(PATHS.cache, 'c4')


def site_distances(dms_id, *, use_cache=True, force=False):
    """Min heavy-atom (and Cb-Cb) distance BETWEEN every pair of the assay's
    mutated positions.

    T09 carries ``min_heavy_dist`` = distance to the OPPOSITE side; L5 needs the
    WITHIN-side residue-residue distance, so it is computed here from the same
    hydrogen-stripped model ``cliff.structure`` uses (``load_heavy_model`` +
    ``_flatten``), cached under ``data/cliff_cache/c4/`` and md5-registered.
    Cb-Cb is reported only -- ``Cb-Cb < 8 A`` is BANNED as an interface
    definition (spec Sec.1.5).
    """
    os.makedirs(C4_CACHE, exist_ok=True)
    path = os.path.join(C4_CACHE, '%s_sitedist.npz' % dms_id)
    if use_cache and not force and os.path.exists(path):
        with np.load(path, allow_pickle=False) as z:
            return dict(keys=[(str(c), int(p)) for c, p in
                              zip(z['chain'], z['seq_idx'])],
                        d_min=z['d_min'], d_cb=z['d_cb'], path=path,
                        resseq=z['resseq'], from_cache=True)
    spec = ASSAYS[dms_id]
    pdb = os.path.join(PATHS.structures, spec.pdb_file)
    _st, model, _nh = _structure.load_heavy_model(pdb)
    flat = _structure._flatten(model)
    idx_of = {k: i for i, k in enumerate(flat['keys'])}
    assay = io_bgym.load_assay(dms_id)
    annot, _entry = _structure.cache_structure(dms_id)
    mut = _structure.map_mutations(assay, annot)
    rows = mut[['chain', 'seq_idx', 'resseq', 'icode']].copy()
    rows['icode'] = rows['icode'].astype(str).replace('nan', '')
    J = len(rows)
    starts, ends, cb = [], [], []
    for r in rows.to_dict('records'):
        k = (str(r['chain']), int(r['resseq']), str(r['icode']))
        if k not in idx_of:
            raise KeyError('%s: mutated residue %r absent from %s'
                           % (dms_id, k, spec.pdb_file))
        i = idx_of[k]
        s = int(flat['starts'][i])
        e = s + int(flat['n_atoms'][i]) if 'n_atoms' in flat else None
        if e is None:
            e = int(flat['starts'][i + 1]) if i + 1 < len(flat['starts']) \
                else flat['coords'].shape[0]
        starts.append(s); ends.append(e)
        res = flat['residues'][i]
        at = None
        for nm in ('CB', 'CA'):
            if nm in res:
                at = res[nm]
                break
        cb.append(at.coord if at is not None else flat['coords'][s])
    coords = flat['coords']
    d_min = np.zeros((J, J))
    for a in range(J):
        ca = coords[starts[a]:ends[a]]
        for b in range(a + 1, J):
            cb_ = coords[starts[b]:ends[b]]
            dd = np.sqrt(((ca[:, None, :] - cb_[None, :, :]) ** 2).sum(-1)).min()
            d_min[a, b] = d_min[b, a] = dd
    cbv = np.asarray(cb, dtype=np.float64)
    d_cb = np.sqrt(((cbv[:, None, :] - cbv[None, :, :]) ** 2).sum(-1))
    np.savez_compressed(path, chain=np.array([str(c) for c in rows['chain']]),
                        seq_idx=rows['seq_idx'].values.astype(np.int64),
                        resseq=rows['resseq'].values.astype(np.int64),
                        d_min=d_min, d_cb=d_cb)
    return dict(keys=[(str(c), int(p)) for c, p in
                      zip(rows['chain'], rows['seq_idx'])],
                d_min=d_min, d_cb=d_cb, resseq=rows['resseq'].values,
                path=path, from_cache=False)


# =========================================================================== #
# 5. C4-P == route L5: eps localises in 3D                                     #
# =========================================================================== #

def sigma_eps_for(dms_id, eps=None, *, registry=None):
    """The two sigma scales every L5 number is reported at (frozen item 6).

    Returns ``(sigma_primary, provenance, sigma_alt, alt_label)``.
    """
    if registry is None:
        registry = _REGISTRY()
    sub = registry[(registry['DMS_id'] == dms_id)
                   & (registry['provenance'] != 'stipulated')]
    sy = float(sub['sigma_y'].iloc[0]) if len(sub) else float('nan')
    se = float(sub['sigma_eps'].iloc[0]) if len(sub) else float('nan')
    prov = str(sub['provenance'].iloc[0]) if len(sub) else 'unknown'
    if np.isfinite(se):
        return se, prov, _SQRT3 * _KRAS_Y_JOIN_SIGMA(), 'sqrt(3)*sigma_y_yjoin (D5)'
    e = np.asarray(eps, dtype=np.float64) if eps is not None else np.zeros(0)
    mad = float(1.4826 * np.median(np.abs(e - np.median(e)))) if e.size > 2 else float('nan')
    alt = _SQRT3 * sy if np.isfinite(sy) else float('nan')
    return mad, 'internal_residual(MAD of eps)', alt, 'sqrt(3)*sigma_y_registry'


_REG_CACHE = {}


def _REGISTRY():
    if 'df' not in _REG_CACHE:
        p = os.path.join(PATHS.artifacts, 'T03_noise_registry.csv')
        _REG_CACHE['df'] = pd.read_csv(p)
    return _REG_CACHE['df']


def _KRAS_Y_JOIN_SIGMA():
    """``sigma_y`` of the KRAS twin's y-LEVEL join -- the 0.1187 that makes
    D5's second threshold ``3 sqrt(3) sigma_y = 0.617``."""
    if 'kras_y' not in _REG_CACHE:
        k = _noise.kras_twin_epsilon()
        _REG_CACHE['kras_y'] = float(k['y_join']['sigma'])
    return _REG_CACHE['kras_y']


def epsilon_structure(dms_id, t09=None, *, verbose=False):
    """The T10 row set: every usable ``eps`` of ``dms_id`` with its two sites'
    geometry, both interface flags, both cliff labels and the site-pair
    aggregates.  ``eps`` comes from ``cliff.noise.epsilon_sitepairs`` -- the
    module that already owns the definition (see the delivery note's
    ``api_notes``); nothing is recomputed here."""
    if t09 is None:
        t09 = read_T09()
    assay = io_bgym.load_assay(dms_id)
    ep = _noise.epsilon_sitepairs(assay)
    if ep['n_usable'] == 0:
        return None, dict(dms_id=dms_id, n_eps=0, reason=ep['reason'])
    sd = site_distances(dms_id)
    pos_of = {k: i for i, k in enumerate(sd['keys'])}
    st = t09[t09['DMS_id'].astype(str) == dms_id].copy()
    st['chain'] = st['chain'].astype(str)
    st['seq_idx'] = st['seq_idx'].astype(np.int64)
    srow = {(r['chain'], int(r['seq_idx'])): r for r in st.to_dict('records')}
    keys, eps = ep['keys'], ep['eps']
    n = len(keys)
    # background count: variants carrying BOTH substitutions
    des = _latent.load_cached_design(dms_id, verify=False)
    X = des['X'].tocsc()
    ci = des['col_index']
    rec = []
    for i in range(n):
        (c1, p1, a1), (c2, p2, a2) = keys[i]
        k1, k2 = (c1, int(p1)), (c2, int(p2))
        r1, r2 = srow.get(k1), srow.get(k2)
        if r1 is None or r2 is None:
            continue
        i1, i2 = pos_of.get(k1), pos_of.get(k2)
        nb = int(X[:, ci[(c1, p1, a1)]].multiply(X[:, ci[(c2, p2, a2)]]).sum())
        rec.append((c1, int(p1), a1, c2, int(p2), a2,
                    (abs(int(p1) - int(p2)) if c1 == c2 else np.nan),
                    (sd['d_min'][i1, i2] if (i1 is not None and i2 is not None) else np.nan),
                    (sd['d_cb'][i1, i2] if (i1 is not None and i2 is not None) else np.nan),
                    str(r1['levy_class']), str(r2['levy_class']),
                    float(r1['rsa_iso']), float(r2['rsa_iso']),
                    bool(r1['is_iface_5A']), bool(r2['is_iface_5A']),
                    bool(r1['is_iface_dsasa']), bool(r2['is_iface_dsasa']),
                    float(eps[i]), nb))
    if not rec:
        return None, dict(dms_id=dms_id, n_eps=0,
                          reason='no eps with both sites structurally annotated')
    cols = ('chain_s', 'site_s', 'aa_s', 'chain_t', 'site_t', 'aa_t',
            'seq_separation', 'd3d_min_heavy', 'd3d_cb', 'levy_s', 'levy_t',
            'rsa_s', 'rsa_t', 'iface5_s', 'iface5_t', 'ifaceD_s', 'ifaceD_t',
            'eps', 'n_backgrounds')
    d = pd.DataFrame.from_records(rec, columns=cols)
    d.insert(0, 'DMS_id', dms_id)
    d['rsa_iso'] = 0.5 * (d['rsa_s'] + d['rsa_t'])
    d['both_iface'] = d['iface5_s'] & d['iface5_t']
    d['both_iface_dsasa'] = d['ifaceD_s'] & d['ifaceD_t']
    sp = list(zip(np.minimum(d['site_s'], d['site_t']),
                  np.maximum(d['site_s'], d['site_t']),
                  d['chain_s'], d['chain_t']))
    d['site_pair_id'] = pd.factorize(pd.Series(sp, index=d.index))[0]
    g = d.groupby('site_pair_id')['eps']
    d['n_aa_combos'] = g.transform('size').astype(np.int64)
    d['eps_sitepair_mean'] = g.transform('mean')
    d['eps_sitepair_sd'] = g.transform(lambda v: v.std(ddof=1))
    sig, prov, sig_alt, alt_label = sigma_eps_for(dms_id, d['eps'].values)
    d['sigma_eps_used'] = sig
    d['sigma_provenance'] = prov
    d['sigma_eps_alt'] = sig_alt
    d['sigma_alt_label'] = alt_label
    d['eps_z'] = d['eps'] / sig if np.isfinite(sig) and sig > 0 else np.nan
    d['is_cliff_3sigma'] = np.abs(d['eps']) >= THRESH['L5_cliff_sigma_mult'] * sig
    d['is_noncliff_1sigma'] = np.abs(d['eps']) < THRESH['L5_noncliff_sigma_mult'] * sig
    if np.isfinite(sig_alt) and sig_alt > 0:
        d['is_cliff_3sigma_alt'] = np.abs(d['eps']) >= THRESH['L5_cliff_sigma_mult'] * sig_alt
        d['is_noncliff_1sigma_alt'] = np.abs(d['eps']) < THRESH['L5_noncliff_sigma_mult'] * sig_alt
    else:
        d['is_cliff_3sigma_alt'] = False
        d['is_noncliff_1sigma_alt'] = False
    icc, n_icc, n_grp = _nulls._icc_oneway(d['eps'].values, d['site_pair_id'].values)
    d['ICC_sitepair'] = icc
    meta = dict(dms_id=dms_id, n_eps=int(len(d)), n_eps_total=int(ep['n_usable']),
                n_site_pairs=int(d['site_pair_id'].nunique()),
                sigma_eps_used=sig, sigma_provenance=prov,
                sigma_eps_alt=sig_alt, ICC_sitepair=icc, n_icc_groups=n_grp,
                n_cross_chain=int((~np.isfinite(d['seq_separation'])).sum()),
                reason='')
    if verbose:
        print('[C4-P] %-38s eps %6d  site pairs %5d  sigma %.4f (%s)  '
              'cliff %5d  noncliff %6d'
              % (dms_id, len(d), meta['n_site_pairs'], sig, prov,
                 int(d['is_cliff_3sigma'].sum()), int(d['is_noncliff_1sigma'].sum())))
    return d, meta


def l5_auroc(d, *, alt=False, dms_id=None, B_perm=None, B_boot=None):
    """AUROC of ``-d3d_min_heavy`` separating cliff from non-cliff eps, with the
    NS2 permutation p and a SITE-PAIR cluster bootstrap CI."""
    ck = 'is_cliff_3sigma_alt' if alt else 'is_cliff_3sigma'
    nk = 'is_noncliff_1sigma_alt' if alt else 'is_noncliff_1sigma'
    use = d[(d[ck] | d[nk]) & np.isfinite(d['d3d_min_heavy'])].copy()
    out = dict(n_class_cliff=int(use[ck].sum()),
               n_class_noncliff=int(use[nk].sum()),
               n_class_total=int(len(use)),
               n_eps_excluded_midband=int(len(d) - len(use)),
               L5_min_eps_gate=int(THRESH['L5_min_eps']),
               L5_feasible=bool(len(use) >= THRESH['L5_min_eps']))
    if out['n_class_cliff'] < 2 or out['n_class_noncliff'] < 2:
        out.update(AUROC_L5=float('nan'), AUROC_lo95=float('nan'),
                   AUROC_hi95=float('nan'), p_NS2=float('nan'), NS2_B=0,
                   NS2_null_mean=float('nan'), NS2_null_sd=float('nan'),
                   NS2_n_strata=0, NS2_wall_s=0.0, AUROC_boot_n=0,
                   AUROC_d3d_median_cliff=float('nan'),
                   AUROC_d3d_median_noncliff=float('nan'),
                   AUROC_check_from_placements=float('nan'),
                   note='no two-class contrast: %d cliff / %d non-cliff eps at '
                        'sigma = %s'
                        % (out['n_class_cliff'], out['n_class_noncliff'],
                           ('%.4f' % d['sigma_eps_used'].iloc[0]) if len(d)
                           else 'n/a'))
        return out, use, None
    score = -use['d3d_min_heavy'].values
    lab = use[ck].values.astype(bool)
    rank = midranks(score)
    a_obs = auroc_from_ranks(rank, lab)
    out['AUROC_L5'] = a_obs
    out['AUROC_d3d_median_cliff'] = float(np.median(use.loc[lab, 'd3d_min_heavy']))
    out['AUROC_d3d_median_noncliff'] = float(np.median(use.loc[~lab, 'd3d_min_heavy']))
    # ---- NS2 --------------------------------------------------------------- #
    B = B_NS2 if B_perm is None else int(B_perm)
    strata = _nulls._strata_from([_nulls._decile(use['seq_separation'].values, 10),
                                  _nulls._decile(use['rsa_iso'].values, 3)])
    permute, groups = group_permuter(strata)
    rng = _rng('perm_NS2', dms_id) if dms_id else _rng('perm_NS2')
    t0 = time.time()
    null = np.empty(B)
    for b in range(B):
        null[b] = auroc_from_ranks(rank, permute(lab, rng))
    p1, nb = empirical_p(a_obs, null, side='greater')
    out.update(p_NS2=p1, NS2_B=int(nb), NS2_null_mean=float(null.mean()),
               NS2_null_sd=float(null.std()), NS2_n_strata=int(np.unique(strata).size),
               NS2_wall_s=time.time() - t0)
    # ---- site-pair cluster bootstrap --------------------------------------- #
    Bb = B_BOOT if B_boot is None else int(B_boot)
    rng2 = _rng('bootstrap_block', dms_id) if dms_id else _rng('bootstrap_block')
    sp = use['site_pair_id'].values
    uniq = np.unique(sp)
    where = {u: np.nonzero(sp == u)[0] for u in uniq}
    boot = np.full(Bb, np.nan)
    for b in range(Bb):
        pick = rng2.integers(0, uniq.size, uniq.size)
        sel = np.concatenate([where[uniq[i]] for i in pick])
        lb = lab[sel]
        if lb.sum() < 2 or (~lb).sum() < 2:
            continue
        boot[b] = auroc_from_ranks(midranks(score[sel]), lb)
    fin = boot[np.isfinite(boot)]
    if fin.size >= 20:
        out['AUROC_lo95'] = float(np.percentile(fin, 2.5))
        out['AUROC_hi95'] = float(np.percentile(fin, 97.5))
    else:
        out['AUROC_lo95'] = out['AUROC_hi95'] = float('nan')
    out['AUROC_boot_n'] = int(fin.size)
    # ---- per-row placement (the AUROC's own decomposition) ----------------- #
    s1 = np.sort(score[lab])
    s0 = np.sort(score[~lab])
    place = np.full(len(use), np.nan)
    lo = np.searchsorted(s0, score[lab], side='left')
    hi = np.searchsorted(s0, score[lab], side='right')
    place[np.nonzero(lab)[0]] = (lo + hi) / (2.0 * s0.size)
    lo1 = np.searchsorted(s1, score[~lab], side='left')
    hi1 = np.searchsorted(s1, score[~lab], side='right')
    place[np.nonzero(~lab)[0]] = 1.0 - (lo1 + hi1) / (2.0 * s1.size)
    out['AUROC_check_from_placements'] = float(np.nanmean(place[lab]))
    return out, use, place


# =========================================================================== #
# 6. cross-assay position alignment (WT-identity-verified, spec G1b's rule)    #
# =========================================================================== #

_WT_CACHE = {}


def wt_map(dms_id):
    """``{(chain, seq_pos): wt_aa}`` over the assay's mutated positions.

    From ``cliff.structure.position_table``, which re-parses ``mutant`` /
    ``mutant_pdb`` and needs no PDB -- so it also works for the two CR9114
    hypercubes, which have no structure (spec G-OPT).
    """
    if dms_id not in _WT_CACHE:
        pt = _structure.position_table(io_bgym.load_assay(dms_id))
        _WT_CACHE[dms_id] = {(str(r['chain']), int(r['seq_idx'])): str(r['wt_aa_mutcol'])
                             for r in pt.to_dict('records')}
    return _WT_CACHE[dms_id]


def resolve_offsets(partners, *, ref=None, span=500):
    """Per-partner, per-chain integer offset onto ``ref``'s seq numbering.

    Scanned over ``[-span, span]``; an offset is accepted only if it produces
    ZERO wt-letter mismatches on the overlap, and the chosen one maximises the
    overlap.  Every candidate is reported so a non-unique solution cannot pass
    silently.  **This is not optional bookkeeping**: a naive ``seq_idx`` join
    across the four KRAS partners disagrees on the WT letter at 156 of 159
    shared positions (measured), i.e. the spec's 163 x 4 matrix would have been
    built out of mismatched residues.
    """
    partners = list(partners)
    ref = ref or partners[0]
    W = {d: wt_map(d) for d in partners}
    chains = {d: sorted(set(c for c, _p in W[d])) for d in partners}
    single = all(len(chains[d]) == 1 for d in partners)
    out = {}
    for d in partners:
        info = dict(dms_id=d, single_chain=single, chains=chains[d],
                    chain_map={}, offset={}, candidates={}, n_overlap={})
        if single:
            cr, cd = chains[ref][0], chains[d][0]
            pairs_ = [(cr, cd)]
            info['chain_map'][cd] = cr
        else:
            if chains[d] != chains[ref]:
                raise ValueError('%s vs %s: chain sets differ (%r vs %r) and '
                                 'neither is single-chain -- the join is not '
                                 'defined' % (d, ref, chains[d], chains[ref]))
            pairs_ = [(c, c) for c in chains[ref]]
            for c in chains[d]:
                info['chain_map'][c] = c
        for cr, cd in pairs_:
            wr = {p: a for (c, p), a in W[ref].items() if c == cr}
            wd = {p: a for (c, p), a in W[d].items() if c == cd}
            cand = []
            for off in range(-span, span + 1):
                sh = {p + off: a for p, a in wd.items()}
                common = set(sh) & set(wr)
                if not common:
                    continue
                if all(sh[p] == wr[p] for p in common):
                    cand.append((len(common), off))
            cand.sort(reverse=True)
            if not cand:
                raise ValueError('%s chain %s: no WT-consistent offset onto %s'
                                 % (d, cd, ref))
            info['offset'][cd] = int(cand[0][1])
            info['n_overlap'][cd] = int(cand[0][0])
            info['candidates'][cd] = [(int(n), int(o)) for n, o in cand[:5]]
            info['unique_at_max'] = sum(1 for n, _o in cand if n == cand[0][0]) == 1
        out[d] = info
    return out, ref


def aligned_positions(dms_id, offs):
    """``{(chain_ref, pos_ref): (chain_local, pos_local)}`` for one partner."""
    info = offs[dms_id]
    out = {}
    for (c, p), _a in wt_map(dms_id).items():
        cr = info['chain_map'].get(c, c)
        out[(cr, p + info['offset'].get(c, 0))] = (c, p)
    return out


# =========================================================================== #
# 7. the same-site channel (the only channel a max_mut = 1 probe has)          #
# =========================================================================== #

def samesite_counts(dms_id, *, tau=TAU_PRIMARY):
    """Per-position same-site substitution-roughness counts.

    ``c_ss = dy / (1.4826 MAD(dy))`` over the assay's same-site pairs
    ``(B u {i->a}, B u {i->b})``; a "cliff" is ``|c_ss| >= tau``.  ``P_a``
    condition (a) (``B != {}``) does NOT apply -- it exists to keep the single
    shared ``y_WT`` out of every nested edge, and no ``y_WT`` term enters a
    same-site difference.  Condition (b) (neither endpoint censored) does.

    This is the channel spec Sec.1.5 assigns to PSD95 ("same-site channel
    only"): a complete single scan has ``e == 0`` in sample and no cross-fit at
    all, so the latent/nested machinery has no content there.
    """
    des = _latent.load_cached_design(dms_id, verify=False)
    z = np.load(os.path.join(PATHS.pairs, dms_id + '_samesite.npz'))
    idx, pos_col = z['idx'], z['pos_col']
    z.close()
    y = des['y']
    cm = des['censor_mask'].astype(bool)
    keep = ~(cm[idx[:, 0]] | cm[idx[:, 1]])
    dy = y[idx[:, 1]] - y[idx[:, 0]]
    scale = float(1.4826 * np.median(np.abs(dy[keep] - np.median(dy[keep])))) \
        if keep.any() else float('nan')
    inv = {v: k for k, v in des['pos_index'].items()}
    npos = len(inv)
    pos = pos_col[keep]
    n_p = np.bincount(pos, minlength=npos)
    c = np.abs(dy[keep]) / scale if (np.isfinite(scale) and scale > 0) else np.full(keep.sum(), np.nan)
    d = pd.DataFrame(dict(DMS_id=dms_id,
                          chain=[inv[i][0] for i in range(npos)],
                          seq_idx=[inv[i][1] for i in range(npos)],
                          n_pairs_at_site=n_p.astype(np.int64)))
    for t in TAU_GRID:
        d['n_cliff_tau%g' % t] = np.bincount(pos[c >= t], minlength=npos).astype(np.int64)
    d['n_cliff_pairs'] = d['n_cliff_tau%g' % tau]
    with np.errstate(invalid='ignore'):
        d['cliff_rate'] = np.where(n_p > 0, d['n_cliff_pairs'] / np.maximum(n_p, 1), np.nan)
    d.attrs.update(dms_id=dms_id, channel='samesite_dy', scale=scale,
                   n_pairs=int(keep.sum()), n_pairs_all=int(idx.shape[0]),
                   n_censor_dropped=int((~keep).sum()),
                   rate_tau_primary=float((c >= tau).mean()) if c.size else float('nan'))
    return d


def partner_counts(dms_id, channel, *, tau=TAU_PRIMARY):
    if channel == 'nested':
        return cliff_counts(dms_id)
    if channel == 'samesite':
        return samesite_counts(dms_id, tau=tau)
    raise ValueError('channel must be nested|samesite')


# =========================================================================== #
# 8. C4-I -- partner specificity, double-centred                               #
# =========================================================================== #

KRAS_C4I_PARTNERS = ('KRAS_RAF1-RBD_norfitness_6VJJ',
                     'KRAS_RALGDS-RBD_norfitness_1LFD',
                     'KRAS_PICK3CG-RBD_norfitness_1HE8',
                     'KRAS_SOS1_norfitness_8BE4')

C4I_FAMILIES = {
    'KRAS': dict(partners=KRAS_C4I_PARTNERS, channel='nested', structure=True,
                 note='the only adequately powered family (spec Sec.1.5); '
                      'KRAS_RAF1_norfitness_6VJJ is the dropped duplicate '
                      '(same RAF1 partner, a 63-position construct)'),
    'PSD95': dict(partners=('PSD95_CRIPT_1BE9', 'PSD95_Tm2F_1BE9'),
                  channel='samesite', structure=True,
                  note='max_mut = 1 => same-site channel only (spec Sec.1.5)'),
    'BH3': dict(partners=('BH3_Bcl-xL_normed_1PQ1', 'BH3_Mcl-1_normed_3KZ0'),
                channel='nested', structure=True,
                note='join at the WT-verified offset (G1b); the naive '
                     'no-offset join gives 97/518 and is banned'),
    '5A12': dict(partners=('5A12_VEGF_fitness_4ZFF', '5A12_Ang2_fitness_4ZFG'),
                 channel='nested', structure=True,
                 note='VEGF is the DESIGNED C4 negative control (0/9 positions '
                      'at the VEGF interface, dSASA exactly 0)'),
    'CR9114': dict(partners=('CR9114_FluAH1_logKd_4FQI', 'CR9114_FluAH3_logKd_4FQY'),
                   channel='nested', structure=False,
                   note='CENSORING-LIMITED ONLY: H3 is 89.05% floored; and no '
                        'PDB in structures/ => W is undefined, M_F not computable '
                        '(spec G-OPT)'),
}


def family_matrix(family, *, t09=None, tau=TAU_PRIMARY, verbose=False):
    """``Z`` (logit cliff rate), ``W`` (-min heavy distance), ``K``, ``N`` and
    every per-cell input C4-I needs, on the WT-verified common frame."""
    spec = C4I_FAMILIES[family]
    partners = list(spec['partners'])
    if t09 is None:
        t09 = read_T09()
    offs, ref = resolve_offsets(partners)
    cnt = {d: partner_counts(d, spec['channel'], tau=tau) for d in partners}
    # per-partner: (chain_ref, pos_ref) -> row
    per = {}
    for d in partners:
        amap = aligned_positions(d, offs)
        loc = {(str(r['chain']), int(r['seq_idx'])): r for r in cnt[d].to_dict('records')}
        st = t09[t09['DMS_id'].astype(str) == d]
        srow = {(str(r['chain']), int(r['seq_idx'])): r for r in st.to_dict('records')}
        per[d] = dict(map={}, struct={})
        for kref, kloc in amap.items():
            if kloc in loc:
                per[d]['map'][kref] = loc[kloc]
            if kloc in srow:
                per[d]['struct'][kref] = srow[kloc]
    common = set(per[partners[0]]['map'])
    for d in partners[1:]:
        common &= set(per[d]['map'])
    keys = sorted(common)
    K = len(partners)
    J = len(keys)
    n_mat = np.zeros((J, K)); k_mat = np.zeros((J, K))
    w_mat = np.full((J, K), np.nan); rsa = np.full((J, K), np.nan)
    ifc = np.zeros((J, K), dtype=bool)
    for a, d in enumerate(partners):
        for j, kk in enumerate(keys):
            r = per[d]['map'][kk]
            n_mat[j, a] = r['n_pairs_at_site']
            k_mat[j, a] = r['n_cliff_pairs']
            s = per[d]['struct'].get(kk)
            if s is not None:
                w_mat[j, a] = -float(s['min_heavy_dist'])
                rsa[j, a] = float(s['rsa_iso'])
                ifc[j, a] = bool(s['is_iface_5A'])
    Z = emp_logit(k_mat, n_mat)
    out = dict(family=family, partners=partners, ref=ref, offsets=offs,
               keys=keys, J=J, K=K, Z=Z, W=w_mat, n=n_mat, k=k_mat,
               rsa=rsa, iface=ifc, channel=spec['channel'],
               structure=bool(spec['structure']), note=spec['note'],
               n_positions_union=len(set().union(*[set(per[d]['map']) for d in partners])),
               n_complete_cells=int((n_mat > 0).all(axis=1).sum()),
               counts=cnt)
    if verbose:
        print('[C4-I] %-7s K=%d  aligned J=%d (union %d, all-n>0 %d)  channel=%s'
              % (family, K, J, out['n_positions_union'],
                 out['n_complete_cells'], spec['channel']))
    return out


def variance_decomposition(M, sigma2_noise):
    """One-way (group x partner) decomposition ``x_gk = mu_g + delta_gk + noise``.

    ``E[MSW] = Var(delta) + sigma2_noise``, ``E[MSB] = K Var(mu) + Var(delta) +
    sigma2_noise``, so ``Var(mu) = (MSB - MSW)/K`` and the noise-corrected
    ``Var(delta) = MSW - sigma2_noise``.  ``F_spec = Var(delta)/(Var(mu)+Var(delta))``.
    """
    M = np.asarray(M, dtype=np.float64)
    ok = np.isfinite(M).all(axis=1)
    M = M[ok]
    G, K = M.shape
    if G < 2 or K < 2:
        return dict(n_groups=int(G), K=int(K), MSB=float('nan'), MSW=float('nan'),
                    var_mu=float('nan'), var_delta=float('nan'),
                    F_spec=float('nan'), F_spec_noise_corrected=float('nan'),
                    var_delta_raw=float('nan'), sigma2_noise=float(sigma2_noise))
    gm = M.mean(axis=1)
    grand = M.mean()
    ssw = float(((M - gm[:, None]) ** 2).sum())
    ssb = float(K * ((gm - grand) ** 2).sum())
    msw = ssw / (G * (K - 1))
    msb = ssb / (G - 1)
    var_mu = (msb - msw) / K
    vd_raw = msw
    vd = msw - float(sigma2_noise)
    def _f(vdelta):
        den = max(var_mu, 0.0) + vdelta
        return float(vdelta / den) if den > 0 else float('nan')
    # ``MSB < MSW`` puts the method-of-moments ``Var(mu)`` BELOW ZERO.  Clamping
    # it to 0 then makes ``F_spec`` exactly 1.0 -- which reads as "completely
    # partner-specific" but actually means "no partner-invariant component is
    # estimable at all", i.e. the estimate is AT THE BOUNDARY.  The flag says so,
    # because 1.0000 quoted without it would be a false positive for C4-I.
    var_mu_neg = bool(var_mu < 0)
    return dict(n_groups=int(G), K=int(K), MSB=msb, MSW=msw, var_mu=var_mu,
                var_delta=vd, var_delta_raw=vd_raw,
                F_spec=_f(vd_raw), F_spec_noise_corrected=_f(max(vd, 0.0)),
                var_delta_negative=bool(vd < 0),
                var_mu_negative=var_mu_neg,
                F_spec_at_boundary=var_mu_neg,
                sigma2_noise=float(sigma2_noise),
                icc_mu=float(max(var_mu, 0.0) / (max(var_mu, 0.0) + vd_raw))
                if (max(var_mu, 0.0) + vd_raw) > 0 else float('nan'))


def fspec_eps(family, fm, *, verbose=False):
    """``F_spec`` on the EPS scale -- the spec's literal definition, available
    only where the SAME substitution pair is measured against every partner."""
    partners = fm['partners']
    offs = fm['offsets']
    single = all(offs[d]['single_chain'] for d in partners)
    tabs = {}
    for d in partners:
        a = io_bgym.load_assay(d)
        ep = _noise.epsilon_sitepairs(a)
        if ep['n_usable'] == 0:
            return dict(available=False,
                        reason='%s: %s' % (d, ep['reason'] or 'no usable eps'))
        info = offs[d]
        dd = {}
        for kk, e in zip(ep['keys'], ep['eps']):
            ak = []
            for (c, p, aa) in kk:
                cr = info['chain_map'].get(c, c)
                ak.append((p + info['offset'].get(c, 0), aa) if single
                          else (cr, p + info['offset'].get(c, 0), aa))
            dd[tuple(sorted(ak))] = float(e)
        tabs[d] = dd
    shared = sorted(set.intersection(*[set(tabs[d]) for d in partners]))
    if len(shared) < 4:
        return dict(available=False, n_shared=len(shared),
                    reason='only %d substitution pairs shared by all %d partners'
                           % (len(shared), len(partners)))
    M = np.array([[tabs[d][kk] for d in partners] for kk in shared])
    s2 = float(THRESH['C4I_sigma_eps_sq'])
    res = variance_decomposition(M, s2)
    # per-partner standardised variant (a partner whose eps spread is simply
    # wider would otherwise look "partner-specific" for a scale reason)
    sc = np.array([1.4826 * np.median(np.abs(M[:, a] - np.median(M[:, a])))
                   for a in range(M.shape[1])])
    sc[sc <= 0] = 1.0
    Ms = M / sc
    res_s = variance_decomposition(Ms, float(np.mean(s2 / sc ** 2)))
    out = dict(available=True, n_shared=len(shared), sigma2_noise=s2,
               F_spec_at_boundary=bool(res['F_spec_at_boundary']),
               eps_sd_per_partner=[float(np.std(M[:, a], ddof=1)) for a in range(M.shape[1])],
               eps_madscale_per_partner=[float(x) for x in sc], scaled=res_s)
    out.update(res)
    if verbose:
        print('      F_spec(eps): shared %d  MSW %.5f MSB %.5f  var_mu %.5f  '
              'F_spec %.4f  noise-corrected %.4f'
              % (len(shared), res['MSW'], res['MSB'], res['var_mu'],
                 res['F_spec'], res['F_spec_noise_corrected']))
    return out


def psi_table(fm, *, alpha=0.05):
    """``PSI_j`` and the per-cell cliff flag.

    A position is "a cliff in partner ``a``" iff its cliff count is enriched
    against THAT partner's own assay-wide rate: one-sided binomial
    ``P(X >= k_ja | n_ja, rate_a) < alpha``.  The cruder "``>= 1`` cliff pair"
    definition is reported beside it (with 1-3% assay-wide rates and hundreds of
    pairs per position it flags nearly every position, which is why it is not
    primary)."""
    n, k = fm['n'], fm['k']
    J, K = n.shape
    rate = np.array([(k[:, a].sum() / n[:, a].sum()) if n[:, a].sum() > 0 else np.nan
                     for a in range(K)])
    flag = np.zeros((J, K), dtype=bool)
    pval = np.full((J, K), np.nan)
    for a in range(K):
        if not np.isfinite(rate[a]) or rate[a] <= 0:
            continue
        pv = _sps.binom.sf(k[:, a] - 1, n[:, a].astype(int), rate[a])
        pval[:, a] = pv
        flag[:, a] = (pv < alpha) & (n[:, a] > 0)
    any1 = k > 0
    have = n > 0
    Kj = have.sum(axis=1).astype(float)
    with np.errstate(invalid='ignore'):
        PSI = np.where(Kj > 0, flag.sum(axis=1) / Kj, np.nan)
        PSI_any = np.where(Kj > 0, (any1 & have).sum(axis=1) / Kj, np.nan)
    return dict(flag=flag, pval=pval, rate=rate, PSI=PSI, PSI_any1=PSI_any,
                K_partners=Kj, is_cliff_pos=flag.any(axis=1),
                is_cliff_pos_any1=(any1 & have).any(axis=1))


def psi_tests(fm, psi, *, B=None, dms_seed=None):
    """The PSI clause, and the reason its literal form is degenerate.

    Spec Sec.1.5 asks for "cliff-position PSI stochastically below non-cliff
    PSI (one-sided Mann-Whitney p < 0.05)".  Taken literally that CANNOT fire:
    a non-cliff position is by construction one flagged in ZERO partners, so its
    ``PSI`` is exactly 0 and no distribution can be stochastically below it.
    Both are therefore reported:

    * ``MW_PSI_p_literal`` -- the literal one-sided Mann-Whitney (cliff < non-cliff).
    * ``MW_PSI_p`` (primary) -- a PERMUTATION test of the same substantive claim
      "cliffs do not recur across partners": the cliff flag is permuted WITHIN
      each partner column independently (partner-independence), and
      ``p_below = (1 + #{mean PSI_b <= mean PSI_obs})/(B+1)``.  Its mirror
      ``p_above`` is the test of the stability-cliff alternative (EXCESS
      recurrence) and is the number that actually discriminates.
    """
    B = B_NS3 if B is None else int(B)
    flag, have = psi['flag'], (fm['n'] > 0)
    J, K = flag.shape
    Kj = psi['K_partners']
    cl = psi['is_cliff_pos']
    obs = psi['PSI']
    out = dict(n_cliff_pos=int(cl.sum()), n_noncliff_pos=int((~cl).sum()),
               median_cliff_PSI=float(np.nanmedian(obs[cl])) if cl.any() else float('nan'),
               mean_cliff_PSI=float(np.nanmean(obs[cl])) if cl.any() else float('nan'),
               median_PSI_all=float(np.nanmedian(obs)),
               median_cliff_PSI_any1=float(np.nanmedian(psi['PSI_any1'][psi['is_cliff_pos_any1']]))
               if psi['is_cliff_pos_any1'].any() else float('nan'))
    if cl.any() and (~cl).any():
        u = _sps.mannwhitneyu(obs[cl], obs[~cl], alternative='less')
        out['MW_PSI_p_literal'] = float(u.pvalue)
        out['MW_U_literal'] = float(u.statistic)
    else:
        out['MW_PSI_p_literal'] = float('nan')
        out['MW_U_literal'] = float('nan')
    rng = _rng('perm_NS3', dms_seed) if dms_seed else _rng('perm_NS3')
    mean_obs = out['mean_cliff_PSI']
    null = np.full(B, np.nan)
    for b in range(B):
        fb = np.zeros_like(flag)
        for a in range(K):
            rows = np.nonzero(have[:, a])[0]
            m = int(flag[rows, a].sum())
            if m:
                fb[rng.choice(rows, size=m, replace=False), a] = True
        cb = fb.any(axis=1)
        if cb.any():
            with np.errstate(invalid='ignore'):
                null[b] = float(np.nanmean(np.where(Kj > 0, fb.sum(axis=1) / Kj, np.nan)[cb]))
    p_below, nb = empirical_p(mean_obs, null, side='less')
    p_above, _ = empirical_p(mean_obs, null, side='greater')
    out.update(MW_PSI_p=p_below, p_PSI_below=p_below, p_PSI_above=p_above,
               PSI_null_mean=float(np.nanmean(null)), PSI_null_B=int(nb),
               PSI_literal_note=('non-cliff PSI is identically 0 by construction, '
                                 'so the spec\'s literal "stochastically below" '
                                 'clause cannot fire; MW_PSI_p carries the '
                                 'partner-independence permutation p instead'))
    return out


def c4i_family(family, *, t09=None, tau=TAU_PRIMARY, B=None, verbose=False):
    """Everything T11 needs for one family."""
    fm = family_matrix(family, t09=t09, tau=tau, verbose=verbose)
    B = B_NS3 if B is None else int(B)
    Z, W, n = fm['Z'], fm['W'], fm['n']
    ok = (n > 0).all(axis=1) & np.isfinite(Z).all(axis=1)
    if fm['structure']:
        ok = ok & np.isfinite(W).all(axis=1)
    Zc, Wc = Z[ok], W[ok]
    res = dict(family=family, J_aligned=fm['J'], J_used=int(ok.sum()),
               K=fm['K'], channel=fm['channel'], partners=fm['partners'],
               structurally_mute=(not fm['structure']), note=fm['note'],
               ref_assay=fm['ref'],
               offsets={d: fm['offsets'][d]['offset'] for d in fm['partners']},
               offsets_unique={d: bool(fm['offsets'][d].get('unique_at_max'))
                               for d in fm['partners']})
    if fm['structure'] and ok.sum() >= 4:
        M = mantel_corr(Zc, Wc)
        rng = _rng('perm_NS3', family)
        null = np.empty(B)
        Wt = double_centre(Wc).ravel()
        for b in range(B):
            null[b] = mantel_corr(_nulls.permute_NS3(Zc, rng), Wc)
        p1, nb = empirical_p(M, null, side='greater')
        p2, _ = empirical_p(M, null, side='two-sided')
        res.update(M_F=M, family_p_NS3=p1, p_NS3_two_sided=p2,
                   NS3_B=int(nb), NS3_null_mean=float(np.nanmean(null)),
                   NS3_null_sd=float(np.nanstd(null)),
                   NS3_exact_signflip=(fm['K'] == 2))
    else:
        res.update(M_F=float('nan'), family_p_NS3=float('nan'),
                   p_NS3_two_sided=float('nan'), NS3_B=0,
                   NS3_null_mean=float('nan'), NS3_null_sd=float('nan'),
                   NS3_exact_signflip=False)
    # ---- fold-axis validation (REQUIRED) ---------------------------------- #
    rowmean = np.nanmean(Z, axis=1)
    rsa = np.nanmean(fm['rsa'], axis=1)
    res['foldaxis_spearman_rowmean_rsa'] = spearman(rowmean[ok], rsa[ok])
    res['foldaxis_spearman_rowmean_burial'] = spearman(rowmean[ok], -rsa[ok])
    res['rowmean_Z_mean'] = float(np.nanmean(rowmean[ok])) if ok.any() else float('nan')
    res['rsa_spread_across_partners'] = float(np.nanmax(
        np.nanmax(fm['rsa'], axis=1) - np.nanmin(fm['rsa'], axis=1))) if fm['J'] else float('nan')
    # ---- PSI --------------------------------------------------------------- #
    psi = psi_table(fm)
    res.update(psi_tests(fm, psi, B=B, dms_seed=family))
    res['per_partner_rate'] = [float(x) for x in psi['rate']]
    # ---- F_spec ------------------------------------------------------------ #
    fe = fspec_eps(family, fm, verbose=verbose)
    zdec = _fspec_Z(fm)
    res['F_spec_Z'] = zdec['F_spec']
    res['F_spec_Z_noise_corrected'] = zdec['F_spec_noise_corrected']
    res['F_spec_Z_at_boundary'] = bool(zdec['F_spec_at_boundary'])
    res['F_spec_var_mu_Z'] = zdec['var_mu']
    res['F_spec_Z_var_mu'] = zdec['var_mu']
    res['F_spec_Z_var_delta'] = zdec['var_delta']
    res['F_spec_Z_sampling_var'] = zdec['sigma2_noise']
    if fe.get('available'):
        res.update(F_spec=fe['F_spec'],
                   F_spec_noise_corrected=fe['F_spec_noise_corrected'],
                   F_spec_at_boundary=bool(fe['F_spec_at_boundary']),
                   F_spec_scale='eps', F_spec_n_shared=fe['n_shared'],
                   F_spec_var_mu=fe['var_mu'], F_spec_var_delta=fe['var_delta'],
                   F_spec_MSW=fe['MSW'], F_spec_MSB=fe['MSB'],
                   F_spec_eps_scaled=fe['scaled']['F_spec_noise_corrected'],
                   F_spec_note='')
    else:
        res.update(F_spec=zdec['F_spec'],
                   F_spec_noise_corrected=zdec['F_spec_noise_corrected'],
                   F_spec_at_boundary=bool(zdec['F_spec_at_boundary']),
                   F_spec_scale='Z', F_spec_n_shared=int(ok.sum()),
                   F_spec_var_mu=zdec['var_mu'], F_spec_var_delta=zdec['var_delta'],
                   F_spec_MSW=zdec['MSW'], F_spec_MSB=zdec['MSB'],
                   F_spec_eps_scaled=float('nan'),
                   F_spec_note='eps scale unavailable (%s); F_spec is on the '
                               'Z = logit(cliff rate) scale, noise-corrected '
                               'with the delta-method sampling variance'
                               % fe.get('reason', ''))
    res['_fm'] = fm
    res['_psi'] = psi
    return res


def _fspec_Z(fm):
    """``F_spec`` on the ``Z = logit(cliff rate)`` scale, noise-corrected with
    the DELTA-METHOD sampling variance ``1/(k+.5) + 1/(n-k+.5)``.

    Available in every family (the eps scale is not), and its noise term is the
    binomial sampling noise of the cliff rate itself rather than an imported
    ``sigma_eps``."""
    Z, n, k = fm['Z'], fm['n'], fm['k']
    ok = (n > 0).all(axis=1) & np.isfinite(Z).all(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        v = 1.0 / (k + 0.5) + 1.0 / (n - k + 0.5)
    s2 = float(np.nanmean(v[ok])) if ok.any() else float('nan')
    return variance_decomposition(Z[ok], s2)


# =========================================================================== #
# 9. artifacts -- T09's CLIFF-DERIVED columns, T10, T11                        #
# =========================================================================== #

T09_NAME = 'T09_structure_sites.csv'
T10_NAME = 'T10_structure_pairs.csv'
T11_NAME = 'T11_partner_specificity.csv'

#: T09's per-POSITION cliff columns (spec Sec.6) -- ``cliff_counts`` supplies them.
T09_SITE_COLUMNS = ('n_pairs_at_site', 'n_cliff_pairs', 'cliff_rate', 'beta_hat_abs')

#: T09's per-ASSAY C4-S columns (spec Sec.6), broadcast to the assay's rows.
#: They carry the PRE-REGISTERED 5.0 A interface definition, which is the one
#: whose GLM ``iface`` coefficient is identified (with ``iface = dSASA > 0`` the
#: Levy dummies alias it exactly).  D6's second, co-primary definition is
#: carried in the ``*_dsasa`` columns of :data:`T09_EXTRA_COLUMNS` -- side by
#: side, never as a replacement.
T09_ASSAY_COLUMNS = ('OR_burial_matched', 'OR_lo95', 'OR_hi95',
                     'beta_iface_unadj', 'beta_iface_adj', 'p_wald', 'p_NS1',
                     'beta_iface_after_rsa')

#: Columns this module APPENDS to T09 beyond the spec's list.  ``verdict.py``
#: already reads ``p_wald_after_rsa`` "when stats_c4.py provides it" (its own
#: docstring), and D6 needs the second interface definition persisted rather
#: than only printed.  Everything here is cliff-derived; no structural column
#: is ever added, removed or rewritten.
T09_EXTRA_COLUMNS = (
    'p_wald_after_rsa',                  # the kill switch's own p (verdict.py reads this first)
    'iface_def_primary',                 # which definition the spec columns carry
    'n_cliff_tau4', 'cliff_rate_tau4',   # TAU_SECONDARY, so no conclusion rests on one cut
    'crude_OR', 'OR_lo95_rbg', 'OR_hi95_rbg', 'p_wald_MH',
    'NS1_strata_level', 'NS1_frac_exchangeable', 'NS1_B',
    'OR_burial_matched_dsasa', 'OR_lo95_dsasa', 'OR_hi95_dsasa',
    'beta_iface_unadj_dsasa', 'beta_iface_after_rsa_dsasa',
    'p_wald_unadj_dsasa', 'p_wald_after_rsa_dsasa', 'p_NS1_dsasa',
    'glm_iface_aliased_dsasa',
    'c4s_stamp',
)

_T09_ALL_CLIFF = (tuple(_structure.T09_PENDING_COLUMNS) + T09_EXTRA_COLUMNS)

#: The orchestrator's stamps.  A stamped assay's numbers are REPORTED but
#: contribute to nothing (they are not in any k-of-n denominator).
STRUCTURALLY_UNIDENTIFIED = {
    'CD19_FMC63_Fitness_7URV':
        'only 62.04% of rows have a finite out-of-fold phi (ORCHESTRATOR)',
}

#: Out of C4-S BY CONSTRUCTION, with the reason, so "not run" is never confused
#: with "run and null".
C4S_OUT_BY_CONSTRUCTION = {
    'Z-domain_ZpA963_HL1_fitness_2M5A':
        '6/6 mutated positions are interface -- the contrast is unfalsifiable',
    'GB1_IgG-Fc_fitness_1FCC_2016':
        'only 4 mutated positions',
    '5A12_VEGF_fitness_4ZFF':
        'DESIGNED NEGATIVE CONTROL: 0/9 mutated positions at the VEGF interface '
        '(dSASA exactly 0.000 at all nine)',
}


def artifact_path(name):
    """Absolute path of one artifact table under ``PATHS.artifacts``."""
    return os.path.join(PATHS.artifacts, name)


_T09_CACHE = {}


def read_T09(*, refresh=False):
    """T09's STRUCTURAL columns -- stage 1's output, with every CLIFF-DERIVED
    column stripped.

    Stripping matters: stage 1 writes the cliff columns as EMPTY placeholders,
    so a merge that kept them would silently produce ``n_pairs_at_site_x`` /
    ``_y`` and a second run of this stage would join against its own output.
    """
    if refresh or 'df' not in _T09_CACHE:
        p = artifact_path(T09_NAME)
        if not os.path.exists(p):
            raise SystemExit('[refuse] %s does not exist -- run stage 1 first' % p)
        df = pd.read_csv(p)
        drop = [c for c in df.columns if c in set(_T09_ALL_CLIFF)]
        _T09_CACHE['df'] = df.drop(columns=drop)
    return _T09_CACHE['df']


def _cell(v):
    """One CSV cell.  ``None`` / non-finite -> the empty string (stage 1's own
    placeholder), integers stay integral, floats get 10 significant digits."""
    if v is None:
        return ''
    if isinstance(v, (bool, np.bool_)):
        return 'True' if bool(v) else 'False'
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return '%d' % int(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return ''
    return '%.10g' % f


class _ArtifactLock(object):
    """``flock`` on one artifact, so two stages writing different tables in the
    same directory cannot interleave a read-modify-write of the SAME file."""

    def __init__(self, name):
        os.makedirs(os.path.join(PATHS.cache, '.locks'), exist_ok=True)
        self.path = os.path.join(PATHS.cache, '.locks', name + '.lock')
        self.fh = None

    def __enter__(self):
        import fcntl
        self.fh = open(self.path, 'a+')
        fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        import fcntl
        if self.fh is not None:
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
            self.fh.close()
            self.fh = None
        return False


def write_T09_cliff_columns(per_site, per_assay, *, verbose=True):
    """Fill T09's CLIFF-DERIVED columns IN PLACE and touch nothing else.

    The file is read back as RAW STRINGS, so every structural column stage 1
    wrote is re-emitted byte-for-byte -- a pandas float round-trip would
    silently reformat ``rsa_iso`` and friends, and this stage does not own them.
    The equality of the structural block before and after is ASSERTED.
    """
    p = artifact_path(T09_NAME)
    with _ArtifactLock('T09'):
        raw = pd.read_csv(p, dtype=str, keep_default_na=False, na_filter=False)
        struct = [c for c in raw.columns if c not in set(_T09_ALL_CLIFF)]
        before = raw[struct].copy()
        site = {(str(r['DMS_id']), str(r['chain']), int(r['seq_idx'])): r
                for r in per_site.to_dict('records')} if len(per_site) else {}
        asy = {str(r['DMS_id']): r for r in per_assay.to_dict('records')} \
            if len(per_assay) else {}
        keys = list(zip(raw['DMS_id'].tolist(), raw['chain'].tolist(),
                        [int(x) for x in raw['seq_idx'].tolist()]))
        n_site_hit = sum(1 for k in keys if k in site)
        n_assay_hit = sum(1 for k in keys if k[0] in asy)
        for col in _T09_ALL_CLIFF:
            vals = []
            for k in keys:
                r = site.get(k)
                if r is not None and col in r:
                    vals.append(_cell(r[col]))
                    continue
                r = asy.get(k[0])
                if r is not None and col in r:
                    vals.append(_cell(r[col]))
                    continue
                vals.append('')
            raw[col] = vals
        # the spec's own column order first, then this module's appended ones
        order = ([c for c in _structure.T09_COLUMNS if c in raw.columns]
                 + [c for c in T09_EXTRA_COLUMNS if c in raw.columns])
        missing = [c for c in raw.columns if c not in order]
        order = order + missing
        raw = raw[order]
        assert raw[struct].equals(before), \
            'a structural column changed -- stage 6 must never rewrite stage 1'
        tmp = p + '.tmp%d' % os.getpid()
        raw.to_csv(tmp, index=False)
        os.replace(tmp, p)
    if verbose:
        print('[T09] filled %d cliff-derived columns; %d/%d rows got per-site '
              'counts, %d/%d got assay-level C4-S scalars  -> %s'
              % (len(_T09_ALL_CLIFF), n_site_hit, len(raw), n_assay_hit,
                 len(raw), os.path.relpath(p, config.REPO)))
    return dict(path=p, n_rows=len(raw), n_site_filled=n_site_hit,
                n_assay_filled=n_assay_hit, columns=list(raw.columns))


#: T10's column list (spec Sec.6), verbatim and in order, then the extras that
#: are actually used downstream.  The per-row string columns of
#: :func:`epsilon_structure` (``sigma_provenance``, ``sigma_alt_label``) are
#: assay-level constants and are NOT written 200,000 times.
T10_COLUMNS = ['DMS_id', 'site_s', 'site_t', 'aa_s', 'aa_t', 'seq_separation',
               'd3d_min_heavy', 'levy_s', 'levy_t', 'both_iface',
               'n_backgrounds', 'n_aa_combos', 'eps', 'eps_z',
               'is_cliff_3sigma', 'ICC_sitepair', 'AUROC_contribution', 'p_NS2']
T10_EXTRA_COLUMNS = ['chain_s', 'chain_t', 'both_iface_dsasa',
                     'is_noncliff_1sigma', 'is_cliff_3sigma_alt',
                     'site_pair_id', 'sigma_eps_used']


def write_T10(frames, *, verbose=True):
    """T10: one row per usable ``eps``, in the spec's column order.

    Written with ``float_format='%.6g'`` -- six significant digits is well past
    the resolution of any DMS score in the benchmark, and the table is large
    enough (>= 10^5 rows) that full repr would double the file for no
    information.
    """
    if not frames:
        return dict(path=artifact_path(T10_NAME), n_rows=0)
    d = pd.concat(frames, ignore_index=True)
    cols = [c for c in T10_COLUMNS + T10_EXTRA_COLUMNS if c in d.columns]
    d = d[cols]
    p = artifact_path(T10_NAME)
    with _ArtifactLock('T10'):
        tmp = p + '.tmp%d' % os.getpid()
        d.to_csv(tmp, index=False, float_format='%.6g')
        os.replace(tmp, p)
    if verbose:
        print('[T10] %d rows x %d cols, %.1f MB -> %s'
              % (len(d), len(cols), os.path.getsize(p) / 1e6,
                 os.path.relpath(p, config.REPO)))
    return dict(path=p, n_rows=int(len(d)), n_cols=len(cols),
                bytes=os.path.getsize(p))


T11_COLUMNS = (['family', 'chain', 'resseq', 'icode', 'wt_aa', 'K_partners']
               + ['cliff_rate_p%d' % i for i in (1, 2, 3, 4)]
               + ['min_heavy_dist_p%d' % i for i in (1, 2, 3, 4)]
               + ['iface_flag_p%d' % i for i in (1, 2, 3, 4)]
               + ['PSI']
               + ['Z_doublecentered_p%d' % i for i in (1, 2, 3, 4)]
               + ['rowmean_Z', 'rsa_iso', 'family_M_stat', 'family_p_NS3',
                  'foldaxis_spearman_rowmean_rsa', 'F_spec',
                  'F_spec_noise_corrected', 'MW_PSI_p',
                  'twin_structure_OR_8BE4', 'twin_structure_OR_5O2S',
                  'classification'])

#: appended by this module: the numbers the spec's own C4-I clauses need but
#: whose columns Sec.6 forgot (``median_cliff_PSI`` is read by ``verdict.py``).
T11_EXTRA_COLUMNS = ['F_spec_at_boundary', 'partners', 'channel',
                     'J_aligned', 'J_used',
                     'median_cliff_PSI', 'mean_cliff_PSI', 'PSI_any1',
                     'p_PSI_below', 'p_PSI_above', 'MW_PSI_p_literal',
                     'F_spec_scale', 'F_spec_n_shared', 'F_spec_Z',
                     'F_spec_Z_noise_corrected', 'p_NS3_two_sided', 'NS3_B',
                     'NS3_null_mean', 'NS3_null_sd', 'n_cliff_pos',
                     'n_noncliff_pos', 'cliff_flag_p1', 'cliff_flag_p2',
                     'cliff_flag_p3', 'cliff_flag_p4',
                     'n_pairs_p1', 'n_pairs_p2', 'n_pairs_p3', 'n_pairs_p4',
                     'structurally_mute', 'twin_structure_note', 'note']


def write_T11(rows, *, verbose=True):
    """T11: one row per (family, aligned position), in the spec's column order,
    with the family-level scalars broadcast down the family's rows (which is
    how ``verdict.py`` reads them -- ``_first_num`` over the group)."""
    d = pd.DataFrame(rows)
    for c in T11_COLUMNS + T11_EXTRA_COLUMNS:
        if c not in d.columns:
            d[c] = np.nan
    d = d[[c for c in T11_COLUMNS + T11_EXTRA_COLUMNS]]
    p = artifact_path(T11_NAME)
    with _ArtifactLock('T11'):
        tmp = p + '.tmp%d' % os.getpid()
        d.to_csv(tmp, index=False, float_format='%.10g')
        os.replace(tmp, p)
    if verbose:
        print('[T11] %d rows x %d cols -> %s'
              % (len(d), d.shape[1], os.path.relpath(p, config.REPO)))
    return dict(path=p, n_rows=int(len(d)))


# =========================================================================== #
# 10. G11 -- the twin-structure control                                        #
# =========================================================================== #

#: The KRAS score table registered against TWO complexes with byte-identical
#: scores (spec Sec.2 rows 7 and 18).  The cliff counts can only come from the
#: side that has a latent fit; the ANNOTATION is what switches.
G11_SCORE_ASSAY = 'KRAS_SOS1_norfitness_8BE4'
G11_ANNOTATIONS = (('8BE4', 'KRAS_SOS1_norfitness_8BE4', 'SOS1'),
                   ('5O2S', 'KRAS_DARPinK27_norfitness_5O2S', 'DARPinK27'))
#: two ORs count as "similar" when neither is below the C4-S support cut and
#: their ratio sits inside this band -- declared here, not chosen afterwards
G11_OR_RATIO_BAND = (1.0 / 1.5, 1.5)


def g11_twin_structures(t09=None, *, tau=TAU_PRIMARY, B=None, B_boot=None,
                        verbose=True):
    """The G11 control: ONE score table, TWO interface localisations.

    KRAS_SOS1_norfitness_8BE4 and KRAS_DARPinK27_norfitness_5O2S share 19,227
    keys with byte-identical scores (T01's own exclusion reason), so the cliff
    positions are literally the same set of positions in both.  At most one of
    the two interfaces can be the CAUSE of their localisation.  If the cliffs
    localise to BOTH with similar odds ratios, interface localisation is
    non-causal for KRAS -- a finding, not a stop.
    """
    if t09 is None:
        t09 = read_T09()
    cc = cliff_counts(G11_SCORE_ASSAY, centred=True)
    out = dict(score_assay=G11_SCORE_ASSAY, tau=tau, per_annotation={},
               n_positions_scored=int(len(cc)))
    for tag, annot_id, partner in G11_ANNOTATIONS:
        st = t09[t09['DMS_id'].astype(str) == annot_id].copy()
        if not len(st):
            out['per_annotation'][tag] = dict(available=False,
                                              reason='no T09 rows for %s' % annot_id)
            continue
        st['chain'] = st['chain'].astype(str)
        st['seq_idx'] = st['seq_idx'].astype(np.int64)
        # the two annotations are of the SAME construct, so the join is on
        # seq_idx; the chain letter is the PDB's and differs between complexes.
        by_pos = {int(r['seq_idx']): r for r in st.to_dict('records')}
        rows = []
        for r in cc.to_dict('records'):
            s = by_pos.get(int(r['seq_idx']))
            if s is None:
                continue
            q = dict(r)
            q.update({k: s[k] for k in ('levy_class', 'rsa_iso', 'dsasa',
                                        'min_heavy_dist', 'is_iface_5A',
                                        'is_iface_dsasa', 'rsa_decile',
                                        'aa_class', 'depth_tertile')})
            q['chain_pdb'] = s['chain']
            rows.append(q)
        use = pd.DataFrame(rows)
        use['rsa_tertile'] = _rank_bins(use['rsa_iso'].values, 3)
        use['beta_decile'] = _rank_bins(use['beta_hat_abs'].values, 10)
        use['beta_tertile'] = _rank_bins(use['beta_hat_abs'].values, 3)
        rec = dict(available=True, annot_assay=annot_id, partner=partner,
                   n_joined=int(len(use)),
                   n_iface_5A=int(use['is_iface_5A'].astype(bool).sum()),
                   n_iface_dsasa=int(use['is_iface_dsasa'].astype(bool).sum()))
        for nm, col in IFACE_DEFS:
            o = c4s_or(use, col, tau=tau, dms_id=annot_id, B=B_boot)
            p = ns1_pvalue(use, col, tau=tau, dms_id=annot_id, B=B,
                           obs=o['OR_burial_matched'], with_glm=False)
            m, _ = c4s_models(use, col, tau=tau)
            rec[nm] = dict(OR=o['OR_burial_matched'], lo=o['OR_lo95'],
                           hi=o['OR_hi95'], crude=o['crude_OR'],
                           p_NS1=p['p_NS1'], NS1_B=p['NS1_B'],
                           rate_iface=o['cliff_rate_iface'],
                           rate_noniface=o['cliff_rate_noniface'],
                           beta_after_rsa=m['beta_iface_after_rsa'],
                           p_after_rsa=m['p_wald_after_rsa'])
        out['per_annotation'][tag] = rec
    a = out['per_annotation'].get('8BE4', {})
    b = out['per_annotation'].get('5O2S', {})
    for nm, _col in IFACE_DEFS:
        oa = a.get(nm, {}).get('OR', float('nan'))
        ob = b.get(nm, {}).get('OR', float('nan'))
        ratio = (oa / ob) if (np.isfinite(oa) and np.isfinite(ob) and ob > 0) \
            else float('nan')
        both_up = (np.isfinite(oa) and np.isfinite(ob)
                   and oa >= THRESH['C4S_OR_sup'] and ob >= THRESH['C4S_OR_sup'])
        similar = bool(np.isfinite(ratio)
                       and G11_OR_RATIO_BAND[0] <= ratio <= G11_OR_RATIO_BAND[1])
        # THREE outcomes, not two.  ``dual_localisation`` (both localisations
        # strong AND indistinguishable) is the spec's refuting flag, but the
        # case that actually occurs here is SIMILAR-BUT-NEITHER-STRONG, and that
        # is NOT a pass: two indistinguishable odds ratios from ONE score table
        # still mean the design cannot attribute the localisation to either
        # interface.  Printing only ``dual_localisation = False`` would read as
        # "G11 cleared", which is the opposite of what was measured.
        if both_up and similar:
            verdict = 'dual_localisation'
        elif similar:
            verdict = 'not_identifiable'
        else:
            verdict = 'distinguishable'
        stronger = ('5O2S' if (np.isfinite(ob)
                               and (not np.isfinite(oa) or ob > oa))
                    else '8BE4')
        out['g11_' + nm] = dict(OR_8BE4=oa, OR_5O2S=ob, ratio=ratio,
                                both_above_sup=bool(both_up), similar=similar,
                                verdict=verdict, stronger=stronger,
                                dual_localisation=bool(both_up and similar))
    out['g11_dual'] = bool(out['g11_5A']['dual_localisation']
                           or out['g11_dsasa']['dual_localisation'])
    if verbose:
        print('\n[G11] one score table (%s), two interface localisations'
              % G11_SCORE_ASSAY)
        for tag, annot_id, partner in G11_ANNOTATIONS:
            r = out['per_annotation'].get(tag, {})
            if not r.get('available'):
                print('      %-5s UNAVAILABLE (%s)' % (tag, r.get('reason')))
                continue
            print('      %-5s (%-10s) joined %3d positions, iface 5A %3d / '
                  'dSASA %3d' % (tag, partner, r['n_joined'], r['n_iface_5A'],
                                 r['n_iface_dsasa']))
            for nm, _c in IFACE_DEFS:
                q = r[nm]
                print('            %-6s OR_MH %.4f [%.4f, %.4f]  crude %.4f  '
                      'p_NS1 %.4f (B=%d)  rate %.4f vs %.4f'
                      % (nm, q['OR'], q['lo'], q['hi'], q['crude'], q['p_NS1'],
                         q['NS1_B'], q['rate_iface'], q['rate_noniface']))
        msg = {
            'dual_localisation':
                'the cliffs localise to BOTH interfaces with indistinguishable '
                'odds ratios AND both clear the support cut => interface '
                'localisation is NON-CAUSAL for KRAS',
            'not_identifiable':
                'the two odds ratios are INDISTINGUISHABLE (so at most one can '
                'be causal and the data cannot say which) but NEITHER clears '
                'the OR >= %.1f support cut, so the spec\'s dual-localisation '
                'flag does not fire -- for a POWER reason, NOT because the two '
                'localisations differ.  G11 is NOT cleared: interface '
                'localisation is NOT IDENTIFIABLE as causal for KRAS'
                % THRESH['C4S_OR_sup'],
            'distinguishable':
                'the two odds ratios differ by more than the declared band, so '
                'one of them can still be causal'}
        for nm, _c in IFACE_DEFS:
            g = out['g11_' + nm]
            print('      verdict[%-6s] OR 8BE4 %.4f vs 5O2S %.4f  ratio %.4f  '
                  'both >= %.1f: %s  indistinguishable: %s  stronger: %s'
                  % (nm, g['OR_8BE4'], g['OR_5O2S'], g['ratio'],
                     THRESH['C4S_OR_sup'], g['both_above_sup'], g['similar'],
                     g['stronger']))
            print('                    => %s' % msg[g['verdict']])
        print('      G11 spec flag (dual localisation, either definition): %s'
              % out['g11_dual'])
        who = set(out['g11_' + nm]['stronger'] for nm, _c in IFACE_DEFS)
        if who == set(['5O2S']):
            print('      READ THIS: the STRONGER -- and the only nominally '
                  'significant -- localisation is against 5O2S/DARPinK27, the '
                  'complex this score table was NEVER measured against.  One '
                  'score table, two interfaces, and the NON-measured one wins, '
                  'so the localisation cannot be attributed to the measured '
                  'interface.')
    return out


# =========================================================================== #
# 11. the stage 6 driver                                                       #
# =========================================================================== #

def c4s_assays(assays=None):
    """The C4-S set: ``eligible_C4S`` and a latent fit and T09 rows.

    Seven assays, which is the spec's own ">= 4 of 7 eligible assays"
    denominator.  ORCHESTRATOR: CD19 is STRUCTURALLY_UNIDENTIFIED and
    contributes to nothing, so the aggregate count is k-of-6.  KRAS_SOS1's
    design-bias flag is WITHDRAWN (D7) and it is in the set.
    """
    ids = list(assays) if assays else list(config.ALL_ASSAYS)
    t09 = read_T09()
    have = set(t09['DMS_id'].astype(str))
    return [a for a in ids
            if ASSAYS[a].eligible_C4S and a in have and has_latent(a)]


def c4p_assays(assays=None):
    """The C4-P set: ``eligible_C4P`` and a latent fit and T09 rows."""
    ids = list(assays) if assays else list(config.ALL_ASSAYS)
    t09 = read_T09()
    have = set(t09['DMS_id'].astype(str))
    return [a for a in ids
            if ASSAYS[a].eligible_C4P and a in have and has_latent(a)]


def c4s_one(dms_id, *, t09=None, tau=TAU_PRIMARY, B=None, B_boot=None,
            verbose=True):
    """C4-S for one assay under BOTH co-primary interface definitions (D6)."""
    if t09 is None:
        t09 = read_T09()
    st = site_table(dms_id, t09, centred=True)
    if st is None:
        return None
    n_exp = int((st['n_pairs_at_site'] > 0).sum())
    rec = dict(DMS_id=dms_id, tau=tau, n_positions=int(len(st)),
               n_positions_with_exposure=n_exp,
               n_iface_5A_with_exposure=int(
                   (st['is_iface_5A'].astype(bool) & (st['n_pairs_at_site'] > 0)).sum()),
               n_iface_dsasa_with_exposure=int(
                   (st['is_iface_dsasa'].astype(bool) & (st['n_pairs_at_site'] > 0)).sum()),
               iface_def_primary=IFACE_DEFS[0][0],
               n_iface_5A=int(st['is_iface_5A'].astype(bool).sum()),
               n_iface_dsasa=int(st['is_iface_dsasa'].astype(bool).sum()),
               n_Pa=int(st.attrs.get('n_Pa', -1)),
               assay_rate_tau_primary=float(st.attrs.get('rate_tau_primary',
                                                         float('nan'))),
               c4s_stamp=STRUCTURALLY_UNIDENTIFIED.get(dms_id, ''),
               per_def={})
    for nm, col in IFACE_DEFS:
        m, use = c4s_models(st, col, tau=tau)
        o = c4s_or(use, col, tau=tau, dms_id=dms_id, B=B_boot)
        p = ns1_pvalue(use, col, tau=tau, dms_id=dms_id, B=B,
                       obs=o['OR_burial_matched'])
        q = dict(m)
        q.pop('_models', None)
        q.pop('_iface_idx', None)
        q.update(o)
        q.update(p)
        q['iface_aliased'] = bool('iface' in (m.get('glm_dropped') or []))
        rec['per_def'][nm] = q
        if verbose:
            print('  %-38s %-6s n=%3d/%3d iface=%3d  OR_MH %6.4f '
                  '[%6.4f,%6.4f] crude %6.4f  p_NS1 %.4f (B=%d, ladder L%d)  '
                  'beta_iface %+.4f (p %.3g) -> after rsa %+.4f (p %.3g)  '
                  '%s'
                  % (dms_id, nm, q['n_positions'], rec['n_positions'],
                     q['n_iface'],
                     q['OR_burial_matched'], q['OR_lo95'], q['OR_hi95'],
                     q['crude_OR'], q['p_NS1'], q['NS1_B'],
                     q['NS1_strata_level'], q['beta_iface_unadj'],
                     q['p_wald_unadj'], q['beta_iface_after_rsa'],
                     q['p_wald_after_rsa'],
                     ('[iface ALIASED by the Levy dummies in the full model]'
                      if q['iface_aliased'] else '')))
    return rec


def _c4s_worker(args):
    dms_id, tau, B, B_boot = args
    try:
        return dms_id, c4s_one(dms_id, tau=tau, B=B, B_boot=B_boot,
                               verbose=False), ''
    except Exception as exc:                                   # pragma: no cover
        return dms_id, None, '%s: %s' % (type(exc).__name__, exc)


def c4s_run(assays=None, *, tau=TAU_PRIMARY, B=None, B_boot=None, nproc=1,
            verbose=True):
    """C4-S over the eligible set, plus the kill switch and the BH-FDR over
    assays (never over the edges)."""
    t09 = read_T09()
    ids = c4s_assays(assays)
    if verbose:
        print('\n=== C4-S  burial-matched site level, %d eligible assays, '
              'BOTH interface definitions (D6 co-primary) ===' % len(ids))
        for a in sorted(C4S_OUT_BY_CONSTRUCTION):
            print('    out of C4-S by construction: %-40s %s'
                  % (a, C4S_OUT_BY_CONSTRUCTION[a]))
    recs = {}
    if nproc and nproc > 1 and len(ids) > 1:
        import multiprocessing as mp
        tasks = [(a, tau, B, B_boot) for a in ids]
        with mp.Pool(processes=min(int(nproc), len(tasks))) as pool:
            for dms_id, rec, err in pool.imap_unordered(_c4s_worker, tasks):
                if err:
                    print('  [C4-S FAILED] %s  %s' % (dms_id, err))
                elif rec is not None:
                    recs[dms_id] = rec
        if verbose:
            for a in ids:
                r = recs.get(a)
                if r is None:
                    continue
                for nm, _c in IFACE_DEFS:
                    q = r['per_def'][nm]
                    print('  %-38s %-6s n=%3d/%3d iface=%3d  OR_MH %6.4f '
                          '[%6.4f,%6.4f] crude %6.4f  p_NS1 %.4f (B=%d, '
                          'ladder L%d)  beta_iface %+.4f (p %.3g) -> after rsa '
                          '%+.4f (p %.3g)  %s'
                          % (a, nm, q['n_positions'], r['n_positions'],
                             q['n_iface'],
                             q['OR_burial_matched'], q['OR_lo95'], q['OR_hi95'],
                             q['crude_OR'], q['p_NS1'], q['NS1_B'],
                             q['NS1_strata_level'], q['beta_iface_unadj'],
                             q['p_wald_unadj'], q['beta_iface_after_rsa'],
                             q['p_wald_after_rsa'],
                             ('[iface ALIASED by the Levy dummies]'
                              if q['iface_aliased'] else '')))
    else:
        for a in ids:
            r = c4s_one(a, t09=t09, tau=tau, B=B, B_boot=B_boot, verbose=verbose)
            if r is not None:
                recs[a] = r
    ids = [a for a in ids if a in recs]
    # ---- BH-FDR over the ASSAYS, per definition --------------------------- #
    for nm, _c in IFACE_DEFS:
        pv = np.array([recs[a]['per_def'][nm]['p_NS1'] for a in ids])
        qv = bh_fdr(pv)
        for a, q in zip(ids, qv):
            recs[a]['per_def'][nm]['q_NS1_BH'] = float(q)
    # ---- the assay-level frame T09 broadcasts ----------------------------- #
    rows = []
    for a in ids:
        r = recs[a]
        p5 = r['per_def']['5A']
        pd_ = r['per_def']['dsasa']
        rows.append(dict(
            DMS_id=a,
            OR_burial_matched=p5['OR_burial_matched'],
            OR_lo95=p5['OR_lo95'], OR_hi95=p5['OR_hi95'],
            beta_iface_unadj=p5['beta_iface_unadj'],
            beta_iface_adj=p5['beta_iface_adj'],
            p_wald=p5['p_wald'], p_NS1=p5['p_NS1'],
            beta_iface_after_rsa=p5['beta_iface_after_rsa'],
            p_wald_after_rsa=p5['p_wald_after_rsa'],
            iface_def_primary='5A_min_heavy<%.1fA' % THRESH['C4_iface_dist_A'],
            crude_OR=p5['crude_OR'], OR_lo95_rbg=p5['OR_lo95_rbg'],
            OR_hi95_rbg=p5['OR_hi95_rbg'], p_wald_MH=p5['p_wald_MH'],
            NS1_strata_level=p5['NS1_strata_level'],
            NS1_frac_exchangeable=p5['NS1_frac_exchangeable'],
            NS1_B=p5['NS1_B'],
            OR_burial_matched_dsasa=pd_['OR_burial_matched'],
            OR_lo95_dsasa=pd_['OR_lo95'], OR_hi95_dsasa=pd_['OR_hi95'],
            beta_iface_unadj_dsasa=pd_['beta_iface_unadj'],
            beta_iface_after_rsa_dsasa=pd_['beta_iface_after_rsa'],
            p_wald_unadj_dsasa=pd_['p_wald_unadj'],
            p_wald_after_rsa_dsasa=pd_['p_wald_after_rsa'],
            p_NS1_dsasa=pd_['p_NS1'],
            glm_iface_aliased_dsasa=pd_['iface_aliased'],
            c4s_stamp=r['c4s_stamp']))
    per_assay = pd.DataFrame(rows)
    if verbose and len(per_assay):
        print('\n  --- C4-S decision arithmetic (spec: SUPPORTED iff OR >= %.1f '
              'with p_NS1 < %.2f in >= 4 of 7; REFUTED iff the CI covers 1 in '
              '>= 5 of 7 or OR < %.1f in >= 3) ---'
              % (THRESH['C4S_OR_sup'], THRESH['C4S_p_NS1_sup'],
                 THRESH['C4S_OR_ref']))
        for a in ids:
            for nm, _c in IFACE_DEFS:
                q = recs[a]['per_def'][nm]
                if q.get('note'):
                    print('    %-38s %-6s %s (%d of %d positions carry any '
                          'P_a exposure)'
                          % (a, nm, q['note'], recs[a]['n_positions_with_exposure'],
                             recs[a]['n_positions']))
        for nm, _c in IFACE_DEFS:
            scored = [a for a in ids if a not in STRUCTURALLY_UNIDENTIFIED]
            def _q(a):
                return recs[a]['per_def'][nm]
            n_sup = sum(1 for a in scored
                        if _q(a)['OR_burial_matched'] >= THRESH['C4S_OR_sup']
                        and _q(a)['p_NS1'] < THRESH['C4S_p_NS1_sup'])
            n_cov = sum(1 for a in scored
                        if np.isfinite(_q(a)['OR_lo95'])
                        and np.isfinite(_q(a)['OR_hi95'])
                        and _q(a)['OR_lo95'] <= 1.0 <= _q(a)['OR_hi95'])
            n_lt1 = sum(1 for a in scored
                        if _q(a)['OR_burial_matched'] < THRESH['C4S_OR_ref'])
            n_kill = sum(1 for a in scored
                         if not (np.isfinite(_q(a)['p_wald_after_rsa'])
                                 and _q(a)['p_wald_after_rsa']
                                 < THRESH['C4S_p_NS1_sup']))
            print('  %-6s  k_support = %d/%d   k_CI_covers_1 = %d/%d   '
                  'k_OR<1 = %d/%d   k_kill_switch_fires = %d/%d'
                  % (nm, n_sup, len(scored), n_cov, len(scored),
                     n_lt1, len(scored), n_kill, len(scored)))
    return dict(records=recs, per_assay=per_assay, assays=ids)


def c4p_run(assays=None, *, t09=None, B=None, B_boot=None, verbose=True):
    """C4-P == route L5, pair level, null NS2.  Writes T10."""
    if t09 is None:
        t09 = read_T09()
    ids = c4p_assays(assays)
    if verbose:
        print('\n=== C4-P  pair level (route L5): AUROC of -d3d_min_heavy '
              'separating |eps| >= %g sigma from < %g sigma, null NS2, '
              '%d assays ==='
              % (THRESH['L5_cliff_sigma_mult'], THRESH['L5_noncliff_sigma_mult'],
                 len(ids)))
    frames, recs = [], {}
    for a in ids:
        d, meta = epsilon_structure(a, t09, verbose=verbose)
        if d is None:
            if verbose:
                print('  %-38s no usable eps: %s' % (a, meta.get('reason')))
            recs[a] = dict(DMS_id=a, available=False, **meta)
            continue
        o, use, place = l5_auroc(d, alt=False, dms_id=a, B_perm=B, B_boot=B_boot)
        oa, _u2, _p2 = l5_auroc(d, alt=True, dms_id=a, B_perm=B, B_boot=B_boot)
        if not np.isfinite(o.get('AUROC_L5', np.nan)):
            if verbose:
                print('  %-38s C4-P NOT COMPUTABLE -- %s'
                      % (a, o.get('note', 'no two-class contrast')))
            rec = dict(DMS_id=a, available=False)
            rec.update(meta)
            rec.update(o)
            rec['reason'] = o.get('note', 'no two-class contrast')
            rec['c4s_stamp'] = STRUCTURALLY_UNIDENTIFIED.get(a, '')
            recs[a] = rec
            d = d.copy()
            d['AUROC_contribution'] = np.nan
            d['p_NS2'] = np.nan
            frames.append(d)
            continue
        d = d.copy()
        d['AUROC_contribution'] = np.nan
        if place is not None:
            d.loc[use.index, 'AUROC_contribution'] = place
        d['p_NS2'] = o.get('p_NS2', np.nan)
        frames.append(d)
        rec = dict(DMS_id=a, available=True)
        rec.update(meta)
        rec.update(o)
        rec.update({('alt_' + k): v for k, v in oa.items()})
        rec['c4s_stamp'] = STRUCTURALLY_UNIDENTIFIED.get(a, '')
        recs[a] = rec
        if verbose:
            print('  %-38s AUROC %.4f [%.4f, %.4f]  p_NS2 %.4f (B=%d)  '
                  'null %.4f+-%.4f  median d3d cliff %.2f vs noncliff %.2f  '
                  'ICC(site pair) %.4f  | D5 alt sigma %.4f: AUROC %.4f '
                  'p_NS2 %.4f  %s'
                  % (a, o['AUROC_L5'], o['AUROC_lo95'], o['AUROC_hi95'],
                     o['p_NS2'], o['NS2_B'], o['NS2_null_mean'],
                     o['NS2_null_sd'], o['AUROC_d3d_median_cliff'],
                     o['AUROC_d3d_median_noncliff'], meta['ICC_sitepair'],
                     meta['sigma_eps_alt'], oa['AUROC_L5'],
                     oa.get('p_NS2', float('nan')), rec['c4s_stamp']))
    ids_ok = [a for a in ids if recs.get(a, {}).get('available')]
    pv = np.array([recs[a]['p_NS2'] for a in ids_ok])
    for a, q in zip(ids_ok, bh_fdr(pv)):
        recs[a]['q_NS2_BH'] = float(q)
    return dict(records=recs, frames=frames, assays=ids_ok)


def c4i_run(families=None, *, t09=None, B=None, g11=None, verbose=True):
    """C4-I over the five families, plus the licence decision.  Writes T11."""
    if t09 is None:
        t09 = read_T09()
    fams = list(families) if families else sorted(C4I_FAMILIES)
    if verbose:
        print('\n=== C4-I  partner specificity, double-centred (the actual '
              'interaction test), %d families ===' % len(fams))
    out, rows, overlap = {}, [], {}
    if verbose:
        print('  --- family joins: variant-key overlap on the WT-verified '
              'offsets vs the BANNED naive no-offset join (spec G1b) ---')
    for fam in fams:
        try:
            overlap[fam] = family_key_overlap(fam, verbose=verbose)
        except Exception as exc:
            print('  [join audit FAILED] %-8s %s: %s'
                  % (fam, type(exc).__name__, exc))
        try:
            r = c4i_family(fam, t09=t09, B=B, verbose=verbose)
        except Exception as exc:
            print('  [C4-I FAILED] %-8s %s: %s' % (fam, type(exc).__name__, exc))
            out[fam] = dict(family=fam, failed='%s: %s' % (type(exc).__name__, exc))
            continue
        out[fam] = r
        if verbose:
            print('  %-7s J=%d/%d K=%d channel=%-8s M_F %s  p_NS3 %s  '
                  'F_spec(%s) %s -> noise-corrected %s  '
                  'PSI: n_cliff %d median %s  MW_p %s (literal %s, '
                  'p_above %s)  fold axis Spearman(rowmean Z, rsa_iso) %s'
                  % (fam, r['J_used'], r['J_aligned'], r['K'], r['channel'],
                     _f4(r['M_F']), _f4(r['family_p_NS3']), r['F_spec_scale'],
                     _f4(r['F_spec']), _f4(r['F_spec_noise_corrected']),
                     r['n_cliff_pos'], _f4(r['median_cliff_PSI']),
                     _f4(r['MW_PSI_p']), _f4(r['MW_PSI_p_literal']),
                     _f4(r['p_PSI_above']),
                     _f4(r['foldaxis_spearman_rowmean_rsa'])))
            if r.get('F_spec_at_boundary'):
                print('           F_spec is AT THE BOUNDARY: the '
                      'method-of-moments Var(mu) came out NEGATIVE '
                      '(MSB < MSW), so F_spec = 1.0000 means "no '
                      'partner-invariant component is estimable", NOT '
                      '"completely partner-specific".  It is not evidence for '
                      'an interaction cliff.')
            print('           offsets %r  unique %r  note: %s'
                  % (r['offsets'], r['offsets_unique'], r['note']))
            if r['F_spec_note']:
                print('           F_spec note: %s' % r['F_spec_note'])
        rows.extend(_t11_rows(r, t09, g11=g11))
    return dict(records=out, rows=rows, families=fams, overlap=overlap)


def _f4(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 'n/a'
    return 'n/a' if not np.isfinite(f) else '%.4f' % f


def _t11_rows(r, t09, g11=None):
    """One T11 row per (family, position) on the family's WT-verified frame."""
    fm = r['_fm']
    psi = r['_psi']
    Z, W, n, k = fm['Z'], fm['W'], fm['n'], fm['k']
    ok = (n > 0).all(axis=1) & np.isfinite(Z).all(axis=1)
    if fm['structure']:
        ok = ok & np.isfinite(W).all(axis=1)
    Zt = np.full(Z.shape, np.nan)
    if ok.sum() >= 2:
        Zt[ok] = double_centre(Z[ok])
    ref = fm['ref']
    st = t09[t09['DMS_id'].astype(str) == ref]
    srow = {(str(q['chain']), int(q['seq_idx'])): q for q in st.to_dict('records')}
    wt = wt_map(ref)
    g5 = (g11 or {}).get('g11_5A', {})
    rows = []
    for j, kk in enumerate(fm['keys']):
        s = srow.get(kk)
        row = dict(family=fm['family'], chain=kk[0],
                   resseq=(int(s['resseq']) if s is not None else kk[1]),
                   icode=(s['icode'] if s is not None else ''),
                   wt_aa=wt.get(kk, ''),
                   K_partners=int(psi['K_partners'][j]),
                   PSI=psi['PSI'][j],
                   rowmean_Z=float(np.nanmean(Z[j])),
                   rsa_iso=float(np.nanmean(fm['rsa'][j])),
                   family_M_stat=r['M_F'], family_p_NS3=r['family_p_NS3'],
                   foldaxis_spearman_rowmean_rsa=r['foldaxis_spearman_rowmean_rsa'],
                   F_spec=r['F_spec'],
                   F_spec_noise_corrected=r['F_spec_noise_corrected'],
                   F_spec_at_boundary=bool(r.get('F_spec_at_boundary', False)),
                   MW_PSI_p=r['MW_PSI_p'],
                   twin_structure_OR_8BE4=(g5.get('OR_8BE4', np.nan)
                                           if fm['family'] == 'KRAS' else np.nan),
                   twin_structure_OR_5O2S=(g5.get('OR_5O2S', np.nan)
                                           if fm['family'] == 'KRAS' else np.nan),
                   twin_structure_note=('G11: one score table, two complexes; '
                                        'dual localisation = %s'
                                        % g5.get('dual_localisation'))
                   if fm['family'] == 'KRAS' else '',
                   classification='',
                   partners='|'.join(fm['partners']), channel=fm['channel'],
                   J_aligned=fm['J'], J_used=int(ok.sum()),
                   median_cliff_PSI=r['median_cliff_PSI'],
                   mean_cliff_PSI=r['mean_cliff_PSI'],
                   PSI_any1=psi['PSI_any1'][j],
                   p_PSI_below=r['p_PSI_below'], p_PSI_above=r['p_PSI_above'],
                   MW_PSI_p_literal=r['MW_PSI_p_literal'],
                   F_spec_scale=r['F_spec_scale'],
                   F_spec_n_shared=r['F_spec_n_shared'],
                   F_spec_Z=r['F_spec_Z'],
                   F_spec_Z_noise_corrected=r['F_spec_Z_noise_corrected'],
                   p_NS3_two_sided=r['p_NS3_two_sided'], NS3_B=r['NS3_B'],
                   NS3_null_mean=r['NS3_null_mean'], NS3_null_sd=r['NS3_null_sd'],
                   n_cliff_pos=r['n_cliff_pos'],
                   n_noncliff_pos=r['n_noncliff_pos'],
                   structurally_mute=r['structurally_mute'], note=r['note'])
        for a in range(4):
            i = a + 1
            if a < fm['K']:
                row['cliff_rate_p%d' % i] = (k[j, a] / n[j, a]) if n[j, a] > 0 \
                    else np.nan
                row['min_heavy_dist_p%d' % i] = (-W[j, a]
                                                 if np.isfinite(W[j, a]) else np.nan)
                row['iface_flag_p%d' % i] = bool(fm['iface'][j, a])
                row['Z_doublecentered_p%d' % i] = Zt[j, a]
                row['cliff_flag_p%d' % i] = bool(psi['flag'][j, a])
                row['n_pairs_p%d' % i] = int(n[j, a])
            else:
                for c in ('cliff_rate_p%d', 'min_heavy_dist_p%d',
                          'iface_flag_p%d', 'Z_doublecentered_p%d',
                          'cliff_flag_p%d', 'n_pairs_p%d'):
                    row[c % i] = np.nan
        rows.append(row)
    return rows


def family_key_overlap(family, *, verbose=True):
    """The VARIANT-level join the spec's family counts quote, verified.

    The spec's "1,577/1,577 shared" (PSD95), "518/518" (BH3), "534 shared"
    (5A12) and "65,093 shared" (CR9114) are counts of shared VARIANT keys, not
    of shared substitution pairs, so they are checked here on their own scale.
    The BANNED naive join -- same numbering, no offset -- is computed beside the
    WT-verified one, because that is the whole reason the offsets exist: on BH3
    the naive join keeps 97 of 518.
    """
    partners = list(C4I_FAMILIES[family]['partners'])
    offs, ref = resolve_offsets(partners)
    single = all(offs[d]['single_chain'] for d in partners)

    def _keys(d, use_offset):
        info = offs[d]
        out = set()
        for k in io_bgym.load_assay(d).keys:
            ak = []
            for (c, ps, aa) in k:
                off = int(info['offset'].get(c, 0)) if use_offset else 0
                cr = info['chain_map'].get(c, c) if not single else None
                ak.append((int(ps) + off, aa) if single
                          else (cr, int(ps) + off, aa))
            out.add(tuple(sorted(ak)))
        return out

    res = dict(family=family, ref=ref, partners=partners,
               offsets={d: offs[d]['offset'] for d in partners},
               n_keys={d: len(io_bgym.load_assay(d).keys) for d in partners})
    for tag, use in (('resolved', True), ('naive_no_offset', False)):
        sets = [_keys(d, use) for d in partners]
        inter = set.intersection(*sets)
        res['n_shared_' + tag] = len(inter)
        res['frac_shared_' + tag] = (len(inter) / min(len(x) for x in sets)
                                     if sets else float('nan'))
    if verbose:
        print('  %-7s variant-key join on the WT-verified offsets %r: '
              '%d shared of %r (%.4f of the smaller table).  The BANNED naive '
              'no-offset join would keep %d (%.4f) -- %s'
              % (family, res['offsets'], res['n_shared_resolved'],
                 res['n_keys'], res['frac_shared_resolved'],
                 res['n_shared_naive_no_offset'],
                 res['frac_shared_naive_no_offset'],
                 'the two agree, so the offset is the identity here'
                 if res['n_shared_resolved'] == res['n_shared_naive_no_offset']
                 else 'a %.1fx difference, which is why the offsets are not '
                      'optional bookkeeping'
                      % (res['n_shared_resolved']
                         / max(res['n_shared_naive_no_offset'], 1))))
    return res


def c4i_licence(rec, g11=None, *, family='KRAS'):
    """The spec's own licence clause, evaluated and NAMED.

    "Interaction cliff" is LICENSED iff ``F_spec >= 0.40`` noise-corrected AND
    ``p_NS3 < 0.05`` in KRAS AND cliff-position PSI stochastically below
    non-cliff PSI (one-sided Mann-Whitney ``p < 0.05``).  If it is refuted the
    correct name is "stability cliff", and this function says so.
    """
    r = rec.get(family, {})
    f = r.get('F_spec_noise_corrected', float('nan'))
    p = r.get('family_p_NS3', float('nan'))
    mw = r.get('MW_PSI_p', float('nan'))
    lit = r.get('MW_PSI_p_literal', float('nan'))
    fold = r.get('foldaxis_spearman_rowmean_rsa', float('nan'))
    cl = dict(
        F_spec_ge_sup=bool(np.isfinite(f) and f >= THRESH['C4I_Fspec_sup']),
        p_NS3_lt_sup=bool(np.isfinite(p) and p < THRESH['C4I_p_NS3_sup']),
        MW_PSI_lt_sup=bool(np.isfinite(mw) and mw < THRESH['C4I_MW_p_sup']),
        MW_PSI_literal_lt_sup=bool(np.isfinite(lit)
                                   and lit < THRESH['C4I_MW_p_sup']),
        F_spec_le_ref=bool(np.isfinite(f) and f <= THRESH['C4I_Fspec_ref']),
        median_PSI_ge_ref=bool(np.isfinite(r.get('median_cliff_PSI', np.nan))
                               and r['median_cliff_PSI']
                               >= THRESH['C4I_median_PSI_ref']),
        G11_dual=bool((g11 or {}).get('g11_dual', False)))
    licensed = (cl['F_spec_ge_sup'] and cl['p_NS3_lt_sup'] and cl['MW_PSI_lt_sup'])
    refuted = (cl['F_spec_le_ref'] or cl['median_PSI_ge_ref'] or cl['G11_dual'])
    name = ('interaction_cliff' if (licensed and not refuted)
            else ('stability_cliff' if refuted else 'undetermined'))
    fold_ok = bool(np.isfinite(fold) and fold > 0)
    if fold_ok:
        fnote = ('Spearman(row mean of Z, rsa_iso) = %s > 0 as the spec '
                 'pre-registers, so the partner-invariant component '
                 'row-centering removes tracks an rsa axis.' % _f4(fold))
    elif np.isfinite(fold):
        fnote = ('Spearman(row mean of Z, rsa_iso) = %s is NOT > 0, so by the '
                 'spec\'s pre-registered criterion the fold interpretation of '
                 'the partner-invariant component is UNSUPPORTED and is '
                 'reported as such rather than assumed.  REPORTED LOUDLY: the '
                 'sign is NEGATIVE, i.e. partner-invariant cliff propensity is '
                 'concentrated at the MORE BURIED positions (low rsa_iso) -- '
                 'which is the direction burial-driven fold destabilisation '
                 'predicts, not the opposite.  So the criterion as written '
                 'fails while the measurement points the way the fold reading '
                 'would want; the spec\'s "> 0" appears to have the sign '
                 'backwards.  Either way the clause is scored as written.'
                 % _f4(fold))
    else:
        fnote = ('Spearman(row mean of Z, rsa_iso) is not computable (no '
                 'structural annotation for this family), so the fold axis is '
                 'UNSUPPORTED by absence of data.')
    return dict(family=family, clauses=cl, licensed=bool(licensed),
                refuted=bool(refuted), classification=name,
                F_spec_noise_corrected=f, p_NS3=p, MW_PSI_p=mw,
                MW_PSI_p_literal=lit, foldaxis_spearman=fold,
                F_spec_at_boundary=bool(r.get('F_spec_at_boundary', False)),
                F_spec_scale=r.get('F_spec_scale', ''),
                median_cliff_PSI=r.get('median_cliff_PSI', float('nan')),
                p_PSI_above=r.get('p_PSI_above', float('nan')),
                M_F=r.get('M_F', float('nan')),
                foldaxis_supported=fold_ok,
                foldaxis_status=('SUPPORTED' if fold_ok else 'UNSUPPORTED'),
                foldaxis_note=fnote)


def negative_control_5A12(t09=None, *, c4i=None, verbose=True):
    """The designed C4 NEGATIVE control, cited in the dSASA form (D6).

    5A12_VEGF's nine mutated positions have ``dSASA`` exactly 0.000 against
    VEGF -- an EXACT zero, which is why the dSASA form is the one cited.  The
    6.4 A form passes by 0.0037 A on the closest position and flips at any
    6.5 A cut, so it is reported but never leaned on.
    """
    if t09 is None:
        t09 = read_T09()
    out = {}
    for a in ('5A12_VEGF_fitness_4ZFF', '5A12_Ang2_fitness_4ZFG'):
        st = t09[t09['DMS_id'].astype(str) == a]
        d = st['dsasa'].values.astype(float)
        mh = st['min_heavy_dist'].values.astype(float)
        out[a] = dict(
            n_positions=int(len(st)),
            n_dsasa_gt0=int((d > 0).sum()),
            max_dsasa=float(np.nanmax(d)) if len(st) else float('nan'),
            all_dsasa_exactly_zero=bool(len(st) and np.all(d == 0.0)),
            n_iface_5A=int(st['is_iface_5A'].astype(bool).sum()),
            min_min_heavy=float(np.nanmin(mh)) if len(st) else float('nan'),
            n_within_6p4=int((mh < 6.4).sum()),
            n_within_6p5=int((mh < 6.5).sum()),
            margin_at_6p4=float(np.nanmin(mh) - 6.4) if len(st) else float('nan'))
    if verbose:
        print('\n=== 5A12_VEGF, the DESIGNED C4 NEGATIVE control ===')
        for a, q in out.items():
            print('  %-26s %d positions  dSASA > 0: %d  max dSASA %.4f  '
                  'all exactly 0.000: %s  |  5A iface: %d  min min-heavy '
                  '%.4f A  within 6.4 A: %d  within 6.5 A: %d'
                  % (a, q['n_positions'], q['n_dsasa_gt0'], q['max_dsasa'],
                     q['all_dsasa_exactly_zero'], q['n_iface_5A'],
                     q['min_min_heavy'], q['n_within_6p4'], q['n_within_6p5']))
        v = out['5A12_VEGF_fitness_4ZFF']
        print('  CITED IN THE dSASA FORM: all %d VEGF positions have dSASA '
              'EXACTLY 0.000.  The 6.4 A form clears the cut by only %.4f A '
              'and would FLIP at a 6.5 A cut (%d of %d positions inside it), '
              'which is why it is reported and not leaned on.'
              % (v['n_positions'], v['margin_at_6p4'], v['n_within_6p5'],
                 v['n_positions']))
        if c4i is not None and '5A12' in c4i:
            r = c4i['5A12']
            print('  C4-I on the 5A12 family (VEGF vs Ang2, the SAME nine '
                  'positions against two partners): M_F %s  p_NS3 %s  '
                  'F_spec(%s) noise-corrected %s -- the control\'s own '
                  'partner-specificity reading'
                  % (_f4(r.get('M_F')), _f4(r.get('family_p_NS3')),
                     r.get('F_spec_scale'), _f4(r.get('F_spec_noise_corrected'))))
    return out


def interface_definition_audit(t09=None, *, verbose=True):
    """D6's own arithmetic, RE-MEASURED on the population C4-S actually uses.

    The spec quotes recall 0.9219 (82 missed of 1,050) and a 6.07 A maximum;
    those are over EVERY residue of the mutated chains.  C4-S runs on the 2,220
    MUTATED positions, and the numbers there are different -- so both are
    printed and the difference is stated rather than inherited.
    """
    if t09 is None:
        t09 = read_T09()
    d = t09['dsasa'].values.astype(float)
    mh = t09['min_heavy_dist'].values.astype(float)
    f5 = t09['is_iface_5A'].values.astype(bool)
    fd = t09['is_iface_dsasa'].values.astype(bool)
    levy = t09['levy_class'].astype(str).values
    burial = np.isin(levy, ('support', 'rim', 'core'))
    out = dict(
        n_positions=int(len(t09)),
        levy_interior_frac=float((levy == 'interior').mean()),
        n_levy_interior=int((levy == 'interior').sum()),
        n_dsasa_gt0=int((d > 0).sum()),
        n_dsasa_gt1=int((d > THRESH['C4_dsasa_min_A2']).sum()),
        n_dsasa_in_0_1=int(((d > 0) & (d <= THRESH['C4_dsasa_min_A2'])).sum()),
        n_levy_burial=int(burial.sum()), n_iface_dsasa=int(fd.sum()),
        n_iface_5A=int(f5.sum()),
        dsasa_cut_is_nonbinding=bool(((d > 0) & (d <= THRESH['C4_dsasa_min_A2'])).sum() == 0),
        dsasa_equals_levy_burial=bool(np.array_equal(d > 0, burial)),
        recall_5A_vs_burial=float((f5 & burial).sum() / max(burial.sum(), 1)),
        n_burial_missed_by_5A=int((~f5 & burial).sum()),
        n_5A_outside_burial=int((f5 & ~burial).sum()),
        max_min_heavy_over_dsasa_gt0=float(np.nanmax(mh[d > 0])) if (d > 0).any()
        else float('nan'),
        spec_recall=0.9219, spec_n_burial=1050, spec_n_missed=82,
        spec_max_min_heavy=THRESH['C4_max_min_heavy_dsasa_pos_A'])
    if verbose:
        print('\n=== D6 interface-definition audit, RE-MEASURED on the %d '
              'MUTATED positions C4-S runs on ===' % out['n_positions'])
        print('  Levy interior: %d/%d = %.4f  <-- buried in the MONOMER fold, '
              'the classic source of large DMS jumps; an interface test that '
              'lumps interior with surface answers the wrong question'
              % (out['n_levy_interior'], out['n_positions'],
                 out['levy_interior_frac']))
        print('  dSASA > 0: %d    dSASA > %g A^2: %d    in (0, %g]: %d'
              % (out['n_dsasa_gt0'], THRESH['C4_dsasa_min_A2'],
                 out['n_dsasa_gt1'], THRESH['C4_dsasa_min_A2'],
                 out['n_dsasa_in_0_1']))
        print('  => the %g A^2 cut is NON-BINDING (no residue anywhere falls in '
              'the interval), so varying it is a FLAT LINE and is never '
              'presented as robustness: %s'
              % (THRESH['C4_dsasa_min_A2'], out['dsasa_cut_is_nonbinding']))
        print('  dSASA > 0 == Levy support+rim+core: %s (%d positions each)'
              % (out['dsasa_equals_levy_burial'], out['n_levy_burial']))
        print('  5.0 A min-heavy flag: %d positions; recall against the burial '
              'definition %.4f (misses %d of %d), flags %d outside it'
              % (out['n_iface_5A'], out['recall_5A_vs_burial'],
                 out['n_burial_missed_by_5A'], out['n_levy_burial'],
                 out['n_5A_outside_burial']))
        print('  max min-heavy over dSASA > 0 positions: %.4f A (the spec\'s '
              '%.2f A is over every residue of the mutated chains, not the '
              'mutated positions)'
              % (out['max_min_heavy_over_dsasa_gt0'], out['spec_max_min_heavy']))
        print('  DISCREPANCY WITH THE BRIEF, stated rather than inherited: the '
              'brief\'s recall 0.9219 (82 missed of 1,050) is over the full '
              'residue population; on the %d mutated positions that actually '
              'enter C4-S it is %.4f (%d missed of %d).'
              % (out['n_positions'], out['recall_5A_vs_burial'],
                 out['n_burial_missed_by_5A'], out['n_levy_burial']))
    return out


def design_bias_audit(assays=None, *, verbose=True):
    """ORCHESTRATOR **D7**, re-derived: KRAS_SOS1's design-bias flag is WITHDRAWN.

    The spec's 0.264 / 0.110 = 2.4x divided a dSASA-based design fraction by
    chain S's own 5 A interface fraction (49/440 = 0.1114) -- the PARTNER side,
    not the KRAS background.  The correct background denominator is "every
    residue of the MUTATED chain(s) present in the PDB", which is exactly what
    :func:`cliff.structure.interface_bias` computes; it is called here rather
    than reimplemented, so there is one definition.
    """
    ids = list(assays) if assays else c4s_assays()
    ids = sorted(set(ids) | {'KRAS_SOS1_norfitness_8BE4',
                             'KRAS_DARPinK27_norfitness_5O2S',
                             'GB1_IgG-Fc_fitness_1FCC'})
    rows = []
    for a in ids:
        try:
            annot, _e = _structure.cache_structure(a)
            assay = io_bgym.load_assay(a)
            mut = _structure.map_mutations(assay, annot)
            offs = _structure.chain_offsets(assay, annot, mut)
            rows.append(_structure.interface_bias(assay, annot, mut, offs))
        except Exception as exc:                               # pragma: no cover
            rows.append(dict(DMS_id=a, note='%s: %s' % (type(exc).__name__, exc)))
    d = pd.DataFrame(rows)
    if verbose:
        print('\n=== D7 design-bias audit (background = every residue of the '
              'MUTATED chains present in the PDB) ===')
        cols = [c for c in ('DMS_id', 'mutated_chains', 'n_design', 'n_bg',
                            'design_iface_frac', 'bg_iface_frac',
                            'iface_bias_factor', 'design_iface_frac_dsasa',
                            'bg_iface_frac_dsasa', 'design_iface_frac_spec',
                            'bg_iface_frac_spec', 'eligible_C4S')
                if c in d.columns]
        with pd.option_context('display.width', 200,
                               'display.max_columns', 40):
            print(d[cols].to_string(index=False))
        k = d[d['DMS_id'] == 'KRAS_SOS1_norfitness_8BE4']
        if len(k):
            r = k.iloc[0]
            print('  D7: KRAS_SOS1 on the MUTATED chain(s) %r -- design %.4f / '
                  'background %.4f = %.4fx, NOT the spec\'s 0.264/0.110 = 2.4x '
                  '(which used the partner side as the denominator).  The flag '
                  'is WITHDRAWN and the assay is eligible: eligible_C4S = %s'
                  % (r['mutated_chains'], r['design_iface_frac'],
                     r['bg_iface_frac'], r['iface_bias_factor'],
                     r['eligible_C4S']))
    return d


def site_counts_all(t09=None, *, verbose=True):
    """Per-position ``P_a`` exposure and cliff counts for EVERY assay that has
    both T09 rows and a latent fit -- T09's own per-site columns, whether or not
    the assay is C4-S eligible."""
    if t09 is None:
        t09 = read_T09()
    ids = [a for a in sorted(set(t09['DMS_id'].astype(str))) if has_latent(a)]
    frames, skipped, d2 = [], {}, []
    for a in ids:
        try:
            cc = cliff_counts(a, centred=True)
            d2.append(_d2_audit_row(a, cc))
        except Exception as exc:
            skipped[a] = '%s: %s' % (type(exc).__name__, exc)
            continue
        cc = cc.copy()
        cc['cliff_rate_tau%g' % TAU_SECONDARY] = np.where(
            cc['n_pairs_at_site'] > 0,
            cc['n_cliff_tau%g' % TAU_SECONDARY]
            / np.maximum(cc['n_pairs_at_site'], 1), np.nan)
        cc = cc.rename(columns={
            'n_cliff_tau%g' % TAU_SECONDARY: 'n_cliff_tau4',
            'cliff_rate_tau%g' % TAU_SECONDARY: 'cliff_rate_tau4'})
        frames.append(cc[['DMS_id', 'chain', 'seq_idx', 'n_pairs_at_site',
                          'n_cliff_pairs', 'cliff_rate', 'beta_hat_abs',
                          'n_cliff_tau4', 'cliff_rate_tau4']])
    d = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=['DMS_id', 'chain', 'seq_idx'])
    d2 = pd.DataFrame(d2)
    if verbose:
        print('\n=== T09 per-site cliff counts: %d assays, %d positions '
              '(c_hat PHI-CENTRED, ORCHESTRATOR D2) ==='
              % (d['DMS_id'].nunique() if len(d) else 0, len(d)))
        for a, why in sorted(skipped.items()):
            print('    skipped %-40s %s' % (a, why))
        print('  --- D2 audit: what phi-centring actually changes ---')
        with pd.option_context('display.width', 200,
                               'display.max_columns', 40):
            print(d2.to_string(index=False))
        if len(d2):
            print('  max |corr(e_oof, phi_oof)| = %.4f (%s); the uncentred '
                  'c_hat would change the tau=%.1f cliff set by %d..%d pairs '
                  'per assay (%.2f%%..%.2f%% of P_a).  Every count in T09 is '
                  'the CENTRED one.'
                  % (d2['corr_e_phi'].abs().max(),
                     d2.loc[d2['corr_e_phi'].abs().idxmax(), 'DMS_id'],
                     TAU_PRIMARY, int(d2['n_cliff_delta'].min()),
                     int(d2['n_cliff_delta'].max()),
                     100 * d2['frac_Pa_delta'].min(),
                     100 * d2['frac_Pa_delta'].max()))
    return d, skipped


def _d2_audit_row(dms_id, cc):
    """ORCHESTRATOR **D2**, audited per assay: ``corr(e_oof, phi_oof)`` and how
    many tau-cliffs the UNCENTRED ``c_hat`` would add or remove.

    The uncentred form manufactures cliffs wherever ``e`` has a phi-dependent
    location, and the brief quotes ``corr(e_oof, phi_oof)`` reaching +0.362 on
    some assay; which assay, and how big the consequence is, is a measurement,
    so it is measured."""
    ctx = _nulls.get_context(dms_id)
    pa = _nulls._pa_mask(ctx, ctx.censor_mask, ctx.oof_finite)
    c_on = _nulls.c_hat(ctx.e_oof, ctx.sigma_oof, ctx.nested_idx, mu=ctx.mu_oof)
    c_off = _nulls.c_hat(ctx.e_oof, ctx.sigma_oof, ctx.nested_idx, mu=None)
    on = np.abs(c_on[pa]) >= TAU_PRIMARY
    off = np.abs(c_off[pa]) >= TAU_PRIMARY
    fin = np.isfinite(ctx.e_oof) & np.isfinite(ctx.phi_oof)
    corr = (float(np.corrcoef(ctx.e_oof[fin], ctx.phi_oof[fin])[0, 1])
            if fin.sum() > 2 else float('nan'))
    n_pa = int(pa.sum())
    return dict(DMS_id=dms_id, n_Pa=n_pa, corr_e_phi=corr,
                n_cliff_centred=int(on.sum()), n_cliff_uncentred=int(off.sum()),
                n_cliff_delta=int((on != off).sum()),
                frac_Pa_delta=(float((on != off).sum()) / n_pa) if n_pa else
                float('nan'))


def tail_vs_localisation_note(c4p, *, verbose=True):
    """The measurement the brief says to expect reflected, MEASURED here.

    The brief reports that the observed tail is LIGHTER than the smooth
    surrogate's on 11 of 17 assays while localisation statistics exceed every
    null on 14 of 17 -- i.e. C4 may be annotating deviations that are real and
    localised but not extreme-tailed.  C4's own localisation column is
    ``p_NS2``, so what is checkable HERE is how many C4-P assays exceed EVERY
    NS2 draw, and whether the AUROC excess over the null mean is large.
    """
    recs = c4p['records']
    ids = c4p['assays']
    n_at_floor = 0
    rows = []
    for a in ids:
        r = recs[a]
        B = int(r.get('NS2_B', 0) or 0)
        floor = 1.0 / (B + 1.0) if B else float('nan')
        at = bool(np.isfinite(r.get('p_NS2', np.nan))
                  and np.isfinite(floor)
                  and abs(r['p_NS2'] - floor) < 1e-12)
        n_at_floor += int(at)
        rows.append(dict(DMS_id=a, AUROC=r['AUROC_L5'],
                         null_mean=r['NS2_null_mean'],
                         null_sd=r['NS2_null_sd'],
                         excess_sd=((r['AUROC_L5'] - r['NS2_null_mean'])
                                    / r['NS2_null_sd'])
                         if r['NS2_null_sd'] else float('nan'),
                         p_NS2=r['p_NS2'], q_NS2_BH=r.get('q_NS2_BH', np.nan),
                         exceeds_every_null=at,
                         median_d3d_cliff=r['AUROC_d3d_median_cliff'],
                         median_d3d_noncliff=r['AUROC_d3d_median_noncliff']))
    d = pd.DataFrame(rows)
    if verbose:
        print('\n=== C4-P localisation vs its own null (the brief\'s '
              '"localised but not extreme-tailed" expectation) ===')
        with pd.option_context('display.width', 200, 'display.max_columns', 40):
            print(d.to_string(index=False))
        print('  %d of %d C4-P assays exceed EVERY NS2 draw (p at the '
              '1/(B+1) floor).  Median 3D distance is %s for cliff eps than '
              'for non-cliff eps in %d of %d assays -- localisation is real '
              'and its direction is the predicted one.'
              % (n_at_floor, len(d),
                 'SHORTER' if (d['median_d3d_cliff']
                               < d['median_d3d_noncliff']).sum() > len(d) / 2
                 else 'LONGER',
                 int((d['median_d3d_cliff'] < d['median_d3d_noncliff']).sum()),
                 len(d)))
    return d


def stage6(assays=None, nproc=1, verbose=True, *, tau=TAU_PRIMARY,
           B_NS1_=None, B_NS2_=None, B_NS3_=None, B_boot=None,
           write=True):
    """Stage 6 (spec Sec.5): C4-S both ways, C4-P, C4-I, G11, and the tables.

    INTERPRETATION ONLY -- nothing here gates C1-C3.  ``cliff/run_all.py``
    resolves this name first (``('stage6', 'run_all', 'run')``) and passes
    ``assays`` / ``nproc`` / ``verbose``.
    """
    t_start = time.time()
    config.assert_env()
    PATHS.ensure_cache_dirs()
    B1 = B_NS1 if B_NS1_ is None else int(B_NS1_)
    B2 = B_NS2 if B_NS2_ is None else int(B_NS2_)
    B3 = B_NS3 if B_NS3_ is None else int(B_NS3_)
    Bb = B_BOOT if B_boot is None else int(B_boot)
    t09 = read_T09(refresh=True)
    if verbose:
        print('=' * 100)
        print('STAGE 6 -- C4: is "interaction cliff" the right name?  '
              '(spec Sec.1.5; INTERPRETATION ONLY, gates nothing)')
        print('  tau_primary %.1f (secondary %.1f) | B: NS1 %d, NS2 %d, NS3 %d, '
              'block bootstrap %d | Monte-Carlo SE of an empirical p at p: '
              'sqrt(p(1-p)/B) <= %.5f (NS1), %.5f (NS2), %.5f (NS3)'
              % (tau, TAU_SECONDARY, B1, B2, B3, Bb,
                 0.5 / math.sqrt(B1), 0.5 / math.sqrt(B2), 0.5 / math.sqrt(B3)))
        print('  T09: %d structural rows over %d assays; nproc=%d'
              % (len(t09), t09['DMS_id'].nunique(), nproc))
        print('=' * 100)

    audit = interface_definition_audit(t09, verbose=verbose)
    bias = design_bias_audit(verbose=verbose)
    sites, skipped = site_counts_all(t09, verbose=verbose)
    c4s = c4s_run(assays, tau=tau, B=B1, B_boot=Bb, nproc=nproc, verbose=verbose)
    c4p = c4p_run(assays, t09=t09, B=B2, B_boot=Bb, verbose=verbose)
    tail = tail_vs_localisation_note(c4p, verbose=verbose)
    g11 = g11_twin_structures(t09, tau=tau, B=B1, B_boot=Bb, verbose=verbose)
    c4i = c4i_run(t09=t09, B=B3, g11=g11, verbose=verbose)
    lic_by_fam = {}
    for fam in sorted(c4i['records']):
        if 'failed' in c4i['records'][fam]:
            continue
        lic_by_fam[fam] = c4i_licence(c4i['records'], g11=g11, family=fam)
    lic = lic_by_fam.get('KRAS', c4i_licence(c4i['records'], g11=g11))
    for r in c4i['rows']:
        q = lic_by_fam.get(r['family'])
        if q is not None:
            r['classification'] = q['classification']
    if verbose:
        print('\n  --- C4-I licence clause, EVERY family (the spec names KRAS '
              'as the one adequately powered family; the others are probes) ---')
        for fam in sorted(lic_by_fam):
            q = lic_by_fam[fam]
            print('  %-7s F_spec(%s) %s%s  p_NS3 %s  MW_p %s  p_above %s  '
                  'median cliff PSI %s  M_F %s  fold axis %s  => %s'
                  % (fam, q['F_spec_scale'],
                     _f4(q['F_spec_noise_corrected']),
                     ' [AT THE BOUNDARY: Var(mu) estimated < 0, so 1.0000 means '
                     '"no partner-invariant component estimable", NOT "fully '
                     'partner-specific"]' if q['F_spec_at_boundary'] else '',
                     _f4(q['p_NS3']), _f4(q['MW_PSI_p']), _f4(q['p_PSI_above']),
                     _f4(q['median_cliff_PSI']), _f4(q['M_F']),
                     q['foldaxis_status'], q['classification'].upper()))
    neg = negative_control_5A12(t09, c4i=c4i['records'], verbose=verbose)

    written = {}
    if write:
        written['T09'] = write_T09_cliff_columns(sites, c4s['per_assay'],
                                                 verbose=verbose)
        written['T10'] = write_T10(c4p['frames'], verbose=verbose)
        written['T11'] = write_T11(c4i['rows'], verbose=verbose)

    if verbose:
        print('\n' + '=' * 100)
        print('C4-I LICENCE CLAUSE (spec Sec.1.5), evaluated on %s' % lic['family'])
        print('  F_spec noise-corrected %s >= %.2f : %s'
              % (_f4(lic['F_spec_noise_corrected']), THRESH['C4I_Fspec_sup'],
                 lic['clauses']['F_spec_ge_sup']))
        print('  p_NS3 %s < %.2f              : %s'
              % (_f4(lic['p_NS3']), THRESH['C4I_p_NS3_sup'],
                 lic['clauses']['p_NS3_lt_sup']))
        print('  MW PSI p %s < %.2f           : %s   (literal form %s: %s)'
              % (_f4(lic['MW_PSI_p']), THRESH['C4I_MW_p_sup'],
                 lic['clauses']['MW_PSI_lt_sup'], _f4(lic['MW_PSI_p_literal']),
                 lic['clauses']['MW_PSI_literal_lt_sup']))
        print('  REFUTING clauses: F_spec <= %.2f: %s | median cliff PSI >= '
              '%.2f: %s | G11 dual localisation: %s'
              % (THRESH['C4I_Fspec_ref'], lic['clauses']['F_spec_le_ref'],
                 THRESH['C4I_median_PSI_ref'], lic['clauses']['median_PSI_ge_ref'],
                 lic['clauses']['G11_dual']))
        print('  FOLD-AXIS VALIDATION (required): %s -- %s'
              % (lic['foldaxis_status'], lic['foldaxis_note']))
        print('  >>> CLASSIFICATION: %s' % lic['classification'].upper())
        if lic['classification'] == 'stability_cliff':
            print('  >>> "interaction cliff" is NOT licensed.  The correct name '
                  'for what this study measures is "STABILITY CLIFF".')
        print('=' * 100)
        print('stage 6 wall %.1f s' % (time.time() - t_start))
    return dict(interface_audit=audit, design_bias=bias, sites=sites,
                sites_skipped=skipped, c4s=c4s, c4p=c4p, c4i=c4i, g11=g11,
                licence=lic, licence_by_family=lic_by_fam,
                negative_control=neg, tail=tail,
                written=written, wall_s=round(time.time() - t_start, 2),
                B=dict(NS1=B1, NS2=B2, NS3=B3, bootstrap=Bb))


#: ``run_all.py`` resolves ``('stage6', 'run_all', 'run')`` in that order.
run_all = stage6
run = stage6


# =========================================================================== #
# 12. self-check                                                              #
# =========================================================================== #

def _selfcheck(verbose=True):
    """Every numerical claim this module makes about its own primitives.

    No statsmodels: the GLM, the Mantel-Haenszel OR, the AUROC, BH-FDR and the
    variance decomposition are all checked against closed forms or brute force
    computed here.
    """
    ok = []

    def _c(name, cond, detail=''):
        ok.append((name, bool(cond), detail))
        if verbose:
            print('  [%s] %-52s %s' % ('ok ' if cond else 'FAIL', name, detail))
        assert cond, name + '  ' + detail

    # ---- 1. empirical p is the study's ONLY form -------------------------- #
    p, n = empirical_p(2.0, np.array([1.0, 2.0, 3.0]))
    _c('empirical_p = (1+#{>=obs})/(B+1)', abs(p - 3.0 / 4.0) < 1e-15,
       'p=%.6f on [1,2,3] at obs=2 (2 draws >= 2)' % p)

    # ---- 2. BH-FDR -------------------------------------------------------- #
    q = bh_fdr([0.01, 0.02, 0.03, 0.04])
    want = np.array([0.04, 0.04, 0.04, 0.04])
    _c('bh_fdr matches the hand computation', np.allclose(q, want),
       '%r' % np.round(q, 6).tolist())
    q2 = bh_fdr([0.001, np.nan, 0.5])
    _c('bh_fdr passes NaN through', np.isnan(q2[1]) and abs(q2[0] - 0.002) < 1e-12,
       '%r' % np.round(q2, 6).tolist())

    # ---- 3. Poisson IRLS against the closed-form two-group rate ratio ----- #
    rng = np.random.default_rng(0)
    n_p = rng.integers(20, 200, 400).astype(float)
    grp = (np.arange(400) % 2).astype(float)
    lam = np.exp(-3.0 + 0.7 * grp)
    y = rng.poisson(lam * n_p).astype(float)
    X = np.column_stack([np.ones(400), grp])
    m = irls_glm(y, X, family='poisson', offset=np.log(n_p),
                 names=['const', 'g'])
    closed = math.log((y[grp == 1].sum() / n_p[grp == 1].sum())
                      / (y[grp == 0].sum() / n_p[grp == 0].sum()))
    _c('Poisson IRLS == closed-form log rate ratio',
       abs(m['coef']['g'] - closed) < 1e-9,
       'IRLS %.10f vs closed %.10f' % (m['coef']['g'], closed))
    _c('Poisson IRLS converged', m['converged'] and m['n_iter'] < 30,
       '%d iterations' % m['n_iter'])

    # ---- 4. binomial IRLS against the closed-form log odds ratio ---------- #
    k = np.minimum(y, n_p)
    mb = irls_glm(k, X, family='binomial', exposure=n_p, names=['const', 'g'])
    a1 = k[grp == 1].sum(); b1 = (n_p - k)[grp == 1].sum()
    a0 = k[grp == 0].sum(); b0 = (n_p - k)[grp == 0].sum()
    closed_b = math.log((a1 / b1) / (a0 / b0))
    _c('binomial IRLS == closed-form log odds ratio',
       abs(mb['coef']['g'] - closed_b) < 1e-9,
       'IRLS %.10f vs closed %.10f' % (mb['coef']['g'], closed_b))

    # ---- 5. exact collinearity is DROPPED and NAMED, never pseudo-inverted  #
    Xc = np.column_stack([np.ones(400), grp, grp])
    mc = irls_glm(y, Xc, family='poisson', offset=np.log(n_p),
                  names=['const', 'g', 'g_copy'])
    _c('_rank_reduce drops the aliased column by name',
       mc['dropped'] == ['g_copy'] and mc['rank'] == 2, '%r' % mc['dropped'])

    # ---- 6. Mantel-Haenszel OR on a textbook two-stratum table ------------ #
    kk = np.array([15.0, 20.0, 9.0, 30.0])
    nn = np.array([100.0, 200.0, 50.0, 300.0])
    ff = np.array([True, False, True, False])
    ss = np.array([0, 0, 1, 1])
    r = mh_or(kk, nn, ff, ss)
    num = den = 0.0
    for s in (0, 1):
        msk = ss == s
        a = kk[msk & ff].sum(); b = (nn - kk)[msk & ff].sum()
        c = kk[msk & ~ff].sum(); d = (nn - kk)[msk & ~ff].sum()
        N = a + b + c + d
        num += a * d / N
        den += b * c / N
    _c('mh_or == sum(ad/N)/sum(bc/N)', abs(r['OR'] - num / den) < 1e-12,
       'MH %.10f' % r['OR'])
    r1 = mh_or(kk, nn, ff, np.zeros(4, dtype=int))
    _c('mh_or on ONE stratum == the crude OR',
       abs(r1['OR'] - r1['crude_OR']) < 1e-12,
       '%.10f vs %.10f' % (r1['OR'], r1['crude_OR']))

    # ---- 7. AUROC from mid-ranks == brute-force U with ties at 0.5 -------- #
    sc = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0])
    lb = np.array([0, 1, 0, 1, 0, 1, 1], dtype=bool)
    a_fast = auroc_from_ranks(midranks(sc), lb)
    tot = 0.0
    for i in np.nonzero(lb)[0]:
        for j in np.nonzero(~lb)[0]:
            tot += 1.0 if sc[i] > sc[j] else (0.5 if sc[i] == sc[j] else 0.0)
    _c('auroc_from_ranks == brute-force U (ties at 0.5)',
       abs(a_fast - tot / (lb.sum() * (~lb).sum())) < 1e-12,
       '%.10f' % a_fast)

    # ---- 8. group_permuter reproduces nulls.permute_within_strata EXACTLY -- #
    st = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    vals = np.arange(9)
    permute, groups = group_permuter(st)
    r1 = permute(vals, np.random.default_rng(7))
    r2 = _nulls.permute_within_strata(vals, st, np.random.default_rng(7))
    _c('group_permuter == nulls.permute_within_strata (same rng state)',
       np.array_equal(r1, r2), '%r' % r1.tolist())

    # ---- 9. double centring and the Mantel correlation -------------------- #
    M = np.random.default_rng(3).normal(size=(6, 4))
    C = double_centre(M)
    _c('double_centre kills both margins',
       abs(C.sum(axis=0)).max() < 1e-12 and abs(C.sum(axis=1)).max() < 1e-12,
       'max |margin| %.2e' % max(abs(C.sum(axis=0)).max(),
                                 abs(C.sum(axis=1)).max()))
    _c('mantel_corr(Z, Z) == 1', abs(mantel_corr(M, M) - 1.0) < 1e-12)
    W = np.random.default_rng(4).normal(size=M.shape)
    _c('mantel_corr is invariant to a per-ROW shift of Z',
       abs(mantel_corr(M, W)
           - mantel_corr(M + np.arange(M.shape[0])[:, None], W)) < 1e-12,
       'row-centering removes the partner-invariant propensity ALGEBRAICALLY; '
       'M_F = %+.6f either way' % mantel_corr(M, W))

    # ---- 10. variance decomposition recovers known variances -------------- #
    G, K = 4000, 4
    rg = np.random.default_rng(11)
    mu = rg.normal(0, 1.0, G)
    dl = rg.normal(0, 0.5, (G, K))
    nz = rg.normal(0, 0.2, (G, K))
    vd = variance_decomposition(mu[:, None] + dl + nz, 0.2 ** 2)
    _c('variance_decomposition recovers Var(mu) = 1.00',
       abs(vd['var_mu'] - 1.0) < 0.06, '%.4f' % vd['var_mu'])
    _c('variance_decomposition recovers Var(delta) = 0.25 after subtracting '
       'the noise', abs(vd['var_delta'] - 0.25) < 0.02, '%.4f' % vd['var_delta'])
    _c('F_spec noise-corrected == 0.25/(1+0.25) = 0.20',
       abs(vd['F_spec_noise_corrected'] - 0.2) < 0.02,
       '%.4f' % vd['F_spec_noise_corrected'])

    # ---- 11. emp_logit is the Cox/Anscombe form --------------------------- #
    e = emp_logit(np.array([0.0]), np.array([10.0]))
    _c('emp_logit(0, 10) == log(0.5/10.5)',
       abs(e[0] - math.log(0.5 / 10.5)) < 1e-15, '%.10f' % e[0])

    # ---- 12. D2: c_hat is PHI-CENTRED, and it matters --------------------- #
    dms = 'KRAS_SOS1_norfitness_8BE4'
    ctx = _nulls.get_context(dms)
    pa = _nulls._pa_mask(ctx, ctx.censor_mask, ctx.oof_finite)
    c_on = _nulls.c_hat(ctx.e_oof, ctx.sigma_oof, ctx.nested_idx, mu=ctx.mu_oof)
    c_off = _nulls.c_hat(ctx.e_oof, ctx.sigma_oof, ctx.nested_idx, mu=None)
    n_on = int((np.abs(c_on[pa]) >= TAU_PRIMARY).sum())
    n_off = int((np.abs(c_off[pa]) >= TAU_PRIMARY).sum())
    fin = np.isfinite(ctx.e_oof) & np.isfinite(ctx.phi_oof)
    corr = float(np.corrcoef(ctx.e_oof[fin], ctx.phi_oof[fin])[0, 1])
    cc = cliff_counts(dms, centred=True)
    _c('cliff_counts(centred=True) honours D2 (phi-centred c_hat)',
       int(cc['n_cliff_pairs'].sum()) == n_on,
       'centred %d cliffs vs uncentred %d on %s; corr(e_oof, phi_oof) = %+.4f'
       % (n_on, n_off, dms, corr))
    _c('the uncentred form would manufacture a DIFFERENT cliff set',
       n_on != n_off, 'the two counts differ by %d pairs' % abs(n_on - n_off))

    # ---- 13. C3's cached eps table and cliff.noise agree on bg_order 0 ---- #
    from cliff import stats_c3 as _c3
    b = _c3.get_bundle(dms)
    tab = _c3.cached_epsilon_table(b)
    inv = {v: k for k, v in b.ctx.col_index.items()} \
        if hasattr(b.ctx, 'col_index') else None
    z0 = tab[tab['bg_order'].values == 0]
    ep = _noise.epsilon_sitepairs(io_bgym.load_assay(dms))
    _c('C3 owns the eps CACHE, cliff.noise owns the eps DEFINITION, and the '
       'bg_order = 0 slice matches',
       len(z0) == ep['n_usable'],
       '%d rows in %s vs %d from cliff.noise.epsilon_sitepairs'
       % (len(z0), os.path.basename(_c3.eps_cache_path(dms)), ep['n_usable']))
    _c('and the eps VALUES agree to 1e-12',
       abs(np.sort(z0['eps'].values) - np.sort(ep['eps'])).max() < 1e-12,
       'max |diff| %.3e'
       % abs(np.sort(z0['eps'].values) - np.sort(ep['eps'])).max())

    # ---- 14. the BH3 offsets are the WT-verified ones, not the naive join -- #
    offs, ref = resolve_offsets(C4I_FAMILIES['BH3']['partners'])
    got = {d: offs[d]['offset'] for d in offs}
    n_ov = {d: offs[d]['n_overlap'] for d in offs}
    _c('resolve_offsets finds a WT-consistent BH3 offset with ZERO mismatches',
       all(offs[d].get('unique_at_max') for d in offs),
       'offsets %r overlaps %r (ref %s)' % (got, n_ov, ref))

    # ---- 15. the D6 non-binding dSASA cut ---------------------------------- #
    a6 = interface_definition_audit(verbose=False)
    _c('dSASA > %g is EXACTLY dSASA > 0 on this benchmark (the cut is '
       'non-binding, so varying it is a flat line)' % THRESH['C4_dsasa_min_A2'],
       a6['dsasa_cut_is_nonbinding'] and a6['n_dsasa_gt0'] == a6['n_dsasa_gt1'],
       '%d == %d, none in (0, %g]'
       % (a6['n_dsasa_gt0'], a6['n_dsasa_gt1'], THRESH['C4_dsasa_min_A2']))
    _c('dSASA > 0 == Levy support+rim+core, position for position',
       a6['dsasa_equals_levy_burial'], '%d positions' % a6['n_levy_burial'])

    # ---- 16. the designed negative control -------------------------------- #
    ng = negative_control_5A12(verbose=False)['5A12_VEGF_fitness_4ZFF']
    _c('5A12_VEGF: all 9 positions have dSASA EXACTLY 0.000',
       ng['all_dsasa_exactly_zero'] and ng['n_positions'] == 9,
       'max dSASA %.6f over %d positions' % (ng['max_dsasa'], ng['n_positions']))
    _c('the 6.4 A form clears the cut by < 0.01 A and flips at 6.5 A',
       0 < ng['margin_at_6p4'] < 0.01 and ng['n_within_6p5'] > 0,
       'closest position clears 6.4 A by %.4f A; %d of %d positions inside '
       '6.5 A'
       % (ng['margin_at_6p4'], ng['n_within_6p5'], ng['n_positions']))

    if verbose:
        print('\n_selfcheck: %d/%d checks passed'
              % (sum(1 for _n, c, _d in ok if c), len(ok)))
    return ok


if __name__ == '__main__':                                     # pragma: no cover
    import argparse as _ap
    ap = _ap.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--selfcheck', action='store_true',
                    help='run the numerical self-check and exit')
    ap.add_argument('--run', action='store_true', help='run stage 6')
    ap.add_argument('--nproc', type=int, default=1)
    ap.add_argument('--B-NS1', type=int, default=None)
    ap.add_argument('--B-NS2', type=int, default=None)
    ap.add_argument('--B-NS3', type=int, default=None)
    ap.add_argument('--B-boot', type=int, default=None)
    ap.add_argument('--no-write', action='store_true')
    ap.add_argument('--assays', default=None,
                    help='comma-separated DMS_ids (default: the C4 set)')
    a = ap.parse_args()
    if a.selfcheck or not a.run:
        print('cliff.stats_c4 self-check')
        _selfcheck()
    if a.run:
        ids = a.assays.split(',') if a.assays else None
        stage6(assays=ids, nproc=a.nproc, verbose=True,
               B_NS1_=a.B_NS1, B_NS2_=a.B_NS2, B_NS3_=a.B_NS3,
               B_boot=a.B_boot, write=not a.no_write)
