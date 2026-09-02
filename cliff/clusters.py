"""BGYM-CLIFF v1 -- the GATED y-blind cluster channel (spec Sec.3 ``clusters.py``, T15).

This is the USER'S OWN formulation of the hypothesis ("group sequences into
neighbourhoods, then ask which member jumps"), so it is implemented seriously and
its weaknesses are reported next to its strengths rather than in place of them.

DO NOT CHANGE, in order of how expensive the mistake is:

1. **``n <= THRESH['cluster_n_max']`` is a hard gate.**  GB1_IgG-Fc_1FCC is
   4.314e9 condensed entries (34.5 GB, and past int32); CR9114 x2 are 17.2 GB
   each; Z-LL1 is 8.3 GB.  All four have ample pair-channel power, so the channel
   is not needed there and is refused there.
2. **The statistic is the leave-one-out robust z of the CROSS-FITTED ADDITIVE
   RESIDUAL ``e``, never of ``y`` and never of ``z``.**  On ``y`` a flagged
   variant conflates a large MAIN EFFECT with a background-dependent jump, and
   that distinction is the entire study.
3. **Clustering is y-blind.**  The tree is a function of the design matrix only.
   Nothing about ``y``, ``e`` or the null may enter ``linkage``.
4. **This channel may only ADD an assay to the C2 count where the pair channel is
   POWERLESS, never override it, and it may NEVER support an epistasis-ORDER
   claim** -- it is order-mixed by construction (a cluster of radius >= 1 mixes
   nested with same-site steps, and ``order_range_within_cluster`` measures how
   much).  ``adds_assay_to_C2_count`` is therefore False unless
   ``n_nested < THRESH['min_nested_for_pair_channel']``.
5. **``scipy.spatial.distance.pdist`` is banned** (single-threaded, and it
   re-derives what one BLAS ``dgemm`` gives exactly): the condensed array comes
   from a blocked Gram, ``D2 = m_u + m_v - 2 X X^T``, which is EXACT in float64
   for a binary design (verified: max|blocked - pdist| = 0).
6. CD19_FMC63_7URV is stamped STRUCTURALLY_UNIDENTIFIED (ORCHESTRATOR D3): it is
   run, reported and flagged, and it may not contribute in either direction.

Every threshold is read from :mod:`cliff.config`; every seed from
``config.SEEDS``.  No verdict is emitted here -- ``cliff/verdict.py`` reads T15.
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
import threading
import time

import numpy as np
import scipy.sparse as sp
import scipy.cluster.hierarchy as sch
from sklearn.metrics import adjusted_rand_score

from cliff import config
from cliff import latent as _latent
from cliff import nulls as _nulls
from cliff import pairs as _pairs
from cliff.config import PATHS, SEEDS, TAUS, THRESH

__all__ = [
    'ELIGIBLE', 'condensed_hamming', 'ward_tree', 'cluster_geometry',
    'target_height', 'loo_robust_z', 'cluster_channel', 'pair_channel_flags',
    'T15_COLUMNS', 'T15_EXTRA_COLUMNS', 'write_T15', 'stage5',
]

# --------------------------------------------------------------------------- #
# who is eligible                                                             #
# --------------------------------------------------------------------------- #

#: The six assays spec Sec.3 lists as eligible (all ``n <= 30,000``, condensed
#: <= 3.6 GB).  Read from the registry, not hard-coded, so the two lists cannot
#: drift apart; the assertion below is the check.
ELIGIBLE = tuple(a for a in config.ALL_ASSAYS
                 if config.ASSAYS[a].eligible_cluster_channel)

_SPEC_ELIGIBLE = (
    '4D5_HER2_fitness_1N8Z', '5A12_VEGF_fitness_4ZFF', 'CD19_FMC63_Fitness_7URV',
    'GB1_IgG-Fc_fitness_1FCC_2016', 'Z-domain_ZpA963_HL1_fitness_2M5A',
    'hYAP65_peptide_FunctioncalScore_1JMQ')
assert set(ELIGIBLE) == set(_SPEC_ELIGIBLE), (sorted(ELIGIBLE),
                                              sorted(_SPEC_ELIGIBLE))

#: ORCHESTRATOR D3.  Runs, is reported, never contributes.
STRUCTURALLY_UNIDENTIFIED = ('CD19_FMC63_Fitness_7URV',)

#: The flag threshold.  ``TAU_WINDOW[0]`` is the low end of the spec's C2
#: consecutive-tau window, i.e. the same "3 sigma" the rest of the study calls a
#: cliff (C3-N's ``3 sigma_eps``, L5's ``|eps| >= 3 sigma``).  A literal here
#: would be a bug; the whole sweep over ``config.TAUS`` is reported in
#: ``x_sweep_json``.
TAU = float(config.TAU_WINDOW[0])

#: Fraction kept by each of the 5 ARI subsamples.  NOT in the spec (which fixes
#: only ``cluster_ari_n_seeds = 5``), so it is named here and reported in the
#: table rather than buried.
SUBSAMPLE_FRAC = 0.80

#: Blocked-Gram row block.  ``block x n x 8`` bytes of scratch: 0.49 GB at
#: ``n = 29,981``, which is the largest eligible assay.
GRAM_BLOCK = 2048

_MAD = THRESH['mad_const']          # 1.4826.  MAD, never sd.


# --------------------------------------------------------------------------- #
# peak-RSS sampling                                                           #
# --------------------------------------------------------------------------- #

def _rss_gb():
    """Resident set size of THIS process, GB, from ``/proc/self/statm``."""
    try:
        with open('/proc/self/statm') as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf('SC_PAGE_SIZE') / 2.0 ** 30
    except (OSError, IndexError, ValueError):       # pragma: no cover
        return float('nan')


class _PeakRSS(object):
    """Per-assay peak RSS.  ``resource.ru_maxrss`` is a high-water mark that
    never falls, so it cannot attribute a peak to one assay; this samples
    ``/proc/self/statm`` instead and is reset per assay."""

    def __init__(self, interval=0.05):
        self.interval = interval
        self.peak = 0.0
        self._stop = threading.Event()
        self._t = None

    def __enter__(self):
        self.peak = _rss_gb()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def _loop(self):
        while not self._stop.wait(self.interval):
            r = _rss_gb()
            if r > self.peak:
                self.peak = r

    def __exit__(self, *exc):
        r = _rss_gb()
        if r > self.peak:
            self.peak = r
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=1.0)
        return False


# --------------------------------------------------------------------------- #
# the condensed distance array                                                #
# --------------------------------------------------------------------------- #

def condensed_hamming(X, block=GRAM_BLOCK, rows=None):
    """Condensed EUCLIDEAN distances ``sqrt(Hamming)`` for a binary design.

    Spec Sec.3: "Blocked BLAS Gram -> condensed float64 -> scipy
    linkage(method='ward'); NEVER scipy's internal single-threaded pdist."

    ``Euclidean^2 == Hamming`` on ``{0,1}`` columns, so
    ``D2[u,v] = m_u + m_v - 2 (X X^T)[u,v]`` with ``m = row nnz``.  One ``dgemm``
    per row block; exact in float64 because every term is a small integer
    (verified against ``pdist``: max abs difference 0).

    ``rows`` restricts to a subset (the ARI subsamples) without materialising a
    second design matrix.  Returns a freshly allocated C-contiguous float64
    array of ``n(n-1)/2`` entries -- note that ``scipy``'s ``nn_chain`` COPIES
    it, so the linkage call costs 2x this array (measured, see T15
    ``peak_RAM_GB``).
    """
    Xs = X if rows is None else X[rows]
    Xd = np.ascontiguousarray(Xs.toarray(), dtype=np.float64)
    n = Xd.shape[0]
    if n < 2:
        return np.empty(0, dtype=np.float64)
    m = Xd.sum(axis=1)
    out = np.empty(n * (n - 1) // 2, dtype=np.float64)
    ar = np.arange(n, dtype=np.int64)
    starts = ar * n - ar * (ar + 1) // 2          # condensed offset of row i
    for i0 in range(0, n, int(block)):
        i1 = min(i0 + int(block), n)
        G = Xd[i0:i1] @ Xd.T
        for i in range(i0, i1):
            if i == n - 1:
                continue
            d2 = m[i] + m[i + 1:] - 2.0 * G[i - i0, i + 1:]
            np.maximum(d2, 0.0, out=d2)           # -0.0 from cancellation only
            out[starts[i]: starts[i] + (n - i - 1)] = np.sqrt(d2)
        del G
    return out


def ward_tree(X, method='ward', block=GRAM_BLOCK, rows=None):
    """``(Z, wall_s)``.  y-BLIND by construction: only ``X`` enters."""
    t0 = time.time()
    cd = condensed_hamming(X, block=block, rows=rows)
    Z = sch.linkage(cd, method=method)
    del cd
    return Z, time.time() - t0


# --------------------------------------------------------------------------- #
# cluster geometry -- closed forms, no pairwise matrix                        #
# --------------------------------------------------------------------------- #

def cluster_geometry(labels, X, mrow, K=None):
    """Within-cluster dispersion of a labelling, in closed form.

    ``SS_c = sum_i ||x_i - xbar_c||^2 = sum_i m_i - ||sum_i x_i||^2 / n_c`` and,
    because ``Euclidean^2 == Hamming`` here, ``SS_c`` is literally the
    within-cluster HAMMING dispersion Ward minimises.  Returns

    * ``cnt``    -- members per cluster;
    * ``radius`` -- ``SS_c / n_c``, the mean HAMMING distance from a member to
      the cluster centroid.  **This is what ``rho`` targets** (spec Sec.3's own
      units: "Ward minimises within-cluster Hamming dispersion"), so ``rho = 1``
      is a cluster one substitution wide;
    * ``rms``    -- ``sqrt(SS_c/n_c)``, the same thing as a Euclidean radius;
    * ``mean_pair`` -- mean pairwise Hamming inside the cluster,
      ``2 SS_c/(n_c-1)``, i.e. ``~2 x radius`` (the exact conversion, so a reader
      who prefers that convention can map any row of T15).

    ``np.bincount``, never ``np.add.at``.
    """
    labels = np.asarray(labels)
    K = int(labels.max()) + 1 if K is None else int(K)
    cnt = np.bincount(labels, minlength=K).astype(np.float64)
    ind = sp.csr_matrix((np.ones(labels.size, dtype=np.float64),
                         (labels, np.arange(labels.size))),
                        shape=(K, labels.size))
    S = ind @ X                                    # K x M cluster column sums
    sq = np.asarray(S.multiply(S).sum(axis=1)).ravel()
    sm = np.bincount(labels, weights=mrow, minlength=K)
    SS = np.maximum(sm - sq / np.maximum(cnt, 1.0), 0.0)
    radius = SS / np.maximum(cnt, 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_pair = np.where(cnt >= 2, 2.0 * SS / np.maximum(cnt - 1.0, 1.0),
                             np.nan)
    return cnt, radius, np.sqrt(radius), mean_pair


def _group_range(labels, values, K):
    """``max - min`` of ``values`` within each of ``K`` groups, by lexsort."""
    out = np.full(int(K), np.nan)
    if labels.size == 0:
        return out
    o = np.lexsort((values, labels))
    ls, vs = labels[o], values[o]
    first = np.concatenate(([True], ls[1:] != ls[:-1]))
    last = np.concatenate((ls[1:] != ls[:-1], [True]))
    out[ls[first]] = vs[last] - vs[first]
    return out


def seq_hamming_radius(labels, codes, K=None):
    """Mean pairwise SEQUENCE Hamming inside each cluster, halved so it is on the
    same footing as :func:`cluster_geometry`'s ``radius``.

    The substitution-set metric puts a SAME-SITE swap at distance 2 and a NESTED
    step at 1, while the study's ``h`` axis (ORCHESTRATOR D1) calls both lag 1.
    Reporting both is how the order-mixing claim is measured rather than
    asserted.  Per position ``p``, the number of differing pairs in cluster ``c``
    is ``(n_c^2 - sum_a k_{c,a}^2)/2`` over the code counts ``k``.
    """
    labels = np.asarray(labels)
    codes = np.asarray(codes)
    K = int(labels.max()) + 1 if K is None else int(K)
    n, P = codes.shape
    cnt = np.bincount(labels, minlength=K).astype(np.float64)
    ncode = int(codes.max()) + 1 if codes.size else 1
    diff = np.zeros(K, dtype=np.float64)
    for p in range(P):
        h = np.bincount(labels.astype(np.int64) * ncode + codes[:, p],
                        minlength=K * ncode).astype(np.float64)
        h = h.reshape(K, ncode)
        diff += 0.5 * (cnt * cnt - (h * h).sum(axis=1))
    npair = cnt * (cnt - 1.0) / 2.0
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(npair > 0, diff / np.maximum(npair, 1.0), np.nan) * 0.5


# --------------------------------------------------------------------------- #
# rho -> cut height                                                           #
# --------------------------------------------------------------------------- #

def _labels_at(Z, h):
    return sch.fcluster(Z, t=h, criterion='distance') - 1


def _achieved(Z, h, X, mrow, eligible, min_size, n):
    """``(radius, K, n_ge, covered)`` at cut height ``h``.

    ``n_c`` counts only ELIGIBLE members -- a cluster whose ``e^oof`` is finite
    for 3 of its 20 rows cannot supply a leave-one-out robust z, so it is not a
    cluster of 20 for this channel's purposes.  The ``rho`` target is read off
    the ``n_c``-weighted mean radius over the KEPT clusters, which is the
    neighbourhood the average covered variant actually sits in.
    """
    lab = _labels_at(Z, h)
    K = int(lab.max()) + 1
    cnt, radius, _rms, _mp = cluster_geometry(lab, X, mrow, K=K)
    ecnt = np.bincount(lab[eligible], minlength=K).astype(np.float64)
    keep = ecnt >= min_size
    w = ecnt[keep]
    if not keep.any() or w.sum() <= 0:
        return float('nan'), K, 0, 0.0, lab, keep, ecnt
    r = float((radius[keep] * w).sum() / w.sum())
    return r, K, int(keep.sum()), float(w.sum() / n), lab, keep, ecnt


def target_height(Z, X, mrow, eligible, rho, min_size, n, verbose=False):
    """Cut height whose kept-cluster mean radius is closest to ``rho``.

    Bisection over the sorted unique merge heights (the radius is monotone
    non-decreasing in the height up to the ``>= min_size`` filter's reshuffling,
    so the bisection is followed by a local scan that makes the answer the true
    arg-min over the ladder rather than trusting monotonicity).

    Returns ``(h, index, degenerate)`` where ``degenerate`` is
    ``''`` | ``'rho_below_min_attainable'`` | ``'rho_above_max_attainable'``.
    The two degenerate ends are the ones spec Sec.3 requires to be reported
    verbatim: as ``rho -> 0`` every cluster is a singleton and **the statistic
    does not exist**; as ``rho -> max`` there is one cluster and it **degenerates
    into a global outlier test on the marginal**.
    """
    hts = np.unique(Z[:, 2])
    eps = 1e-9

    def ev(i):
        return _achieved(Z, hts[i] + eps, X, mrow, eligible, min_size, n)[0]

    lo, hi = 0, hts.size - 1
    r_hi = ev(hi)
    # the coarsest cut is one cluster; if even that radius is below rho the
    # target is unattainable and the answer is the root
    if np.isfinite(r_hi) and r_hi < rho:
        return hts[hi] + eps, hi, 'rho_above_max_attainable'
    # smallest index that has any kept cluster at all
    j0 = 0
    for i in np.unique(np.linspace(0, hts.size - 1, 40).astype(int)):
        if np.isfinite(ev(int(i))):
            j0 = int(i)
            break
    else:                                            # pragma: no cover
        return hts[hi] + eps, hi, 'no_cluster_reaches_min_size'
    # refine j0 downward to the first index with a finite radius
    a, b = 0, j0
    while a < b:
        mid = (a + b) // 2
        if np.isfinite(ev(mid)):
            b = mid
        else:
            a = mid + 1
    j0 = a
    r_min = ev(j0)
    if np.isfinite(r_min) and r_min > rho:
        return hts[j0] + eps, j0, 'rho_below_min_attainable'
    lo, hi = j0, hts.size - 1
    while lo < hi:                                   # first index with r >= rho
        mid = (lo + hi) // 2
        r = ev(mid)
        if np.isfinite(r) and r >= rho:
            hi = mid
        else:
            lo = mid + 1
    cands = [i for i in range(max(j0, lo - 4), min(hts.size, lo + 5))]
    best, best_d = lo, float('inf')
    for i in cands:
        r = ev(i)
        if not np.isfinite(r):
            continue
        d = abs(r - rho)
        if d < best_d - 1e-12:
            best, best_d = i, d
    if verbose:
        print('        rho=%.2f -> h=%.4f (idx %d/%d) radius=%.4f'
              % (rho, hts[best], best, hts.size - 1, ev(best)))
    return hts[best] + eps, best, ''


# --------------------------------------------------------------------------- #
# the statistic: leave-one-out robust z of the cross-fitted residual          #
# --------------------------------------------------------------------------- #

def _loo_median_from_sorted(vs, rank):
    """Median of ``vs`` (sorted, length ``m``) with the element at sorted
    position ``rank`` removed, as a closed-form order statistic.

    ``m`` even  -> ``m-1`` odd : one middle,  ``vs[m//2]`` if ``rank < m//2``
                                 else ``vs[m//2-1]``.
    ``m`` odd   -> ``m-1`` even: two middles ``k1 = (m-3)//2``, ``k2 = k1+1``,
                                 each shifted up by one when it sits at or after
                                 the hole.
    """
    m = vs.size
    if m < 2:
        return np.full(rank.shape, np.nan)
    if m % 2 == 0:
        k = m // 2 - 1
        return np.where(rank <= k, vs[k + 1], vs[k])
    k1 = (m - 3) // 2
    k2 = k1 + 1
    a = np.where(k1 < rank, vs[k1], vs[k1 + 1])
    b = np.where(k2 < rank, vs[k2], vs[k2 + 1])
    return 0.5 * (a + b)


def _loo_med_and_mad(v):
    """Exact leave-one-out median and MAD of ``v`` in ``O(m log m)``.

    ``med^{-i}`` takes at most THREE distinct values over ``i`` (two when ``m``
    is even), because it is an order statistic of the same sorted array with one
    hole in it.  So ``MAD^{-i} = median_{l != i} |v_l - med^{-i}|`` needs at most
    three sorts, not the ``O(m^2)`` matrix -- which is what makes a 200-replicate
    permutation null on a 30,000-variant assay affordable.  Verified against a
    brute-force ``O(m^2)`` implementation in :func:`_selfcheck`.
    """
    m = v.size
    out_med = np.full(m, np.nan)
    out_mad = np.full(m, np.nan)
    if m < 2:
        return out_med, out_mad
    order = np.argsort(v, kind='stable')
    vs = v[order]
    rank = np.empty(m, dtype=np.int64)
    rank[order] = np.arange(m, dtype=np.int64)
    out_med = _loo_median_from_sorted(vs, rank)
    for c in np.unique(out_med):
        sel = out_med == c
        a = np.abs(v - c)
        ao = np.argsort(a, kind='stable')
        asort = a[ao]
        arank = np.empty(m, dtype=np.int64)
        arank[ao] = np.arange(m, dtype=np.int64)
        out_mad[sel] = _loo_median_from_sorted(asort, arank[sel])
    return out_med, out_mad


def loo_robust_z(values, labels, keep_mask, sigma_noise, min_size):
    """``t_i = (e_i - med_c^{-i}) / max(1.4826 MAD_c^{-i}, sigma_noise_a)``.

    ``values`` is the CROSS-FITTED ADDITIVE RESIDUAL ``e^oof`` (spec Sec.3: "not
    of ``z``"), ``labels`` the y-blind cluster labelling, ``keep_mask`` the
    eligible rows (finite ``phi^oof``, uncensored).  ``nan`` for a row outside a
    kept cluster; the ``sigma_noise`` floor is what stops a cluster whose members
    happen to agree to 1e-16 from manufacturing an infinite z.
    """
    t = np.full(values.size, np.nan)
    den = np.full(values.size, np.nan)
    idx = np.flatnonzero(keep_mask)
    if idx.size == 0:
        return t, den
    lab = labels[idx]
    order = np.argsort(lab, kind='stable')
    idx = idx[order]
    lab = lab[order]
    bnd = np.flatnonzero(np.diff(lab)) + 1
    for a, b in zip(np.concatenate(([0], bnd)),
                    np.concatenate((bnd, [lab.size]))):
        if b - a < min_size:
            continue
        rows = idx[a:b]
        v = values[rows]
        med, mad = _loo_med_and_mad(v)
        d = np.maximum(_MAD * mad, sigma_noise)
        t[rows] = (v - med) / d
        den[rows] = d
    return t, den


def eta2_residual(values, labels, keep_mask, kept_clusters):
    """One-way ``eta^2 = SSB/SST`` of the residual over the kept clusters.

    Reported WITH its N2 mean, because ``eta^2`` rises mechanically with the
    number of groups: a finer cut always explains more.  The excess over the
    permutation null is the part that means anything.
    """
    sel = keep_mask & kept_clusters[labels]
    v = values[sel]
    if v.size < 4:
        return float('nan'), 0, 0
    g = labels[sel]
    _u, gi = np.unique(g, return_inverse=True)
    ng = gi.max() + 1
    cnt = np.bincount(gi, minlength=ng).astype(np.float64)
    tot = np.bincount(gi, weights=v, minlength=ng)
    gm = tot / cnt
    grand = v.mean()
    ssb = float((cnt * (gm - grand) ** 2).sum())
    sst = float(((v - grand) ** 2).sum())
    return (ssb / sst if sst > 0 else float('nan')), int(v.size), int(ng)


# --------------------------------------------------------------------------- #
# the pair channel, for the rho = 1 Jaccard                                   #
# --------------------------------------------------------------------------- #

def pair_channel_flags(ctx, tau=TAU):
    """The pair channel's NESTED result, as a per-variant flag.

    ``c_hat`` is ORCHESTRATOR D2's phi-centred form on ``P_a`` (conditions
    (a) ``B != {}``, (b) uncensored, (c) finite ``phi^oof``), taken straight from
    :mod:`cliff.nulls` so this is the same number T06 will carry -- T06 does not
    exist yet, and re-deriving ``c_hat`` here rather than importing it would have
    been a second definition of the study's central statistic.

    The two channels flag different objects (edges vs variants), so the Jaccard
    needs a common universe: a variant is pair-flagged iff it is an ENDPOINT of a
    nested edge in ``P_a`` with ``|c_hat| >= tau``.  ``flag_v`` is the stricter
    reading (only the larger endpoint ``B u {i}``, the one whose deviation
    ``c_hat`` actually attributes the jump to) and is reported beside it.
    """
    keep = _nulls._pa_mask(ctx, ctx.censor_mask, ctx.oof_finite)
    idx = ctx.nested_idx[keep]
    c = _nulls.c_hat(ctx.e_oof, ctx.sigma_oof, idx, mu=ctx.mu_oof)
    fin = np.isfinite(c)
    idx, c = idx[fin], c[fin]
    covered = np.zeros(ctx.n, dtype=bool)
    covered[idx.ravel()] = True
    hit = np.abs(c) >= tau
    flag = np.zeros(ctx.n, dtype=bool)
    flag[idx[hit].ravel()] = True
    flag_v = np.zeros(ctx.n, dtype=bool)
    flag_v[idx[hit][:, 1]] = True
    return dict(covered=covered, flag=flag, flag_v=flag_v,
                n_Pa=int(idx.shape[0]), n_edge_flag=int(hit.sum()),
                rate_edge=float(hit.mean()) if hit.size else float('nan'))


def _jaccard(a, b):
    u = int((a | b).sum())
    return (float((a & b).sum()) / u) if u else float('nan')


def _bh_fdr(p):
    """Benjamini-Hochberg over the ASSAYS (never over edges).  ~10 lines on
    numpy because statsmodels is deliberately not a dependency (spec Sec.4)."""
    p = np.asarray(p, dtype=np.float64)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    if not ok.any():
        return q
    pv = p[ok]
    m = pv.size
    o = np.argsort(pv, kind='stable')
    ranked = pv[o] * m / (np.arange(m) + 1.0)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


# --------------------------------------------------------------------------- #
# T15                                                                         #
# --------------------------------------------------------------------------- #

#: spec Sec.6 T15, verbatim and in order.  Nothing is renamed, reordered or
#: dropped; the ``x_*`` block below is APPENDED (verdict.py reads T15 by name).
T15_COLUMNS = [
    'DMS_id', 'linkage', 'rho_target', 'K', 'n_clusters_ge8',
    'frac_variants_covered', 'coverage_gate_pass', 'mean_within_radius',
    'order_range_within_cluster', 'eta2_residual', 's_rho',
    'ari_ward_vs_average', 'ari_subsample_5seed', 'n_cliff', 'cliff_rate',
    'T_N2', 'jaccard_vs_pair_channel_rho1', 'peak_RAM_GB', 'wall_s',
    'adds_assay_to_C2_count',
]

#: Everything a reader needs to audit the 20 spec columns without re-running the
#: channel.  ``x_`` marks them as beyond spec Sec.6.
T15_EXTRA_COLUMNS = [
    'x_n', 'x_n_eligible', 'x_n_covered', 'x_frac_eligible_covered',
    'x_tau', 'x_sigma_noise_a', 'x_sigma_noise_provenance',
    'x_mean_within_rms', 'x_mean_within_pairwise_hamming',
    'x_mean_within_radius_seq', 'x_radius_min_attainable',
    'x_radius_max_attainable', 'x_degenerate_end',
    'x_eta2_N2_mean', 'x_eta2_p_N2', 'x_cliff_rate_N2_mean', 'x_p_perm_N2',
    'x_q_BH', 'x_B_null', 'x_ari_subsample_sd', 'x_ari_subsample_frac',
    'x_n_flag_cluster', 'x_n_flag_pair', 'x_n_universe_jaccard',
    'x_n_flag_both', 'x_jaccard_v_endpoint_only', 'x_jaccard_unrestricted',
    'x_n_Pa', 'x_n_nested', 'x_pair_channel_powerless',
    'x_structurally_unidentified', 'x_order_mixed_no_order_claim',
    'x_sweep_json', 'x_note',
]

ALL_COLUMNS = T15_COLUMNS + T15_EXTRA_COLUMNS

#: The two degenerate ends spec Sec.3 requires to be "reported verbatim".
_DEGENERATE_TEXT = {
    'rho_below_min_attainable':
        'rho is BELOW the smallest attainable within-cluster radius on this '
        'tree: no cut has a cluster of the required size at that radius.  As '
        'rho -> 0 every cluster is a singleton and THE STATISTIC DOES NOT '
        'EXIST (there is no leave-one-out median of an empty set).  Reported '
        'at the finest attainable cut instead, and rho_target is therefore NOT '
        'the radius achieved -- read mean_within_radius.',
    'rho_above_max_attainable':
        'rho is ABOVE the radius of the ROOT: the whole assay is one cluster, '
        'so at this rho the channel DEGENERATES INTO A GLOBAL OUTLIER TEST ON '
        'THE MARGINAL residual and is no longer a neighbourhood test at all.  '
        'Reported at the root; rho_target is NOT the radius achieved.',
    'no_cluster_reaches_min_size':
        'NO cut of this tree has a single cluster with enough eligible '
        'members; the channel does not exist for this assay.',
}


# --------------------------------------------------------------------------- #
# one assay                                                                   #
# --------------------------------------------------------------------------- #

def cluster_channel(dms_id, rho_targets=None, *, n_max=None, min_size=None,
                    B=None, linkages=('ward', 'average'), taus=TAUS,
                    verbose=True):
    """The gated channel for one assay -> a list of T15 row dicts.

    ``RUNS ONLY IF n <= n_max`` (spec Sec.3).  Everything else is measured:
    the coverage gate is not a filter on what gets reported, it is itself the
    reported finding when it fails (4D5 is the pre-declared test case).
    """
    rho_targets = tuple(THRESH['cluster_rho_targets'] if rho_targets is None
                        else rho_targets)
    n_max = THRESH['cluster_n_max'] if n_max is None else int(n_max)
    min_size = THRESH['cluster_min_size'] if min_size is None else int(min_size)
    B = THRESH['null_B'] if B is None else int(B)
    t_assay = time.time()
    mon = _PeakRSS()
    mon.__enter__()

    des = _latent.load_cached_design(dms_id, verify=True)
    n = int(des['y'].size)
    if n > n_max:
        raise ValueError(
            'cluster channel REFUSED for %s: n = %d > cluster_n_max = %d; '
            'condensed would be %.3g entries = %.1f GB'
            % (dms_id, n, n_max, n * (n - 1) / 2, n * (n - 1) / 2 * 8 / 2 ** 30))

    ctx = _nulls.get_context(dms_id, verify=True)
    X = ctx.X.tocsr()
    mrow = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    codes = des['codes']
    e = ctx.e_oof
    eligible = ctx.oof_finite & ~ctx.censor_mask & np.isfinite(e)
    n_elig = int(eligible.sum())

    # --- sigma_noise_a: the assay's own CROSS-FITTED residual scale ---------- #
    # The registry's `internal_residual` provenance (spec Sec.1.0 / T03) is "the
    # assay's own cross-fitted residual MAD per phi-decile", and the numerator
    # here is a deviation of e^oof, so the floor has to be on the OOF scale too.
    # T03 reports the median of the FULL-fit (in-sample) sigma knots, which on
    # 4D5 is 0.0952 against an OOF residual MAD of 1.0847 -- 11x apart, because
    # 4D5's in-sample additive fit is nearly saturated.  Using the in-sample
    # number as a floor for an OOF numerator would be no floor at all.
    sig_oof = ctx.sigma_oof[eligible]
    sigma_noise = float(np.median(sig_oof[np.isfinite(sig_oof)]))
    sigma_prov = ('median sigma^oof over eligible rows (internal_residual, '
                  'LATENT scale, cross-fitted to match the numerator)')
    pair = pair_channel_flags(ctx, tau=TAU)
    n_nested = int(ctx.nested_idx.shape[0])
    powerless = n_nested < THRESH['min_nested_for_pair_channel']
    unident = dms_id in STRUCTURALLY_UNIDENTIFIED

    if verbose:
        print('  [clusters] %s n=%d eligible=%d (%.4f) M=%d P=%d '
              'sigma_noise=%.5f  n_nested=%d |P_a|=%d powerless=%s'
              % (dms_id, n, n_elig, n_elig / n, ctx.M, ctx.P, sigma_noise,
                 n_nested, pair['n_Pa'], powerless))

    trees, cut = {}, {}
    try:
        for meth in linkages:
            Z, tw = ward_tree(X, method=meth)
            trees[meth] = Z
            if verbose:
                print('    linkage=%-8s %.1fs  %d merge heights  %.3f..%.3f'
                      % (meth, tw, np.unique(Z[:, 2]).size, Z[0, 2], Z[-1, 2]))
        # attainable radius range, on the primary tree
        Zp = trees[linkages[0]]
        hts = np.unique(Zp[:, 2])
        r_max = _achieved(Zp, hts[-1] + 1e-9, X, mrow, eligible, min_size, n)[0]
        r_min = float('nan')
        for i in range(hts.size):
            r = _achieved(Zp, hts[i] + 1e-9, X, mrow, eligible, min_size, n)[0]
            if np.isfinite(r):
                r_min = r
                break
        # ---- the cuts -------------------------------------------------------- #
        for meth in linkages:
            Z = trees[meth]
            for rho in rho_targets:
                h, _i, degen = target_height(Z, X, mrow, eligible, float(rho),
                                             min_size, n, verbose=verbose)
                r, K, n_ge, cov, lab, keep, ecnt = _achieved(
                    Z, h, X, mrow, eligible, min_size, n)
                cut[(meth, rho)] = dict(h=h, degen=degen, radius=r, K=K,
                                        n_ge=n_ge, cov=cov, labels=lab,
                                        keep=keep, ecnt=ecnt)
        # ---- ward vs average ARI, at MATCHED K ------------------------------- #
        # the two linkages' heights are not commensurable (average's are mean
        # distances, ward's are Lance-Williams increments), so the honest
        # comparison cuts the average tree to the ward K rather than to the same
        # height.
        for rho in rho_targets:
            a = cut.get(('ward', rho))
            if a is None or 'average' not in trees:
                continue
            Kw = int(a['labels'].max()) + 1
            lb = sch.fcluster(trees['average'], t=Kw, criterion='maxclust') - 1
            ari = float(adjusted_rand_score(a['labels'], lb))
            a['ari_wa'] = ari
            if ('average', rho) in cut:
                cut[('average', rho)]['ari_wa'] = ari
        # ---- 5-seed subsample ARI -------------------------------------------- #
        ss = {}
        rng = np.random.default_rng(config.assay_seed('cluster_subsample',
                                                      dms_id))
        n_sub = int(round(SUBSAMPLE_FRAC * n))
        subs = [np.sort(rng.choice(n, size=n_sub, replace=False))
                for _ in range(int(THRESH['cluster_ari_n_seeds']))]
        for meth in linkages:
            per_rho = {rho: [] for rho in rho_targets}
            for s, rows in enumerate(subs):
                Zs, tws = ward_tree(X, method=meth, rows=rows)
                Xs = X[rows]
                ms = mrow[rows]
                es = eligible[rows]
                for rho in rho_targets:
                    hs, _j, _d = target_height(Zs, Xs, ms, es, float(rho),
                                               min_size, rows.size)
                    ls = _labels_at(Zs, hs)
                    lf = cut[(meth, rho)]['labels'][rows]
                    per_rho[rho].append(float(adjusted_rand_score(lf, ls)))
                del Zs, Xs
                if verbose and meth == linkages[0]:
                    print('    subsample %d/%d (%d rows) %.1fs  ARI %s'
                          % (s + 1, len(subs), rows.size, tws,
                             ' '.join('%.3f' % per_rho[r][-1]
                                      for r in rho_targets)))
            ss[meth] = per_rho
        del trees, Zp
        gc.collect()

        # ---- observed statistic + the N2 permutation null -------------------- #
        obs = {}
        for (meth, rho), cc in cut.items():
            t, _den = loo_robust_z(e, cc['labels'], eligible, sigma_noise,
                                   min_size)
            fin = np.isfinite(t)
            eta, n_eta, k_eta = eta2_residual(e, cc['labels'], eligible,
                                              cc['ecnt'] >= min_size)
            obs[(meth, rho)] = dict(
                t=t, fin=fin, n_cov=int(fin.sum()), eta2=eta,
                rate={tau: (float((np.abs(t[fin]) >= tau).mean())
                            if fin.any() else float('nan')) for tau in taus},
                n_cliff={tau: int((np.abs(t[fin]) >= tau).sum())
                         for tau in taus})

        keys = sorted(cut, key=lambda k: (k[0], k[1]))

        def _stat_fn(_ctx, rep):
            """N2 -> the SAME leave-one-out statistic on the permuted residual.

            N2 keeps ``beta, phi, g, sigma`` fixed and exchanges ``e`` within
            (mutation order x phi-decile x censoring) strata, so it destroys the
            assignment of a residual to a GENOTYPE -- hence to a cluster -- while
            preserving the residual marginal exactly.  That is precisely the null
            this channel needs, and it needs no refit, so the tree is built once
            and only the values move.
            """
            en = rep['e_oof']
            out = {}
            for k in keys:
                cc = cut[k]
                tt, _d = loo_robust_z(en, cc['labels'], eligible, sigma_noise,
                                      min_size)
                ff = np.isfinite(tt)
                nm = '%s|%g' % k
                for tau in taus:
                    out['rate_%s_tau%g' % (nm, tau)] = (
                        float((np.abs(tt[ff]) >= tau).mean())
                        if ff.any() else float('nan'))
                out['eta2_' + nm] = eta2_residual(
                    en, cc['labels'], eligible, cc['ecnt'] >= min_size)[0]
            return out

        t0 = time.time()
        nul = _nulls.run_ensemble(dms_id, 'N2', B=B, stat_fn=_stat_fn, nproc=1,
                                  use_cache=False, write=False, verify=False,
                                  verbose=False)
        if verbose:
            print('    N2 x %d (custom stat_fn, non-cacheable) %.1fs'
                  % (B, time.time() - t0))

    finally:
        mon.__exit__(None, None, None)
    peak_gb = mon.peak

    # ---- assemble rows ----------------------------------------------------- #
    rows = []
    for meth, rho in keys:
        cc = cut[(meth, rho)]
        ob = obs[(meth, rho)]
        nm = '%s|%g' % (meth, rho)
        cnt, radius, rms, mp = cluster_geometry(cc['labels'], X, mrow,
                                                K=cc['K'])
        seqr = seq_hamming_radius(cc['labels'], codes, K=cc['K'])
        kp = cc['ecnt'] >= min_size
        w = cc['ecnt'][kp]
        wsum = w.sum() if kp.any() else float('nan')

        def _wm(v):
            if not kp.any():
                return float('nan')
            vv = v[kp]
            ok = np.isfinite(vv)
            return (float((vv[ok] * w[ok]).sum() / w[ok].sum())
                    if ok.any() else float('nan'))

        # order range: max - min mutation order among ELIGIBLE members.  By
        # lexsort, not ``np.maximum.at``: min/max-by-group is the one reduction
        # ``np.bincount`` cannot do, and ufunc.at is the slow path the spec's
        # numeric hygiene rule exists to avoid.
        orng = _group_range(cc['labels'][eligible],
                            ctx.n_muts[eligible].astype(np.float64), cc['K'])

        rate_null = {tau: nul['rate_%s_tau%g' % (nm, tau)].to_numpy()
                     for tau in taus}
        eta_null = nul['eta2_' + nm].to_numpy()
        r_obs = ob['rate'][TAU]
        rn = rate_null[TAU]
        rn_ok = rn[np.isfinite(rn)]
        r_null_mean = float(rn_ok.mean()) if rn_ok.size else float('nan')
        T_N2 = (r_obs / r_null_mean) if (np.isfinite(r_obs)
                                         and r_null_mean > 0) else float('nan')
        p_perm = ((1.0 + float((rn_ok >= r_obs).sum())) / (rn_ok.size + 1.0)
                  if (rn_ok.size and np.isfinite(r_obs)) else float('nan'))
        e_ok = eta_null[np.isfinite(eta_null)]
        p_eta = ((1.0 + float((e_ok >= ob['eta2']).sum())) / (e_ok.size + 1.0)
                 if (e_ok.size and np.isfinite(ob['eta2'])) else float('nan'))

        # ---- Jaccard against the pair channel, on the SHARED universe ------- #
        clu_cov = ob['fin']
        clu_flag = clu_cov & (np.abs(np.nan_to_num(ob['t'], nan=0.0)) >= TAU)
        uni = clu_cov & pair['covered']
        jac = _jaccard(clu_flag & uni, pair['flag'] & uni)
        jac_v = _jaccard(clu_flag & uni, pair['flag_v'] & uni)
        uni_all = clu_cov | pair['covered']
        jac_all = _jaccard(clu_flag & uni_all, pair['flag'] & uni_all)

        gate = bool(cc['n_ge'] >= THRESH['cluster_min_n_clusters']
                    and cc['cov'] >= THRESH['cluster_min_frac_covered'])
        sweep = dict(
            tau=[float(t) for t in taus],
            n_cliff=[ob['n_cliff'][t] for t in taus],
            rate_obs=[ob['rate'][t] for t in taus],
            rate_N2_mean=[float(np.nanmean(rate_null[t]))
                          if np.isfinite(rate_null[t]).any() else None
                          for t in taus],
            T_N2=[(ob['rate'][t] / float(np.nanmean(rate_null[t]))
                   if (np.isfinite(rate_null[t]).any()
                       and float(np.nanmean(rate_null[t])) > 0) else None)
                  for t in taus],
            p_perm_N2=[((1.0 + float((rate_null[t][np.isfinite(rate_null[t])]
                                      >= ob['rate'][t]).sum()))
                        / (np.isfinite(rate_null[t]).sum() + 1.0)
                        if np.isfinite(rate_null[t]).any() else None)
                       for t in taus],
            sigma_mult=_sigma_mult_sweep(e, cc, eligible, sigma_noise,
                                         min_size, taus),
            sigma_floor_only=_floor_only(e, cc, eligible, ctx.sigma_floor,
                                         min_size, taus),
        )
        note = []
        if cc['degen']:
            note.append('DEGENERATE END (%s): %s'
                        % (cc['degen'], _DEGENERATE_TEXT[cc['degen']]))
        if not gate:
            note.append('COVERAGE GATE FAILED (%d clusters >= %d of the %d '
                        'required, %.3f of variants covered of the %.2f '
                        'required) -- spec Sec.3: the assay is declared '
                        'STRUCTURALLY INCAPABLE of testing this hypothesis and '
                        'THAT IS THE REPORTED FINDING.'
                        % (cc['n_ge'], min_size,
                           THRESH['cluster_min_n_clusters'], cc['cov'],
                           THRESH['cluster_min_frac_covered']))
        if unident:
            note.append('ORCHESTRATOR D3: STRUCTURALLY_UNIDENTIFIED -- '
                        'only %.2f%% of rows have a finite phi^oof and '
                        'resid_mad_oof/resid_mad_in = 3.26x, so its '
                        'standardised residuals are not identified.  Numbers '
                        'reported, contributes in NEITHER direction.'
                        % (100.0 * n_elig / n))
        if not powerless:
            note.append('Pair channel is NOT powerless here (n_nested = %d >= '
                        '%d), so this channel may not add the assay to the C2 '
                        'count and may not override the pair verdict.'
                        % (n_nested, THRESH['min_nested_for_pair_channel']))
        rows.append({
            'DMS_id': dms_id, 'linkage': meth, 'rho_target': float(rho),
            'K': cc['K'], 'n_clusters_ge8': cc['n_ge'],
            'frac_variants_covered': round(cc['cov'], 6),
            'coverage_gate_pass': gate,
            'mean_within_radius': _round(cc['radius'], 6),
            'order_range_within_cluster': _round(_wm(orng), 4),
            'eta2_residual': _round(ob['eta2'], 6),
            's_rho': _round(cc['h'], 6),
            'ari_ward_vs_average': _round(cc.get('ari_wa'), 6),
            'ari_subsample_5seed': _round(float(np.mean(ss[meth][rho])), 6),
            'n_cliff': ob['n_cliff'][TAU], 'cliff_rate': _round(r_obs, 8),
            'T_N2': _round(T_N2, 4),
            'jaccard_vs_pair_channel_rho1': (_round(jac, 6) if rho == 1
                                             else ''),
            'peak_RAM_GB': round(peak_gb, 3), 'wall_s': '',
            'adds_assay_to_C2_count': '',
            # ---- x_ block ---------------------------------------------------
            'x_n': n, 'x_n_eligible': n_elig, 'x_n_covered': ob['n_cov'],
            'x_frac_eligible_covered': _round(ob['n_cov'] / max(n_elig, 1), 6),
            'x_tau': TAU, 'x_sigma_noise_a': _round(sigma_noise, 8),
            'x_sigma_noise_provenance': sigma_prov,
            'x_mean_within_rms': _round(_wm(rms), 6),
            'x_mean_within_pairwise_hamming': _round(_wm(mp), 6),
            'x_mean_within_radius_seq': _round(_wm(seqr), 6),
            'x_radius_min_attainable': _round(r_min, 6),
            'x_radius_max_attainable': _round(r_max, 6),
            'x_degenerate_end': cc['degen'],
            'x_eta2_N2_mean': _round(float(np.nanmean(eta_null))
                                     if e_ok.size else float('nan'), 6),
            'x_eta2_p_N2': _round(p_eta, 6),
            'x_cliff_rate_N2_mean': _round(r_null_mean, 8),
            'x_p_perm_N2': _round(p_perm, 6), 'x_q_BH': '',
            'x_B_null': int(B),
            'x_ari_subsample_sd': _round(float(np.std(ss[meth][rho])), 6),
            'x_ari_subsample_frac': SUBSAMPLE_FRAC,
            'x_n_flag_cluster': int((clu_flag & uni).sum()),
            'x_n_flag_pair': int((pair['flag'] & uni).sum()),
            'x_n_universe_jaccard': int(uni.sum()),
            'x_n_flag_both': int((clu_flag & pair['flag'] & uni).sum()),
            'x_jaccard_v_endpoint_only': _round(jac_v, 6),
            'x_jaccard_unrestricted': _round(jac_all, 6),
            'x_n_Pa': pair['n_Pa'], 'x_n_nested': n_nested,
            'x_pair_channel_powerless': powerless,
            'x_structurally_unidentified': unident,
            'x_order_mixed_no_order_claim': True,
            'x_sweep_json': json.dumps(sweep, separators=(',', ':'),
                                       default=_jsafe),
            'x_note': '  '.join(note),
        })
    wall = time.time() - t_assay
    for r in rows:
        r['wall_s'] = round(wall, 2)
    if verbose:
        print('    %s done  %.1fs  peak %.2f GB' % (dms_id, wall, peak_gb))
    _nulls.clear_context_cache()
    gc.collect()
    return rows


def _sigma_mult_sweep(e, cc, eligible, sigma_noise, min_size, taus):
    """``sigma x {0.5, 1, 2}`` on the FLOOR (spec Sec.1.0 requires the surface
    to accompany every headline number)."""
    out = {}
    for m in config.SIGMA_MULTIPLIERS:
        t, _d = loo_robust_z(e, cc['labels'], eligible, m * sigma_noise,
                             min_size)
        f = np.isfinite(t)
        out['%g' % m] = dict(
            n_covered=int(f.sum()),
            rate=[float((np.abs(t[f]) >= tau).mean()) if f.any() else None
                  for tau in taus])
    return out


def _floor_only(e, cc, eligible, sigma_floor, min_size, taus):
    """The other reading of ``sigma_noise_a``: floor only at the assay's decimal
    grid (``quantum/sqrt(12)``), i.e. let each cluster's OWN dispersion set the
    scale.  On these six assays the quantum is 1e-16, so this is effectively no
    floor at all -- which is exactly why it is a sensitivity and not the
    primary."""
    t, _d = loo_robust_z(e, cc['labels'], eligible, float(sigma_floor),
                         min_size)
    f = np.isfinite(t)
    return dict(sigma_floor=float(sigma_floor), n_covered=int(f.sum()),
                rate=[float((np.abs(t[f]) >= tau).mean()) if f.any() else None
                      for tau in taus])


def _round(v, k):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return ''
    return '' if not np.isfinite(v) else round(v, k)


def _jsafe(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# --------------------------------------------------------------------------- #
# the stage                                                                   #
# --------------------------------------------------------------------------- #

def write_T15(df, *, write=True):
    p = os.path.join(PATHS.artifacts, 'T15_cluster_channel.csv')
    if write:
        PATHS.ensure_cache_dirs()
        df.to_csv(p, index=False)
    return p


def run_all(assays=None, *, B=None, linkages=('ward', 'average'), write=True,
            verbose=True, nproc=None):
    """Spec Sec.5 stage 5: the six eligible assays, SERIAL.

    ``nproc`` is accepted and IGNORED on purpose: the peak here is one 3.3 GB
    condensed array plus scipy's copy of it (measured 6.8 GB on 5A12_VEGF), and
    the spec's own scheduling rule is that stage 5 never shares the box with
    stage 3/4.  Running two of these concurrently would double a figure the spec
    already under-states, for no wall-clock gain worth having.
    """
    import pandas as pd
    config.assert_env()
    ids = list(ELIGIBLE if assays is None else assays)
    bad = [a for a in ids if not config.ASSAYS[a].eligible_cluster_channel]
    if bad:
        raise ValueError('not eligible for the cluster channel (spec Sec.3 '
                         'n <= %d gate): %r' % (THRESH['cluster_n_max'], bad))
    if nproc not in (None, 1) and verbose:
        print('[clusters] nproc=%r ignored: stage 5 is SERIAL (peak is one '
              'condensed array + scipy nn_chain\'s copy of it)' % (nproc,))
    # ascending n so a failure on the big one still leaves five rows
    ids.sort(key=lambda a: int(config.ASSAYS[a].n_spec or 0))
    rows = []
    t0 = time.time()
    for i, a in enumerate(ids):
        if verbose:
            print('[clusters] %d/%d %s' % (i + 1, len(ids), a))
        rows.extend(cluster_channel(a, B=B, linkages=linkages,
                                    verbose=verbose))
    df = pd.DataFrame(rows)
    # ---- BH-FDR over the ASSAYS (never over edges), per (linkage, rho) ------ #
    for (meth, rho), g in df.groupby(['linkage', 'rho_target']):
        p = np.array([float(v) if v != '' else np.nan
                      for v in g['x_p_perm_N2']], dtype=np.float64)
        q = _bh_fdr(p)
        df.loc[g.index, 'x_q_BH'] = [_round(v, 6) for v in q]
    # ---- adds_assay_to_C2_count ------------------------------------------- #
    # spec Sec.3: "may only ADD an assay to the C2 count when the pair channel
    # is powerless (4D5), never override it".  Only the rho = 1 ward row can
    # ever claim it, and D3's STRUCTURALLY_UNIDENTIFIED assay never can.
    adds = []
    for _i, r in df.iterrows():
        ok = (r['linkage'] == 'ward' and float(r['rho_target']) == 1.0
              and bool(r['coverage_gate_pass'])
              and bool(r['x_pair_channel_powerless'])
              and not bool(r['x_structurally_unidentified'])
              and _f(r['T_N2']) >= THRESH['C2_T_sup']
              and _f(r['x_q_BH']) < THRESH['C2_q_BH_sup'])
        adds.append(bool(ok))
    df['adds_assay_to_C2_count'] = adds
    df = df[ALL_COLUMNS]
    p = write_T15(df, write=write)
    if verbose:
        print('[clusters] %d rows -> %s   total %.1fs' % (len(df), p,
                                                          time.time() - t0))
    return df


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float('nan')
    return x


def stage5(assays=None, nproc=None, verbose=True):
    """The name ``run_all.py``'s stage table looks for first."""
    df = run_all(assays=assays, nproc=nproc, verbose=verbose)
    bad = _pairs.verify_manifest()
    if bad:
        raise RuntimeError('MANIFEST verification failed after stage 5: %r'
                           % (bad,))
    if verbose:
        print('[clusters] MANIFEST verified clean (this stage writes no cache '
              'of its own: the tree is a deterministic function of X and the 5 '
              'subsamples are seeded from SEEDS["cluster_subsample"]=%d)'
              % SEEDS['cluster_subsample'])
    return df


# --------------------------------------------------------------------------- #
# self-check -- RUNS on real data and prints the numbers                      #
# --------------------------------------------------------------------------- #

def _selfcheck(argv=()):
    import pandas as pd
    config.assert_env()
    ok = True

    # ---- 1. LOO median/MAD against brute force -------------------------- #
    rng = np.random.default_rng(12345)
    worst = 0.0
    for m in list(range(2, 40)) + [64, 101]:
        for trial in range(6):
            v = rng.normal(size=m)
            if trial == 3:
                v = np.round(v, 1)                 # heavy ties
            if trial == 4:
                v = np.zeros(m)                    # MAD == 0 exactly
            if trial == 5:
                v = np.repeat(rng.normal(size=max(m // 3, 1)),
                              int(np.ceil(m / max(m // 3, 1))))[:m]
            med, mad = _loo_med_and_mad(v)
            for i in range(m):
                w = np.delete(v, i)
                bm = np.median(w)
                bd = np.median(np.abs(w - bm))
                worst = max(worst, abs(med[i] - bm), abs(mad[i] - bd))
    print('[selfcheck] LOO med/MAD vs brute force, m=2..101 incl. ties and '
          'MAD==0: max abs diff = %.3g' % worst)
    ok &= worst < 1e-12

    # ---- 2. blocked Gram against pdist ---------------------------------- #
    from scipy.spatial.distance import pdist
    a = 'Z-domain_ZpA963_HL1_fitness_2M5A'
    ctx = _nulls.get_context(a, verify=True)
    c1 = condensed_hamming(ctx.X)
    c2 = pdist(np.asarray(ctx.X.toarray(), dtype=np.float64))
    print('[selfcheck] blocked BLAS Gram vs pdist on %s (n=%d): max abs diff '
          '= %.3g ; squared distances integral: %s'
          % (a, ctx.n, np.abs(c1 - c2).max(),
             bool(np.array_equal(c1 ** 2, np.round(c1 ** 2)))))
    ok &= np.abs(c1 - c2).max() == 0.0

    # ---- 3. geometry closed form against an explicit pairwise mean ------- #
    Z = sch.linkage(c1.copy(), method='ward')
    lab = _labels_at(Z, 4.0)
    mrow = np.asarray(ctx.X.sum(axis=1)).ravel().astype(np.float64)
    cnt, radius, rms, mp = cluster_geometry(lab, ctx.X.tocsr(), mrow)
    from scipy.spatial.distance import squareform
    D2 = squareform(c1) ** 2
    err = 0.0
    for c in np.flatnonzero(cnt >= 2)[:25]:
        ix = np.flatnonzero(lab == c)
        sub = D2[np.ix_(ix, ix)]
        brute = sub[np.triu_indices(ix.size, 1)].mean()
        err = max(err, abs(brute - mp[c]))
    print('[selfcheck] mean pairwise Hamming closed form vs brute force '
          '(25 clusters): max abs diff = %.3g' % err)
    ok &= err < 1e-9
    print('[selfcheck] radius identity  mean_pair == 2*radius*n/(n-1): %s'
          % bool(np.allclose(mp[cnt >= 2],
                             2 * radius[cnt >= 2] * cnt[cnt >= 2]
                             / (cnt[cnt >= 2] - 1))))

    # ---- 4. y-blindness -------------------------------------------------- #
    Z2 = ward_tree(ctx.X.tocsr())[0]
    print('[selfcheck] tree is y-blind (same X, same tree, y never touched): '
          '%s' % bool(np.array_equal(Z, Z2)))
    ok &= bool(np.array_equal(Z, Z2))

    # ---- 5. the n_max gate refuses GB1_1FCC ------------------------------ #
    try:
        cluster_channel('GB1_IgG-Fc_fitness_1FCC', verbose=False)
        print('[selfcheck] n_max gate: FAILED TO REFUSE GB1_IgG-Fc_1FCC')
        ok = False
    except ValueError as exc:
        print('[selfcheck] n_max gate refuses GB1_IgG-Fc_1FCC: %s' % exc)

    # ---- 6. the whole channel on the two smallest assays ----------------- #
    df = pd.DataFrame(cluster_channel('4D5_HER2_fitness_1N8Z', B=20,
                                      verbose=True))
    cols = ['linkage', 'rho_target', 'K', 'n_clusters_ge8',
            'frac_variants_covered', 'coverage_gate_pass',
            'mean_within_radius', 'eta2_residual', 'n_cliff', 'cliff_rate',
            'T_N2', 'ari_ward_vs_average', 'ari_subsample_5seed']
    print(df[cols].to_string(index=False))
    print('[selfcheck] %s' % ('ALL OK' if ok else 'FAILURES ABOVE'))
    return 0 if ok else 1


def _main(argv):
    if argv and argv[0] in ('--selfcheck', 'selfcheck'):
        return _selfcheck(argv[1:])
    verbose = '--quiet' not in argv
    assays = None
    if '--assays' in argv:
        assays = argv[argv.index('--assays') + 1].split(',')
    B = None
    if '--B' in argv:
        B = int(argv[argv.index('--B') + 1])
    df = run_all(assays=assays, B=B, verbose=verbose)
    print(df[['DMS_id', 'linkage', 'rho_target', 'K', 'n_clusters_ge8',
              'frac_variants_covered', 'coverage_gate_pass',
              'mean_within_radius', 'eta2_residual', 'n_cliff', 'cliff_rate',
              'T_N2', 'jaccard_vs_pair_channel_rho1',
              'adds_assay_to_C2_count']].to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(_main(sys.argv[1:]))
