"""BGYM-CLIFF v1 -- C1: is the landscape smooth in mutation degree? (spec Sec.1.2)

Statistics implemented here, with the spec Sec.3 signatures verbatim:

* :func:`gini_mean_difference` -- closed form on sorted ``y``, O(n log n).
* :func:`variogram_exact`      -- ``V(h), G(h)`` from a cached pair index array,
  reduced with ``np.bincount(H, weights=d2)``.  **Never** ``np.add.at``.
* :func:`variogram_sampled`    -- every ``h`` in ONE pass over the ONE seeded
  2e7 random-pair sample per assay.
* :func:`gamma_background`     -- ``gamma(1)``, ``gamma(m)``, CI by 2,000
  bootstraps over SITE PAIRS (never over variants).
* :func:`roughness_to_slope`   -- Szendro 2013 ``r/s`` on the additive LS fit.

THE h AXIS -- decided once, in writing, as the foundation demanded
-----------------------------------------------------------------
Spec Sec.1.0 types a nested pair as ``|K_u Delta K_v| = 1`` and a same-site swap
as ``= 2``, while Sec.1.2's sampler computes "Hamming from the P-length int8 code
vector by block XOR-nonzero-count", which is the number of differing
``(chain, pos)`` SLOTS and equals 1 for BOTH classes.  The two metrics disagree,
so ``V(1)`` was ambiguous as written.

**Resolved empirically, not by taste.**  Spec Sec.1.2 pre-declares ``SI =
G(1)/GMD`` for ten assays.  Measured here (see :func:`_selfcheck`), the
CODE-VECTOR metric -- ``h`` = number of differing ``(chain, pos)`` slots, so that
``h = 1`` is ``nested UNION same-site`` -- reproduces all ten to three decimals
(GB1_1FCC 0.2677 vs 0.268; Z-ZSPA1-LL1 1.3977 vs 1.398; ...), while the
nested-only reading misses every one of them by 0.05-0.28.  The primary axis is
therefore the code-vector metric, and it is the same metric as the cached
``hamming`` array, so no pair is ever re-enumerated.

``h = 1`` is provably ``nested (+) same-site``, disjointly: one differing slot
with a zero on one side is a nested pair, with both sides non-zero a same-site
swap.  Both channels are ALSO reported on their own T05 rows (``h='1_nested'``,
``h='1_samesite'``), so the same-site channel remains a separate reference
(T04 ``samesite_SI_reference``) and nothing is silently pooled.

Everything numeric comes from :mod:`cliff.config`; seeds only from
``config.SEEDS`` via ``config.assay_seed``.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from cliff import config
from cliff import pairs as _pairs
from cliff.config import PATHS, SEEDS, THRESH
from cliff.io_bgym import load_assay

# --------------------------------------------------------------------------- #
# module constants that are DEFINITIONS, not decision boundaries              #
# --------------------------------------------------------------------------- #

#: performance knob for the blocked ``np.bincount`` reduction (rows per block).
_REDUCE_BLOCK = 4_000_000

#: the "90" in the T04 column name ``V_range_h90``: the variogram RANGE is the
#: lag at which V(h) first reaches this fraction of the sill V(inf).  This is
#: part of the column's definition, not a verdict threshold.
_RANGE_SILL_FRAC = 0.9

#: percentile pair for every 95% CI in this module.
_CI_Q = (2.5, 97.5)

#: bootstrap replicates are chunked so that ``B x K`` gathers stay small.
_BOOT_CHUNK = 200

T04_COLUMNS = [
    'DMS_id', 'SI', 'SI_lo95', 'SI_hi95', 'V1_over_Vinf', 'V_monotone_h1_h4',
    'V_range_h90', 'gamma1', 'gamma1_lo95', 'gamma1_hi95', 'gamma_decay_json',
    'r_rough', 's_slope', 'rs', 'rs_N1_mean', 'rs_N3_mean', 'pos_rs',
    'R2_add_raw', 'R2_add_latent', 'link_R2_gain', 'SI_N1_p975', 'SI_N3_p025',
    'samesite_SI_reference', 'verdict_C1', 'failing_criterion',
]

T05_COLUMNS = [
    'DMS_id', 'h', 'N_h', 'exact_or_sampled', 'V_h', 'V_h_se', 'G_h', 'G_h_se',
    'V_h_over_Vinf', 'G_h_over_GMD', 'V_h_N1_mean', 'V_h_N1_lo', 'V_h_N1_hi',
    'V_h_N2_mean',
]

#: T04/T05 columns this module deliberately leaves EMPTY -- they are null-ensemble
#: quantities and belong to ``cliff/nulls.py`` (N1 needs the latent fit; N2 needs
#: the cross-fitted residuals).  ``pos_rs`` needs BOTH N1 and N3 and so waits too.
NULL_COLUMNS_DEFERRED = ('rs_N1_mean', 'rs_N3_mean', 'pos_rs', 'SI_N1_p975',
                         'SI_N3_p025', 'V_h_N1_mean', 'V_h_N1_lo', 'V_h_N1_hi',
                         'V_h_N2_mean')

#: T04 columns that need ``cliff/latent.py``.
LATENT_COLUMNS_DEFERRED = ('R2_add_latent', 'link_R2_gain')


# --------------------------------------------------------------------------- #
# closed forms                                                                #
# --------------------------------------------------------------------------- #

def gini_mean_difference(y):
    """``GMD = [2/(n(n-1))] Sum_k (2k-n-1) y_(k)`` -- spec Sec.1.2/Sec.3, closed
    form on sorted ``y``, O(n log n).  Equals the mean ``|y_u - y_v|`` over all
    ``C(n,2)`` pairs; no random-pair enumeration anywhere."""
    y = np.sort(np.asarray(y, dtype=np.float64))
    n = y.size
    if n < 2:
        return float('nan')
    i = np.arange(1, n + 1, dtype=np.float64)
    return float(2.0 * ((2.0 * i - n - 1.0) * y).sum() / (n * (n - 1.0)))


def v_infinity(y):
    """``V(inf) = Var(y) * n/(n-1)`` -- spec Sec.1.2, closed form.

    Identically the mean of ``(y_u-y_v)^2 / 2`` over all ``C(n,2)`` pairs, i.e.
    the ddof=1 sample variance (asserted in :func:`_selfcheck`)."""
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n < 2:
        return float('nan')
    return float(y.var() * n / (n - 1.0))


# --------------------------------------------------------------------------- #
# pair reductions -- np.bincount, never np.add.at                             #
# --------------------------------------------------------------------------- #

def _moments(y, idx, H, hmax, block=None):
    """Blocked ``np.bincount`` reduction of the pair differences.

    Returns ``(N, S1, S2, S4)``, each length ``hmax+1``, where ``S1 = Sum|d|``,
    ``S2 = Sum d^2``, ``S4 = Sum d^4`` per lag.  ``H`` is either an int scalar
    (all pairs share the lag, the exact-channel case) or a per-pair int array.
    """
    L = int(hmax) + 1
    N = np.zeros(L, dtype=np.int64)
    S1 = np.zeros(L, dtype=np.float64)
    S2 = np.zeros(L, dtype=np.float64)
    S4 = np.zeros(L, dtype=np.float64)
    idx = np.asarray(idx)
    m = int(idx.shape[0])
    if m == 0:
        return N, S1, S2, S4
    y = np.asarray(y, dtype=np.float64)
    scalar_h = np.isscalar(H) or (getattr(H, 'ndim', 1) == 0)
    blk = int(block or _REDUCE_BLOCK)
    for s in range(0, m, blk):
        e = min(s + blk, m)
        d = y[idx[s:e, 0]] - y[idx[s:e, 1]]
        ad = np.abs(d)
        d2 = d * d
        if scalar_h:
            hb = np.full(e - s, int(H), dtype=np.intp)
        else:
            hb = np.asarray(H[s:e], dtype=np.intp)
        N += np.bincount(hb, minlength=L)[:L]
        S1 += np.bincount(hb, weights=ad, minlength=L)[:L]
        S2 += np.bincount(hb, weights=d2, minlength=L)[:L]
        S4 += np.bincount(hb, weights=d2 * d2, minlength=L)[:L]
    return N, S1, S2, S4


def _stats(N, S1, S2, S4, h):
    """``(N_h, V_h, G_h, V_h_se, G_h_se)`` from the per-lag moments.

    The SEs are i.i.d.-pair standard errors of the mean.  For a SAMPLED lag that
    is the honest Monte-Carlo SE (pairs are drawn independently).  For an EXACT
    lag it is only a nominal scale: the pairs share variants, so it understates
    the true uncertainty and must never be read as a CI -- which is why the C1
    decision uses the site-pair / position cluster bootstraps instead.
    """
    n = int(N[h])
    if n == 0:
        return 0, float('nan'), float('nan'), float('nan'), float('nan')
    m1 = S1[h] / n
    m2 = S2[h] / n
    m4 = S4[h] / n
    var_g = max(m2 - m1 * m1, 0.0)
    var_v = max(m4 - m2 * m2, 0.0) / 4.0
    return (n, float(0.5 * m2), float(m1),
            float(np.sqrt(var_v / n)), float(np.sqrt(var_g / n)))


def variogram_exact(y, idx, h):
    """``(N_h, V_h, G_h)`` for one cached pair class, all of whose pairs sit at
    lag ``h`` -- spec Sec.1.2 ("exact from the cached bucketing pair index
    arrays, reduced with ``np.bincount(H, weights=d2)``")."""
    h = int(h)
    N, S1, S2, S4 = _moments(y, idx, h, h)
    n, V, G, _, _ = _stats(N, S1, S2, S4, h)
    return n, V, G


def variogram_sampled(y, codes, samp_idx, *, hamming=None, block=None):
    """Every lag in ONE pass over the seeded random-pair sample.

    ``hamming`` may be passed to reuse the ``hamming`` array already cached in
    ``randpairs/{DMS_id}_2e7_seed*.npz`` (identical to
    ``pairs.hamming_from_codes(codes, samp_idx)``, asserted in
    :func:`_selfcheck`); otherwise it is recomputed by block XOR-nonzero-count.

    Returns a DataFrame indexed by ``h`` with ``N_h, V_h, G_h, V_h_se, G_h_se``.
    """
    samp_idx = np.asarray(samp_idx)
    if hamming is None:
        H = _pairs.hamming_from_codes(codes, samp_idx, block=THRESH['hamming_block'])
    else:
        H = np.asarray(hamming)
    if H.size != samp_idx.shape[0]:
        raise ValueError('hamming/idx length mismatch: %d vs %d'
                         % (H.size, samp_idx.shape[0]))
    hmax = int(H.max()) if H.size else 0
    N, S1, S2, S4 = _moments(y, samp_idx, H, hmax, block=block)
    if N[0] != 0:
        raise AssertionError('lag 0 in the random-pair sample: duplicate genotypes '
                             '(G1 says there are none)')
    rows = []
    for h in range(1, hmax + 1):
        if N[h] == 0:
            continue
        n, V, G, Vse, Gse = _stats(N, S1, S2, S4, h)
        rows.append(dict(h=h, N_h=n, V_h=V, G_h=G, V_h_se=Vse, G_h_se=Gse))
    return pd.DataFrame(rows, columns=['h', 'N_h', 'V_h', 'G_h', 'V_h_se', 'G_h_se'])


# --------------------------------------------------------------------------- #
# cache readers (never re-enumerate pairs)                                    #
# --------------------------------------------------------------------------- #

def load_nested(dms_id):
    """``pairs/{DMS_id}_nested.npz`` -- idx, add_col, sibling_count, masks."""
    p = os.path.join(PATHS.pairs, dms_id + '_nested.npz')
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def load_samesite(dms_id):
    """``pairs/{DMS_id}_samesite.npz`` -- idx, pos_col."""
    p = os.path.join(PATHS.pairs, dms_id + '_samesite.npz')
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def randpairs_path(dms_id, n_draw=None, seed_name='randpairs'):
    if n_draw is None:
        n_draw = THRESH['randpair_n_draw']
    return os.path.join(PATHS.randpairs, '%s_%s_seed%d.npz'
                        % (dms_id, _pairs._sci(n_draw), SEEDS[seed_name]))


def load_randpairs(assay, *, create_if_missing=True):
    """The ONE seeded random-pair sample for this assay.

    Stage 0 materialised it for the 14 PRIMARY+ARM assays only; the three
    CONTROL assays are plotted on the same V(h) axes (spec F1) and the five
    pre-declared C1 refutations are EXCLUDED-tier, so the remaining 14 samples
    are created here through the same seeded, md5'd API and appended to
    ``MANIFEST.json`` (never rewritten -- see :func:`manifest_extend`).
    """
    p = randpairs_path(assay.dms_id)
    created = None
    if not os.path.exists(p):
        if not create_if_missing:
            return None
        created = _pairs.cache_randpairs(assay)
    with np.load(p) as z:
        out = dict(idx=z['idx'], hamming=z['hamming'], exact=bool(z['exact']),
                   n=int(z['n']), n_draw=int(z['n_draw']), seed=z['seed'].tolist())
    out['path'] = p
    out['created'] = created
    return out


#: the only cache subtrees C1 reads.  ``verify_inputs`` refuses to run on an md5
#: mismatch HERE; a mismatch anywhere else (another module's cache, mid-run) is
#: reported and left to its owner -- spec Sec.5 says the md5 is verified
#: "before use", and C1 uses exactly these three.
INPUT_PREFIXES = ('data/cliff_cache/keys/', 'data/cliff_cache/pairs/',
                  'data/cliff_cache/randpairs/')


def verify_inputs(*, verbose=True):
    """Verify the md5 of every cache file C1 reads and REFUSE to run on a
    mismatch (spec Sec.5)."""
    bad = _pairs.verify_manifest()
    mine = [b for b in bad if b[0].startswith(INPUT_PREFIXES)]
    if mine:
        raise RuntimeError(
            'MANIFEST.json md5 mismatch on %d C1 input file(s); refusing to run '
            'downstream of a changed cache: %r' % (len(mine), mine[:5]))
    if bad and verbose:
        print('[C1] NOTE: %d manifest md5 mismatch(es) outside the C1 inputs '
              '(%s) -- not consumed here, left to their owning module: %s'
              % (len(bad), ', '.join(INPUT_PREFIXES),
                 ', '.join(sorted(set(os.path.dirname(b[0]) for b in bad)))))
    return bad


def manifest_extend(entries, block_name, block):
    """Read-modify-write ``MANIFEST.json``: add md5 entries for artefacts created
    here WITHOUT touching anything stage 0 wrote.

    ``pairs.write_manifest`` rebuilds the file from the entry list it is handed,
    so calling it here would drop stage 0's 98 entries.  Every sampled artefact
    must still be md5'd into the manifest (the repo's own convention), hence this
    merge."""
    with open(PATHS.manifest) as fh:
        man = json.load(fh)
    for e in entries:
        man['files'][e['path']] = {'md5': e['md5'], 'bytes': e['bytes']}
    blocks = man.get(block_name, {})
    blocks.update(block)
    man[block_name] = blocks
    tmp = PATHS.manifest + '.vgtmp'
    with open(tmp, 'w') as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
    os.replace(tmp, PATHS.manifest)
    return len(man['files'])


# --------------------------------------------------------------------------- #
# design matrix + Szendro r/s                                                 #
# --------------------------------------------------------------------------- #

def design_matrix(assay):
    """``X in {0,1}^{n x M}``, ``M`` = distinct observed ``(chain, seq_pos,
    aa_mut)``, row nnz = ``num_muts`` (spec Sec.1.0).

    The foundation caches ``codes`` (position -> aa) but no design matrix, so it
    is built here; ``latent.py`` may import this helper rather than duplicating
    the column order, which is ``assay.col_index``'s and therefore sorted."""
    import scipy.sparse as sp
    nm = assay.n_muts.astype(np.int64)
    total = int(nm.sum())
    rows = np.repeat(np.arange(assay.n, dtype=np.int32), nm)
    cols = np.empty(total, dtype=np.int32)
    ci = assay.col_index
    p = 0
    for k in assay.keys:
        for t in k:
            cols[p] = ci[t]
            p += 1
    if p != total:
        raise AssertionError('design matrix nnz mismatch %d != %d' % (p, total))
    return sp.csr_matrix((np.ones(total, dtype=np.float64), (rows, cols)),
                         shape=(assay.n, assay.M))


def roughness_to_slope(X, y, *, atol=1e-12, btol=1e-12, iter_lim=None):
    """Szendro 2013 roughness-to-slope ratio of the additive landscape.

    Fit ``y ~ c0 + X c`` by least squares (``scipy.sparse.linalg.lsqr``, which
    converges to the MINIMUM-NORM LS solution, so ``s`` is well defined even when
    ``X`` is rank deficient -- the Z-domain libraries mutate every position in
    every variant, making each position's substitution columns sum to the
    intercept).  Then

    * ``r`` = RMS residual  (roughness),
    * ``s`` = ``mean |c_j|`` over the ``M`` substitution coefficients (slope),
    * ``rs = r/s``, ``R2_add = 1 - SSres/SStot``.

    ``X`` is the ``{0,1}`` substitution indicator of spec Sec.1.0.  Under a
    ``{-1,+1}`` coding every ``c_j`` halves and ``r/s`` doubles, so the coding is
    part of the definition and is stated here.

    ``degenerate=True`` marks a saturated design (one column per variant, as in
    the five ``max_mut == 1`` assays): the additive fit is then exact, ``r == 0``
    and ``r/s == 0`` carries no information about smoothness.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import lsqr
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    X = sp.csr_matrix(X)
    A = sp.hstack([np.ones((n, 1)), X], format='csr')
    out = lsqr(A, y, atol=atol, btol=btol,
               iter_lim=int(iter_lim or 4 * A.shape[1] + 100))
    c, istop, itn = out[0], int(out[1]), int(out[2])
    resid = y - A @ c
    coef = c[1:]
    ss_res = float((resid * resid).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r = float(np.sqrt(ss_res / n))
    s = float(np.abs(coef).mean()) if coef.size else float('nan')
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    return dict(r=r, s=s, rs=(r / s if s > 0 else float('nan')), R2_add=r2,
                M=int(coef.size), n_iter=itn, istop=istop,
                intercept=float(c[0]), beta=coef,
                s_median=float(np.median(np.abs(coef))) if coef.size else float('nan'),
                degenerate=bool(coef.size >= n - 1 or r == 0.0))


def rs_null_N3(X, y, *, B=None, seed_name='nulls_N3', dms_id=None, **kw):
    """N3 (House-of-Cards) calibration of ``r/s``: free permutation of ``y``.

    Handed to ``cliff/nulls.py`` ready-made -- N3 is the one null in the spec that
    needs no latent fit -- but the T04 ``rs_N3_mean`` / ``pos_rs`` cells are left
    EMPTY by this module, because ``pos_rs`` also needs N1's mean and N1 belongs
    to ``nulls.py``.  Costs one ``lsqr`` per replicate (0.25-0.8 s on the larger
    assays), so ``B`` defaults to ``THRESH['null_B']`` but is worth overriding for
    a smoke test."""
    if B is None:
        B = THRESH['null_B']
    rng = np.random.default_rng(config.assay_seed(seed_name, dms_id) if dms_id
                                else SEEDS[seed_name])
    y = np.asarray(y, dtype=np.float64)
    vals = []
    for _ in range(int(B)):
        vals.append(roughness_to_slope(X, rng.permutation(y), **kw)['rs'])
    v = np.asarray(vals, dtype=np.float64)
    return dict(B=int(B), rs_mean=float(np.nanmean(v)), rs_sd=float(np.nanstd(v)),
                rs_p025=float(np.nanpercentile(v, _CI_Q[0])),
                rs_p975=float(np.nanpercentile(v, _CI_Q[1])), rs=v)


def si_null_N3(y, idx_lag1, *, B=None, seed_name='nulls_N3', dms_id=None):
    """N3 (House-of-Cards) calibration of ``SI``: free permutation of ``y``.

    ``GMD`` is a function of the multiset of ``y`` and so is permutation
    invariant, and the mean ``|y_u - y_v|`` over a uniformly random pair IS
    ``GMD``, so ``E[SI_N3] = 1`` exactly.  That identity is what makes the spec's
    own sentence about Z-ZSPA1-LL2 ("SI 1.001 -- neighbours as different as
    random pairs") literally true, and it fixes the scale on which every other
    ``SI`` is read.  Measured here for the record and handed to ``cliff/nulls.py``;
    T04's ``SI_N3_p025`` cell is left EMPTY by this module.
    """
    if B is None:
        B = THRESH['null_B']
    y = np.asarray(y, dtype=np.float64)
    gmd = gini_mean_difference(y)
    rng = np.random.default_rng(config.assay_seed(seed_name, dms_id) if dms_id
                                else SEEDS[seed_name])
    idx_lag1 = np.asarray(idx_lag1)
    out = np.empty(int(B), dtype=np.float64)
    for b in range(int(B)):
        yp = rng.permutation(y)
        out[b] = np.abs(yp[idx_lag1[:, 0]] - yp[idx_lag1[:, 1]]).mean() / gmd
    return dict(B=int(B), SI_mean=float(out.mean()), SI_sd=float(out.std()),
                SI_p025=float(np.percentile(out, _CI_Q[0])),
                SI_p975=float(np.percentile(out, _CI_Q[1])), SI=out)


# --------------------------------------------------------------------------- #
# cluster bootstraps (sufficient statistics, exact and cheap)                  #
# --------------------------------------------------------------------------- #

def _pearson(nn, sx, sy, sxx, syy, sxy):
    num = nn * sxy - sx * sy
    den = np.sqrt(np.maximum(nn * sxx - sx * sx, 0.0)
                  * np.maximum(nn * syy - sy * sy, 0.0))
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return out


def pearson_cluster_bootstrap(x, y, cluster, *, B, rng):
    """Pearson ``r`` with a CLUSTER bootstrap CI.

    Pearson is a function of six additive sufficient statistics, so resampling
    clusters is exactly ``S[draw].sum(axis=0)`` -- identical to gathering every
    observation of every drawn cluster, at ``O(B*K)`` instead of ``O(B*n)``.
    That is what makes 2,000 bootstraps over GB1_1FCC's 183,690 observations
    instantaneous.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0:
        return dict(r=float('nan'), lo=float('nan'), hi=float('nan'),
                    n_obs=0, n_clusters=0, B=0)
    uc, inv = np.unique(np.asarray(cluster), return_inverse=True)
    K = int(uc.size)
    n_c = np.bincount(inv, minlength=K).astype(np.float64)
    sx = np.bincount(inv, weights=x, minlength=K)
    sy = np.bincount(inv, weights=y, minlength=K)
    sxx = np.bincount(inv, weights=x * x, minlength=K)
    syy = np.bincount(inv, weights=y * y, minlength=K)
    sxy = np.bincount(inv, weights=x * y, minlength=K)
    r_obs = float(_pearson(n_c.sum(), sx.sum(), sy.sum(), sxx.sum(), syy.sum(),
                           sxy.sum()))
    lo = hi = float('nan')
    if B and K >= 2:
        reps = []
        done = 0
        while done < B:
            b = min(_BOOT_CHUNK, B - done)
            sel = rng.integers(0, K, size=(b, K))
            reps.append(_pearson(n_c[sel].sum(1), sx[sel].sum(1), sy[sel].sum(1),
                                 sxx[sel].sum(1), syy[sel].sum(1), sxy[sel].sum(1)))
            done += b
        rb = np.concatenate(reps)
        if np.isfinite(rb).any():
            lo, hi = [float(v) for v in np.nanpercentile(rb, list(_CI_Q))]
    return dict(r=r_obs, lo=lo, hi=hi, n_obs=int(x.size), n_clusters=K, B=int(B))


def si_cluster_bootstrap(absd, cluster, gmd, *, B, rng):
    """CI for ``SI = G(1)/GMD`` by resampling the MUTATED POSITION of each lag-1
    pair (its one differing ``(chain,pos)`` slot).

    The spec gives ``SI`` two CI columns in T04 but does not name an estimator.
    Resampling variants would break the pair graph; resampling pairs would ignore
    the dominant dependence (every pair that adds or swaps the same position
    shares that position's effect).  So this is the same block bootstrap the spec
    prescribes for the C2 sweep, applied to ``G(1)``: resample the position set,
    take all lag-1 pairs whose differing position is in the resample.  ``GMD`` is
    a marginal statistic of ``y`` and is held fixed.
    """
    absd = np.asarray(absd, dtype=np.float64)
    if absd.size == 0 or not np.isfinite(gmd) or gmd == 0:
        return float('nan'), float('nan')
    uc, inv = np.unique(np.asarray(cluster), return_inverse=True)
    K = int(uc.size)
    n_c = np.bincount(inv, minlength=K).astype(np.float64)
    s1 = np.bincount(inv, weights=absd, minlength=K)
    if K < 2:
        return float('nan'), float('nan')
    reps = []
    done = 0
    while done < B:
        b = min(_BOOT_CHUNK, B - done)
        sel = rng.integers(0, K, size=(b, K))
        reps.append(s1[sel].sum(1) / n_c[sel].sum(1) / gmd)
        done += b
    v = np.concatenate(reps)
    return tuple(float(q) for q in np.nanpercentile(v, list(_CI_Q)))


# --------------------------------------------------------------------------- #
# gamma -- background-dependence of a single mutation's effect                 #
# --------------------------------------------------------------------------- #

def _col_to_pos(assay):
    """X-column -> code-vector column (i.e. ``(chain, seq_pos)`` slot)."""
    out = np.full(assay.M, -1, dtype=np.int32)
    for (ch, pos, aa), c in assay.col_index.items():
        out[c] = assay.pos_index[(ch, pos)]
    if (out < 0).any():
        raise AssertionError('col_index/pos_index disagree')
    return out


def _single_slot(assay):
    """Per-row code-vector slot of the mutated position, for singles only
    (``-1`` elsewhere)."""
    out = np.full(assay.n, -1, dtype=np.int32)
    rows = np.where(assay.n_muts == 1)[0]
    if rows.size:
        out[rows] = (assay.codes[rows] != 0).argmax(axis=1).astype(np.int32)
    return out


def beta_from_singles(assay):
    """``beta_hat_i`` for every X column, as the OBSERVED single-mutant effect
    ``y_i - y_WT`` (NaN where that single is not observed).

    Spec Sec.1.2 writes ``gamma(1) = Pearson(Delta_i(empty), Delta_i({j}))`` with
    ``Delta_i(empty) = y_i - y_WT``, i.e. it uses exactly this estimator for the
    ``m = 1`` case, and ``gamma(m) = Pearson(Delta_i(B), beta_hat_i)`` for deeper
    backgrounds.  Using the same estimator for both makes the gamma(m) decay one
    consistent curve whose ``m = 0`` value is identically 1 (asserted below), and
    it needs no latent fit.  ``cliff/latent.py``'s cross-fitted ``beta_hat`` can
    be passed in later through ``gamma_background(..., beta=...)``.
    """
    beta = np.full(assay.M, np.nan, dtype=np.float64)
    if assay.wt_row < 0:
        return beta, 'unavailable_no_wt_row'
    ywt = float(assay.y[assay.wt_row])
    for r in np.where(assay.n_muts == 1)[0]:
        beta[assay.col_index[assay.keys[r][0]]] = assay.y[r] - ywt
    return beta, 'observed_single_effect_y_minus_yWT'


def gamma_background(assay, *, beta=None, B=None, seed_name='bootstrap_gamma',
                     nested=None, max_m=None):
    """``gamma(1)``, ``gamma(m)`` and the site-pair bootstrap CI (spec Sec.1.2).

    ``gamma(1)`` = Pearson over all ``(i, {j})`` with singles ``i``, ``j`` and the
    double ``{i,j}`` all observed, between ``Delta_i(empty) = y_i - y_WT`` and
    ``Delta_i({j}) = y_ij - y_j``.  Those observations are exactly the cached
    NESTED edges whose background is a single, so no pair is re-enumerated:
    ``Delta_i(B) = y[idx[:,1]] - y[idx[:,0]]`` and ``i = add_col``.

    ``gamma(m)`` = Pearson(``Delta_i(B)``, ``beta_hat_i``) over backgrounds of
    size ``m``.  ``gamma(0)`` is identically 1 by construction and is asserted, as
    a check that ``beta`` and ``add_col`` line up.

    CI: 2,000 bootstraps over SITE PAIRS ``(pos_i, pos_j)`` for ``m = 1``
    (the spec's own unit).  For ``m >= 2`` a background is a set, not a site, so
    the cluster is the ADDED position ``pos_i`` -- stated, not silent.
    """
    if B is None:
        B = THRESH['C1_bootstrap_B']
    if nested is None:
        nested = load_nested(assay.dms_id)
    idx = nested['idx']
    add_col = nested['add_col']
    if beta is None:
        beta, beta_source = beta_from_singles(assay)
    else:
        beta = np.asarray(beta, dtype=np.float64)
        beta_source = 'supplied'
    rng = np.random.default_rng(config.assay_seed(seed_name, assay.dms_id))

    out = dict(beta_source=beta_source, n_nested=int(idx.shape[0]),
               gamma=dict(), gamma1=float('nan'), gamma1_lo95=float('nan'),
               gamma1_hi95=float('nan'), gamma1_n_obs=0, gamma1_n_sitepairs=0,
               gamma0_check=float('nan'), note='')
    if idx.shape[0] == 0:
        out['note'] = 'no nested edges'
        return out
    if beta_source.startswith('unavailable'):
        out['note'] = ('gamma undefined: no WT row, so Delta_i(empty) and hence '
                       'beta_hat_i cannot be formed from singles')
        return out

    col2pos = _col_to_pos(assay)
    pos_add = col2pos[add_col]
    slot = _single_slot(assay)
    m_bg = assay.n_muts[idx[:, 0]].astype(np.int64)
    dy = assay.y[idx[:, 1]] - assay.y[idx[:, 0]]
    b = beta[add_col]
    ok = np.isfinite(b)

    ms = sorted(set(int(v) for v in np.unique(m_bg[ok])))
    if max_m is not None:
        ms = [m for m in ms if m <= max_m]
    for m in ms:
        sel = ok & (m_bg == m)
        if not sel.any():
            continue
        if m == 1:
            pa = pos_add[sel]
            pb = slot[idx[sel, 0]]
            lo_p = np.minimum(pa, pb).astype(np.int64)
            hi_p = np.maximum(pa, pb).astype(np.int64)
            clus = lo_p * (assay.P + 1) + hi_p
            unit = 'site_pair'
        else:
            clus = pos_add[sel]
            unit = 'added_position'
        res = pearson_cluster_bootstrap(b[sel], dy[sel], clus,
                                        B=(B if m == 1 else 0), rng=rng)
        out['gamma'][m] = dict(gamma=res['r'], n_obs=res['n_obs'],
                               n_clusters=res['n_clusters'], unit=unit,
                               lo95=res['lo'], hi95=res['hi'])
        if m == 0:
            out['gamma0_check'] = res['r']
        if m == 1:
            out['gamma1'] = res['r']
            out['gamma1_lo95'] = res['lo']
            out['gamma1_hi95'] = res['hi']
            out['gamma1_n_obs'] = res['n_obs']
            out['gamma1_n_sitepairs'] = res['n_clusters']
    if 1 not in out['gamma']:
        if int(assay.n_muts.max()) <= 1:
            out['note'] = ('gamma(1) undefined: max_mut == 1, so no nested edge '
                           'has a mutated background')
        elif not ok.any():
            out['note'] = ('gamma(1) undefined: no added substitution has an '
                           'observed single, so beta_hat is empty (no singles)')
        else:
            out['note'] = ('gamma(1) undefined: no nested edge has a '
                           'single-mutant background with an observed single '
                           'for the added substitution (doubles gap)')
    elif not np.isfinite(out['gamma1']):
        sel = ok & (m_bg == 1)
        out['note'] = ('gamma(1) not identifiable on %d observations: '
                       'sd(beta_hat) = %.3g, sd(Delta) = %.3g -- Pearson needs '
                       'both to vary (all singles equal, e.g. sitting on a '
                       'censoring floor)'
                       % (int(sel.sum()), float(np.std(b[sel])),
                          float(np.std(dy[sel]))))
    return out


# --------------------------------------------------------------------------- #
# per-assay C1 driver                                                         #
# --------------------------------------------------------------------------- #

def _range_h90(hs, vs, vinf, frac=_RANGE_SILL_FRAC):
    """Variogram RANGE: the lag at which V(h) first reaches ``frac * V(inf)``,
    linearly interpolated between the bracketing integer lags."""
    if not np.isfinite(vinf) or vinf <= 0 or not len(hs):
        return float('nan')
    tgt = frac * vinf
    for k in range(len(hs)):
        if np.isfinite(vs[k]) and vs[k] >= tgt:
            if k == 0:
                return float(hs[0])
            v0, v1 = vs[k - 1], vs[k]
            if not np.isfinite(v0) or v1 == v0:
                return float(hs[k])
            return float(hs[k - 1] + (tgt - v0) / (v1 - v0) * (hs[k] - hs[k - 1]))
    return float('nan')


def _split_lag1_sampled(codes, idx, H):
    """Boolean 'is nested' for the sampled lag-1 pairs.

    A lag-1 pair differs at exactly one ``(chain,pos)`` slot; it is NESTED when
    one side is WT there (code 0) and a SAME-SITE swap when both sides carry a
    substitution.  Only the lag-1 rows are gathered, so this costs nothing."""
    sel = np.where(H == 1)[0]
    if sel.size == 0:
        return sel, np.zeros(0, dtype=bool)
    ii = idx[sel]
    ca = codes[ii[:, 0]]
    cb = codes[ii[:, 1]]
    dif = ca != cb
    if not (dif.sum(axis=1) == 1).all():
        raise AssertionError('a lag-1 pair differs at != 1 slot')
    slot = dif.argmax(axis=1)
    ar = np.arange(sel.size)
    a = ca[ar, slot]
    bb = cb[ar, slot]
    return sel, ((a == 0) | (bb == 0))


def c1_assay(assay, *, B_boot=None, verbose=True, create_randpairs=True):
    """Every observed C1 statistic for one assay: the T05 rows + the T04 row."""
    t0 = time.time()
    if B_boot is None:
        B_boot = THRESH['C1_bootstrap_B']
    y = assay.y
    n = assay.n
    gmd = gini_mean_difference(y)
    vinf = v_infinity(y)

    nz = load_nested(assay.dms_id)
    sz = load_samesite(assay.dms_id)
    n_idx, s_idx = nz['idx'], sz['idx']
    col2pos = _col_to_pos(assay)

    # ---- exact lag 1: the two channels, and their (disjoint) union ---------- #
    Nn, S1n, S2n, S4n = _moments(y, n_idx, 1, 1)
    Ns, S1s, S2s, S4s = _moments(y, s_idx, 1, 1)
    chan = {}
    chan['1_nested'] = _stats(Nn, S1n, S2n, S4n, 1)
    chan['1_samesite'] = _stats(Ns, S1s, S2s, S4s, 1)
    chan['1'] = _stats(Nn + Ns, S1n + S1s, S2n + S2s, S4n + S4s, 1)

    # ---- SI and its position-cluster bootstrap ------------------------------ #
    si = chan['1'][2] / gmd if np.isfinite(gmd) and gmd != 0 else float('nan')
    si_nested = chan['1_nested'][2] / gmd if np.isfinite(gmd) and gmd else float('nan')
    si_ss = chan['1_samesite'][2] / gmd if np.isfinite(gmd) and gmd else float('nan')
    # SI's CI is a POSITION block bootstrap, so it draws on SEEDS['bootstrap_block']
    # (the spec's own name for that resampling unit), not on the gamma stream.
    rng = np.random.default_rng(config.assay_seed('bootstrap_block', assay.dms_id))
    absd = np.concatenate([
        np.abs(y[n_idx[:, 0]] - y[n_idx[:, 1]]) if n_idx.shape[0] else np.zeros(0),
        np.abs(y[s_idx[:, 0]] - y[s_idx[:, 1]]) if s_idx.shape[0] else np.zeros(0)])
    clus = np.concatenate([
        col2pos[nz['add_col']] if n_idx.shape[0] else np.zeros(0, dtype=np.int32),
        sz['pos_col'] if s_idx.shape[0] else np.zeros(0, dtype=np.int32)])
    si_lo, si_hi = si_cluster_bootstrap(absd, clus, gmd, B=B_boot, rng=rng)
    del absd, clus

    # ---- sampled lags ------------------------------------------------------- #
    rp = load_randpairs(assay, create_if_missing=create_randpairs)
    samp = None
    xcheck = dict(DMS_id=assay.dms_id)
    kind = ''
    rp_created = (rp or {}).get('created')
    if rp is not None:
        H = rp['hamming']
        kind = 'exact_enum' if rp['exact'] else 'sampled'
        samp = variogram_sampled(y, assay.codes, rp['idx'], hamming=H)
        # cross-check the sample against the exact channels, per channel
        sel1, is_nested = _split_lag1_sampled(assay.codes, rp['idx'], H)
        samp_lag1 = {}
        for name, mask in (('1', None), ('1_nested', is_nested),
                           ('1_samesite', ~is_nested)):
            ss = sel1 if mask is None else sel1[mask]
            Nx, S1x, S2x, S4x = _moments(y, rp['idx'][ss], 1, 1)
            st = _stats(Nx, S1x, S2x, S4x, 1)
            samp_lag1[name] = st
            ex = chan[name]
            tag = {'1': 'all', '1_nested': 'nest', '1_samesite': 'ss'}[name]
            xcheck['N_exact_' + tag] = ex[0]
            xcheck['N_samp_' + tag] = st[0]
            xcheck['V_exact_' + tag] = ex[1]
            xcheck['V_samp_' + tag] = st[1]
            xcheck['G_exact_' + tag] = ex[2]
            xcheck['G_samp_' + tag] = st[2]
            xcheck['zV_' + tag] = (((st[1] - ex[1]) / st[3])
                                   if (np.isfinite(st[3]) and st[3] > 0)
                                   else float('nan'))
            xcheck['zG_' + tag] = (((st[2] - ex[2]) / st[4])
                                   if (np.isfinite(st[4]) and st[4] > 0)
                                   else float('nan'))
        xcheck['exact_enum'] = bool(rp['exact'])
        del sel1, is_nested
        del rp['idx'], rp['hamming']

    # ---- T05 rows ----------------------------------------------------------- #
    t05 = []

    def _row(hlabel, src, st):
        N_h, V, G, Vse, Gse = st
        t05.append(dict(
            DMS_id=assay.dms_id, h=hlabel, N_h=N_h, exact_or_sampled=src,
            V_h=V, V_h_se=Vse, G_h=G, G_h_se=Gse,
            V_h_over_Vinf=(V / vinf if np.isfinite(vinf) and vinf else float('nan')),
            G_h_over_GMD=(G / gmd if np.isfinite(gmd) and gmd else float('nan')),
            V_h_N1_mean='', V_h_N1_lo='', V_h_N1_hi='', V_h_N2_mean=''))

    _row('1', 'exact', chan['1'])
    _row('1_nested', 'exact', chan['1_nested'])
    _row('1_samesite', 'exact', chan['1_samesite'])
    v_by_h, vse_by_h, hs = {}, {}, []
    if samp is not None:
        for name in ('1', '1_nested', '1_samesite'):
            _row(name + '_sampled', kind, samp_lag1[name])
        for rr in samp.itertuples():
            if int(rr.h) == 1:
                continue
            _row(int(rr.h), kind,
                 (int(rr.N_h), rr.V_h, rr.G_h, rr.V_h_se, rr.G_h_se))
            v_by_h[int(rr.h)] = rr.V_h
            vse_by_h[int(rr.h)] = rr.V_h_se
        hs = sorted(v_by_h)
    # V(1) on the axis is the EXACT merged value, never the sampled one
    v_by_h[1] = chan['1'][1]
    vse_by_h[1] = chan['1'][3]
    hs = [1] + hs
    t05.append(dict(DMS_id=assay.dms_id, h='random', N_h=n * (n - 1) // 2,
                    exact_or_sampled='closed_form', V_h=vinf, V_h_se='',
                    G_h=gmd, G_h_se='', V_h_over_Vinf=1.0, G_h_over_GMD=1.0,
                    V_h_N1_mean='', V_h_N1_lo='', V_h_N1_hi='', V_h_N2_mean=''))

    # ---- monotonicity over h = 1..4, and the range ------------------------- #
    upto = int(THRESH['C1_h_monotone_upto'])
    hseq = [h for h in hs if h <= upto]
    seq = [v_by_h[h] for h in hseq]
    mono = ''
    mono_break = None
    if len(seq) >= 2:
        mono = bool(np.all(np.diff(np.asarray(seq, dtype=np.float64)) >= 0))
        if not mono:
            for k in range(1, len(seq)):
                if seq[k] < seq[k - 1]:
                    se = float(np.sqrt(vse_by_h.get(hseq[k], np.nan) ** 2
                                       + vse_by_h.get(hseq[k - 1], np.nan) ** 2))
                    drop = seq[k] - seq[k - 1]
                    mono_break = dict(
                        h=hseq[k], drop=drop, se=se,
                        n_se=(abs(drop) / se if se > 0 else float('inf')),
                        source=('exact_enum' if kind == 'exact_enum' else 'sampled'),
                        rel_from=(seq[k - 1] / vinf if vinf else float('nan')),
                        rel_to=(seq[k] / vinf if vinf else float('nan')))
                    break
    v1_over_vinf = (v_by_h[1] / vinf) if (np.isfinite(vinf) and vinf) else float('nan')
    v_range = _range_h90(hs, [v_by_h[h] for h in hs], vinf)

    # ---- gamma -------------------------------------------------------------- #
    gam = gamma_background(assay, B=B_boot, nested=nz)
    if np.isfinite(gam.get('gamma0_check', np.nan)):
        if abs(gam['gamma0_check'] - 1.0) > 1e-9:
            raise AssertionError('gamma(0) != 1 (%r): beta/add_col misaligned'
                                 % gam['gamma0_check'])

    # ---- r/s ---------------------------------------------------------------- #
    X = design_matrix(assay)
    rs = roughness_to_slope(X, y)

    row = dict(
        DMS_id=assay.dms_id,
        SI=si, SI_lo95=si_lo, SI_hi95=si_hi,
        V1_over_Vinf=v1_over_vinf,
        V_monotone_h1_h4=mono, V_range_h90=v_range,
        gamma1=gam['gamma1'], gamma1_lo95=gam['gamma1_lo95'],
        gamma1_hi95=gam['gamma1_hi95'],
        gamma_decay_json=json.dumps({str(k): (None if not np.isfinite(v['gamma'])
                                              else round(float(v['gamma']), 6))
                                     for k, v in sorted(gam['gamma'].items())}),
        r_rough=rs['r'], s_slope=rs['s'], rs=rs['rs'],
        rs_N1_mean='', rs_N3_mean='', pos_rs='',
        R2_add_raw=rs['R2_add'], R2_add_latent='', link_R2_gain='',
        SI_N1_p975='', SI_N3_p025='',
        samesite_SI_reference=si_ss, verdict_C1='',
        failing_criterion='')
    row['failing_criterion'] = c1_failing_criterion(row, gam, rs, mono, v_by_h,
                                                     mono_break)

    extra = dict(SI_nested_only=si_nested, GMD=gmd, V_inf=vinf,
                 N1_merged=chan['1'][0], N1_nested=chan['1_nested'][0],
                 N1_samesite=chan['1_samesite'][0],
                 h_max=(max(hs) if hs else 0), n=n, M=assay.M, P=assay.P,
                 rs_degenerate=rs['degenerate'], rs_istop=rs['istop'],
                 rs_n_iter=rs['n_iter'],
                 gamma1_n_obs=gam['gamma1_n_obs'],
                 gamma1_n_sitepairs=gam['gamma1_n_sitepairs'],
                 beta_source=gam['beta_source'], gamma_note=gam['note'],
                 randpairs=kind, randpairs_created=rp_created,
                 wall_s=round(time.time() - t0, 2))
    if verbose:
        print('[C1] %-44s SI=%6.4f [%6.4f,%6.4f] SI_nest=%6.4f SI_ss=%6.4f '
              'V1/Vinf=%7.4f mono=%-5s h90=%5s g1=%7s r/s=%6.4f R2=%6.4f %5.1fs'
              % (assay.dms_id, si, si_lo, si_hi, si_nested, si_ss, v1_over_vinf,
                 mono, ('%.2f' % v_range) if np.isfinite(v_range) else 'nan',
                 ('%.4f' % gam['gamma1']) if np.isfinite(gam['gamma1']) else 'n/a',
                 rs['rs'], rs['R2_add'], extra['wall_s']), flush=True)
    return dict(t04=row, t05=t05, xcheck=xcheck, extra=extra, gamma=gam)


def c1_failing_criterion(row, gam, rs, mono, v_by_h, mono_break=None):
    """Which spec Sec.1.2 clauses fail / fire, as a diagnosis string.

    NOT a verdict -- ``verdict_C1`` is ``cliff/verdict.py``'s column and is left
    empty here.  Thresholds are read from ``config.THRESH`` only."""
    T = THRESH
    si, v1 = row['SI'], row['V1_over_Vinf']
    g1, g1lo, g1hi = row['gamma1'], row['gamma1_lo95'], row['gamma1_hi95']
    fire, supfail, uneval = [], [], []
    # REFUTED clauses
    if np.isfinite(si) and si >= T['C1_SI_ref']:
        fire.append('SI>=%.2f' % T['C1_SI_ref'])
    if np.isfinite(v1) and v1 >= T['C1_V1_over_Vinf_ref']:
        fire.append('V1/Vinf>=%.2f' % T['C1_V1_over_Vinf_ref'])
    # the REFUTED clause is literally V(1) > V(2); the SUPPORTED clause is the
    # stronger "non-decreasing over h = 1..4".  They are NOT the same test and
    # are kept apart (BH3_Bcl-xL has V(1) < V(2) but V(3) < V(2)).
    v1a, v2a = v_by_h.get(1, float('nan')), v_by_h.get(2, float('nan'))
    if np.isfinite(v1a) and np.isfinite(v2a) and v1a > v2a:
        fire.append('V(1)>V(2)')
    if np.isfinite(g1) and np.isfinite(g1hi) and g1 <= T['C1_gamma1_ref'] \
            and g1hi < T['C1_gamma1_ci_hi_ref']:
        fire.append('gamma1<=%.2f with CI_hi<%.2f'
                    % (T['C1_gamma1_ref'], T['C1_gamma1_ci_hi_ref']))
    # SUPPORTED clauses
    if not (np.isfinite(si) and si <= T['C1_SI_sup']):
        supfail.append('SI<=%.2f' % T['C1_SI_sup'])
    if not (np.isfinite(v1) and v1 <= T['C1_V1_over_Vinf_sup']):
        supfail.append('V1/Vinf<=%.2f' % T['C1_V1_over_Vinf_sup'])
    if mono is not True:
        tag = ''
        if mono_break is not None:
            # a SAMPLED V(h) can dip on Monte-Carlo noise, so the dip is given in
            # SE units; when the lag came from the full pair enumeration the dip
            # is exact.  V/Vinf on both sides shows whether the "violation" is a
            # real early plateau or a 1-3% overshoot at the sill -- verdict.py
            # needs that distinction, the literal clause does not make it.
            tag = (' [breaks at h=%d: %+.4g (%s%s), V/Vinf %.3f -> %.3f]'
                   % (mono_break['h'], mono_break['drop'],
                      mono_break['source'],
                      ('' if mono_break['source'] == 'exact_enum'
                       else ', %.1f sampling SE' % mono_break['n_se']),
                      mono_break['rel_from'], mono_break['rel_to']))
        supfail.append('V non-decreasing h1..h%d%s'
                       % (T['C1_h_monotone_upto'], tag))
    if not np.isfinite(v2a):
        uneval.append('V(1)>V(2) (no lag-2 estimate)')
    if not (np.isfinite(g1) and g1 >= T['C1_gamma1_sup']
            and np.isfinite(g1lo) and g1lo > T['C1_gamma1_ci_lo_sup']):
        supfail.append('gamma1>=%.2f & CI_lo>%.2f'
                       % (T['C1_gamma1_sup'], T['C1_gamma1_ci_lo_sup']))
    # unevaluable
    uneval.append('pos_rs>=%.2f (needs N1+N3: nulls.py)' % T['C1_pos_rs_ref'])
    if mono == '':
        uneval.append('V monotonicity (h axis has < 2 lags)')
    if not np.isfinite(g1):
        uneval.append('gamma1 (%s)' % (gam['note'] or 'undefined'))
    if rs['degenerate']:
        uneval.append('r/s (saturated additive design: R2==1, r==0)')
    parts = []
    if fire:
        parts.append('REFUTED_ON[' + '; '.join(fire) + ']')
    if supfail:
        parts.append('SUP_FAIL[' + '; '.join(supfail) + ']')
    else:
        parts.append('SUP_ALL_MEASURABLE_MET')
    if uneval:
        parts.append('UNEVAL[' + '; '.join(uneval) + ']')
    return ' | '.join(parts)


# --------------------------------------------------------------------------- #
# whole-benchmark driver                                                      #
# --------------------------------------------------------------------------- #

def _verify_cached_keys(assay):
    """The cached ``keys/{DMS_id}.npz`` must reproduce what ``load_assay`` just
    parsed -- otherwise the pair index arrays refer to different rows."""
    p = os.path.join(PATHS.keys, assay.dms_id + '.npz')
    with np.load(p, allow_pickle=False) as z:
        if not np.array_equal(z['codes'], assay.codes):
            raise AssertionError('%s: cached codes != parsed codes' % assay.dms_id)
        if not np.array_equal(z['row_index'], assay.row_index):
            raise AssertionError('%s: cached row_index != parsed' % assay.dms_id)
        yc = z['y']
        if not np.allclose(yc, assay.y, rtol=0, atol=0, equal_nan=True):
            raise AssertionError('%s: cached y != parsed y' % assay.dms_id)


def predeclared_report(t04):
    """Spec Sec.1.2's pre-declared ``SI`` values against the measurement.

    Five are the pre-declared REFUTATIONS ("must reproduce, else the
    implementation is wrong"); GB1_1FCC 0.268 and 5A12_VEGF 0.250 are the two
    predictions; the remaining three (4D5, BH3 x2) come from the same Sec.2
    profile table and are free extra checks.  All ten live in
    ``config.ASSAYS[...].SI_spec`` / ``config.EXPECTED``."""
    t04 = t04.set_index('DMS_id')
    ref = config.EXPECTED['C1_predeclared_refutations']
    rows = []
    for d in config.ALL_ASSAYS:
        want = config.ASSAYS[d].SI_spec
        if not np.isfinite(want) or d not in t04.index:
            continue
        got = float(t04.loc[d, 'SI'])
        got_n = float(t04.loc[d, 'SI_nested_only']) if 'SI_nested_only' in t04 \
            else float('nan')
        rows.append(dict(DMS_id=d, tier=config.ASSAYS[d].tier,
                         role=('pre-declared REFUTATION' if d in ref
                               else 'pre-declared prediction'),
                         SI_spec=want, SI_measured=got, abs_diff=abs(got - want),
                         reproduces=bool(abs(got - want) < 5e-4),
                         SI_nested_only=got_n))
    return pd.DataFrame(rows)


def c1_all(assays=None, *, B_boot=None, write=True, verbose=True,
           create_randpairs=True):
    """C1 for every assay: writes T05_variogram.csv and the observed columns of
    T04_smoothness_C1.csv."""
    import resource
    config.assert_env()
    PATHS.ensure_cache_dirs()
    verify_inputs(verbose=verbose)
    ids = list(config.ALL_ASSAYS) if assays is None else list(assays)
    t_start = time.time()
    if verbose:
        print('#' * 110)
        print('# C1 -- smoothness in mutation degree (spec Sec.1.2).  '
              'h = differing (chain,pos) slots; h=1 == nested (+) same-site.')
        print('#' * 110, flush=True)
    t04_rows, t05_rows, xchecks, extras, new_entries, new_block = [], [], [], {}, [], {}
    for k, d in enumerate(ids):
        a = load_assay(d)
        _verify_cached_keys(a)
        res = c1_assay(a, B_boot=B_boot, verbose=verbose,
                       create_randpairs=create_randpairs)
        row = dict(res['t04'])
        row['SI_nested_only'] = res['extra']['SI_nested_only']
        t04_rows.append(row)
        t05_rows.extend(res['t05'])
        xchecks.append(res['xcheck'])
        extras[d] = res['extra']
        cr = res['extra'].pop('randpairs_created', None)
        if cr:
            new_entries.extend(cr['manifest'])
            new_block[d] = dict(exact=cr['exact'], n_drawn=cr['n_drawn'],
                                n_possible_pairs=cr['n_possible_pairs'],
                                hamming_hist=cr['hamming_hist'],
                                created_by='cliff.variogram')
        del a, res

    t04 = pd.DataFrame(t04_rows)
    t05 = pd.DataFrame(t05_rows)[T05_COLUMNS]
    xc = pd.DataFrame(xchecks)

    if new_entries:
        nf = manifest_extend(new_entries, 'randpairs_variogram', new_block)
        if verbose:
            print('\n[C1] MANIFEST.json extended with %d randpair artefact(s); '
                  '%d files now fingerprinted' % (len(new_entries), nf))
        bad2 = [b for b in _pairs.verify_manifest()
                if b[0].startswith(INPUT_PREFIXES)]
        if bad2:
            raise RuntimeError('manifest_extend broke the manifest: %r' % bad2[:5])

    if write:
        p05 = os.path.join(PATHS.artifacts, 'T05_variogram.csv')
        t05.to_csv(p05, index=False)
        p04 = os.path.join(PATHS.artifacts, 'T04_smoothness_C1.csv')
        t04[T04_COLUMNS].to_csv(p04, index=False)
        if verbose:
            print('[C1] wrote %s (%d x %d)' % (p05, len(t05), len(t05.columns)))
            print('[C1] wrote %s (%d x %d)' % (p04, len(t04), len(T04_COLUMNS)))
            print('[C1] T04 columns left EMPTY for their owning module: %s'
                  % ', '.join(c for c in NULL_COLUMNS_DEFERRED + LATENT_COLUMNS_DEFERRED
                              if c in T04_COLUMNS))
    if verbose:
        print('[C1] wall %.1fs  peak RSS %.2f GB'
              % (time.time() - t_start,
                 resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))
    return dict(T04=t04, T05=t05, xcheck=xc, extra=extras,
                predeclared=predeclared_report(t04))


# --------------------------------------------------------------------------- #
# self-check -- runs on the real data and prints what the spec predicts        #
# --------------------------------------------------------------------------- #

def _selfcheck():
    import scipy.sparse as sp
    config.assert_env()
    print('=' * 110)
    print('cliff.variogram self-check -- closed forms, then C1 on all %d assays'
          % len(config.ALL_ASSAYS))
    print('=' * 110)

    # ---- closed forms against brute force --------------------------------- #
    rng = np.random.default_rng(SEEDS['base'])
    yy = rng.normal(size=400) * 3.0 + 1.0
    bf_gmd = float(np.abs(yy[:, None] - yy[None, :]).sum() / (400 * 399))
    bf_vinf = float(((yy[:, None] - yy[None, :]) ** 2).sum() / (2 * 400 * 399))
    print('[closed form] GMD    closed %.12f  brute %.12f  |d|=%.2e'
          % (gini_mean_difference(yy), bf_gmd, abs(gini_mean_difference(yy) - bf_gmd)))
    print('[closed form] V(inf) closed %.12f  brute %.12f  |d|=%.2e'
          % (v_infinity(yy), bf_vinf, abs(v_infinity(yy) - bf_vinf)))
    assert abs(gini_mean_difference(yy) - bf_gmd) < 1e-10
    assert abs(v_infinity(yy) - bf_vinf) < 1e-10

    # ---- variogram_exact vs a naive loop ---------------------------------- #
    idx = _pairs.all_pairs_exact(60).astype(np.int32)
    y60 = yy[:60]
    N, V, G = variogram_exact(y60, idx, 7)
    d = y60[idx[:, 0]] - y60[idx[:, 1]]
    print('[exact]   N=%d  V=%.12f (naive %.12f)  G=%.12f (naive %.12f)'
          % (N, V, 0.5 * (d ** 2).mean(), G, np.abs(d).mean()))
    assert N == idx.shape[0] and abs(V - 0.5 * (d ** 2).mean()) < 1e-12
    assert abs(G - np.abs(d).mean()) < 1e-12
    # exact V over ALL pairs must equal the closed-form V(inf) / GMD
    N, V, G = variogram_exact(y60, idx, 1)
    print('[exact]   all-pairs V == V(inf): %.12f vs %.12f | G == GMD: %.12f vs %.12f'
          % (V, v_infinity(y60), G, gini_mean_difference(y60)))
    assert abs(V - v_infinity(y60)) < 1e-10 and abs(G - gini_mean_difference(y60)) < 1e-10

    # ---- cached hamming == pairs.hamming_from_codes ----------------------- #
    a = load_assay('Z-domain_ZpA963_HL1_fitness_2M5A')
    rp = load_randpairs(a, create_if_missing=False)
    hh = _pairs.hamming_from_codes(a.codes, rp['idx'][:2_000_000],
                                   block=THRESH['hamming_block'])
    same = int((hh == rp['hamming'][:2_000_000]).sum())
    print('[cache]   cached hamming == hamming_from_codes on 2e6 pairs: %d/%d'
          % (same, hh.size))
    assert same == hh.size
    # and the lag-1 split reproduces the two cached class counts exactly
    nz = load_nested(a.dms_id)
    sz = load_samesite(a.dms_id)
    sel, isn = _split_lag1_sampled(a.codes, rp['idx'], rp['hamming'])
    print('[cache]   Z-HL1 sample is exact=%s: lag-1 split %d nested / %d same-site '
          'vs cached %d / %d' % (rp['exact'], int(isn.sum()), int((~isn).sum()),
                                 nz['idx'].shape[0], sz['idx'].shape[0]))
    if rp['exact']:
        assert int(isn.sum()) == nz['idx'].shape[0]
        assert int((~isn).sum()) == sz['idx'].shape[0]
    del a, rp, hh, nz, sz

    # ---- r/s: lsqr min-norm vs dense lstsq on a small assay --------------- #
    a = load_assay('Z-domain_ZpA963_HL2_fitness_2M5A')
    X = design_matrix(a)
    rs = roughness_to_slope(X, a.y)
    A = np.hstack([np.ones((a.n, 1)), X.toarray()])
    c, _, rank, _ = np.linalg.lstsq(A, a.y, rcond=None)
    res = a.y - A @ c
    print('[r/s]     Z-HL2 lsqr r=%.6f s=%.6f rs=%.6f | dense-lstsq r=%.6f s=%.6f '
          'rs=%.6f  rank=%d/%d'
          % (rs['r'], rs['s'], rs['rs'], np.sqrt((res ** 2).mean()),
             np.abs(c[1:]).mean(), np.sqrt((res ** 2).mean()) / np.abs(c[1:]).mean(),
             rank, A.shape[1]))
    assert abs(rs['r'] - np.sqrt((res ** 2).mean())) < 1e-6
    del a, X, A

    # ---- the whole benchmark ---------------------------------------------- #
    out = c1_all()
    t04, t05, xc = out['T04'], out['T05'], out['xcheck']

    print('\n' + '=' * 110)
    print('PRE-DECLARED SI (spec Sec.1.2 / Sec.2) vs MEASURED -- '
          'the acceptance test for this module')
    print('=' * 110)
    pr = out['predeclared']
    print('%-40s %-24s %8s %9s %9s %6s %9s'
          % ('DMS_id', 'role', 'SI_spec', 'measured', 'abs_diff', 'repro',
             'SI_nested'))
    for r in pr.itertuples():
        print('%-40s %-24s %8.3f %9.4f %9.5f %6s %9.4f'
              % (r.DMS_id[:40], r.role, r.SI_spec, r.SI_measured, r.abs_diff,
                 r.reproduces, r.SI_nested_only))
    print('reproduced: %d/%d within 5e-4' % (int(pr['reproduces'].sum()), len(pr)))

    print('\n' + '=' * 110)
    print('SAMPLED vs EXACT lag-1 cross-check (spec Sec.1.2: "must agree within '
          'the sampling SE").  z = (sampled - exact)/SE_sampled')
    print('=' * 110)
    cols = ['DMS_id', 'exact_enum', 'N_exact_all', 'N_samp_all', 'zV_all',
            'zG_all', 'zV_nest', 'zG_nest', 'zV_ss', 'zG_ss']
    xv = xc[[c for c in cols if c in xc.columns]].dropna(subset=['N_samp_all'])
    with pd.option_context('display.width', 250, 'display.max_columns', 30,
                           'display.float_format', lambda v: '%.3f' % v):
        print(xv.to_string(index=False))
    zs = xv[[c for c in xv.columns if c.startswith('z')]].values.astype(float).ravel()
    zs = np.abs(zs[np.isfinite(zs)])
    print('|z| over %d channel-statistic comparisons: median %.2f  p90 %.2f  max %.2f'
          % (zs.size, np.median(zs), np.percentile(zs, 90), zs.max()))

    print('\n' + '=' * 110)
    print('T04 observed columns (all %d assays)' % len(t04))
    print('=' * 110)
    show = ['DMS_id', 'SI', 'SI_lo95', 'SI_hi95', 'SI_nested_only',
            'samesite_SI_reference', 'V1_over_Vinf', 'V_monotone_h1_h4',
            'V_range_h90', 'gamma1', 'gamma1_lo95', 'gamma1_hi95', 'r_rough',
            's_slope', 'rs', 'R2_add_raw']
    with pd.option_context('display.width', 250, 'display.max_columns', 40,
                           'display.float_format', lambda v: '%.4f' % v):
        print(t04[show].to_string(index=False))

    print('\nT05 shape %s; lag labels present: %s'
          % (t05.shape, sorted(set(str(v) for v in t05['h']))))
    print('\nN3 (House-of-Cards) helpers -- handed to nulls.py, NOT written into T04:')
    a = load_assay('Z-domain_ZpA963_HL2_fitness_2M5A')
    n3 = rs_null_N3(design_matrix(a), a.y, B=20, dms_id=a.dms_id)
    print('  Z-HL2  rs_obs=%.4f  rs_N3_mean=%.4f +- %.4f (B=20)'
          % (float(t04.set_index('DMS_id').loc[a.dms_id, 'rs']),
             n3['rs_mean'], n3['rs_sd']))
    for d in ('Z-domain_ZSPA-1_LL2_fitness_1LP1', 'GB1_IgG-Fc_fitness_1FCC'):
        aa = load_assay(d)
        i1 = np.vstack([load_nested(d)['idx'], load_samesite(d)['idx']])
        s3 = si_null_N3(aa.y, i1, B=THRESH['null_B'], dms_id=d)
        print('  %-36s SI_obs=%.4f  SI_N3=%.4f +- %.4f  [%.4f, %.4f] (B=%d) '
              '-- E[SI_N3] == 1 exactly'
              % (d[:36], float(t04.set_index('DMS_id').loc[d, 'SI']),
                 s3['SI_mean'], s3['SI_sd'], s3['SI_p025'], s3['SI_p975'], s3['B']))
        del aa, i1
    print('\n[variogram] self-check complete.')
    return out


if __name__ == '__main__':
    _selfcheck()
