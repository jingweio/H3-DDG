"""BGYM-CLIFF v1 -- pair enumeration, the seeded random-pair sample, and the npz
cache (spec Sec.3 / Sec.5).

Three algorithms, all grafted from the ``sali`` design (spec Sec.0):

* **wildcard bucketing** for the two pair classes -- O(sum |K_v|) set lookups, not
  O(n^2).  Nested and same-site are enumerated into SEPARATE arrays and are never
  merged in any statistic (spec Sec.1.0: in GB1_1FCC 184,735 nested vs 861,874
  same-site, so merging would mislabel 82% of pairs as epistasis).
* **``np.bincount``, never ``np.add.at``** for every pair reduction (spec Sec.1.2).
* **one seeded 2e7 random-pair sample** per assay for the ``h >= 3`` variogram,
  materialised once, md5'd into ``MANIFEST.json`` and read-only downstream.

Pair-index arrays are ``int32`` (spec's numeric hygiene); code vectors are ``int8``.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import subprocess
import time

import numpy as np

from cliff import config
from cliff.config import PATHS, SEEDS, THRESH
from cliff.io_bgym import md5_of

# --------------------------------------------------------------------------- #
# enumeration                                                                 #
# --------------------------------------------------------------------------- #

def enumerate_nested(keys, col_index=None):
    """Nested pairs ``(B, B u {i})``, i.e. ``|K_u Delta K_v| = 1``.

    Wildcard bucketing: hold ``{K}`` in a set; for each variant ``v`` and each
    ``i in range(len(K_v))``, test membership of ``K_v \\ i``.

    Returns ``(idx, add_col)`` with ``idx`` ``(m, 2) int32``, **column 0 = the
    smaller set**, and ``add_col`` ``(m,) int32`` = the X-column of the added
    substitution.  Each unordered pair is counted exactly once (it is found only
    from the larger side).
    """
    if col_index is None:
        from cliff.io_bgym import build_col_index
        col_index, _ = build_col_index(keys)
    pos_of = {}
    for i, k in enumerate(keys):
        if k in pos_of:
            raise ValueError('duplicate canonical key at row %d: %r' % (i, k))
        pos_of[k] = i
    lo, hi, add = [], [], []
    get = pos_of.get
    for i, k in enumerate(keys):
        for j in range(len(k)):
            sub = k[:j] + k[j + 1:]
            p = get(sub)
            if p is not None:
                lo.append(p)
                hi.append(i)
                add.append(col_index[k[j]])
    idx = np.empty((len(lo), 2), dtype=np.int32)
    idx[:, 0] = lo
    idx[:, 1] = hi
    return idx, np.asarray(add, dtype=np.int32)


def enumerate_samesite(keys, pos_index=None):
    """Same-site swaps ``(B u {i->a}, B u {i->b})``, i.e. ``|K_u Delta K_v| = 2``
    with identical ``(chain, pos)``.

    Bucket each variant under ``(K \\ i, chain_i, pos_i)``; a bucket of size ``k``
    contributes exactly ``C(k, 2)`` pairs; every aa in a bucket is distinct, so no
    same-aa correction and no dedup.

    Returns ``(idx, pos_col)`` with ``idx`` ``(m, 2) int32`` ordered ``i < j`` by
    row number, and ``pos_col`` ``(m,) int32`` = the code-vector column of the
    swapped ``(chain, pos)``.
    """
    if pos_index is None:
        from cliff.io_bgym import build_col_index
        _, pos_index = build_col_index(keys)
    buckets = {}
    for i, k in enumerate(keys):
        for j in range(len(k)):
            ch, ps, _aa = k[j]
            b = (k[:j] + k[j + 1:], ch, ps)
            lst = buckets.get(b)
            if lst is None:
                buckets[b] = [i]
            else:
                lst.append(i)
    a_l, b_l, p_l = [], [], []
    for (bg, ch, ps), lst in buckets.items():
        if len(lst) < 2:
            continue
        c = pos_index[(ch, ps)]
        m = len(lst)
        for x in range(m):
            for y in range(x + 1, m):
                u, v = lst[x], lst[y]
                if u > v:
                    u, v = v, u
                a_l.append(u)
                b_l.append(v)
                p_l.append(c)
    idx = np.empty((len(a_l), 2), dtype=np.int32)
    idx[:, 0] = a_l
    idx[:, 1] = b_l
    return idx, np.asarray(p_l, dtype=np.int32)


# --------------------------------------------------------------------------- #
# random pairs / Hamming                                                      #
# --------------------------------------------------------------------------- #

def sample_random_pairs(n, n_draw, seed):
    """``(n_draw, 2) int32`` uniformly random unordered pairs with ``i < j``.

    Sampling is WITH replacement over the C(n,2) unordered pairs, which is what
    makes V(h)/G(h) unbiased moment estimators; ``i == j`` draws are rejected and
    redrawn.  ``seed`` is passed straight to ``np.random.default_rng`` (accepts an
    int or the ``[base, ordinal]`` entropy list from ``config.assay_seed``).
    """
    if n < 2:
        return np.empty((0, 2), dtype=np.int32)
    rng = np.random.default_rng(seed)
    out = np.empty((n_draw, 2), dtype=np.int32)
    filled = 0
    while filled < n_draw:
        need = n_draw - filled
        cand = rng.integers(0, n, size=(int(need * 1.05) + 8, 2), dtype=np.int64)
        keep = cand[cand[:, 0] != cand[:, 1]][:need]
        lo = np.minimum(keep[:, 0], keep[:, 1])
        hi = np.maximum(keep[:, 0], keep[:, 1])
        out[filled:filled + len(keep), 0] = lo
        out[filled:filled + len(keep), 1] = hi
        filled += len(keep)
    return out


def all_pairs_exact(n):
    """Every unordered pair, ``(C(n,2), 2) int32`` with ``i < j``."""
    i, j = np.triu_indices(int(n), k=1)
    out = np.empty((i.size, 2), dtype=np.int32)
    out[:, 0] = i
    out[:, 1] = j
    return out


def hamming_from_codes(codes, idx, block=1_000_000):
    """Number of positions at which the two code vectors differ, by block
    XOR-nonzero-count over the ``(n, P) int8`` code array.

    NOTE (reported, not hidden): this is the CODE-VECTOR Hamming distance, i.e.
    the number of differing ``(chain, pos)`` slots.  It equals 1 for a nested pair
    AND for a same-site swap, while the spec's symmetric-difference metric
    ``|K_u Delta K_v|`` gives 1 and 2 respectively.  ``variogram.py`` must
    therefore decide, once and explicitly, which of the two cached pair classes
    feeds ``V(1)``; the two class counts are cached separately for exactly that
    reason.
    """
    codes = np.ascontiguousarray(codes)
    idx = np.asarray(idx)
    m = idx.shape[0]
    if m == 0:
        return np.empty(0, dtype=np.int32)
    P = max(int(codes.shape[1]), 1)
    # keep each gathered block near 64 MB so peak RSS stays flat in P
    blk = max(1, min(int(block), int(64_000_000 // P)))
    out = np.empty(m, dtype=np.int32)
    for s in range(0, m, blk):
        e = min(s + blk, m)
        a = codes[idx[s:e, 0]]
        b = codes[idx[s:e, 1]]
        np.bitwise_xor(a, b, out=a)
        out[s:e] = np.count_nonzero(a, axis=1)
    return out


# --------------------------------------------------------------------------- #
# derived pair statistics (T01)                                              #
# --------------------------------------------------------------------------- #

def degrees(n, idx):
    """Node degree in a pair graph -- ``np.bincount``, never ``np.add.at``."""
    if idx.shape[0] == 0:
        return np.zeros(int(n), dtype=np.int64)
    return np.bincount(idx.reshape(-1), minlength=int(n))


def sibling_counts(idx, add_col, keys):
    """``|S(e)|`` for every nested edge ``e = (B, B u {i})``, where
    ``S(e) = {(B', B' u {i}) : |B xor B'| = 1}`` (spec Sec.1.4 L1).

    Within each ``add_col`` group the backgrounds form a set; ``|S(e)|`` is the
    degree of ``B`` in the nested graph restricted to that set.  On CR9114-H1
    (2^16 at 99.33%) every edge should have ~15 siblings, which is the spec's own
    stated check.
    """
    m = idx.shape[0]
    out = np.zeros(m, dtype=np.int32)
    if m == 0:
        return out
    order = np.argsort(add_col, kind='stable')
    ac = add_col[order]
    bounds = np.flatnonzero(np.diff(ac)) + 1
    for g in np.split(order, bounds):
        bg = [keys[idx[e, 0]] for e in g]
        loc = {}
        for t, k in enumerate(bg):
            if k in loc:
                raise ValueError('duplicate background inside an add_col group')
            loc[k] = t
        deg = np.zeros(len(g), dtype=np.int32)
        get = loc.get
        for t, k in enumerate(bg):
            for j in range(len(k)):
                u = get(k[:j] + k[j + 1:])
                if u is not None:
                    deg[t] += 1
                    deg[u] += 1
        out[g] = deg
    return out


def pairwise_column_stats(keys, col_index, min_obs=None):
    """Observations per interaction (``Z``) column, for the C3-L route L4 gate.

    A ``Z`` column is an ordered pair of X-columns, i.e. a pair of SUBSTITUTIONS
    ``(chain,pos,aa) x (chain,pos,aa)``, not a pair of sites: that is what makes
    GB1_1FCC infeasible at 1 observation per column (each of its 91,845 doubles
    is the only variant carrying its own substitution pair) while its 1,485 SITE
    pairs carry ~62 aa-combinations each -- the distinction the spec draws between
    the infeasible L4 and the feasible L2'.
    """
    if min_obs is None:
        min_obs = THRESH['L4_min_obs_per_col']
    M = max(len(col_index), 1)
    a_l, b_l = [], []
    for k in keys:
        if len(k) < 2:
            continue
        c = sorted(col_index[s] for s in k)
        for x in range(len(c)):
            cx = c[x]
            for y in range(x + 1, len(c)):
                a_l.append(cx)
                b_l.append(c[y])
    if not a_l:
        return dict(n_cols=0, n_obs=0, mean_obs_per_col=float('nan'),
                    n_cols_ge_min=0, frac_cols_ge_min=float('nan'),
                    n_cols_ge_N2b=0, feasible=False)
    flat = np.asarray(a_l, dtype=np.int64) * M + np.asarray(b_l, dtype=np.int64)
    _, cnt = np.unique(flat, return_counts=True)
    n_cols = int(cnt.size)
    return dict(n_cols=n_cols, n_obs=int(cnt.sum()),
                mean_obs_per_col=float(cnt.mean()),
                n_cols_ge_min=int((cnt >= min_obs).sum()),
                frac_cols_ge_min=float((cnt >= min_obs).mean()),
                n_cols_ge_N2b=int((cnt >= THRESH['N2b_min_cooccur']).sum()),
                feasible=bool(cnt.mean() >= min_obs))


def site_pair_stats(keys):
    """Backgrounds and aa-combinations per SITE pair -- the C3-L L2 / L2' gates."""
    from collections import defaultdict
    combos = defaultdict(set)
    for k in keys:
        if len(k) < 2:
            continue
        s = sorted(k)
        for x in range(len(s)):
            for y in range(x + 1, len(s)):
                sp = ((s[x][0], s[x][1]), (s[y][0], s[y][1]))
                combos[sp].add((s[x][2], s[y][2]))
    if not combos:
        return dict(n_site_pairs=0, median_aa_combos_per_site_pair=float('nan'),
                    n_site_pairs_ge_L2p=0)
    v = np.array([len(x) for x in combos.values()])
    return dict(n_site_pairs=int(v.size),
                median_aa_combos_per_site_pair=float(np.median(v)),
                n_site_pairs_ge_L2p=int((v >= THRESH['L2p_min_aa_combos']).sum()))


def primary_nested_set(assay, idx):
    """Boolean mask of the PRIMARY NESTED SET ``P_a`` conditions computable at
    stage 0 (spec Sec.1.0):

    (a) ``B != {}``       -- excludes the WT hub
    (b) neither endpoint at a detected censoring level

    (c) ``finite phi^oof`` at both endpoints is deferred to ``latent.py`` and
    (d) the tier filter (PRIMARY or ARM) is a downstream selection, not a
    property of the assay -- the count is reported for every assay because G5
    needs CR9114-H3's, and it is a CONTROL.
    """
    if idx.shape[0] == 0:
        z = np.zeros(0, dtype=bool)
        return z, z, z
    wt_anchored = np.zeros(idx.shape[0], dtype=bool)
    if assay.wt_row >= 0:
        wt_anchored = (idx[:, 0] == assay.wt_row) | (idx[:, 1] == assay.wt_row)
    cm = assay.censor_mask
    censor_touch = cm[idx[:, 0]] | cm[idx[:, 1]]
    return (~wt_anchored) & (~censor_touch), wt_anchored, censor_touch


# --------------------------------------------------------------------------- #
# cache                                                                       #
# --------------------------------------------------------------------------- #

def _save(path, **arrays):
    # np.savez appends '.npz' unless the name already ends in it, so the temp
    # name must keep the suffix last
    tmp = path[:-4] + '.tmp.npz' if path.endswith('.npz') else path + '.tmp'
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return dict(path=os.path.relpath(path, config.REPO), md5=md5_of(path),
                bytes=os.path.getsize(path))


def cache_keys(assay):
    """``data/cliff_cache/keys/{DMS_id}.npz``: codes, indices, row_index, n_muts, y."""
    PATHS.ensure_cache_dirs()
    p = os.path.join(PATHS.keys, assay.dms_id + '.npz')
    return _save(
        p,
        codes=assay.codes, y=assay.y, y_raw=assay.y_raw,
        row_index=assay.row_index, n_muts=assay.n_muts,
        censor_mask=assay.censor_mask,
        censor_levels=np.asarray(assay.censor_levels, dtype=np.float64),
        wt_row=np.int64(assay.wt_row), quantum=np.float64(assay.quantum),
        modal_decimals=np.int64(assay.modal_decimals),
        transform=np.array(assay.transform),
        col_index_json=np.array(json.dumps(
            [[list(k), v] for k, v in sorted(assay.col_index.items(), key=lambda t: t[1])])),
        pos_index_json=np.array(json.dumps(
            [[list(k), v] for k, v in sorted(assay.pos_index.items(), key=lambda t: t[1])])),
    )


def cache_pairs(assay):
    """Enumerate + cache both pair classes; returns the T01 pair statistics.

    Writes ``data/cliff_cache/pairs/{DMS_id}_{nested,samesite}.npz`` and returns a
    dict carrying the md5 manifest entries and every pair-derived T01 column.
    """
    PATHS.ensure_cache_dirs()
    t0 = time.time()
    n_idx, add_col = enumerate_nested(assay.keys, assay.col_index)
    t_nested = time.time() - t0
    t1 = time.time()
    s_idx, pos_col = enumerate_samesite(assay.keys, assay.pos_index)
    t_samesite = time.time() - t1

    pa_mask, wt_anchored, censor_touch = primary_nested_set(assay, n_idx)
    nd = degrees(assay.n, n_idx)
    sd = degrees(assay.n, s_idx)
    sib = sibling_counts(n_idx, add_col, assay.keys)

    ent = []
    ent.append(_save(os.path.join(PATHS.pairs, assay.dms_id + '_nested.npz'),
                     idx=n_idx, add_col=add_col, sibling_count=sib,
                     wt_anchored=wt_anchored, censor_touch=censor_touch,
                     primary_Pa=pa_mask))
    ent.append(_save(os.path.join(PATHS.pairs, assay.dms_id + '_samesite.npz'),
                     idx=s_idx, pos_col=pos_col))

    pw = pairwise_column_stats(assay.keys, assay.col_index)
    sp = site_pair_stats(assay.keys)
    n_nested = int(n_idx.shape[0])
    n_samesite = int(s_idx.shape[0])
    ge3 = sib >= THRESH['L1_min_siblings']
    return dict(
        DMS_id=assay.dms_id, manifest=ent,
        n_nested=n_nested, n_samesite=n_samesite,
        n_nested_wt_anchored=int(wt_anchored.sum()),
        n_nested_censor_touching=int(censor_touch.sum()),
        n_primary_Pa=int(pa_mask.sum()),
        pairs_per_variant=(n_nested + n_samesite) / float(assay.n),
        mean_nested_degree=2.0 * n_nested / float(assay.n),
        wt_degree=(int(nd[assay.wt_row]) if assay.wt_row >= 0 else 0),
        max_degree=int(nd.max()) if assay.n else 0,
        max_total_degree=int((nd + sd).max()) if assay.n else 0,
        # T01's n_edges_ge3_siblings is counted OVER P_a, because that is the set
        # the L1 gate acts on.  Counting over all nested edges would make
        # GB1_1FCC look L1-feasible on the strength of its WT hub alone (1,045
        # WT-anchored edges with up to 824 siblings each), which is exactly the
        # assay the spec lists as NOT L1-feasible.
        n_edges_ge3_siblings=int((ge3 & pa_mask).sum()),
        n_edges_ge3_siblings_all_nested=int(ge3.sum()),
        median_siblings=float(np.median(sib)) if sib.size else float('nan'),
        median_siblings_Pa=(float(np.median(sib[pa_mask])) if pa_mask.any()
                            else float('nan')),
        max_siblings=int(sib.max()) if sib.size else 0,
        mean_obs_per_pairwise_col=pw['mean_obs_per_col'],
        n_pairwise_cols=pw['n_cols'], pairwise_feasible=pw['feasible'],
        n_pairwise_cols_ge_N2b=pw['n_cols_ge_N2b'],
        n_site_pairs=sp['n_site_pairs'],
        median_aa_combos_per_site_pair=sp['median_aa_combos_per_site_pair'],
        n_site_pairs_ge_L2p=sp['n_site_pairs_ge_L2p'],
        wall_nested_s=round(t_nested, 2), wall_samesite_s=round(t_samesite, 2),
    )


def cache_randpairs(assay, n_draw=None, seed_name='randpairs'):
    """The ``h >= 3`` random-pair sample, materialised once and md5'd.

    DEVIATION (reported): when ``C(n, 2) <= n_draw`` the EXACT full pair set is
    written instead of a 2e7 sample with ~11x replacement.  Sampling 2e7 pairs
    from CR6261's 1.78e6 possible pairs would be strictly worse than enumerating
    them, and the exact set makes V(h) exact for that assay.  ``exact=True`` is
    stored in the npz and reported in T05's ``exact_or_sampled`` column.
    """
    PATHS.ensure_cache_dirs()
    if n_draw is None:
        n_draw = THRESH['randpair_n_draw']
    n = assay.n
    n_possible = n * (n - 1) // 2
    tag = '%s_%s_seed%d.npz' % (assay.dms_id, _sci(n_draw), SEEDS[seed_name])
    p = os.path.join(PATHS.randpairs, tag)
    t0 = time.time()
    if n_possible <= n_draw:
        idx = all_pairs_exact(n)
        exact = True
    else:
        idx = sample_random_pairs(n, n_draw, config.assay_seed(seed_name, assay.dms_id))
        exact = False
    h = hamming_from_codes(assay.codes, idx, block=THRESH['hamming_block'])
    ent = _save(p, idx=idx, hamming=h.astype(np.int16), exact=np.bool_(exact),
                n=np.int64(n), n_draw=np.int64(idx.shape[0]),
                seed=np.asarray(config.assay_seed(seed_name, assay.dms_id)))
    with open(p + '.md5', 'w') as fh:
        fh.write('%s  %s\n' % (ent['md5'], os.path.basename(p)))
    hb = np.bincount(h, minlength=assay.P + 1)
    return dict(DMS_id=assay.dms_id, manifest=[ent], exact=exact,
                n_possible_pairs=int(n_possible), n_drawn=int(idx.shape[0]),
                n_unique_drawn=(int(np.unique(idx[:, 0].astype(np.int64) * n
                                              + idx[:, 1]).size)
                                if not exact else int(idx.shape[0])),
                hamming_hist={int(k): int(v) for k, v in enumerate(hb) if v},
                wall_s=round(time.time() - t0, 2))


def _sci(v):
    """20000000 -> '2e7' (the spec's cache-file naming)."""
    s = '%e' % v
    m, e = s.split('e')
    m = m.rstrip('0').rstrip('.')
    return '%se%d' % (m, int(e))


# --------------------------------------------------------------------------- #
# MANIFEST.json                                                               #
# --------------------------------------------------------------------------- #

def git_provenance():
    try:
        head = subprocess.check_output(['git', '-C', config.REPO, 'rev-parse', 'HEAD'],
                                       stderr=subprocess.DEVNULL).decode().strip()
        dirty = len(subprocess.check_output(
            ['git', '-C', config.REPO, 'status', '--porcelain'],
            stderr=subprocess.DEVNULL).decode().splitlines())
    except Exception:
        head, dirty = '', -1
    return dict(commit=head, dirty_files=dirty)


#: Reserved top-level MANIFEST keys -- rebuilt on every write, never carried
#: forward.  Everything else at top level is a stage's own provenance block
#: (``latent``, ``structure``, ``variogram``, ``nulls``, ...) and IS carried
#: forward, so a concurrent writer cannot drop another stage's block.
_MANIFEST_RESERVED = ('schema', 'written_utc', 'env', 'env_observed', 'git',
                      'seed_base', 'seeds', 'assay_ordinal', 'taus',
                      'bindinggym_input', 'files')


@contextlib.contextmanager
def manifest_lock(timeout=120.0, poll=0.05):
    """Exclusive ``flock`` serialising the MANIFEST read-modify-write.

    Spec Sec.5 requires every cache file to be md5'd into ``MANIFEST.json``, and
    every stage module writes into the SAME file.  ``write_manifest`` used to be
    a bare replace, so two stage modules running concurrently could interleave
    (A reads, B reads, A writes, B writes) and B's write would silently drop
    every entry A had added -- observed losing entries in exactly that pattern.
    The lock file lives beside the manifest and is never deleted (deleting it is
    what makes lock files racy).
    """
    d = os.path.join(PATHS.cache, '.locks')
    os.makedirs(d, exist_ok=True)
    fh = open(os.path.join(d, 'manifest.lock'), 'a+')
    t0 = time.time()
    try:
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.time() - t0 > timeout:
                    raise RuntimeError(
                        'MANIFEST.json lock held for > %.0f s -- another writer '
                        'is stuck; inspect %s' % (timeout, d))
                time.sleep(poll)
        fh.seek(0)
        fh.truncate()
        fh.write('pid %d  %s\n' % (os.getpid(),
                                   time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                                 time.gmtime())))
        fh.flush()
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


def write_manifest(entries, extra=None):
    """``data/cliff_cache/MANIFEST.json``: md5 of every cache file + env tuple +
    git commit + seeds (spec Sec.5).  Downstream code verifies the md5 before use
    and refuses to run on a mismatch (:func:`verify_manifest`).

    **Concurrency (D8).**  The whole read-modify-write is serialised by
    :func:`manifest_lock`, and the ``files`` map is now a UNION of what is on
    disk *inside the lock* with ``entries`` (``entries`` winning on a collision)
    rather than a replacement.  Union, not replacement, is what makes the fix
    complete: a caller that read the manifest BEFORE taking the lock (which is
    what ``latent._update_manifest`` and ``structure._merge_manifest`` do) can no
    longer drop an entry another writer added in between.  Non-reserved
    top-level blocks are carried forward the same way.  The one thing this gives
    up is the ability to REMOVE a stale entry through this function; nothing in
    the pipeline does that, and :func:`verify_manifest` reports a missing file as
    ``MISSING`` rather than silently passing.
    """
    PATHS.ensure_cache_dirs()
    with manifest_lock():
        on_disk_files, on_disk_extra = {}, {}
        if os.path.exists(PATHS.manifest):
            try:
                with open(PATHS.manifest) as fh:
                    prev = json.load(fh)
                on_disk_files = prev.get('files', {}) or {}
                on_disk_extra = {k: v for k, v in prev.items()
                                 if k not in _MANIFEST_RESERVED}
            except (ValueError, OSError):
                on_disk_files, on_disk_extra = {}, {}
        files = dict(on_disk_files)
        files.update({e['path']: {'md5': e['md5'], 'bytes': e['bytes']}
                      for e in entries})
        man = dict(
            schema='bgym-cliff-v1/MANIFEST',
            written_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            env=list(config.EXPECTED_ENV),
            env_observed=list(config.assert_env()),
            git=git_provenance(),
            seed_base=config.SEED_BASE,
            seeds=dict(SEEDS),
            assay_ordinal=dict(config.ASSAY_ORDINAL),
            taus=list(config.TAUS),
            bindinggym_input=PATHS.bgym_input,
            files=files,
        )
        man.update(on_disk_extra)
        if extra:
            man.update(extra)
        tmp = '%s.tmp.%d' % (PATHS.manifest, os.getpid())
        with open(tmp, 'w') as fh:
            json.dump(man, fh, indent=1, sort_keys=True)
        os.replace(tmp, PATHS.manifest)
    return man


def verify_manifest():
    """Recompute every md5 in ``MANIFEST.json``.  Returns the list of mismatches;
    downstream code must refuse to run on a non-empty result."""
    with open(PATHS.manifest) as fh:
        man = json.load(fh)
    bad = []
    for rel, meta in sorted(man['files'].items()):
        p = os.path.join(config.REPO, rel)
        if not os.path.exists(p):
            bad.append((rel, 'MISSING', meta['md5']))
        else:
            got = md5_of(p)
            if got != meta['md5']:
                bad.append((rel, got, meta['md5']))
    return bad


# --------------------------------------------------------------------------- #
# stage 0 -- caches + T01 + T02                                               #
# --------------------------------------------------------------------------- #

#: spec Sec.6, verbatim and in order.  Columns needing a downstream statistic
#: (structure.py's interface fractions, G8's power) are written EMPTY, never
#: dropped and never back-filled from the spec's own expectations.
T01_COLUMNS = [
    'DMS_id', 'filename', 'registered', 'tier', 'family_id', 'structure_cluster_id',
    'exclusion_reason', 'n_rows', 'n_unique_keys', 'n_dup_keys', 'poi', 'pdb_file',
    'pdb_exists', 'side0_chains', 'side1_chains', 'scale_type', 'transform_applied',
    'sign_convention', 'has_wt_row', 'wt_row_index', 'wt_value', 'wt_percentile',
    'rho_depth_score', 'max_mut', 'mut_count_hist', 'n_positions',
    'aa_per_pos_median', 'y_min', 'y_max', 'y_sd', 'y_mad', 'n_distinct_values',
    'modal_decimals', 'quantum', 'floor_value', 'floor_frac', 'ceil_value',
    'ceil_frac', 'modal_value_frac', 'n_nested', 'n_samesite',
    'n_nested_wt_anchored', 'n_nested_censor_touching', 'n_primary_Pa',
    'pairs_per_variant', 'wt_degree', 'max_degree', 'n_edges_ge3_siblings',
    'mean_obs_per_pairwise_col', 'pairwise_feasible', 'design_iface_frac',
    'bg_iface_frac', 'iface_bias_factor', 'eligible_C1', 'eligible_C2',
    'eligible_C3L', 'eligible_C4S', 'eligible_C4P', 'eligible_C4I',
    'eligible_cluster_channel', 'underpowered_G8',
]

T02_COLUMNS = ['gate_id', 'gate_name', 'assay', 'statistic', 'expected', 'observed',
               'tolerance', 'PASS/FAIL', 'consequence_if_fail', 'halts_study']


def _mad(v):
    v = np.asarray(v, dtype=np.float64)
    return float(THRESH['mad_const'] * np.median(np.abs(v - np.median(v))))


def t01_row(assay, pair_stats):
    """One T01 row: everything measurable at stage 0."""
    import pandas as pd
    from scipy.stats import spearmanr
    spec = config.ASSAYS[assay.dms_id]
    yr = assay.y_raw
    n = assay.n
    u, c = np.unique(yr, return_counts=True)
    _lv, _mk, cmeta = detect_censoring_raw(yr)
    # per-position distinct mutant aa
    per_pos = {}
    for k in assay.keys:
        for (ch, ps, aa) in k:
            per_pos.setdefault((ch, ps), set()).add(aa)
    aa_per_pos = np.array([len(v) for v in per_pos.values()]) if per_pos else np.array([0])
    hist = np.bincount(assay.n_muts.astype(np.int64))
    wt = assay.wt_row
    rho = spearmanr(assay.n_muts.astype(np.float64), assay.y)
    row = dict(
        DMS_id=assay.dms_id, filename=spec.filename, registered=spec.registered,
        tier=spec.tier, family_id=spec.family_id,
        structure_cluster_id=spec.structure_cluster_id,
        exclusion_reason=spec.exclusion_reason,
        n_rows=n, n_unique_keys=len(set(assay.keys)),
        n_dup_keys=n - len(set(assay.keys)),
        poi=spec.poi, pdb_file=spec.pdb_file,
        pdb_exists=os.path.exists(os.path.join(PATHS.structures, spec.pdb_file)),
        side0_chains=spec.side0_chains, side1_chains=spec.side1_chains,
        scale_type=spec.scale_type, transform_applied=spec.transform,
        sign_convention=spec.sign_convention,
        has_wt_row=(wt >= 0), wt_row_index=(wt if wt >= 0 else ''),
        wt_value=(('%.10g' % yr[wt]) if wt >= 0 else ''),
        wt_percentile=(round(100.0 * float((yr < yr[wt]).mean()), 4) if wt >= 0 else ''),
        rho_depth_score=(round(float(rho.correlation), 6)
                         if np.isfinite(rho.correlation) else ''),
        max_mut=int(assay.n_muts.max()),
        mut_count_hist=';'.join('%d:%d' % (i, v) for i, v in enumerate(hist) if v),
        n_positions=assay.P, aa_per_pos_median=float(np.median(aa_per_pos)),
        y_min=float(yr.min()), y_max=float(yr.max()),
        y_sd=float(yr.std(ddof=1)), y_mad=_mad(yr),
        n_distinct_values=int(u.size), modal_decimals=assay.modal_decimals,
        quantum=assay.quantum,
        floor_value=cmeta['floor_value'], floor_frac=cmeta['floor_frac'],
        ceil_value=cmeta['ceil_value'], ceil_frac=cmeta['ceil_frac'],
        modal_value_frac=float(c.max()) / n,
        n_nested=pair_stats['n_nested'], n_samesite=pair_stats['n_samesite'],
        n_nested_wt_anchored=pair_stats['n_nested_wt_anchored'],
        n_nested_censor_touching=pair_stats['n_nested_censor_touching'],
        n_primary_Pa=pair_stats['n_primary_Pa'],
        pairs_per_variant=round(pair_stats['pairs_per_variant'], 4),
        wt_degree=pair_stats['wt_degree'], max_degree=pair_stats['max_degree'],
        n_edges_ge3_siblings=pair_stats['n_edges_ge3_siblings'],
        mean_obs_per_pairwise_col=(round(pair_stats['mean_obs_per_pairwise_col'], 4)
                                   if np.isfinite(pair_stats['mean_obs_per_pairwise_col'])
                                   else ''),
        pairwise_feasible=pair_stats['pairwise_feasible'],
        # ---- deferred to cliff/structure.py (measured, never transcribed) ----
        design_iface_frac='', bg_iface_frac='', iface_bias_factor='',
        eligible_C1=spec.eligible_C1, eligible_C2=spec.eligible_C2,
        eligible_C3L=spec.eligible_C3L, eligible_C4S=spec.eligible_C4S,
        eligible_C4P=spec.eligible_C4P, eligible_C4I=spec.eligible_C4I,
        eligible_cluster_channel=spec.eligible_cluster_channel,
        # ---- deferred to G8 (cliff/calibrate.py) ----
        underpowered_G8='',
    )
    _ = pd
    assert set(row) == set(T01_COLUMNS), (
        sorted(set(row) ^ set(T01_COLUMNS)))
    return row


def detect_censoring_raw(y_raw):
    """:func:`cliff.io_bgym.detect_censoring` on the RAW score column.

    T01's distributional columns are on the raw scale, because that is the scale
    every profile number the spec quotes lives on (hYAP65 "min 0.00911, max
    15.56"; SARS2-RBD "-4.84/-4.76"; CR9114-H3 "89.05% at exactly 6.000").  The
    ANALYSIS-scale detection (post-log10 for hYAP65) is what ``Assay.censor_mask``
    carries; log10 is strictly monotone, and hYAP65 has no censoring at all, so
    the two agree row-for-row on all 28 assays.
    """
    from cliff.io_bgym import detect_censoring
    return detect_censoring(y_raw)


def stage0(assays=None, *, do_randpairs=True, verbose=True):
    """Load 28 -> caches -> T01 -> randpair samples -> MANIFEST -> T02."""
    import resource
    import pandas as pd
    from cliff.io_bgym import audit_all, load_assay
    config.assert_env()
    PATHS.ensure_cache_dirs()
    ids = list(config.ALL_ASSAYS) if assays is None else list(assays)
    t_start = time.time()

    if verbose:
        print('#' * 100)
        print('# STAGE 0 -- parse audit, pair enumeration, random-pair sample')
        print('#' * 100, flush=True)
    audit = audit_all(ids, deep=True, literal_eval_check=True, verbose=verbose)

    rows, manifest, pstats, rstats, bench = [], [], {}, {}, {}
    for k, dms_id in enumerate(ids):
        r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        t0 = time.time()
        a = load_assay(dms_id)
        t_load = time.time() - t0
        manifest.append(cache_keys(a))
        ps = cache_pairs(a)
        manifest.extend(ps.pop('manifest'))
        pstats[dms_id] = ps
        rows.append(t01_row(a, ps))
        if do_randpairs and config.ASSAYS[dms_id].tier in ('PRIMARY', 'ARM'):
            rs = cache_randpairs(a)
            manifest.extend(rs.pop('manifest'))
            rstats[dms_id] = rs
        r1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        bench[dms_id] = dict(load_s=round(t_load, 2),
                             enum_s=round(ps['wall_nested_s'] + ps['wall_samesite_s'], 2),
                             randpair_s=rstats.get(dms_id, {}).get('wall_s', ''),
                             peak_rss_gb=round(r1, 3))
        if verbose:
            print('[stage0] %2d/%2d %-44s nested=%-7d samesite=%-7d wt_anch=%-6d '
                  'censor=%-6d Pa=%-7d ppv=%6.2f sib>=3(Pa)=%-7d enum=%5.2fs '
                  'rand=%-6s peakRSS=%.2fGB'
                  % (k + 1, len(ids), dms_id, ps['n_nested'], ps['n_samesite'],
                     ps['n_nested_wt_anchored'], ps['n_nested_censor_touching'],
                     ps['n_primary_Pa'], ps['pairs_per_variant'],
                     ps['n_edges_ge3_siblings'], bench[dms_id]['enum_s'],
                     bench[dms_id]['randpair_s'], r1), flush=True)
        del a

    t01 = pd.DataFrame(rows)[T01_COLUMNS]
    p01 = os.path.join(PATHS.artifacts, 'T01_assay_manifest.csv')
    t01.to_csv(p01, index=False)

    man = write_manifest(manifest, extra=dict(
        stage0=dict(wall_s=round(time.time() - t_start, 1),
                    peak_rss_gb=round(resource.getrusage(
                        resource.RUSAGE_SELF).ru_maxrss / 1e6, 3),
                    n_assays=len(ids)),
        bench=bench,
        randpairs={k: dict(exact=v['exact'], n_drawn=v['n_drawn'],
                           n_possible_pairs=v['n_possible_pairs'],
                           hamming_hist=v['hamming_hist']) for k, v in rstats.items()}))

    t02 = build_T02(audit, pstats, bench)
    p02 = os.path.join(PATHS.artifacts, 'T02_gates.csv')
    t02.to_csv(p02, index=False)
    if verbose:
        print('\n[stage0] wrote %s (%d x %d)' % (p01, len(t01), len(t01.columns)))
        print('[stage0] wrote %s (%d x %d)' % (p02, len(t02), len(t02.columns)))
        print('[stage0] wrote %s (%d cached files)' % (PATHS.manifest, len(man['files'])))
        print('[stage0] wall %.1fs  peak RSS %.2f GB'
              % (time.time() - t_start,
                 resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))
    return dict(T01=t01, T02=t02, audit=audit, pair_stats=pstats,
                randpair_stats=rstats, manifest=man, bench=bench)


def _g(gate_id, name, assay, stat, expected, observed, tol, consequence, halts,
       mode='eq'):
    """One T02 row.  ``mode``: 'eq' (|obs-exp| <= tol), 'le' (obs <= exp, a budget)
    or 'ge'.  An empty observed value is PENDING, never PASS."""
    if observed == '' or observed is None or expected == '':
        verdict = 'PENDING'
    else:
        try:
            o, e = float(observed), float(expected)
            if mode == 'le':
                ok = o <= e
            elif mode == 'ge':
                ok = o >= e
            else:
                ok = abs(o - e) <= float(tol or 0)
        except (TypeError, ValueError):
            ok = (str(observed) == str(expected))
        verdict = 'PASS' if ok else 'FAIL'
    return dict(gate_id=gate_id, gate_name=name, assay=assay, statistic=stat,
                expected=expected, observed=observed, tolerance=tol,
                **{'PASS/FAIL': verdict}, consequence_if_fail=consequence,
                halts_study=halts)


def build_T02(audit, pstats, bench):
    """T02 with the exact spec Sec.6 columns.  G0/G1/G1b/G2/G3 are observed here;
    G4-G11 / G-UP / G-OPT rows are written with an empty observed value and
    ``PENDING``, so the table is structurally complete from stage 0 on."""
    import pandas as pd
    E = config.EXPECTED
    G1 = audit.attrs['G1']
    G1b = audit.attrs['G1b']
    G2 = audit.attrs['G2']
    G3 = audit.attrs['G3']
    reg = [k for k in audit['DMS_id'] if config.ASSAYS[k].registered]
    inst_reg = int(audit.set_index('DMS_id').loc[reg, 'n_mutation_instances'].sum())
    npos_reg = int(audit.set_index('DMS_id').loc[reg, 'n_distinct_pdb_keys'].sum())
    STOP = 'STOP -- the data is not what the profile describes'
    rows = []

    # ---------------------------- G0 ---------------------------- #
    for dms_id in ('GB1_IgG-Fc_fitness_1FCC', 'CR9114_FluAH1_logKd_4FQI'):
        b = bench.get(dms_id)
        if b is None:
            continue
        rows.append(_g('G0', 'pre-flight benchmark: pair enumeration wall', dms_id,
                       'enumeration wall (s), <= budget', THRESH['G0_enum_wall_s'],
                       b['enum_s'], 'budget',
                       'switch to the n=40,000 x 5-subsample path; SI between-subsample '
                       'SE must be <= 0.03 else INCONCLUSIVE', 'no', mode='le'))
        rows.append(_g('G0', 'pre-flight benchmark: peak RSS', dms_id,
                       'process peak-RSS high-water mark (GB) after this assay -- an '
                       'UPPER BOUND on the per-assay figure, since ru_maxrss never '
                       'decreases', THRESH['G0_enum_rss_gb'],
                       b['peak_rss_gb'], 'budget', 'as above', 'no', mode='le'))
    tot_enum = round(sum(v['enum_s'] for v in bench.values()), 2)
    peak = max(v['peak_rss_gb'] for v in bench.values()) if bench else ''
    rows.append(_g('G0', 'whole-benchmark enumeration wall', 'ALL_28',
                   'sum of per-assay enumeration wall (s)', E['enum_all_wall_s'],
                   tot_enum, 'budget',
                   'runtime plan Sec.5 is wrong; re-price stage 0', 'no', mode='le'))
    rows.append(_g('G0', 'whole-benchmark peak RSS', 'ALL_28', 'peak RSS (GB)',
                   E['enum_all_rss_gb'], peak, 'budget', 'as above', 'no', mode='le'))
    rows.append(_g('G0', 'fit_latent budget', 'GB1_IgG-Fc_fitness_1FCC',
                   'fit_latent 10 iterations (s)', THRESH['G0_fit_latent_s'], '',
                   'budget', 'subsample path', 'no', mode='le'))
    rows.append(_g('G0', 'one null replicate budget', 'GB1_IgG-Fc_fitness_1FCC',
                   'one replicate (s)', THRESH['G0_one_replicate_s'], '',
                   'budget', 'subsample path', 'no', mode='le'))

    # ---------------------------- G1 ---------------------------- #
    rows.append(_g('G1', 'parse audit: rows', 'ALL_28', 'total rows',
                   E['G1_n_rows_total'], G1['n_rows'], 0, STOP, 'YES'))
    rows.append(_g('G1', 'parse audit: unique canonical keys', 'ALL_28',
                   'sum of per-file unique canonical keys',
                   E['G1_n_unique_keys_total'], G1['n_unique_keys'], 0, STOP, 'YES'))
    rows.append(_g('G1', 'parse audit: duplicate canonical keys', 'ALL_28',
                   'duplicate rows under the canonical key (chain retained)', 0,
                   G1['n_dup_keys'], 0, STOP, 'YES'))
    rows.append(_g('G1', 'parse audit: mutation instances (25 registered)',
                   'REGISTERED_25', 'mutation tokens',
                   E['G1_n_mutation_instances_registered25'], inst_reg, 0, STOP, 'YES'))
    rows.append(_g('G1', 'parse audit: mutation instances (all 28)', 'ALL_28',
                   'mutation tokens; the spec quotes the 25-registered figure in '
                   'the same line as the 28-file row count',
                   E['G1_n_mutation_instances_all28'], G1['n_mutation_instances'], 0,
                   STOP, 'YES'))
    rows.append(_g('G1', 'parse audit: mutated positions (25 registered)',
                   'REGISTERED_25', 'distinct (chain,resseq,icode) mutated positions',
                   E['G1_n_mutated_positions_registered25'], npos_reg, 0,
                   'Sec.1.5 burial denominators change', 'no'))
    rows.append(_g('G1', 'parse audit: wt-letter mismatches, mutant vs mutant_pdb',
                   'ALL_28', 'mismatching instances', 0,
                   int(audit['n_wt_mismatch_mutant_vs_pdbcol'].sum()), 0, STOP, 'YES'))
    rows.append(_g('G1', 'parse audit: wt-letter mismatches vs wildtype_sequence',
                   'ALL_28', 'mismatching instances', 0,
                   int(audit['n_wt_mismatch_vs_wildtype_sequence'].sum()), 0,
                   STOP, 'YES'))
    rows.append(_g('G1', 'parse audit: wt-letter mismatches vs PDB residue',
                   'REGISTERED_25', 'mismatching instances (22 PDBs; the 3 '
                   'unregistered assays have no PDB in structures/)', 0,
                   int(sum(v for v in audit['n_wt_mismatch_vs_pdb_residue']
                           if v != '')), 0, STOP, 'YES'))
    rows.append(_g('G1', 'parse audit: PDB residue keys not found', 'REGISTERED_25',
                   '(chain,resseq,icode) absent from the PDB', 0,
                   int(sum(v for v in audit['n_pdb_key_missing'] if v != '')), 0,
                   'mutant_pdb lookup is not a total function; map_mutations breaks',
                   'YES'))
    rows.append(_g('G1', 'parse audit: X-hits', 'ALL_28',
                   'mutations landing on an X of wildtype_sequence', E['G1_n_X_hits'],
                   G1['n_X_hits'], 0, STOP, 'YES'))
    rows.append(_g('G1', 'parse audit: * tokens', 'ALL_28', 'stop-codon tokens',
                   E['G1_n_star_tokens'], G1['n_star_tokens'], 0, STOP, 'YES'))
    rows.append(_g('G1', 'parse audit: identity mutations', 'ALL_28',
                   'tokens with wt == mut', E['G1_n_identity_mutations'],
                   G1['n_identity_mutations'], 0, STOP, 'YES'))
    rows.append(_g('G1', 'parse audit: per-chain token counts agree', 'ALL_28',
                   'mutant vs mutant_pdb per-chain token totals', 'True',
                   str(G1['all_chain_token_counts_agree']), 0, STOP, 'YES'))
    rows.append(_g('G1', 'parse audit: mutated_sequence reconstruction', 'ALL_28',
                   'rows whose mutated_sequence does not rebuild from '
                   'wildtype_sequence + mutant', 0, G1['n_reconstruction_fail'], 0,
                   'Hamming != mutation-set distance; all pair work is invalid',
                   'YES'))
    rows.append(_g('G1', 'parse audit: indels', 'ALL_28',
                   'rows where len(mutated) != len(wildtype) on any chain',
                   E['G1_n_indels'], G1['n_indels'], 0, STOP, 'YES'))
    rows.append(_g('G1', 'parse audit: regex dict parser vs ast.literal_eval',
                   'ALL_28', 'disagreements over %d dict strings'
                   % G1['n_literal_eval_checked'], 0, G1['n_literal_eval_diff'], 0,
                   'the fast parser is unsound; revert to ast.literal_eval', 'YES'))
    hla = audit.attrs['HLA_duplicate_column']
    rows.append(_g('G1', 'HLA-A2 duplicated DMS_score column is identical',
                   'HLA-A2_TAPBPR_meanscore_5WER',
                   'rows differing between the two DMS_score columns', 0,
                   hla['n_diff_between_duplicates'], 0,
                   'usecols silently picks one of two different scores', 'YES'))

    # ---------------------------- G1b --------------------------- #
    for tag in ('mutant_seq', 'mutant_pdb'):
        v = G1b[tag]
        rows.append(_g('G1b', 'BH3 cross-assay join via %s + WT-residue identity' % tag,
                       'BH3_Bcl-xL_normed_1PQ1 x BH3_Mcl-1_normed_3KZ0',
                       'shared keys at the WT-consistent offset %+d' % v['offset'],
                       E['G1b_n_shared'], v['n_shared'], 0,
                       'the C4-I BH3 probe has no join', 'no'))
    rows.append(_g('G1b', 'BH3 naive (pos,aa) join -- BANNED, measured to show why',
                   'BH3_Bcl-xL_normed_1PQ1 x BH3_Mcl-1_normed_3KZ0',
                   'shared keys with NO offset, mutant numbering',
                   E['G1b_naive_join_n_shared'], G1b['mutant_seq']['naive_n_shared'], 0,
                   'n/a -- this row exists to document the banned route', 'no'))
    rows.append(_g('G1b', 'BH3 r disagreement resolved: Pearson',
                   'BH3_Bcl-xL_normed_1PQ1 x BH3_Mcl-1_normed_3KZ0',
                   'Pearson r on the 518/518 join',
                   E['G1b_r_claim_a'], round(G1b['mutant_seq']['pearson'], 4), 0.001,
                   'n/a', 'no'))
    rows.append(_g('G1b', 'BH3 r disagreement resolved: Spearman',
                   'BH3_Bcl-xL_normed_1PQ1 x BH3_Mcl-1_normed_3KZ0',
                   'Spearman rho on the SAME 518/518 join',
                   E['G1b_r_claim_b'], round(G1b['mutant_seq']['spearman'], 4), 0.001,
                   'n/a', 'no'))
    rows.append(_g('G1b', 'BH3 seq-numbering offset', 'BH3 pair',
                   '3KZ0 chain C seq pos - 1PQ1 chain B seq pos',
                   E['G1b_offset_seq'], G1b['mutant_seq']['offset'], 0,
                   'the -2 correction in Sec.2 is wrong', 'no'))

    # ---------------------------- G2 ---------------------------- #
    rows.append(_g('G2', 'twin-assay shared keys',
                   'KRAS_SOS1_norfitness_8BE4 x KRAS_DARPinK27_norfitness_5O2S',
                   'shared keys (side0 token multiset)', E['G2_n_shared_keys'],
                   G2['n_shared'], 0,
                   'STOP -- the de-duplication premise has changed', 'YES'))
    rows.append(_g('G2', 'twin-assay byte identity',
                   'KRAS_SOS1_norfitness_8BE4 x KRAS_DARPinK27_norfitness_5O2S',
                   'max|Delta| on the raw score STRINGS', E['G2_max_abs_delta'],
                   G2['max_abs_delta'], 0,
                   'STOP -- the de-duplication premise has changed', 'YES'))
    rows.append(_g('G2', 'twin-assay raw-string differences',
                   'KRAS_SOS1_norfitness_8BE4 x KRAS_DARPinK27_norfitness_5O2S',
                   'shared keys whose DMS_score strings differ byte-for-byte', 0,
                   G2['n_raw_string_differences'], 0,
                   'STOP -- the de-duplication premise has changed', 'YES'))

    # ---------------------------- G3 ---------------------------- #
    for r in G3.itertuples():
        rows.append(_g('G3', 'chain-key integrity: duplicates WITH the chain label',
                       r.DMS_id, 'duplicate genotypes (canonical key)',
                       r.expected_dups_with_chain, r.n_dup_keys_with_chain, 0,
                       'the within-genotype SDs are REAL and become the primary noise '
                       'floor for this assay', 'no'))
        rows.append(_g('G3', 'chain-key integrity: duplicates WITHOUT the chain label',
                       r.DMS_id, 'duplicate genotypes keyed on (seq_pos,wt_aa,mut_aa)',
                       r.expected_dups_without_chain, r.n_dup_keys_without_chain, 0,
                       'the chain-collision diagnosis is wrong', 'no'))

    # ------------------- downstream gates, PENDING -------------- #
    pend = [
        ('G4', 'null self-calibration: T(tau) = 1.00 +/- 0.05 on a held-out N1',
         'PRIMARY+ARM', 'T(tau) over 199 N1 surrogates, tolerance +/-%g'
         % THRESH['G4_T_tol'], 1.00, '',
         'STOP -- the surrogate machinery is biased; no observed number is readable',
         'YES'),
        ('G4', 'null self-calibration: 200 empirical p-values uniform',
         'PRIMARY+ARM', 'KS p', '>%g' % THRESH['G4_ks_p_min'], '', 'as above', 'YES'),
        ('G5', 'censoring positive control: unmasked T(4)',
         'CR9114_FluAH3_logKd_4FQY', 'T(4) unmasked',
         '>=%g' % THRESH['G5_unmasked_T4_min'], '',
         'STOP -- the pipeline cannot tell a detection limit from a cliff', 'YES'),
        ('G5', 'censoring positive control: |P_a| collapse after floor masking',
         'CR9114_FluAH3_logKd_4FQY', '|P_a| after masking',
         '<=%d' % THRESH['G5_Pa_after_max'], '', 'as above', 'YES'),
        ('G6', 'anti-smooth negative control: C1 REFUTED and T(4) inside the N2 band',
         'Z-domain_ZSPA-1_LL1_fitness_1LP1 / LL2', 'verdict_C1 and T(4)',
         'REFUTED / inside', '',
         'STOP -- the pipeline is fooled by selection-dependent library membership',
         'YES'),
        ('G7', 'scale-mixture discrimination: does N2c inflate TR / T(tau)?',
         'PRIMARY+ARM', 'TR and T(tau) on 200 N2c surrogates', 'sets the verdict rule',
         '', 'if localisation is ALSO inflated under N2c -> STOP', 'YES'),
        ('G8', 'power & bias: detection power at a = 4 sigma, pi = 0.005',
         '6 representative assays', 'power',
         '>=%g' % THRESH['G8_power_min'], '',
         'the assay is stamped UNDERPOWERED and reports INCONCLUSIVE', 'no'),
        ('G9', 'aggregate-rule FPR of the k-of-7 rule', '50 N1 datasets',
         'family-level FPR', '<=%g' % THRESH['G9_family_fpr_max'], '',
         'tighten k until <= 0.10 and record the change before any observed value',
         'no'),
        ('G10', 'censoring-mask composition (order x degree-decile x phi-decile)',
         'censored assays', 'max absolute bin-proportion difference',
         '<=%g' % THRESH['G10_max_bin_prop_diff'], '',
         'the clamp replay in the null is mis-specified; flag every claim as '
         'conditional', 'no'),
        ('G11', 'twin-structure control: at most ONE KRAS interface can be causal',
         'KRAS_SOS1_norfitness_8BE4 / KRAS_DARPinK27_norfitness_5O2S',
         'OR at 8BE4 vs OR at 5O2S', 'not both', '',
         'reported as a finding, not a stop', 'no'),
        ('G-UP', 'optional upstream per-variant SE arm',
         'SARS2-RBD / GB1 / CR9114', 'per-variant SE or read counts obtained',
         'obtained', '',
         'every C3-N verdict is stamped conditional and the record must say '
         '"effect size relative to one contaminated replicate bound, not a '
         'calibrated significance"', 'no'),
        ('G-OPT', 'optional structural recovery for the hypercube arm',
         'CR9114_FluAH1_logKd_4FQI / 4FQY / CR6261',
         '4FQI/4FQY/3GBN fetched and the 16/11 somatic sites mapped at 100%',
         '100%', '',
         'skip the structural half of the arm and say so; NEVER substitute a '
         'constant offset', 'no'),
    ]
    for gid, name, assay, stat, exp, obs, cons, halt in pend:
        rows.append(_g(gid, name, assay, stat, exp, obs, '', cons, halt))

    df = pd.DataFrame(rows)[T02_COLUMNS]
    return df


# --------------------------------------------------------------------------- #
# self-check                                                                  #
# --------------------------------------------------------------------------- #

def _brute_nested(keys):
    ks = set(keys)
    out = 0
    for k in keys:
        for j in range(len(k)):
            if k[:j] + k[j + 1:] in ks:
                out += 1
    return out


def _brute_pairs(keys):
    """O(n^2) reference for both classes -- only for tiny assays."""
    n_nested = n_ss = 0
    for i in range(len(keys)):
        a = set(keys[i])
        for j in range(i + 1, len(keys)):
            b = set(keys[j])
            d = a ^ b
            if len(d) == 1:
                n_nested += 1
            elif len(d) == 2:
                x, y = sorted(d)
                if (x[0], x[1]) == (y[0], y[1]):
                    n_ss += 1
    return n_nested, n_ss


def _selfcheck():
    import resource
    from cliff.io_bgym import load_assay
    config.assert_env()
    print('=' * 100)
    print('cliff.pairs self-check -- runs on the real data')
    print('=' * 100)

    # ---- brute-force agreement on a small real assay ----
    for dms_id in ('Z-domain_ZpA963_HL2_fitness_2M5A', 'BH3_Bcl-xL_normed_1PQ1',
                   'PSD95_CRIPT_1BE9'):
        a = load_assay(dms_id)
        ni, ac = enumerate_nested(a.keys, a.col_index)
        si, pc = enumerate_samesite(a.keys, a.pos_index)
        bn, bs = _brute_pairs(a.keys)
        print('[brute] %-34s n=%-5d nested %6d vs brute %6d %s   samesite %6d vs '
              'brute %6d %s'
              % (dms_id, a.n, ni.shape[0], bn, 'OK' if ni.shape[0] == bn else 'FAIL',
                 si.shape[0], bs, 'OK' if si.shape[0] == bs else 'FAIL'))
        assert ni.shape[0] == bn and si.shape[0] == bs
        assert (ni[:, 0] != ni[:, 1]).all() and (si[:, 0] < si[:, 1]).all()
        # column 0 must be the smaller set
        assert all(len(a.keys[u]) + 1 == len(a.keys[v]) for u, v in ni)

    # ---- hamming_from_codes against a naive reference ----
    a = load_assay('Z-domain_ZpA963_HL1_fitness_2M5A')
    rng = np.random.default_rng(config.SEEDS['base'])
    ridx = np.stack([rng.integers(0, a.n, 20000), rng.integers(0, a.n, 20000)], 1
                    ).astype(np.int32)
    h_fast = hamming_from_codes(a.codes, ridx, block=997)
    h_naive = np.array([(a.codes[i] != a.codes[j]).sum() for i, j in ridx])
    assert np.array_equal(h_fast, h_naive), 'hamming mismatch'
    from scipy.spatial.distance import hamming as sp_ham
    h_scipy = np.array([round(sp_ham(a.codes[i], a.codes[j]) * a.P) for i, j in ridx[:500]])
    assert np.array_equal(h_fast[:500], h_scipy)
    print('[hamming] 20,000 pairs: block XOR == naive != ; first 500 == scipy.hamming  OK')
    # code-Hamming vs symmetric difference, on the two cached classes
    ni, ac = enumerate_nested(a.keys, a.col_index)
    si, pc = enumerate_samesite(a.keys, a.pos_index)
    hn = hamming_from_codes(a.codes, ni[:5000])
    hs = hamming_from_codes(a.codes, si[:5000])
    print('[hamming] nested code-H: %s   samesite code-H: %s  '
          '(symmetric-difference metric would be 1 and 2)'
          % (sorted(set(hn.tolist())), sorted(set(hs.tolist()))))
    assert set(hn.tolist()) == {1} and set(hs.tolist()) == {1}

    # ---- sample_random_pairs determinism + uniformity ----
    s1 = sample_random_pairs(1000, 50000, config.assay_seed('randpairs', 'PSD95_CRIPT_1BE9'))
    s2 = sample_random_pairs(1000, 50000, config.assay_seed('randpairs', 'PSD95_CRIPT_1BE9'))
    s3 = sample_random_pairs(1000, 50000, config.assay_seed('randpairs', 'PSD95_Tm2F_1BE9'))
    assert np.array_equal(s1, s2) and not np.array_equal(s1, s3)
    assert (s1[:, 0] < s1[:, 1]).all() and s1.dtype == np.int32
    print('[sample] deterministic per (seed name, assay); two assays with the same n get '
          'DIFFERENT streams; i<j, int32  OK')

    # ---- G0 pre-flight benchmark ----
    print('-' * 100)
    print('[G0] pre-flight benchmark (spec: enumeration <= %.0f s / %.0f GB per assay)'
          % (THRESH['G0_enum_wall_s'], THRESH['G0_enum_rss_gb']))
    for dms_id in ('GB1_IgG-Fc_fitness_1FCC', 'CR9114_FluAH1_logKd_4FQI'):
        r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        t0 = time.time()
        a = load_assay(dms_id)
        t_load = time.time() - t0
        t1 = time.time()
        ni, ac = enumerate_nested(a.keys, a.col_index)
        t_n = time.time() - t1
        t1 = time.time()
        si, pc = enumerate_samesite(a.keys, a.pos_index)
        t_s = time.time() - t1
        t1 = time.time()
        sib = sibling_counts(ni, ac, a.keys)
        t_sib = time.time() - t1
        r1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        print('[G0] %-30s load %5.2fs  nested %6.2fs (%8d)  samesite %6.2fs (%8d)  '
              'siblings %5.2fs (median %g, max %d)  enum total %6.2fs  peak RSS %.2f GB'
              % (dms_id, t_load, t_n, ni.shape[0], t_s, si.shape[0], t_sib,
                 np.median(sib), sib.max(), t_n + t_s, r1))
        assert t_n + t_s <= THRESH['G0_enum_wall_s'], 'G0 wall budget blown'
        assert r1 <= THRESH['G0_enum_rss_gb'] * 2, 'G0 RSS far over budget'
        _ = r0
    print('[pairs] SELF-CHECK PASSED')


if __name__ == '__main__':
    _selfcheck()
