# -*- coding: utf-8 -*-
"""C5 -- cliff-aware evaluation: does any CPU-only model get a cliff's DIRECTION
right?  (spec Sec.1.6, the FOURTH way this study returns a negative.)

The statistic
-------------
For a catalogued cliff edge ``p = (B, B u {i})`` of ``P_a``,

    ``PSA(tau) = mean over {p : |c_hat_p| >= tau} of 1[sign(dy_p) == sign(dyhat_p)]``

with ``dy = y_v - y_u`` (the MEASURED step, ``v`` the larger set) and
``dyhat = yhat_v - yhat_u`` (the MODEL's step).  Ties in ``dyhat`` count 0.5, so
chance is exactly 0.5 whatever the model's tie mass.  ``AUPSA`` is the mean over
the ``tau`` sweep.  Reported beside it: ``PSA`` on the NON-cliff nested edges of
the same ``P_a`` (``|c_hat| < tau``) and the per-assay Spearman over all rows in
``bindinggym_metrics.py``'s own dialect.

Three CPU-only models (the ProteinMPNN OOF arm is out of scope -- it needs a GPU
and ``diagnostics/oof/`` is not on this branch):

* **M1** ``additive_isotonic`` -- the fit already cached in ``latent/``:
  ``yhat = g(X beta)``.  Nothing is refitted here.
* **M2** ``physchem`` -- OLS of ``y`` on the per-variant SUM of the five
  per-mutation features ``[BLOSUM62, d_hydrophobicity, d_volume, rsa_iso,
  iface]``.
* **M3** ``msa_site_indep`` -- site-independent MSA log-odds
  ``sum_mutations log f(mut)/f(wt)`` from ``$BINDINGGYM_INPUT/msas/*.a2m``,
  streamed in bounded-memory batches (two files in that directory are 112 MB).

THE HONEST FRAMING, which every number below has to be read with
--------------------------------------------------------------------
**All three models are background-independent by construction.**  On a nested
edge the two endpoints differ by exactly one substitution ``i``, so

* M1's ``dyhat = g(phi_B + beta_i) - g(phi_B)``, whose sign is ``sign(beta_i)``
  wherever ``g`` is strictly increasing -- i.e. M1's prediction for the edge is
  the added substitution's AVERAGE (main) effect;
* M2's ``dyhat = theta . f(i)`` and M3's ``dyhat = log f(mut_i)/f(wt_i)`` --
  both functions of the substitution alone, with no reference to ``B`` at all.

So ``PSA_cliff`` does not ask "is this model any good"; it asks **whether the
added substitution's context-free effect predicts the sign of its effect in this
particular background**, on exactly the edges where the background-relative
deviation ``c_hat`` is large.  That is the blind spot C5 is about, and M1 is the
sharpest form of the question because it is fitted ON THIS ASSAY: if the
in-sample additive fit still cannot call the direction, no context-free model
can.  Symmetrically, ``PSA_cliff(M1) >= 0.75`` would mean a purely additive
model already gets cliff directions right, and C2 -- however statistically
sound -- would be practically empty (spec Sec.1.6 / Sec.7 item 10).

Decisions taken here, and why
-----------------------------
* the cliff catalogue is the ORCHESTRATOR-D2 **phi-centred** ``c_hat`` on
  **nested** pairs of ``P_a`` (D1), in ``sigma`` units -- the same edges T06 and
  ``cliff_catalogue_*`` call cliffs.  The uncentred and MAD-unit catalogues are
  computed as sensitivity variants, never as the primary.
* an edge with ``dy == 0`` has no direction to predict and is dropped from the
  PSA denominator (measured: **zero** such edges at every ``tau`` on all 14
  assays, so this is a guard, not a filter).  An edge whose model prediction is
  UNDEFINED (no MSA column, no structural annotation) is dropped as well and
  counted separately -- "cannot be applied" is not the same as "predicts no
  change", and folding the former into the 0.5 tie credit would drag PSA toward
  chance and manufacture a blind spot.
* CIs are the study's mandatory block bootstrap over **mutated positions**
  (``THRESH['C2_block_bootstrap_B']`` resamples, seeded from
  ``SEEDS['bootstrap_block']``), never over edges.

Runs on the 14 PRIMARY+ARM assays (spec Sec.5 stage 7).  Writes
``local-records/bindingGYM-cliff/artifacts/T12_cliff_aware_eval.csv`` with the
spec's exact 15 columns; ``verdict_blindspot`` / ``verdict_practical_emptiness``
are left EMPTY -- they are ``verdict.py``'s write-back columns.
"""
from __future__ import print_function

import itertools
import json
import math
import os
import sys

# ---- BLAS threads: 1, and it has to be set BEFORE numpy is imported ------- #
# Same reasoning (and the same tuple) as ``nulls.THREAD_ENV``, which is the
# source of truth; the assertion below keeps the duplicate honest.  This module
# is single-process, but its OLS/interp kernels would still build an 80-thread
# OpenBLAS pool per call for no gain.
_THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
for _v in _THREAD_ENV:
    os.environ.setdefault(_v, '1')

import numpy as np
import pandas as pd

from . import config
from . import io_bgym
from . import latent as _latent
from . import nulls
from .config import PATHS, SEEDS, TAUS, THRESH
from .latent import g_apply

assert set(_THREAD_ENV) == set(nulls.THREAD_ENV), \
    'stats_c5._THREAD_ENV drifted from nulls.THREAD_ENV'

__all__ = [
    'C5_MODELS', 'T12_COLUMNS', 'KD_HYDROPATHY', 'AA_VOLUME', 'blosum62',
    'msa_path', 'msa_counts', 'msa_chain_offsets', 'msa_log_odds',
    'wt_letters', 'site_features', 'mutation_feature_matrix',
    'predict_M1', 'predict_M2', 'predict_M3', 'model_predictions',
    'pa_edges', 'psa_point', 'block_bootstrap_psa', 'spearman_bgym',
    'rmse_calibrated', 'psa_table', 'run_assay', 'build_T12', 'family_summary',
    'stage7', 'run_all', 'run',
]

#: The model enum, verbatim from spec Sec.6's T12 column list and from
#: ``verdict.C5_MODELS`` (which reads this table).  Order matters: ``verdict.py``
#: takes ``spearman_all_rows`` from the FIRST row it finds for an assay, so M1 --
#: the model the practical-emptiness clause is about -- is written first.
C5_MODELS = ('M1_additive_isotonic', 'M2_physchem', 'M3_msa_site_indep')

#: spec Sec.6 "T12 cliff_aware_eval" -- EXACTLY these columns, in this order.
T12_COLUMNS = ['DMS_id', 'model', 'tau', 'n_cliff_edges', 'PSA_cliff',
               'PSA_lo95', 'PSA_hi95', 'PSA_noncliff', 'AUPSA',
               'spearman_all_rows', 'rmse_all', 'rmse_cliff', 'n_pred_ties',
               'verdict_blindspot', 'verdict_practical_emptiness']

#: Columns ``verdict.py`` owns and writes back; emitted empty here.
_VERDICT_COLUMNS = ('verdict_blindspot', 'verdict_practical_emptiness')

#: Kyte & Doolittle 1982 hydropathy index (J Mol Biol 157:105).  A physical
#: constant of the 20 amino acids, not a threshold, hence not in ``config``.
KD_HYDROPATHY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
    'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}

#: Residue volumes in A^3 (Zamyatnin 1972, Prog Biophys Mol Biol 24:107).
AA_VOLUME = {
    'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5, 'Q': 143.8,
    'E': 138.4, 'G': 60.1, 'H': 153.2, 'I': 166.7, 'L': 166.7, 'K': 168.6,
    'M': 162.9, 'F': 189.9, 'P': 112.7, 'S': 89.0, 'T': 116.1, 'W': 227.8,
    'Y': 193.6, 'V': 140.0,
}

#: M2's five features, in the spec's own order.
M2_FEATURES = ('blosum62', 'd_hydrophobicity', 'd_volume', 'rsa_iso', 'iface')

#: The two interface definitions ORCHESTRATOR D6 makes co-primary.
IFACE_COLUMNS = ('is_iface_5A', 'is_iface_dsasa')

_BLOSUM = {}


def blosum62():
    """``{(wt, mut): score}`` over the 20 canonical amino acids (biopython)."""
    if not _BLOSUM:
        from Bio.Align import substitution_matrices
        m = substitution_matrices.load('BLOSUM62')
        for a in config.AA20:
            for b in config.AA20:
                _BLOSUM[(a, b)] = float(m[a][b])
    return _BLOSUM


# =========================================================================== #
# M3 -- the MSA site-independent model                                        #
# =========================================================================== #

#: Batch size for the streaming a2m reader: records, not bytes.  8,192 x 750
#: columns is 6 MB, so peak RSS is bounded by the batch and the (L, 256) count
#: matrix regardless of the file -- ``msas/`` holds two 112 MB alignments.
_MSA_BATCH = 8192

_MSA_CACHE = {}


def msa_path(dms_id):
    """``$BINDINGGYM_INPUT/msas/{pdb stem}.a2m`` -- the same stem as the assay's
    ``pdb_file``, which is how the benchmark names its alignments (verified on
    all 22 files present).  Returns ``None`` when the assay has none: the two
    ARM hypercubes (CR9114-H1, CR6261) and CR9114-H3 are unregistered, have no
    PDB in ``structures/`` and no a2m in ``msas/`` either."""
    stem = os.path.splitext(config.ASSAYS[dms_id].pdb_file)[0]
    p = os.path.join(PATHS.msas, stem + '.a2m')
    return p if os.path.exists(p) else None


def _msa_counts_full(path, *, batch=_MSA_BATCH):
    """``(query, counts20)`` with upper- and lowercase folded together.

    One pass, one ``(L, 256)`` byte accumulator, then the two case columns of
    each amino acid are added: in these alignments the case marks the focus
    region, not the residue.  Every record is the same length (verified:
    ``n_bad_len == 0`` on all 22 files), so column ``j`` is one alignment site
    for every sequence, and a column's depth is ``counts[j].sum()`` -- gaps
    (``-``, ``.``) and non-canonical letters (``X``, ``B``, ``Z``, ``*``) are
    simply not counted.

    Bounded memory: records are reduced ``batch`` at a time with one
    ``np.bincount`` per batch (``np.bincount``, never ``np.add.at``), so peak
    RSS is the batch plus the ``(L, 256)`` matrix however big the file --
    ``msas/`` holds two 112 MB alignments.  Wrapped by :func:`msa_counts`, which
    memoises so the two 6VJJ assays share one pass.
    """
    query = None
    L = 0
    n_seq = 0
    n_bad = 0
    acc = None
    buf = []
    cur = []

    def _flush(rows):
        if not rows:
            return
        arr = np.frombuffer(b''.join(rows), dtype=np.uint8).reshape(len(rows), L)
        col = np.repeat(np.arange(L, dtype=np.int64)[None, :], arr.shape[0], 0)
        flat = col.ravel() * 256 + arr.ravel().astype(np.int64)
        acc[:] += np.bincount(flat, minlength=L * 256).reshape(L, 256)

    def _start(s):
        nonlocal query, L, acc
        query = s.decode()
        L = len(s)
        acc = np.zeros((L, 256), dtype=np.int64)

    with open(path, 'rb') as fh:
        for line in fh:
            if line[:1] == b'>':
                if cur:
                    s = b''.join(cur)
                    del cur[:]
                    if query is None:
                        _start(s)
                    n_seq += 1
                    if len(s) != L:
                        n_bad += 1
                    else:
                        buf.append(s)
                        if len(buf) >= batch:
                            _flush(buf)
                            del buf[:]
                continue
            cur.append(line.strip())
    if cur:
        s = b''.join(cur)
        if query is None:
            _start(s)
        n_seq += 1
        if len(s) != L:
            n_bad += 1
        else:
            buf.append(s)
    _flush(buf)
    if query is None:
        raise ValueError('%s: no sequence records' % path)
    up = np.array([ord(c) for c in config.AA20], dtype=np.int64)
    counts = acc[:, up] + acc[:, up + 32]
    counts_upper_only = acc[:, up]
    return dict(query=query, L=L, n_seq=n_seq, n_bad_len=n_bad,
                counts=counts, counts_upper=counts_upper_only,
                n_gap=int(acc[:, ord('-')].sum() + acc[:, ord('.')].sum()))


def msa_counts(path):
    """Memoised :func:`_msa_counts_full` -- the two 6VJJ assays share one pass."""
    if path not in _MSA_CACHE:
        _MSA_CACHE[path] = _msa_counts_full(path)
    return _MSA_CACHE[path]


_WT_SEQ_CACHE = {}


def _wt_sequences(dms_id):
    """``{chain: sequence}`` from ``BindingGYM.csv``, else from the DMS csv's own
    ``wildtype_sequence`` column (the 3 unregistered files)."""
    if dms_id in _WT_SEQ_CACHE:
        return _WT_SEQ_CACHE[dms_id]
    out = None
    try:
        reg = pd.read_csv(PATHS.registry_csv).set_index('DMS_id')
        if dms_id in reg.index:
            out = io_bgym._parse_dict_str(str(reg.loc[dms_id, 'wildtype_sequence']))
    except (OSError, KeyError, ValueError):
        out = None
    if out is None:
        col = pd.read_csv(PATHS.dms_csv(dms_id), usecols=['wildtype_sequence'],
                          dtype=str)['wildtype_sequence'].values
        out = io_bgym._parse_dict_str(str(pd.unique(col)[0]))
    _WT_SEQ_CACHE[dms_id] = out
    return out


def msa_chain_offsets(dms_id, query):
    """``{chain: offset}`` locating each chain inside the alignment's query row.

    The a2m query is the concatenation of SOME ordered subset of the assay's
    chains: one chain for 10 of the 12 (GB1 -> C, KRAS -> A/B/R, SARS2-RBD -> E,
    hYAP65 -> A, CD19 -> C) and two for the other two (5A12_VEGF -> H+L,
    Z-ZpA963 -> A+B).  Resolved by EXACT case-insensitive match against the
    registry's ``wildtype_sequence`` over ordered subsets whose total length is
    ``len(query)`` -- never by alignment, and never by a constant offset (spec
    Sec.3 ``map_mutations``).  Raises when nothing matches.
    """
    wt = _wt_sequences(dms_id)
    qu = query.upper()
    chains = sorted(wt)
    for r in range(1, len(chains) + 1):
        for perm in itertools.permutations(chains, r):
            if sum(len(wt[c]) for c in perm) != len(query):
                continue
            if ''.join(wt[c] for c in perm).upper() == qu:
                off, k = {}, 0
                for c in perm:
                    off[c] = k
                    k += len(wt[c])
                return off, perm
    raise RuntimeError('%s: no ordered chain subset of %r reproduces the a2m '
                       'query (len %d)' % (dms_id, chains, len(query)))


def msa_log_odds(dms_id, pos_index, wt_of, *, pseudocount=1.0,
                 uppercase_only=False, verbose=False):
    """Site-independent log-odds ``log f(mut)/f(wt)`` per mutated site.

    Returns ``(lo, meta)`` with ``lo[(chain, seq_pos)]`` a ``(20,)`` array over
    ``config.AA20`` (``nan`` for a site with no alignment column) and ``meta``
    carrying the coverage and depth numbers the record has to quote.

    ``f`` is the column's UNWEIGHTED amino-acid frequency with a Laplace
    pseudocount: ``f(a) = (n_a + alpha)/(sum_b n_b + 20 alpha)``.  Sequence
    reweighting (ProteinGym's ``theta = 0.2``) is NOT applied -- the spec does
    not specify it and an exact computation is ``O(N^2 L)`` on a 260,949-sequence
    alignment.  The WT letter at every mutated site is asserted against the
    ``mutant`` column's own letter, and that assertion passes 100% on all 12
    assays that have an alignment.
    """
    p = msa_path(dms_id)
    meta = dict(dms_id=dms_id, msa=None, n_seq=0, L=0, n_sites=len(pos_index),
                n_sites_mapped=0, n_sites_upper=0, depth_min=float('nan'),
                depth_median=float('nan'), n_wt_mismatch=0, chain_order='',
                pseudocount=float(pseudocount), uppercase_only=bool(uppercase_only))
    if p is None:
        meta['note'] = 'no a2m in msas/ for %s' % config.ASSAYS[dms_id].pdb_file
        return {}, meta
    c = msa_counts(p)
    query = c['query']
    counts = c['counts_upper'] if uppercase_only else c['counts']
    off, perm = msa_chain_offsets(dms_id, query)
    meta.update(msa=os.path.basename(p), n_seq=c['n_seq'], L=c['L'],
                chain_order=''.join(perm), n_bad_len=c['n_bad_len'])
    aa_ix = {a: i for i, a in enumerate(config.AA20)}
    lo, depths = {}, []
    n_up = 0
    n_bad = 0
    for (chain, ps) in sorted(pos_index):
        if chain not in off:
            continue
        j = off[chain] + int(ps) - 1
        if not (0 <= j < c['L']):
            continue
        qa = query[j]
        wt = wt_of.get((chain, int(ps)))
        if wt is not None and qa.upper() != wt:
            n_bad += 1
            continue
        if uppercase_only and not qa.isupper():
            continue
        n_up += int(qa.isupper())
        n = counts[j].astype(np.float64)
        tot = n.sum()
        if tot <= 0:
            continue
        f = (n + pseudocount) / (tot + pseudocount * len(config.AA20))
        wt_letter = wt if wt is not None else qa.upper()
        if wt_letter not in aa_ix:
            continue
        lo[(chain, int(ps))] = np.log(f) - math.log(f[aa_ix[wt_letter]])
        depths.append(int(tot))
    meta.update(n_sites_mapped=len(lo), n_sites_upper=n_up, n_wt_mismatch=n_bad,
                depth_min=(min(depths) if depths else float('nan')),
                depth_median=(float(np.median(depths)) if depths else float('nan')))
    meta['note'] = ''
    if verbose:
        print('[C5:M3] %-40s %s L=%d nseq=%d chains=%s mapped=%d/%d '
              'upper=%d depth med=%s min=%s wt_mismatch=%d'
              % (dms_id, meta['msa'], meta['L'], meta['n_seq'],
                 meta['chain_order'], meta['n_sites_mapped'], meta['n_sites'],
                 meta['n_sites_upper'], meta['depth_median'], meta['depth_min'],
                 n_bad))
    return lo, meta


# =========================================================================== #
# wild-type letters and the M2 site features                                  #
# =========================================================================== #

_WT_LETTER_CACHE = {}


def wt_letters(dms_id, pos_index):
    """``{(chain, seq_pos): wt_aa}`` from the assay's own ``mutant`` column.

    One ``usecols=['mutant']`` read with early termination once every position
    of ``pos_index`` has been seen, and an assertion that a position never
    reports two different WT letters.  The csv is the primary source (it is the
    only one the two unregistered ARM assays have); :func:`site_features`
    cross-checks it against T09's ``wt_aa``.
    """
    if dms_id in _WT_LETTER_CACHE:
        return _WT_LETTER_CACHE[dms_id]
    col = pd.read_csv(PATHS.dms_csv(dms_id), usecols=['mutant'])['mutant'].values
    want = set(pos_index)
    seen = {}
    for i in range(len(col)):
        d = io_bgym._parse_dict_str(col[i])
        for chain, v in d.items():
            if not v:
                continue
            for tok in v.split(':'):
                wt, pos, ic, _mut = io_bgym.parse_mut_token(tok)
                k = (chain, pos)
                if k in seen:
                    if seen[k] != wt:
                        raise ValueError('%s: (%s,%d) WT letter %r then %r'
                                         % (dms_id, chain, pos, seen[k], wt))
                else:
                    seen[k] = wt
        if len(seen) >= len(want) and want <= set(seen):
            break
    missing = want - set(seen)
    if missing:
        raise ValueError('%s: no WT letter for %d positions (e.g. %r)'
                         % (dms_id, len(missing), sorted(missing)[:3]))
    out = {k: seen[k] for k in want}
    _WT_LETTER_CACHE[dms_id] = out
    return out


_T09 = {}


def _t09():
    """T09 as ``{(DMS_id, chain, seq_idx): row}``, read once."""
    if not _T09:
        p = os.path.join(PATHS.artifacts, 'T09_structure_sites.csv')
        _T09['df'] = pd.read_csv(p) if os.path.exists(p) else None
    return _T09['df']


def site_features(dms_id, pos_index, wt_of):
    """``(feat, meta)`` -- the per-site half of M2.

    ``feat[(chain, seq_pos)] = (rsa_iso, is_iface_5A, is_iface_dsasa)``, read
    from T09 (stage 1's structural annotation).  Sites with no annotation, and
    assays with none at all (CR9114-H1, CR6261: unregistered, no PDB), are
    reported rather than imputed -- M2 then runs on its three
    substitution-only features and says so.
    """
    meta = dict(dms_id=dms_id, n_sites=len(pos_index), n_annotated=0,
                structural=False, n_wt_disagree=0, note='')
    t09 = _t09()
    out = {}
    if t09 is None:
        meta['note'] = 'T09_structure_sites.csv absent'
        return out, meta
    sub = t09[t09['DMS_id'].astype(str) == dms_id]
    if len(sub) == 0:
        meta['note'] = 'no T09 rows (assay is structurally mute)'
        return out, meta
    bad = 0
    for _, r in sub.iterrows():
        k = (str(r['chain']), int(r['seq_idx']))
        if k not in pos_index:
            continue
        rsa = float(r['rsa_iso']) if pd.notna(r['rsa_iso']) else float('nan')
        if not np.isfinite(rsa):
            continue
        w = wt_of.get(k)
        if w is not None and str(r['wt_aa']) != w:
            bad += 1
            continue
        out[k] = (rsa,
                  1.0 if bool(r['is_iface_5A']) else 0.0,
                  1.0 if bool(r['is_iface_dsasa']) else 0.0)
    meta.update(n_annotated=len(out), structural=len(out) > 0, n_wt_disagree=bad)
    if bad:
        raise ValueError('%s: T09 wt_aa disagrees with the mutant column at %d '
                         'sites' % (dms_id, bad))
    return out, meta


def mutation_feature_matrix(dms_id, col_index, pos_index, *, iface='is_iface_5A',
                            verbose=False):
    """``(F, ok, meta)`` -- M2's per-X-column feature matrix.

    ``F`` is ``(M, 5)`` over :data:`M2_FEATURES` for the substitution owning each
    column of ``X``; ``ok`` is the ``(M,)`` mask of columns whose features are
    all defined.  A variant's feature row is then ``X @ F`` -- i.e. M2 is
    additive over the variant's substitutions, exactly like M1, which is what
    makes its nested-edge prediction a function of the added substitution alone.
    """
    if iface not in IFACE_COLUMNS:
        raise ValueError('iface must be one of %r' % (IFACE_COLUMNS,))
    wt_of = wt_letters(dms_id, pos_index)
    feat, fmeta = site_features(dms_id, pos_index, wt_of)
    bl = blosum62()
    M = len(col_index)
    F = np.full((M, len(M2_FEATURES)), np.nan, dtype=np.float64)
    ok = np.zeros(M, dtype=bool)
    n_struct = 0
    for (chain, ps, mut), c in col_index.items():
        wt = wt_of[(chain, int(ps))]
        if wt not in KD_HYDROPATHY or mut not in KD_HYDROPATHY:
            continue                                    # X / non-canonical WT
        row = [bl[(wt, mut)],
               KD_HYDROPATHY[mut] - KD_HYDROPATHY[wt],
               AA_VOLUME[mut] - AA_VOLUME[wt]]
        s = feat.get((chain, int(ps)))
        if s is None:
            row += [np.nan, np.nan]
        else:
            n_struct += 1
            row += [s[0], s[1] if iface == 'is_iface_5A' else s[2]]
        F[c] = row
        ok[c] = True
    used = list(M2_FEATURES)
    if not np.isfinite(F[ok][:, 3:]).all():
        # no (or partial) structural annotation -> drop the two site features
        # for the WHOLE assay rather than impute a burial for some columns and
        # not others, and say so in the note
        F = F[:, :3]
        used = list(M2_FEATURES[:3])
    meta = dict(fmeta, iface_col=iface, n_cols=M, n_cols_ok=int(ok.sum()),
                n_cols_structural=n_struct, features=','.join(used))
    if verbose:
        print('[C5:M2] %-40s cols=%d ok=%d feats=%s (%s)'
              % (dms_id, M, int(ok.sum()), meta['features'], fmeta['note'] or 'T09 ok'))
    return F, ok, meta


# =========================================================================== #
# the three models -- per-row predictions on the y scale (or a monotone proxy) #
# =========================================================================== #

def predict_M1(ctx, *, oof=False, link=True):
    """M1: ``yhat = g(phi)`` from the CACHED additive-isotonic fit.

    ``oof=True`` swaps ``phi`` for the cached cross-fitted ``phi_oof`` (the
    link stays the full fit's -- a fold's own ``g`` is not cached, and a monotone
    link can only change ties, never a sign or a rank).  ``link=False`` returns
    ``phi`` itself, the tie-free monotone proxy.
    """
    phi = ctx.phi_oof if oof else ctx.phi
    if not link:
        return np.asarray(phi, dtype=np.float64)
    return g_apply(ctx.g_knots, phi)


def _ols_fit(A, y):
    """Least squares with an intercept prepended; returns coefficients."""
    beta, _res, _rk, _sv = np.linalg.lstsq(A, y, rcond=None)
    return beta


def predict_M2(ctx, F, ok, *, oof=False):
    """M2: OLS of ``y`` on the per-variant sum of the per-mutation features.

    Rows carrying a column whose features are undefined get ``nan`` (the model
    cannot be applied), never an imputed 0.  ``oof=True`` refits on the cached
    5-fold partition and predicts each fold from the other four.
    """
    n = ctx.n
    Xc = ctx.X.tocsc()
    bad = np.zeros(n, dtype=bool)
    if (~ok).any():
        bad = np.asarray((ctx.X[:, ~ok]).sum(axis=1)).ravel() > 0
    Z = np.asarray((ctx.X @ np.nan_to_num(F, nan=0.0)), dtype=np.float64)
    A = np.column_stack([np.ones(n), Z])
    y = ctx.y
    fit_ok = ~bad & np.isfinite(y)
    out = np.full(n, np.nan, dtype=np.float64)
    if not oof:
        b = _ols_fit(A[fit_ok], y[fit_ok])
        out[~bad] = A[~bad] @ b
        return out
    folds = np.asarray(ctx.folds)
    for f in np.unique(folds):
        tr = fit_ok & (folds != f)
        te = (~bad) & (folds == f)
        if tr.sum() <= A.shape[1] or not te.any():
            continue
        b = _ols_fit(A[tr], y[tr])
        out[te] = A[te] @ b
    _ = Xc
    return out


def predict_M3(ctx, col_index, lo):
    """M3: the per-variant sum of site-independent MSA log-odds.

    A variant carrying a substitution at a site with no alignment column is
    ``nan``: the model has no opinion there, which is not the same as predicting
    no change.
    """
    M = len(col_index)
    w = np.full(M, np.nan, dtype=np.float64)
    aa_ix = {a: i for i, a in enumerate(config.AA20)}
    for (chain, ps, mut), c in col_index.items():
        v = lo.get((chain, int(ps)))
        if v is None or mut not in aa_ix:
            continue
        w[c] = v[aa_ix[mut]]
    okc = np.isfinite(w)
    out = np.asarray(ctx.X @ np.nan_to_num(w, nan=0.0), dtype=np.float64).ravel()
    if (~okc).any():
        bad = np.asarray((ctx.X[:, ~okc]).sum(axis=1)).ravel() > 0
        out[bad] = np.nan
    return out, int(okc.sum()), M


def model_predictions(dms_id, ctx, *, variant='primary', verbose=False):
    """``{model: yhat}`` plus a metadata dict, for one variant of the sweep.

    Variants: ``primary`` (M1 in-sample with the isotonic link, M2 in-sample,
    5 features, ``is_iface_5A``), ``oof`` (M1 and M2 cross-fitted; M3 is y-blind
    so it is unchanged), ``nolink`` (M1 on ``phi``, no link ties),
    ``iface_dsasa`` (M2's interface feature switched to the dSASA/Levy
    definition, ORCHESTRATOR D6) and ``msa_upper`` (M3 restricted to the a2m's
    uppercase focus columns).
    """
    des = _latent.load_cached_design(dms_id, verify=False)
    col_index, pos_index = des['col_index'], des['pos_index']
    iface = 'is_iface_dsasa' if variant == 'iface_dsasa' else 'is_iface_5A'
    F, ok, m2meta = mutation_feature_matrix(dms_id, col_index, pos_index,
                                            iface=iface, verbose=verbose)
    wt_of = wt_letters(dms_id, pos_index)
    lo, m3meta = msa_log_odds(dms_id, pos_index, wt_of,
                              uppercase_only=(variant == 'msa_upper'),
                              verbose=verbose)
    oof = (variant == 'oof')
    p1 = predict_M1(ctx, oof=oof, link=(variant != 'nolink'))
    p2 = predict_M2(ctx, F, ok, oof=oof)
    p3, n_ok3, n_col3 = predict_M3(ctx, col_index, lo)
    meta = dict(variant=variant, M2=m2meta, M3=dict(m3meta, n_cols_ok=n_ok3,
                                                    n_cols=n_col3))
    return {C5_MODELS[0]: p1, C5_MODELS[1]: p2, C5_MODELS[2]: p3}, meta


# =========================================================================== #
# PSA                                                                         #
# =========================================================================== #

def pa_edges(ctx, *, centred=True, unit='sigma'):
    """The ``P_a`` edge set with its cliff statistic (ORCHESTRATOR D1 + D2).

    Returns a dict with ``idx`` (the surviving nested pairs, column 0 = the
    smaller set), ``c`` (``c_hat``, ``sigma`` or MAD units), ``pos`` (the added
    substitution's position index, for the block bootstrap), ``dy`` and
    ``n_Pa``.
    """
    if unit not in ('sigma', 'mad'):
        raise ValueError('unit must be sigma or mad')
    keep = nulls._pa_mask(ctx, ctx.censor_mask, ctx.oof_finite)
    sub = ctx.nested_idx[keep]
    mu = ctx.mu_oof if centred else None
    c = nulls.c_hat(ctx.e_oof, ctx.sigma_oof, sub, mu=mu)
    if unit == 'mad':
        e = ctx.e_oof - (ctx.mu_oof if centred else 0.0)
        num = e[sub[:, 1]] - e[sub[:, 0]]
        s = _latent.mad_scaled(num[np.isfinite(num)])
        c = num / s if s > 0 else np.full(num.shape, np.nan)
    e = ctx.e_oof - (ctx.mu_oof if centred else 0.0)
    num = e[ctx.nested_idx[keep][:, 1]] - e[ctx.nested_idx[keep][:, 0]]
    good = np.isfinite(c)
    sub = sub[good]
    add_col = ctx.add_col[keep][good]
    return dict(idx=sub, c=c[good], pos=ctx.pos_of_add[keep][good],
                add_col=add_col, num=num[good],
                beta_add=np.asarray(ctx.beta[1:], dtype=np.float64)[add_col],
                dy=ctx.y[sub[:, 1]] - ctx.y[sub[:, 0]],
                n_Pa=int(keep.sum()), n_finite=int(good.sum()))


def psa_point(dy, dyhat):
    """``(PSA, n_scored, n_pred_ties, n_dy_zero, n_undefined, hits)``.

    ``hits`` is per edge: 1 for a sign match, 0.5 for a tie in ``dyhat``, 0 for
    a mismatch -- so chance is exactly 0.5.  Edges with ``dy == 0`` (no direction
    to predict) or an undefined ``dyhat`` are excluded and counted.
    """
    dy = np.asarray(dy, dtype=np.float64)
    dyhat = np.asarray(dyhat, dtype=np.float64)
    defined = np.isfinite(dyhat) & np.isfinite(dy)
    n_undef = int((~defined).sum())
    zero = defined & (dy == 0)
    use = defined & (dy != 0)
    hits = np.where(dyhat[use] == 0, 0.5,
                    (np.sign(dy[use]) == np.sign(dyhat[use])).astype(np.float64))
    psa = float(hits.mean()) if hits.size else float('nan')
    return (psa, int(use.sum()), int((dyhat[use] == 0).sum()), int(zero.sum()),
            n_undef, hits, use)


def block_bootstrap_psa(hits, pos, dms_id, *, B=None, seed_name='bootstrap_block'):
    """95% CI for PSA by block bootstrap over MUTATED POSITIONS (the study's
    ground rule: never over edges).

    The position set resampled is the set of added-substitution positions that
    appear in ``P_a``; a position drawn twice contributes its edges twice, which
    is what makes this the block bootstrap and not a subsample.  Implemented as
    one multinomial draw of the position multiplicities per replicate and a
    weighted mean -- ``np.bincount``, never ``np.add.at``.
    """
    if B is None:
        B = THRESH['C2_block_bootstrap_B']
    hits = np.asarray(hits, dtype=np.float64)
    if hits.size == 0:
        return float('nan'), float('nan'), 0, 0
    u, gi = np.unique(np.asarray(pos), return_inverse=True)
    npos = u.size
    S = np.bincount(gi, weights=hits, minlength=npos)
    N = np.bincount(gi, minlength=npos).astype(np.float64)
    rng = np.random.default_rng(config.assay_seed(seed_name, dms_id))
    W = rng.multinomial(npos, np.full(npos, 1.0 / npos), size=int(B)).astype(np.float64)
    num = W @ S
    den = W @ N
    with np.errstate(invalid='ignore', divide='ignore'):
        vals = num / den
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float('nan'), float('nan'), npos, 0
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), npos, int(vals.size)


def spearman_bgym(y, pred):
    """Per-assay Spearman in ``bindinggym_metrics.py``'s dialect, verbatim:
    ``df[label].rank().corr(df[pred].rank())`` -- Pearson of the average-tie
    ranks, on the rows where both are finite.  ``pred`` is oriented larger =
    binds tighter, the same as ``DMS_score``; all three models predict on that
    orientation already, so nothing is negated here (contrast
    ``bindinggym_metrics.evaluate_oof``, which negates a ddG head).
    """
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(pred)
    if m.sum() < 3:
        return float('nan'), int(m.sum())
    a, b = pd.Series(y[m]), pd.Series(pred[m])
    return float(a.rank().corr(b.rank())), int(m.sum())


def rmse_calibrated(y, pred, *, subset=None):
    """The repo's own RMSE dialect (``utils.overall_rmse_mae``, mirrored in
    ``bindinggym_metrics._rmse_calibrated``): fit ``y ~ a + b*pred`` by OLS on
    the finite rows and take the residual RMSE.

    Affine-invariant, which is the only defensible choice here: M3's log-odds
    and M2's 5-feature score are not on the assay's y scale at all, so a raw
    ``sqrt(mean((pred-y)^2))`` would report a unit mismatch as model error.
    ``subset`` evaluates the SAME fitted map on a sub-population (used for
    ``rmse_cliff``), so the two numbers are comparable.
    """
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(pred)
    if m.sum() < 3:
        return float('nan'), 0
    A = np.column_stack([np.ones(int(m.sum())), pred[m]])
    b = _ols_fit(A, y[m])
    ev = m if subset is None else (m & np.asarray(subset, dtype=bool))
    if ev.sum() == 0:
        return float('nan'), 0
    r = y[ev] - (b[0] + b[1] * pred[ev])
    return float(np.sqrt((r * r).mean())), int(ev.sum())


# =========================================================================== #
# per assay                                                                   #
# =========================================================================== #

#: The sweep of variants.  ``primary`` is the only one T12 carries.
VARIANTS = ('primary', 'oof', 'nolink', 'iface_dsasa', 'msa_upper',
            'uncentred', 'mad_unit')

_CATALOGUE_VARIANT = {'uncentred': dict(centred=False, unit='sigma'),
                      'mad_unit': dict(centred=True, unit='mad')}


def run_assay(dms_id, *, variants=('primary',), bootstrap=True, verbose=True):
    """Every C5 number for one assay, as a long DataFrame (one row per
    ``(variant, model, tau)``)."""
    ctx = nulls.build_context(dms_id, verify=False)
    rows = []
    meta_all = {}
    for variant in variants:
        cat = _CATALOGUE_VARIANT.get(variant, dict(centred=True, unit='sigma'))
        ed = pa_edges(ctx, **cat)
        pv = 'primary' if variant in _CATALOGUE_VARIANT else variant
        preds, meta = model_predictions(dms_id, ctx, variant=pv, verbose=verbose)
        meta_all[variant] = meta
        gg = nulls._grid_guard_taus(ctx, cat['unit'])
        ac = np.abs(ed['c'])
        u, v = ed['idx'][:, 0], ed['idx'][:, 1]
        # WHY a PSA comes out where it does, per edge set.  For a nested pair
        # Delta z = num + beta_add exactly (spec Sec.1.0), so
        # sign(dy) == sign(beta_add) -- which is M1's prediction -- iff either
        # the two agree in sign or |beta_add| > |num|.  Both fractions are
        # properties of the CATALOGUE and the additive fit, not of a model, so
        # they are repeated on every model's row.
        dom = np.abs(ed['beta_add']) > np.abs(ed['num'])
        agr = np.sign(ed['num']) == np.sign(ed['beta_add'])
        for model in C5_MODELS:
            yhat = preds[model]
            dyhat_all = yhat[v] - yhat[u]
            sp, n_sp = spearman_bgym(ctx.y, yhat)
            rm_all, n_rm = rmse_calibrated(ctx.y, yhat)
            per_tau = []
            for t in TAUS:
                cliff = ac >= t
                inc = np.zeros(ctx.n, dtype=bool)
                inc[u[cliff]] = True
                inc[v[cliff]] = True
                rm_c, n_rc = rmse_calibrated(ctx.y, yhat, subset=inc)
                psa, n_sc, n_tie, n_z, n_ud, hits, use = psa_point(
                    ed['dy'][cliff], dyhat_all[cliff])
                psa_nc = psa_point(ed['dy'][~cliff], dyhat_all[~cliff])
                lo = hi = float('nan')
                npos = nboot = 0
                if bootstrap and n_sc > 0:
                    lo, hi, npos, nboot = block_bootstrap_psa(
                        hits, ed['pos'][cliff][use], dms_id)
                r = dict(DMS_id=dms_id, family_id=ctx.family_id, tier=ctx.tier,
                         variant=variant, unit=cat['unit'],
                         centred=cat['centred'], model=model, tau=t,
                         grid_guard_pass=bool(gg[t]),
                         n_Pa=ed['n_Pa'], n_cliff_catalogued=int(cliff.sum()),
                         n_cliff_edges=n_sc, n_pred_ties=n_tie,
                         n_dy_zero=n_z, n_undefined=n_ud,
                         PSA_cliff=psa, PSA_lo95=lo, PSA_hi95=hi,
                         n_boot_pos=npos, n_boot_ok=nboot,
                         PSA_noncliff=psa_nc[0], n_noncliff_edges=psa_nc[1],
                         n_noncliff_ties=psa_nc[2],
                         spearman_all_rows=sp, n_rows_scored=n_sp,
                         rmse_all=rm_all, rmse_cliff=rm_c, n_rows_cliff=n_rc,
                         frac_dy_neg=(float((ed['dy'][cliff] < 0).mean())
                                      if cliff.any() else float('nan')),
                         frac_dyhat_neg=(
                             float((dyhat_all[cliff][use] < 0).mean())
                             if n_sc else float('nan')),
                         frac_beta_dominates=(float(dom[cliff].mean())
                                              if cliff.any() else float('nan')),
                         frac_sign_agree_num_beta=(float(agr[cliff].mean())
                                                   if cliff.any() else float('nan')))
                per_tau.append(r)
            live = [r for r in per_tau if r['grid_guard_pass']
                    and r['n_cliff_edges'] > 0 and np.isfinite(r['PSA_cliff'])]
            aupsa = float(np.mean([r['PSA_cliff'] for r in live])) if live else float('nan')
            wt = float(np.sum([r['PSA_cliff'] * r['n_cliff_edges'] for r in live])
                       / np.sum([r['n_cliff_edges'] for r in live])) if live else float('nan')
            for r in per_tau:
                r['AUPSA'] = aupsa
                r['AUPSA_n_tau'] = len(live)
                r['AUPSA_edge_weighted'] = wt
            rows += per_tau
        if verbose:
            p = [r for r in rows if r['variant'] == variant and r['tau'] == 3]
            print('[C5] %-40s %-11s n_Pa=%7d  '
                  % (dms_id, variant, ed['n_Pa'])
                  + '  '.join('%s AUPSA=%.3f PSA(3)=%s rho=%.3f'
                              % (r['model'].split('_')[0], r['AUPSA'],
                                 ('%.3f' % r['PSA_cliff']) if np.isfinite(r['PSA_cliff']) else 'na',
                                 r['spearman_all_rows']) for r in p))
    out = pd.DataFrame(rows)
    out.attrs['meta'] = meta_all
    return out


# =========================================================================== #
# T12                                                                         #
# =========================================================================== #

def build_T12(assays=None, *, write=True, variants=VARIANTS, verbose=True):
    """Run C5 on the 14 PRIMARY+ARM assays and write T12.

    Returns ``(t12, long)``: the spec-shaped table and the full variant sweep.
    """
    config.assert_env()
    ids = list(assays) if assays else list(config.PRIMARY_AND_ARM)
    frames, metas = [], {}
    for a in ids:
        f = run_assay(a, variants=variants, verbose=verbose)
        metas[a] = f.attrs.get('meta', {})
        frames.append(f)
    long = pd.concat(frames, ignore_index=True)
    long.attrs['meta'] = metas
    prim = long[(long['variant'] == 'primary')].copy()
    # verdict.py reads spearman_all_rows from the FIRST row of an assay, so the
    # model order has to be M1, M2, M3 and tau ascending.
    prim['_m'] = prim['model'].map({m: i for i, m in enumerate(C5_MODELS)})
    prim['_a'] = prim['DMS_id'].map(config.ASSAY_ORDINAL)
    prim = prim.sort_values(['_a', '_m', 'tau'])
    t12 = prim.reindex(columns=T12_COLUMNS)
    for c in _VERDICT_COLUMNS:
        t12[c] = ''                       # verdict.py owns these
    if write:
        PATHS.ensure_cache_dirs()
        p = os.path.join(PATHS.artifacts, 'T12_cliff_aware_eval.csv')
        t12.to_csv(p, index=False)
        if verbose:
            print('[C5] wrote %s  (%d rows x %d cols)' % (p, len(t12), t12.shape[1]))
    return t12, long


def family_summary(long, *, spearman_from='own'):
    """The spec Sec.1.6 aggregate, per family.

    Blind spot demonstrated for an assay iff ``AUPSA <= THRESH['C5_PSA_blindspot']``
    for ALL of M1-M3 while the per-assay Spearman is ``>= THRESH['C5_spearman_min']``;
    the family call is the majority rule ``verdict.family_call`` uses.  ``K = 6``
    (ORCHESTRATOR D3: CD19/F7 is STRUCTURALLY_UNIDENTIFIED and leaves the
    aggregate denominator), and F8 (the hypercube arm) has its own denominator.

    ``spearman_from='own'`` requires each model to clear the Spearman floor on
    its own predictions; ``'M1'`` reads M1's, which is what ``verdict.py``'s
    ``_first_num`` actually picks up from the table.
    """
    T = THRESH
    prim = long[long['variant'] == 'primary']
    rows = []
    for a, sub in prim.groupby('DMS_id', sort=False):
        rec = dict(DMS_id=a, family_id=sub['family_id'].iloc[0],
                   tier=sub['tier'].iloc[0])
        psa, sp = {}, {}
        for m in C5_MODELS:
            s = sub[sub['model'] == m]
            psa[m] = float(s['AUPSA'].iloc[0]) if len(s) else float('nan')
            sp[m] = float(s['spearman_all_rows'].iloc[0]) if len(s) else float('nan')
            rec['AUPSA_' + m.split('_')[0]] = psa[m]
            rec['rho_' + m.split('_')[0]] = sp[m]
        ref = sp[C5_MODELS[0]] if spearman_from == 'M1' else None
        all_lo = all(np.isfinite(psa[m]) and psa[m] <= T['C5_PSA_blindspot']
                     for m in C5_MODELS)
        acc = (np.isfinite(ref) and ref >= T['C5_spearman_min']) if ref is not None \
            else all(np.isfinite(sp[m]) and sp[m] >= T['C5_spearman_min']
                     for m in C5_MODELS)
        n_missing = sum(1 for m in C5_MODELS if not np.isfinite(psa[m]))
        rec['blindspot'] = ('SUPPORTED' if (all_lo and acc) else
                            ('INCONCLUSIVE' if n_missing else 'REFUTED'))
        rec['practical_emptiness'] = (
            bool(psa[C5_MODELS[0]] >= T['C5_PSA_practically_empty'])
            if np.isfinite(psa[C5_MODELS[0]]) else None)
        rows.append(rec)
    per_assay = pd.DataFrame(rows)
    fam = []
    for f, sub in per_assay.groupby('family_id', sort=True):
        o = list(sub['blindspot'])
        pos = sum(1 for x in o if x == 'SUPPORTED')
        neg = sum(1 for x in o if x == 'REFUTED')
        inc = sum(1 for x in o if x == 'INCONCLUSIVE')
        call = ('SUPPORTED' if (pos > neg and pos > inc) else
                ('REFUTED' if (neg > pos and neg >= inc) else 'INCONCLUSIVE'))
        fam.append(dict(family_id=f, members=','.join(sub['DMS_id']),
                        n=len(sub), n_pos=pos, n_neg=neg, n_inc=inc,
                        family_call=call,
                        in_denominator=(f in ('F1', 'F2', 'F3', 'F4', 'F5', 'F6')),
                        practical_emptiness_any=bool(
                            sub['practical_emptiness'].fillna(False).any())))
    fam = pd.DataFrame(fam)
    den = fam[fam['in_denominator']]
    k = int((den['family_call'] == 'SUPPORTED').sum())
    agg = dict(K=int(len(den)), k=k, k_required=THRESH['C5_family_k_blindspot'],
               blindspot_demonstrated=bool(k >= THRESH['C5_family_k_blindspot']),
               spearman_from=spearman_from,
               practical_emptiness_families=','.join(
                   fam.loc[fam['practical_emptiness_any'], 'family_id']))
    per_assay.attrs['family'] = fam
    per_assay.attrs['aggregate'] = agg
    return per_assay


# =========================================================================== #
# stage entry point                                                           #
# =========================================================================== #

def stage7(assays=None, nproc=1, verbose=True):
    """spec Sec.5 stage 7: "C5: three CPU models + PSA/AUPSA x 14 (MSA read from
    ``msas/*.a2m``)".  Single-process by construction -- the whole stage is a few
    seconds of numpy plus ~240 MB of streamed alignment, so ``nproc`` is accepted
    and ignored (announced, never silently)."""
    if verbose and nproc and int(nproc) != 1:
        print('[C5] nproc=%s ignored: stage 7 is one process (the cost is the '
              'streamed a2m read, which is I/O bound)' % (nproc,))
    t12, long = build_T12(assays, write=True, verbose=verbose)
    pa = family_summary(long)
    if verbose:
        _report(t12, long, pa)
    ok = _pairs_verify()
    if verbose:
        print('[C5] pairs.verify_manifest(): %s' % (ok if ok else 'clean'))
    return t12


def run_all(assays=None, nproc=1, verbose=True):
    return stage7(assays, nproc=nproc, verbose=verbose)


def run(assays=None, nproc=1, verbose=True):
    return stage7(assays, nproc=nproc, verbose=verbose)


def _pairs_verify():
    """ORCHESTRATOR D8: nothing here writes a cache, so this only confirms that
    no other stage's entry went missing while C5 ran."""
    from . import pairs as _p
    try:
        return _p.verify_manifest()
    except Exception as exc:                                # pragma: no cover
        return ['verify_manifest raised: %r' % (exc,)]


def _report(t12, long, pa):
    T = THRESH
    prim = long[long['variant'] == 'primary']
    print('\n=== C5 headline: AUPSA per assay x model (chance 0.5, blind spot '
          '<= %.2f, practical emptiness >= %.2f for M1) ==='
          % (T['C5_PSA_blindspot'], T['C5_PSA_practically_empty']))
    piv = prim.pivot_table(index='DMS_id', columns='model', values='AUPSA',
                           aggfunc='first')
    rho = prim.pivot_table(index='DMS_id', columns='model',
                           values='spearman_all_rows', aggfunc='first')
    nc = prim[prim['tau'] == 3].pivot_table(index='DMS_id', columns='model',
                                            values='n_cliff_edges', aggfunc='first')
    tab = pd.concat([piv.add_prefix('AUPSA_'), rho.add_prefix('rho_'),
                     nc.add_prefix('n3_')], axis=1)
    tab.columns = [c.replace('_additive_isotonic', '1').replace('_physchem', '2')
                   .replace('_msa_site_indep', '3') for c in tab.columns]
    print(tab.to_string(float_format=lambda v: '%.3f' % v))
    print('\n=== family aggregate (K = 6, ORCHESTRATOR D3) ===')
    print(pa.attrs['family'].to_string(index=False))
    print(pa.attrs['aggregate'])
    print('\n=== per-assay blind-spot / practical-emptiness calls ===')
    print(pa.to_string(index=False, float_format=lambda v: '%.3f' % v))


# =========================================================================== #
# self-check                                                                  #
# =========================================================================== #

def _selfcheck(argv=()):
    """Runs on real data and prints the numbers (house rule)."""
    config.assert_env()
    ids = list(argv) if argv else list(config.PRIMARY_AND_ARM)
    print('[C5] env %s' % (config.EXPECTED_ENV,))

    # --- 1. the metric dialects, against bindinggym_metrics.py itself ------- #
    if config.REPO not in sys.path:
        sys.path.insert(0, config.REPO)
    import bindinggym_metrics as bgm
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    p = 0.6 * y + rng.normal(size=500)
    d = pd.DataFrame(dict(DMS_score=y, pred=p))
    ref = bgm.bindinggym_metrics_one_assay(d)['Spearman']
    got = spearman_bgym(y, p)[0]
    assert abs(ref - got) < 1e-12, (ref, got)
    r_ref = bgm._rmse_calibrated(y, p)[0]
    r_got = rmse_calibrated(y, p)[0]
    assert abs(r_ref - r_got) < 1e-10, (r_ref, r_got)
    print('[C5] dialect check vs bindinggym_metrics.py: Spearman %.12f == %.12f, '
          'rmse_calib %.10f == %.10f  OK' % (ref, got, r_ref, r_got))

    # --- 2. PSA is exactly 0.5 under a coin-flip prediction ---------------- #
    dy = rng.normal(size=100000)
    psa = psa_point(dy, rng.normal(size=100000))[0]
    assert abs(psa - 0.5) < 0.01, psa
    tie = psa_point(dy, np.zeros(100000))[0]
    assert tie == 0.5, tie
    perf = psa_point(dy, dy)[0]
    assert perf == 1.0, perf
    print('[C5] PSA calibration: random %.4f, all-ties %.1f, oracle %.1f  OK'
          % (psa, tie, perf))

    # --- 3. M1's nested-edge prediction IS the added beta ------------------ #
    a = 'GB1_IgG-Fc_fitness_1FCC'
    ctx = nulls.build_context(a, verify=False)
    ed = pa_edges(ctx)
    yh = predict_M1(ctx, link=False)
    dphi = yh[ed['idx'][:, 1]] - yh[ed['idx'][:, 0]]
    b = ctx.beta[1:][ed['add_col']]
    assert np.abs(dphi - b).max() < 1e-9, np.abs(dphi - b).max()
    print('[C5] M1 identity on %s: max|dphi - beta_added| = %.2e over %d edges'
          % (a, np.abs(dphi - b).max(), dphi.size))

    # --- 4. the full sweep ------------------------------------------------- #
    t12, long = build_T12(ids, write=True, verbose=True)
    pa = family_summary(long)
    _report(t12, long, pa)

    print('\n=== T12 shape / columns ===')
    print(list(t12.columns))
    assert list(t12.columns) == T12_COLUMNS, list(t12.columns)
    print('%d rows, %d assays x %d models x %d tau'
          % (len(t12), t12['DMS_id'].nunique(), t12['model'].nunique(),
             t12['tau'].nunique()))

    print('\n=== sensitivity: AUPSA by variant (M1 / M2 / M3) ===')
    sv = long.pivot_table(index='variant', columns='model', values='AUPSA',
                          aggfunc='mean')
    print(sv.to_string(float_format=lambda v: '%.4f' % v))
    print('\n=== per-assay AUPSA by variant, M1 ===')
    print(long[long['model'] == C5_MODELS[0]]
          .pivot_table(index='DMS_id', columns='variant', values='AUPSA',
                       aggfunc='first')
          .to_string(float_format=lambda v: '%.3f' % v))

    print('\n=== M3 MSA provenance ===')
    m = long.attrs['meta']
    for a2 in ids:
        mm = m.get(a2, {}).get('primary', {}).get('M3', {})
        m2 = m.get(a2, {}).get('primary', {}).get('M2', {})
        print('%-40s msa=%-16s nseq=%-7s chains=%-3s mapped=%s/%s depth_med=%s '
              'cols=%s/%s | M2 feats=%s'
              % (a2, mm.get('msa'), mm.get('n_seq'), mm.get('chain_order'),
                 mm.get('n_sites_mapped'), mm.get('n_sites'),
                 mm.get('depth_median'), mm.get('n_cols_ok'), mm.get('n_cols'),
                 m2.get('features')))

    print('\n=== PSA_cliff vs PSA_noncliff, tau sweep, M1 ===')
    p1 = long[(long['variant'] == 'primary') & (long['model'] == C5_MODELS[0])]
    print(p1.pivot_table(index='DMS_id', columns='tau',
                         values='PSA_cliff', aggfunc='first')
          .to_string(float_format=lambda v: '%.3f' % v))
    print(p1.pivot_table(index='DMS_id', columns='tau',
                         values='PSA_noncliff', aggfunc='first')
          .to_string(float_format=lambda v: '%.3f' % v))
    print('\n[C5] verify_manifest: %s' % (_pairs_verify() or 'clean'))
    return t12, long, pa


def _main(argv):
    if argv and argv[0] == '--stage7':
        stage7(argv[1:] or None)
        return 0
    _selfcheck(tuple(argv))
    return 0


if __name__ == '__main__':
    sys.exit(_main(sys.argv[1:]))
