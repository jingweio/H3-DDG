"""BGYM-CLIFF v1 -- stage 1: structural annotation (spec Sec.3 ``structure.py``,
Sec.1.5 C4, gates G11 / G-OPT).

Two public entry points, exactly as the spec's Sec.3 signature block names them:

* :func:`annotate_structure` -- per-residue geometry of ONE complex.
* :func:`map_mutations`      -- the assay's mutated positions joined onto that
  annotation, **by lookup through** ``mutant_pdb``, never by alignment.

plus the stage-1 driver :func:`stage1` (T09 + the ``structure/`` cache + the
per-assay interface-bias table + the G11 twin-structure control).

Everything numeric that is a DECISION BOUNDARY is read from
:data:`cliff.config.THRESH`.  Three published REFERENCE tables are defined here
because ``config.py`` does not carry them and this module does not own that file
(see the module docstring section "reference data, not thresholds" below).

Python 3.9: no ``match``, no runtime ``X | Y`` unions.
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

from cliff import config
from cliff import io_bgym
from cliff import pairs as _pairs
from cliff.config import ASSAYS, PATHS, THRESH

# --------------------------------------------------------------------------- #
# reference data, NOT thresholds                                              #
# --------------------------------------------------------------------------- #
# ``config.py`` is the frozen home of every numeric DECISION BOUNDARY and this
# module must not edit it.  The three tables below are published reference data
# (a physical constants table and the algebraic definition of a published
# classification), not decision boundaries: no verdict compares a statistic
# against them.  They are kept here, named, sourced and immutable.  Reported as
# a deviation in the delivery note.

#: Tien et al. 2013 (PLoS ONE 8:e80635, Table 1) maximum allowed solvent
#: accessibility per residue, A^2.  THEORETICAL column (Gly-X-Gly tripeptide).
TIEN2013_THEORETICAL = {
    'A': 129.0, 'R': 274.0, 'N': 195.0, 'D': 193.0, 'C': 167.0,
    'Q': 225.0, 'E': 223.0, 'G': 104.0, 'H': 224.0, 'I': 197.0,
    'L': 201.0, 'K': 236.0, 'M': 224.0, 'F': 240.0, 'P': 159.0,
    'S': 155.0, 'T': 172.0, 'W': 285.0, 'Y': 263.0, 'V': 174.0,
}

#: Tien et al. 2013, EMPIRICAL column (largest observed in a PDB survey).
TIEN2013_EMPIRICAL = {
    'A': 121.0, 'R': 265.0, 'N': 187.0, 'D': 187.0, 'C': 148.0,
    'Q': 214.0, 'E': 214.0, 'G': 97.0, 'H': 216.0, 'I': 195.0,
    'L': 191.0, 'K': 230.0, 'M': 203.0, 'F': 228.0, 'P': 154.0,
    'S': 143.0, 'T': 163.0, 'W': 264.0, 'Y': 255.0, 'V': 165.0,
}

#: Levy 2010 (J Mol Biol 403:660) structural regions.  The 25% relative-ASA cut
#: is part of the DEFINITION of the five classes, not a tunable threshold.
LEVY_RSA_CUT = 0.25
LEVY_CLASSES = ('interior', 'surface', 'support', 'rim', 'core')

#: coarse amino-acid grouping for the T09 ``aa_class`` stratification column
#: (used only as an NS1 stratifier downstream; no verdict reads it).
AA_CLASS = {}
for _aas, _cls in (('AVLIMC', 'hydrophobic'), ('FWY', 'aromatic'),
                   ('STNQ', 'polar'), ('KRH', 'positive'),
                   ('DE', 'negative'), ('GP', 'special')):
    for _a in _aas:
        AA_CLASS[_a] = _cls

_AA3TO1 = io_bgym._AA3TO1

#: Shrake-Rupley settings.  Biopython 1.81 defaults, pinned so a later default
#: change cannot silently move every SASA in the study.
SASA_PROBE_RADIUS = 1.40
SASA_N_POINTS = 100

#: hydrogen element symbols stripped before any SASA call (spec Sec.3).
HYDROGEN_ELEMENTS = ('H', 'D')

# --------------------------------------------------------------------------- #
# T09 columns (spec Sec.6, verbatim and in order)                             #
# --------------------------------------------------------------------------- #

T09_COLUMNS = [
    'DMS_id', 'chain', 'resseq', 'icode', 'seq_idx', 'wt_aa', 'levy_class',
    'rsa_iso', 'rsa_cplx', 'dsasa', 'min_heavy_dist', 'cb_dist', 'is_iface_5A',
    'is_iface_dsasa', 'n_variants_at_site', 'n_pairs_at_site', 'n_cliff_pairs',
    'cliff_rate', 'beta_hat_abs', 'rsa_decile', 'aa_class', 'depth_tertile',
    'OR_burial_matched', 'OR_lo95', 'OR_hi95', 'beta_iface_unadj',
    'beta_iface_adj', 'p_wald', 'p_NS1', 'beta_iface_after_rsa',
    'assay_permissible',
]

#: the T09 columns this module OWNS.  Everything else is stats_c4.py's and is
#: written EMPTY -- never dropped, never back-filled from the spec's own
#: expectations (same convention the stage-0 author used for T01).
T09_STRUCTURAL_COLUMNS = [
    'DMS_id', 'chain', 'resseq', 'icode', 'seq_idx', 'wt_aa', 'levy_class',
    'rsa_iso', 'rsa_cplx', 'dsasa', 'min_heavy_dist', 'cb_dist', 'is_iface_5A',
    'is_iface_dsasa', 'n_variants_at_site', 'rsa_decile', 'aa_class',
    'depth_tertile', 'assay_permissible',
]

T09_PENDING_COLUMNS = [c for c in T09_COLUMNS if c not in T09_STRUCTURAL_COLUMNS]

#: extra structural columns kept in the npz cache and in the returned frames but
#: NOT in the spec's T09 list (provenance / audit only).
ANNOT_EXTRA_COLUMNS = [
    'resname', 'side', 'n_heavy_atoms', 'sasa_iso', 'sasa_cplx', 'rsa_iso_raw',
    'rsa_cplx_raw', 'drsa', 'depth_A', 'is_mutated',
]


# --------------------------------------------------------------------------- #
# PDB loading + hydrogen strip                                                #
# --------------------------------------------------------------------------- #

def _rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


def strip_hydrogens(model):
    """Detach every ``element in ('H','D')`` atom from ``model`` IN PLACE.

    Returns the number of atoms removed.  Spec Sec.3: all 22 PDBs are protonated
    models, Biopython gives H a 1.20 A radius, and a naive ShrakeRupley call
    distorts the buried tail 2.5x (37 vs 15 residues with SASA < 1 on 1PQ1).
    """
    n = 0
    for chain in model:
        for res in chain:
            dead = [a.get_id() for a in res if a.element in HYDROGEN_ELEMENTS]
            for aid in dead:
                res.detach_child(aid)
                n += 1
    return n


def load_heavy_model(pdb_path, keep_chains=None, *, strip_h=True):
    """Parse ``pdb_path``, keep only ``keep_chains`` (all when ``None``), and
    strip hydrogens.  Returns ``(structure, model, n_hydrogens_stripped)``.

    A fresh parse per call is deliberate: detaching chains from a shared model
    and re-attaching them would leave Biopython's cached ``full_id`` stale, and
    ``ShrakeRupley`` keys its per-entity aggregation on ``full_id``.
    """
    st = PDBParser(QUIET=True).get_structure('s', pdb_path)
    n_models = len(st)
    if n_models != 1:
        raise ValueError('%s: %d MODELs, expected 1' % (pdb_path, n_models))
    model = st[0]
    if keep_chains is not None:
        keep = set(keep_chains)
        for cid in [ch.id for ch in model]:
            if cid not in keep:
                model.detach_child(cid)
        got = set(ch.id for ch in model)
        if got != keep:
            raise ValueError('%s: chains %r requested, %r present'
                             % (pdb_path, sorted(keep), sorted(got)))
    # no waters, no hetero-residues, no altlocs anywhere in this tree (checked)
    for chain in model:
        for res in chain:
            if res.id[0] != ' ':
                raise ValueError('%s: hetero/water residue %r' % (pdb_path, res.id))
            if res.get_resname() not in _AA3TO1:
                raise ValueError('%s: unknown residue %r' % (pdb_path, res.get_resname()))
    n_h = strip_hydrogens(model) if strip_h else 0
    return st, model, n_h


def _residue_key(chain_id, res):
    """The residue key is the TRIPLE ``(chain, resseq, icode)`` (spec Sec.3)."""
    _het, resseq, icode = res.id
    return (chain_id, int(resseq), icode.strip())


def _flatten(model, order=None):
    """Flatten ``model`` into canonical per-residue and per-atom arrays.

    Returns a dict with, in the canonical residue order (sorted by
    ``(chain, resseq, icode)``): ``keys`` (list of triples), ``chain``,
    ``resseq``, ``icode``, ``resname``, ``wt_aa``, ``n_atoms``, ``starts``
    (reduceat offsets), ``coords`` (n_atoms,3), ``atom_res`` (n_atoms,),
    ``atom_names``, ``residues`` (the Biopython objects).
    """
    recs = []
    for chain in model:
        for res in chain:
            recs.append((_residue_key(chain.id, res), res))
    recs.sort(key=lambda t: t[0])
    if order is not None:
        pos = {k: i for i, (k, _r) in enumerate(recs)}
        if sorted(pos) != sorted(order):
            raise ValueError('residue key sets differ between SASA passes')
        recs = [recs[pos[k]] for k in order]

    keys = [k for k, _r in recs]
    coords_list, atom_res, names, counts = [], [], [], []
    for i, (_k, res) in enumerate(recs):
        atoms = list(res)
        if not atoms:
            raise ValueError('residue %r has no heavy atoms' % (_k,))
        counts.append(len(atoms))
        for a in atoms:
            coords_list.append(a.coord)
            atom_res.append(i)
            names.append(a.get_id())
    counts = np.asarray(counts, dtype=np.int64)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    return dict(
        keys=keys,
        chain=np.array([k[0] for k in keys], dtype=object),
        resseq=np.array([k[1] for k in keys], dtype=np.int32),
        icode=np.array([k[2] for k in keys], dtype=object),
        resname=np.array([r.get_resname() for _k, r in recs], dtype=object),
        wt_aa=np.array([_AA3TO1[r.get_resname()] for _k, r in recs], dtype=object),
        n_atoms=counts, starts=starts,
        coords=np.asarray(coords_list, dtype=np.float64),
        atom_res=np.asarray(atom_res, dtype=np.int64),
        atom_names=names,
        residues=[r for _k, r in recs],
    )


# --------------------------------------------------------------------------- #
# annotate_structure                                                          #
# --------------------------------------------------------------------------- #

def annotate_structure(poi, pdb_path, side0, side1, *, tien='theoretical',
                       cache_dir=None, use_cache=True, force=False,
                       verbose=False):
    """Per-residue structural annotation of one two-sided complex.

    ``side0`` / ``side1`` are iterables of PDB chain letters, taken from
    ``data_splits/assay_chain_sides.tsv`` (never guessed).

    Columns (one row per residue of the complex, sorted by
    ``(chain, resseq, icode)``):

    ==================  =========================================================
    ``min_heavy_dist``  min heavy-atom distance to the OPPOSITE side, A.
                        cKDTree per side, ``np.minimum`` reduction over atoms.
    ``cb_dist``         same but C-beta to C-beta (CA for Gly / missing CB).
                        REPORTED ONLY -- ``Cb-Cb < 8 A`` is BANNED as an
                        interface definition (spec Sec.1.5).
    ``sasa_iso``        Shrake-Rupley SASA of the residue with only its OWN side
                        present (the side is one protein, so a two-chain Fab is
                        isolated as a unit).
    ``sasa_cplx``       Shrake-Rupley SASA in the full complex.
    ``dsasa``           ``sasa_iso - sasa_cplx`` (>= 0 exactly: adding atoms can
                        only occlude probe points).
    ``rsa_iso``         ``sasa_iso / Tien2013[wt_aa]``, clipped to
                        ``THRESH['C4_rsa_clip']``.  ``rsa_iso_raw`` unclipped.
    ``levy_class``      Levy 2010 in {interior, surface, support, rim, core}.
    ``depth_A``         burial depth: distance from the residue's nearest heavy
                        atom to the nearest SOLVENT-EXPOSED atom of its own
                        isolated side (0 for an exposed residue).
    ==================  =========================================================

    ``df.attrs`` carries the provenance (n hydrogens stripped, timings, the
    with-H control counts, and the side definitions).
    """
    side0 = tuple(side0)
    side1 = tuple(side1)
    if set(side0) & set(side1):
        raise ValueError('sides overlap: %r / %r' % (side0, side1))
    if tien == 'theoretical':
        maxacc = TIEN2013_THEORETICAL
    elif tien == 'empirical':
        maxacc = TIEN2013_EMPIRICAL
    else:
        raise ValueError('tien must be theoretical|empirical, got %r' % (tien,))

    cache_dir = cache_dir or PATHS.structure_cache
    cpath = os.path.join(cache_dir, '%s_%s_%s.npz'
                         % (poi, ''.join(side0), ''.join(side1)))
    if use_cache and not force and os.path.exists(cpath):
        df = _read_annot_cache(cpath)
        df.attrs['from_cache'] = True
        return df

    t0 = time.time()
    sr = ShrakeRupley(probe_radius=SASA_PROBE_RADIUS, n_points=SASA_N_POINTS)

    # ---- complex ---------------------------------------------------------- #
    _st, m_all, n_h_all = load_heavy_model(pdb_path)
    chains_present = tuple(ch.id for ch in m_all)
    if set(chains_present) != set(side0) | set(side1):
        raise ValueError('%s: chains %r != side0 %r + side1 %r'
                         % (pdb_path, chains_present, side0, side1))
    flat = _flatten(m_all)
    keys = flat['keys']
    t_parse = time.time() - t0

    t1 = time.time()
    sr.compute(m_all, level='R')
    sasa_cplx = np.array([r.sasa for r in flat['residues']], dtype=np.float64)
    t_sasa_cplx = time.time() - t1

    # ---- isolated sides --------------------------------------------------- #
    sasa_iso = np.full(len(keys), np.nan)
    depth = np.full(len(keys), np.nan)
    side_of = np.empty(len(keys), dtype=object)
    idx_of = {k: i for i, k in enumerate(keys)}
    t_sasa_iso = 0.0
    for label, chs in ((0, side0), (1, side1)):
        t2 = time.time()
        _sti, m_side, _nh = load_heavy_model(pdb_path, keep_chains=chs)
        fs = _flatten(m_side)
        sr.compute(m_side, level='R')
        t_sasa_iso += time.time() - t2
        s_iso = np.array([r.sasa for r in fs['residues']], dtype=np.float64)
        # burial depth: nearest solvent-exposed atom of the SAME isolated side
        atom_sasa = np.array([a.sasa for r in fs['residues'] for a in r],
                             dtype=np.float64)
        exposed = atom_sasa > 0.0
        if exposed.any():
            tree_exp = cKDTree(fs['coords'][exposed])
            d_at = tree_exp.query(fs['coords'], k=1)[0]
        else:                                          # pragma: no cover
            d_at = np.zeros(fs['coords'].shape[0])
        d_res = np.minimum.reduceat(d_at, fs['starts'])
        for j, k in enumerate(fs['keys']):
            i = idx_of[k]
            sasa_iso[i] = s_iso[j]
            depth[i] = d_res[j]
            side_of[i] = label
    if np.isnan(sasa_iso).any():
        raise RuntimeError('%s: %d residues covered by no side'
                           % (poi, int(np.isnan(sasa_iso).sum())))

    dsasa = sasa_iso - sasa_cplx
    if dsasa.min() < -1e-6:
        raise RuntimeError('%s: negative dSASA %.3g -- side split is wrong'
                           % (poi, dsasa.min()))
    dsasa = np.maximum(dsasa, 0.0)

    # ---- min heavy-atom distance to the OPPOSITE side --------------------- #
    t3 = time.time()
    coords = flat['coords']
    res_side1 = np.array([c in side1 for c in flat['chain']], dtype=bool)
    m1 = res_side1[flat['atom_res']]
    m0 = ~m1
    tree0 = cKDTree(coords[m0])
    tree1 = cKDTree(coords[m1])
    d_atom = np.empty(coords.shape[0], dtype=np.float64)
    d_atom[m0] = tree1.query(coords[m0], k=1)[0]
    d_atom[m1] = tree0.query(coords[m1], k=1)[0]
    min_heavy = np.minimum.reduceat(d_atom, flat['starts'])

    # ---- C-beta distance (reported, BANNED as a definition) --------------- #
    cb = np.full((len(keys), 3), np.nan)
    for i, res in enumerate(flat['residues']):
        if 'CB' in res:
            cb[i] = res['CB'].coord
        elif 'CA' in res:
            cb[i] = res['CA'].coord
    have = ~np.isnan(cb[:, 0])
    s1_res = res_side1
    cb_dist = np.full(len(keys), np.nan)
    for lab in (0, 1):
        src = have & ((s1_res) if lab == 1 else (~s1_res))
        tgt = have & ((~s1_res) if lab == 1 else (s1_res))
        if src.any() and tgt.any():
            cb_dist[src] = cKDTree(cb[tgt]).query(cb[src], k=1)[0]
    t_dist = time.time() - t3

    # ---- RSA + Levy ------------------------------------------------------- #
    wt_aa = flat['wt_aa']
    denom = np.array([maxacc[a] for a in wt_aa], dtype=np.float64)
    rsa_iso_raw = sasa_iso / denom
    rsa_cplx_raw = sasa_cplx / denom
    lo, hi = THRESH['C4_rsa_clip']
    rsa_iso = np.clip(rsa_iso_raw, lo, hi)
    rsa_cplx = np.clip(rsa_cplx_raw, lo, hi)
    levy = levy_class(rsa_iso, rsa_cplx, dsasa)

    df = pd.DataFrame(dict(
        chain=flat['chain'], resseq=flat['resseq'], icode=flat['icode'],
        resname=flat['resname'], wt_aa=wt_aa, side=side_of.astype(np.int8),
        n_heavy_atoms=flat['n_atoms'].astype(np.int32),
        sasa_iso=sasa_iso, sasa_cplx=sasa_cplx, dsasa=dsasa,
        rsa_iso=rsa_iso, rsa_cplx=rsa_cplx,
        rsa_iso_raw=rsa_iso_raw, rsa_cplx_raw=rsa_cplx_raw,
        drsa=rsa_iso - rsa_cplx, levy_class=levy,
        min_heavy_dist=min_heavy, cb_dist=cb_dist, depth_A=depth,
        is_iface_5A=min_heavy < THRESH['C4_iface_dist_A'],
        is_iface_dsasa=dsasa > THRESH['C4_dsasa_min_A2'],
    ))
    df['rsa_decile'] = _rank_bins(df['rsa_iso'].values, 10)
    df['depth_tertile'] = _rank_bins(df['depth_A'].values, 3)
    df['aa_class'] = [AA_CLASS[a] for a in wt_aa]
    df.attrs.update(
        poi=poi, pdb_path=pdb_path, pdb_md5=io_bgym.md5_of(pdb_path),
        side0=''.join(side0), side1=''.join(side1),
        n_residues=int(len(keys)), n_heavy_atoms=int(coords.shape[0]),
        n_hydrogens_stripped=int(n_h_all), tien=tien,
        probe_radius=SASA_PROBE_RADIUS, n_points=SASA_N_POINTS,
        wall_parse_s=round(t_parse, 3), wall_sasa_cplx_s=round(t_sasa_cplx, 3),
        wall_sasa_iso_s=round(t_sasa_iso, 3), wall_dist_s=round(t_dist, 3),
        wall_s=round(time.time() - t0, 3),
    )
    if use_cache:
        _write_annot_cache(cpath, df)
    if verbose:
        print('[annot] %-14s %-4s|%-4s res=%4d heavy=%5d H-stripped=%5d  %.2fs'
              % (poi, ''.join(side0), ''.join(side1), len(keys),
                 coords.shape[0], n_h_all, df.attrs['wall_s']))
    return df


def levy_class(rsa_iso, rsa_cplx, dsasa):
    """Levy 2010's five structural regions, vectorised.

    ``interior``  rASA_u <  25%, dASA == 0      ``surface`` rASA_u >= 25%, dASA == 0
    ``support``   rASA_u <  25%, dASA >  0      ``rim``     rASA_c >= 25%, dASA >  0
    ``core``      rASA_u >= 25%, rASA_c < 25%, dASA > 0
    """
    rsa_iso = np.asarray(rsa_iso, dtype=np.float64)
    rsa_cplx = np.asarray(rsa_cplx, dtype=np.float64)
    iface = np.asarray(dsasa, dtype=np.float64) > 0.0
    buried_u = rsa_iso < LEVY_RSA_CUT
    out = np.empty(rsa_iso.shape, dtype=object)
    out[(~iface) & buried_u] = 'interior'
    out[(~iface) & (~buried_u)] = 'surface'
    out[iface & buried_u] = 'support'
    out[iface & (~buried_u) & (rsa_cplx < LEVY_RSA_CUT)] = 'core'
    out[iface & (~buried_u) & (rsa_cplx >= LEVY_RSA_CUT)] = 'rim'
    if any(v is None for v in out):                    # pragma: no cover
        raise RuntimeError('levy_class left %d residues unassigned'
                           % sum(v is None for v in out))
    return out


def _rank_bins(v, k):
    """0-based rank bin (decile / tertile) that is stable under ties: the bin of
    a value is ``floor(k * (#strictly-smaller + 0.5*#equal) / n)``."""
    v = np.asarray(v, dtype=np.float64)
    n = v.size
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    order = np.argsort(v, kind='stable')
    sv = v[order]
    lo = np.searchsorted(sv, v, side='left')
    hi = np.searchsorted(sv, v, side='right')
    mid = 0.5 * (lo + hi)
    b = np.floor(k * mid / n).astype(np.int64)
    return np.clip(b, 0, k - 1).astype(np.int8)


# --------------------------------------------------------------------------- #
# npz cache                                                                   #
# --------------------------------------------------------------------------- #

_OBJ_COLS = ('chain', 'icode', 'resname', 'wt_aa', 'levy_class', 'aa_class')


def _write_annot_cache(path, df):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = {}
    for c in df.columns:
        v = df[c].values
        if c in _OBJ_COLS:
            # width from the data -- a fixed 'U8' silently truncated
            # aa_class 'hydrophobic' (11 chars) and broke the round-trip
            w = max(1, max((len(str(x)) for x in v), default=1))
            out[c] = np.array([str(x) for x in v], dtype='U%d' % w)
        else:
            out[c] = v
    out['__columns__'] = np.array(list(df.columns), dtype='U32')
    out['__attrs__'] = np.array(json.dumps(df.attrs, sort_keys=True))
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)
    return path


def _read_annot_cache(path):
    z = np.load(path, allow_pickle=False)
    cols = [str(c) for c in z['__columns__']]
    d = {}
    for c in cols:
        v = z[c]
        d[c] = np.array([str(x) for x in v], dtype=object) if c in _OBJ_COLS else v
    df = pd.DataFrame(d, columns=cols)
    df.attrs.update(json.loads(str(z['__attrs__'])))
    return df


def cache_structure(dms_id, **kw):
    """Annotate the complex behind ``dms_id`` and return the manifest entry
    ``{path, md5, bytes}`` for ``MANIFEST.json`` alongside the frame."""
    spec = ASSAYS[dms_id]
    pdb_path = os.path.join(PATHS.structures, spec.pdb_file)
    df = annotate_structure(spec.poi, pdb_path, spec.side0_chains,
                            spec.side1_chains, **kw)
    cpath = os.path.join(PATHS.structure_cache, '%s_%s_%s.npz'
                         % (spec.poi, spec.side0_chains, spec.side1_chains))
    entry = dict(path=os.path.relpath(cpath, config.REPO),
                 md5=io_bgym.md5_of(cpath), bytes=os.path.getsize(cpath))
    return df, entry


# --------------------------------------------------------------------------- #
# mutation -> residue lookup                                                  #
# --------------------------------------------------------------------------- #

def _wildtype_sequences(dms_id):
    """The assay's ``wildtype_sequence`` dict.  Asserted constant over rows."""
    col = pd.read_csv(PATHS.dms_csv(dms_id), usecols=['wildtype_sequence'],
                      dtype=str)['wildtype_sequence'].values
    uniq = pd.unique(col)
    if len(uniq) != 1:
        raise ValueError('%s: %d distinct wildtype_sequence values'
                         % (dms_id, len(uniq)))
    return io_bgym._parse_dict_str(uniq[0]), len(col)


def position_table(assay):
    """One row per DISTINCT mutated ``(chain, seq_pos)`` of ``assay``.

    Built by re-parsing ``mutant`` / ``mutant_pdb`` (the :class:`Assay` keeps the
    canonical key and the PDB key in different orders, so the two cannot be
    zipped) with early termination once every ``(chain, seq_pos)`` in
    ``assay.pos_index`` has been seen.  ``n_variants_at_site`` comes from the
    cached ``codes`` matrix, so it is exact regardless of the early stop.
    """
    df = pd.read_csv(PATHS.dms_csv(assay.dms_id), usecols=['mutant', 'mutant_pdb'])
    mut = df['mutant'].values
    pdb = df['mutant_pdb'].values
    want = set(assay.pos_index)
    seen = {}
    n_scanned = 0
    for i in range(len(mut)):
        n_scanned += 1
        for chain, pos, wt, _m, resseq, icode in io_bgym.parse_pair_dicts(mut[i], pdb[i]):
            k = (chain, pos)
            rec = (resseq, icode, wt)
            if k in seen:
                if seen[k] != rec:
                    raise ValueError('%s: (%s,%d) maps to %r and %r'
                                     % (assay.dms_id, chain, pos, seen[k], rec))
            else:
                seen[k] = rec
        if len(seen) == len(want):
            break
    if set(seen) != want:
        raise ValueError('%s: position_table covered %d of %d pos_index entries'
                         % (assay.dms_id, len(seen), len(want)))

    n_var = (assay.codes != 0).sum(axis=0)
    n_aa = np.zeros(assay.P, dtype=np.int32)
    for (chain, pos, _aa), _c in assay.col_index.items():
        n_aa[assay.pos_index[(chain, pos)]] += 1

    rows = []
    for (chain, pos), (resseq, icode, wt) in sorted(seen.items()):
        col = assay.pos_index[(chain, pos)]
        rows.append(dict(chain=chain, seq_idx=int(pos), resseq=int(resseq),
                         icode=icode, wt_aa_mutcol=wt,
                         n_variants_at_site=int(n_var[col]),
                         n_aa_observed=int(n_aa[col]), code_col=int(col)))
    out = pd.DataFrame(rows)
    out.attrs.update(dms_id=assay.dms_id, n_rows_scanned=int(n_scanned),
                     n_rows_total=int(len(mut)), n_positions=int(len(rows)))
    return out


def map_mutations(assay, annot, *, verbose=False):
    """Join ``assay``'s mutated positions onto ``annot`` BY LOOKUP through
    ``mutant_pdb`` -- never by alignment.

    Asserts all FOUR wt-letter sources agree on every mutated position:
    ``mutant`` / ``mutant_pdb`` (already cross-checked token-by-token inside
    :func:`cliff.io_bgym.parse_pair_dicts`), ``wildtype_sequence``, and the PDB
    residue itself (from ``annot``, i.e. Biopython's own parse).

    Returns one row per mutated position with every structural column, plus
    ``df.attrs`` carrying the four-source audit counters.
    """
    pt = position_table(assay)
    wt_seq, _n = _wildtype_sequences(assay.dms_id)

    key_to_i = {(c, int(r), str(ic)): i for i, (c, r, ic)
                in enumerate(zip(annot['chain'].values, annot['resseq'].values,
                                 annot['icode'].values))}
    n_missing = 0
    n_mm_seqcol = n_mm_pdbcol = n_mm_wtseq = n_mm_pdbres = 0
    rows = []
    for rec in pt.to_dict('records'):
        k = (rec['chain'], rec['resseq'], rec['icode'])
        if k not in key_to_i:
            n_missing += 1
            raise KeyError('%s: mutated residue %r absent from %s'
                           % (assay.dms_id, k, annot.attrs.get('poi')))
        i = key_to_i[k]
        wt_mutcol = rec['wt_aa_mutcol']          # source 1 == source 2 (io_bgym)
        seq = wt_seq.get(rec['chain'], '')
        j = rec['seq_idx'] - 1
        wt_seqfile = seq[j] if 0 <= j < len(seq) else '?'
        wt_pdbres = str(annot['wt_aa'].values[i])
        if wt_seqfile != wt_mutcol:
            n_mm_wtseq += 1
        if wt_pdbres != wt_mutcol:
            n_mm_pdbres += 1
        r = dict(rec)
        r['wt_aa'] = wt_mutcol
        r['wt_aa_wildtype_sequence'] = wt_seqfile
        r['wt_aa_pdb_residue'] = wt_pdbres
        r['annot_row'] = i
        rows.append(r)
    out = pd.DataFrame(rows)
    for c in ANNOT_EXTRA_COLUMNS + ['levy_class', 'rsa_iso', 'rsa_cplx', 'dsasa',
                                    'min_heavy_dist', 'cb_dist', 'is_iface_5A',
                                    'is_iface_dsasa', 'rsa_decile',
                                    'depth_tertile', 'aa_class']:
        if c in annot.columns:
            out[c] = annot[c].values[out['annot_row'].values]
    out['DMS_id'] = assay.dms_id

    bad = (n_mm_seqcol + n_mm_pdbcol + n_mm_wtseq + n_mm_pdbres + n_missing)
    out.attrs.update(
        dms_id=assay.dms_id, poi=annot.attrs.get('poi'),
        n_positions=int(len(out)),
        n_mutation_instances=int(assay.n_muts.sum()),
        n_missing_in_pdb=int(n_missing),
        n_wt_mismatch_mutant=int(n_mm_seqcol),
        n_wt_mismatch_mutant_pdb=int(n_mm_pdbcol),
        n_wt_mismatch_wildtype_sequence=int(n_mm_wtseq),
        n_wt_mismatch_pdb_residue=int(n_mm_pdbres),
        n_wt_mismatch_total=int(bad),
        n_rows_scanned=pt.attrs['n_rows_scanned'],
        mutated_chains=''.join(sorted(set(out['chain']))),
    )
    if bad:
        raise AssertionError(
            '%s: %d wt-letter disagreements across the four sources '
            '(mutant=%d mutant_pdb=%d wildtype_sequence=%d PDB=%d missing=%d)'
            % (assay.dms_id, bad, n_mm_seqcol, n_mm_pdbcol, n_mm_wtseq,
               n_mm_pdbres, n_missing))
    if verbose:
        print('[map ] %-42s %3d positions on chains %-3s  4/4 wt sources agree'
              % (assay.dms_id, len(out), out.attrs['mutated_chains']))
    return out


# --------------------------------------------------------------------------- #
# background positions via a CONSTANT seq->pdb offset (guarded)               #
# --------------------------------------------------------------------------- #

def chain_offsets(assay, annot, mut):
    """Per mutated chain: the modal ``resseq - seq_pos`` offset and whether it
    survives a PER-POSITION IDENTITY ASSERTION over the whole chain.

    Spec Sec.3: constant offsets may be used ONLY to build background
    (non-mutated) position sets, ONLY on the verified-clean chains, and ONLY
    with a per-position identity assertion -- they FAIL on 4ZFG-H/L and
    4ZFF-H/L (168 mismatches of 219 for 4ZFG-H).  Nothing here is ever used for
    the mutation lookup itself, which goes through ``mutant_pdb`` only.

    The identity check is run for EVERY chain, including the ones whose offset
    is already disqualified (non-constant, or an icode in the mutated set), so
    that the size of the failure is measured rather than assumed.  A chain is
    ``verified_clean`` only when all four conditions hold: one offset, no
    icode, 0 letter mismatches and 0 absent residues.
    """
    wt_seq, _n = _wildtype_sequences(assay.dms_id)
    letters = {(c, int(r), str(ic)): str(a) for c, r, ic, a
               in zip(annot['chain'].values, annot['resseq'].values,
                      annot['icode'].values, annot['wt_aa'].values)}
    out = []
    for chain, sub in mut.groupby('chain'):
        deltas = [int(r) - int(p) for r, p in zip(sub['resseq'], sub['seq_idx'])]
        offs = sorted(set(deltas))
        vals, cnts = np.unique(np.asarray(deltas, dtype=np.int64),
                               return_counts=True)
        off = int(vals[int(np.argmax(cnts))])          # modal offset
        has_icode = bool(any(str(x) != '' for x in sub['icode']))
        seq = wt_seq.get(chain, '')
        n_mm = n_abs = 0
        for j in range(len(seq)):
            k = (chain, j + 1 + off, '')
            if k not in letters:
                n_abs += 1
            elif letters[k] != seq[j]:
                n_mm += 1
        why = []
        if len(offs) != 1:
            why.append('%d distinct offsets %s' % (len(offs), offs[:5]))
        if has_icode:
            why.append('icode present in the mutated set')
        if n_mm:
            why.append('%d letter mismatches of %d' % (n_mm, len(seq)))
        if n_abs:
            why.append('%d absent of %d' % (n_abs, len(seq)))
        out.append(dict(DMS_id=assay.dms_id, chain=chain, offset=off,
                        offset_constant=(len(offs) == 1),
                        n_offsets_seen=len(offs), has_icode=has_icode,
                        seq_len=len(seq), n_checked=len(seq), n_mismatch=n_mm,
                        n_absent=n_abs,
                        verified_clean=(len(offs) == 1 and not has_icode
                                        and n_mm == 0 and n_abs == 0),
                        reason='; '.join(why)))
    return pd.DataFrame(out)


def background_positions(assay, annot, mut, offsets):
    """Non-mutated positions of the mutated chains, ``seq_idx`` filled in only
    on the offset-verified-clean chains (empty elsewhere)."""
    mutated = set(zip(mut['chain'], mut['resseq'].astype(int),
                      mut['icode'].astype(str)))
    off_of = {r['chain']: (int(r['offset']) if r['verified_clean'] else None)
              for r in offsets.to_dict('records')}
    chains = set(off_of)
    rows = []
    for i in range(len(annot)):
        c = str(annot['chain'].values[i])
        if c not in chains:
            continue
        k = (c, int(annot['resseq'].values[i]), str(annot['icode'].values[i]))
        if k in mutated:
            continue
        off = off_of[c]
        rows.append(dict(DMS_id=assay.dms_id, chain=c, resseq=k[1], icode=k[2],
                         seq_idx=(k[1] - off) if off is not None else -1,
                         annot_row=i))
    out = pd.DataFrame(rows, columns=['DMS_id', 'chain', 'resseq', 'icode',
                                      'seq_idx', 'annot_row'])
    if len(out):
        for c in ['wt_aa', 'levy_class', 'rsa_iso', 'rsa_cplx', 'dsasa',
                  'min_heavy_dist', 'cb_dist', 'is_iface_5A', 'is_iface_dsasa',
                  'rsa_decile', 'depth_tertile', 'aa_class', 'depth_A', 'side',
                  'sasa_iso', 'sasa_cplx', 'rsa_iso_raw', 'rsa_cplx_raw',
                  'drsa', 'resname', 'n_heavy_atoms']:
            if c in annot.columns:
                out[c] = annot[c].values[out['annot_row'].values]
    return out


def interface_bias(assay, annot, mut, offsets):
    """``design_iface_frac`` vs ``bg_iface_frac`` and the design-bias factor.

    ``design`` = the assay's mutated positions.  ``bg`` = EVERY residue of the
    mutated chains present in the PDB (the denominator the spec's own numbers
    use: GB1_1FCC 18/55 = 0.327 design against 18/56 = 0.321 background).
    """
    chains = set(mut['chain'])
    in_chain = np.array([str(c) in chains for c in annot['chain'].values])
    iface = annot['is_iface_5A'].values.astype(bool)
    dsz = annot['is_iface_dsasa'].values.astype(bool)
    d_i = mut['is_iface_5A'].values.astype(bool)
    d_d = mut['is_iface_dsasa'].values.astype(bool)
    n_d = len(mut)
    n_b = int(in_chain.sum())
    dif = float(d_i.mean()) if n_d else float('nan')
    bif = float(iface[in_chain].mean()) if n_b else float('nan')
    spec = ASSAYS[assay.dms_id]
    return dict(
        DMS_id=assay.dms_id, poi=spec.poi, tier=spec.tier,
        mutated_chains=''.join(sorted(chains)),
        n_design=n_d, n_bg=n_b,
        n_design_iface=int(d_i.sum()), n_bg_iface=int(iface[in_chain].sum()),
        design_iface_frac=dif, bg_iface_frac=bif,
        iface_bias_factor=(dif / bif) if (bif and bif == bif and bif > 0)
        else float('nan'),
        design_iface_frac_dsasa=float(d_d.mean()) if n_d else float('nan'),
        bg_iface_frac_dsasa=float(dsz[in_chain].mean()) if n_b else float('nan'),
        design_iface_frac_spec=spec.design_iface_frac_spec,
        bg_iface_frac_spec=spec.bg_iface_frac_spec,
        eligible_C4S=spec.eligible_C4S,
        n_bg_offset_verified=int(sum(1 for r in offsets.to_dict('records')
                                     if r['verified_clean'])),
        n_bg_offset_failed=int(sum(1 for r in offsets.to_dict('records')
                                   if not r['verified_clean'])),
    )


# --------------------------------------------------------------------------- #
# stage 1 driver                                                              #
# --------------------------------------------------------------------------- #

def structural_assays():
    """The assays whose PDB actually exists (25 registered; the two CR9114
    hypercubes and CR6261 have no PDB in ``structures/`` -- spec G-OPT)."""
    out = []
    for k in config.ALL_ASSAYS:
        p = os.path.join(PATHS.structures, ASSAYS[k].pdb_file)
        if os.path.exists(p):
            out.append(k)
    return tuple(out)


def build_T09(mut_by_assay):
    """T09 with the spec's exact column list; the stats_c4 columns stay EMPTY."""
    frames = []
    for dms_id, mut in mut_by_assay.items():
        d = pd.DataFrame(index=range(len(mut)))
        d['DMS_id'] = dms_id
        for c in ('chain', 'resseq', 'icode', 'seq_idx', 'wt_aa', 'levy_class',
                  'rsa_iso', 'rsa_cplx', 'dsasa', 'min_heavy_dist', 'cb_dist',
                  'is_iface_5A', 'is_iface_dsasa', 'n_variants_at_site',
                  'rsa_decile', 'aa_class', 'depth_tertile'):
            d[c] = mut[c].values
        d['assay_permissible'] = ASSAYS[dms_id].eligible_C4S
        for c in T09_PENDING_COLUMNS:
            d[c] = ''
        frames.append(d[T09_COLUMNS])
    t09 = pd.concat(frames, ignore_index=True)
    assert list(t09.columns) == T09_COLUMNS
    return t09


def build_T09_all_residues(annot_by_assay):
    """The full per-(assay, residue) annotation -- 9,493 rows over the 25
    structural assays.  This is the frame every audited denominator in the
    spec's Sec.3 docstring is quoted against (35/9,493 rsa_iso > 1.0)."""
    frames = []
    for dms_id, a in annot_by_assay.items():
        d = a.copy()
        d.insert(0, 'DMS_id', dms_id)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def _merge_manifest(new_entries, extra=None):
    """Add ``new_entries`` to ``MANIFEST.json`` without dropping stage 0's."""
    old = {}
    if os.path.exists(PATHS.manifest):
        with open(PATHS.manifest) as fh:
            old = json.load(fh)
    reserved = ('schema', 'written_utc', 'env', 'env_observed', 'git',
                'seed_base', 'seeds', 'assay_ordinal', 'taus',
                'bindinggym_input', 'files')
    keep = {k: v for k, v in old.items() if k not in reserved}
    if extra:
        keep.update(extra)
    new_paths = set(e['path'] for e in new_entries)
    entries = [dict(path=p, md5=m['md5'], bytes=m['bytes'])
               for p, m in old.get('files', {}).items() if p not in new_paths]
    man = _pairs.write_manifest(entries + list(new_entries), extra=keep)
    # MANIFEST.json is shared, unlocked state: another stage writing between our
    # read and our write would silently drop its own entries.  Detect it rather
    # than pretend it cannot happen.
    lost = set(old.get('files', {})) - set(man['files'])
    if lost:
        sys.stderr.write('[structure] WARNING: %d manifest entries lost in the '
                         'merge (concurrent writer?): %s\n'
                         % (len(lost), sorted(lost)[:5]))
    return man


def stage1(assays=None, *, verbose=True, use_cache=True, force=False,
           write_t09=True):
    """Structural annotation for every assay with a PDB, T09, and the gates.

    Returns ``{'T09','T09_all_residues','annot','mut','bg','offsets','bias',
    'manifest','gates','bench'}``.
    """
    config.assert_env()
    PATHS.ensure_cache_dirs()
    ids = tuple(assays) if assays is not None else structural_assays()
    t_start = time.time()

    annot_by, mut_by, bg_by = {}, {}, {}
    off_rows, bias_rows, entries, bench = [], [], [], []
    for dms_id in ids:
        t0 = time.time()
        df, entry = cache_structure(dms_id, use_cache=use_cache, force=force)
        t_annot = time.time() - t0
        entries.append(entry)
        assay = io_bgym.load_assay(dms_id)
        mut = map_mutations(assay, df, verbose=verbose)
        offs = chain_offsets(assay, df, mut)
        bg = background_positions(assay, df, mut, offs)
        annot_by[dms_id] = df
        mut_by[dms_id] = mut
        bg_by[dms_id] = bg
        off_rows.append(offs)
        bias_rows.append(interface_bias(assay, df, mut, offs))
        bench.append(dict(DMS_id=dms_id, poi=ASSAYS[dms_id].poi,
                          n_residues=len(df), n_heavy=df.attrs['n_heavy_atoms'],
                          n_H_stripped=df.attrs['n_hydrogens_stripped'],
                          wall_annot_s=round(t_annot, 3),
                          wall_annot_cached=bool(df.attrs.get('from_cache', False)),
                          wall_total_s=round(time.time() - t0, 3),
                          rss_gb=round(_rss_gb(), 3)))
    offsets = pd.concat(off_rows, ignore_index=True)
    bias = pd.DataFrame(bias_rows)
    bench = pd.DataFrame(bench)

    t09 = build_T09(mut_by)
    t09_all = build_T09_all_residues(annot_by)
    if write_t09:
        os.makedirs(PATHS.artifacts, exist_ok=True)
        t09.to_csv(os.path.join(PATHS.artifacts, 'T09_structure_sites.csv'),
                   index=False)

    man = _merge_manifest(entries, extra=dict(structure=dict(
        assays=list(ids), n_assays=len(ids),
        n_residues_total=int(len(t09_all)),
        n_mutated_positions=int(len(t09)),
        tien='theoretical', levy_rsa_cut=LEVY_RSA_CUT,
        probe_radius=SASA_PROBE_RADIUS, n_points=SASA_N_POINTS,
        wall_s=round(time.time() - t_start, 2))))

    gates = _gate_rows(t09, t09_all, bias, annot_by, bench)
    return dict(T09=t09, T09_all_residues=t09_all, annot=annot_by, mut=mut_by,
                bg=bg_by, offsets=offsets, bias=bias, manifest=man,
                gates=gates, bench=bench,
                wall_s=round(time.time() - t_start, 2))


def _g(gate_id, name, assay, statistic, expected, observed, tol='', cons='',
       halts=False, ok=None):
    if ok is None:
        ok = '' if observed in ('', None) else (observed == expected)
    return {'gate_id': gate_id, 'gate_name': name, 'assay': assay,
            'statistic': statistic, 'expected': expected, 'observed': observed,
            'tolerance': tol, 'PASS/FAIL': ('' if ok == '' else
                                            ('PASS' if ok else 'FAIL')),
            'consequence_if_fail': cons, 'halts_study': halts}


def _gate_rows(t09, t09_all, bias, annot_by, bench):
    """The structural half of T02: G11, the C4 interface-definition audits, and
    the spec's Sec.1.5 pre-declared structural facts."""
    rows = []
    ri = t09_all['rsa_iso_raw'].values
    n_gt1 = int((ri > 1.0).sum())
    rows.append(_g('G-STR', 'rsa_iso exceeding 1.0 before clipping', 'ALL',
                   'n(rsa_iso_raw > 1.0) / n_residues', '35 / 9493',
                   '%d / %d' % (n_gt1, len(t09_all)),
                   cons='Tien 2013 maxima are the wrong table'))
    rows.append(_g('G-STR', 'max rsa_iso before clipping', 'ALL',
                   'max rsa_iso_raw', 1.36, round(float(ri.max()), 4),
                   tol='+-0.01', ok=abs(float(ri.max()) - 1.36) <= 0.01))
    dz = t09_all['is_iface_dsasa'].values.astype(bool)
    mh = t09_all['min_heavy_dist'].values
    rows.append(_g('G-STR', 'max min-heavy distance over all dSASA>1 residues',
                   'ALL', 'max min_heavy_dist | dsasa > 1 A^2', 6.07,
                   round(float(mh[dz].max()), 2), tol='+-0.02',
                   ok=abs(float(mh[dz].max()) - 6.07) <= 0.02,
                   cons='the 5.0/6.0 A interface cut is not a superset of dSASA>1'))
    cb = t09_all['cb_dist'].values
    flag = cb < THRESH['C4_cb_dist_banned_A']
    rows.append(_g('G-STR', 'BANNED Cbeta-Cbeta < 8 A: n flagged', 'ALL',
                   'n(cb_dist < 8 A)', 911, int(np.nansum(flag)),
                   cons='Cbeta-Cbeta stays BANNED either way'))
    rows.append(_g('G-STR', 'BANNED Cbeta-Cbeta < 8 A: recall vs dSASA>1', 'ALL',
                   'n(cb<8 & dsasa>1) / n(dsasa>1)', '825 / 1050',
                   '%d / %d' % (int(np.nansum(flag & dz)), int(dz.sum())),
                   cons='Cbeta-Cbeta stays BANNED either way'))
    lv = t09['levy_class'].values
    frac_int = float((lv == 'interior').mean())
    rows.append(_g('G-STR', 'mutated positions that are Levy interior', 'ALL',
                   'frac levy_class == interior', 0.437, round(frac_int, 4),
                   tol='+-0.01', ok=abs(frac_int - 0.437) <= 0.01,
                   cons='the C4 burial-matching premise changes'))
    rows.append(_g('G-STR', 'total mutated positions, 25 structural assays',
                   'ALL', 'n rows of T09',
                   config.EXPECTED['n_mutated_positions_total'], int(len(t09))))
    # independent end-to-end check of the whole mutant_pdb lookup: summing
    # n_variants_at_site over T09 must reproduce G1's mutation-instance count
    n_inst = int(t09['n_variants_at_site'].values.sum())
    rows.append(_g('G-STR', 'mutation instances recovered by the lookup', 'ALL',
                   'sum(n_variants_at_site) over T09',
                   config.EXPECTED['G1_n_mutation_instances_registered25'],
                   n_inst,
                   cons='the mutant_pdb lookup lost or duplicated mutations'))
    rows.append(_g('G-STR', 'registered assays with a PDB in structures/', 'ALL',
                   'n assays annotated', config.EXPECTED['n_registered'],
                   int(t09['DMS_id'].nunique()),
                   cons='G-OPT: the hypercube arm stays structurally mute'))
    rows.append(_g('G-OPT', 'PDBs absent from structures/ for the arm', 'ARM',
                   'n assays with no PDB', 3,
                   len(config.ALL_ASSAYS) - int(t09['DMS_id'].nunique()),
                   cons='skip the structural half for the arm and SAY SO; '
                        'NEVER substitute a constant offset'))
    # 5A12_VEGF designed C4 negative control
    v = t09[t09['DMS_id'] == '5A12_VEGF_fitness_4ZFF']
    if len(v):
        n_close = int((v['min_heavy_dist'].values < 6.4).sum())
        rows.append(_g('G-STR', '5A12_VEGF designed C4 NEGATIVE control',
                       '5A12_VEGF_fitness_4ZFF',
                       'n mutated positions within 6.4 A of VEGF', '0 / 9',
                       '%d / %d' % (n_close, len(v)),
                       cons='the designed negative control is not negative'))
    # Z-ZpA963_HL1: 6/6 interface => out of C4-S
    z = t09[t09['DMS_id'] == 'Z-domain_ZpA963_HL1_fitness_2M5A']
    if len(z):
        rows.append(_g('G-STR', 'Z-ZpA963_HL1 is all-interface => out of C4-S',
                       'Z-domain_ZpA963_HL1_fitness_2M5A',
                       'n iface / n mutated positions', '6 / 6',
                       '%d / %d' % (int(z['is_iface_5A'].values.sum()), len(z)),
                       cons='C4-S would become falsifiable there'))
    # G11 twin structure
    a, b = 'KRAS_RAF1_norfitness_6VJJ', 'KRAS_RAF1-RBD_norfitness_6VJJ'
    if a in annot_by and b in annot_by:
        ca = annot_by[a].to_csv(index=False).encode()
        cb_ = annot_by[b].to_csv(index=False).encode()
        rows.append(_g('G11', 'twin-structure control: byte-identical annotation',
                       '%s vs %s' % (a, b), 'md5 of the per-residue annotation',
                       'identical', 'identical' if ca == cb_ else 'DIFFER',
                       cons='at most ONE of the two KRAS interfaces can be causal'))
    # dSASA > 1 A^2 is non-binding on this benchmark: no residue has
    # 0 < dSASA <= 1, so {dSASA>1} == {dSASA>0} == Levy {support,rim,core}
    d0 = int((t09_all['dsasa'].values > 0.0).sum())
    rows.append(_g('G-STR', 'dSASA > 1 A^2 threshold is non-binding', 'ALL',
                   'n(dsasa > 0) vs n(dsasa > 1)', '%d == %d' % (d0, d0),
                   '%d == %d' % (d0, int(dz.sum())),
                   cons='the 1 A^2 cut would start to matter'))
    # per-assay design vs background interface fraction
    for r in bias.to_dict('records'):
        for which, spec_key, obs_key in (('design', 'design_iface_frac_spec',
                                          'design_iface_frac'),
                                         ('bg', 'bg_iface_frac_spec',
                                          'bg_iface_frac')):
            exp = r[spec_key]
            obs = r[obs_key]
            ok = '' if exp != exp else (abs(obs - exp) <= 0.002)
            rows.append(_g('G-STR', '%s interface fraction' % which, r['DMS_id'],
                           '%s_iface_frac  (5.0 A; dSASA>1 gives %.4f; bias %.3fx)'
                           % (which,
                              r['design_iface_frac_dsasa'] if which == 'design'
                              else r['bg_iface_frac_dsasa'],
                              r['iface_bias_factor']),
                           '' if exp != exp else round(exp, 4), round(obs, 4),
                           tol='+-0.002', ok=ok,
                           cons='C4-S eligibility for this assay is mis-declared'))
    # the one pre-declared value that does NOT reproduce, and its resolution
    if 'KRAS_SOS1_norfitness_8BE4' in annot_by:
        a8 = annot_by['KRAS_SOS1_norfitness_8BE4']
        s_side = a8['chain'].values == 'S'
        f5 = a8['min_heavy_dist'].values < THRESH['C4_iface_dist_A']
        rows.append(_g('G-STR', "KRAS_SOS1 bg 0.110 is the PARTNER side's fraction",
                       'KRAS_SOS1_norfitness_8BE4',
                       "5.0 A iface frac of chain S (SOS1) = the spec's bg_iface_frac",
                       0.110, round(float(f5[s_side].mean()), 4), tol='+-0.002',
                       ok=abs(float(f5[s_side].mean()) - 0.110) <= 0.002,
                       cons="the spec's 2.4x SOS1 design bias is a side-attribution "
                            'error; measured bias on chain R is 1.01x'))
    rows.append(_g('G-STR', 'stage-1 wall clock, all structural assays', 'ALL',
                   'seconds, one core', config.EXPECTED['structure_all_s'],
                   round(float(bench['wall_total_s'].sum()), 1),
                   tol='<= 2x budget',
                   ok=float(bench['wall_total_s'].sum())
                   <= 2 * config.EXPECTED['structure_all_s']))
    return pd.DataFrame(rows, columns=_pairs.T02_COLUMNS)


# --------------------------------------------------------------------------- #
# self-check                                                                  #
# --------------------------------------------------------------------------- #

def _hydrogen_control(pdb_path='1PQ1_hm.pdb'):
    """The spec's own H-strip evidence, both numbers: residues with SASA < 1 with
    hydrogens present vs stripped (spec Sec.3: 37 vs 15 on 1PQ1)."""
    p = os.path.join(PATHS.structures, pdb_path)
    sr = ShrakeRupley(probe_radius=SASA_PROBE_RADIUS, n_points=SASA_N_POINTS)
    out = {}
    for label, strip in (('with_H', False), ('heavy_only', True)):
        t0 = time.time()
        _st, m, n_h = load_heavy_model(p, strip_h=strip)
        sr.compute(m, level='R')
        res = list(m.get_residues())
        out[label] = dict(n_residues=len(res), n_atoms=len(list(m.get_atoms())),
                          n_H_stripped=n_h,
                          n_sasa_lt_1=int(sum(1 for r in res if r.sasa < 1.0)),
                          wall_s=round(time.time() - t0, 3))
    a, b = out['with_H']['n_sasa_lt_1'], out['heavy_only']['n_sasa_lt_1']
    out['ratio'] = round(a / float(b), 3) if b else float('inf')
    return out


def _selfcheck():
    config.assert_env()
    print('=' * 100)
    print('cliff/structure.py self-check -- BGYM-CLIFF v1 stage 1')
    print('  env      ', config.assert_env())
    print('  input    ', PATHS.bgym_input)
    print('  git      ', _pairs.git_provenance())
    print('=' * 100)

    # ---- G0-style H-strip control ---------------------------------------- #
    print('\n[1] HYDROGEN STRIP CONTROL (spec Sec.3: 37 vs 15 on 1PQ1)')
    hc = _hydrogen_control()
    for k in ('with_H', 'heavy_only'):
        d = hc[k]
        print('    %-11s atoms=%5d  H_stripped=%5d  residues=%3d  '
              'SASA<1 -> %3d   (%.2f s)'
              % (k, d['n_atoms'], d['n_H_stripped'], d['n_residues'],
                 d['n_sasa_lt_1'], d['wall_s']))
    print('    distortion of the buried tail: %.2fx  (spec: 2.5x, 37 vs 15)'
          % hc['ratio'])

    # ---- stage 1 ---------------------------------------------------------- #
    ids = structural_assays()
    print('\n[2] STAGE 1 -- %d assays with a PDB (%d PDB files on disk)'
          % (len(ids), len(set(ASSAYS[i].pdb_file for i in ids))))
    missing = [k for k in config.ALL_ASSAYS if k not in ids]
    print('    no PDB in structures/ (spec G-OPT): %s' % (missing,))
    t0 = time.time()
    res = stage1(ids, verbose=False, force=True)
    wall = time.time() - t0
    t09, t09all, bias = res['T09'], res['T09_all_residues'], res['bias']
    print('    T09 rows (mutated positions) : %d   (spec 2,220)' % len(t09))
    print('    mutation instances recovered  : %d   (G1: 1,173,273 over the 25 '
          'registered files)' % int(t09['n_variants_at_site'].values.sum()))
    print('    per-(assay,residue) rows      : %d   (spec denominator 9,493)'
          % len(t09all))
    print('    wall %.1f s (spec 37.9 s), peak RSS %.2f GB' % (wall, _rss_gb()))

    print('\n[3] LEVY CLASS COMPOSITION')
    for name, frame in (('mutated positions (2,220)', t09),
                        ('all residues (9,493)', t09all)):
        vc = frame['levy_class'].value_counts()
        print('    %-26s %s' % (name, '  '.join(
            '%s=%d(%.1f%%)' % (c, vc.get(c, 0), 100.0 * vc.get(c, 0) / len(frame))
            for c in LEVY_CLASSES)))

    print('\n[4] INTERFACE DEFINITIONS (spec Sec.1.5)')
    all_ = t09all
    dz = all_['is_iface_dsasa'].values.astype(bool)
    mh = all_['min_heavy_dist'].values
    cbv = all_['cb_dist'].values
    for cut in (5.0, 6.0, 6.07, 8.0):
        f = mh < cut
        print('    min-heavy < %-5.2f A : flagged %5d   recall vs dSASA>1 %4d/%4d '
              '= %.4f   precision %.4f'
              % (cut, int(f.sum()), int((f & dz).sum()), int(dz.sum()),
                 (f & dz).sum() / max(dz.sum(), 1),
                 (f & dz).sum() / max(f.sum(), 1)))
    fcb = cbv < THRESH['C4_cb_dist_banned_A']
    print('    BANNED Cb-Cb < 8.0 A : flagged %5d   recall vs dSASA>1 %4d/%4d '
          '= %.4f   precision %.4f   (spec: 911 flagged, 825/1050)'
          % (int(np.nansum(fcb)), int(np.nansum(fcb & dz)), int(dz.sum()),
             np.nansum(fcb & dz) / max(dz.sum(), 1),
             np.nansum(fcb & dz) / max(np.nansum(fcb), 1)))
    print('    max min-heavy over dSASA>1 residues: %.4f A  (spec 6.07)'
          % mh[dz].max())
    d0 = all_['dsasa'].values > 0.0
    print('    dSASA > 0 : %d   dSASA > 1 A^2 : %d   =>  the 1 A^2 cut is %s; '
          'the burial-defined interface IS Levy {support,rim,core}'
          % (int(d0.sum()), int(dz.sum()),
             'NON-BINDING on this benchmark' if int(d0.sum()) == int(dz.sum())
             else 'BINDING'))
    dep = all_['depth_A'].values
    print('    burial depth_A: 0 for %d/%d (surface), median of the rest %.2f A, '
          'max %.2f A' % (int((dep == 0).sum()), len(dep),
                          float(np.median(dep[dep > 0])), float(dep.max())))
    rr = all_['rsa_iso_raw'].values
    print('    rsa_iso_raw > 1.0 : %d / %d  max %.4f  (spec 35 / 9,493, max 1.36)'
          % (int((rr > 1.0).sum()), len(rr), rr.max()))
    rre = np.concatenate([
        (all_.loc[all_['DMS_id'] == d, 'sasa_iso'].values
         / np.array([TIEN2013_EMPIRICAL[a] for a in
                     all_.loc[all_['DMS_id'] == d, 'wt_aa'].values]))
        for d in pd.unique(all_['DMS_id'])])
    print('    same with the Tien EMPIRICAL column: %d / %d  max %.4f'
          % (int((rre > 1.0).sum()), len(rre), rre.max()))

    print('\n[5] PER-ASSAY design vs background interface fraction')
    print('    %-42s %-12s %5s %5s %8s %8s %7s %8s %8s %s'
          % ('DMS_id', 'chains', 'nDes', 'nBg', 'design', 'bg', 'bias',
             'spec_des', 'spec_bg', 'C4S'))
    for r in bias.sort_values('DMS_id').to_dict('records'):
        print('    %-42s %-12s %5d %5d %8.4f %8.4f %6.2fx %8s %8s %s'
              % (r['DMS_id'], r['mutated_chains'], r['n_design'], r['n_bg'],
                 r['design_iface_frac'], r['bg_iface_frac'],
                 r['iface_bias_factor'],
                 '-' if r['design_iface_frac_spec'] != r['design_iface_frac_spec']
                 else '%.3f' % r['design_iface_frac_spec'],
                 '-' if r['bg_iface_frac_spec'] != r['bg_iface_frac_spec']
                 else '%.3f' % r['bg_iface_frac_spec'],
                 'Y' if r['eligible_C4S'] else '.'))

    print('\n[6] CONSTANT seq->pdb OFFSETS (background sets ONLY; the mutation '
          'lookup never uses them)')
    off = res['offsets']
    print('    %-42s %-5s %7s %5s %7s %8s %8s %s'
          % ('DMS_id', 'chain', 'offset', 'const', 'seq_len', 'mismatch',
             'absent', 'clean / why'))
    for r in off.to_dict('records'):
        print('    %-42s %-5s %7d %5s %7d %8d %8d %s %s'
              % (r['DMS_id'], r['chain'], r['offset'],
                 'Y' if r['offset_constant'] else 'N', r['seq_len'],
                 r['n_mismatch'], r['n_absent'],
                 'YES' if r['verified_clean'] else 'NO ', r['reason']))
    poi_of = {k: ASSAYS[k].poi for k in res['annot']}
    dist = off.assign(pdb_chain=[poi_of[d] + '-' + c for d, c
                                 in zip(off['DMS_id'], off['chain'])])
    dd = dist.drop_duplicates('pdb_chain')
    print('    verified clean: %d of %d (assay x chain) | %d of %d distinct '
          '(pdb x chain)   [spec: 19 clean]'
          % (int(off['verified_clean'].sum()), len(off),
             int(dd['verified_clean'].sum()), len(dd)))
    print('    the spec\'s named failures, measured:')
    for d, c in (('5A12_Ang2_fitness_4ZFG', 'H'), ('5A12_Ang2_fitness_4ZFG', 'L'),
                 ('5A12_VEGF_fitness_4ZFF', 'H'), ('5A12_VEGF_fitness_4ZFF', 'L')):
        r = off[(off['DMS_id'] == d) & (off['chain'] == c)].iloc[0]
        print('      %-26s-%s  offset %+d  %3d mismatches of %3d  (spec 4ZFG-H: '
              '168 of 219)' % (ASSAYS[d].poi, c, r['offset'], r['n_mismatch'],
                               r['seq_len']))
    b1 = off[(off['DMS_id'] == 'BH3_Bcl-xL_normed_1PQ1')].iloc[0]['offset']
    b2 = off[(off['DMS_id'] == 'BH3_Mcl-1_normed_3KZ0')].iloc[0]['offset']
    print('    independent G1b cross-check: 1PQ1-B seq->pdb %+d, 3KZ0-C seq->pdb '
          '%+d  =>  seq offset %+d and pdb offset %+d  (spec G1b: -2 and -84)'
          % (b1, b2, 0 - 2, b2 - b1 - 2))

    print('\n[6b] KRAS_SOS1_8BE4: the ONE pre-declared value that does not '
          'reproduce')
    a8 = res['annot']['KRAS_SOS1_norfitness_8BE4']
    m8 = res['mut']['KRAS_SOS1_norfitness_8BE4']
    f5 = a8['min_heavy_dist'].values < THRESH['C4_iface_dist_A']
    fz = a8['is_iface_dsasa'].values.astype(bool)
    for lab, msk in (('chain R = KRAS, the MUTATED side', a8['chain'].values == 'R'),
                     ('chain S = SOS1, the PARTNER side', a8['chain'].values == 'S')):
        print('    %-36s 5.0 A %3d/%3d = %.4f   dSASA>1 %3d/%3d = %.4f'
              % (lab, int(f5[msk].sum()), int(msk.sum()), f5[msk].mean(),
                 int(fz[msk].sum()), int(msk.sum()), fz[msk].mean()))
    print('    design (163 mutated positions of chain R): 5.0 A %.4f, '
          'dSASA>1 %.4f  (spec design 0.264)'
          % (m8['is_iface_5A'].values.mean(), m8['is_iface_dsasa'].values.mean()))
    print("    => the spec's bg 0.110 is chain S's 5.0 A fraction (%.4f), i.e. the "
          'PARTNER side.' % f5[a8['chain'].values == 'S'].mean())
    print('    => the library mutates 163 of the 165 resolved chain-R residues, so '
          'design ~ bg')
    print('       BY CONSTRUCTION (bias 1.01x at EVERY cut 4-8 A).  A 2.4x is '
          'arithmetically')
    print('       impossible here; it is possible in KRAS_RAF1 (63 of 168 '
          'positions) and reproduces')
    print('       there exactly: design %.4f vs bg %.4f = %.2fx (spec 0.238 / '
          '0.101 / 2.4x).'
          % tuple(res['bias'].set_index('DMS_id')
                  .loc['KRAS_RAF1_norfitness_6VJJ',
                       ['design_iface_frac', 'bg_iface_frac',
                        'iface_bias_factor']].values))

    print('\n[7] SPEC-NAMED STRUCTURAL FACTS')
    v = t09[t09['DMS_id'] == '5A12_VEGF_fitness_4ZFF']
    print('    5A12_VEGF within 6.4 A of VEGF : %d / %d   (spec 0 / 9); '
          'min min_heavy = %.2f A, max dSASA = %.3f A^2'
          % (int((v['min_heavy_dist'].values < 6.4).sum()), len(v),
             v['min_heavy_dist'].values.min(), v['dsasa'].values.max()))
    z = t09[t09['DMS_id'] == 'Z-domain_ZpA963_HL1_fitness_2M5A']
    print('    Z-ZpA963_HL1 interface         : %d / %d   (spec 6 / 6)'
          % (int(z['is_iface_5A'].values.sum()), len(z)))
    a, b = 'KRAS_RAF1_norfitness_6VJJ', 'KRAS_RAF1-RBD_norfitness_6VJJ'
    ca = res['annot'][a].to_csv(index=False).encode()
    cb2 = res['annot'][b].to_csv(index=False).encode()
    import hashlib
    print('    G11 KRAS_RAF1 vs RAF1-RBD      : %s  (md5 %s / %s)'
          % ('BYTE-IDENTICAL' if ca == cb2 else 'DIFFER',
             hashlib.md5(ca).hexdigest()[:12], hashlib.md5(cb2).hexdigest()[:12]))
    ta = t09[t09['DMS_id'] == a].drop(columns=['DMS_id'])
    tb = t09[t09['DMS_id'] == b].drop(columns=['DMS_id'])
    print('    G11 T09 rows                   : %d vs %d positions, '
          'shared %d, structural columns identical on the shared set: %s'
          % (len(ta), len(tb),
             len(set(zip(ta['chain'], ta['resseq'])) & set(zip(tb['chain'], tb['resseq']))),
             _shared_identical(ta, tb)))

    print('\n[8] GATE TABLE (structural half of T02)')
    g = res['gates']
    with pd.option_context('display.width', 250, 'display.max_colwidth', 62,
                           'display.max_rows', 200):
        print(g[['gate_id', 'gate_name', 'assay', 'expected', 'observed',
                 'PASS/FAIL']].to_string(index=False))
    n_fail = int((g['PASS/FAIL'] == 'FAIL').sum())
    print('\n    %d PASS / %d FAIL / %d reported-only'
          % (int((g['PASS/FAIL'] == 'PASS').sum()), n_fail,
             int((g['PASS/FAIL'] == '').sum())))

    print('\n[9] MANIFEST + CACHE ROUND-TRIP')
    bad = _pairs.verify_manifest()
    print('    verify_manifest(): %s' % ('CLEAN' if not bad else bad))
    n_rt = n_rt_bad = 0
    for k, a in res['annot'].items():
        sp = ASSAYS[k]
        cp = os.path.join(PATHS.structure_cache, '%s_%s_%s.npz'
                          % (sp.poi, sp.side0_chains, sp.side1_chains))
        b = _read_annot_cache(cp)
        n_rt += 1
        if (list(b.columns) != list(a.columns)
                or b.to_csv(index=False) != a.to_csv(index=False)):
            n_rt_bad += 1
    print('    npz round-trip: %d frames, %d differ from the in-memory frame'
          % (n_rt, n_rt_bad))
    print('    structure cache  : %d npz, %.2f MB'
          % (len(os.listdir(PATHS.structure_cache)),
             sum(os.path.getsize(os.path.join(PATHS.structure_cache, f))
                 for f in os.listdir(PATHS.structure_cache)) / 1e6))
    print('    T09 written      : %s (%d rows x %d cols)'
          % (os.path.join(PATHS.artifacts, 'T09_structure_sites.csv'),
             len(t09), len(t09.columns)))
    print('\n[10] TIMING per assay (spec budget 37.9 s total, one core)')
    bn = res['bench'].sort_values('wall_total_s', ascending=False)
    for r in bn.to_dict('records'):
        print('    %-42s res=%4d heavy=%5d  annot %6.2fs  total %6.2fs'
              % (r['DMS_id'], r['n_residues'], r['n_heavy'],
                 r['wall_annot_s'], r['wall_total_s']))
    print('    TOTAL annot %.1f s   TOTAL stage1 %.1f s   peak RSS %.2f GB'
          % (bn['wall_annot_s'].sum(), bn['wall_total_s'].sum(), _rss_gb()))
    print('=' * 100)
    return res


def _shared_identical(ta, tb):
    ka = list(zip(ta['chain'], ta['resseq'], ta['icode']))
    kb = list(zip(tb['chain'], tb['resseq'], tb['icode']))
    shared = sorted(set(ka) & set(kb))
    cols = ['levy_class', 'rsa_iso', 'rsa_cplx', 'dsasa', 'min_heavy_dist',
            'cb_dist', 'is_iface_5A', 'is_iface_dsasa']
    ia = {k: i for i, k in enumerate(ka)}
    ib = {k: i for i, k in enumerate(kb)}
    for c in cols:
        va = ta[c].values[[ia[k] for k in shared]]
        vb = tb[c].values[[ib[k] for k in shared]]
        if not np.array_equal(va, vb):
            return 'NO (%s)' % c
    return 'YES'


if __name__ == '__main__':
    _selfcheck()
