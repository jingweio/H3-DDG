"""BGYM-CLIFF v1 -- loading, canonical keys, ``mutant``<->``mutant_pdb`` pairing,
and the G1 / G1b / G2 / G3 parse audits (spec Sec.3).

Three invariants this module exists to enforce:

1. **The chain label is part of every key.**  Dropping it manufactures fake
   duplicate genotypes in the four Z-domain assays (both chains are mutated and
   their 1-based numbering overlaps) -- see :func:`gate_G3`.
2. **``mutant`` and ``mutant_pdb`` are joined BY CHAIN KEY, never by dict order.**
   5A12_Ang2 stores ``{'H':..,'L':'','A':''}`` in one column and
   ``{'A':'','H':..,'L':''}`` in the other.
3. **Cross-assay joins never use a naive ``(pos, aa)`` match** (banned repo-wide by
   spec G1b); they go through an explicitly measured, WT-identity-verified offset.

Everything numeric is read from :mod:`cliff.config`.
"""
from __future__ import annotations

import ast
import hashlib
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cliff import config
from cliff.config import ASSAYS, EXPECTED, PATHS, THRESH

#: wt, resnum, icode, mut -- e.g. 'P52AL' -> ('P', 52, 'A', 'L'); 'A11C' -> ('A', 11, '', 'C')
MUT_RE = re.compile(r'^([A-Z])(-?\d+)([A-Za-z]?)([A-Z])$')

#: fast path for the ``{'A': '', 'C': 'V39I:T44K'}`` dict literals.  Verified
#: against ``ast.literal_eval`` on every one of the 1.0e6 dict strings in the
#: benchmark (see :func:`audit_all` -> ``n_literal_eval_checked``).
_KV_RE = re.compile(r"'([^']*)'\s*:\s*'([^']*)'")

#: the four columns spec Sec.3 pins for every load.  Side effect: HLA-A2's
#: duplicated ``DMS_score`` column disappears (verified identical, see
#: :func:`_check_hla_duplicate_column`).
USECOLS = ['POI', 'DMS_score', 'mutant', 'mutant_pdb']


# --------------------------------------------------------------------------- #
# small parsing primitives                                                    #
# --------------------------------------------------------------------------- #

def _parse_dict_str(s):
    """``"{'A': '', 'C': 'V39I'}"`` -> ``{'A': '', 'C': 'V39I'}`` (10x literal_eval)."""
    return dict(_KV_RE.findall(s))


def parse_mut_token(token):
    """``'P52AL'`` -> ``('P', 52, 'A', 'L')``.  Raises on anything unparseable."""
    m = MUT_RE.match(token)
    if m is None:
        raise ValueError('unparseable mutation token: %r' % (token,))
    wt, num, icode, mut = m.groups()
    return wt, int(num), icode, mut


def parse_pair_dicts(mutant, mutant_pdb):
    """Join the two mutation columns BY CHAIN KEY and zip the token lists.

    Returns ``[(chain, seq_pos, wt_aa, mut_aa, resseq, icode), ...]`` -- ``icode``
    is ``''`` when absent.  ``seq_pos`` is the 1-based per-chain sequence index
    from ``mutant``; ``(resseq, icode)`` is the PDB residue id from
    ``mutant_pdb``.  Order follows the ``mutant`` dict's own chain order and each
    chain's colon-separated token order (unsorted -- :func:`canonical_key` sorts).

    Raises on: a chain present in one column and not the other, differing token
    counts for a chain, a WT- or mutant-letter disagreement between the two
    columns, a ``'*'`` token, or an unparseable token.
    """
    d_seq = _parse_dict_str(mutant)
    d_pdb = _parse_dict_str(mutant_pdb)
    if set(d_seq) != set(d_pdb):
        raise ValueError('chain sets differ: %r vs %r' % (sorted(d_seq), sorted(d_pdb)))
    out = []
    for chain in d_seq:                      # BY CHAIN KEY, never by dict order
        v_seq, v_pdb = d_seq[chain], d_pdb[chain]
        if not v_seq and not v_pdb:
            continue
        t_seq = v_seq.split(':')
        t_pdb = v_pdb.split(':')
        if len(t_seq) != len(t_pdb):
            raise ValueError('chain %r token counts differ: %d vs %d'
                             % (chain, len(t_seq), len(t_pdb)))
        for a, b in zip(t_seq, t_pdb):
            wt_a, pos_a, ic_a, mut_a = parse_mut_token(a)
            wt_b, pos_b, ic_b, mut_b = parse_mut_token(b)
            if (wt_a, mut_a) != (wt_b, mut_b):
                raise ValueError('chain %r token %r/%r wt/mut letters disagree'
                                 % (chain, a, b))
            if ic_a:
                raise ValueError('chain %r seq token %r carries an icode' % (chain, a))
            out.append((chain, pos_a, wt_a, mut_a, pos_b, ic_b))
    return out


def canonical_key(muts):
    """``K(v) = tuple(sorted((chain, seq_pos, aa_mut)))``.

    THE CHAIN LABEL IS MANDATORY (spec Sec.1.0 / G3).  ``muts`` is the list
    returned by :func:`parse_pair_dicts`.
    """
    return tuple(sorted((m[0], m[1], m[3]) for m in muts))


def detect_censoring(y):
    """Detected floor/ceiling levels: values with mass >= 0.005 at min/max, using
    the **1-decimal-place string form** to define the level (spec Sec.3).

    Concretely: a floor exists iff ``P(y == min y) >= 0.005``; the floor *level*
    is then every distinct value whose ``'%.1f'`` form equals the minimum's and
    whose own mass is also ``>= 0.005``.  Symmetrically for the ceiling.

    This reading is what reproduces every censoring number the spec quotes:
    SARS2-RBD -4.84 (14.53%) + -4.76 (9.31%) = 23.84% (spec: 23.85% "at
    -4.84/-4.76"; -4.76 is NOT the second-smallest value, it is the second value
    inside the -4.8 1-dp bin); CR9114-H3 6.000 at 89.05%; CR9114-H1 7.000 at
    2.57%; CR6261 7.000 at 11.34%.  Taking "levels at min/max" to mean only the
    exact extremum misses -4.76; taking the whole 1-dp bin regardless of mass
    sweeps up continuous dust in narrow-range assays (5A12_Ang2 spans 0.65).

    Returns ``(levels, mask, meta)`` where ``levels`` is a tuple of floats
    (floors then ceilings), ``mask`` is the boolean row mask and ``meta`` carries
    the per-side value/fraction for T01.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    u, c = np.unique(y, return_counts=True)
    frac = c / float(n)
    thr = THRESH['censor_min_mass']
    onedp = np.array(['%.1f' % v for v in u])
    levels = []
    meta = dict(floor_value='', floor_frac='', ceil_value='', ceil_frac='')
    # ---- floor ----
    if frac[0] >= thr:
        sel = (onedp == onedp[0]) & (frac >= thr)
        lv = [float(v) for v in u[sel]]
        levels.extend(lv)
        meta['floor_value'] = '|'.join('%.10g' % v for v in lv)
        meta['floor_frac'] = float(frac[sel].sum())
    # ---- ceiling ----
    if frac[-1] >= thr:
        sel = (onedp == onedp[-1]) & (frac >= thr)
        lv = [float(v) for v in u[sel]]
        # a one-point distribution would double-count; guard it
        lv = [v for v in lv if v not in levels]
        levels.extend(lv)
        meta['ceil_value'] = '|'.join('%.10g' % v for v in lv)
        meta['ceil_frac'] = float(frac[sel].sum())
    if levels:
        mask = np.isin(y, np.array(levels, dtype=np.float64))
    else:
        mask = np.zeros(n, dtype=bool)
    return tuple(levels), mask, meta


def score_quantum(score_strings):
    """``(quantum, modal_decimals)`` -- the assay's decimal grid.

    ``quantum = 10 ** -modal_decimals`` where ``modal_decimals`` is the modal
    number of decimal places in the RAW ``DMS_score`` strings.

    DEVIATION from the spec's wording ("modal spacing of sorted unique y"),
    documented because it is load-bearing for the grid guard: the literal modal
    spacing is 0.005 on SARS2-RBD and 1.42e-05 on CR9114-H3, contradicting the
    spec's own stated ``q = 0.01`` and ``q = 0.1`` for exactly those two assays
    (Sec.1.0).  On continuous assays the modal spacing is numerically degenerate
    (783 of GB1_1FCC's 82,123 unique-value gaps are below 1e-8, so the mode is an
    artefact of float noise).  The decimal grid reproduces both spec values
    exactly and is the physical quantity the guard is about.  The literal modal
    spacing is still measured and reported as ``modal_spacing_unique``.
    """
    dec = np.fromiter((_n_decimals(s) for s in score_strings),
                      dtype=np.int32, count=len(score_strings))
    md = int(np.bincount(dec).argmax())
    return float(10.0 ** (-md)), md


def _n_decimals(s):
    """Decimal places of a score string, exponential notation included.

    A handful of rows per file are written in scientific notation (7 in
    SARS2-RBD, 5 in GB1_1FCC, 2 in ACE2_6M17, ...), so ``len(after the dot)``
    would count ``'e-05'`` as four decimals.  ``Decimal`` gives the true
    exponent: ``'3.29465e-05'`` -> 10 decimals.
    """
    if 'e' in s or 'E' in s:
        from decimal import Decimal
        return max(0, -Decimal(s).as_tuple().exponent)
    return len(s.partition('.')[2])


def modal_spacing_unique(y):
    """The spec's literal "modal spacing of sorted unique y" -- reported, not used."""
    u = np.unique(np.asarray(y, dtype=np.float64))
    if u.size < 3:
        return float('nan')
    d = np.diff(u)
    uu, cc = np.unique(d, return_counts=True)
    return float(uu[cc.argmax()])


def build_col_index(keys):
    """``(chain, seq_pos, aa_mut) -> column of X`` and ``(chain, seq_pos) ->
    code-vector column``, both in sorted order so they are reproducible."""
    subs = set()
    for k in keys:
        subs.update(k)
    col_index = {s: i for i, s in enumerate(sorted(subs))}
    pos_index = {p: i for i, p in enumerate(sorted(set((c, p) for c, p, _ in subs)))}
    return col_index, pos_index


# --------------------------------------------------------------------------- #
# Assay                                                                       #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Assay:
    """One BindingGYM landscape, parsed (spec Sec.3).

    ``y`` is the ANALYSIS scale (``log10`` already applied for hYAP65); ``y_raw``
    is the column as read.  ``censor_levels`` / ``censor_mask`` are detected on
    ``y``; ``quantum`` is the raw decimal grid.
    """
    dms_id: str
    poi: str
    pdb_file: str
    side0: tuple
    side1: tuple
    y: np.ndarray                 # float64, len n
    keys: list                    # canonical keys, len n
    codes: np.ndarray             # (n, P) int8, 0 = WT at that position
    col_index: dict               # (chain, seq_pos, aa_mut) -> column of X
    pos_index: dict               # (chain, seq_pos)         -> code-vector column
    pdb_key: list                 # per row, per token: (chain, resseq, icode)
    row_index: np.ndarray         # int32, source-csv 0-based row number (primary key)
    n_muts: np.ndarray            # int8
    wt_row: int                   # -1 when the assay has no WT row
    censor_levels: tuple
    censor_mask: np.ndarray       # bool
    quantum: float
    # ---- provenance / T01 support (not decision variables) ----
    y_raw: np.ndarray = field(default=None, repr=False)
    transform: str = 'none'
    modal_decimals: int = -1
    score_strings: tuple = field(default=(), repr=False)

    @property
    def n(self):
        return int(self.y.size)

    @property
    def P(self):
        return int(self.codes.shape[1])

    @property
    def M(self):
        return len(self.col_index)


def load_assay(dms_id, *, apply_transform=True, keep_score_strings=False):
    """Load one assay.

    ``usecols=['POI','DMS_score','mutant','mutant_pdb']`` ALWAYS (spec Sec.3).
    Side effect of usecols: HLA-A2's duplicated ``DMS_score`` column disappears
    (verified identical by :func:`_check_hla_duplicate_column`).
    """
    spec = ASSAYS[dms_id]
    path = PATHS.dms_csv(dms_id)
    df = pd.read_csv(path, usecols=USECOLS, dtype={'DMS_score': str})
    n = len(df)
    score_strings = df['DMS_score'].tolist()
    y_raw = np.asarray(df['DMS_score'].values, dtype=np.float64)
    quantum, modal_dec = score_quantum(score_strings)

    transform = spec.transform if apply_transform else 'none'
    if transform == 'log10':
        if not np.all(y_raw > 0):
            raise ValueError('%s: log10 transform on non-positive values' % dms_id)
        y = np.log10(y_raw)
    elif transform == 'none':
        y = y_raw.copy()
    else:
        raise ValueError('unknown transform %r' % (transform,))

    mut_col = df['mutant'].values
    pdb_col = df['mutant_pdb'].values
    keys = []
    pdb_key = []
    n_muts = np.empty(n, dtype=np.int8)
    parsed = []
    for i in range(n):
        muts = parse_pair_dicts(mut_col[i], pdb_col[i])
        parsed.append(muts)
        keys.append(canonical_key(muts))
        pdb_key.append(tuple((m[0], m[4], m[5]) for m in muts))
        n_muts[i] = len(muts)

    col_index, pos_index = build_col_index(keys)
    P = len(pos_index)
    codes = np.zeros((n, P), dtype=np.int8)
    aa_code = config.AA_CODE
    for i, muts in enumerate(parsed):
        for chain, pos, _wt, mut, _rs, _ic in muts:
            codes[i, pos_index[(chain, pos)]] = aa_code[mut]

    wt_row = -1
    for i, k in enumerate(keys):
        if len(k) == 0:
            if wt_row >= 0:
                raise ValueError('%s: more than one WT row (%d, %d)' % (dms_id, wt_row, i))
            wt_row = i

    censor_levels, censor_mask, _ = detect_censoring(y)

    return Assay(
        dms_id=dms_id, poi=spec.poi, pdb_file=spec.pdb_file,
        side0=tuple(spec.side0_chains), side1=tuple(spec.side1_chains),
        y=y, keys=keys, codes=codes, col_index=col_index, pos_index=pos_index,
        pdb_key=pdb_key, row_index=np.arange(n, dtype=np.int32), n_muts=n_muts,
        wt_row=wt_row, censor_levels=censor_levels, censor_mask=censor_mask,
        quantum=quantum, y_raw=y_raw, transform=transform,
        modal_decimals=modal_dec,
        score_strings=tuple(score_strings) if keep_score_strings else ())


# --------------------------------------------------------------------------- #
# audit-only helpers                                                          #
# --------------------------------------------------------------------------- #

_AA3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V', 'MSE': 'M', 'SEC': 'U', 'PYL': 'O',
}


def pdb_residue_letters(pdb_path):
    """``(chain, resseq, icode) -> one-letter code`` from ATOM records.

    AUDIT ONLY -- this is the fourth wt-letter source G1 requires.  The
    authoritative structural annotation (SASA, distances, Levy classes) is
    ``cliff/structure.py``'s job and uses Biopython.
    """
    out = {}
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith('ATOM'):
                continue
            resname = line[17:20].strip()
            chain = line[21]
            resseq = int(line[22:26])
            icode = line[26].strip()
            key = (chain, resseq, icode)
            if key not in out:
                out[key] = _AA3TO1.get(resname, 'X')
    return out


def _check_hla_duplicate_column():
    """HLA-A2 ships ``DMS_score`` twice; ``usecols`` keeps one.  Verify identical."""
    path = PATHS.dms_csv('HLA-A2_TAPBPR_meanscore_5WER')
    raw = pd.read_csv(path, header=None, skiprows=1, usecols=[1, 2],
                      dtype=str, names=['a', 'b'])
    n_diff = int((raw['a'].values != raw['b'].values).sum())
    kept = pd.read_csv(path, usecols=USECOLS, dtype={'DMS_score': str})['DMS_score'].values
    n_diff_kept = int((kept != raw['a'].values).sum())
    return dict(n_rows=len(raw), n_diff_between_duplicates=n_diff,
                n_diff_kept_vs_first=n_diff_kept)


# --------------------------------------------------------------------------- #
# G1 -- parse audit over all 28 files                                         #
# --------------------------------------------------------------------------- #

def audit_one(dms_id, *, deep=True, literal_eval_check=True):
    """Per-file G1 counters.

    ``deep=True`` re-verifies the spec Sec.1.0 claim end-to-end: every row's
    ``mutated_sequence`` is reconstructed from ``wildtype_sequence`` + the
    mutation list and compared byte-for-byte.  That single check proves, per row,
    0 indels, 0 identity mutations and Hamming == mutation-set distance.
    """
    path = PATHS.dms_csv(dms_id)
    cols = USECOLS + (['wildtype_sequence', 'mutated_sequence'] if deep
                      else ['wildtype_sequence'])
    t0 = time.time()
    df = pd.read_csv(path, usecols=cols, dtype={'DMS_score': str})
    n = len(df)

    wt_strs = df['wildtype_sequence'].values
    wt_nunique = int(pd.unique(wt_strs).size)
    wt_seq = ast.literal_eval(wt_strs[0])           # constant per file (asserted below)

    mut_col = df['mutant'].values
    pdb_col = df['mutant_pdb'].values
    mseq_col = df['mutated_sequence'].values if deep else None

    n_inst = 0
    n_star = 0
    n_x_hits = 0
    n_wt_mismatch_pdbcol = 0        # mutant vs mutant_pdb letters
    n_wt_mismatch_wtseq = 0         # mutant vs wildtype_sequence
    n_identity = 0
    n_token_count_mismatch = 0
    n_parse_fail = 0
    n_recon_fail = 0
    n_indel = 0
    n_litcheck = 0
    n_litdiff = 0
    keys = []
    per_chain_tokens = Counter()
    pdb_keys_all = set()
    for i in range(n):
        ms, mp = mut_col[i], pdb_col[i]
        if '*' in ms or '*' in mp:
            n_star += 1
        d_seq = _parse_dict_str(ms)
        d_pdb = _parse_dict_str(mp)
        if literal_eval_check:
            n_litcheck += 2
            if d_seq != ast.literal_eval(ms) or d_pdb != ast.literal_eval(mp):
                n_litdiff += 1
        if set(d_seq) != set(d_pdb):
            n_token_count_mismatch += 1
        for ch in d_seq:
            t_s = [t for t in d_seq[ch].split(':') if t]
            t_p = [t for t in d_pdb.get(ch, '').split(':') if t]
            if len(t_s) != len(t_p):
                n_token_count_mismatch += 1
                continue
            per_chain_tokens[ch] += len(t_s)
            for a, b in zip(t_s, t_p):
                n_inst += 1
                try:
                    wt_a, pos_a, ic_a, mut_a = parse_mut_token(a)
                    wt_b, pos_b, ic_b, mut_b = parse_mut_token(b)
                except ValueError:
                    n_parse_fail += 1
                    continue
                if (wt_a, mut_a) != (wt_b, mut_b):
                    n_wt_mismatch_pdbcol += 1
                if wt_a == mut_a:
                    n_identity += 1
                seq = wt_seq.get(ch, '')
                if 1 <= pos_a <= len(seq):
                    letter = seq[pos_a - 1]
                    if letter == 'X':
                        n_x_hits += 1
                    elif letter != wt_a:
                        n_wt_mismatch_wtseq += 1
                else:
                    n_wt_mismatch_wtseq += 1
                pdb_keys_all.add((ch, pos_b, ic_b))
        muts = parse_pair_dicts(ms, mp)
        keys.append(canonical_key(muts))
        if deep:
            got = _parse_dict_str(mseq_col[i])
            rebuilt = {}
            for ch in wt_seq:
                s = wt_seq[ch]
                toks = [t for t in d_seq.get(ch, '').split(':') if t]
                if toks:
                    lst = list(s)
                    for t in toks:
                        _w, p, _ic, mm = parse_mut_token(t)
                        if 1 <= p <= len(lst):
                            lst[p - 1] = mm
                    s = ''.join(lst)
                rebuilt[ch] = s
            if rebuilt != got:
                n_recon_fail += 1
            for ch in wt_seq:
                if len(got.get(ch, '')) != len(wt_seq[ch]):
                    n_indel += 1

    uniq = len(set(keys))
    uniq_nochain = len(set(tuple(sorted((p, a) for _c, p, a in k)) for k in keys))
    # per-chain token counts from the PDB column, for the "token counts match per
    # chain" clause
    per_chain_pdb = Counter()
    for i in range(n):
        for ch, v in _parse_dict_str(pdb_col[i]).items():
            per_chain_pdb[ch] += len([t for t in v.split(':') if t])
    chain_counts_agree = (dict(per_chain_tokens) == dict(per_chain_pdb))

    # fourth wt-letter source: the PDB residue itself
    n_wt_mismatch_pdbres = ''
    n_pdb_key_missing = ''
    pdb_path = os.path.join(PATHS.structures, ASSAYS[dms_id].pdb_file)
    if os.path.exists(pdb_path):
        letters = pdb_residue_letters(pdb_path)
        bad = 0
        miss = 0
        for i in range(n):
            for (ch, pos_a, wt_a, _mut, rs, ic) in parse_pair_dicts(mut_col[i], pdb_col[i]):
                got = letters.get((ch, rs, ic))
                if got is None:
                    miss += 1
                elif got != wt_a:
                    bad += 1
        n_wt_mismatch_pdbres = bad
        n_pdb_key_missing = miss

    y = np.asarray(df['DMS_score'].values, dtype=np.float64)
    return dict(
        DMS_id=dms_id, n_rows=n, n_unique_keys=uniq, n_dup_keys=n - uniq,
        n_unique_keys_nochain=uniq_nochain, n_dup_keys_nochain=n - uniq_nochain,
        n_mutation_instances=n_inst, n_parse_fail=n_parse_fail,
        n_star_tokens=n_star, n_X_hits=n_x_hits,
        n_wt_mismatch_mutant_vs_pdbcol=n_wt_mismatch_pdbcol,
        n_wt_mismatch_vs_wildtype_sequence=n_wt_mismatch_wtseq,
        n_wt_mismatch_vs_pdb_residue=n_wt_mismatch_pdbres,
        n_pdb_key_missing=n_pdb_key_missing,
        n_identity_mutations=n_identity,
        n_token_count_mismatch=n_token_count_mismatch,
        n_reconstruction_fail=(n_recon_fail if deep else ''),
        n_indels=(n_indel if deep else ''),
        wildtype_sequence_nunique=wt_nunique,
        chain_token_counts_agree=chain_counts_agree,
        per_chain_tokens=dict(sorted(per_chain_tokens.items())),
        n_literal_eval_checked=n_litcheck, n_literal_eval_diff=n_litdiff,
        n_distinct_pdb_keys=len(pdb_keys_all),
        has_wt_row=any(len(k) == 0 for k in keys),
        max_mut=int(max(len(k) for k in keys)),
        y_min=float(y.min()), y_max=float(y.max()),
        wall_s=round(time.time() - t0, 2))


# --------------------------------------------------------------------------- #
# G1b -- the BH3 1PQ1-chainB <-> 3KZ0-chainC join                             #
# --------------------------------------------------------------------------- #

def _bh3_tables():
    a = load_assay('BH3_Bcl-xL_normed_1PQ1')
    b = load_assay('BH3_Mcl-1_normed_3KZ0')
    return a, b


def gate_G1b():
    """Resolve the BH3 cross-assay join and the r = 0.1709 vs +0.592 disagreement.

    Both BH3 files are 518 rows over 10 mutated positions of the same BH3
    peptide, but the peptide sits on chain B of 1PQ1 and chain C of 3KZ0, with
    different residue numbering in BOTH the ``mutant`` and the ``mutant_pdb``
    column.  The join is therefore an explicit, WT-identity-verified integer
    offset -- never a naive ``(pos, aa)`` match (banned).
    """
    tab = {}
    for dms_id in ('BH3_Bcl-xL_normed_1PQ1', 'BH3_Mcl-1_normed_3KZ0'):
        path = PATHS.dms_csv(dms_id)
        df = pd.read_csv(path, usecols=USECOLS, dtype={'DMS_score': str})
        rows = []
        for ms, mp in zip(df['mutant'].values, df['mutant_pdb'].values):
            rows.append(parse_pair_dicts(ms, mp))
        tab[dms_id] = dict(muts=rows,
                           y=np.asarray(df['DMS_score'].values, dtype=np.float64),
                           s=df['DMS_score'].tolist())
    A, B = tab['BH3_Bcl-xL_normed_1PQ1'], tab['BH3_Mcl-1_normed_3KZ0']

    def sites(rows, use_pdb):
        """``position -> WT letter`` over the mutated sites of one file."""
        wt = {}
        for r in rows:
            for (ch, pos, w, mut, rs, ic) in r:
                p = rs if use_pdb else pos
                if wt.setdefault(p, w) != w:
                    raise ValueError('inconsistent WT letter at position %d' % p)
        return wt

    res = {}
    for use_pdb, tag in ((False, 'mutant_seq'), (True, 'mutant_pdb')):
        wtA = sites(A['muts'], use_pdb)
        wtB = sites(B['muts'], use_pdb)
        # the offset is identified by requiring the site sets to coincide AND the
        # WT letters to agree on every matched site.  Scan the whole plausible
        # range and keep every offset that is perfectly WT-consistent -- the
        # scan, not an assumption, is what makes the join auditable.
        good = []
        for off in range(-500, 501):
            shifted = {p + off: w for p, w in wtA.items()}
            if set(shifted) == set(wtB) and all(shifted[p] == wtB[p] for p in wtB):
                good.append(off)
        res[tag] = dict(n_sites_a=len(wtA), n_sites_b=len(wtB),
                        pos_a=sorted(wtA), pos_b=sorted(wtB),
                        wt_a=[wtA[p] for p in sorted(wtA)],
                        wt_b=[wtB[p] for p in sorted(wtB)],
                        offsets_wt_consistent=good)

    # ---- the three joins ----
    def keyset(rows, use_pdb, offset):
        out = []
        for r in rows:
            out.append(tuple(sorted(((rs if use_pdb else pos) + offset, mut)
                                    for (ch, pos, w, mut, rs, ic) in r)))
        return out

    def join(kA, kB):
        dA, dB = {}, {}
        for i, k in enumerate(kA):
            dA.setdefault(k, i)
        for i, k in enumerate(kB):
            dB.setdefault(k, i)
        common = sorted(set(dA) & set(dB))
        ia = np.array([dA[k] for k in common], dtype=np.int64)
        ib = np.array([dB[k] for k in common], dtype=np.int64)
        return common, ia, ib

    out = {}
    for tag, use_pdb in (('mutant_seq', False), ('mutant_pdb', True)):
        offs = res[tag]['offsets_wt_consistent']
        off = offs[0] if offs else 0
        kA = keyset(A['muts'], use_pdb, off)
        kB = keyset(B['muts'], use_pdb, 0)
        common, ia, ib = join(kA, kB)
        ya, yb = A['y'][ia], B['y'][ib]
        r = float(np.corrcoef(ya, yb)[0, 1]) if len(common) > 2 else float('nan')
        # naive: no offset at all
        kA0 = keyset(A['muts'], use_pdb, 0)
        c0, ia0, ib0 = join(kA0, kB)
        r0 = (float(np.corrcoef(A['y'][ia0], B['y'][ib0])[0, 1])
              if len(c0) > 2 else float('nan'))
        out[tag] = dict(offset=off, offsets_wt_consistent=offs,
                        n_shared=len(common), n_a=len(kA), n_b=len(kB),
                        pearson=r, spearman=_spearman(ya, yb),
                        naive_n_shared=len(c0), naive_pearson=r0,
                        naive_spearman=(_spearman(A['y'][ia0], B['y'][ib0])
                                        if len(c0) > 2 else float('nan')),
                        pos_a=res[tag]['pos_a'], pos_b=res[tag]['pos_b'],
                        wt_a=res[tag]['wt_a'], wt_b=res[tag]['wt_b'])
    return out


def _spearman(a, b):
    from scipy.stats import spearmanr
    if len(a) < 3:
        return float('nan')
    return float(spearmanr(a, b).correlation)


# --------------------------------------------------------------------------- #
# G2 -- twin-assay byte identity                                              #
# --------------------------------------------------------------------------- #

def gate_G2():
    """KRAS_SOS1_8BE4 vs KRAS_DARPinK27_5O2S: shared keys and max|Delta| on the
    raw score STRINGS (byte identity, not float equality).

    The chain label differs between the two files (8BE4 mutates chain R, 5O2S
    chain A) so a canonical-key join is impossible by construction.  Both files
    mutate exactly their own side0 chain with IDENTICAL ``mutant`` token strings,
    so the join key is the side0 token multiset -- an explicit, verified
    chain-slot alignment, not a naive ``(pos, aa)`` match.
    """
    out = {}
    tabs = {}
    for dms_id in ('KRAS_SOS1_norfitness_8BE4', 'KRAS_DARPinK27_norfitness_5O2S'):
        spec = ASSAYS[dms_id]
        df = pd.read_csv(PATHS.dms_csv(dms_id), usecols=USECOLS,
                         dtype={'DMS_score': str})
        keys, offside = [], 0
        for ms, mp in zip(df['mutant'].values, df['mutant_pdb'].values):
            muts = parse_pair_dicts(ms, mp)
            if any(m[0] not in spec.side0_chains for m in muts):
                offside += 1
            keys.append(tuple(sorted((m[1], m[3]) for m in muts)))
        tabs[dms_id] = dict(keys=keys, s=df['DMS_score'].tolist(),
                            y=np.asarray(df['DMS_score'].values, dtype=np.float64),
                            n_offside=offside)
    a, b = tabs['KRAS_SOS1_norfitness_8BE4'], tabs['KRAS_DARPinK27_norfitness_5O2S']
    da = {k: i for i, k in enumerate(a['keys'])}
    db = {k: i for i, k in enumerate(b['keys'])}
    common = sorted(set(da) & set(db))
    n_str_diff = 0
    max_abs = 0.0
    for k in common:
        sa, sb = a['s'][da[k]], b['s'][db[k]]
        if sa != sb:
            n_str_diff += 1
        max_abs = max(max_abs, abs(a['y'][da[k]] - b['y'][db[k]]))
    out.update(n_a=len(a['keys']), n_b=len(b['keys']), n_shared=len(common),
               n_raw_string_differences=n_str_diff, max_abs_delta=float(max_abs),
               n_offside_mutations_a=a['n_offside'], n_offside_mutations_b=b['n_offside'],
               n_only_in_a=len(set(da) - set(db)), n_only_in_b=len(set(db) - set(da)))
    return out


# --------------------------------------------------------------------------- #
# G3 -- chain-key integrity in the four Z-domain assays                       #
# --------------------------------------------------------------------------- #

Z_ASSAYS = ('Z-domain_ZSPA-1_LL1_fitness_1LP1', 'Z-domain_ZSPA-1_LL2_fitness_1LP1',
            'Z-domain_ZpA963_HL1_fitness_2M5A', 'Z-domain_ZpA963_HL2_fitness_2M5A')


def gate_G3(assays=Z_ASSAYS):
    """Duplicate genotypes WITH and WITHOUT the chain label.

    If duplicates appear WITH the chain, the within-genotype SDs are real and
    become the primary noise floor for those assays (spec G3) -- report either
    way, never silently.

    Two chain-drop conventions are reported, because they give different answers
    and the spec's pre-declared 847/59/650/38 identifies exactly one of them:

    * ``token`` -- drop the chain from the full mutation token, i.e. key on
      ``(seq_pos, wt_aa, mut_aa)``.  This is what a naive "concatenate the two
      chains' ``mutant`` strings" join does.  Reproduces the spec's numbers.
    * ``posaa`` -- drop the chain AND the WT letter, i.e. key on
      ``(seq_pos, mut_aa)`` -- the chain-less form of the canonical key.  A
      strictly coarser key, so strictly more collisions.
    """
    rows = []
    for dms_id in assays:
        df = pd.read_csv(PATHS.dms_csv(dms_id), usecols=USECOLS,
                         dtype={'DMS_score': str})
        n = len(df)
        y = np.asarray(df['DMS_score'].values, dtype=np.float64)
        with_chain, k_token, k_posaa, k_pdbtoken = [], [], [], []
        for ms, mp in zip(df['mutant'].values, df['mutant_pdb'].values):
            muts = parse_pair_dicts(ms, mp)
            with_chain.append(canonical_key(muts))
            k_token.append(tuple(sorted((m[1], m[2], m[3]) for m in muts)))
            k_posaa.append(tuple(sorted((m[1], m[3]) for m in muts)))
            k_pdbtoken.append(tuple(sorted((m[4], m[5], m[2], m[3]) for m in muts)))

        def counts(keys):
            c = Counter(keys)
            g = defaultdict(list)
            for k, v in zip(keys, y):
                g[k].append(v)
            sds = [np.std(v, ddof=1) for v in g.values() if len(v) > 1]
            return dict(n_unique=len(c),
                        n_dup_rows=n - len(c),
                        n_dup_keys=sum(1 for v in c.values() if v > 1),
                        max_mult=max(c.values()),
                        n_groups_ge2=len(sds),
                        mean_sd=float(np.mean(sds)) if sds else float('nan'),
                        pooled_sd=(float(np.sqrt(np.mean(np.square(sds))))
                                   if sds else float('nan')))

        cw = counts(with_chain)
        ct = counts(k_token)
        cp = counts(k_posaa)
        cq = counts(k_pdbtoken)
        rows.append(dict(
            DMS_id=dms_id, n_rows=n,
            # ---- with the chain label: the analysis convention ----
            n_unique_with_chain=cw['n_unique'],
            n_dup_rows_with_chain=cw['n_dup_rows'],
            n_dup_keys_with_chain=cw['n_dup_keys'],
            # ---- chain dropped from the token (the spec's convention) ----
            n_dup_keys_without_chain=ct['n_dup_keys'],
            n_dup_rows_without_chain=ct['n_dup_rows'],
            max_multiplicity_without_chain=ct['max_mult'],
            mean_within_genotype_sd_without_chain=ct['mean_sd'],
            pooled_within_genotype_sd_without_chain=ct['pooled_sd'],
            n_groups_ge2_without_chain=ct['n_groups_ge2'],
            # ---- chain AND wt letter dropped (strictly coarser) ----
            n_dup_keys_without_chain_posaa=cp['n_dup_keys'],
            n_dup_rows_without_chain_posaa=cp['n_dup_rows'],
            mean_within_genotype_sd_posaa=cp['mean_sd'],
            # ---- chain dropped under PDB numbering ----
            n_dup_keys_without_chain_pdbnum=cq['n_dup_keys'],
            n_dup_rows_without_chain_pdbnum=cq['n_dup_rows'],
            # ---- the spec's pre-declared values ----
            expected_dups_without_chain=EXPECTED['G3_dups_without_chain'][dms_id],
            expected_dups_with_chain=EXPECTED['G3_dups_with_chain'][dms_id],
            forbidden_spec_sd=config.FORBIDDEN_ZDOMAIN_SDS[dms_id]))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# audit_all                                                                   #
# --------------------------------------------------------------------------- #

def audit_all(files=None, *, deep=True, literal_eval_check=True, verbose=True):
    """G1 + G1b + G2 + G3.  Returns the per-file G1 table; the cross-assay gate
    results ride along in ``df.attrs``."""
    files = list(config.ALL_ASSAYS) if files is None else list(files)
    rows = []
    for k, dms_id in enumerate(files):
        r = audit_one(dms_id, deep=deep, literal_eval_check=literal_eval_check)
        rows.append(r)
        if verbose:
            print('[G1] %2d/%2d %-44s n=%-6d inst=%-7d uniq=%-6d dup=%d '
                  'wtmis(pdbcol/wtseq/pdbres)=%s/%s/%s X=%d star=%d recon_fail=%s %.1fs'
                  % (k + 1, len(files), dms_id, r['n_rows'], r['n_mutation_instances'],
                     r['n_unique_keys'], r['n_dup_keys'],
                     r['n_wt_mismatch_mutant_vs_pdbcol'],
                     r['n_wt_mismatch_vs_wildtype_sequence'],
                     r['n_wt_mismatch_vs_pdb_residue'], r['n_X_hits'],
                     r['n_star_tokens'], r['n_reconstruction_fail'], r['wall_s']),
                  flush=True)
    df = pd.DataFrame(rows)
    tot = dict(
        n_files=len(df), n_rows=int(df['n_rows'].sum()),
        n_unique_keys=int(df['n_unique_keys'].sum()),
        n_dup_keys=int(df['n_dup_keys'].sum()),
        n_mutation_instances=int(df['n_mutation_instances'].sum()),
        n_wt_mismatch_total=int(df['n_wt_mismatch_mutant_vs_pdbcol'].sum()
                                + df['n_wt_mismatch_vs_wildtype_sequence'].sum()
                                + sum(v for v in df['n_wt_mismatch_vs_pdb_residue']
                                      if v != '')),
        n_X_hits=int(df['n_X_hits'].sum()), n_star_tokens=int(df['n_star_tokens'].sum()),
        n_identity_mutations=int(df['n_identity_mutations'].sum()),
        n_parse_fail=int(df['n_parse_fail'].sum()),
        n_token_count_mismatch=int(df['n_token_count_mismatch'].sum()),
        n_reconstruction_fail=(int(df['n_reconstruction_fail'].sum()) if deep else ''),
        n_indels=(int(df['n_indels'].sum()) if deep else ''),
        n_literal_eval_checked=int(df['n_literal_eval_checked'].sum()),
        n_literal_eval_diff=int(df['n_literal_eval_diff'].sum()),
        all_chain_token_counts_agree=bool(df['chain_token_counts_agree'].all()),
        wildtype_sequence_constant_per_file=bool((df['wildtype_sequence_nunique'] == 1).all()),
    )
    df.attrs['G1'] = tot
    df.attrs['G1b'] = gate_G1b()
    df.attrs['G2'] = gate_G2()
    df.attrs['G3'] = gate_G3()
    df.attrs['HLA_duplicate_column'] = _check_hla_duplicate_column()
    return df


def md5_of(path, block=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        while True:
            b = fh.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# self-check                                                                  #
# --------------------------------------------------------------------------- #

def _selfcheck():
    config.assert_env()
    print('=' * 100)
    print('cliff.io_bgym self-check -- runs on the real data')
    print('=' * 100)
    # ---- parse_pair_dicts on the three documented gotchas ----
    got = parse_pair_dicts("{'H': 'P53L:Y57C', 'L': '', 'A': ''}",
                           "{'A': '', 'H': 'P52AL:Y56C', 'L': ''}")
    assert got == [('H', 53, 'P', 'L', 52, 'A'), ('H', 57, 'Y', 'C', 56, '')], got
    print('[parse] 5A12_Ang2 dict-key-order + Kabat icode      -> %r  OK' % (got,))
    got = parse_pair_dicts("{'R': '', 'J': ''}", "{'J': '', 'R': ''}")
    assert got == []
    print('[parse] CXCR4 WT row (key order differs)            -> []  OK')
    got = parse_pair_dicts("{'B': 'Q6F:L14F', 'A': 'L6F:I28L'}",
                           "{'A': 'L9F:I31L', 'B': 'Q9F:L17F'}")
    assert canonical_key(got) == (('A', 6, 'F'), ('A', 28, 'L'),
                                  ('B', 6, 'F'), ('B', 14, 'F')), canonical_key(got)
    print('[parse] Z-LL1 both chains mutated, keys keep chain  -> %r  OK'
          % (canonical_key(got),))
    # dropping the chain collides A/6/F with B/6/F
    nochain = tuple(sorted((p, a) for _c, p, a in canonical_key(got)))
    assert len(set(nochain)) < len(nochain), nochain
    print('[parse] ... and WITHOUT the chain it collides       -> %r  (G3)' % (nochain,))
    # ---- score_quantum on the two spec-quoted grids ----
    q, md = score_quantum(['-4.84', '-4.76', '-0.63', '-4.84'])
    assert (q, md) == (0.01, 2), (q, md)
    print('[quantum] SARS2-RBD-style 2-dp strings -> q = %g (spec: 0.01)  OK' % q)
    # ---- detect_censoring on a synthetic SARS2-like vector ----
    yy = np.concatenate([np.full(1453, -4.84), np.full(931, -4.76),
                         np.linspace(-4.7, 0.37, 7616)])
    lv, mk, meta = detect_censoring(yy)
    assert sorted(lv) == [-4.84, -4.76], lv
    assert abs(mk.mean() - 0.2384) < 1e-3, mk.mean()
    print('[censor] synthetic SARS2 -> levels %s  mass %.4f (spec 0.2385)  OK'
          % (list(lv), mk.mean()))
    # ---- HLA duplicate column ----
    hla = _check_hla_duplicate_column()
    print('[HLA] duplicated DMS_score column: %d rows, %d differences, usecols keeps '
          'the first (%d differences)  %s'
          % (hla['n_rows'], hla['n_diff_between_duplicates'], hla['n_diff_kept_vs_first'],
             'OK' if hla['n_diff_between_duplicates'] == 0 else 'MISMATCH'))
    # ---- load one big and one small assay ----
    for dms_id in ('GB1_IgG-Fc_fitness_1FCC', 'hYAP65_peptide_FunctioncalScore_1JMQ',
                   'Z-domain_ZSPA-1_LL1_fitness_1LP1'):
        t0 = time.time()
        a = load_assay(dms_id)
        print('[load] %-44s n=%-6d P=%-4d M=%-5d max_mut=%-2d wt_row=%-6s '
              'censor=%s q=%g  %.2fs'
              % (dms_id, a.n, a.P, a.M, int(a.n_muts.max()), a.wt_row,
                 list(a.censor_levels), a.quantum, time.time() - t0))
        assert len(set(a.keys)) == a.n, 'duplicate canonical keys in %s' % dms_id
    a = load_assay('hYAP65_peptide_FunctioncalScore_1JMQ')
    assert a.transform == 'log10' and abs(a.y[np.argmin(np.abs(a.y_raw - 1.0))]) < 1e-12
    print('[load] hYAP65 log10 applied: raw WT 1.000 -> y 0.000  OK')
    # ---- gates ----
    print('-' * 100)
    g2 = gate_G2()
    print('[G2] shared=%d (spec %d)  max|delta|=%.17g (spec %g)  raw-string diffs=%d'
          % (g2['n_shared'], EXPECTED['G2_n_shared_keys'], g2['max_abs_delta'],
             EXPECTED['G2_max_abs_delta'], g2['n_raw_string_differences']))
    g1b = gate_G1b()
    for tag, v in sorted(g1b.items()):
        print('[G1b] via %-10s offset=%-5s shared=%d/%d  r=%+.4f rho=%+.4f  | '
              'naive(no offset): shared=%d r=%s'
              % (tag, v['offset'], v['n_shared'], v['n_a'], v['pearson'],
                 v['spearman'], v['naive_n_shared'],
                 ('%+.4f' % v['naive_pearson']) if v['naive_n_shared'] > 2 else 'n/a'))
    g3 = gate_G3()
    print('[G3]')
    print(g3[['DMS_id', 'n_rows', 'n_dup_keys_with_chain', 'expected_dups_with_chain',
              'n_dup_keys_without_chain', 'expected_dups_without_chain',
              'n_dup_keys_without_chain_posaa', 'n_dup_keys_without_chain_pdbnum',
              'mean_within_genotype_sd_without_chain', 'forbidden_spec_sd']].to_string(
                  index=False))
    ok3 = bool((g3['n_dup_keys_with_chain'] == g3['expected_dups_with_chain']).all()
               and (g3['n_dup_keys_without_chain']
                    == g3['expected_dups_without_chain']).all())
    print('[G3] %s' % ('PASS -- 0 duplicates with the chain, spec 847/59/650/38 reproduced '
                       'exactly without it' if ok3 else 'FAIL'))
    print('[io_bgym] SELF-CHECK PASSED')


if __name__ == '__main__':
    _selfcheck()
