"""BGYM-CLIFF v1 -- verdict plumbing.  Spec Sec.1.2-1.6, table T14.

DO NOT CHANGE: no numeric decision boundary may be written here -- every one is read
from ``config.THRESH`` (a literal in this file is a bug).  And do not turn a missing
or unevaluable input into a pass: absence is INCONCLUSIVE with a named reason, an
UNDERPOWERED (G8) assay is INCONCLUSIVE whatever its numbers, and a failed halting
gate makes the whole study INCONCLUSIVE.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from cliff import config
from cliff.config import PATHS, THRESH

# --------------------------------------------------------------------------- #
# the three-way outcome                                                       #
# --------------------------------------------------------------------------- #

SUPPORTED = 'SUPPORTED'
REFUTED = 'REFUTED'
INCONCLUSIVE = 'INCONCLUSIVE'
OUTCOMES = (SUPPORTED, REFUTED, INCONCLUSIVE)

#: Family-level wording.  A family is not an assay: it either supports the claim,
#: refutes it, or cannot say.  Kept as the same three tokens so the k-of-7 counts
#: are countable without a second vocabulary.
NOT_TESTABLE = 'NOT_TESTABLE'

#: Aggregate wording for a claim over the 7 families (spec Sec.1.2-1.4 say
#: "C1 TRUE iff ... / C1 REFUTED iff ...").
TRUE = 'TRUE'


# --------------------------------------------------------------------------- #
# artifact tables                                                             #
# --------------------------------------------------------------------------- #

#: spec Sec.6 deliverables.  verdict.py reads these and never writes anything but
#: the ``verdict_*`` / ``failing_criterion`` / ``classification`` columns back.
TABLES = {
    'T01': 'T01_assay_manifest.csv',
    'T02': 'T02_gates.csv',
    'T03': 'T03_noise_registry.csv',
    'T04': 'T04_smoothness_C1.csv',
    'T05': 'T05_variogram.csv',
    'T06': 'T06_cliff_tail_C2.csv',
    'T07': 'T07_localisation_C3.csv',
    'T08': 'T08_epsilon_replication.csv',
    'T09': 'T09_structure_sites.csv',
    'T10': 'T10_structure_pairs.csv',
    'T11': 'T11_partner_specificity.csv',
    'T12': 'T12_cliff_aware_eval.csv',
    'T13': 'T13_sensitivity.csv',
    'T15': 'T15_cluster_channel.csv',
}
T14_NAME = 'T14_verdict_by_family.csv'
T14A_NAME = 'T14a_verdict_by_assay.csv'

#: spec Sec.6 T14, verbatim and in order.  ``C1_pos/C1_neg`` really is one column
#: whose name contains a slash -- exactly like T02's ``PASS/FAIL``.
T14_COLUMNS = [
    'family_id', 'member_assays', 'n_eligible',
    'C1_pos/C1_neg', 'C2_pos/C2_neg', 'C3L_pos/C3L_neg',
    'C3N_result', 'C4S_result', 'C4I_result', 'C5_result',
    'family_verdict_C1', 'family_verdict_C2', 'family_verdict_C3', 'all_three',
    'meta_effect', 'meta_ci_lo', 'meta_ci_hi', 'notes',
]

#: The verdict columns verdict.py owns inside the statistics tables (spec Sec.3:
#: "verdict.py  # applies THRESH; emits T1..T12").  Nothing else is ever written.
WRITE_BACK = {
    'T04': ('verdict_C1', 'failing_criterion'),
    'T06': ('verdict_C2', 'failing_criterion'),
    'T07': ('verdict_C3L', 'verdict_C3A', 'failing_criterion'),
    'T08': ('verdict_C3N', 'verdict_stamp'),
    'T11': ('classification',),
    'T12': ('verdict_blindspot', 'verdict_practical_emptiness'),
}


def artifact_path(name):
    return os.path.join(PATHS.artifacts, TABLES.get(name, name))


def load_table(name, verbose=False):
    """``(DataFrame or None, status)``.  ``status`` is ``'ok'`` or the reason the
    table is unusable -- ``table_missing`` / ``table_empty``.  A missing table is
    NEVER a silent pass; every verdict that needed it reports ``table_missing``."""
    import pandas as pd
    p = artifact_path(name)
    if not os.path.exists(p):
        return None, 'table_missing'
    try:
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
    except Exception as exc:                                  # pragma: no cover
        return None, 'table_unreadable:%s' % type(exc).__name__
    if len(df) == 0:
        return df, 'table_empty'
    if verbose:
        print('[verdict] read %-30s %5d x %2d' % (TABLES.get(name, name),
                                                  len(df), len(df.columns)))
    return df, 'ok'


def load_all_tables(verbose=False):
    """Every Sec.6 table, read as strings (a verdict never depends on pandas'
    dtype inference).  Returns ``{name: (df|None, status)}``."""
    return {k: load_table(k, verbose=verbose) for k in sorted(TABLES)}


def table_status(tables=None):
    """One row per Sec.6 table: does it exist, how big, which verdicts it feeds."""
    import pandas as pd
    tables = load_all_tables() if tables is None else tables
    feeds = {'T01': 'eligibility, UNDERPOWERED stamp', 'T02': 'all gates',
             'T03': 'C3-N stamp', 'T04': 'C1', 'T05': 'C1 (V(1)>V(2))',
             'T06': 'C2', 'T07': 'C3-L, C3-A', 'T08': 'C3-N', 'T09': 'C4-S',
             'T10': '(C4-P detail)', 'T11': 'C4-I', 'T12': 'C5',
             'T13': '(sensitivity)', 'T15': 'cluster channel -> C2 count'}
    rows = []
    for k in sorted(TABLES):
        df, st = tables[k]
        rows.append(dict(table=k, filename=TABLES[k], status=st,
                         n_rows=(0 if df is None else len(df)),
                         n_cols=(0 if df is None else len(df.columns)),
                         feeds=feeds.get(k, '')))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# value readers -- everything comes back as (value, why-not)                   #
# --------------------------------------------------------------------------- #

_TRUE_TOKENS = ('true', 't', 'yes', 'y', '1', 'pass', 'passed', 'ok')
_FALSE_TOKENS = ('false', 'f', 'no', 'n', '0', 'fail', 'failed')
_EMPTY_TOKENS = ('', 'nan', 'none', 'null', 'na', 'n/a', 'pending', '-', '--')


def _s(row, col):
    """Raw string of ``row[col]``, or None when the column or value is absent."""
    if row is None or col not in row:
        return None
    v = row[col]
    if v is None:
        return None
    s = str(v).strip()
    return None if s.lower() in _EMPTY_TOKENS else s


def num(row, col):
    """Float value of ``row[col]``, else None."""
    s = _s(row, col)
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def flag(row, col):
    """Tri-state boolean of ``row[col]``: True / False / None (absent)."""
    s = _s(row, col)
    if s is None:
        return None
    t = s.lower()
    if t in _TRUE_TOKENS:
        return True
    if t in _FALSE_TOKENS:
        return False
    return None


# --------------------------------------------------------------------------- #
# clauses                                                                     #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Clause:
    """One clause of a spec decision rule.

    ``ok`` is tri-state on purpose: ``None`` means the clause could not be
    evaluated (a missing table, row, column or value), which must never be read
    as "passed".
    """
    name: str
    kind: str            # 'support' | 'refute'
    ok: object           # True | False | None
    detail: str = ''


def _sup(name, ok, detail=''):
    return Clause(name, 'support', ok, detail)


def _ref(name, ok, detail=''):
    return Clause(name, 'refute', ok, detail)


def _le(v, t):
    return None if v is None else bool(v <= t)


def _ge(v, t):
    return None if v is None else bool(v >= t)


def _lt(v, t):
    return None if v is None else bool(v < t)


def _gt(v, t):
    return None if v is None else bool(v > t)


def _and3(vals):
    """Three-valued AND (Kleene).  ONE definitely-false term settles a
    conjunction even when the others are missing -- without this, a single absent
    column would make every clause "unevaluable" and hide a real failure."""
    vals = list(vals)
    if any(v is False for v in vals):
        return False
    if vals and all(v is True for v in vals):
        return True
    return None


def _or3(vals):
    """Three-valued OR.  One definitely-true term settles a disjunction."""
    vals = list(vals)
    if any(v is True for v in vals):
        return True
    if vals and all(v is False for v in vals):
        return False
    return None


def _nand3(vals):
    """Three-valued NOT-AND: the negation of a conjunction, used where the spec
    states a clause as "REFUTED if A and B" but the support side needs its
    complement."""
    v = _and3(vals)
    return None if v is None else (not v)


def _fmt(v):
    if v is None:
        return 'NA'
    return ('%g' % v) if isinstance(v, float) else str(v)


@dataclass
class Decision:
    outcome: str
    failing_criterion: str
    clauses: tuple = ()
    detail: dict = field(default_factory=dict)

    def as_row(self, prefix):
        d = {prefix: self.outcome, prefix + '_failing_criterion': self.failing_criterion,
             prefix + '_clauses': ' | '.join(
                 '%s=%s(%s)' % (c.name, {True: 'T', False: 'F', None: '?'}[c.ok],
                                c.detail) for c in self.clauses)}
        for k, v in self.detail.items():
            d['%s_%s' % (prefix, k)] = v
        return d


def decide(clauses, detail=None):
    """Apply the spec's own logic: REFUTED is an OR over refutation clauses,
    SUPPORTED is an AND over support clauses, and anything else -- a support
    clause that definitively failed, or any clause that could not be evaluated --
    is INCONCLUSIVE with the deciding clause named.

    A refutation clause firing *while* every support clause also passes is a real
    possibility in these rules (they are not complements); it is reported as
    REFUTED with a ``CONFLICT:`` prefix so a reader sees it rather than a
    silently-chosen precedence.
    """
    clauses = tuple(clauses)
    fired = [c for c in clauses if c.kind == 'refute' and c.ok is True]
    sup = [c for c in clauses if c.kind == 'support']
    sup_fail = [c for c in sup if c.ok is False]
    unk = [c for c in clauses if c.ok is None]
    det = dict(detail or {})
    if fired:
        why = 'refuted_by:' + '+'.join(c.name for c in fired)
        if sup and not sup_fail and not unk:
            why = 'CONFLICT:' + why + ';every_support_clause_also_passed'
        return Decision(REFUTED, why, clauses, det)
    if sup_fail:
        return Decision(INCONCLUSIVE,
                        'support_failed:' + '+'.join(c.name for c in sup_fail),
                        clauses, det)
    if unk:
        return Decision(INCONCLUSIVE,
                        'unevaluable:' + '+'.join(c.name for c in unk), clauses, det)
    if not sup:
        return Decision(INCONCLUSIVE, 'no_support_clause_defined', clauses, det)
    return Decision(SUPPORTED, '', clauses, det)


# --------------------------------------------------------------------------- #
# gates -- G7's rule switch, G8's UNDERPOWERED stamp, G9's k, the halts        #
# --------------------------------------------------------------------------- #

#: What ``calibrate.py`` must write into T02's G7 ``observed`` cell.  G7 does not
#: pass or fail -- it *sets* the C2 rule (spec Sec.1.1) -- so its verdict is read
#: from ``observed``, never from ``PASS/FAIL``.
G7_FLAG_CONVENTION = (
    "T02 gate_id 'G7': observed = 'inflated' (N2c lifts TR / T(tau) -> C2 alone is "
    "inadmissible, C2 AND C3-L becomes mandatory) or 'not_inflated' (N2c leaves them "
    "at 1.00 -> C2 alone is admissible); a numeric observed value is read as the N2c "
    "inflation factor and counts as inflated iff > 1 + THRESH['G4_T_tol']. A SECOND "
    "G7 row whose gate_name or statistic mentions 'localisation' carries the same "
    "encoding for the localisation axis; 'inflated' there is the STOP branch. An "
    "empty cell means undetermined, and C2 is then decided both ways (see g7_flags)."
)
_G7_INFLATED = ('inflated', 'tail_inflated', 'inflatable', 'yes', 'true', '1')
_G7_NOT = ('not_inflated', 'no_inflation', 'at_null', 'no', 'false', '0', 'null')


def _g7_read(cell):
    """'inflated'/'not_inflated'/number -> True/False; anything else -> None."""
    if cell is None:
        return None
    t = str(cell).strip().lower().replace(' ', '_').replace('-', '_')
    if t in _G7_INFLATED:
        return True
    if t in _G7_NOT:
        return False
    try:
        return bool(float(t) > 1.0 + THRESH['G4_T_tol'])
    except ValueError:
        return None


def g7_flags(t02, override=None):
    """Read the G7 rule switch out of the gates table (spec Sec.1.3 clause 4).

    Returns ``{'tail_inflatable', 'localisation_inflated', 'source', 'per_assay'}``
    with tri-state values.  Nothing about the C2 rule is hardcoded here: an empty
    G7 cell yields ``None`` and :func:`verdict_C2` then decides the assay both
    ways and reports INCONCLUSIVE only if the two answers differ.
    """
    out = dict(tail_inflatable=None, localisation_inflated=None,
               source='T02 absent', per_assay={})
    if override is not None and 'tail_inflatable' in override:
        out.update(override)
        out['source'] = 'caller override'
        return out
    if t02 is None or 'gate_id' not in getattr(t02, 'columns', []):
        return out
    rows = t02[t02['gate_id'].astype(str).str.strip() == 'G7']
    if len(rows) == 0:
        out['source'] = 'T02 has no G7 row'
        return out
    seen = []
    for _, r in rows.iterrows():
        txt = (str(r.get('gate_name', '')) + ' ' + str(r.get('statistic', ''))).lower()
        val = _g7_read(_s(r, 'observed'))
        which = 'localisation_inflated' if ('localis' in txt or 'localiz' in txt) \
            else 'tail_inflatable'
        assay = str(r.get('assay', '')).strip()
        if assay in config.ASSAYS:
            out['per_assay'].setdefault(assay, {})[which] = val
        elif out[which] is None:
            out[which] = val
        seen.append('%s=%s' % (which, val))
    out['source'] = 'T02 G7 rows (%s)' % ', '.join(seen)
    return out


def halting_gate_failures(t02):
    """Spec Sec.7: gates whose failure stops the study.  A FAIL on any row with
    ``halts_study == YES`` makes every verdict INCONCLUSIVE."""
    if t02 is None or 'gate_id' not in getattr(t02, 'columns', []):
        return []
    out = []
    for _, r in t02.iterrows():
        if str(r.get('halts_study', '')).strip().upper() not in ('YES', 'TRUE', '1'):
            continue
        if str(r.get('PASS/FAIL', '')).strip().upper() == 'FAIL':
            out.append((str(r['gate_id']), str(r.get('assay', '')),
                        str(r.get('statistic', ''))[:60]))
    return out


def underpowered_assays(t01, t02):
    """The G8 UNDERPOWERED set (spec Sec.1.1 G8 / Sec.7 item 11).

    Read from T01's ``underpowered_G8`` column first (that is where the stamp
    lives, spec Sec.6 T01) and then from any per-assay T02 G8 row whose observed
    power is below ``THRESH['G8_power_min']``.  Plus the STOP branch of G7: if
    localisation is *also* inflated under N2c there is no discriminating axis at
    all, which the caller handles as a halt, not as an underpower stamp.
    """
    out = {}
    if t01 is not None and 'underpowered_G8' in t01.columns:
        for _, r in t01.iterrows():
            if flag(r, 'underpowered_G8') is True:
                out[str(r['DMS_id'])] = 'T01.underpowered_G8'
    if t02 is not None and 'gate_id' in getattr(t02, 'columns', []):
        for _, r in t02[t02['gate_id'].astype(str).str.strip() == 'G8'].iterrows():
            a = str(r.get('assay', '')).strip()
            p = num(r, 'observed')
            if a in config.ASSAYS and p is not None and p < THRESH['G8_power_min']:
                out.setdefault(a, 'T02.G8 power %.3g < %.3g'
                               % (p, THRESH['G8_power_min']))
    return out


def g9_rule_calibration(t02):
    """G9: the measured family-level FPR of the k-of-7 rule, and the tightened k
    if one was recorded.  Spec Sec.1.3: C2's k is "subject to G9 tightening", and
    Sec.1.1 G9 says the change must be recorded *before* any observed value is
    inspected -- so it is read from the gates table, never chosen here.

    ``artifacts/G9_k_tightening.json`` (written by calibrate.py) may carry
    ``{"C1": k, "C2": k, "C3": k}``.
    """
    out = dict(fpr=None, k=dict(), source='T02 absent')
    if t02 is not None and 'gate_id' in getattr(t02, 'columns', []):
        rows = t02[t02['gate_id'].astype(str).str.strip() == 'G9']
        if len(rows):
            out['fpr'] = num(rows.iloc[0], 'observed')
            out['source'] = 'T02 G9 observed'
        else:
            out['source'] = 'T02 has no G9 row'
    p = os.path.join(PATHS.artifacts, 'G9_k_tightening.json')
    if os.path.exists(p):
        try:
            with open(p) as fh:
                out['k'] = {str(k): int(v) for k, v in json.load(fh).items()}
            out['source'] += ' + G9_k_tightening.json'
        except Exception:                                     # pragma: no cover
            pass
    return out


@dataclass
class Gates:
    halts: tuple = ()
    underpowered: dict = field(default_factory=dict)
    g7: dict = field(default_factory=dict)
    g9: dict = field(default_factory=dict)
    gup_obtained: object = None      # G-UP: upstream per-variant SEs
    gopt_ok: object = None           # G-OPT: hypercube structures recovered
    g11_dual: object = None          # G11: KRAS cliffs localise to BOTH interfaces

    @property
    def study_halted(self):
        return len(self.halts) > 0

    @property
    def halt_reason(self):
        return 'GATE_FAILED_STUDY_HALTED:' + '+'.join(
            '%s(%s)' % (g, a) if a else g for g, a, _ in self.halts)


def read_gates(t02, t01=None, g7_override=None):
    g7 = g7_flags(t02, override=g7_override)
    halts = list(halting_gate_failures(t02))
    # spec Sec.1.1 G7: "If localisation is ALSO inflated under N2c -> STOP, the
    # localisation axis has no discriminating power either."
    if g7.get('localisation_inflated') is True:
        halts.append(('G7', 'PRIMARY+ARM',
                      'localisation ALSO inflated under N2c -> no discriminating axis'))
    gup = gopt = g11 = None
    if t02 is not None and 'gate_id' in getattr(t02, 'columns', []):
        gid = t02['gate_id'].astype(str).str.strip()
        for key, name in (('G-UP', 'gup'), ('G-OPT', 'gopt'), ('G11', 'g11')):
            rows = t02[gid == key]
            if len(rows) == 0:
                continue
            obs = _s(rows.iloc[0], 'observed')
            val = None if obs is None else flag(rows.iloc[0], 'observed')
            if val is None and obs is not None:
                val = str(obs)
            if name == 'gup':
                gup = val
            elif name == 'gopt':
                gopt = val
            else:
                g11 = val
    return Gates(halts=tuple(halts), underpowered=underpowered_assays(t01, t02),
                 g7=g7, g9=g9_rule_calibration(t02), gup_obtained=gup,
                 gopt_ok=gopt, g11_dual=(True if g11 is True else
                                         (False if g11 is False else None)))


# --------------------------------------------------------------------------- #
# row helpers                                                                 #
# --------------------------------------------------------------------------- #

def _rows_for(df, status, key, value):
    """Rows of ``df`` whose ``key`` equals ``value``, plus a status token."""
    if df is None:
        return None, status
    if key not in df.columns:
        return None, 'column_missing:%s' % key
    sub = df[df[key].astype(str).str.strip() == str(value)]
    if len(sub) == 0:
        return None, 'row_missing'
    return sub, 'ok'


def _one(df, status, key, value):
    sub, st = _rows_for(df, status, key, value)
    if sub is None:
        return None, st
    return sub.iloc[0], 'ok'


def _first_num(sub, col):
    """First non-empty numeric value of ``col`` over a group of rows (T06/T07/T09
    repeat their per-assay columns on every row)."""
    if sub is None or col not in sub.columns:
        return None
    for _, r in sub.iterrows():
        v = num(r, col)
        if v is not None:
            return v
    return None


def _first_flag(sub, col):
    if sub is None or col not in sub.columns:
        return None
    for _, r in sub.iterrows():
        v = flag(r, col)
        if v is not None:
            return v
    return None


def _first_str(sub, col):
    if sub is None or col not in sub.columns:
        return None
    for _, r in sub.iterrows():
        v = _s(r, col)
        if v is not None:
            return v
    return None


def _missing(prefix, status):
    return Decision(INCONCLUSIVE, '%s:%s' % (status, prefix), (), {})


# --------------------------------------------------------------------------- #
# C1 -- the landscape is smooth in mutation degree (spec Sec.1.2)             #
# --------------------------------------------------------------------------- #

def verdict_C1(dms_id, t04, t04_status, t05=None, t05_status='table_missing'):
    """Spec Sec.1.2.

    SUPPORTED iff SI <= 0.50 and V(1)/V(inf) <= 0.35 and V(h) non-decreasing over
    h=1..4 and gamma(1) >= 0.60 with 95% CI lower bound > 0.45.
    REFUTED iff SI >= 0.80 or V(1)/V(inf) >= 0.70 or V(1) > V(2) or gamma(1) <= 0.20
    with CI upper < 0.45 or pos_rs >= 0.70.
    """
    row, st = _one(t04, t04_status, 'DMS_id', dms_id)
    if row is None:
        return _missing('T04', st)
    SI = num(row, 'SI')
    v1 = num(row, 'V1_over_Vinf')
    mono = flag(row, 'V_monotone_h1_h4')
    g1 = num(row, 'gamma1')
    g1lo = num(row, 'gamma1_lo95')
    g1hi = num(row, 'gamma1_hi95')
    prs = num(row, 'pos_rs')

    # V(1) > V(2): available directly from T05 (one row per h); when T05 is absent
    # a TRUE monotone flag still settles it, and a FALSE one does not (the
    # violation could be at h=3), which is reported rather than assumed either way.
    v1gt2, v12_src = None, 'V_monotone_h1_h4'
    if mono is True:
        v1gt2 = False
    if t05 is not None and 'h' in getattr(t05, 'columns', []):
        sub, st5 = _rows_for(t05, t05_status, 'DMS_id', dms_id)
        if sub is not None:
            vh = {}
            for _, r in sub.iterrows():
                hs = _s(r, 'h')
                vv = num(r, 'V_h')
                if hs is not None and vv is not None:
                    vh[hs] = vv
            if '1' in vh and '2' in vh:
                v1gt2, v12_src = bool(vh['1'] > vh['2']), 'T05 V_h'
            _ = st5

    cl = [
        _sup('C1.SI<=%g' % THRESH['C1_SI_sup'], _le(SI, THRESH['C1_SI_sup']),
             'SI=%s' % _fmt(SI)),
        _sup('C1.V1/Vinf<=%g' % THRESH['C1_V1_over_Vinf_sup'],
             _le(v1, THRESH['C1_V1_over_Vinf_sup']), 'V1/Vinf=%s' % _fmt(v1)),
        _sup('C1.V_monotone_h1_h%d' % THRESH['C1_h_monotone_upto'], mono,
             'monotone=%s' % _fmt(mono)),
        _sup('C1.gamma1>=%g&CIlo>%g' % (THRESH['C1_gamma1_sup'],
                                        THRESH['C1_gamma1_ci_lo_sup']),
             _and3([_ge(g1, THRESH['C1_gamma1_sup']),
                    _gt(g1lo, THRESH['C1_gamma1_ci_lo_sup'])]),
             'gamma1=%s lo95=%s' % (_fmt(g1), _fmt(g1lo))),
        _ref('C1.SI>=%g' % THRESH['C1_SI_ref'], _ge(SI, THRESH['C1_SI_ref']),
             'SI=%s' % _fmt(SI)),
        _ref('C1.V1/Vinf>=%g' % THRESH['C1_V1_over_Vinf_ref'],
             _ge(v1, THRESH['C1_V1_over_Vinf_ref']), 'V1/Vinf=%s' % _fmt(v1)),
        _ref('C1.V1>V2', v1gt2, 'from %s' % v12_src),
        _ref('C1.gamma1<=%g&CIhi<%g' % (THRESH['C1_gamma1_ref'],
                                        THRESH['C1_gamma1_ci_hi_ref']),
             _and3([_le(g1, THRESH['C1_gamma1_ref']),
                    _lt(g1hi, THRESH['C1_gamma1_ci_hi_ref'])]),
             'gamma1=%s hi95=%s' % (_fmt(g1), _fmt(g1hi))),
        _ref('C1.pos_rs>=%g' % THRESH['C1_pos_rs_ref'],
             _ge(prs, THRESH['C1_pos_rs_ref']), 'pos_rs=%s' % _fmt(prs)),
    ]
    return decide(cl, dict(SI=SI, V1_over_Vinf=v1, gamma1=g1, pos_rs=prs))


# --------------------------------------------------------------------------- #
# C2 -- a minority of sequence-near pairs jump (spec Sec.1.3)                 #
# --------------------------------------------------------------------------- #

#: spec Sec.1.3 / 1.4: the consecutive-tau window the C2 verdict is read off.
THRESH_TAU_WINDOW = config.TAU_WINDOW


def _tau_window(taus):
    lo, hi = THRESH_TAU_WINDOW
    return [t for t in taus if lo <= t <= hi]


def _longest_run(seq):
    """Longest run of consecutive True in a list, over the *surviving* grid."""
    best = cur = 0
    for v in seq:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def _c2_tau_clause(sub, unit):
    """Clause 2 for one unit system: T(tau) >= 2.0 with q_BH < 0.05 for >= 4
    consecutive tau in [3,8].  A tau dropped by the grid guard is REMOVED from the
    sequence (spec Sec.1.0: "any tau whose absolute cut < 3q_a is dropped from the
    sweep"), not counted as a failure."""
    if sub is None or 'unit' not in sub.columns or 'tau' not in sub.columns:
        return None, 'no tau rows', None
    rows = sub[sub['unit'].astype(str).str.strip().str.lower() == unit.lower()]
    if len(rows) == 0:
        return None, 'unit %s absent' % unit, None
    tau_num = []
    for _, r in rows.iterrows():
        try:
            tau_num.append(float(str(r['tau']).strip()))
        except (ValueError, TypeError):
            tau_num.append(float('nan'))
    keep, dropped = [], []
    for t in _tau_window(config.TAUS):
        rr = rows[[abs(v - t) < 1e-9 for v in tau_num]]
        if len(rr) == 0:
            # An ABSENT tau row is unevaluable, NOT "removed from the sweep":
            # only grid_guard_pass == False removes a tau.  Treating a missing row
            # as removed would let 3,4,6,8 count as four CONSECUTIVE tau.
            keep.append((t, None))
            continue
        r = rr.iloc[0]
        if flag(r, 'grid_guard_pass') is False:
            dropped.append(t)
            continue
        T = num(r, 'T_N2')
        q = num(r, 'q_BH')
        if T is None or q is None:
            keep.append((t, None))
        else:
            keep.append((t, bool(T >= THRESH['C2_T_sup'] and q < THRESH['C2_q_BH_sup'])))
    if not keep:
        return None, 'grid_guard_removed_every_tau_in_%s' % (THRESH_TAU_WINDOW,), None
    if any(v is None for _, v in keep):
        return None, 'value_missing:T_N2/q_BH at tau %s' % (
            [t for t, v in keep if v is None],), None
    run = _longest_run([v for _, v in keep])
    txt = 'run=%d/%d over tau %s%s' % (
        run, THRESH['C2_n_consecutive_tau'], [t for t, _ in keep],
        (' guard-dropped %s' % (dropped,)) if dropped else '')
    return bool(run >= THRESH['C2_n_consecutive_tau']), txt, run


def verdict_C2(dms_id, t06, t06_status, n_Pa=None, c3l_ok=None, g7=None,
               scale='latent'):
    """Spec Sec.1.3, on the requested scale (``latent`` is primary, ``raw`` the
    sensitivity column; Sec.1.4 C3-A.4: a verdict holding on raw but not on latent
    is discarded, never reported as positive).

    Clause 4 -- "if G7 shows the tail is inflatable by heteroscedasticity, the
    assay also passes >= 1 localisation route (C3-L)" -- is driven entirely by the
    gates table.  When G7 is undetermined the assay is decided BOTH ways and the
    answer is reported only if the conjunct does not change it.
    """
    sub, st = _rows_for(t06, t06_status, 'DMS_id', dms_id)
    if sub is None:
        return _missing('T06', st)
    if 'scale' in sub.columns:
        s2 = sub[sub['scale'].astype(str).str.strip().str.lower() == scale]
        if len(s2) == 0:
            return _missing('T06', 'scale_missing:%s' % scale)
        sub = s2

    n_pa = _first_num(sub, 'n_Pa')
    if n_pa is None:
        n_pa = n_Pa
    TR = _first_num(sub, 'TR')
    tr95 = _first_num(sub, 'TR_N1_p95')
    tr995 = _first_num(sub, 'TR_N1_p995')
    pi = _first_num(sub, 'pi_hat')
    pilo = _first_num(sub, 'pi_lo95')
    pihi = _first_num(sub, 'pi_hi95')
    rho = _first_num(sub, 'rho_hat')
    dbic = _first_num(sub, 'dBIC')

    # --- TR regime (spec Sec.1.3: no tail-ratio verdict below 2,000) ---
    if n_pa is None:
        regime = None
    elif n_pa >= THRESH['C2_TR1_min_Pa']:
        regime = 'TR1'
    elif n_pa >= THRESH['C2_TR2_min_Pa']:
        regime = 'TR2'
    else:
        regime = 'none'

    cl, regime_note = [], ''
    if regime == 'none':
        # spec Sec.1.3: "no tail-ratio verdict below 2,000 (mixture + localisation
        # routes only)".  The TR clauses are not added at all -- a clause that is
        # trivially true would read as evidence.
        regime_note = ('|P_a|=%s < %d: NO tail-ratio verdict; C2 rests on the '
                       'mixture and localisation routes only'
                       % (_fmt(n_pa), THRESH['C2_TR2_min_Pa']))
    elif regime is None:
        cl.append(_sup('C2.1_TR', None,
                       'n_Pa unknown (T01/T06 carry no |P_a|): the TR regime '
                       'cannot be chosen'))
    else:
        cl.append(_sup('C2.1_TR>N1_p%g' % THRESH['C2_TR_sup_pctile'],
                       None if (TR is None or tr995 is None) else bool(TR > tr995),
                       '%s TR=%s N1p995=%s' % (regime, _fmt(TR), _fmt(tr995))))
        cl.append(_ref('C2.TR<N1_p%g' % THRESH['C2_TR_ref_pctile'],
                       None if (TR is None or tr95 is None) else bool(TR < tr95),
                       'TR=%s N1p95=%s' % (_fmt(TR), _fmt(tr95))))

    # --- clause 2: the tau sweep, in BOTH unit systems ---
    runs = {}
    for unit in ('sigma', 'MAD'):
        ok, txt, run = _c2_tau_clause(sub, unit)
        runs[unit] = run
        cl.append(_sup('C2.2_tau_run_%s' % unit, ok, txt))
    # the table's own count, when stats_c2.py wrote one, is cross-checked
    declared = _first_num(sub, 'n_consecutive_tau_passing')
    mismatch = ''
    if declared is not None and runs.get('sigma') is not None \
            and int(declared) != int(runs['sigma']):
        mismatch = ('T06.n_consecutive_tau_passing=%g but recomputed %d'
                    % (declared, runs['sigma']))

    # --- clause 3: the mixture ---
    mix = _and3([_le(dbic, THRESH['C2_dBIC_sup']),
                 _ge(pi, THRESH['C2_pi_lo']), _le(pi, THRESH['C2_pi_hi']),
                 _gt(pilo, THRESH['C2_pi_ci_lo_sup']),
                 _ge(rho, THRESH['C2_rho_sup'])])
    cl.append(_sup('C2.3_mixture', mix,
                   'dBIC=%s pi=%s [%s,%s] rho=%s'
                   % (_fmt(dbic), _fmt(pi), _fmt(pilo), _fmt(pihi), _fmt(rho))))

    # --- refutations: max_tau T(tau) and the mixture ---
    tmax, tmax_hi, tmax_unit = None, None, ''
    if 'tau' in sub.columns:
        for _, r in sub.iterrows():
            try:
                tv = float(str(r['tau']).strip())
            except (ValueError, KeyError):
                continue
            if not (THRESH_TAU_WINDOW[0] <= tv <= THRESH_TAU_WINDOW[1]):
                continue
            T = num(r, 'T_N2')
            if T is None:
                continue
            if tmax is None or T > tmax:
                tmax, tmax_hi = T, num(r, 'T_N2_hi95')
                tmax_unit = '%s@tau%g' % (_s(r, 'unit') or '?', tv)
    cl.append(_ref('C2.maxT<%g&CIhi<%g' % (THRESH['C2_T_ref'], THRESH['C2_T_ref_ci_hi']),
                   _and3([_lt(tmax, THRESH['C2_T_ref']),
                          _lt(tmax_hi, THRESH['C2_T_ref_ci_hi'])]),
                   'maxT=%s (%s) hi95=%s' % (_fmt(tmax), tmax_unit, _fmt(tmax_hi))))
    cl.append(_ref('C2.dBIC>%g' % THRESH['C2_dBIC_sup'],
                   _gt(dbic, THRESH['C2_dBIC_sup']), 'dBIC=%s' % _fmt(dbic)))
    cl.append(_ref('C2.pi_CIhi<%g' % THRESH['C2_pi_ci_hi_ref'],
                   _lt(pihi, THRESH['C2_pi_ci_hi_ref']), 'pi_hi95=%s' % _fmt(pihi)))

    det = dict(n_Pa=n_pa, TR=TR, TR_regime=regime, pi_hat=pi, rho_hat=rho, dBIC=dbic,
               tau_run_sigma=runs.get('sigma'), tau_run_MAD=runs.get('MAD'),
               max_T=tmax, note='; '.join(x for x in (regime_note, mismatch) if x))

    # --- clause 4: the G7-conditional localisation conjunct ---
    g7 = g7 or {}
    infl = g7.get('per_assay', {}).get(dms_id, {}).get('tail_inflatable',
                                                       g7.get('tail_inflatable'))
    conj = _sup('C2.4_C3L_conjunct(G7)', c3l_ok,
                'G7 tail_inflatable=%s -> C3-L mandatory; C3-L=%s'
                % (infl, _fmt(c3l_ok)))
    if infl is True:
        det['G7'] = 'inflated: C2 alone inadmissible, C2 AND C3-L mandatory'
        return decide(cl + [conj], det)
    if infl is False:
        det['G7'] = 'not inflated: C2 alone admissible'
        return decide(cl, det)
    with_c = decide(cl + [conj], det)
    without_c = decide(cl, det)
    det['G7'] = 'undetermined; decided both ways'
    if with_c.outcome == without_c.outcome:
        return Decision(with_c.outcome,
                        without_c.failing_criterion or with_c.failing_criterion,
                        tuple(cl) + (conj,), det)
    return Decision(INCONCLUSIVE,
                    'G7_undetermined_and_the_C3L_conjunct_decides(%s_vs_%s)'
                    % (without_c.outcome, with_c.outcome), tuple(cl) + (conj,), det)


# --------------------------------------------------------------------------- #
# C3-N / C3-L / C3-A (spec Sec.1.4)                                           #
# --------------------------------------------------------------------------- #

def upstream_se_obtained(t03, gates=None):
    """Did G-UP deliver upstream per-variant SEs?  Spec Sec.1.4 / Sec.7 item 12:
    every C3-N verdict is stamped ``conditional`` unless it did.

    Two independent sources, and BOTH must say yes: T02's G-UP row and T03's own
    ``upstream_SE_obtained`` column.  Absence is ``conditional``, never
    ``calibrated``.
    """
    gup = None if gates is None else gates.gup_obtained
    t03_ok = None
    if t03 is not None and 'upstream_SE_obtained' in getattr(t03, 'columns', []):
        vals = [flag(r, 'upstream_SE_obtained') for _, r in t03.iterrows()]
        vals = [v for v in vals if v is not None]
        t03_ok = (all(vals) if vals else None)
    src = 'T02.G-UP=%s, T03.upstream_SE_obtained=%s' % (gup, t03_ok)
    if gup is True and t03_ok is not False:
        return True, src
    if t03_ok is True and gup is not False:
        return True, src
    return False, src


def verdict_C3N(t08, t08_status, family='F2', gates=None, t03=None):
    """Spec Sec.1.4 C3-N, testable in exactly one family (KRAS).

    SUPPORTED iff R >= 0.70 with permutation chance <= 0.10 and sign agreement
    >= 0.85.  REFUTED iff R <= 0.35 or sign agreement <= 0.60 or > 50% of
    catalogued cliffs have |Delta| < 3 sigma.  Every verdict is stamped
    ``conditional`` unless G-UP completed.
    """
    if t08 is None:
        return _missing('T08', t08_status)
    members = set(config.FAMILIES.get(family, ()))
    rows = []
    for _, r in t08.iterrows():
        a, b = str(r.get('assay_a', '')), str(r.get('assay_b', ''))
        if a in members and b in members and num(r, 'R') is not None:
            rows.append(r)
    if not rows:
        return _missing('T08', 'row_missing:no within-%s replicate pair' % family)
    r = rows[0]
    R = num(r, 'R')
    chance = num(r, 'R_chance_perm')
    if chance is None:
        chance = num(r, 'perm_p')
    sign = num(r, 'sign_agreement')
    frac_lo = num(r, 'frac_cliffs_below_3sigma')      # optional, spec has no column
    cl = [
        _sup('C3N.R>=%g&chance<=%g' % (THRESH['C3N_R_sup'],
                                       THRESH['C3N_perm_chance_max']),
             _and3([_ge(R, THRESH['C3N_R_sup']),
                    _le(chance, THRESH['C3N_perm_chance_max'])]),
             'R=%s chance=%s' % (_fmt(R), _fmt(chance))),
        _sup('C3N.sign>=%g' % THRESH['C3N_sign_agreement_sup'],
             _ge(sign, THRESH['C3N_sign_agreement_sup']), 'sign=%s' % _fmt(sign)),
        _ref('C3N.R<=%g' % THRESH['C3N_R_ref'], _le(R, THRESH['C3N_R_ref']),
             'R=%s' % _fmt(R)),
        _ref('C3N.sign<=%g' % THRESH['C3N_sign_agreement_ref'],
             _le(sign, THRESH['C3N_sign_agreement_ref']), 'sign=%s' % _fmt(sign)),
    ]
    if frac_lo is not None:
        cl.append(_ref('C3N.frac|D|<3sigma>%g' % THRESH['C3N_frac_below_3sigma_ref'],
                       _gt(frac_lo, THRESH['C3N_frac_below_3sigma_ref']),
                       'frac=%s' % _fmt(frac_lo)))
    d = decide(cl, dict(R=R, sign_agreement=sign, n_shared=num(r, 'n_shared')))
    gup, gup_src = upstream_se_obtained(t03, gates)
    d.detail['stamp'] = 'calibrated' if gup else 'conditional'
    d.detail['stamp_source'] = gup_src
    d.detail['stamp_reason'] = (
        '' if gup else
        'G-UP incomplete: effect size relative to one contaminated replicate '
        'bound, not a calibrated significance (spec Sec.7 item 12)')
    if frac_lo is None:
        d.detail['note'] = ('T08 has no frac_cliffs_below_3sigma column, so the '
                            'third refutation clause of C3-N was not evaluable')
    return d


def verdict_C3L(dms_id, t07, t07_status):
    """Spec Sec.1.4 C3-L (five routes, each with a hard feasibility gate).

    SUPPORTED iff beta_a exceeds the 99.5th percentile of its N2 null AND
    (ICC >= 0.30 with CI lower > 0.15 OR dR2_oos >= 0.02 with CI lower > 0.005 OR
    AUROC_L5 >= 0.60 with p_NS2 < 0.01), "using whichever routes are feasible".
    REFUTED iff beta_a inside the N2 95% band AND ICC CI upper < 0.15 AND
    dR2_oos CI upper < 0.02.

    Reading of "whichever routes are feasible": the L1 conjunct is required when
    L1 is feasible and simply unavailable when it is not (an infeasible route
    cannot be a clause), and likewise for the recurrence disjunction.  Both the
    strict (L1 mandatory) and the feasibility-relaxed answer are recorded.
    """
    sub, st = _rows_for(t07, t07_status, 'DMS_id', dms_id)
    if sub is None:
        return _missing('T07', st)
    routes = {}
    for _, r in sub.iterrows():
        rt = (_s(r, 'route') or '').upper().replace("'", 'P').replace('L2P', 'L2P')
        routes[rt] = r
    feas = {k: flag(v, 'feasible') for k, v in routes.items()}

    # ---- L1: the sibling slope against its N2 null ----
    l1 = routes.get('L1')
    l1_ok, l1_txt = None, 'L1 absent from T07'
    l1_feasible = feas.get('L1')
    if l1 is not None and l1_feasible is not False:
        b = num(l1, 'beta_sibling')
        p995 = num(l1, 'beta_N2_p995')
        if b is not None and p995 is not None:
            l1_ok = bool(b > p995)
        l1_txt = 'beta=%s N2p995=%s n=%s' % (_fmt(b), _fmt(p995),
                                             _fmt(num(l1, 'n_units')))
    elif l1_feasible is False:
        l1_txt = 'L1 INFEASIBLE (< %d edges with |S| >= %d)' % (
            THRESH['L1_min_edges'], THRESH['L1_min_siblings'])

    # ---- the recurrence disjunction: ICC (L2/L2') or dR2 (L4) or AUROC (L5) ----
    dis, dis_txt = [], []
    for rt in ('L2', 'L2P', "L2'"):
        r = routes.get(rt)
        if r is None or flag(r, 'feasible') is False:
            continue
        icc, lo, hi = num(r, 'ICC'), num(r, 'ICC_lo95'), num(r, 'ICC_hi95')
        dis.append(_and3([_ge(icc, THRESH['C3L_ICC_sup']),
                          _gt(lo, THRESH['C3L_ICC_ci_lo_sup'])]))
        dis_txt.append('%s ICC=%s[%s,%s]' % (rt, _fmt(icc), _fmt(lo), _fmt(hi)))
    r = routes.get('L4')
    if r is not None and flag(r, 'feasible') is not False:
        d2, lo = num(r, 'dR2_oos'), num(r, 'dR2_lo95')
        dis.append(_and3([_ge(d2, THRESH['C3L_dR2_sup']),
                          _gt(lo, THRESH['C3L_dR2_ci_lo_sup'])]))
        dis_txt.append('L4 dR2=%s lo=%s' % (_fmt(d2), _fmt(lo)))
    r = routes.get('L5')
    if r is not None and flag(r, 'feasible') is not False:
        au, p = num(r, 'AUROC_L5'), num(r, 'p_NS2')
        dis.append(_and3([_ge(au, THRESH['C3L_AUROC_sup']),
                          _lt(p, THRESH['C3L_p_NS2_sup'])]))
        dis_txt.append('L5 AUROC=%s p_NS2=%s' % (_fmt(au), _fmt(p)))
    r = routes.get('L3')
    if r is not None and flag(r, 'feasible') is not False:
        dis_txt.append('L3 present (scored as C3-N)')

    dis_ok = _or3(dis)
    cl = []
    if l1_feasible is False:
        cl.append(_sup('C3L.L1(unavailable)', None,
                       l1_txt + ' -- the beta conjunct cannot be evaluated'))
    else:
        cl.append(_sup('C3L.beta>N2_p%g' % THRESH['C3L_beta_sup_pctile'], l1_ok,
                       l1_txt))
    cl.append(_sup('C3L.recurs(ICC|dR2|AUROC)', dis_ok,
                   '; '.join(dis_txt) or 'no feasible recurrence route'))

    # ---- refutation: beta inside the N2 band AND ICC hi < .15 AND dR2 hi < .02 ----
    inband = None if l1 is None else flag(l1, 'beta_in_N2_band')
    icc_his = [v for v in (num(routes.get(k), 'ICC_hi95')
                           for k in ('L2', 'L2P', "L2'") if k in routes)
               if v is not None]
    icc_hi = min(icc_his) if icc_his else None
    dr2_hi = num(routes.get('L4'), 'dR2_hi95') if 'L4' in routes else None
    # A conjunction: beta NOT in the band settles it as False on its own, so a
    # missing ICC / dR2 column can no longer stall the refutation clause.
    ref_ok = _and3([inband,
                    _lt(icc_hi, THRESH['C3L_ICC_ci_hi_ref']),
                    _lt(dr2_hi, THRESH['C3L_dR2_ci_hi_ref'])])
    cl.append(_ref('C3L.beta_in_N2_band&ICChi<%g&dR2hi<%g'
                   % (THRESH['C3L_ICC_ci_hi_ref'], THRESH['C3L_dR2_ci_hi_ref']),
                   ref_ok, 'in_band=%s ICChi=%s dR2hi=%s'
                   % (_fmt(inband), _fmt(icc_hi), _fmt(dr2_hi))))

    d = decide(cl, dict(routes_present=','.join(sorted(routes)),
                        routes_feasible=','.join(sorted(k for k, v in feas.items()
                                                        if v is True)),
                        L1_pass=l1_ok, recurrence_pass=dis_ok))
    d.detail['strict_L1_mandatory'] = decide(
        [_sup('C3L.beta>N2_p%g' % THRESH['C3L_beta_sup_pctile'], l1_ok, l1_txt),
         _sup('C3L.recurs(ICC|dR2|AUROC)', dis_ok, '')]).outcome
    return d


def verdict_C3A(dms_id, t07, t07_status):
    """Spec Sec.1.4 C3-A artefact clauses, all four of which must pass.

    1 sampling depth, 2 density (enrichment in BOTH the top and bottom
    neighbourhood-density quintile; monotone in density => sequencing-depth
    artefact), 3 floor invariance, 4 scale invariance.  C3-A has no support
    clauses of its own: it is clean or it is refuted.
    """
    sub, st = _rows_for(t07, t07_status, 'DMS_id', dms_id)
    if sub is None:
        return _missing('T07', st)
    depth = _first_num(sub, 'depth_spearman')
    best = _first_num(sub, 'best_struct_covariate')
    q1 = _first_num(sub, 'density_q1_rate')
    q5 = _first_num(sub, 'density_q5_rate')
    mono = _first_flag(sub, 'density_monotone')
    floor_ok = _first_flag(sub, 'floor_mask_invariant')
    scale_ok = _first_flag(sub, 'latent_raw_consistent')
    cl = [
        _sup('C3A.1_depth<=%g|struct>=%g' % (THRESH['C3A_depth_spearman_ref'],
                                             THRESH['C3A_struct_cov_min']),
             _nand3([_gt(depth, THRESH['C3A_depth_spearman_ref']),
                     _lt(best, THRESH['C3A_struct_cov_min'])]),
             'depth_rho=%s best_struct=%s' % (_fmt(depth), _fmt(best))),
        _sup('C3A.2_density_both_quintiles&not_monotone',
             _and3([_gt(q1, 0.0), _gt(q5, 0.0),
                    (None if mono is None else (not mono))]),
             'q1=%s q5=%s monotone=%s' % (_fmt(q1), _fmt(q5), _fmt(mono))),
        _sup('C3A.3_floor_mask_invariant', floor_ok, 'invariant=%s' % _fmt(floor_ok)),
        _sup('C3A.4_latent_raw_consistent', scale_ok,
             'consistent=%s' % _fmt(scale_ok)),
    ]
    d = decide(cl, dict(depth_spearman=depth, density_monotone=mono,
                        floor_mask_invariant=floor_ok,
                        latent_raw_consistent=scale_ok))
    # C3-A is an artefact screen: "not clean" is a refutation of the assay's
    # positive claims, so a definite failure is reported as REFUTED, while a
    # missing input stays INCONCLUSIVE.
    if d.outcome == INCONCLUSIVE and d.failing_criterion.startswith('support_failed'):
        d = Decision(REFUTED, d.failing_criterion.replace('support_failed',
                                                          'artefact_clause_failed'),
                     d.clauses, d.detail)
    return d


# --------------------------------------------------------------------------- #
# C4-S / C4-I (spec Sec.1.5)                                                  #
# --------------------------------------------------------------------------- #

def verdict_C4S(dms_id, t09, t09_status):
    """Spec Sec.1.5 C4-S, per assay: burial-matched OR >= 1.5 with p_NS1 < 0.01.
    Kill switch: REFUTED if beta_iface loses significance when rsa_iso enters."""
    sub, st = _rows_for(t09, t09_status, 'DMS_id', dms_id)
    if sub is None:
        return _missing('T09', st)
    permissible = _first_flag(sub, 'assay_permissible')
    if permissible is False:
        return Decision(INCONCLUSIVE, 'assay_not_permissible_for_C4S', (),
                        dict(assay_permissible=False))
    OR = _first_num(sub, 'OR_burial_matched')
    lo = _first_num(sub, 'OR_lo95')
    hi = _first_num(sub, 'OR_hi95')
    pns1 = _first_num(sub, 'p_NS1')
    b_adj = _first_num(sub, 'beta_iface_after_rsa')
    # The kill switch needs the Wald p of beta_iface *after* rsa_iso enters.  T09's
    # spec columns name only an ambiguous ``p_wald``; ``p_wald_after_rsa`` is read
    # first when stats_c4.py provides it, and which one was used is recorded.
    p_adj = _first_num(sub, 'p_wald_after_rsa')
    p_src = 'p_wald_after_rsa'
    if p_adj is None:
        p_adj, p_src = _first_num(sub, 'p_wald'), 'p_wald (ambiguous; see docstring)'
    cl = [
        _sup('C4S.OR>=%g&p_NS1<%g' % (THRESH['C4S_OR_sup'], THRESH['C4S_p_NS1_sup']),
             _and3([_ge(OR, THRESH['C4S_OR_sup']),
                    _lt(pns1, THRESH['C4S_p_NS1_sup'])]),
             'OR=%s [%s,%s] p_NS1=%s' % (_fmt(OR), _fmt(lo), _fmt(hi), _fmt(pns1))),
        _ref('C4S.OR<%g' % THRESH['C4S_OR_ref'], _lt(OR, THRESH['C4S_OR_ref']),
             'OR=%s' % _fmt(OR)),
        _ref('C4S.kill_switch_beta_iface_loses_sig_with_rsa',
             None if p_adj is None else bool(p_adj >= THRESH['C4S_p_NS1_sup']),
             'beta_after_rsa=%s %s=%s (alpha = C4S_p_NS1_sup = %g)'
             % (_fmt(b_adj), p_src, _fmt(p_adj), THRESH['C4S_p_NS1_sup'])),
    ]
    d = decide(cl, dict(OR=OR, OR_lo95=lo, OR_hi95=hi, p_NS1=pns1,
                        OR_ci_covers_1=(None if (lo is None or hi is None)
                                        else bool(lo <= 1.0 <= hi))))
    return d


def verdict_C4I(family, t11, t11_status, gates=None):
    """Spec Sec.1.5 C4-I: "interaction cliff" LICENSED iff F_spec >= 0.40
    (noise-corrected) and p_NS3 < 0.05 in KRAS and cliff-position PSI
    stochastically below non-cliff PSI (one-sided Mann-Whitney p < 0.05).
    REFUTED (=> the correct name is "stability cliff") iff F_spec <= 0.15 or
    median cliff PSI >= 0.75 or G11 shows dual localisation.
    """
    sub, st = _rows_for(t11, t11_status, 'family', family)
    if sub is None:
        return _missing('T11', st)
    fspec = _first_num(sub, 'F_spec_noise_corrected')
    if fspec is None:
        fspec = _first_num(sub, 'F_spec')
    pns3 = _first_num(sub, 'family_p_NS3')
    mwp = _first_num(sub, 'MW_PSI_p')
    psi_med = _first_num(sub, 'median_cliff_PSI')
    fold = _first_num(sub, 'foldaxis_spearman_rowmean_rsa')
    g11 = None if gates is None else gates.g11_dual
    cl = [
        _sup('C4I.Fspec>=%g' % THRESH['C4I_Fspec_sup'],
             _ge(fspec, THRESH['C4I_Fspec_sup']), 'F_spec=%s' % _fmt(fspec)),
        _sup('C4I.p_NS3<%g' % THRESH['C4I_p_NS3_sup'],
             _lt(pns3, THRESH['C4I_p_NS3_sup']), 'p_NS3=%s' % _fmt(pns3)),
        _sup('C4I.MW_PSI_p<%g' % THRESH['C4I_MW_p_sup'],
             _lt(mwp, THRESH['C4I_MW_p_sup']), 'MW_p=%s' % _fmt(mwp)),
        _ref('C4I.Fspec<=%g' % THRESH['C4I_Fspec_ref'],
             _le(fspec, THRESH['C4I_Fspec_ref']), 'F_spec=%s' % _fmt(fspec)),
        _ref('C4I.median_cliff_PSI>=%g' % THRESH['C4I_median_PSI_ref'],
             _ge(psi_med, THRESH['C4I_median_PSI_ref']), 'PSI=%s' % _fmt(psi_med)),
        _ref('C4I.G11_dual_localisation', g11, 'G11=%s' % _fmt(g11)),
    ]
    d = decide(cl, dict(F_spec_noise_corrected=fspec, p_NS3=pns3, MW_PSI_p=mwp,
                        foldaxis_spearman=fold))
    d.detail['classification'] = {SUPPORTED: 'interaction_cliff',
                                  REFUTED: 'stability_cliff',
                                  INCONCLUSIVE: 'undetermined'}[d.outcome]
    d.detail['foldaxis_validated'] = (None if fold is None else bool(fold > 0))
    if fold is not None and fold <= 0:
        d.detail['foldaxis_note'] = (
            'Spearman(row mean of Z, rsa_iso) = %g is NOT > 0: the fold '
            'interpretation of the partner-invariant component is UNSUPPORTED '
            '(spec Sec.1.5 fold-axis validation)' % fold)
    return d


# --------------------------------------------------------------------------- #
# C5 (spec Sec.1.6)                                                           #
# --------------------------------------------------------------------------- #

C5_MODELS = ('M1_additive_isotonic', 'M2_physchem', 'M3_msa_site_indep')


def verdict_C5(dms_id, t12, t12_status):
    """Spec Sec.1.6.  Blind spot demonstrated iff PSA_cliff <= 0.60 for all of
    M1-M3 while per-assay Spearman >= 0.30.  Practical-emptiness refutation iff
    PSA_cliff >= 0.75 for M1 -- reported even if C1-C3 all passed.

    The per-(assay, model) summary is ``AUPSA`` (the mean over the tau sweep,
    spec Sec.1.6) when the table carries it, else the mean of ``PSA_cliff`` over
    the available tau -- never a single hand-picked tau.
    """
    sub, st = _rows_for(t12, t12_status, 'DMS_id', dms_id)
    if sub is None:
        return _missing('T12', st)
    spear = _first_num(sub, 'spearman_all_rows')
    psa, src = {}, {}
    for m in C5_MODELS:
        rows = sub[sub['model'].astype(str).str.strip() == m] \
            if 'model' in sub.columns else None
        if rows is None or len(rows) == 0:
            psa[m], src[m] = None, 'model_missing'
            continue
        a = _first_num(rows, 'AUPSA')
        if a is not None:
            psa[m], src[m] = a, 'AUPSA'
            continue
        vals = [num(r, 'PSA_cliff') for _, r in rows.iterrows()]
        vals = [v for v in vals if v is not None]
        psa[m] = (sum(vals) / len(vals)) if vals else None
        src[m] = 'mean PSA_cliff over %d tau' % len(vals) if vals else 'value_missing'
    all_lo = _and3([_le(psa[mo], THRESH['C5_PSA_blindspot']) for mo in C5_MODELS])
    cl = [
        _sup('C5.PSA_cliff<=%g for M1-M3' % THRESH['C5_PSA_blindspot'], all_lo,
             ' '.join('%s=%s' % (m.split('_')[0], _fmt(psa[m])) for m in C5_MODELS)),
        _sup('C5.spearman>=%g' % THRESH['C5_spearman_min'],
             _ge(spear, THRESH['C5_spearman_min']), 'spearman=%s' % _fmt(spear)),
    ]
    d = decide(cl, dict(spearman_all_rows=spear,
                        **{'PSA_' + m.split('_')[0]: psa[m] for m in C5_MODELS}))
    m1 = psa[C5_MODELS[0]]
    d.detail['practical_emptiness'] = (
        None if m1 is None else bool(m1 >= THRESH['C5_PSA_practically_empty']))
    d.detail['psa_source'] = ','.join('%s:%s' % (m.split('_')[0], src[m])
                                      for m in C5_MODELS)
    if d.detail['practical_emptiness'] is True:
        d.detail['practical_emptiness_note'] = (
            'PSA_cliff(M1) = %.3f >= %g: a purely additive model gets cliff '
            'directions right, so C2 may be statistically true and practically '
            'EMPTY.  Spec Sec.1.6 / Sec.7 item 10: report it that way even if '
            'C1-C3 all passed.' % (m1, THRESH['C5_PSA_practically_empty']))
    return d


# --------------------------------------------------------------------------- #
# per-assay assembly                                                          #
# --------------------------------------------------------------------------- #

def _cluster_channel_additions(t15, t01):
    """Spec Sec.3 clusters.py: the cluster channel "may only ADD an assay to the
    C2 count when the pair channel is powerless (4D5), never override it"."""
    add = {}
    if t15 is None or 'DMS_id' not in getattr(t15, 'columns', []):
        return add
    nested = {}
    if t01 is not None and 'n_nested' in t01.columns:
        nested = {str(r['DMS_id']): num(r, 'n_nested') for _, r in t01.iterrows()}
    for _, r in t15.iterrows():
        a = str(r['DMS_id']).strip()
        if flag(r, 'adds_assay_to_C2_count') is not True:
            continue
        nn = nested.get(a)
        powerless = (nn is not None
                     and nn < THRESH['min_nested_for_pair_channel'])
        add[a] = dict(powerless=powerless, n_nested=nn,
                      allowed=bool(powerless),
                      family=config.ASSAYS[a].family_id if a in config.ASSAYS else '')
    return add


def per_assay_verdicts(tables=None, verbose=False):
    """One row per assay: every criterion's outcome, the clause that decided it,
    and the provenance of every input.  Assays outside the primary/arm tiers are
    still scored -- G6 needs the Z-LL1/LL2 C1 verdicts and Sec.1.2's pre-declared
    refutations name two EXCLUDED assays -- but they never contribute to a family
    count (``contributes`` is False)."""
    import pandas as pd
    tables = load_all_tables(verbose=verbose) if tables is None else tables
    t = {k: tables[k][0] for k in tables}
    s = {k: tables[k][1] for k in tables}
    gates = read_gates(t['T02'], t['T01'])
    clusters = _cluster_channel_additions(t['T15'], t['T01'])

    t01_by = {}
    if t['T01'] is not None and 'DMS_id' in t['T01'].columns:
        t01_by = {str(r['DMS_id']): r for _, r in t['T01'].iterrows()}

    rows = []
    for dms_id in config.ALL_ASSAYS:
        spec = config.ASSAYS[dms_id]
        t01r = t01_by.get(dms_id)
        n_pa = num(t01r, 'n_primary_Pa') if t01r is not None else None
        halted = gates.study_halted
        under = dms_id in gates.underpowered

        def _stamp(d, criterion):
            """Sec.7 items 1-6 and 11: a halted study and an UNDERPOWERED assay
            are INCONCLUSIVE whatever the numbers say."""
            if halted:
                return Decision(INCONCLUSIVE, gates.halt_reason, d.clauses, d.detail)
            if under:
                det = dict(d.detail)
                det['underpowered_source'] = gates.underpowered[dms_id]
                det['numbers_as_measured'] = d.outcome
                return Decision(INCONCLUSIVE, 'UNDERPOWERED_G8', d.clauses, det)
            _ = criterion
            return d

        c1 = _stamp(verdict_C1(dms_id, t['T04'], s['T04'], t['T05'], s['T05']), 'C1')
        c3l = _stamp(verdict_C3L(dms_id, t['T07'], s['T07']), 'C3L')
        c3a = _stamp(verdict_C3A(dms_id, t['T07'], s['T07']), 'C3A')
        c3l_ok = {SUPPORTED: True, REFUTED: False}.get(c3l.outcome, None)
        c2 = _stamp(verdict_C2(dms_id, t['T06'], s['T06'], n_Pa=n_pa,
                               c3l_ok=c3l_ok, g7=gates.g7, scale='latent'), 'C2')
        c2raw = verdict_C2(dms_id, t['T06'], s['T06'], n_Pa=n_pa, c3l_ok=c3l_ok,
                           g7=gates.g7, scale='raw')
        # Sec.1.4 C3-A.4: a positive that holds on raw but not on latent is
        # DISCARDED, never reported as positive.  The latent verdict is the one
        # that ships; the raw column is the sensitivity record.
        c4s = _stamp(verdict_C4S(dms_id, t['T09'], s['T09']), 'C4S')
        c5 = _stamp(verdict_C5(dms_id, t['T12'], s['T12']), 'C5')

        r = dict(DMS_id=dms_id, tier=spec.tier, family_id=spec.family_id,
                 registered=spec.registered, n_primary_Pa=('' if n_pa is None
                                                           else int(n_pa)),
                 underpowered_G8=under,
                 underpowered_source=gates.underpowered.get(dms_id, ''),
                 study_halted=halted,
                 # Sec.7 item 11: an UNDERPOWERED assay "never contributes to a
                 # family count".  ``counts_in_family`` is membership of its own
                 # family row (F8 included); ``contributes`` is membership of the
                 # k-of-7 primary count, which F8 is excluded from by construction.
                 counts_in_family=bool(spec.family_id and not under and not halted),
                 contributes=bool(spec.tier == 'PRIMARY' and spec.family_id
                                  and not under and not halted),
                 eligible_C1=spec.eligible_C1, eligible_C2=spec.eligible_C2,
                 eligible_C3L=spec.eligible_C3L, eligible_C4S=spec.eligible_C4S,
                 eligible_C4I=spec.eligible_C4I,
                 cluster_channel_adds_C2=(dms_id in clusters),
                 cluster_channel_allowed=clusters.get(dms_id, {}).get('allowed', ''))
        r.update(c1.as_row('verdict_C1'))
        r.update(c2.as_row('verdict_C2'))
        r['verdict_C2_raw'] = c2raw.outcome
        r['latent_raw_consistent'] = (c2.outcome == c2raw.outcome)
        r['verdict_C2_raw_only_positive_discarded'] = bool(
            c2raw.outcome == SUPPORTED and c2.outcome != SUPPORTED)
        r.update(c3l.as_row('verdict_C3L'))
        r.update(c3a.as_row('verdict_C3A'))
        r.update(c4s.as_row('verdict_C4S'))
        r.update(c5.as_row('verdict_C5'))
        rows.append(r)
    df = pd.DataFrame(rows)
    df.attrs['gates'] = gates
    df.attrs['table_status'] = s
    df.attrs['clusters'] = clusters
    return df


# --------------------------------------------------------------------------- #
# family aggregation                                                          #
# --------------------------------------------------------------------------- #

#: How a family with more than one member is called from its members.  Spec Sec.1
#: counts "families supported", never "assays supported", but does not say how a
#: multi-assay family votes.  Only F1 (2 assays) and F2 (5 KRAS) are affected.
#: MAJORITY is the default because Sec.2/Sec.8 insist the five KRAS assays are
#: ~one independent system: 'any' would let one KRAS assay carry the family,
#: 'all' would let one failure sink it.  A 50/50 split is INCONCLUSIVE, and
#: ``C1_pos/C1_neg`` always carries the raw counts so any rule is re-derivable.
FAMILY_RULES = ('majority', 'any', 'all')


def family_call(outcomes, rule='majority'):
    """(call, n_pos, n_neg, n_inconclusive) for one family and one criterion."""
    pos = sum(1 for o in outcomes if o == SUPPORTED)
    neg = sum(1 for o in outcomes if o == REFUTED)
    inc = sum(1 for o in outcomes if o == INCONCLUSIVE)
    if not outcomes:
        return NOT_TESTABLE, 0, 0, 0
    if rule == 'any':
        call = SUPPORTED if pos else (REFUTED if neg else INCONCLUSIVE)
    elif rule == 'all':
        call = (SUPPORTED if pos == len(outcomes) else
                (REFUTED if neg else INCONCLUSIVE))
    else:
        if pos > neg and pos > inc:
            call = SUPPORTED
        elif neg > pos and neg >= inc:
            call = REFUTED
        else:
            call = INCONCLUSIVE
    return call, pos, neg, inc


# --------------------------------------------------------------------------- #
# meta-analysis (spec Sec.1: "the evidence is the per-family CIs and a           #
# cluster-df random-effects meta-analysis")                                    #
# --------------------------------------------------------------------------- #

def meta_random_effects(effects, ses):
    """DerSimonian-Laird random effects with a Knapp-Hartung (cluster-df) t
    interval.  Not a new statistic: it pools per-assay quantities that the
    statistics tables already report *with* their CIs.  ``statsmodels`` is not a
    dependency (spec Sec.4), so this is the ~20 lines on scipy it asks for.

    Returns ``{effect, ci_lo, ci_hi, tau2, Q, I2, k, method}``; ``k == 1`` falls
    back to that single estimate's own normal interval and says so.
    """
    import numpy as np
    from scipy import stats
    y = np.asarray([e for e, s in zip(effects, ses)
                    if e is not None and s is not None and s > 0], dtype=float)
    se = np.asarray([s for e, s in zip(effects, ses)
                     if e is not None and s is not None and s > 0], dtype=float)
    k = y.size
    if k == 0:
        return dict(effect=None, ci_lo=None, ci_hi=None, tau2=None, Q=None,
                    I2=None, k=0, method='no usable effect/SE pair')
    z = float(stats.norm.ppf(0.975))
    if k == 1:
        return dict(effect=float(y[0]), ci_lo=float(y[0] - z * se[0]),
                    ci_hi=float(y[0] + z * se[0]), tau2=None, Q=None, I2=None,
                    k=1, method='single estimate; no between-assay heterogeneity '
                               'estimable, normal interval')
    v = se ** 2
    w = 1.0 / v
    fe = float((w * y).sum() / w.sum())
    Q = float((w * (y - fe) ** 2).sum())
    C = float(w.sum() - (w ** 2).sum() / w.sum())
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    ws = 1.0 / (v + tau2)
    mu = float((ws * y).sum() / ws.sum())
    var = 1.0 / ws.sum()
    q_hk = float((ws * (y - mu) ** 2).sum() / (k - 1))
    se_hk = float(np.sqrt(max(q_hk, 1.0) * var)) if k > 1 else float(np.sqrt(var))
    tcrit = float(stats.t.ppf(0.975, k - 1))
    I2 = float(max(0.0, (Q - (k - 1)) / Q) * 100.0) if Q > 0 else 0.0
    return dict(effect=mu, ci_lo=mu - tcrit * se_hk, ci_hi=mu + tcrit * se_hk,
                tau2=tau2, Q=Q, I2=I2, k=k,
                method='DerSimonian-Laird tau^2 + Knapp-Hartung t_%d interval'
                       % (k - 1))


def meta_effect_per_assay(t06, t06_status, tau=None, scale='latent', unit='sigma'):
    """The per-assay effect fed to the meta-analysis: ``log T_N2`` at one tau,
    with an SE recovered from the table's own block-bootstrap CI.

    ``tau`` defaults to ``min(config.TAU_WINDOW)`` -- the lower edge of the
    verdict window, where the sweep has the most edges and therefore the tightest
    CI.  It is an argument, not a constant, so a reviewer can re-pool at any tau.
    """
    import numpy as np
    from scipy import stats
    out = {}
    if t06 is None or 'tau' not in getattr(t06, 'columns', []):
        return out, 'T06 %s' % t06_status
    tau = min(config.TAU_WINDOW) if tau is None else tau
    z = float(stats.norm.ppf(0.975))
    for _, r in t06.iterrows():
        if 'scale' in t06.columns and (_s(r, 'scale') or '').lower() != scale:
            continue
        if 'unit' in t06.columns and (_s(r, 'unit') or '').lower() != unit.lower():
            continue
        try:
            if float(str(r['tau']).strip()) != float(tau):
                continue
        except (ValueError, TypeError):
            continue
        T, lo, hi = num(r, 'T_N2'), num(r, 'T_N2_lo95'), num(r, 'T_N2_hi95')
        if T is None or T <= 0:
            continue
        se = None
        if lo is not None and hi is not None and lo > 0 and hi > lo:
            se = (np.log(hi) - np.log(lo)) / (2.0 * z)
        out[str(r['DMS_id'])] = (float(np.log(T)), se)
    return out, 'log T_N2 at tau=%g, %s scale, %s units' % (tau, scale, unit)


# --------------------------------------------------------------------------- #
# T14                                                                         #
# --------------------------------------------------------------------------- #

def _pn(pos, neg):
    return '%d/%d' % (pos, neg)


def _fmt_family_result(d, detail_key):
    """One T14 result cell: NOT_TESTABLE when the family simply has no such row,
    ``INCONCLUSIVE(<reason>)`` when the input is absent, else the outcome plus its
    qualifier (the C3-N stamp, the C4-I classification)."""
    if d is None:
        return NOT_TESTABLE
    if d.failing_criterion.startswith('row_missing'):
        return NOT_TESTABLE
    if d.failing_criterion.startswith(('table_missing', 'table_empty',
                                       'column_missing', 'table_unreadable')):
        return '%s(%s)' % (d.outcome, d.failing_criterion.split(':')[0])
    q = d.detail.get(detail_key, '')
    return '%s(%s)' % (d.outcome, q) if q else d.outcome


def build_T14(per_assay, tables=None, rule='majority', meta_tau=None, verbose=False):
    """T14 with the exact spec Sec.6 columns plus its footer rows: the k-of-7
    counts, the G9 empirical FPR of the rule, the binomial p of the count
    (0.2266 at 5/7), and the aggregate SUPPORTED / REFUTED / INCONCLUSIVE call."""
    import pandas as pd
    from scipy import stats
    tables = load_all_tables() if tables is None else tables
    t = {k: tables[k][0] for k in tables}
    s = {k: tables[k][1] for k in tables}
    gates = per_assay.attrs.get('gates') or read_gates(t['T02'])
    by = {str(r['DMS_id']): r for _, r in per_assay.iterrows()}
    eff, eff_src = meta_effect_per_assay(t['T06'], s['T06'], tau=meta_tau)

    counts = {'C1': {}, 'C2': {}, 'C3': {}}
    rows, fam_meta = [], {}
    for fam, members in config.FAMILIES.items():
        contrib = [m for m in members if by[m]['counts_in_family']]
        o = {}
        for crit, col in (('C1', 'verdict_C1'), ('C2', 'verdict_C2'),
                          ('C3L', 'verdict_C3L'), ('C3A', 'verdict_C3A'),
                          ('C5', 'verdict_C5')):
            o[crit] = [by[m][col] for m in contrib]
        # Sec.1.5 C4-S counts over "7 ELIGIBLE assays", so its denominator is the
        # eligible subset, not every member: an assay whose interface contrast is
        # undefined (5A12_VEGF 0/9, Z-ZpA963_HL1 6/6) is not a C4-S abstention.
        o['C4S'] = [by[m]['verdict_C4S'] for m in contrib if by[m]['eligible_C4S']]
        c1 = family_call(o['C1'], rule)
        c2 = family_call(o['C2'], rule)
        c3l = family_call(o['C3L'], rule)
        c3a = family_call(o['C3A'], rule)
        c4s = family_call(o['C4S'], rule)
        c5 = family_call(o['C5'], rule)
        # C3-N needs two measurements of the same site pair, so a single-member
        # family cannot test it at all (spec Sec.1.4: "testable in exactly one
        # family (KRAS)") -- that is NOT_TESTABLE, not INCONCLUSIVE.
        if len(members) < 2:
            c3n, c3n_res = None, NOT_TESTABLE
        else:
            c3n = verdict_C3N(t['T08'], s['T08'], family=fam, gates=gates,
                              t03=t['T03'])
            c3n_res = _fmt_family_result(c3n, 'stamp')
        c4i = verdict_C4I(fam, t['T11'], s['T11'], gates=gates)
        c4i_res = _fmt_family_result(c4i, 'classification')
        # C3 as a whole: C3-L is the load-bearing route; C3-A must be clean.
        if c3l[0] == SUPPORTED and c3a[0] == REFUTED:
            fam_c3 = INCONCLUSIVE
        elif c3l[0] == SUPPORTED:
            fam_c3 = SUPPORTED
        elif c3l[0] == REFUTED:
            fam_c3 = REFUTED
        else:
            fam_c3 = c3l[0]
        m = meta_random_effects([eff.get(x, (None, None))[0] for x in contrib],
                               [eff.get(x, (None, None))[1] for x in contrib])
        fam_meta[fam] = m
        if fam != 'F8':
            counts['C1'][fam] = c1[0]
            counts['C2'][fam] = c2[0]
            counts['C3'][fam] = fam_c3
        note = []
        if fam == 'F8':
            note.append('ARM: hypercube arm, its OWN denominator; never folded '
                        'into the k-of-7 primary count')
        dropped = [m2 for m2 in members if m2 not in contrib]
        if dropped:
            note.append('excluded from the family count: ' + ', '.join(
                '%s(%s)' % (d, 'UNDERPOWERED_G8' if by[d]['underpowered_G8']
                            else ('STUDY_HALTED' if by[d]['study_halted']
                                  else by[d]['tier'])) for d in dropped))
        if m['k'] and m.get('method'):
            note.append('meta: %s' % m['method'])
        rows.append({
            'family_id': fam,
            'member_assays': ';'.join(members),
            'n_eligible': len(contrib),
            'C1_pos/C1_neg': _pn(c1[1], c1[2]),
            'C2_pos/C2_neg': _pn(c2[1], c2[2]),
            'C3L_pos/C3L_neg': _pn(c3l[1], c3l[2]),
            'C3N_result': c3n_res,
            'C4S_result': '%s %s' % (c4s[0], _pn(c4s[1], c4s[2])),
            'C4I_result': c4i_res,
            'C5_result': '%s %s' % (c5[0], _pn(c5[1], c5[2])),
            'family_verdict_C1': c1[0],
            'family_verdict_C2': c2[0],
            'family_verdict_C3': fam_c3,
            'all_three': (SUPPORTED if (c1[0] == c2[0] == fam_c3 == SUPPORTED)
                          else (REFUTED if REFUTED in (c1[0], c2[0], fam_c3)
                                else INCONCLUSIVE)),
            'meta_effect': '' if m['effect'] is None else round(m['effect'], 6),
            'meta_ci_lo': '' if m['ci_lo'] is None else round(m['ci_lo'], 6),
            'meta_ci_hi': '' if m['ci_hi'] is None else round(m['ci_hi'], 6),
            'notes': ' | '.join(note),
        })

    # ------------------------------- footers ------------------------------- #
    K = config.K_FAMILIES
    k_cfg = dict(C1_true=THRESH['C1_family_k_true'],
                 C1_ref=THRESH['C1_family_k_refuted'],
                 C2_true=THRESH['C2_family_k_true'],
                 C2_ref=THRESH['C2_family_k_refuted'],
                 C3_true=THRESH['C3_family_k_true'],
                 C3_ref=THRESH['C3_family_k_refuted'])
    k_eff = dict(k_cfg)
    for crit in ('C1', 'C2', 'C3'):
        if crit in gates.g9.get('k', {}):
            k_eff['%s_true' % crit] = int(gates.g9['k'][crit])
    npos = {c: sum(1 for v in counts[c].values() if v == SUPPORTED)
            for c in counts}
    nneg = {c: sum(1 for v in counts[c].values() if v == REFUTED) for c in counts}
    #: Families that returned a DETERMINATE call.  The load-bearing guard below.
    ndet = {c: npos[c] + nneg[c] for c in counts}

    # Sec.1.4's two extra conjuncts for the C3 aggregate.
    c3n_fam = ''
    c3n_out = INCONCLUSIVE
    for fam in config.FAMILIES:
        if fam == 'F8' or len(config.FAMILIES[fam]) < 2:
            continue
        d = verdict_C3N(t['T08'], s['T08'], family=fam, gates=gates, t03=t['T03'])
        if not d.failing_criterion.startswith('row_missing'):
            c3n_fam, c3n_out = fam, d.outcome
            if d.outcome == SUPPORTED:
                break
    c3_extra = dict(
        c3a_dirty=[str(r['DMS_id']) for _, r in per_assay.iterrows()
                   if r['contributes'] and r['verdict_C3A'] == REFUTED],
        c3n=c3n_out, c3n_family=c3n_fam or 'no family carries a replicate pair')

    def _agg(crit):
        """Spec Sec.1.2-1.4 aggregate rules, in their own words -- plus the one
        guard those rules need and do not state.

        "C2 REFUTED iff supported in <= 1 of 7" is *vacuously* satisfied when no
        family was evaluated at all (0 <= 1), which would turn a missing table
        into the published negative headline.  A "supported in at most k"
        refutation therefore fires only when EVERY family actually returned a
        determinate call; otherwise the aggregate is INCONCLUSIVE and says how
        many families were determinate.  The "supported in >= k" and "refuted in
        >= k" rules need no guard: they require k real observations.

        C3 additionally carries Sec.1.4's two conjuncts: C3-A clean everywhere and
        C3-N supported in the one family where it is testable.
        """
        if gates.study_halted:
            return INCONCLUSIVE, gates.halt_reason
        p, n, det = npos[crit], nneg[crit], ndet[crit]
        if crit == 'C1':
            if n >= k_eff['C1_ref']:
                return REFUTED, 'refuted in %d of %d families (>= %d)' % (
                    n, K, k_eff['C1_ref'])
            if p >= k_eff['C1_true']:
                return TRUE, 'supported in %d of %d families (>= %d)' % (
                    p, K, k_eff['C1_true'])
            return INCONCLUSIVE, 'supported in %d of %d (< %d), refuted in %d ' \
                                 '(< %d); %d/%d families determinate' % (
                                     p, K, k_eff['C1_true'], n,
                                     k_eff['C1_ref'], det, K)
        label, kt, kr = (('', k_eff['C2_true'], k_eff['C2_ref']) if crit == 'C2'
                         else ('C3-L ', k_eff['C3_true'], k_eff['C3_ref']))
        if p >= kt:
            if crit == 'C3':
                # Sec.1.4: "C3 TRUE iff C3-L supported in >= 3 of 7 families,
                # C3-A clean everywhere, and C3-N supported in the one family
                # where it is testable."
                if c3_extra['c3a_dirty']:
                    return INCONCLUSIVE, (
                        'C3-L supported in %d of %d (>= %d) but C3-A is NOT clean '
                        'everywhere: %s' % (p, K, kt,
                                            ','.join(c3_extra['c3a_dirty'])))
                if c3_extra['c3n'] != SUPPORTED:
                    return INCONCLUSIVE, (
                        'C3-L supported in %d of %d (>= %d) but C3-N is %s in the '
                        'one family where it is testable (%s)'
                        % (p, K, kt, c3_extra['c3n'], c3_extra['c3n_family']))
            return TRUE, '%ssupported in %d of %d families (>= %d)' % (
                label, p, K, kt)
        if p <= kr:
            if det < K:
                return INCONCLUSIVE, (
                    '%ssupported in %d of %d, which would meet the "<= %d" '
                    'refutation -- but only %d of %d families returned a '
                    'determinate call, so the refutation would be VACUOUS '
                    '(see _agg docstring)' % (label, p, K, kr, det, K))
            return REFUTED, '%ssupported in only %d of %d families (<= %d), all ' \
                            '%d determinate' % (label, p, K, kr, K)
        return INCONCLUSIVE, '%ssupported in %d of %d (< %d) but > %d; %d/%d ' \
                             'determinate' % (label, p, K, kt, kr, det, K)

    agg = {c: _agg(c) for c in ('C1', 'C2', 'C3')}
    binom = {c: float(stats.binom.sf(npos[c] - 1, K, 0.5)) if npos[c] > 0 else 1.0
             for c in npos}
    ref5of7 = float(stats.binom.sf(THRESH['C1_family_k_true'] - 1, K, 0.5))
    assert abs(ref5of7 - config.BINOM_P_5OF7) < 5e-5, \
        'binomial p of %d-of-%d = %.6f, config says %.4f' % (
            THRESH['C1_family_k_true'], K, ref5of7, config.BINOM_P_5OF7)

    fam_meta_eff = [fam_meta[f]['effect'] for f in config.FAMILIES if f != 'F8']
    fam_meta_se = []
    zq = float(stats.norm.ppf(0.975))
    for f in config.FAMILIES:
        if f == 'F8':
            continue
        mm = fam_meta[f]
        fam_meta_se.append(None if (mm['ci_lo'] is None or mm['ci_hi'] is None)
                           else (mm['ci_hi'] - mm['ci_lo']) / (2 * zq))
    study_meta = meta_random_effects(fam_meta_eff, fam_meta_se)

    def _f(fid, note, **kw):
        r = {c: '' for c in T14_COLUMNS}
        r['family_id'] = fid
        r['notes'] = note
        r.update(kw)
        rows.append(r)

    _f('FOOTER:k_of_%d' % K,
       'family-level counts over F1..F%d (F8 arm excluded by construction); '
       'rule for a multi-assay family = %s (see FAMILY_RULES).  DETERMINATE '
       'families: C1 %d/%d, C2 %d/%d, C3-L %d/%d -- a "supported in <= k" '
       'refutation is withheld as VACUOUS unless the count is %d/%d.'
       % (K, rule, ndet['C1'], K, ndet['C2'], K, ndet['C3'], K, K, K),
       n_eligible=sum(1 for f in config.FAMILIES if f != 'F8'
                      and counts['C1'].get(f) != NOT_TESTABLE),
       **{'C1_pos/C1_neg': _pn(npos['C1'], nneg['C1']),
          'C2_pos/C2_neg': _pn(npos['C2'], nneg['C2']),
          'C3L_pos/C3L_neg': _pn(npos['C3'], nneg['C3'])})
    _f('FOOTER:k_thresholds',
       'config THRESH k-of-%d thresholds%s' % (
           K, '' if k_eff == k_cfg else '; G9 TIGHTENED: %s' % (
               {a: b for a, b in k_eff.items() if k_cfg[a] != b},)),
       **{'C1_pos/C1_neg': 'true>=%d,ref>=%d' % (k_eff['C1_true'], k_eff['C1_ref']),
          'C2_pos/C2_neg': 'true>=%d,ref<=%d' % (k_eff['C2_true'], k_eff['C2_ref']),
          'C3L_pos/C3L_neg': 'true>=%d,ref<=%d' % (k_eff['C3_true'],
                                                   k_eff['C3_ref'])})
    _f('FOOTER:G9_rule_FPR',
       'G9 empirical family-level FPR of the k-of-%d rule over %d complete N1 '
       'surrogate datasets; ceiling %g.  Source: %s' % (
           K, THRESH['G9_n_datasets'], THRESH['G9_family_fpr_max'],
           gates.g9['source']),
       meta_effect=('PENDING' if gates.g9['fpr'] is None else gates.g9['fpr']),
       n_eligible=('' if gates.g9['fpr'] is None
                   else ('PASS' if gates.g9['fpr'] <= THRESH['G9_family_fpr_max']
                         else 'FAIL')))
    _f('FOOTER:binomial_p_of_count',
       'one-sided binomial P(X >= k) with X ~ Binom(%d, 0.5).  Reference: '
       '%d-of-%d gives p = %.4f (config.BINOM_P_5OF7 = %.4f).  Spec Sec.1: this '
       'is a GENERALITY statement, never the evidence -- the evidence is the '
       'per-family CIs and the cluster-df random-effects meta-analysis.'
       % (K, THRESH['C1_family_k_true'], K, ref5of7, config.BINOM_P_5OF7),
       **{'C1_pos/C1_neg': round(binom['C1'], 4),
          'C2_pos/C2_neg': round(binom['C2'], 4),
          'C3L_pos/C3L_neg': round(binom['C3'], 4)})
    _f('FOOTER:meta_analysis',
       'cluster-df random-effects meta-analysis over the %d primary families. '
       'Effect = %s.  %s.  Per-family rows carry the within-family pool.'
       % (K, eff_src, study_meta['method']),
       meta_effect=('' if study_meta['effect'] is None
                    else round(study_meta['effect'], 6)),
       meta_ci_lo=('' if study_meta['ci_lo'] is None
                   else round(study_meta['ci_lo'], 6)),
       meta_ci_hi=('' if study_meta['ci_hi'] is None
                   else round(study_meta['ci_hi'], 6)),
       n_eligible=study_meta['k'])

    pre = check_predeclared_C1_refutations(t['T04'], s['T04'], verbose=False)
    n_rep = sum(1 for r in pre if r['reproduces'])
    _f('FOOTER:C1_predeclared_refutations',
       'Spec Sec.1.2 "must reproduce, else the implementation is wrong": %d/%d '
       'reproduce as REFUTED.  %s' % (
           n_rep, len(pre),
           '; '.join('%s SI_spec=%.3f SI_obs=%s -> %s'
                     % (r['DMS_id'].split('_')[0] + '/' + r['DMS_id'][-8:],
                        r['SI_spec'], _fmt(r['SI_obs']), r['verdict'])
                     for r in pre)),
       n_eligible='%d/%d' % (n_rep, len(pre)))

    headline, all_three = aggregate_headline(agg, per_assay, gates)
    _f('FOOTER:AGGREGATE', headline,
       family_verdict_C1='%s (%s)' % agg['C1'], family_verdict_C2='%s (%s)' % agg['C2'],
       family_verdict_C3='%s (%s)' % agg['C3'], all_three=all_three)
    _f('FOOTER:HEADLINE_LIMITATION', config.HEADLINE_LIMITATION)
    _f('FOOTER:PROVENANCE',
       'env=%s | git=%s dirty=%s | tables: %s | G7=%s | UNDERPOWERED=%s | '
       'family rule=%s | %s'
       % ('.'.join(config.EXPECTED_ENV), _git()['commit'][:12], _git()['dirty_files'],
          ','.join('%s:%s' % (k, v) for k, v in sorted(s.items()) if v != 'ok')
          or 'all present',
          gates.g7['source'],
          ','.join(sorted(gates.underpowered)) or 'none',
          rule, 'STUDY HALTED: %s' % gates.halt_reason if gates.study_halted
          else 'no halting gate has failed'))

    df = pd.DataFrame(rows)
    for c in T14_COLUMNS:
        if c not in df.columns:
            df[c] = ''
    df = df[T14_COLUMNS]
    df.attrs['aggregate'] = agg
    df.attrs['counts'] = counts
    df.attrs['binomial_p'] = binom
    df.attrs['k_effective'] = k_eff
    df.attrs['headline'] = headline
    df.attrs['study_meta'] = study_meta
    if verbose:
        print('[verdict] T14: %d family rows + %d footer rows'
              % (len(config.FAMILIES), len(df) - len(config.FAMILIES)))
    return df


def aggregate_headline(agg, per_assay, gates):
    """The aggregate SUPPORTED / REFUTED / INCONCLUSIVE call, in the spec's own
    words for each of the four ways this study can return a negative."""
    c1, c2, c3 = agg['C1'][0], agg['C2'][0], agg['C3'][0]
    if gates.study_halted:
        return ('INCONCLUSIVE -- STUDY HALTED. %s.  Spec Sec.7: no observed '
                'number is readable.' % gates.halt_reason), INCONCLUSIVE
    empt = [str(r['DMS_id']) for _, r in per_assay.iterrows()
            if r.get('verdict_C5_practical_emptiness') is True]
    parts = []
    if c2 == REFUTED:
        parts.append('AGGREGATE = REFUTED (C2). Spec Sec.1.3 / Sec.7 item 8: '
                     '"BindingGYM binding landscapes are additive-plus-monotone-'
                     'link-plus-heteroscedastic-noise to within the resolution of '
                     'the data; no cliff component is detectable." That is '
                     'publishable and must be written as the finding.')
    if c1 == REFUTED:
        parts.append('C1 REFUTED: "premise C1 is not a general property of '
                     'BindingGYM landscapes" (Sec.7 item 7) -- smoothness is a '
                     'property of well-sampled designed libraries, not of '
                     'mutation landscapes in general.')
    if c3 == REFUTED:
        parts.append('C3-L REFUTED: the deviations do not recur, i.e. they are '
                     'indistinguishable from heteroscedastic measurement noise '
                     '(Sec.7 item 9).  Do not rename it a cliff.')
    if empt:
        parts.append('PRACTICAL EMPTINESS fires on %s: PSA_cliff(M1) >= %g, so C2 '
                     'may be statistically true and practically empty (Sec.7 item '
                     '10) -- report it that way even if C1-C3 all passed.'
                     % (','.join(empt), THRESH['C5_PSA_practically_empty']))
    if c1 == c2 == c3 == TRUE:
        call = SUPPORTED
        parts.insert(0, 'AGGREGATE = SUPPORTED: C1, C2 and C3 all TRUE over the '
                        '%d families.' % config.K_FAMILIES)
    elif REFUTED in (c1, c2, c3):
        call = REFUTED
        if not parts or not parts[0].startswith('AGGREGATE'):
            parts.insert(0, 'AGGREGATE = REFUTED: C1=%s, C2=%s, C3=%s.'
                            % (c1, c2, c3))
    else:
        call = INCONCLUSIVE
        parts.insert(0, 'AGGREGATE = INCONCLUSIVE: C1=%s, C2=%s, C3=%s.'
                        % (c1, c2, c3))
    parts.append('Generality is a count over %d CORRELATED families, binomial p = '
                 '%.4f at %d-of-%d; the evidence is the per-family CIs.'
                 % (config.K_FAMILIES, config.BINOM_P_5OF7,
                    THRESH['C1_family_k_true'], config.K_FAMILIES))
    return ' '.join(parts), call


def _git():
    try:
        from cliff.pairs import git_provenance
        return git_provenance()
    except Exception:                                          # pragma: no cover
        return dict(commit='', dirty_files=-1)


# --------------------------------------------------------------------------- #
# write-back                                                                  #
# --------------------------------------------------------------------------- #

def write_back_verdicts(per_assay, tables=None, verbose=True):
    """Fill the ``verdict_*`` / ``failing_criterion`` / ``classification`` columns
    of T04/T06/T07/T08/T11/T12 (spec Sec.3: verdict.py "applies THRESH; emits
    T1..T12").  NOTHING else in those tables is ever touched: every other column
    is written back byte-identically.

    Each target is RE-READ from disk immediately before it is rewritten, so a
    table another stage produced after this verdict pass started is never
    clobbered -- only the verdict columns are overlaid onto the fresh content,
    and a row that has appeared since gets an empty verdict rather than a stale
    one."""
    import pandas as pd
    tables = load_all_tables() if tables is None else tables
    by = {str(r['DMS_id']): r for _, r in per_assay.iterrows()}
    gates = per_assay.attrs.get('gates')
    t = {k: tables[k][0] for k in tables}
    s = {k: tables[k][1] for k in tables}
    done = []
    for name, cols in sorted(WRITE_BACK.items()):
        if t.get(name) is None or len(t[name]) == 0:
            continue
        df, st_fresh = load_table(name)          # re-read: close the RMW window
        if df is None or len(df) == 0 or st_fresh != 'ok':
            continue
        if len(df) != len(t[name]):
            print('[verdict] NOTE %s changed on disk during this pass (%d -> %d '
                  'rows); overlaying the verdict columns onto the FRESH copy'
                  % (name, len(t[name]), len(df)))
        t[name] = df
        df = df.copy()
        key = 'family' if name == 'T11' else ('assay_a' if name == 'T08' else 'DMS_id')
        if key not in df.columns:
            continue
        vals = {c: [] for c in cols}
        for _, r in df.iterrows():
            kv = str(r[key]).strip()
            if name == 'T04':
                a = by.get(kv)
                vals['verdict_C1'].append('' if a is None else a['verdict_C1'])
                vals['failing_criterion'].append(
                    '' if a is None else a['verdict_C1_failing_criterion'])
            elif name == 'T06':
                a = by.get(kv)
                sc = (_s(r, 'scale') or 'latent').lower()
                col = 'verdict_C2_raw' if sc == 'raw' else 'verdict_C2'
                vals['verdict_C2'].append('' if a is None else a[col])
                vals['failing_criterion'].append(
                    '' if a is None or sc == 'raw'
                    else a['verdict_C2_failing_criterion'])
            elif name == 'T07':
                a = by.get(kv)
                vals['verdict_C3L'].append('' if a is None else a['verdict_C3L'])
                vals['verdict_C3A'].append('' if a is None else a['verdict_C3A'])
                vals['failing_criterion'].append(
                    '' if a is None else a['verdict_C3L_failing_criterion'])
            elif name == 'T08':
                fam = _family_of_pair(kv, str(r.get('assay_b', '')).strip())
                d = verdict_C3N(t['T08'], s['T08'], family=fam or 'F2', gates=gates,
                                t03=t.get('T03'))
                vals['verdict_C3N'].append(d.outcome)
                vals['verdict_stamp'].append(d.detail.get('stamp', 'conditional'))
            elif name == 'T11':
                d = verdict_C4I(kv, t['T11'], s['T11'], gates=gates)
                vals['classification'].append(d.detail.get('classification',
                                                           'undetermined'))
            elif name == 'T12':
                a = by.get(kv)
                vals['verdict_blindspot'].append('' if a is None else a['verdict_C5'])
                vals['verdict_practical_emptiness'].append(
                    '' if a is None else
                    str(a.get('verdict_C5_practical_emptiness', '')))
        for c in cols:
            df[c] = vals[c]
        p = artifact_path(name)
        df.to_csv(p, index=False)
        done.append('%s(%s)' % (name, ','.join(cols)))
        if verbose:
            print('[verdict] wrote back %s -> %s' % (','.join(cols), p))
    _ = pd
    return done


def _family_of_pair(a, b):
    for fam, members in config.FAMILIES.items():
        if a in members and b in members:
            return fam
    return config.ASSAYS[a].family_id if a in config.ASSAYS else ''


# --------------------------------------------------------------------------- #
# driver                                                                      #
# --------------------------------------------------------------------------- #

def run(write=True, write_back=True, rule='majority', meta_tau=None, verbose=True):
    """Read the Sec.6 tables, apply THRESH, write T14 (+ T14a) and, unless
    ``write_back=False``, the verdict columns of T04/T06/T07/T08/T11/T12."""
    config.assert_env()
    tables = load_all_tables(verbose=verbose)
    pa = per_assay_verdicts(tables)
    t14 = build_T14(pa, tables, rule=rule, meta_tau=meta_tau, verbose=verbose)
    out = {}
    if write:
        os.makedirs(PATHS.artifacts, exist_ok=True)
        p14 = os.path.join(PATHS.artifacts, T14_NAME)
        t14.to_csv(p14, index=False)
        p14a = os.path.join(PATHS.artifacts, T14A_NAME)
        pa.to_csv(p14a, index=False)
        out['T14'] = p14
        out['T14a'] = p14a
        if verbose:
            print('[verdict] wrote %s (%d x %d)' % (p14, len(t14), len(t14.columns)))
            print('[verdict] wrote %s (%d x %d)' % (p14a, len(pa), len(pa.columns)))
    if write_back:
        out['write_back'] = write_back_verdicts(pa, tables, verbose=verbose)
    return dict(per_assay=pa, T14=t14, tables=tables, paths=out,
                gates=pa.attrs['gates'], aggregate=t14.attrs['aggregate'],
                headline=t14.attrs['headline'])


# --------------------------------------------------------------------------- #
# SELF-CHECK FIXTURES.  Never data, never written to artifacts/.               #
# --------------------------------------------------------------------------- #

def measured_C1_inputs(assays=None):
    """SELF-CHECK FIXTURE, not a producer of T04 -- ``variogram.py`` owns T04.

    Computes SI = G(1)/GMD and V(1)/V(inf) straight from the cached pair index
    arrays with the spec's own closed forms (Sec.1.2), so that the C1 rule can be
    exercised on REAL numbers before variogram.py exists.  ``h = 1`` is taken as
    the union of the nested and same-site pair sets, because that is the set with
    code-vector Hamming 1 and it is the ONLY one of the three candidate readings
    that reproduces the spec's own pre-declared SI values (see the module's
    self-check output).
    """
    import numpy as np
    import pandas as pd
    ids = list(config.ALL_ASSAYS) if assays is None else list(assays)
    rows = []
    for a in ids:
        kp = os.path.join(PATHS.keys, a + '.npz')
        if not os.path.exists(kp):
            continue
        y = np.load(kp, allow_pickle=False)['y'].astype(np.float64)
        n = y.size
        ys = np.sort(y)
        i = np.arange(1, n + 1)
        gmd = float(2.0 * ((2 * i - n - 1) * ys).sum() / (n * (n - 1)))
        vinf = float(np.var(y) * n / (n - 1))
        idxs = []
        for kind in ('nested', 'samesite'):
            p = os.path.join(PATHS.pairs, '%s_%s.npz' % (a, kind))
            if os.path.exists(p):
                q = np.load(p, allow_pickle=False)['idx']
                if q.shape[0]:
                    idxs.append(q)
        if not idxs or gmd <= 0:
            continue
        idx = np.vstack(idxs)
        d = y[idx[:, 1]] - y[idx[:, 0]]
        rows.append(dict(DMS_id=a, SI=float(np.abs(d).mean() / gmd),
                         V1_over_Vinf=float(0.5 * np.mean(d * d) / vinf),
                         V_monotone_h1_h4='', gamma1='', gamma1_lo95='',
                         gamma1_hi95='', pos_rs='', n_h1=int(idx.shape[0]),
                         GMD=gmd, V_inf=vinf))
    return pd.DataFrame(rows)


#: One synthetic T04/T06/T07/T09/T12 row set per mode.  These are HAND-CHOSEN to
#: sit unambiguously on one side of every THRESH boundary, so the self-check
#: proves the *rule* is wired correctly.  They are never written to artifacts/.
_FIXTURE_MODES = ('sup', 'ref', 'incon')


def _fixture(mode_by_assay, artefact=(), g7=None, underpowered=(), halt=False):
    """Build in-memory Sec.6 tables from a ``{dms_id: 'sup'|'ref'|'incon'}`` map.

    Every value is expressed as an OFFSET from the ``THRESH`` boundary it is meant
    to straddle, never as a literal: the fixture then stays correct if a threshold
    is ever re-pinned, and no decision boundary is re-declared in this file.
    """
    import pandas as pd
    T = THRESH
    bp = 0.2                     # a stand-in N2 99.5th percentile for beta_a
    t04, t06, t07, t09, t12 = [], [], [], [], []
    for a, m in sorted(mode_by_assay.items()):
        # ---------------------------- T04 / C1 ---------------------------- #
        if m == 'sup':
            t04.append(dict(DMS_id=a,
                            SI=T['C1_SI_sup'] - 0.20,
                            V1_over_Vinf=T['C1_V1_over_Vinf_sup'] - 0.15,
                            V_monotone_h1_h4=True,
                            gamma1=T['C1_gamma1_sup'] + 0.10,
                            gamma1_lo95=T['C1_gamma1_ci_lo_sup'] + 0.10,
                            gamma1_hi95=T['C1_gamma1_sup'] + 0.25,
                            pos_rs=T['C1_pos_rs_ref'] - 0.45))
        elif m == 'ref':
            t04.append(dict(DMS_id=a,
                            SI=T['C1_SI_ref'] + 0.15,
                            V1_over_Vinf=T['C1_V1_over_Vinf_ref'] + 0.20,
                            V_monotone_h1_h4=False,
                            gamma1=T['C1_gamma1_ref'] - 0.10,
                            gamma1_lo95=T['C1_gamma1_ref'] - 0.18,
                            gamma1_hi95=T['C1_gamma1_ci_hi_ref'] - 0.25,
                            pos_rs=T['C1_pos_rs_ref'] + 0.15))
        else:
            mid_si = 0.5 * (T['C1_SI_sup'] + T['C1_SI_ref'])
            mid_g = 0.5 * (T['C1_gamma1_ref'] + T['C1_gamma1_sup'])
            t04.append(dict(DMS_id=a, SI=mid_si,
                            V1_over_Vinf=0.5 * (T['C1_V1_over_Vinf_sup']
                                                + T['C1_V1_over_Vinf_ref']),
                            V_monotone_h1_h4=True, gamma1=mid_g,
                            gamma1_lo95=mid_g - 0.10, gamma1_hi95=mid_g + 0.10,
                            pos_rs=T['C1_pos_rs_ref'] - 0.30))
        # ---------------------------- T06 / C2 ---------------------------- #
        p95 = T['C2_TR1_gauss']
        p995 = p95 + 0.14
        head = dict(DMS_id=a, scale='latent',
                    n_Pa=int(T['C2_TR1_min_Pa'] * 2.5), frac_c_exact_zero=0.0,
                    TR_N1_p95=p95, TR_N1_p995=p995)
        if m == 'sup':
            head.update(TR=p995 + 1.0,
                        pi_hat=0.5 * (T['C2_pi_lo'] + T['C2_pi_hi']),
                        pi_lo95=T['C2_pi_ci_lo_sup'] * 2,
                        pi_hi95=T['C2_pi_hi'], rho_hat=T['C2_rho_sup'] + 1.0,
                        dBIC=T['C2_dBIC_sup'] - 40.0)
            tq = dict(T_N2=T['C2_T_sup'] + 1.0, T_N2_lo95=T['C2_T_sup'] + 0.5,
                      T_N2_hi95=T['C2_T_sup'] + 1.5, q_BH=T['C2_q_BH_sup'] / 5)
        elif m == 'ref':
            head.update(TR=p95 - 0.9, pi_hat=T['C2_pi_lo'] / 5,
                        pi_lo95=T['C2_pi_ci_lo_sup'] / 5,
                        pi_hi95=T['C2_pi_ci_hi_ref'] / 2,
                        rho_hat=T['C2_rho_sup'] - 1.8,
                        dBIC=T['C2_dBIC_sup'] + 8.0)
            tq = dict(T_N2=T['C2_T_ref'] - 0.4, T_N2_lo95=T['C2_T_ref'] - 0.6,
                      T_N2_hi95=T['C2_T_ref_ci_hi'] - 0.7, q_BH=0.6)
        else:
            head.update(TR=p95 + 0.05, pi_hat=T['C2_pi_hi'] * 2,
                        pi_lo95=T['C2_pi_hi'], pi_hi95=T['C2_pi_hi'] * 3,
                        rho_hat=T['C2_rho_sup'] + 1.0,
                        dBIC=T['C2_dBIC_sup'] - 10.0)
            tq = dict(T_N2=0.5 * (T['C2_T_ref'] + T['C2_T_sup']),
                      T_N2_lo95=T['C2_T_ref'] + 0.1,
                      T_N2_hi95=T['C2_T_sup'] + 0.4, q_BH=T['C2_q_BH_sup'] / 2)
        for unit in ('sigma', 'MAD'):
            for tau in config.TAUS:
                r = dict(head)
                r.update(unit=unit, tau=tau, tau_absolute=tau * 0.1,
                         grid_guard_pass=True, n_consecutive_tau_passing='', **tq)
                t06.append(r)
        # ------------------------- T07 / C3-L, C3-A ----------------------- #
        c3a = dict(depth_spearman=T['C3A_depth_spearman_ref'] - 0.30,
                   best_struct_covariate=T['C3A_struct_cov_min'] + 0.10,
                   density_q1_rate=0.02, density_q5_rate=0.03,
                   density_monotone=False, floor_mask_invariant=True,
                   latent_raw_consistent=True)
        if a in artefact:
            c3a.update(density_monotone=True)
        if m == 'sup':
            t07.append(dict(DMS_id=a, route='L1', feasible=True, n_units=5000,
                            beta_sibling=bp * 2.5, se_hc3=0.05,
                            beta_N2_p995=bp, beta_in_N2_band=False, **c3a))
            t07.append(dict(DMS_id=a, route='L5', feasible=True, n_units=900,
                            AUROC_L5=T['C3L_AUROC_sup'] + 0.10,
                            p_NS2=T['C3L_p_NS2_sup'] / 10, **c3a))
        elif m == 'ref':
            t07.append(dict(DMS_id=a, route='L1', feasible=True, n_units=5000,
                            beta_sibling=bp / 20, se_hc3=0.05,
                            beta_N2_p995=bp, beta_in_N2_band=True, **c3a))
            t07.append(dict(DMS_id=a, route='L2', feasible=True, n_units=400,
                            ICC=T['C3L_ICC_sup'] / 6, ICC_lo95=0.0,
                            ICC_hi95=T['C3L_ICC_ci_hi_ref'] - 0.05, **c3a))
            t07.append(dict(DMS_id=a, route='L4', feasible=True, n_units=1300,
                            dR2_oos=T['C3L_dR2_sup'] / 20, dR2_lo95=-0.001,
                            dR2_hi95=T['C3L_dR2_ci_hi_ref'] - 0.01, **c3a))
        else:
            t07.append(dict(DMS_id=a, route='L1', feasible=True, n_units=5000,
                            beta_sibling=bp + 0.01, se_hc3=0.05,
                            beta_N2_p995=bp, beta_in_N2_band=False, **c3a))
            t07.append(dict(DMS_id=a, route='L2', feasible=True, n_units=400,
                            ICC=T['C3L_ICC_sup'] - 0.10,
                            ICC_lo95=T['C3L_ICC_ci_lo_sup'] - 0.05,
                            ICC_hi95=T['C3L_ICC_sup'], **c3a))
        # ---------------------------- T09 / C4-S -------------------------- #
        s9 = dict(DMS_id=a, chain='A', resseq=1, icode='', assay_permissible=True)
        if m == 'sup':
            s9.update(OR_burial_matched=T['C4S_OR_sup'] + 0.5,
                      OR_lo95=T['C4S_OR_sup'], OR_hi95=T['C4S_OR_sup'] + 1.2,
                      p_NS1=T['C4S_p_NS1_sup'] / 10,
                      p_wald_after_rsa=T['C4S_p_NS1_sup'] / 10,
                      beta_iface_after_rsa=0.8)
        elif m == 'ref':
            s9.update(OR_burial_matched=T['C4S_OR_ref'] - 0.2,
                      OR_lo95=T['C4S_OR_ref'] - 0.4,
                      OR_hi95=T['C4S_OR_ref'] + 0.05, p_NS1=0.4,
                      p_wald_after_rsa=0.5, beta_iface_after_rsa=0.0)
        else:
            s9.update(OR_burial_matched=0.5 * (T['C4S_OR_ref'] + T['C4S_OR_sup']),
                      OR_lo95=T['C4S_OR_ref'] - 0.1,
                      OR_hi95=T['C4S_OR_sup'] + 0.4,
                      p_NS1=T['C4S_p_NS1_sup'] * 5,
                      p_wald_after_rsa=T['C4S_p_NS1_sup'] / 2,
                      beta_iface_after_rsa=0.3)
        t09.append(s9)
        # ----------------------------- T12 / C5 --------------------------- #
        for mo in C5_MODELS:
            if m == 'sup':
                psa = T['C5_PSA_blindspot'] - 0.05
            elif mo == C5_MODELS[0]:
                psa = T['C5_PSA_practically_empty'] + 0.05
            else:
                psa = T['C5_PSA_blindspot'] + 0.10
            t12.append(dict(DMS_id=a, model=mo, tau=4, AUPSA=psa, PSA_cliff=psa,
                            PSA_lo95=psa - 0.05, PSA_hi95=psa + 0.05,
                            spearman_all_rows=T['C5_spearman_min'] + 0.15))
    # ------------------------------ T08 / C3-N ---------------------------- #
    t08 = [dict(assay_a='KRAS_RAF1_norfitness_6VJJ',
                assay_b='KRAS_RAF1-RBD_norfitness_6VJJ',
                relation='same_interaction_diff_study', join_method='canonical_key',
                n_shared=config.NOISE['KRAS']['n_source'],
                R=THRESH['C3N_R_sup'] + 0.10,
                R_chance_perm=THRESH['C3N_perm_chance_max'] / 2, perm_p=0.001,
                sign_agreement=THRESH['C3N_sign_agreement_sup'] + 0.05)]
    # ------------------------------ T11 / C4-I ---------------------------- #
    t11 = [dict(family='F2', chain='A', resseq=12,
                F_spec_noise_corrected=THRESH['C4I_Fspec_sup'] + 0.15,
                family_p_NS3=THRESH['C4I_p_NS3_sup'] / 5,
                MW_PSI_p=THRESH['C4I_MW_p_sup'] / 5,
                median_cliff_PSI=THRESH['C4I_median_PSI_ref'] - 0.30,
                foldaxis_spearman_rowmean_rsa=0.3)]
    t01 = pd.DataFrame([dict(DMS_id=a,
                             n_nested=THRESH['min_nested_for_pair_channel'] * 2,
                             n_primary_Pa=int(THRESH['C2_TR1_min_Pa'] * 2.5),
                             underpowered_G8=(a in underpowered))
                        for a in config.ALL_ASSAYS])
    g2 = [dict(gate_id='G1', gate_name='parse audit', assay='ALL', statistic='rows',
               expected=1, observed=1, tolerance=0,
               **{'PASS/FAIL': 'PASS'}, consequence_if_fail='STOP',
               halts_study='YES')]
    if g7 is not None:
        g2.append(dict(gate_id='G7', gate_name='scale-mixture discrimination',
                       assay='PRIMARY+ARM', statistic='TR and T(tau) on N2c',
                       expected='sets the verdict rule', observed=g7, tolerance='',
                       **{'PASS/FAIL': ''}, consequence_if_fail='C2 AND C3-L',
                       halts_study='YES'))
    if halt:
        g2.append(dict(gate_id='G5', gate_name='censoring positive control',
                       assay='CR9114_FluAH3_logKd_4FQY', statistic='T(4) unmasked',
                       expected='>=%g' % THRESH['G5_unmasked_T4_min'],
                       observed='1.0', tolerance='', **{'PASS/FAIL': 'FAIL'},
                       consequence_if_fail='STOP', halts_study='YES'))
    out = {k: (None, 'table_missing') for k in TABLES}
    for k, v in (('T01', t01), ('T02', pd.DataFrame(g2)), ('T04', pd.DataFrame(t04)),
                 ('T06', pd.DataFrame(t06)), ('T07', pd.DataFrame(t07)),
                 ('T08', pd.DataFrame(t08)), ('T09', pd.DataFrame(t09)),
                 ('T11', pd.DataFrame(t11)), ('T12', pd.DataFrame(t12))):
        out[k] = (v.astype(str) if len(v) else v, 'ok')
    return out


# --------------------------------------------------------------------------- #
# self-check -- runs on whatever tables exist right now                        #
# --------------------------------------------------------------------------- #

def check_predeclared_C1_refutations(t04=None, status=None, verbose=True):
    """Spec Sec.1.2: "Pre-declared refutations (must reproduce, else the
    implementation is wrong)"."""
    if t04 is None:
        t04, status = load_table('T04')
    out = []
    for dms_id, si_spec in sorted(config.EXPECTED['C1_predeclared_refutations'].items()):
        d = verdict_C1(dms_id, t04, status or 'table_missing')
        si = d.detail.get('SI')
        out.append(dict(DMS_id=dms_id, SI_spec=si_spec, SI_obs=si,
                        verdict=d.outcome, reproduces=(d.outcome == REFUTED),
                        failing_criterion=d.failing_criterion))
        if verbose:
            print('  %-42s SI_spec=%.3f SI_obs=%-7s verdict=%-12s %s'
                  % (dms_id, si_spec, _fmt(si), d.outcome, d.failing_criterion[:44]))
    return out


def _selfcheck():
    import pandas as pd
    config.assert_env()
    print('=' * 100)
    print('cliff.verdict self-check -- real artifacts/, real THRESH, no invented data')
    print('=' * 100)
    st = table_status()
    print(st.to_string(index=False))
    n_ok = int((st['status'] == 'ok').sum())
    print('\n[verdict] %d/%d Sec.6 tables present' % (n_ok, len(st)))

    tables = load_all_tables()
    gates = read_gates(tables['T02'][0])
    print('[verdict] halting gate failures : %s'
          % (list(gates.halts) or 'none -- the study is not halted'))
    print('[verdict] G7 rule switch        : tail_inflatable=%s  '
          'localisation_inflated=%s  (%s)'
          % (gates.g7['tail_inflatable'], gates.g7['localisation_inflated'],
             gates.g7['source']))
    print('[verdict] G8 UNDERPOWERED       : %s'
          % (gates.underpowered or 'none stamped yet'))
    print('[verdict] G9 rule FPR / k       : fpr=%s k=%s (%s)'
          % (gates.g9['fpr'], gates.g9['k'] or '{} -> config k', gates.g9['source']))
    print('[verdict] G-UP / G-OPT / G11    : %s / %s / %s'
          % (gates.gup_obtained, gates.gopt_ok, gates.g11_dual))

    pa = per_assay_verdicts(tables)
    print('\n[verdict] per-assay verdicts (%d assays x %d columns)'
          % (len(pa), len(pa.columns)))
    show = ['DMS_id', 'tier', 'family_id', 'contributes', 'verdict_C1',
            'verdict_C2', 'verdict_C3L', 'verdict_C3A', 'verdict_C4S', 'verdict_C5']
    with pd.option_context('display.width', 200, 'display.max_colwidth', 24):
        print(pa[show].to_string(index=False))
    print('\n[verdict] failing_criterion, one assay per distinct reason:')
    seen = set()
    for _, r in pa.iterrows():
        for crit in ('C1', 'C2', 'C3L', 'C3A', 'C4S', 'C5'):
            why = r['verdict_%s_failing_criterion' % crit]
            if why and why not in seen:
                seen.add(why)
                print('  %-4s %-40s %s' % (crit, r['DMS_id'], why))
    assert set(pa['verdict_C1']) <= set(OUTCOMES), 'C1 emitted a non-outcome'
    assert set(pa['verdict_C2']) <= set(OUTCOMES), 'C2 emitted a non-outcome'
    print('\n[verdict] every emitted verdict is one of %s  OK' % (OUTCOMES,))

    print('\n[verdict] Sec.1.2 pre-declared C1 refutations (must reproduce):')
    check_predeclared_C1_refutations(*tables['T04'])

    t14 = build_T14(pa, tables, verbose=True)
    print('\n[verdict] T14 (%d rows x %d cols); columns match the spec: %s'
          % (len(t14), len(t14.columns), list(t14.columns) == T14_COLUMNS))
    assert list(t14.columns) == T14_COLUMNS
    with pd.option_context('display.width', 250, 'display.max_colwidth', 18):
        print(t14[['family_id', 'n_eligible', 'C1_pos/C1_neg', 'C2_pos/C2_neg',
                   'C3L_pos/C3L_neg', 'C3N_result', 'C4I_result',
                   'family_verdict_C1', 'family_verdict_C2', 'family_verdict_C3',
                   'all_three']].to_string(index=False))
    print('\n[verdict] footer rows:')
    for _, r in t14[t14['family_id'].str.startswith('FOOTER')].iterrows():
        print('  %-28s C1=%-9s C2=%-9s C3=%-9s meta=%-10s' % (
            r['family_id'], r['C1_pos/C1_neg'], r['C2_pos/C2_neg'],
            r['C3L_pos/C3L_neg'], r['meta_effect']))
        print('      %s' % str(r['notes'])[:300])
    print('\n[verdict] aggregate call: %s' % (t14.attrs['aggregate'],))
    print('[verdict] binomial p of the counts: %s' % t14.attrs['binomial_p'])

    # the one closed-form number the spec pins: binom p at 5/7
    from scipy import stats
    p57 = float(stats.binom.sf(4, 7, 0.5))
    print('[verdict] binomial P(X >= 5 | Binom(7, 0.5)) = %.6f  (spec 0.2266, '
          'config.BINOM_P_5OF7 = %.4f)  %s'
          % (p57, config.BINOM_P_5OF7,
             'MATCH' if abs(p57 - config.BINOM_P_5OF7) < 5e-5 else 'MISMATCH'))
    assert abs(p57 - config.BINOM_P_5OF7) < 5e-5

    # meta-analysis closed-form check against a hand-computable case
    m = meta_random_effects([1.0, 1.0, 1.0], [0.5, 0.5, 0.5])
    print('[verdict] meta_random_effects([1,1,1],[.5,.5,.5]) -> effect=%.6f '
          'tau2=%.6f Q=%.6f  (homogeneous: effect must be 1, tau2 and Q must be 0)'
          % (m['effect'], m['tau2'], m['Q']))
    assert abs(m['effect'] - 1.0) < 1e-12 and m['tau2'] == 0.0 and m['Q'] < 1e-12
    m2 = meta_random_effects([0.0, 2.0], [0.1, 0.1])
    print('[verdict] meta_random_effects([0,2],[.1,.1]) -> effect=%.6f tau2=%.6f '
          'I2=%.2f%% k=%d  (maximally heterogeneous: effect 1, large tau2, I2->100)'
          % (m2['effect'], m2['tau2'], m2['I2'], m2['k']))
    assert abs(m2['effect'] - 1.0) < 1e-9 and m2['tau2'] > 0.9 and m2['I2'] > 99

    # decide() truth table
    assert decide([_sup('a', True)]).outcome == SUPPORTED
    assert decide([_sup('a', False)]).outcome == INCONCLUSIVE
    assert decide([_sup('a', None)]).outcome == INCONCLUSIVE
    assert decide([_sup('a', True), _ref('b', True)]).outcome == REFUTED
    assert decide([_sup('a', True), _ref('b', True)]
                  ).failing_criterion.startswith('CONFLICT:')
    assert decide([_sup('a', False), _ref('b', True)]).outcome == REFUTED
    print('[verdict] decide() truth table: SUPPORTED/REFUTED/INCONCLUSIVE + the '
          'CONFLICT case  OK')

    # G7 both-ways resolution
    g7_unknown = dict(tail_inflatable=None, localisation_inflated=None,
                      source='test', per_assay={})
    d = verdict_C2('GB1_IgG-Fc_fitness_1FCC', tables['T06'][0], tables['T06'][1],
                   c3l_ok=None, g7=g7_unknown)
    print('[verdict] G7-undetermined C2 on a missing T06 -> %s (%s)'
          % (d.outcome, d.failing_criterion))
    assert d.outcome == INCONCLUSIVE

    # ------------------------------------------------------------------ #
    # PART B -- the C1 rule on REAL numbers computed from the pair cache  #
    # ------------------------------------------------------------------ #
    print('\n' + '=' * 100)
    print('PART B -- C1 on REAL data: SI and V(1)/V(inf) from the cached pair')
    print('  indices via the spec Sec.1.2 closed forms (self-check fixture only;')
    print('  variogram.py owns T04).  h = 1 is nested UNION same-site.')
    print('=' * 100)
    m = measured_C1_inputs()
    print('%d assays measured from data/cliff_cache/{keys,pairs}' % len(m))
    pre = config.EXPECTED['C1_predeclared_refutations']
    hdr = ('%-38s %8s %8s %6s %9s %-13s %s'
           % ('DMS_id', 'SI_spec', 'SI_obs', 'd', 'V1/Vinf', 'verdict_C1',
              'failing_criterion'))
    print(hdr)
    n_rep, n_chk = 0, 0
    for _, r in m.sort_values('SI').iterrows():
        a = r['DMS_id']
        d = verdict_C1(a, m.astype(str), 'ok')
        spec_si = pre.get(a)
        tag = ''
        if spec_si is not None:
            n_chk += 1
            if d.outcome == REFUTED:
                n_rep += 1
            tag = '%8.3f %6.4f' % (spec_si, abs(spec_si - r['SI']))
        else:
            tag = '%8s %6s' % ('-', '-')
        print('%-38s %s %8.4f %9.4f %-13s %s'
              % (a[:38], tag, r['SI'], r['V1_over_Vinf'], d.outcome,
                 d.failing_criterion[:52]))
    print('\n[verdict] Sec.1.2 pre-declared refutations reproduced as REFUTED: '
          '%d/%d' % (n_rep, n_chk))
    assert n_rep == n_chk == len(pre), 'a pre-declared C1 refutation did not reproduce'
    gb1 = verdict_C1('GB1_IgG-Fc_fitness_1FCC', m.astype(str), 'ok')
    print('[verdict] GB1_IgG-Fc_fitness_1FCC (the flagship SMOOTH landscape) -> %s '
          '(%s)' % (gb1.outcome, gb1.failing_criterion))
    assert gb1.outcome != REFUTED, 'the flagship smooth assay was refuted'

    # ------------------------------------------------------------------ #
    # PART C -- the rule wiring, on hand-chosen fixtures                  #
    # ------------------------------------------------------------------ #
    print('\n' + '=' * 100)
    print('PART C -- rule wiring on hand-chosen fixtures (NOT data; never written)')
    print('=' * 100)
    fam1 = {f: config.FAMILIES[f] for f in config.FAMILIES if f != 'F8'}
    pos = {}
    for f, mem in fam1.items():
        for a in mem:
            pos[a] = 'sup' if f not in ('F6', 'F7') else 'ref'

    def _scen(name, tabs, expect=None):
        p = per_assay_verdicts(tabs)
        t = build_T14(p, tabs, verbose=False)
        agg = t.attrs['aggregate']
        row = t[t['family_id'] == 'FOOTER:k_of_%d' % config.K_FAMILIES].iloc[0]
        print('\n--- %s' % name)
        print('    counts   C1 %s   C2 %s   C3L %s'
              % (row['C1_pos/C1_neg'], row['C2_pos/C2_neg'], row['C3L_pos/C3L_neg']))
        for c in ('C1', 'C2', 'C3'):
            print('    %-3s %-13s %s' % (c, agg[c][0], agg[c][1][:110]))
        print('    HEADLINE %s' % t.attrs['headline'][:150])
        if expect:
            for c, e in expect.items():
                assert agg[c][0] == e, '%s: %s = %s, expected %s' % (
                    name, c, agg[c][0], e)
        return p, t, agg

    _scen('POSITIVE: F1-F5 supported, F6-F7 refuted, G7 not_inflated',
          _fixture(pos, g7='not_inflated'),
          dict(C1=TRUE, C2=TRUE, C3=TRUE))
    _scen('NEGATIVE: every family refuted',
          _fixture({a: 'ref' for a in pos}, g7='not_inflated'),
          dict(C1=REFUTED, C2=REFUTED, C3=REFUTED))
    _scen('VACUOUS GUARD: only F1 evaluated, the rest of T04/T06/T07 absent',
          _fixture({a: 'ref' for a in config.FAMILIES['F1']}, g7='not_inflated'),
          dict(C2=INCONCLUSIVE, C3=INCONCLUSIVE))

    # G7 rule switch: C2-supported assays whose C3-L is REFUTED
    mixed = dict(pos)
    for f in ('F1', 'F2', 'F3'):
        for a in config.FAMILIES[f]:
            mixed[a] = 'sup'
    t_no = _fixture(mixed, g7='not_inflated')
    t_yes = _fixture(mixed, g7='inflated')
    for k in ('T07',):
        df = t_yes[k][0].copy()
        df.loc[df['DMS_id'].isin(config.FAMILIES['F1'] + config.FAMILIES['F2']),
               'beta_sibling'] = '0.01'
        df.loc[df['DMS_id'].isin(config.FAMILIES['F1'] + config.FAMILIES['F2']),
               'beta_in_N2_band'] = 'True'
        t_yes[k] = (df, 'ok')
        t_no[k] = (df.copy(), 'ok')
    _, _, a_no = _scen('G7 = not_inflated  (C2 alone admissible; C3-L broken on '
                       'F1+F2)', t_no)
    _, _, a_yes = _scen('G7 = inflated      (C2 AND C3-L mandatory; same tables)',
                        t_yes)
    print('\n[verdict] the G7 switch alone moves C2 from %s to %s -- the rule is '
          'gate-driven, not hardcoded' % (a_no['C2'][0], a_yes['C2'][0]))
    assert a_no['C2'][0] != a_yes['C2'][0], 'the G7 switch changed nothing'

    _scen('UNDERPOWERED: G8 stamps every F3/F4/F5 assay',
          _fixture(pos, g7='not_inflated',
                   underpowered=tuple(config.FAMILIES['F3'] + config.FAMILIES['F4']
                                      + config.FAMILIES['F5'])),
          dict(C1=INCONCLUSIVE))
    _scen('HALTING GATE: G5 FAILs with halts_study = YES',
          _fixture(pos, g7='not_inflated', halt=True),
          dict(C1=INCONCLUSIVE, C2=INCONCLUSIVE, C3=INCONCLUSIVE))
    _scen('C3-A DIRTY: density monotone on two supported assays',
          _fixture(pos, g7='not_inflated',
                   artefact=tuple(config.FAMILIES['F1'])),
          dict(C3=INCONCLUSIVE))

    print('\n' + '=' * 100)
    print('PART D -- write the real T14 from the real artifacts/')
    print('=' * 100)
    res = run(write=True, write_back=True, verbose=False)
    for k in ('T14', 'T14a'):
        if k in res['paths']:
            print('\n[verdict] wrote %s' % res['paths'][k])
    print('[verdict] wrote back into: %s'
          % (res['paths'].get('write_back') or 'nothing (no stats table exists yet)'))
    print('[verdict] HEADLINE: %s' % res['headline'][:600])
    print('\n[verdict] SELF-CHECK PASSED')


if __name__ == '__main__':
    _selfcheck()
