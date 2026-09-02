"""BGYM-CLIFF v1 -- the seven publication figures F1..F7.  Spec Sec.6.

Every number drawn here is read from a table in ``local-records/bindingGYM-cliff/
artifacts/``.  Nothing is smoothed, interpolated, extrapolated or invented; no
example data exists in this module, not even for layout testing.  Style comes
from :mod:`cliff.figstyle` (rcParams, the two permitted widths, the fixed
family -> colour map, the bold 9 pt panel letters) and is never overridden here.

THREE RULES THIS MODULE ENFORCES ON ITSELF
------------------------------------------
1. A shaded region is either a REAL null envelope drawn through
   ``figstyle.null_band`` (5th-95th percentile, from the table) or a frozen
   DECISION region drawn through :func:`decision_region` (hatched, always
   labelled "decision region, not a null").  Observed-data uncertainty is drawn
   as ERROR BARS, never as a fill, precisely so that a filled region in F1..F7
   always means "null" or "decision" and never "the observed line, thickened".
2. A panel whose input table (or whose input *column*) is absent is stamped by
   :func:`stamp` with a visible grey "table not available: T0x" box naming the
   table AND the column, and what the panel would have shown.  It is never left
   empty and never silently substituted.
3. Decision lines come from ``config.THRESH`` by key.  There is no numeric
   literal in this module that is also a threshold in the spec.

WHAT IS AND IS NOT DERIVED
--------------------------
The only arithmetic performed on tabulated numbers is:
  * ``V_h_se / V_inf`` -- putting a tabulated SE on the tabulated ratio's scale,
    where ``V_inf`` is T05's own ``h == 'random'`` row (exact, by construction:
    ``V_h / V_inf == V_h_over_Vinf`` for every row was checked, not assumed);
  * ``T = rate_obs / rate_null`` -- the spec's own definition of T(tau), taken
    from T02f's already-tabulated ``*_T_N1`` column wherever it exists;
  * ``n_nested - n_nested_censor_touching`` for the G5 masking panel, which
    T01 also tabulates directly as ``n_primary_Pa`` (both are plotted, and they
    agree to the unit on every censored assay -- the panel says so).
Nothing else.  No fit, no kernel, no rolling mean.

DECISIONS TAKEN WHERE SPEC Sec.6 WAS SILENT OR THE TABLES DISAGREED WITH IT
---------------------------------------------------------------------------
* F1 "the peak": Sec.6 predates the finding that ``V(h)`` rises and then FALLS.
  The peak marked is the FIRST INTERIOR LOCAL MAXIMUM of ``V(h)/V(inf)`` that is
  resolved beyond the tabulated SEs (rise and fall each > 2 x the pooled SE of
  the two neighbouring h).  This rule has no free parameter, needs no
  ``N_h`` cutoff, and reproduces every peak in the study log (SARS2-RBD h=7,
  hYAP65 h=5, KRAS_RAF1 / KRAS_SOS1 h=3).  Where the curve is still rising at
  ``h_max`` no peak is marked, because no fall has been observed.  The x axis is
  never truncated: hYAP65 runs to h=28 and SARS2-RBD to h=18 so that the decay
  to the absorbing floor is visible.
* F1 N1/N2 ribbons: ``T05.V_h_N1_lo / V_h_N1_hi / V_h_N2_mean`` are EMPTY in
  this run (0/373 rows finite).  Per rule 1 no band is drawn and the legend
  cell says so.  Rule 2's stamp is not used because the observed curve -- the
  panel's actual subject -- is live.
* F1 decision-line annotation: 0.35 / 0.70 are annotated in panel (a) and in
  the legend cell rather than in all 15 panels, which at 45 mm per panel would
  collide with the curve.
* F2: NO table tabulates ``P(|c_hat| >= tau)`` over the whole tau grid for the
  observed data AND all four nulls.  T06 carries one row per (assay, scale,
  unit) at the SELECTED tau; T02f carries tau = 3 and 4, sigma units only.  So
  F2 is drawn as "observed against the four nulls" dot plots over the assay
  axis at the taus that exist, plus the two tail-shape statistics that carry
  all four nulls (TR1 and excess kurtosis of e) -- excess kurtosis being the
  N2c claim in its most direct form.  The mixture panels (e)/(f), which
  Sec.6 asks for as per-assay insets, are promoted to full panels: a per-assay
  inset cannot be placed inside a dot plot, and pi_hat with its CI plus the
  ``c_hat = 0`` spike mass is exactly the inset's content.
* F3 MAD units: no table carries the OBSERVED MAD-unit rate (T02a carries the
  N1 reference mean for ``rate_mad_tau*``, not the observation), so the
  right-hand panel is stamped.  The left panel plots T(tau) against N1
  (``T02f.rate_sigma_tau*_T_N1``), which is what exists; the C2 verdict is read
  off T_N2, so the axis label names the null explicitly.
* F6(d): Sec.6 asks for "the QQ of the 200 null p-values".  The 200
  leave-one-out empirical p-values behind each G4 KS test are not tabulated;
  T02a tabulates the KS p-value per (assay, statistic).  A p-value is uniform
  under the null, so the panel is the QQ of those -- for BOTH the randomised
  form that G4 is read off and the tie-inflated conservative form, which is the
  honest way to show why the randomised form is used.
* F7 tau: Sec.6 does not name one and T12 carries six.  tau = 3 is used
  (``TAU_WINDOW`` lower edge, and the C3-N cliff definition), stated on the axis.
* F7 "AUPSA on cliff vs non-cliff edges": T12's AUPSA is ONE number per
  (assay, model) -- it does not split by cliff status (and is constant in tau).
  The cliff / non-cliff split in T12 is ``PSA_cliff`` vs ``PSA_noncliff``, so
  that is what the bars show, with AUPSA marked as a separate reference glyph.

Entry points: :func:`stage8` (= :func:`run_all` = :func:`run`), dispatched by
``cliff/run_all.py`` stage 8; ``python -m cliff.figures`` runs the self-check.
"""
from __future__ import annotations

import os
import string
import textwrap
import traceback

import numpy as np
import pandas as pd

from cliff import config, figstyle
from cliff.config import PATHS, THRESH

figstyle.apply()                      # installs rcParams + selects Agg

# A decision region is hatched (see decision_region); at matplotlib's default
# 1.0 pt / black the hatch swamps the data it sits behind, so it is thinned
# here.  These two are NOT among spec Sec.6's sixteen rcParams -- they are
# hatch cosmetics, and figstyle.RC_PARAMS stays checkable against Sec.6.
import matplotlib as _mpl                                        # noqa: E402
_mpl.rcParams['hatch.linewidth'] = 0.35
_mpl.rcParams['hatch.color'] = '#C9C9C9'

import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402
from matplotlib.patches import Patch, Rectangle                   # noqa: E402
from matplotlib.ticker import MaxNLocator                          # noqa: E402

# --------------------------------------------------------------------------- #
# the tables -- spec Sec.6 "Table columns (exact)"                            #
# --------------------------------------------------------------------------- #

#: name -> filename, exactly the Sec.6 artifacts listing.
TABLES = {
    'T01': 'T01_assay_manifest.csv',
    'T02': 'T02_gates.csv',
    'T02a': 'T02a_G4_selfcal.csv',
    'T02b': 'T02b_N2_power.csv',
    'T02c': 'T02c_null_ensemble_cost.csv',
    'T02d': 'T02d_link_clamp_audit.csv',
    'T02e': 'T02e_N2c_kurtosis_room.csv',
    'T02f': 'T02f_obs_vs_nulls.csv',
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
    'T14': 'T14_verdict_by_family.csv',
    'T14a': 'T14a_verdict_by_assay.csv',
    'T15': 'T15_cluster_channel.csv',
    # written by stage 4 / stage 6 beside the Sec.6 list; they carry statistics
    # Sec.6's figures ask for that no Sec.6 table has a column for
    'T02g': 'T02g_G5_censoring.csv',
    'T02gb': 'T02g_G5_band_N2.csv',
    'T02h': 'T02h_G6_antismooth.csv',
    'T02j': 'T02j_G8_power_raw.csv',
    'T02m': 'T02m_T_N2_structural.csv',
}

_CACHE = {}


def artifacts_dir(outdir=None):
    return PATHS.artifacts if outdir is None else outdir


def load(name, outdir=None, verbose=False, usecols=None):
    """Read one artifact table.  Returns ``None`` when it does not exist yet --
    never raises, so a figure can degrade instead of taking the stage down.

    ``usecols`` is for T10, which is ~180k rows x 25 columns: the figures need
    six of them, and this box runs three other jobs."""
    key = (name, artifacts_dir(outdir), tuple(usecols) if usecols else None)
    if key in _CACHE:
        return _CACHE[key]
    path = os.path.join(artifacts_dir(outdir), TABLES[name])
    if not os.path.exists(path) and os.path.exists(path + '.gz'):
        path += '.gz'                      # a big table may be gzipped in place
    df = None
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, usecols=list(usecols) if usecols else None)
        except Exception as exc:                                   # pragma: no cover
            print('[figures] %s unreadable (%s) -- treated as absent' % (name, exc))
            df = None
    if verbose:
        print('[figures] %-5s %s' % (name, ('%d x %d' % df.shape) if df is not None
                                     else 'ABSENT'))
    _CACHE[key] = df
    return df


def have(df, *cols):
    """True when ``df`` exists and every named column has at least one finite
    value.  A column of NaN is as absent as a missing file -- and is the case
    that silently produces an empty panel if it is not checked."""
    if df is None:
        return False
    for c in cols:
        if c not in df.columns:
            return False
        if not df[c].notna().any():
            return False
    return True


def n_finite(df, col):
    return 0 if df is None or col not in df.columns else int(df[col].notna().sum())


# --------------------------------------------------------------------------- #
# labels and per-assay styling (colour = family, from figstyle; marker and     #
# linestyle vary WITHIN a family so the figures survive greyscale printing)    #
# --------------------------------------------------------------------------- #

_LABEL = {
    'GB1_IgG-Fc_fitness_1FCC': 'GB1-IgG-Fc (1FCC)',
    'GB1_IgG-Fc_fitness_1FCC_2016': 'GB1-IgG-Fc (1FCC, 2016)',
    'KRAS_RAF1_norfitness_6VJJ': 'KRAS-RAF1 (6VJJ)',
    'KRAS_RAF1-RBD_norfitness_6VJJ': 'KRAS-RAF1-RBD (6VJJ)',
    'KRAS_RALGDS-RBD_norfitness_1LFD': 'KRAS-RALGDS (1LFD)',
    'KRAS_PICK3CG-RBD_norfitness_1HE8': 'KRAS-PICK3CG (1HE8)',
    'KRAS_SOS1_norfitness_8BE4': 'KRAS-SOS1 (8BE4)',
    'KRAS_DARPinK27_norfitness_5O2S': 'KRAS-DARPinK27 (5O2S)',
    'SARS2-RBD_ACE2_deltaKd_6M0J': 'SARS2-RBD-ACE2 (6M0J)',
    '5A12_VEGF_fitness_4ZFF': '5A12-VEGF (4ZFF)',
    '5A12_Ang2_fitness_4ZFG': '5A12-Ang2 (4ZFG)',
    'Z-domain_ZpA963_HL1_fitness_2M5A': 'Z-ZpA963-HL1 (2M5A)',
    'Z-domain_ZpA963_HL2_fitness_2M5A': 'Z-ZpA963-HL2 (2M5A)',
    'hYAP65_peptide_FunctioncalScore_1JMQ': 'hYAP65 (1JMQ)',
    'CD19_FMC63_Fitness_7URV': 'CD19-FMC63 (7URV)',
    'CR9114_FluAH1_logKd_4FQI': 'CR9114-H1 (4FQI)',
    'CR6261_FluAH1_logKd_3GBN': 'CR6261-H1 (3GBN)',
    'CR9114_FluAH3_logKd_4FQY': 'CR9114-H3 (4FQY)',
    'Z-domain_ZSPA-1_LL1_fitness_1LP1': 'Z-LL1 (1LP1)',
    'Z-domain_ZSPA-1_LL2_fitness_1LP1': 'Z-LL2 (1LP1)',
}

#: within-family marker / linestyle cycles.  Index = position in
#: ``config.FAMILIES[fam]``, so a given assay keeps one glyph in all figures.
_MARKERS = ('o', 's', '^', 'D', 'v', 'P', 'X', '*')
_LINESTYLES = ('-', (0, (4, 1.5)), (0, (1, 1.2)), (0, (5, 1.2, 1, 1.2)),
               (0, (3, 1, 3, 1, 1, 1)), (0, (7, 1.5)), (0, (2, 0.8)), (0, (1, 2)))


def label_of(dms_id):
    if dms_id in _LABEL:
        return _LABEL[dms_id]
    parts = str(dms_id).split('_')
    if len(parts) >= 3:
        return '%s-%s (%s)' % (parts[0], parts[1], parts[-1])
    return str(dms_id)


def short_of(dms_id):
    """Label without the PDB code, for a rotated tick label -- but never
    ambiguous: the two GB1 assays differ only in the parenthetical, so the
    disambiguating token is kept."""
    lab = label_of(dms_id).split(' (')[0]
    if '2016' in str(dms_id):
        lab += '/2016'
    return lab


def _family_index(dms_id):
    fam = config.ASSAYS[dms_id].family_id if dms_id in config.ASSAYS else ''
    members = config.FAMILIES.get(fam, ())
    return fam, (members.index(dms_id) if dms_id in members else 0)


def assay_style(dms_id, for_line=True):
    """Colour from :data:`figstyle.FAMILY_COLOR`; marker and linestyle from the
    assay's position inside its family, so two members of one family are
    distinguishable with the colour removed."""
    if dms_id in config.ASSAYS and config.ASSAYS[dms_id].tier in ('PRIMARY', 'ARM'):
        fam, i = _family_index(dms_id)
        st = dict(color=figstyle.FAMILY_COLOR[fam],
                  markeredgecolor=figstyle.FAMILY_EDGE[fam],
                  marker=_MARKERS[i % len(_MARKERS)])
        if for_line:
            st['linestyle'] = _LINESTYLES[i % len(_LINESTYLES)]
        return st
    st = dict(figstyle.CONTROL_STYLE)
    # CONTROL / EXCLUDED assays all take figstyle's black dashed control style;
    # the marker and dash pattern still vary by position in the tier, or three
    # controls become three identical legend rows (they did)
    tier = config.ASSAYS[dms_id].tier if dms_id in config.ASSAYS else 'CONTROL'
    peers = [a for a in config.ALL_ASSAYS if config.ASSAYS[a].tier == tier]
    i = peers.index(dms_id) if dms_id in peers else 0
    st['marker'] = ('x', '+', '1', '2', '3', '4')[i % 6]
    st['markeredgecolor'] = st['color']
    if for_line:
        st['linestyle'] = ((0, (4, 1.5)), (0, (5, 1.2, 1, 1.2)),
                           (0, (1, 1.2)), (0, (7, 1.5)))[i % 4]
    else:
        st.pop('linestyle', None)
        st.pop('linewidth', None)
    return st


def assay_order(tiers=('PRIMARY',), families=None):
    """Assays grouped by family in F1..F8 order -- the order every figure uses,
    so a reader finds the same assay in the same place in all seven."""
    out = []
    for fam in (families or config.FAMILIES):
        for a in config.FAMILIES[fam]:
            if config.ASSAYS[a].tier in tiers:
                out.append(a)
    for a in config.ALL_ASSAYS:                     # tiers with no family
        if config.ASSAYS[a].tier in tiers and a not in out:
            out.append(a)
    return out


PRIMARY_BY_FAMILY = assay_order(('PRIMARY',))
CONTROL_ROW = ('CR9114_FluAH3_logKd_4FQY',
               'Z-domain_ZSPA-1_LL1_fitness_1LP1',
               'Z-domain_ZSPA-1_LL2_fitness_1LP1')     # spec F1's control row

# --------------------------------------------------------------------------- #
# panel primitives                                                            #
# --------------------------------------------------------------------------- #

STAMP_FACE = '#EFEFEF'
STAMP_EDGE = '#8A8A8A'
STAMP_TEXT = '#2B2B2B'


def stamp(ax, tables, what=None, extra=None, wrap=42):
    """Mark a panel as awaiting an input.  Spec Sec.6 + the study's own rule:
    a panel that cannot be drawn says so, in the axes, naming the table."""
    tabs = [tables] if isinstance(tables, str) else list(tables)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ('left', 'bottom'):
        ax.spines[s].set_visible(False)
    ax.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0, transform=ax.transAxes,
                           facecolor=STAMP_FACE, edgecolor=STAMP_EDGE,
                           linestyle=(0, (3.5, 2.5)), linewidth=0.6, zorder=0,
                           clip_on=False))
    head = 'table not available: %s' % ', '.join(tabs)
    body = [head]
    if what:
        body += ['', '\n'.join(textwrap.wrap('would show: ' + what, wrap))]
    if extra:
        body += ['', '\n'.join(textwrap.wrap(extra, wrap))]
    ax.text(0.5, 0.5, '\n'.join(body), transform=ax.transAxes, ha='center',
            va='center', fontsize=6.0, color=STAMP_TEXT, linespacing=1.45,
            zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return 'awaiting %s' % ', '.join(tabs)


def notes_axes(ax, title, lines, handles=None, fontsize=6.0,
               legend_loc='lower left', legend_bbox=None):
    """A frame-less cell carrying the figure's legend and its caveats.  Every
    figure has one: a stamped panel is only honest if the reader is also told
    what the live panels do and do not include."""
    ax.set_axis_off()
    ax.text(0.0, 1.0, title, transform=ax.transAxes, ha='left', va='top',
            fontsize=7, fontweight='bold')
    # wrap to the CELL, not to an assumed page width: an un-wrapped note line
    # runs off the figure edge (it did, before this was added)
    w_in = ax.figure.get_size_inches()[0] * ax.get_position().width
    ncol = max(24, int(w_in * 72.0 / (0.52 * fontsize)))
    wrapped = []
    for ln in lines:
        wrapped += textwrap.wrap(ln, ncol) or ['']
    ax.text(0.0, 0.93, '\n'.join(wrapped), transform=ax.transAxes, ha='left',
            va='top', fontsize=fontsize, linespacing=1.5, color='#1A1A1A')
    if handles:
        kw = dict(loc=legend_loc, fontsize=6.0, handlelength=1.8, borderpad=0.2,
                  labelspacing=0.35, frameon=False)
        if legend_bbox is not None:
            kw['bbox_to_anchor'] = legend_bbox
        ax.legend(handles=handles, **kw)
    return ax


def footer(fig, text, fontsize=5.4, pad_mm=1.4, handles=None, ncol=5,
           legend_fontsize=6.0):
    """A caveat footer (and, optionally, the figure legend) under the panels.

    ``constrained_layout`` does not reserve space for a ``fig.text`` or a
    ``fig.legend``, so the layout rectangle is shrunk by exactly the height
    they need first -- otherwise both land on top of the bottom row (they did).
    """
    w_in, h_in = fig.get_size_inches()
    wrap_chars = max(40, int(w_in * 72.0 / (0.545 * fontsize)))
    lines = []
    for para in (text if isinstance(text, (list, tuple)) else [text]):
        lines += textwrap.wrap(para, wrap_chars) or ['']
    pad_pt = 72.0 * pad_mm / figstyle.MM_PER_IN
    text_pt = len(lines) * fontsize * 1.45
    leg_rows = 0 if not handles else int(np.ceil(len(handles) / float(ncol)))
    leg_pt = leg_rows * legend_fontsize * 1.9
    reserve = (text_pt + leg_pt + pad_pt) / 72.0 / h_in
    eng = fig.get_layout_engine()
    if eng is not None:
        eng.set(rect=(0.0, reserve, 1.0, 1.0 - reserve))
    y_text = (text_pt + pad_pt * 0.35) / 72.0 / h_in
    fig.text(0.006, y_text, '\n'.join(lines), fontsize=fontsize, ha='left',
             va='top', color='#333333', linespacing=1.45)
    if handles:
        fig.legend(handles=handles, loc='lower center', ncol=ncol,
                   fontsize=legend_fontsize, frameon=False,
                   bbox_to_anchor=(0.5, y_text + pad_pt * 0.15 / 72.0 / h_in))
    return lines


def decision_region(ax, lo, hi, label, orient='h'):
    """A frozen DECISION region (spec Sec.6 F3 "refutation region shaded").
    Hatched, grey, and always labelled as a decision region so it can never be
    read as a null envelope -- the only other thing this figure set shades."""
    fn = ax.axhspan if orient == 'h' else ax.axvspan
    return fn(lo, hi, facecolor='#F7F7F7', edgecolor='#C9C9C9', hatch='//',
              linewidth=0.0, zorder=0, label=label)


def integer_x(ax, values):
    """Integer ticks on an integer axis (spec F1 "integer ticks 1...max")."""
    v = np.asarray(sorted(set(int(x) for x in values)))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5, min_n_ticks=3))
    ax.set_xlim(v.min() - 0.4, v.max() + 0.4)


def tiny_title(ax, text, color=None, size=7.0):
    """``size=None`` keeps the spec's ``axes.titlesize`` (9 pt)."""
    kw = dict(color=(color or 'black'), pad=2.5)
    if size is not None:
        kw['fontsize'] = size
    ax.set_title(text, **kw)


def _log_ylim(ax, y, pad=1.6):
    """Keep a log axis from collapsing onto one decade when the data span less,
    and from clipping the decision lines."""
    y = np.asarray([v for v in np.ravel(y) if np.isfinite(v) and v > 0], float)
    if y.size == 0:
        return
    lo, hi = y.min() / pad, y.max() * pad
    ax.set_ylim(lo, hi)


# --------------------------------------------------------------------------- #
# status bookkeeping -- what the report has to be able to say                 #
# --------------------------------------------------------------------------- #

class FigStatus(object):
    def __init__(self, stem, title):
        self.stem, self.title, self.panels, self.paths = stem, title, [], []

    def live(self, letter, detail=''):
        self.panels.append((letter, 'live', detail))
        return 'live'

    def awaiting(self, letter, detail):
        self.panels.append((letter, 'awaiting', detail))
        return detail

    def note(self, letter, state, detail=''):
        self.panels.append((letter, state, detail))

    @property
    def n_live(self):
        return sum(1 for _, s, _ in self.panels if s == 'live')

    def __repr__(self):
        return '<%s %d/%d live>' % (self.stem, self.n_live, len(self.panels))


def _placeholder(stem, title, notes, outdir=None, status=None):
    """A figure whose inputs are ENTIRELY missing still writes a page, carrying
    the same "table not available" note -- so a missing figure is visible in the
    deliverables rather than being an absent file nobody notices."""
    st = status or FigStatus(stem, title)
    fig, ax = figstyle.figure_double(70.0)
    ax.set_axis_off()
    ax.text(0.5, 0.72, title, ha='center', va='center', fontsize=9,
            fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.42, '\n'.join(notes), ha='center', va='center', fontsize=7,
            transform=ax.transAxes, linespacing=1.6)
    ax.add_patch(Rectangle((0.02, 0.06), 0.96, 0.88, transform=ax.transAxes,
                           facecolor=STAMP_FACE, edgecolor=STAMP_EDGE,
                           linestyle=(0, (3.5, 2.5)), linewidth=0.6, zorder=0))
    figstyle.panel_letter(ax, '')
    st.paths = figstyle.savefig_both(fig, stem, outdir=outdir)
    return st


def _outline(fig, axes, label=None, pad_mm=1.6, pad_top_mm=4.4, color='#333333'):
    """Draw one rectangle around a group of axes (spec F1's "separately
    outlined control row") and then FREEZE the layout, so ``savefig`` cannot
    re-run constrained_layout and leave the rectangle behind."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    boxes = [a.get_tightbbox(rend).transformed(inv) for a in axes]
    w_in, h_in = fig.get_size_inches()
    px, py = pad_mm / figstyle.MM_PER_IN / w_in, pad_mm / figstyle.MM_PER_IN / h_in
    x0 = min(b.x0 for b in boxes) - px
    x1 = max(b.x1 for b in boxes) + px
    y0 = min(b.y0 for b in boxes) - py
    y1 = max(b.y1 for b in boxes) + pad_top_mm / figstyle.MM_PER_IN / h_in
    fig.add_artist(Rectangle((x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure,
                             fill=False, edgecolor=color, linewidth=0.7,
                             linestyle=(0, (5, 2)), zorder=5))
    if label:                                   # INSIDE the box's top pad
        fig.text(x1 - px * 1.5, y1 - py * 0.5, label, fontsize=6.5, ha='right',
                 va='top', color=color, fontweight='bold')
    fig.set_layout_engine('none')


# --------------------------------------------------------------------------- #
# F1 -- variogram panel (C1)                                                  #
# --------------------------------------------------------------------------- #

def _variogram_series(t5, dms_id):
    """The integer-h rows plus ``V(inf)`` from T05's own ``h == 'random'`` row.
    Returns ``None`` when the assay is not in the table."""
    d = t5[t5.DMS_id == dms_id].copy()
    if d.empty:
        return None
    d['h_int'] = pd.to_numeric(d['h'], errors='coerce')
    rand = d[d['h'].astype(str) == 'random']
    v_inf = float(rand['V_h'].iloc[0]) if len(rand) else np.nan
    d = d[d.h_int.notna()].sort_values('h_int')
    if d.empty:
        return None
    return dict(h=d.h_int.values.astype(int),
                y=d.V_h_over_Vinf.values.astype(float),
                se=(d.V_h_se.values.astype(float) / v_inf
                    if np.isfinite(v_inf) and v_inf > 0 else
                    np.full(len(d), np.nan)),
                n=d.N_h.values.astype(float),
                exact=np.array([str(s).startswith('exact')
                                for s in d.exact_or_sampled.values]),
                v_inf=v_inf)


def peak_of(h, y, se):
    """First interior local maximum of ``y`` resolved beyond the tabulated SEs.

    "Resolved" = the rise from ``h-1`` and the fall to ``h+1`` each exceed twice
    the pooled SE of the two points compared.  No free parameter, no ``N_h``
    cutoff; returns ``(None, None)`` when the curve is still rising at ``h_max``,
    because then no fall has been observed and there is no peak to claim.
    """
    for i in range(len(y)):
        if i + 1 >= len(y):
            break
        s_up = np.hypot(se[i], se[i - 1]) if i > 0 else 0.0
        s_dn = np.hypot(se[i], se[i + 1])
        rose = (i == 0) or (y[i] - y[i - 1] > 2.0 * (s_up if np.isfinite(s_up) else 0.0))
        fell = y[i] - y[i + 1] > 2.0 * (s_dn if np.isfinite(s_dn) else 0.0)
        if rose and fell:
            return int(h[i]), float(y[i])
    return None, None


def _f1_panel(ax, dms_id, t5, t4, annotate_lines=False):
    s = _variogram_series(t5, dms_id)
    if s is None:
        return stamp(ax, 'T05', 'V(h)/V(inf) for %s' % label_of(dms_id))
    st = assay_style(dms_id)
    col = st['color']
    ok = np.isfinite(s['y']) & (s['y'] > 0)

    figstyle.decision_line(ax, 1.0, linestyle='--', color='#5A5A5A', linewidth=0.7)
    for key in ('C1_V1_over_Vinf_sup', 'C1_V1_over_Vinf_ref'):
        figstyle.decision_line(ax, THRESH[key])
        if annotate_lines:                     # annotated once, at the left edge
            ax.annotate('%g' % THRESH[key], xy=(0.025, THRESH[key]),
                        xycoords=('axes fraction', 'data'), ha='left',
                        va='bottom', fontsize=5.6, color='#4D4D4D')

    ax.errorbar(s['h'][ok], s['y'][ok], yerr=s['se'][ok], color=col,
                linestyle=st.get('linestyle', '-'), linewidth=1.1, marker='none',
                elinewidth=0.6, capsize=1.2, zorder=3)
    # filled = exact enumeration, open = sampled: the same glyph in all panels
    for mask, face in ((s['exact'] & ok, col), (~s['exact'] & ok, 'white')):
        if mask.any():
            ax.plot(s['h'][mask], s['y'][mask], linestyle='none',
                    marker=st['marker'], markersize=2.6, markerfacecolor=face,
                    markeredgecolor=st.get('markeredgecolor', col),
                    markeredgewidth=0.6, zorder=4)
    # h = 1 is where the C1 decision is read
    if ok[0] and s['h'][0] == 1:
        ax.plot([1], [s['y'][0]], linestyle='none', marker='o', markersize=5.2,
                markerfacecolor='none', markeredgecolor='#1A1A1A',
                markeredgewidth=0.7, zorder=5)
    hp, vp = peak_of(s['h'], s['y'], s['se'])
    ax.set_yscale('log')
    integer_x(ax, s['h'])
    lo = np.nanmin(s['y'][ok]) if ok.any() else 0.1
    hi = np.nanmax(s['y'][ok]) if ok.any() else 1.0
    # headroom above the maximum so the peak label sits inside the axes and
    # clear of the title; cheap on a log axis
    ax.set_ylim(min(lo / 1.9, THRESH['C1_V1_over_Vinf_sup'] / 1.3),
                max(hi * (2.4 if hp is not None else 1.7), 1.5))
    if hp is not None:
        ax.plot([hp], [vp], linestyle='none', marker='v', markersize=4.0,
                markerfacecolor='#1A1A1A', markeredgecolor='#1A1A1A', zorder=6)
        # BESIDE the marker, one line: above it collides with the panel title
        # whenever the peak is also the maximum (measured, then moved)
        x0, x1 = ax.get_xlim()
        right = (hp - x0) / (x1 - x0) < 0.5
        ax.annotate('peak $h$=%d, %.2f' % (hp, vp), xy=(hp, vp),
                    xytext=(4.5 if right else -4.5, 0.0),
                    textcoords='offset points', fontsize=5.4, color='#1A1A1A',
                    ha='left' if right else 'right', va='center', zorder=7,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none',
                              pad=0.5))
        # the fall AFTER the peak is the finding, so name where it ends
        h_end, v_end = int(s['h'][ok][-1]), float(s['y'][ok][-1])
        if h_end > hp and v_end < vp / 2.0:
            ax.text(0.97, 0.035, r'$\rightarrow$%.2g at $h$=%d' % (v_end, h_end),
                    transform=ax.transAxes, fontsize=5.4, ha='right',
                    va='bottom', color='#1A1A1A', zorder=7,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none',
                              pad=0.5))

    v1 = s['y'][0] if (len(s['y']) and s['h'][0] == 1) else np.nan
    if np.isfinite(v1):
        ax.annotate('%.3f' % v1, xy=(1, v1), xytext=(5.0, -1.5),
                    textcoords='offset points', fontsize=5.4, ha='left',
                    va='top', color='#1A1A1A', zorder=7,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none',
                              pad=0.5))
    verdict = ''
    if t4 is not None and 'verdict_C1' in t4.columns:
        r = t4[t4.DMS_id == dms_id]
        if len(r):
            verdict = str(r.verdict_C1.iloc[0])
    fam = config.ASSAYS[dms_id].family_id
    # The verdict rides in the TITLE, not in a corner of the axes: at four
    # columns every corner is claimed by the curve, the peak label or the
    # h=1 value on at least one assay (measured, then moved).
    tag = label_of(dms_id)
    if verdict:
        tag += ' | C1 %s' % {'REFUTED': 'REF', 'INCONCLUSIVE': 'INC',
                             'SUPPORTED': 'SUP'}.get(verdict, verdict[:3])
    # 7 pt, not the spec's axes.titlesize=9: at 15 small multiples in 183 mm
    # the 9 pt titles of adjacent panels collide (measured, then reduced).
    tiny_title(ax, tag,
               color=(figstyle.FAMILY_EDGE.get(fam, '#000000')
                      if config.ASSAYS[dms_id].tier != 'CONTROL' else '#000000'),
               size=6.6)
    _ = col
    return 'live'


def fig1_variogram_panel(outdir=None, verbose=True):
    """F1 -- 12 primary small multiples grouped by family + the outlined control
    row.  ``V(h)/V(inf)`` on log y against integer Hamming distance."""
    st = FigStatus('F1_variogram_panel', 'F1  variogram: is the landscape smooth in mutation degree?')
    t5, t4 = load('T05', outdir), load('T04', outdir)
    if t5 is None:
        return _placeholder(st.stem, st.title,
                            ['table not available: T05',
                             'F1 needs T05_variogram.csv (V(h), N_h, V(h)/V(inf) and '
                             'the h="random" closed-form V(inf) row).',
                             'No other table carries V(h); nothing is drawn.'],
                            outdir=outdir, status=st)

    ids = list(PRIMARY_BY_FAMILY) + list(CONTROL_ROW)
    fig, axes = figstyle.figure_grid(4, 4, 176.0, width='double')
    flat = axes.ravel()
    letters = list(string.ascii_lowercase)
    for k, dms_id in enumerate(ids):
        ax = flat[k]
        state = _f1_panel(ax, dms_id, t5, t4, annotate_lines=(k == 0))
        figstyle.panel_letter(ax, letters[k])
        st.note(letters[k], 'live' if state == 'live' else 'awaiting',
                '%s%s' % (label_of(dms_id),
                          '' if state == 'live' else ' -- ' + str(state)))
        if k % 4 == 0:
            ax.set_ylabel(r'$V(h)/V(\infty)$')
        if k >= 8:                       # bottom row of each block
            ax.set_xlabel('$h$ (mutations)')

    n_null = sum(n_finite(t5, c) for c in
                 ('V_h_N1_lo', 'V_h_N1_hi', 'V_h_N2_mean'))
    handles = [
        Line2D([0], [0], color='#444444', marker='o', markersize=2.6,
               markerfacecolor='#444444',
               label='observed: filled = exact, open = sampled'),
        Line2D([0], [0], color='none', marker='o', markersize=5.2,
               markerfacecolor='none', markeredgecolor='#1A1A1A',
               label=r'$h=1$, where C1 is decided'),
        Line2D([0], [0], color='none', marker='v', markersize=4.0,
               markerfacecolor='#1A1A1A', label='first SE-resolved peak'),
        Line2D([0], [0], color='#5A5A5A', linestyle='--', linewidth=0.7,
               label=r'$V(h)=V(\infty)$'),
        Line2D([0], [0], color='#4D4D4D', linestyle=':', linewidth=0.6,
               label=r'C1 support $\leq$%.2f / refute $\geq$%.2f'
                     % (THRESH['C1_V1_over_Vinf_sup'],
                        THRESH['C1_V1_over_Vinf_ref'])),
        Patch(facecolor='none', edgecolor='#333333', linestyle='--',
              linewidth=0.7, label='outlined row: CONTROL tier'),
    ]
    notes_axes(flat[15], 'F1  legend', [], handles=handles,
               legend_loc='upper left', legend_bbox=(0.0, 0.90))
    st.note('p', 'live', 'legend / caveats cell')

    footer(fig, [
        r'$x$: Hamming distance $h$ between the two variants of a pair, '
        'integer ticks.  $y$: $V(h)/V(\\infty)$, log scale.  The number beside '
        'the open $h=1$ circle is $V(1)/V(\\infty)$, the statistic the C1 '
        "clause is decided on; the title carries T04's C1 verdict "
        '(INC = INCONCLUSIVE, REF = REFUTED).',
        'Error bars are $\\pm$1 SE from $N_h$ (T05.V_h_se, put on the ratio '
        'scale with the table\'s own $V(\\infty)$ from its $h$="random" row); they '
        'are bars and not a band, because in F1-F7 a shaded region only ever '
        'means a real null envelope or a frozen decision region.  '
        'The N1 and N2 ribbons spec Sec.6 asks for are NOT drawn: '
        'T05.V_h_N1_lo / V_h_N1_hi / V_h_N2_mean carry %d finite values in this '
        'run, and a band that is not a real 5th-95th null envelope is '
        'forbidden.' % n_null,
        'The peak marked is the first interior local maximum resolved beyond '
        'the tabulated SEs (rise and fall each $>2\\times$ the pooled SE of the '
        'compared points); where the curve is still rising at $h_{\\max}$ no peak '
        'is marked, because no fall has been observed.  Bottom row = CONTROL '
        'tier, not data points (outlined): Z-LL1 and Z-LL2 sit ABOVE '
        '$V(\\infty)$ at $h=1$, the pre-declared negative result, on the same '
        'axes as the positive ones.'])

    _outline(fig, [flat[12], flat[13], flat[14]], pad_top_mm=1.0)
    st.paths = figstyle.savefig_both(fig, st.stem, outdir=outdir)
    return st


# --------------------------------------------------------------------------- #
# F2 -- tail survival against the four nulls                                  #
# --------------------------------------------------------------------------- #

_NULLS4 = ('N1', 'N2', 'N2b', 'N2c')
_NULL_MARK = {'N1': ('s', '#9A9A9A'), 'N2': ('D', figstyle.OKABE_ITO['blue']),
              'N2b': ('^', figstyle.OKABE_ITO['bluishgreen']),
              'N2c': ('v', figstyle.OKABE_ITO['vermillion'])}


def _dotplot(ax, ids, obs, nulls, sd=None, xlabel='', logx=True, obs_label='observed'):
    """One row per assay: the observed value and the four null means.

    ``sd`` optionally carries a tabulated per-null SD, drawn as an error BAR
    (T02b tabulates mean and SD over B=200; the 5th-95th percentiles are not
    tabulated, so no band may be drawn -- spec Sec.6)."""
    y = np.arange(len(ids))[::-1]
    for j, a in enumerate(ids):
        for nm in _NULLS4:
            v = nulls.get(nm, {}).get(a, np.nan)
            if not np.isfinite(v) or (logx and v <= 0):
                continue
            mk, cl = _NULL_MARK[nm]
            e = (sd or {}).get(nm, {}).get(a, np.nan)
            if np.isfinite(e) and e > 0:
                ax.errorbar([v], [y[j]], xerr=[e], color=cl, elinewidth=0.6,
                            capsize=1.2, linestyle='none', zorder=2)
            ax.plot([v], [y[j]], marker=mk, markersize=2.9, linestyle='none',
                    markerfacecolor='none', markeredgecolor=cl,
                    markeredgewidth=0.7, zorder=3)
        v = obs.get(a, np.nan)
        if np.isfinite(v) and (v > 0 or not logx):
            stl = assay_style(a, for_line=False)
            # black edge: the observed marker must not be mistaken for the N2
            # diamond on an assay whose family colour is also blue
            ax.plot([v], [y[j]], marker='o', markersize=3.8, linestyle='none',
                    markerfacecolor=stl['color'], markeredgecolor='#000000',
                    markeredgewidth=0.5, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels([label_of(a) for a in ids], fontsize=5.6)
    for j, a in enumerate(ids):
        if config.ASSAYS[a].tier == 'CONTROL':
            ax.get_yticklabels()[j].set_style('italic')
    if logx:
        ax.set_xscale('log')
        vals = [v for v in list(obs.values())
                + [v for n in nulls.values() for v in n.values()]
                if np.isfinite(v) and v > 0]
        if vals:
            ax.set_xlim(min(vals) / 1.55, max(vals) * 1.55)
    # the extra half-row at the bottom is where a vertical decision line's
    # label lives; without it the label sits on the last assay's markers
    ax.set_ylim(-1.35, len(ids) - 0.3)
    ax.grid(axis='y', color='#EDEDED', linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    _ = obs_label


def fig2_tail_survival_vs_nulls(outdir=None, verbose=True):
    """F2 -- the observed tail against FOUR nulls (N1, N2, N2b, N2c).  N2c is
    the point: if heteroscedasticity alone reproduces the tail, localisation is
    a mandatory conjunct, not a nicety."""
    st = FigStatus('F2_tail_survival_vs_nulls',
                   'F2  tail of |c_hat| against the four nulls')
    t2f, t2b, t6 = load('T02f', outdir), load('T02b', outdir), load('T06', outdir)
    if t2f is None and t6 is None:
        return _placeholder(st.stem, st.title,
                            ['table not available: T06, T02f',
                             'F2 needs the observed tail rate and the N1 / N2 / N2b / '
                             'N2c ensembles at each tau.',
                             'No other table carries P(|c_hat| >= tau).'],
                            outdir=outdir, status=st)

    ids = [a for a in assay_order(('PRIMARY', 'ARM', 'CONTROL'))
           if t2f is not None and a in set(t2f.DMS_id)]
    fig, axes = figstyle.figure_grid(2, 3, 132.0, width='double')
    flat = axes.ravel()
    lets = list(string.ascii_lowercase)

    def series(col_fmt, table):
        obs = {a: np.nan for a in ids}
        nul = {n: {} for n in _NULLS4}
        if table is None:
            return obs, nul
        idx = table.set_index('DMS_id')
        for a in ids:
            if a not in idx.index:
                continue
            r = idx.loc[a]
            c = col_fmt % 'obs'
            obs[a] = float(r[c]) if c in idx.columns and pd.notna(r[c]) else np.nan
            for n in _NULLS4:
                c = col_fmt % n
                nul[n][a] = (float(r[c]) if c in idx.columns and pd.notna(r[c])
                             else np.nan)
        return obs, nul

    # ---- (a), (b): tail rate at the two taus that are tabulated -------------
    for k, tau in enumerate((3, 4)):
        ax = flat[k]
        obs, nul = series('rate_sigma_tau%d_%%s' % tau, t2f)
        sd = None
        col = 'rate_sigma_tau%d_N2_sd' % tau
        if t2b is not None and col in t2b.columns:
            sd = {'N2': dict(zip(t2b.DMS_id, t2b[col]))}
        _dotplot(ax, ids, obs, nul, sd=sd,
                 xlabel=r'$P(|\hat c| \geq %d\,\sigma)$' % tau)
        tiny_title(ax, r'tail rate at $\tau=%d\,\sigma$' % tau)
        figstyle.panel_letter(ax, lets[k])
        st.live(lets[k], 'rate at tau=%d, observed + 4 null means (T02f)' % tau)

    # ---- (c): TR1, the tail-ratio statistic --------------------------------
    ax = flat[2]
    obs, nul = series('TR1_%s', t2f)
    sd = None
    if t2b is not None and 'TR1_N2_sd' in t2b.columns:
        sd = {'N2': dict(zip(t2b.DMS_id, t2b.TR1_N2_sd))}
    _dotplot(ax, ids, obs, nul, sd=sd,
             xlabel=r'$TR_1 = Q_{.999}/Q_{.75}$ of $|\hat c|$')
    figstyle.decision_line(ax, THRESH['C2_TR1_gauss'], orient='v')
    ax.annotate('Gaussian %.2f' % THRESH['C2_TR1_gauss'],
                xy=(THRESH['C2_TR1_gauss'], 0.008),
                xycoords=('data', 'axes fraction'), ha='left', va='bottom',
                fontsize=5.6, color='#4D4D4D')
    tiny_title(ax, 'tail ratio $TR_1$')
    figstyle.panel_letter(ax, lets[2])
    st.live(lets[2], 'TR1 observed + 4 null means, N2 +/-1 SD from T02b')

    # ---- (d): excess kurtosis -- the N2c claim in its most direct form -----
    ax = flat[3]
    obs, nul = series('kurt_e_%s', t2f)
    # linear x: the kurtosis range (2.4 to 18.8) is under one decade, where a
    # log axis shows a single labelled tick and reads as a collapsed scale
    _dotplot(ax, ids, obs, nul, xlabel=r'kurtosis of $e$ (Gaussian = 3)',
             logx=False)
    allv = [v for v in list(obs.values())
            + [v for n in nul.values() for v in n.values()] if np.isfinite(v)]
    if allv:
        pad = 0.06 * (max(allv) - min(allv))
        ax.set_xlim(min(min(allv) - pad, 3.0 - pad), max(allv) + pad)
    figstyle.decision_line(ax, 3.0, orient='v')
    ax.annotate('Gaussian 3', xy=(3.0, 0.008),
                xycoords=('data', 'axes fraction'), ha='left', va='bottom',
                fontsize=5.6, color='#4D4D4D')
    tiny_title(ax, 'kurtosis of the interaction residual')
    figstyle.panel_letter(ax, lets[3])
    st.live(lets[3], 'kurtosis of e, observed + 4 null means (T02f)')

    # ---- (e), (f): the mixture fit and the c_hat = 0 spike -----------------
    ax = flat[4]
    if have(t6, 'pi_hat'):
        d = t6[t6.pi_hat.notna()]
        scale = 'latent'
        if 'unit' in d.columns:
            d = d[d.unit.astype(str).str.lower().str.startswith('sigma')]
        if 'scale' in d.columns and (d.scale == scale).any():
            d = d[d.scale == scale]
        d = d.drop_duplicates('DMS_id').set_index('DMS_id')
        y = np.arange(len(ids))[::-1]
        for j, a in enumerate(ids):
            if a not in d.index:
                continue
            r = d.loc[a]
            lo = float(r.get('pi_lo95', np.nan))
            hi = float(r.get('pi_hi95', np.nan))
            stl = assay_style(a, for_line=False)
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [y[j], y[j]], color=stl['color'], linewidth=0.8,
                        solid_capstyle='butt', zorder=2)
            ax.plot([float(r.pi_hat)], [y[j]], marker='o', markersize=3.4,
                    linestyle='none', markerfacecolor=stl['color'],
                    markeredgecolor=stl.get('markeredgecolor', stl['color']),
                    zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels([label_of(a) for a in ids], fontsize=5.6)
        ax.set_xscale('log')
        ax.set_ylim(-1.35, len(ids) - 0.3)
        for key, ha in (('C2_pi_lo', 'right'), ('C2_pi_hi', 'left')):
            figstyle.decision_line(ax, THRESH[key], orient='v')
            ax.annotate('%g' % THRESH[key], xy=(THRESH[key], 0.008),
                        xycoords=('data', 'axes fraction'), ha=ha,
                        va='bottom', fontsize=5.4, color='#4D4D4D')
        ax.set_xlabel(r'$\hat\pi$, mixture weight of the wide component')
        tiny_title(ax, r'mixture fit (%s, $\sigma$; 95%% CI)' % scale)
        st.live(lets[4], 'pi_hat with 95%% CI from T06 (%s scale, sigma units); '
                         'the C2 prior window is %g-%g and every assay sits '
                         'above it' % (scale, THRESH['C2_pi_lo'],
                                       THRESH['C2_pi_hi']))
    else:
        tiny_title(ax, 'mixture fit')
        st.awaiting(lets[4], stamp(
            ax, 'T06', 'the two-component mixture weight pi_hat with its 95%% CI '
            'per assay, against the %g-%g prior window'
            % (THRESH['C2_pi_lo'], THRESH['C2_pi_hi']),
            extra='T06_cliff_tail_C2.csv is written by stage 3 '
                  '(cliff.stats_c2), which had not finished when this figure '
                  'was rendered.'))
    figstyle.panel_letter(ax, lets[4])

    ax = flat[5]
    if 'frac_c_exact_zero' in (t6.columns if t6 is not None else []):
        d = t6.dropna(subset=['frac_c_exact_zero']).drop_duplicates('DMS_id')
        d = d.set_index('DMS_id')
        y = np.arange(len(ids))[::-1]
        vals = []
        for j, a in enumerate(ids):
            if a not in d.index:
                continue
            v = float(d.loc[a].frac_c_exact_zero)
            vals.append(v)
            stl = assay_style(a, for_line=False)
            ax.plot([0, v], [y[j], y[j]], color=stl['color'], linewidth=0.9,
                    zorder=2)                                  # the spike, as a stem
            ax.plot([v], [y[j]], marker='|', markersize=5.0, linestyle='none',
                    markeredgecolor=stl['color'], markeredgewidth=1.0, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels([label_of(a) for a in ids], fontsize=5.6)
        ax.set_ylim(-1.35, len(ids) - 0.3)
        allzero = bool(len(vals)) and max(vals) == 0.0
        if allzero:
            # a panel of zero-length stems reads as a broken panel, so the
            # result is written on it in words
            ax.set_xlim(-0.004, 0.10)
            ax.text(0.60, 0.5, 'frac_c_exact_zero = 0 EXACTLY\nin all %d assays: '
                    'there is no\natom at $\\hat c = 0$ to model'
                    % len(vals), transform=ax.transAxes, ha='center',
                    va='center', fontsize=6.0, color='#8A2A00',
                    linespacing=1.5)
        ax.set_xlabel(r'$P(\hat c \equiv 0)$, the atom at zero')
        tiny_title(ax, r'the $\hat c = 0$ spike')
        st.live(lets[5], 'frac_c_exact_zero for %d assays (T06); it is 0.0 in '
                         'every one, so the spike-and-slab has no spike'
                % len(vals) if allzero else
                'frac_c_exact_zero for %d assays (T06)' % len(vals))
    else:
        tiny_title(ax, r'the $\hat c = 0$ spike')
        st.awaiting(lets[5], stamp(
            ax, 'T06', r'the atom at $\hat c = 0$ (frac_c_exact_zero) as a '
            'separate marked stem per assay, which is what makes the mixture '
            'fit a spike-and-slab rather than a two-Gaussian fit',
            extra='Spec Sec.6 asks for this as a per-assay inset; a dot plot '
                  'has no room for 17 insets, so it is promoted to a panel.'))
    figstyle.panel_letter(ax, lets[5])

    handles = [Line2D([0], [0], marker='o', linestyle='none', markersize=3.8,
                      markerfacecolor='#444444', markeredgecolor='#000000',
                      label='observed (fill = family colour)')]
    handles += [Line2D([0], [0], marker=_NULL_MARK[n][0], linestyle='none',
                       markersize=2.9, markerfacecolor='none',
                       markeredgecolor=_NULL_MARK[n][1], label='%s mean' % n)
                for n in _NULLS4]
    footer(fig, [
        'Null markers are ENSEMBLE MEANS over B=%d replicates; the N2 error '
        'bars are $\\pm$1 SD (T02b).  No 5th-95th percentile is tabulated for '
        'these statistics, so no band is shaded -- a band that is not a real '
        'null envelope is forbidden (spec Sec.6).  Italic assay label = CONTROL '
        'tier, not a data point.' % THRESH['null_B'],
        'Spec Sec.6 asks for the survival curve $P(|\\hat c|\\geq\\tau)$ over '
        'the whole $\\tau$ grid against all four nulls: no table carries it.  '
        'T02f carries $\\tau$ = 3 and 4 in $\\sigma$ units (panels a, b) and the '
        'two tail-shape statistics that do carry all four nulls (c, d); T06 '
        'carries one selected $\\tau$ per assay.  N2c is the panel-d claim: if '
        'heteroscedasticity alone reproduces the observed kurtosis, the tail is '
        'not evidence of interaction and localisation is a mandatory conjunct.'],
        handles=handles, ncol=5)
    st.paths = figstyle.savefig_both(fig, st.stem, outdir=outdir)
    return st


# --------------------------------------------------------------------------- #
# F3 -- enrichment sweep                                                      #
# --------------------------------------------------------------------------- #

def _sweep_from_t13(t13, unit):
    """T(tau) per assay from T13's tau knob, when stage 8's sensitivity table
    exists.  Returns {assay: (taus, T)}."""
    if t13 is None or 'knob' not in t13.columns:
        return {}
    d = t13[t13.knob.astype(str) == 'tau'].copy()
    if 'unit' in d.columns:
        d = d[d.unit.astype(str).str.lower().str.startswith(unit)]
    if d.empty or 'T_N2' not in d.columns:
        return {}
    out = {}
    for a, g in d.groupby('DMS_id'):
        g = g.copy()
        g['tau'] = pd.to_numeric(g.value, errors='coerce')
        g = g[g.tau.notna() & g.T_N2.notna()].sort_values('tau')
        if len(g):
            out[a] = (g.tau.values.astype(float), g.T_N2.values.astype(float))
    return out


def _f3_panel(ax, sweeps, ylabel, null_name):
    decision_region(ax, 1e-3, THRESH['C2_T_ref'],
                    label=r'C2 refutation region ($T<%.1f$), decision region, '
                          'not a null' % THRESH['C2_T_ref'])
    figstyle.decision_line(ax, 1.0, linestyle='--', color='#5A5A5A',
                           linewidth=0.7, label='T=1 (no enrichment)')
    figstyle.decision_line(ax, THRESH['C2_T_sup'],
                           label='C2 support T=%g' % THRESH['C2_T_sup'])
    ymin, ymax = 1.0, 1.0
    for a, (tau, t) in sweeps.items():
        stl = assay_style(a)
        m = np.isfinite(t) & (t > 0)
        if not m.any():
            continue
        ax.plot(tau[m], t[m], color=stl['color'], linestyle=stl.get('linestyle', '-'),
                marker=stl['marker'], markersize=2.6,
                markeredgecolor=stl.get('markeredgecolor', stl['color']),
                markeredgewidth=0.5, linewidth=1.0, label=label_of(a))
        ymin, ymax = min(ymin, np.nanmin(t[m])), max(ymax, np.nanmax(t[m]))
    ax.set_yscale('log')
    ax.set_xticks(list(config.TAUS))
    ax.set_xlim(min(config.TAUS) - 0.4, max(config.TAUS) + 0.4)
    ax.set_ylim(ymin / 2.0, max(ymax * 2.0, THRESH['C2_T_sup'] * 1.6))
    ax.set_xlabel(r'$\tau$  (sweep grid %s)'
                  % ', '.join(str(t) for t in config.TAUS))
    ax.set_ylabel(ylabel)
    _ = null_name


def fig3_enrichment_sweep(outdir=None, verbose=True):
    """F3 -- T(tau) over the whole sweep grid, so the conclusion is read off the
    curve rather than off one magic threshold."""
    st = FigStatus('F3_enrichment_sweep', 'F3  enrichment sweep T(tau)')
    t2f, t13, t6 = load('T02f', outdir), load('T13', outdir), load('T06', outdir)

    t2m = load('T02m', outdir)
    sw_sigma = _sweep_from_t13(t13, 'sigma')
    sw_mad = _sweep_from_t13(t13, 'mad')
    null_sigma, null_mad = 'N2', 'N2'

    def _two_tau(table, fmt, null):
        """T(tau) at the taus a two-tau table carries."""
        out = {}
        if table is None:
            return out, null
        cols = {tau: fmt % tau for tau in (3, 4)}
        if not all(c in table.columns for c in cols.values()):
            return out, null
        idx = table.set_index('DMS_id')
        for a in assay_order(('PRIMARY', 'ARM', 'CONTROL')):
            if a not in idx.index:
                continue
            taus, ts = [], []
            for tau, c in cols.items():
                v = idx.loc[a, c]
                if pd.notna(v):
                    taus.append(float(tau))
                    ts.append(float(v))
            if taus:
                out[a] = (np.array(taus), np.array(ts))
        return out, null

    if not sw_sigma:
        # T02m carries T against N2 -- the null the C2 verdict is read off --
        # at tau = 3 and 4.  Preferred over T02f's T against N1.
        sw_sigma, null_sigma = _two_tau(t2m, 'T_N2_tau%d', 'N2')
    if not sw_sigma:
        sw_sigma, null_sigma = _two_tau(t2f, 'rate_sigma_tau%d_T_N1', 'N1')

    if not sw_sigma and not sw_mad:
        return _placeholder(st.stem, st.title,
                            ['table not available: T13 (knob="tau"), T06, T02f',
                             'F3 needs T(tau) over the tau grid in both unit systems.',
                             'Nothing is drawn: an enrichment curve through one point '
                             'would be a claim the tables do not support.'],
                            outdir=outdir, status=st)

    fig, axes = figstyle.figure_grid(1, 3, 74.0, width='double',
                                     gridspec_kw=dict(width_ratios=(1.0, 1.0, 0.62)))
    ax = axes[0]
    if sw_sigma:
        _f3_panel(ax, sw_sigma,
                  r'$T(\tau) = $ rate$_{\rm obs}$ / rate$_{\rm %s}$' % null_sigma,
                  null_sigma)
        # the only tabulated null INTERVAL for T: T02h's G6 band, two controls.
        # Drawn as a whisker, not a shaded band: the column names do not state
        # the band's coverage, and spec Sec.6 only licenses a 5th-95th shade.
        t2h = load('T02h', outdir)
        n_band = 0
        if t2h is not None and 'T_N2_band_lo_tau3' in t2h.columns:
            for _, r in t2h.iterrows():
                for tau in (3, 4):
                    lo = r.get('T_N2_band_lo_tau%d' % tau, np.nan)
                    hi = r.get('T_N2_band_hi_tau%d' % tau, np.nan)
                    if not (pd.notna(lo) and pd.notna(hi)):
                        continue
                    xx = tau + (0.16 if n_band % 2 else -0.16)
                    ax.plot([xx, xx], [lo, hi], color=figstyle.OKABE_ITO['blue'],
                            linewidth=1.4, alpha=0.75, solid_capstyle='butt',
                            zorder=1)
                    n_band += 1
        tiny_title(ax, r'$\sigma$ units  (null: %s)' % null_sigma)
        if len(next(iter(sw_sigma.values()))[0]) < len(config.TAUS):
            got = sorted({int(t) for v in sw_sigma.values() for t in v[0]})
            ax.text(0.5, 0.02, 'only $\\tau$ = %s tabulated'
                    % ', '.join(str(g) for g in got), transform=ax.transAxes,
                    ha='center', va='bottom', fontsize=5.8, color='#8A2A00')
        st.live('a', 'T(tau) vs %s, sigma units, tau = %s'
                % (null_sigma,
                   ','.join(str(int(t)) for t in
                            sorted({t for v in sw_sigma.values() for t in v[0]}))))
    else:
        tiny_title(ax, r'$\sigma$ units')
        st.awaiting('a', stamp(ax, ('T13', 'T06'),
                               r'$T(\tau)$ for every assay over $\tau$ = %s in '
                               r'$\sigma$ units'
                               % ', '.join(str(t) for t in config.TAUS)))
    figstyle.panel_letter(ax, 'a')

    ax = axes[1]
    if sw_mad:
        _f3_panel(ax, sw_mad, r'$T(\tau)$ (null: %s)' % null_mad, null_mad)
        tiny_title(ax, 'MAD units  (null: %s)' % null_mad)
        st.live('b', 'T(tau) vs %s, MAD units' % null_mad)
    else:
        tiny_title(ax, 'MAD units')
        st.awaiting('b', stamp(
            ax, ('T13', 'T06'),
            r'the same sweep in MAD units ($1.4826\times$MAD, never SD), which '
            'is what answers "you picked a magic scale"',
            extra='No table carries the OBSERVED MAD-unit rate: T02a carries '
                  'the N1 reference mean for rate_mad_tau*, not the '
                  'observation.'))
    figstyle.panel_letter(ax, 'b')

    _ord = assay_order(('PRIMARY', 'ARM', 'CONTROL'))
    handles = []
    for a in sorted(set(list(sw_sigma) + list(sw_mad)),
                    key=lambda x: _ord.index(x) if x in _ord else 99):
        kw = dict(assay_style(a))
        kw.setdefault('linewidth', 1.0)
        kw['markersize'] = 2.6
        handles.append(Line2D([0], [0], label=label_of(a), **kw))
    handles.append(Patch(facecolor='#F7F7F7', edgecolor='#C9C9C9', hatch='//',
                         label='refutation region (decision, not a null)'))
    handles.append(Line2D([0], [0], color=figstyle.OKABE_ITO['blue'],
                          linewidth=1.4, alpha=0.75,
                          label='T02h N2 band for $T$ (2 controls only)'))
    notes_axes(axes[2], 'F3  legend', [], handles=handles,
               legend_loc='upper left', legend_bbox=(0.0, 0.92))
    st.note('c', 'live', 'legend cell')
    footer(fig, [
        'One line per assay: colour = family, marker and linestyle fixed per '
        'assay across F1-F7, so the figure survives greyscale printing.  '
        'The C2 verdict is read off $T$ against N2; the tabulated quantity '
        'plotted is named on the axis.  $T$ against N1 and $T$ against N2 are '
        'NOT interchangeable -- N1 destroys the additive structure, N2 '
        'preserves it -- so the two are never drawn on one axis.',
        'The hatched band is the frozen C2 refutation region '
        '($\\max_\\tau T < %.1f$ with CI upper $< %.1f$): a DECISION region, '
        'the only shading in F1-F7 that is not a real null envelope.  The '
        'whole sweep grid $\\tau$ = %s is drawn on the $x$ axis even where the '
        'tables carry no point, so what is missing is visible.'
        % (THRESH['C2_T_ref'], THRESH['C2_T_ref_ci_hi'],
           ', '.join(str(t) for t in config.TAUS))])
    _ = t6
    st.paths = figstyle.savefig_both(fig, st.stem, outdir=outdir)
    return st


# --------------------------------------------------------------------------- #
# F4 -- localisation                                                          #
# --------------------------------------------------------------------------- #

def fig4_localisation(outdir=None, verbose=True):
    """F4 -- is the jump localised to a site pair, or is it noise?  (a) sibling
    regression against its N2 null band; (b) the KRAS twin; (c) the shared vs
    partner-specific split; (d) site-pair ICC."""
    st = FigStatus('F4_localisation', 'F4  localisation of the jumps')
    t7, t8, t11 = load('T07', outdir), load('T08', outdir), load('T11', outdir)
    t10 = load('T10', outdir, usecols=('DMS_id', 'ICC_sitepair', 'n_aa_combos',
                                       'n_backgrounds', 'eps', 'is_cliff_3sigma'))
    fig, axes = figstyle.figure_grid(2, 2, 138.0, width='double')
    flat = axes.ravel()
    _facts = {}

    # ---- (a) sibling slope against its N2 null band ------------------------
    ax = flat[0]
    if have(t7, 'beta_sibling', 'beta_N2_p995'):
        d = t7[(t7.route.astype(str) == 'L1') & t7.beta_sibling.notna()]
        ids = [a for a in assay_order(('PRIMARY', 'ARM', 'CONTROL'))
               if a in set(d.DMS_id)]
        n_infeas = int((t7[(t7.route.astype(str) == 'L1')].feasible
                        .astype(str).str.lower() == 'false').sum())
        y = np.arange(len(ids))[::-1]
        for j, a in enumerate(ids):
            r = d[d.DMS_id == a].iloc[0]
            stl = assay_style(a, for_line=False)
            # the REAL N2 null interval, 2.5th-97.5th, from the table
            lo = float(r.get('beta_N2_p025', np.nan))
            hi = float(r.get('beta_N2_p975', np.nan))
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [y[j], y[j]],
                        color=figstyle.OKABE_ITO['blue'], linewidth=3.0,
                        alpha=0.30, solid_capstyle='butt', zorder=1)
            p995 = float(r.beta_N2_p995)
            if np.isfinite(p995):
                ax.plot([p995], [y[j]], marker='|', markersize=6.5,
                        linestyle='none',
                        markeredgecolor=figstyle.OKABE_ITO['blue'],
                        markeredgewidth=0.9, zorder=3)
            se, b = float(r.get('se_hc3', np.nan)), float(r.beta_sibling)
            if np.isfinite(se):
                ax.errorbar([b], [y[j]], xerr=[1.96 * se], color=stl['color'],
                            elinewidth=0.6, capsize=1.2, linestyle='none',
                            zorder=4)
            ax.plot([b], [y[j]], marker='o', markersize=3.6, linestyle='none',
                    markerfacecolor=stl['color'], markeredgecolor='#000000',
                    markeredgewidth=0.4, zorder=5)
        ax.set_yticks(y)
        ax.set_yticklabels([label_of(a) for a in ids], fontsize=5.6)
        ax.set_ylim(-1.0, len(ids) - 0.3)
        figstyle.decision_line(ax, 0.0, orient='v', linestyle='--',
                               color='#5A5A5A', linewidth=0.7)
        ax.annotate(r'$\beta_a=0$', xy=(0.0, 0.008),
                    xycoords=('data', 'axes fraction'), ha='right',
                    va='bottom', fontsize=5.2, color='#4D4D4D')
        ax.set_xlabel(r'$\beta_a$: slope of $e$ on its sibling mean')
        tiny_title(ax, r'localisation: $\beta_a$ vs its N2 null', size=7.0)
        st.live('a', 'beta_sibling with HC3 CI against the tabulated N2 '
                     '2.5-97.5 interval and its 99.5th percentile, %d feasible '
                     'L1 assays (%d infeasible, T07)' % (len(ids), n_infeas))
    else:
        ax.set_xlabel('sibling mean of $e$ at the same site pair')
        ax.set_ylabel(r'$e$ (interaction residual)')
        st.awaiting('a', stamp(
            ax, 'T07', r'$e$ against its sibling mean with the fitted $\beta_a$, '
            'the N2 null band (not an analytic zero), cliff edges highlighted '
            r'and $\beta_a(\tau)$ inset',
            extra='T07 columns needed: beta_sibling, se_hc3, beta_N2_p995 '
                  '(route L1).  Written by stage 3, cliff.stats_c3.'))
    figstyle.panel_letter(ax, 'a')

    # ---- (b) the KRAS twin: does a flagged cliff replicate? ----------------
    ax = flat[1]
    if have(t8, 'R', 'pearson_raw'):
        d = t8[t8.R.notna()].copy()
        if 'row_role' in d.columns:
            d = d.sort_values(['threshold_label', 'sigma_mult'])
        y = np.arange(len(d))[::-1]
        labs = []
        for j, (_, r) in enumerate(d.iterrows()):
            prim = str(r.get('row_role', '')) == 'primary'
            lo, hi = float(r.get('R_lo95', np.nan)), float(r.get('R_hi95', np.nan))
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [y[j], y[j]], color='#333333',
                        linewidth=1.1 if prim else 0.7, solid_capstyle='butt',
                        zorder=2)
            ax.plot([float(r.R)], [y[j]], marker='o',
                    markersize=4.6 if prim else 3.4, linestyle='none',
                    markerfacecolor=figstyle.FAMILY_COLOR['F2'],
                    markeredgecolor='#000000',
                    markeredgewidth=0.7 if prim else 0.4, zorder=5)
            ax.plot([float(r.R_chance_perm)], [y[j]], marker='s',
                    markersize=2.8, linestyle='none', markerfacecolor='none',
                    markeredgecolor='#7A7A7A', markeredgewidth=0.7, zorder=4)
            thr = {'3sigma_eps_measured': r'$3\sigma_\varepsilon$',
                   '3sqrt3sigma_y_contrast': r'$3\sqrt{3}\sigma_y$'}.get(
                       str(r.get('threshold_label', '')),
                       str(r.get('threshold_label', ''))[:12])
            labs.append('%s $\\times$%g%s\n(n=%d flagged)'
                        % (thr, float(r.get('sigma_mult', 1)),
                           ' primary' if prim else '',
                           int(r.get('n_flagged_a', 0))))
        ax.set_yticks(y)
        ax.set_yticklabels(labs, fontsize=5.2)
        ax.set_ylim(-1.0, len(d) - 0.3)
        ax.set_xlim(0.0, 1.06)
        for key, lab, yy, va in (('C3N_R_sup', 'C3-N support', 0.99, 'top'),
                                 ('C3N_R_ref', 'refute', 0.008, 'bottom')):
            figstyle.decision_line(ax, THRESH[key], orient='v')
            ax.annotate('%s %.2f' % (lab, THRESH[key]), xy=(THRESH[key], yy),
                        xycoords=('data', 'axes fraction'), ha='left', va=va,
                        fontsize=5.2, color='#4D4D4D')
        r0 = d[d.get('row_role', '') == 'primary']
        r0 = r0.iloc[0] if len(r0) else d.iloc[0]
        _facts['twin'] = (
            '%s vs %s, n=%d shared site pairs, r=%.3f, OLS slope %.3f, '
            'sign agreement %.3f against a %.3f chance level, permutation '
            'p=%.0e.' % (short_of(str(r0.assay_a)), short_of(str(r0.assay_b)),
                         int(r0.n_shared), float(r0.pearson_raw),
                         float(r0.ols_slope), float(r0.sign_agreement),
                         float(r0.get('sign_agreement_chance', np.nan)),
                         float(r0.perm_p)))
        ax.set_xlabel(r'$R$: fraction of flagged cliffs that replicate')
        tiny_title(ax, 'KRAS twin: does a flagged cliff replicate?', size=7.0)
        st.live('b', 'R with 95%% CI against its permutation chance level over '
                     'the %d tabulated threshold variants of the one testable '
                     'twin (T08)' % len(d))
    else:
        st.awaiting('b', stamp(
            ax, 'T08', r'the KRAS twin hexbin of $\varepsilon_a$ vs '
            r'$\varepsilon_b$ over the shared site pairs, the affine line, the '
            r'$\pm3\sigma$ box, the measured $r$, and the ROC with a bootstrap '
            'band',
            extra='T10 carries per-assay eps, but joining SOS1/8BE4 to '
                  'DARPinK27/5O2S is T08\'s job: a naive (pos, aa) cross-assay '
                  'join is BANNED (gate G1b), so it is not improvised here.'))
    figstyle.panel_letter(ax, 'b')

    # ---- (c) shared vs partner-specific ------------------------------------
    ax = flat[2]
    if have(t11, 'F_spec', 'F_spec_noise_corrected'):
        d = (t11.dropna(subset=['F_spec'])
             .drop_duplicates(subset=['family', 'channel']))
        d = d.sort_values('F_spec_noise_corrected', ascending=False)
        x = np.arange(len(d), dtype=float)
        w, labs = 0.38, []
        for k, (_, r) in enumerate(d.iterrows()):
            ax.bar(x[k] - w / 2, r.F_spec, w, color='#C8C8C8',
                   edgecolor='#333333', linewidth=0.4, zorder=2)
            ax.bar(x[k] + w / 2, r.F_spec_noise_corrected, w,
                   color=figstyle.OKABE_ITO['vermillion'], alpha=0.85,
                   edgecolor='#333333', linewidth=0.4, zorder=2)
            note = []
            if bool(r.get('F_spec_at_boundary', False)):
                note.append('at bound')
            if bool(r.get('structurally_mute', False)):
                note.append('mute')
            labs.append('%s\n%s, K=%d\nn=%d%s'
                        % (r.family, r.channel, int(r.K_partners),
                           int(r.get('F_spec_n_shared', 0)),
                           '\n' + ', '.join(note) if note else ''))
        for key, lab, va in (('C4I_Fspec_sup', 'C4-I support', 'bottom'),
                             ('C4I_Fspec_ref', 'refute', 'top')):
            figstyle.decision_line(ax, THRESH[key])
            ax.annotate('%s %.2f' % (lab, THRESH[key]),
                        xy=(0.012, THRESH[key]),
                        xycoords=('axes fraction', 'data'), ha='left', va=va,
                        fontsize=5.2, color='#4D4D4D', zorder=6,
                        bbox=dict(facecolor='white', alpha=0.75,
                                  edgecolor='none', pad=0.4))
        ax.set_xticks(x)
        ax.set_xticklabels(labs, fontsize=5.0)
        ax.set_xlim(-0.7, len(d) - 0.3)
        ax.set_ylim(0, 1.34)
        ax.set_ylabel(r'$F_{\rm spec}$ (partner-specific share)')
        ax.legend(handles=[
            Patch(facecolor='#C8C8C8', edgecolor='#333333', label='raw'),
            Patch(facecolor=figstyle.OKABE_ITO['vermillion'], alpha=0.85,
                  edgecolor='#333333', label='noise-corrected (the one that counts)')],
            fontsize=5.4, loc='upper center', ncol=2)
        tiny_title(ax, r'$F_{\rm spec}$: shared $\mu$ vs partner-specific $\delta$',
                   size=7.0)
        st.live('c', '%d family x channel rows of F_spec / F_spec_noise_corrected '
                     '(T11)' % len(d))
    else:
        st.awaiting('c', stamp(
            ax, ('T11', 'T08'),
            r'the variance decomposition (shared $\mu$ / partner-specific '
            r'$\delta$ / noise) per family with $F_{\rm spec}$'))
    figstyle.panel_letter(ax, 'c')

    # ---- (d) site-pair ICC --------------------------------------------------
    ax = flat[3]
    if have(t10, 'ICC_sitepair'):
        g = (t10.dropna(subset=['ICC_sitepair'])
             .groupby('DMS_id')
             .agg(icc=('ICC_sitepair', 'first'),
                  nuniq=('ICC_sitepair', 'nunique'),
                  n_pairs=('ICC_sitepair', 'size'),
                  aa_max=('n_aa_combos', 'max')))
        ids = [a for a in assay_order(('PRIMARY', 'ARM', 'CONTROL'))
               if a in g.index]
        y = np.arange(len(ids))[::-1]
        for j, a in enumerate(ids):
            stl = assay_style(a, for_line=False)
            ax.plot([g.loc[a, 'icc']], [y[j]], marker=stl['marker'],
                    markersize=4.0, linestyle='none',
                    markerfacecolor=stl['color'], markeredgecolor='#000000',
                    markeredgewidth=0.4, zorder=4)
            ax.annotate('n=%d, $\\leq$%d aa combos'
                        % (int(g.loc[a, 'n_pairs']), int(g.loc[a, 'aa_max'])),
                        xy=(g.loc[a, 'icc'], y[j]), xytext=(5, 0),
                        textcoords='offset points', va='center', ha='left',
                        fontsize=4.8, color='#555555')
        ax.set_yticks(y)
        ax.set_yticklabels([label_of(a) for a in ids], fontsize=5.6)
        ax.set_ylim(-1.0, len(ids) - 0.3)
        ax.set_xlim(0, 1.24)
        for key, lab, yy, va in (('C3L_ICC_sup', 'C3-L support', 0.99, 'top'),
                                 ('C3L_ICC_ci_hi_ref', 'refute (CI upper)',
                                  0.008, 'bottom')):
            figstyle.decision_line(ax, THRESH[key], orient='v')
            ax.annotate('%s %.2f' % (lab, THRESH[key]), xy=(THRESH[key], yy),
                        xycoords=('data', 'axes fraction'), ha='left', va=va,
                        fontsize=5.2, color='#4D4D4D')
        ax.set_xlabel('ICC of $e$ within a site pair')
        tiny_title(ax, 'site-pair ICC (one value per assay)', size=7.0)
        nuq = int(g.nuniq.max())
        st.live('d', 'ICC_sitepair for %d assays (T10); constant within an '
                     'assay (max %d distinct value per assay), so the '
                     'across-aa-combination breakdown Sec.6 asks for is not '
                     'tabulated' % (len(ids), nuq))
    else:
        st.awaiting('d', stamp(
            ax, 'T10', 'the GB1_IgG-Fc (1FCC) site-pair ICC across amino-acid '
            'combinations'))
    figstyle.panel_letter(ax, 'd')

    footer(fig, [
        (_facts.get('twin', '') + '  ') +
        'Panel (b): spec Sec.6 asks for a hexbin of $\\varepsilon_a$ vs '
        r'$\varepsilon_b$ over the shared site pairs.  T08 tabulates that '
        'join\'s SUMMARY (n_shared, pearson_raw, ols_slope, sign agreement, '
        r'$R$ and its permutation chance level) but not the per-pair '
        r'$\varepsilon$; T10 carries per-assay $\varepsilon$, and joining two '
        'assays by hand is exactly the naive cross-assay join gate G1b bans, '
        'so the panel shows what T08 does tabulate: $R$ at every threshold '
        'variant against its own permutation chance level; the measured '
        'summary statistics open this footer.',
        'Panel (c): T11 tabulates the RATIO $F_{\\rm spec}$ (and its '
        'noise-corrected form, with $\\sigma_\\varepsilon^2$ = %.5f subtracted), '
        'not the three variance components separately, so the panel shows the '
        'ratio and not the stacked shared / specific / noise decomposition spec '
        'Sec.6 describes.  "at bound" = F_spec_at_boundary, i.e. the estimate '
        'sits at 1.00 and the family carries no usable shared component.'
        % THRESH['C4I_sigma_eps_sq'],
        'Panel (d): T10\'s ICC_sitepair is ONE value per assay repeated on '
        'every site-pair row (verified: at most one distinct value per assay), '
        'so Sec.6\'s "ICC across amino-acid combinations" cannot be drawn from '
        'it; the per-assay value is shown instead and the count of site pairs '
        'and the largest number of amino-acid combinations behind it are '
        'annotated.'],
        handles=[
            Line2D([0], [0], marker='o', linestyle='-', markersize=3.6,
                   color='#444444', markerfacecolor='#444444',
                   markeredgecolor='#000000',
                   label=r'(a) observed $\beta_a$ $\pm$1.96 HC3 SE'),
            Patch(facecolor=figstyle.OKABE_ITO['blue'], alpha=0.30,
                  edgecolor='none', label='(a) N2 null 2.5th-97.5th'),
            Line2D([0], [0], marker='|', linestyle='none', markersize=6.5,
                   markeredgecolor=figstyle.OKABE_ITO['blue'],
                   label='(a) N2 99.5th pct'),
            Line2D([0], [0], marker='o', linestyle='-', markersize=4.0,
                   color='#333333', markerfacecolor=figstyle.FAMILY_COLOR['F2'],
                   markeredgecolor='#000000', label=r'(b) $R$ with 95% CI'),
            Line2D([0], [0], marker='s', linestyle='none', markersize=2.8,
                   markerfacecolor='none', markeredgecolor='#7A7A7A',
                   label='(b) permutation chance level')], ncol=5)
    st.paths = figstyle.savefig_both(fig, st.stem, outdir=outdir)
    return st


# --------------------------------------------------------------------------- #
# F5 -- structure                                                             #
# --------------------------------------------------------------------------- #

def fig5_structure(outdir=None, verbose=True):
    """F5 -- is the cliff at the interface?  (a) burial-matched cliff rate by
    Levy class; (b) AUROC(-d3d) forest against NS2; (c) the KRAS double-centred
    heatmap; (d) NS3; (e) the twin-structure control; (f) the designed
    negative."""
    st = FigStatus('F5_structure', 'F5  structural localisation')
    t9, t11 = load('T09', outdir), load('T11', outdir)
    t10 = load('T10', outdir, usecols=('DMS_id', 'AUROC_contribution', 'p_NS2',
                                       'is_cliff_3sigma', 'is_noncliff_1sigma'))
    if t9 is None and t10 is None and t11 is None:
        return _placeholder(st.stem, st.title,
                            ['table not available: T09, T10, T11'],
                            outdir=outdir, status=st)

    fig, axes = figstyle.figure_grid(2, 3, 136.0, width='double')
    flat = axes.ravel()

    # ---- (a) burial-matched cliff rate by Levy class -----------------------
    ax = flat[0]
    if have(t9, 'cliff_rate', 'n_pairs_at_site', 'n_cliff_pairs'):
        perm = t9[t9.assay_permissible.astype(str).str.lower().isin(('true', '1'))]
        d = perm.dropna(subset=['n_pairs_at_site', 'n_cliff_pairs'])
        n_assay = d.DMS_id.nunique()
        classes = [c for c in ('core', 'support', 'rim', 'interior', 'surface')
                   if c in set(d.levy_class)]
        marks = dict(zip(('core', 'support', 'rim', 'interior', 'surface'),
                         ('o', 's', '^', 'D', 'v')))
        lines = dict(zip(('core', 'support', 'rim', 'interior', 'surface'),
                         ('-', (0, (4, 1.5)), (0, (1, 1.2)),
                          (0, (5, 1.2, 1, 1.2)), (0, (7, 1.5)))))
        cols = dict(zip(('core', 'support', 'rim', 'interior', 'surface'),
                        (figstyle.OKABE_ITO['vermillion'],
                         figstyle.OKABE_ITO['orange'],
                         figstyle.OKABE_ITO['reddishpurple'],
                         figstyle.OKABE_ITO['blue'],
                         figstyle.OKABE_ITO['skyblue'])))
        for cls in classes:
            g = d[d.levy_class == cls].groupby('rsa_decile')[
                ['n_cliff_pairs', 'n_pairs_at_site']].sum()
            g = g[g.n_pairs_at_site > 0]
            if not len(g):
                continue
            rate = (g.n_cliff_pairs / g.n_pairs_at_site).values
            se = np.sqrt(np.clip(rate * (1 - rate), 0, None)
                         / g.n_pairs_at_site.values)
            ax.errorbar(g.index.values, rate, yerr=se, color=cols[cls],
                        linestyle=lines[cls], marker=marks[cls], markersize=2.8,
                        linewidth=1.0, elinewidth=0.5, capsize=1.0,
                        markeredgecolor='#000000', markeredgewidth=0.3,
                        label='%s (%d sites)' % (cls, len(d[d.levy_class == cls])))
        ax.set_xlabel(r'$rsa_{\rm iso}$ decile (1 = most buried)')
        ax.set_ylabel(r'pooled cliff rate at the site, $\tau=3\sigma$')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.set_ylim(bottom=0.0)
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks(range(1, 11))
        ax.tick_params(axis='x', labelsize=5.6)
        tiny_title(ax, 'cliff rate by Levy class', size=6.8)
        st.live('a', 'pooled cliff rate over levy_class x rsa_decile for the '
                     '%d permissible assays (T09); NS1 bands NOT drawn -- T09 '
                     'carries p_NS1 but no NS1 rate quantiles' % n_assay)
    else:
        st.awaiting('a', stamp(
            ax, 'T09', 'the burial-matched cliff rate by Levy class within '
            r'$rsa_{\rm iso}$ deciles, with NS1 bands'))
    figstyle.panel_letter(ax, 'a')

    # ---- (b) AUROC(-d3d) forest -------------------------------------------
    ax = flat[1]
    if have(t10, 'AUROC_contribution'):
        d = t10.dropna(subset=['AUROC_contribution'])
        g = d.groupby('DMS_id').agg(auroc=('AUROC_contribution', 'mean'),
                                    n=('AUROC_contribution', 'size'))
        pn = t10.dropna(subset=['p_NS2']).groupby('DMS_id').p_NS2.first()
        nn = (t10[t10.is_noncliff_1sigma.astype(str).str.lower()
                  .isin(('true', '1'))].groupby('DMS_id').size())
        ids = [a for a in assay_order(('PRIMARY', 'ARM', 'CONTROL'))
               if a in g.index]
        y = np.arange(len(ids))[::-1]
        for j, a in enumerate(ids):
            stl = assay_style(a, for_line=False)
            ax.plot([g.loc[a, 'auroc']], [y[j]], marker=stl['marker'],
                    markersize=4.0, linestyle='none',
                    markerfacecolor=stl['color'], markeredgecolor='#000000',
                    markeredgewidth=0.4, zorder=4)
            txt = '%d/%d' % (int(g.loc[a, 'n']), int(nn.get(a, 0)))
            if a in pn.index:
                txt += ', $p_{NS2}$%s%.0e' % ('$\\leq$' if pn[a] <= 1e-4 else '=',
                                              pn[a])
            ax.annotate(txt, xy=(g.loc[a, 'auroc'], y[j]), xytext=(5, 0),
                        textcoords='offset points', va='center', ha='left',
                        fontsize=4.8, color='#555555')
        ax.set_yticks(y)
        ax.set_yticklabels([label_of(a) for a in ids], fontsize=5.6)
        ax.set_ylim(-1.0, len(ids) - 0.3)
        ax.set_xlim(0.40, 1.16)
        figstyle.decision_line(ax, 0.5, orient='v', linestyle='--',
                               color='#5A5A5A', linewidth=0.7)
        ax.annotate('0.5 chance', xy=(0.5, 0.008),
                    xycoords=('data', 'axes fraction'), ha='right', va='bottom',
                    fontsize=5.2, color='#4D4D4D')
        figstyle.decision_line(ax, THRESH['C3L_AUROC_sup'], orient='v')
        ax.annotate('L5 support %.2f' % THRESH['C3L_AUROC_sup'],
                    xy=(THRESH['C3L_AUROC_sup'], 0.955),
                    xycoords=('data', 'axes fraction'), ha='left', va='top',
                    fontsize=5.2, color='#4D4D4D')
        ax.set_xlabel(r'AUROC($-d_{\rm 3D}$), cliff vs non-cliff pairs')
        tiny_title(ax, 'are cliff pairs closer in 3D?', size=6.8)
        st.live('b', 'AUROC per assay = mean of the tabulated per-cliff-pair '
                     'AUROC_contribution, %d assays (T10)' % len(ids))
    else:
        st.awaiting('b', stamp(
            ax, 'T10', r'a forest of AUROC($-d_{\rm 3D}$) per assay against its '
            'NS2 permutation null'))
    figstyle.panel_letter(ax, 'b')

    # ---- (c) the KRAS double-centred heatmap ------------------------------
    ax = flat[2]
    zc = ['Z_doublecentered_p%d' % k for k in (1, 2, 3, 4)]
    dc = ['min_heavy_dist_p%d' % k for k in (1, 2, 3, 4)]
    fc = ['iface_flag_p%d' % k for k in (1, 2, 3, 4)]
    kras = None if t11 is None else t11[t11.family == 'KRAS']
    if kras is not None and len(kras) and kras[zc].notna().any().all():
        d = kras.dropna(subset=zc).sort_values('rowmean_Z')
        Z = d[zc].values.astype(float)
        vmax = np.nanmax(np.abs(Z))
        im = ax.imshow(Z, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       interpolation='nearest', origin='lower')
        # interface membership per partner, on top of the cell it belongs to
        for k in range(4):
            if fc[k] not in d.columns:
                continue
            m = d[fc[k]].astype(str).str.lower().isin(('true', '1')).values
            if m.any():
                ax.plot(np.full(m.sum(), k), np.nonzero(m)[0], marker='.',
                        markersize=1.1, linestyle='none', color='#000000',
                        zorder=3)
        parts = str(d.partners.iloc[0]).split('|')
        ax.set_xticks(range(4))
        ax.set_xticklabels(['%s\n%.1f$\\AA$'
                            % (p.split('_')[1].replace('-RBD', ''),
                               np.nanmedian(d[dc[k]].values))
                            for k, p in enumerate(parts[:4])], fontsize=4.6,
                           rotation=45, ha='right')
        ax.set_ylabel('KRAS site, ordered by row mean $Z$')
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label('double-centred $Z$ of the cliff rate', fontsize=5.6)
        cb.ax.tick_params(labelsize=5.2)
        tiny_title(ax, 'KRAS %d sites $\\times$ 4 partners' % len(d), size=6.8)
        st.live('c', 'KRAS %d x 4 double-centred Z (T11), interface sites '
                     'marked; spec Sec.6 says 163 x 4' % len(d))
    else:
        st.awaiting('c', stamp(
            ax, 'T11', 'the KRAS double-centred cliff-rate heatmap with each '
            'partner\'s interface distance annotated'))
    figstyle.panel_letter(ax, 'c')

    # ---- (d) NS3 ----------------------------------------------------------
    ax = flat[3]
    if have(t11, 'family_M_stat', 'NS3_null_sd'):
        d = (t11.dropna(subset=['family_M_stat'])
             .drop_duplicates(subset=['family', 'channel'])
             .sort_values('family_M_stat'))
        y = np.arange(len(d))[::-1]
        for j, (_, r) in enumerate(d.iterrows()):
            sd = float(r.NS3_null_sd)
            mu = float(r.NS3_null_mean)
            if np.isfinite(sd):
                ax.errorbar([mu], [y[j]], xerr=[sd], color='#9A9A9A',
                            elinewidth=0.8, capsize=1.5, linestyle='none',
                            zorder=2)
                ax.plot([mu], [y[j]], marker='s', markersize=2.6,
                        markerfacecolor='none', markeredgecolor='#7A7A7A',
                        linestyle='none', zorder=3)
            ax.plot([float(r.family_M_stat)], [y[j]], marker='o',
                    markersize=4.0, markerfacecolor=figstyle.OKABE_ITO['vermillion'],
                    markeredgecolor='#000000', markeredgewidth=0.4,
                    linestyle='none', zorder=5)
            ax.text(0.985, y[j], ('$p_{NS3}$=%.3f' % float(r.family_p_NS3))
                    if np.isfinite(float(r.family_p_NS3)) else '$p_{NS3}$ n/a',
                    transform=ax.get_yaxis_transform(), va='center',
                    ha='right', fontsize=5.0, color='#555555')
        ax.set_yticks(y)
        ax.set_yticklabels(['%s (%s, K=%d)' % (r.family, r.channel,
                                               int(r.K_partners))
                            for _, r in d.iterrows()], fontsize=5.6)
        ax.set_ylim(-0.8, len(d) - 0.2)
        figstyle.decision_line(ax, 0.0, orient='v', linestyle='--',
                               color='#5A5A5A', linewidth=0.7)
        ax.set_xlabel(r'$M$, partner-specificity statistic')
        _ns3_b = int(d.NS3_B.max() or 0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.tick_params(axis='x', labelsize=5.6)
        lo = float(np.nanmin([d.family_M_stat.min(),
                              (d.NS3_null_mean - d.NS3_null_sd).min()]))
        hi = float(np.nanmax([d.family_M_stat.max(),
                              (d.NS3_null_mean + d.NS3_null_sd).max()]))
        span = max(hi - lo, 1e-6)
        ax.set_xlim(lo - 0.10 * span, hi + 0.75 * span)   # room for the p labels
        tiny_title(ax, 'NS3 partner-label permutation', size=6.8)
        st.live('d', 'M_obs vs the NS3 null mean +/-1 SD for %d family x '
                     'channel rows (T11); the %d permutation draws themselves '
                     'are not tabulated, so no histogram'
                % (len(d), int(d.NS3_B.max() or 0)))
    else:
        st.awaiting('d', stamp(ax, 'T11', 'the NS3 permutation histogram and '
                                          r'$M_{\rm obs}$'))
    figstyle.panel_letter(ax, 'd')

    # ---- (e) twin-structure control ---------------------------------------
    ax = flat[4]
    if have(t11, 'twin_structure_OR_8BE4', 'twin_structure_OR_5O2S'):
        r = t11.dropna(subset=['twin_structure_OR_8BE4']).iloc[0]
        ors = [('KRAS-SOS1 (8BE4)', float(r.twin_structure_OR_8BE4),
                'KRAS_SOS1_norfitness_8BE4'),
               ('KRAS-DARPinK27 (5O2S)', float(r.twin_structure_OR_5O2S),
                'KRAS_DARPinK27_norfitness_5O2S')]
        y = np.arange(len(ors), dtype=float)[::-1]
        for k, (lab, v, dms) in enumerate(ors):
            lo = hi = np.nan
            if t9 is not None and 'OR_lo95' in t9.columns:
                q = t9[(t9.DMS_id == dms) & t9.OR_burial_matched.notna()]
                if len(q):
                    lo, hi = float(q.OR_lo95.iloc[0]), float(q.OR_hi95.iloc[0])
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [y[k], y[k]], color='#333333', linewidth=0.9,
                        solid_capstyle='butt', zorder=2)
                for e in (lo, hi):
                    ax.plot([e, e], [y[k] - 0.07, y[k] + 0.07], color='#333333',
                            linewidth=0.9, zorder=2)
            ax.plot([v], [y[k]], marker='o', markersize=5.0,
                    markerfacecolor=figstyle.FAMILY_COLOR['F2'],
                    markeredgecolor='#000000', markeredgewidth=0.5,
                    linestyle='none', zorder=5)
            ax.annotate('OR=%.2f%s' % (v, ' (95%% CI %.2f-%.2f)' % (lo, hi)
                                       if np.isfinite(lo) else
                                       ' (no CI tabulated)'),
                        xy=(v, y[k]), xytext=(0, 9), textcoords='offset points',
                        va='bottom', ha='center', fontsize=5.4)
        figstyle.decision_line(ax, 1.0, orient='v', linestyle='--',
                               color='#5A5A5A', linewidth=0.7)
        ax.annotate('OR=1', xy=(1.0, 0.01), xycoords=('data', 'axes fraction'),
                    ha='right', va='bottom', fontsize=5.2, color='#4D4D4D')
        figstyle.decision_line(ax, THRESH['C4S_OR_sup'], orient='v')
        ax.annotate('C4-S support %.1f' % THRESH['C4S_OR_sup'],
                    xy=(THRESH['C4S_OR_sup'], 0.01),
                    xycoords=('data', 'axes fraction'), ha='left', va='bottom',
                    fontsize=5.2, color='#4D4D4D')
        ax.set_yticks(y)
        ax.set_yticklabels([l for l, _, _ in ors], fontsize=5.6)
        ax.set_ylim(-0.6, len(ors) - 0.25)
        ax.set_xlabel('burial-matched interface OR')
        tiny_title(ax, 'G11 twin structure', size=6.8)
        st.live('e', 'twin-structure ORs 8BE4 / 5O2S (T11) with the 8BE4 CI '
                     'from T09')
    else:
        st.awaiting('e', stamp(ax, 'T11', 'the twin-structure control, OR at '
                                          '8BE4 vs OR at 5O2S'))
    figstyle.panel_letter(ax, 'e')

    # ---- (f) the designed negative ----------------------------------------
    ax = flat[5]
    neg = None if t11 is None else t11[t11.family == '5A12']
    if neg is not None and len(neg) and neg.cliff_rate_p1.notna().any():
        d = neg.dropna(subset=['cliff_rate_p1', 'cliff_rate_p2'])
        parts = str(d.partners.iloc[0]).split('|')
        x = np.arange(len(d), dtype=float)
        for k, (_, r) in enumerate(d.iterrows()):
            ax.plot([x[k], x[k]], [r.cliff_rate_p1, r.cliff_rate_p2],
                    color='#BBBBBB', linewidth=0.6, zorder=2)
        ax.plot(x, d.cliff_rate_p1.values, marker='o', linestyle='none',
                markersize=3.6, markerfacecolor=figstyle.FAMILY_COLOR['F4'],
                markeredgecolor='#000000', markeredgewidth=0.4, zorder=4,
                label='%s (negative)' % parts[0].split('_')[1])
        ax.plot(x, d.cliff_rate_p2.values, marker='s', linestyle='none',
                markersize=3.6, markerfacecolor='white',
                markeredgecolor=figstyle.FAMILY_EDGE['F4'], markeredgewidth=0.7,
                zorder=4, label=(parts[1].split('_')[1]
                                 if len(parts) > 1 else 'partner 2'))
        ax.set_xticks(x)
        ax.set_xticklabels(['%s%s' % (r.wt_aa, int(r.resseq))
                            for _, r in d.iterrows()], fontsize=5.2,
                           rotation=45, ha='right')
        ax.set_ylabel(r'cliff rate at the site, $\tau=3\sigma$')
        ax.legend(fontsize=5.2, loc='upper right')
        iface = int(sum(str(v).lower() in ('true', '1')
                        for v in d.iface_flag_p1.values))
        ax.set_ylim(bottom=0.0)
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y1 + 0.35 * (y1 - y0))
        tiny_title(ax, '5A12: designed negative\n%d of %d sites at the %s '
                   'interface' % (iface, len(d), parts[0].split('_')[1]),
                   size=6.4)
        st.live('f', '5A12 per-site cliff rate for both partners, %d sites '
                     '(T11)' % len(d))
    else:
        st.awaiting('f', stamp(ax, 'T11', 'the 5A12-VEGF designed-negative '
                                          'panel'))
    figstyle.panel_letter(ax, 'f')

    notes = []
    if t11 is not None and 'note' in t11.columns:
        nz = [n for n in t11[t11.family == '5A12'].note.dropna().unique()]
        if nz:
            notes.append('T11 on the designed negative, verbatim: "%s".'
                         % str(nz[0]))
    footer(fig, [
        'Panel (a) plots the POOLED rate $\\sum n_{\\rm cliff}/\\sum n_{\\rm pairs}$ '
        'over the permissible assays in each (Levy class, decile) cell, with '
        r'$\pm$1 binomial SE from those same tabulated counts -- the only '
        'arithmetic in this figure beyond a ratio and a mean.  NS1 bands are '
        'NOT drawn: T09 carries p_NS1 and the burial-matched OR with its CI, '
        'but no NS1 quantiles for the rate itself, and a band that is not a '
        'real null envelope is forbidden (spec Sec.6).',
        'Panel (b): the assay AUROC is the mean over cliff pairs of T10\'s '
        'per-pair AUROC_contribution (the Mann-Whitney identity), annotated '
        'with n(cliff)/n(non-cliff) and the NS2 permutation p; p is at the '
        'B=10,000 floor for every assay.  No bootstrap band is tabulated, so '
        'none is drawn.  Panel (c): black dots mark iface_flag per partner.  '
        'Panel (d): the NS3 draws are not tabulated (only mean and SD), so the '
        'histogram spec Sec.6 asks for is replaced by the null mean $\\pm$1 SD.']
        + notes,
        handles=[Line2D([0], [0], color=c, linestyle=l, marker=m,
                        markersize=2.8, markeredgecolor='#000000',
                        markeredgewidth=0.3, label='(a) ' + k)
                 for k, c, l, m in (
                     ('core', figstyle.OKABE_ITO['vermillion'], '-', 'o'),
                     ('support', figstyle.OKABE_ITO['orange'], (0, (4, 1.5)), 's'),
                     ('rim', figstyle.OKABE_ITO['reddishpurple'], (0, (1, 1.2)), '^'),
                     ('interior', figstyle.OKABE_ITO['blue'],
                      (0, (5, 1.2, 1, 1.2)), 'D'),
                     ('surface', figstyle.OKABE_ITO['skyblue'], (0, (7, 1.5)), 'v'))]
        + [Line2D([0], [0], marker='o', linestyle='none', markersize=4.0,
                  markerfacecolor=figstyle.OKABE_ITO['vermillion'],
                  markeredgecolor='#000000', label=r'(d) $M_{\rm obs}$'),
           Line2D([0], [0], marker='s', linestyle='-', markersize=2.6,
                  markerfacecolor='none', color='#9A9A9A',
                  label=r'(d) NS3 null mean $\pm$1 SD')],
        ncol=4)
    st.paths = figstyle.savefig_both(fig, st.stem, outdir=outdir)
    return st


# --------------------------------------------------------------------------- #
# F6 -- gates and calibration                                                 #
# --------------------------------------------------------------------------- #

def _parse_list(v):
    """T02h stores its per-bin vectors as a bracketed string; read it back
    without eval.  Returns an empty array when the cell is empty."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return np.array([])
    txt = str(v).strip().strip('[]')
    if not txt:
        return np.array([])
    out = []
    for tok in txt.replace(';', ',').split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            return np.array([])
    return np.asarray(out, float)


def _gate_rows(t2, gate_id):
    if t2 is None or 'gate_id' not in t2.columns:
        return pd.DataFrame()
    return t2[t2.gate_id.astype(str) == gate_id]


def _pending(rows):
    if rows.empty:
        return True
    return bool((rows['PASS/FAIL'].astype(str) == 'PENDING').all())


def fig6_gates_and_calibration(outdir=None, verbose=True):
    """F6 -- the figure a hostile referee is shown first."""
    st = FigStatus('F6_gates_and_calibration', 'F6  gates and calibration')
    t1, t2, t2a = load('T01', outdir), load('T02', outdir), load('T02a', outdir)
    t2b, t2h, t2j = load('T02b', outdir), load('T02h', outdir), load('T02j', outdir)
    t7 = load('T07', outdir)
    if t1 is None and t2 is None and t2a is None:
        return _placeholder(st.stem, st.title,
                            ['table not available: T01, T02, T02a'],
                            outdir=outdir, status=st)

    fig, axes = figstyle.figure_grid(2, 3, 132.0, width='double')
    flat = axes.ravel()
    _facts = {}

    # ---- (a) floor masking: |P_a| before / after ---------------------------
    ax = flat[0]
    g5 = _gate_rows(t2, 'G5')
    if have(t1, 'n_nested', 'n_nested_censor_touching'):
        d = t1[(t1.n_nested_censor_touching.fillna(0) > 0)
               & (t1.n_primary_Pa.fillna(0) > 0)].copy()
        d = d.sort_values('n_nested', ascending=False)
        ids = list(d.DMS_id)
        x = np.arange(len(ids), dtype=float)
        before = d.n_nested.values.astype(float)
        after = d.n_primary_Pa.values.astype(float)
        derived = before - d.n_nested_censor_touching.values.astype(float)
        n_agree = int(np.sum(np.isclose(after, derived)))
        _facts['n_agree'], _facts['n_cens'] = n_agree, len(ids)
        w = 0.38
        for k, a in enumerate(ids):
            hi = a == 'CR9114_FluAH3_logKd_4FQY'
            ax.bar(x[k] - w / 2, before[k], w, color='#C8C8C8',
                   edgecolor='#333333', linewidth=(0.9 if hi else 0.4), zorder=2)
            ax.bar(x[k] + w / 2, after[k], w,
                   color=figstyle.OKABE_ITO['vermillion'], alpha=0.9,
                   edgecolor='#333333', linewidth=(0.9 if hi else 0.4), zorder=2)
            fac = before[k] / after[k] if after[k] > 0 else np.nan
            if np.isfinite(fac):
                ax.annotate(r'$\times$%.1f' % fac,
                            xy=(x[k], max(before[k], after[k])), xytext=(0, 2.5),
                            textcoords='offset points', ha='center', va='bottom',
                            fontsize=5.2, fontweight=('bold' if hi else 'normal'))
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels([short_of(a) for a in ids],
                           rotation=30, ha='right', fontsize=5.4)
        ax.set_ylabel(r'$|P_a|$ (nested pairs)')
        ax.set_ylim(top=np.nanmax(before) * 40.0)   # headroom for the G5 block
        figstyle.decision_line(ax, THRESH['G5_Pa_after_expected'])
        ax.annotate('G5 expects %d after masking'
                    % THRESH['G5_Pa_after_expected'],
                    xy=(0.02, THRESH['G5_Pa_after_expected']),
                    xycoords=('axes fraction', 'data'), ha='left', va='bottom',
                    fontsize=5.2, color='#4D4D4D', zorder=6,
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='none',
                              pad=0.4))
        ax.legend(handles=[
            Patch(facecolor='#C8C8C8', edgecolor='#333333', label='before masking'),
            Patch(facecolor=figstyle.OKABE_ITO['vermillion'], alpha=0.9,
                  edgecolor='#333333', label='after (T01.n_primary_Pa)')],
            fontsize=5.2, loc='upper right', bbox_to_anchor=(1.0, 0.72))
        t2g = load('T02g', outdir)
        g5txt, g5col = 'G5 $T(4)$: PENDING', '#8A2A00'
        if have(t2g, 'T4_unmasked_N1'):
            r0 = t2g.iloc[0]
            g5txt = ('G5 (%s): $T(4)$ unmasked/N1 = %.2f (wants $\\geq$%g)\n'
                     '$T(4)$ masked/N2 = %.2f, N2 band %.2f-%.2f\n'
                     'collapse $\\times$%.1f, verdict flips: %s'
                     % (short_of(str(r0.DMS_id)), float(r0.T4_unmasked_N1),
                        THRESH['G5_unmasked_T4_min'], float(r0.T4_masked_N2),
                        float(r0.T4_masked_N2_band_lo),
                        float(r0.T4_masked_N2_band_hi),
                        float(r0.Pa_collapse_factor),
                        str(r0.get('verdict_flips', ''))))
            g5col = '#1A1A1A'
            _facts['g5'] = True
        ax.text(0.06, 0.985, g5txt, transform=ax.transAxes, fontsize=5.0,
                ha='left', va='top', color=g5col, linespacing=1.4, zorder=6,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none',
                          pad=0.5))
        tiny_title(ax, r'floor masking: $|P_a|$ before / after', size=6.8)
        st.live('a', 'censoring collapse of |P_a| for the %d censored assays '
                     'with a pair channel (T01); before - censor_touching == '
                     'n_primary_Pa on %d/%d; the G5 T(4) half is %s'
                % (len(ids), n_agree, len(ids),
                   'live from T02g' if _facts.get('g5') else 'PENDING'))
    else:
        st.awaiting('a', stamp(ax, ('T01', 'T02'),
                               'CR9114-H3 before / after floor masking'))
    figstyle.panel_letter(ax, 'a')

    # ---- (b) cliff rate by density quintile --------------------------------
    ax = flat[1]
    if have(t2h, 'density_rates_tau3'):
        nb = THRESH['C3A_n_density_bins']
        drew, lo_all = 0, []
        for _, r in t2h.iterrows():
            a = str(r.DMS_id)
            ns = _parse_list(r.get('density_n_per_bin', None))
            tot = int(ns.sum()) if len(ns) else 0
            for col, ls, lab in (('density_rates_tau3', '-', r'$\tau$=3'),
                                 ('density_rates_tau4', (0, (4, 1.5)),
                                  r'$\tau$=4')):
                rates = _parse_list(r.get(col, None))
                if not len(rates):
                    continue
                stl = assay_style(a, for_line=False)
                ax.plot(np.arange(1, len(rates) + 1), rates,
                        marker=stl.get('marker', 'o'), linestyle=ls,
                        linewidth=1.0, markersize=4.2,
                        color=stl['color'], markerfacecolor=stl['color'],
                        markeredgecolor=stl['color'], markeredgewidth=0.9,
                        label='%s %s (n=%d)'
                              % (short_of(a), lab, tot))
                lo_all += [v for v in rates if v > 0]
                drew += 1
        ax.set_yscale('log')
        if lo_all:
            ax.set_ylim(min(lo_all) / 6.0, max(lo_all) * 2.2)
        ax.set_xticks(range(1, nb + 1))
        ax.set_xlabel('density quintile (1 = sparsest)')
        ax.set_ylabel('cliff rate')
        ax.legend(fontsize=4.8, loc='lower left', ncol=2, columnspacing=0.8)
        rho = '   '.join('%s $\\rho$=%+.1f'
                         % (short_of(str(r.DMS_id)),
                            float(r.density_spearman_tau3))
                         for _, r in t2h.iterrows()
                         if pd.notna(r.get('density_spearman_tau3', None)))
        ax.text(0.98, 0.985, 'Spearman(rate, quintile)\n' + rho,
                transform=ax.transAxes, ha='right', va='top', fontsize=4.8,
                color='#333333', linespacing=1.4)
        tiny_title(ax, 'cliff rate by density quintile (G6)', size=6.8)
        st.live('b', '%d rate curves over %d density quintiles for the G6 '
                     'anti-smooth controls (T02h)' % (drew, nb))
    elif have(t7, 'density_q1_rate', 'density_q5_rate'):
        d = t7.dropna(subset=['density_q1_rate', 'density_q5_rate'])
        d = d.drop_duplicates('DMS_id')
        for _, r in d.iterrows():
            stl = assay_style(str(r.DMS_id), for_line=False)
            ax.plot([1, THRESH['C3A_n_density_bins']],
                    [float(r.density_q1_rate), float(r.density_q5_rate)],
                    marker=stl.get('marker', 'o'), linestyle='-',
                    linewidth=1.0, markersize=4.0, color=stl['color'],
                    markerfacecolor=stl['color'], markeredgecolor=stl['color'],
                    label=short_of(str(r.DMS_id)))
        ax.set_yscale('log')
        ax.set_xticks(range(1, THRESH['C3A_n_density_bins'] + 1))
        ax.set_xlabel('density quintile (1 = sparsest)')
        ax.set_ylabel('cliff rate')
        ax.legend(fontsize=5.0, loc='best')
        ax.text(0.5, 0.02, 'only quintiles 1 and %d are tabulated in T07'
                % THRESH['C3A_n_density_bins'], transform=ax.transAxes,
                ha='center', va='bottom', fontsize=5.4, color='#8A2A00')
        tiny_title(ax, 'cliff rate by density quintile', size=6.8)
        st.live('b', 'density_q1_rate and density_q5_rate for %d assays (T07); '
                     'only the two extreme quintiles are tabulated' % len(d))
    else:
        st.awaiting('b', stamp(
            ax, ('T02h', 'T07'),
            'Z-LL1 cliff rate by neighbourhood-density quintile -- the '
            'anti-smooth negative control: if the rate tracks density, the '
            'statistic is measuring sampling, not the landscape'))
    figstyle.panel_letter(ax, 'b')

    # ---- (c) G8 injection power -------------------------------------------
    ax = flat[2]
    g8 = _gate_rows(t2, 'G8')
    if have(t2j, 'supported', 'amp', 'pi'):
        keys = ['DMS_id', 'pi', 'amp'] if 'DMS_id' in t2j.columns \
            else ['pi', 'amp']
        g = (t2j.groupby(keys)
             .agg(power=('supported', 'mean'), n=('supported', 'size'),
                  ref=('refuted', 'mean')).reset_index())
        assays = (sorted(set(g.DMS_id)) if 'DMS_id' in g.columns else [None])
        for ai, a in enumerate(assays):
            ga = g if a is None else g[g.DMS_id == a]
            mk = 'o' if a is None else assay_style(a, for_line=False)['marker']
            for i, pi in enumerate(sorted(ga.pi.unique())):
                d = ga[ga.pi == pi].sort_values('amp')
                ax.plot(d.amp.values, d.power.values, marker=mk,
                        linestyle=('-', (0, (4, 1.5)), (0, (1, 1.2)))[i % 3],
                        markersize=3.4, linewidth=1.0,
                        color=figstyle.PALETTE[i % len(figstyle.PALETTE)],
                        markeredgecolor='#000000', markeredgewidth=0.3,
                        label=(r'$\pi$=%g' % pi) if ai == 0 else None)
        figstyle.decision_line(ax, THRESH['G8_power_min'], linestyle='--',
                               color='#5A5A5A', linewidth=0.8)
        ax.annotate('G8 underpower line %.2f' % THRESH['G8_power_min'],
                    xy=(0.02, THRESH['G8_power_min']),
                    xycoords=('axes fraction', 'data'), ha='left',
                    va='bottom', fontsize=5.2, color='#4D4D4D')
        figstyle.decision_line(ax, THRESH['G8_power_ref_amplitude'], orient='v')
        ax.set_ylim(-0.05, 1.10)
        ax.set_xticks(list(THRESH['G8_amplitudes_sigma']))
        ax.set_xlabel(r'injected cliff amplitude $a$ ($\sigma$)')
        ax.set_ylabel('detection power')
        ax.legend(fontsize=5.2, loc='center right', title='injected rate',
                  title_fontsize=5.2)
        ref = g[(g.amp == THRESH['G8_power_ref_amplitude'])
                & (np.isclose(g.pi, THRESH['G8_power_ref_rate']))]
        msg = ''
        if len(ref):
            msg = ('power %s at the reference cell\n'
                   r'($a$=%g$\sigma$, $\pi$=%g, %s reps each)' '\n'
                   'REFUTED in %s of them'
                   % ('/'.join('%.2f' % v for v in ref.power.values),
                      THRESH['G8_power_ref_amplitude'],
                      THRESH['G8_power_ref_rate'],
                      '/'.join('%d' % v for v in ref.n.values),
                      '/'.join('%.0f%%' % (100 * v) for v in ref.ref.values)))
            ax.text(0.04, 0.94, msg + '\n' + r'$\Rightarrow$ UNDERPOWERED',
                    transform=ax.transAxes, ha='left', va='top',
                    fontsize=5.2, color='#8A2A00', linespacing=1.4)
        tiny_title(ax, 'G8 injection power, %s'
                   % (label_of(assays[0]) if len(assays) == 1
                      else '%d assays' % len(assays)), size=6.8)
        st.live('c', 'G8 power = mean(supported) over %d reps per cell, %d '
                     'amplitudes x %d rates x %d of the 6 representative '
                     'assays (T02j); power is %.2f at every cell'
                % (int(g.n.max()), g.amp.nunique(), g.pi.nunique(),
                   len(assays), float(g.power.max())))
    elif not _pending(g8) and 'observed' in g8.columns and g8.observed.notna().any():
        d = g8[g8.observed.notna()]
        vals = pd.to_numeric(d.observed, errors='coerce').dropna()
        ax.plot([THRESH['G8_power_ref_amplitude']] * len(vals), vals.values,
                marker='o', linestyle='none', markersize=4.0,
                markerfacecolor=figstyle.OKABE_ITO['vermillion'],
                markeredgecolor='#000000', markeredgewidth=0.4)
        figstyle.decision_line(ax, THRESH['G8_power_min'], linestyle='--',
                               color='#5A5A5A', linewidth=0.8)
        ax.annotate('G8 underpower line %.2f' % THRESH['G8_power_min'],
                    xy=(0.02, THRESH['G8_power_min']),
                    xycoords=('axes fraction', 'data'), ha='left', va='bottom',
                    fontsize=5.2, color='#4D4D4D')
        ax.set_xticks(list(THRESH['G8_amplitudes_sigma']))
        ax.set_ylim(-0.05, 1.10)
        ax.set_xlabel(r'injected cliff amplitude $a$ ($\sigma$)')
        ax.set_ylabel('detection power')
        ax.text(0.5, 0.5, 'T02 carries only the reference cell\n'
                r'($a$=%g$\sigma$, $\pi$=%g); the amplitude'
                '\nsweep needs T02j'
                % (THRESH['G8_power_ref_amplitude'],
                   THRESH['G8_power_ref_rate']), transform=ax.transAxes,
                ha='center', va='center', fontsize=5.4, color='#8A2A00')
        tiny_title(ax, 'G8 injection power (reference cell only)', size=6.8)
        st.live('c', '%d G8 power value(s) at the reference cell from T02; the '
                     'amplitude sweep needs T02j' % len(vals))
    else:
        st.awaiting('c', stamp(
            ax, 'T02 (G8)', 'detection power vs injected amplitude (%s sigma), '
            'one line per injected rate (%s), per assay, with the %.2f '
            'underpower line'
            % ('/'.join(str(a) for a in THRESH['G8_amplitudes_sigma']),
               '/'.join(str(r) for r in THRESH['G8_rates']),
               THRESH['G8_power_min']),
            extra='T02\'s G8 row exists but is PENDING (stage 4, '
                  'cliff.calibrate).  T02b\'s N2_power column is a DIFFERENT '
                  'quantity -- N2-vs-observed discrimination -- and is not '
                  'substituted here.'))
    figstyle.panel_letter(ax, 'c')

    # ---- (d) G4 uniformity QQ ---------------------------------------------
    ax = flat[3]
    if have(t2a, 'ks_p_rand'):
        d = t2a[t2a.live.astype(str).str.lower().isin(('true', '1'))]
        mins = {}
        for col, cl, mk, lab in (
                ('ks_p_rand', figstyle.OKABE_ITO['blue'], 'o',
                 'randomised (G4 is read off this)'),
                ('ks_p_cons', figstyle.OKABE_ITO['vermillion'], 's',
                 'conservative (tie-inflated)')):
            if col not in d.columns:
                continue
            p = np.sort(d[col].dropna().values.astype(float))
            if p.size == 0:
                continue
            mins[col] = p.min()
            q = (np.arange(1, p.size + 1) - 0.5) / p.size
            ax.plot(q, p, linestyle='none', marker=mk, markersize=2.2,
                    markerfacecolor='none', markeredgecolor=cl,
                    markeredgewidth=0.5, label='%s (n=%d)' % (lab, p.size))
        ax.plot([0, 1], [0, 1], color='#5A5A5A', linestyle='--', linewidth=0.7,
                label='uniform (the diagonal)')
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel('expected uniform quantile')
        ax.set_ylabel(r'G4 KS $p$ (per assay $\times$ statistic)')
        figstyle.decision_line(ax, THRESH['G4_ks_p_min'])
        ax.annotate('G4 floor %g' % THRESH['G4_ks_p_min'],
                    xy=(0.02, THRESH['G4_ks_p_min']),
                    xycoords=('axes fraction', 'data'), ha='left', va='bottom',
                    fontsize=5.2, color='#4D4D4D')
        ax.legend(fontsize=5.0, loc='lower right')
        ax.text(0.03, 0.55, 'min randomised $p$ = %.3f\nmin conservative $p$ = '
                '%.1e' % (mins.get('ks_p_rand', np.nan),
                          mins.get('ks_p_cons', np.nan)),
                transform=ax.transAxes, fontsize=5.2, va='top',
                linespacing=1.4)
        tiny_title(ax, r'G4 uniformity of the null $p$-values', size=6.8)
        st.live('d', 'QQ of %d live per-(assay, statistic) KS p-values, both '
                     'forms (T02a); every randomised p exceeds the %g floor'
                % (len(d), THRESH['G4_ks_p_min']))
    else:
        st.awaiting('d', stamp(ax, 'T02a', 'the G4 uniformity QQ of the null '
                                           '$p$-values'))
    figstyle.panel_letter(ax, 'd')

    # ---- (e) G9 aggregate-rule FPR ----------------------------------------
    ax = flat[4]
    g9 = _gate_rows(t2, 'G9')
    t14 = load('T14', outdir)
    if not _pending(g9) and g9.observed.notna().any():
        vals = pd.to_numeric(g9.observed, errors='coerce').dropna()
        x = np.arange(len(vals), dtype=float)
        ax.bar(x, vals.values, 0.5,
               color=figstyle.OKABE_ITO['vermillion'], alpha=0.9,
               edgecolor='#333333', linewidth=0.4, zorder=2)
        figstyle.decision_line(ax, THRESH['G9_family_fpr_max'], linestyle='--',
                               color='#5A5A5A', linewidth=0.8)
        ax.annotate('G9 ceiling %.2f' % THRESH['G9_family_fpr_max'],
                    xy=(0.02, THRESH['G9_family_fpr_max']),
                    xycoords=('axes fraction', 'data'), ha='left', va='bottom',
                    fontsize=5.2, color='#4D4D4D')
        ax.set_xticks(x)
        ax.set_xticklabels([str(a)[:18] for a in g9.assay.values[:len(vals)]],
                           rotation=30, ha='right', fontsize=5.2)
        ax.set_ylabel('empirical family-level FPR of the k-of-7 rule')
        ax.set_ylim(0, max(float(vals.max()) * 1.4,
                           THRESH['G9_family_fpr_max'] * 1.6))
        ax.text(0.98, 0.96, 'over %d complete N1 datasets'
                % THRESH['G9_n_datasets'], transform=ax.transAxes, ha='right',
                va='top', fontsize=5.4)
        tiny_title(ax, 'G9 aggregate-rule FPR', size=6.8)
        st.live('e', '%d G9 FPR value(s) against the %.2f ceiling (T02)'
                % (len(vals), THRESH['G9_family_fpr_max']))
    else:
        g9_row = ''
        if t14 is not None and 'family_id' in t14.columns:
            q = t14[t14.family_id.astype(str) == 'FOOTER:G9_rule_FPR']
            if len(q):
                g9_row = ('T14\'s FOOTER:G9_rule_FPR row says "%s".'
                          % str(q.iloc[0].get('meta_effect', '')))
        st.awaiting('e', stamp(
            ax, 'T02 (G9)', 'the empirical family-level FPR of the k-of-7 '
            'aggregate rule over %d complete N1 datasets, against its %.2f '
            'ceiling' % (THRESH['G9_n_datasets'], THRESH['G9_family_fpr_max']),
            extra=('T02\'s G9 row is PENDING (stage 4).  %s Until it lands k '
                   'stays at the pre-declared %d-of-7 and no aggregate call is '
                   'readable.' % (g9_row, THRESH['C1_family_k_true']))))
    figstyle.panel_letter(ax, 'e')

    # ---- (f) the gate ledger ----------------------------------------------
    n_pass = 0 if t2 is None else int((t2['PASS/FAIL'] == 'PASS').sum())
    n_pend = 0 if t2 is None else int((t2['PASS/FAIL'] == 'PENDING').sum())
    n_fail = 0 if t2 is None else int((t2['PASS/FAIL'] == 'FAIL').sum())
    ledger = ['T02_gates.csv: %d PASS, %d FAIL, %d PENDING.'
              % (n_pass, n_fail, n_pend), '']
    if t2 is not None:
        pend = (t2[t2['PASS/FAIL'] == 'PENDING']
                .groupby('gate_id').size().sort_index())
        ledger.append('still PENDING, by gate:')
        ledger += ['   %-6s %d row%s' % (g, n, '' if n == 1 else 's')
                   for g, n in pend.items()]
    ledger += ['', 'A PENDING gate is not a passed gate.']
    notes_axes(flat[5], 'F6  gate ledger', ledger, fontsize=5.8)
    st.note('f', 'live', 'gate ledger cell')

    footer(fig, [
        'Panel (a) plots two tabulated pair counts and nothing derived: '
        'T01.n_nested before masking against T01.n_primary_Pa after it, for '
        'every assay that has both a floor and a pair channel.  On %d of the '
        '%d, n_nested $-$ n_nested_censor_touching lands exactly on '
        'n_primary_Pa; on the rest n_primary_Pa is smaller still, because it '
        'also drops pairs excluded for reasons other than censoring -- which '
        'is why the bars are the two counts themselves and not their '
        'difference.  G5 also requires unmasked $T(4)\\geq$%g and a '
        '$\\geq$%gx collapse; the $T(4)$ half is PENDING, so the panel shows '
        'the collapse; the $T(4)$ numbers themselves are printed on the panel '
        'from T02g_G5_censoring.csv, which is not in the Sec.6 table list.'
        % (_facts.get('n_agree', 0), _facts.get('n_cens', 0),
           THRESH['G5_unmasked_T4_min'], THRESH['G5_Pa_collapse_factor']),
        'Panel (c): power is the fraction of the %d injected-surrogate '
        'replicates per (assay, amplitude, rate) cell that the pipeline stamps '
        'SUPPORTED, aggregated from T02j\'s per-replicate rows; one line per '
        'injected rate, one marker shape per assay.  Panel (d) reads G4 off '
        'the RANDOMISED empirical '
        'p-value; the conservative form is plotted beside it because ties make '
        'it reject uniformity on assays whose statistic is discrete -- that is '
        'a property of the tie handling, not evidence that the surrogate '
        'machinery is biased, and showing only the favourable form would hide '
        'it.  Points ABOVE the diagonal are conservative, not anti-conservative.'
        % THRESH['G8_n_reps']])
    st.paths = figstyle.savefig_both(fig, st.stem, outdir=outdir)
    return st


# --------------------------------------------------------------------------- #
# F7 -- the cliff blind spot                                                  #
# --------------------------------------------------------------------------- #

_MODEL_MARK = {'M1_additive_isotonic': ('o', 'M1 additive+isotonic'),
               'M2_physchem': ('s', 'M2 phys-chem'),
               'M3_msa_site_indep': ('^', 'M3 MSA site-independent')}
F7_TAU = config.TAU_WINDOW[0]          # spec Sec.6 does not name one; see docstring


def fig7_cliff_blind_spot(outdir=None, verbose=True):
    """F7 -- a model can rank the whole assay well and still be at chance on the
    cliff edges.  Scatter of per-assay Spearman against PSA on cliff edges."""
    st = FigStatus('F7_cliff_blind_spot', 'F7  the cliff blind spot')
    t12 = load('T12', outdir)
    if not have(t12, 'PSA_cliff', 'spearman_all_rows'):
        return _placeholder(
            st.stem, st.title,
            ['table not available: T12',
             'F7 needs T12_cliff_aware_eval.csv (PSA_cliff with its CI, '
             'PSA_noncliff, AUPSA and spearman_all_rows for M1-M3).'],
            outdir=outdir, status=st)

    d = t12[t12.tau == F7_TAU].copy()
    fig, axes = figstyle.figure_grid(
        2, 1, 152.0, width='double',
        gridspec_kw=dict(height_ratios=(1.0, 0.78)))

    # ---- (a) Spearman vs PSA_cliff ----------------------------------------
    ax = axes[0]
    decision_region(ax, 0.0, THRESH['C5_PSA_blindspot'],
                    label='blind-spot region (decision, not a null)')
    figstyle.decision_line(ax, 0.5, linestyle='--', color='#5A5A5A',
                           linewidth=0.7, label='0.5 chance')
    figstyle.decision_line(ax, THRESH['C5_PSA_blindspot'],
                           label='blind spot %.2f' % THRESH['C5_PSA_blindspot'])
    figstyle.decision_line(ax, THRESH['C5_PSA_practically_empty'],
                           label='practical emptiness %.2f'
                                 % THRESH['C5_PSA_practically_empty'])
    figstyle.decision_line(ax, THRESH['C5_spearman_min'], orient='v')
    ax.annotate('C5 needs $\\rho\\geq$%.2f' % THRESH['C5_spearman_min'],
                xy=(THRESH['C5_spearman_min'], 0.012),
                xycoords=('data', 'axes fraction'), ha='left', va='bottom',
                fontsize=5.4, color='#4D4D4D', zorder=6,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none',
                          pad=0.4))
    n_pt = 0
    for _, r in d.iterrows():
        if not (np.isfinite(r.PSA_cliff) and np.isfinite(r.spearman_all_rows)):
            continue
        mk = _MODEL_MARK.get(str(r.model), ('P', str(r.model)))[0]
        stl = assay_style(r.DMS_id, for_line=False)
        lo, hi = r.get('PSA_lo95', np.nan), r.get('PSA_hi95', np.nan)
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([r.spearman_all_rows] * 2, [lo, hi], color=stl['color'],
                    linewidth=0.5, alpha=0.75, zorder=2, solid_capstyle='butt')
        ax.plot([r.spearman_all_rows], [r.PSA_cliff], marker=mk, markersize=4.0,
                linestyle='none', markerfacecolor=stl['color'],
                markeredgecolor=stl.get('markeredgecolor', stl['color']),
                markeredgewidth=0.5, zorder=4)
        n_pt += 1
    ax.set_xlabel(r'per-assay Spearman $\rho$ over all rows (T12)')
    ax.set_ylabel(r'PSA on cliff edges, $\tau=%d$' % F7_TAU)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(-0.62, 1.02)
    ax.legend(handles=[Line2D([0], [0], marker=m, linestyle='none',
                              markersize=4.0, color='#444444', label=lab)
                       for m, lab in _MODEL_MARK.values()]
              + figstyle.family_legend_handles(
                  [f for f in config.FAMILIES
                   if any(a in set(d.DMS_id) for a in config.FAMILIES[f])]),
              ncol=4, fontsize=5.8, loc='lower left')
    tiny_title(ax, 'whole-assay ranking says nothing about the cliff edges',
               size=7.4)
    st.live('a', '%d (assay, model) points at tau=%d with 95%% PSA CIs (T12)'
            % (n_pt, F7_TAU))
    figstyle.panel_letter(ax, 'a')

    # ---- (b) PSA on cliff vs non-cliff edges ------------------------------
    ax = axes[1]
    ids = [a for a in assay_order(('PRIMARY', 'ARM')) if a in set(d.DMS_id)]
    models = [m for m in _MODEL_MARK if m in set(d.model)]
    x = np.arange(len(ids), dtype=float)
    w = 0.8 / max(len(models), 1)
    for mi, m in enumerate(models):
        dm = d[d.model == m].set_index('DMS_id')
        off = (mi - (len(models) - 1) / 2.0) * w
        for k, a in enumerate(ids):
            if a not in dm.index:
                continue
            r = dm.loc[a]
            stl = assay_style(a, for_line=False)
            if np.isfinite(r.PSA_cliff):
                ax.bar(x[k] + off, r.PSA_cliff, w * 0.88, color=stl['color'],
                       alpha=(0.95 - 0.22 * mi), edgecolor='#333333',
                       linewidth=0.35, zorder=2)
            if np.isfinite(r.PSA_noncliff):
                ax.plot([x[k] + off - w * 0.44, x[k] + off + w * 0.44],
                        [r.PSA_noncliff] * 2, color='#1A1A1A', linewidth=0.9,
                        zorder=4, solid_capstyle='butt')
            if np.isfinite(r.get('AUPSA', np.nan)):
                ax.plot([x[k] + off], [r.AUPSA], marker='D', markersize=2.2,
                        markerfacecolor='none', markeredgecolor='#1A1A1A',
                        markeredgewidth=0.5, linestyle='none', zorder=5)
    figstyle.decision_line(ax, 0.5, linestyle='--', color='#5A5A5A',
                           linewidth=0.7)
    figstyle.decision_line(ax, THRESH['C5_PSA_blindspot'])
    ax.set_xticks(x)
    ax.set_xticklabels([short_of(a) for a in ids], rotation=30,
                       ha='right', fontsize=5.6)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel(r'PSA at $\tau=%d$' % F7_TAU)
    ax.legend(handles=[
        Patch(facecolor='#8A8A8A', edgecolor='#333333', label='bar: PSA on cliff edges'),
        Line2D([0], [0], color='#1A1A1A', linewidth=0.9,
               label='tick: PSA on non-cliff edges'),
        Line2D([0], [0], marker='D', linestyle='none', markersize=2.2,
               markerfacecolor='none', markeredgecolor='#1A1A1A',
               label='AUPSA (not split by cliff status in T12)'),
        Line2D([0], [0], color='#5A5A5A', linestyle='--', linewidth=0.7,
               label='0.5 chance'),
    ], ncol=4, fontsize=5.8, loc='lower left')
    tiny_title(ax, 'PSA on cliff edges (bars) vs non-cliff edges (ticks); '
                   'bar shade M1 dark $\\rightarrow$ M3 light', size=7.4)
    st.live('b', '%d assays x %d models, PSA_cliff bars with PSA_noncliff ticks'
            % (len(ids), len(models)))
    figstyle.panel_letter(ax, 'b')
    footer(fig, [
        '$\\tau=%d$: the C2 window\'s lower edge and the C3-N cliff '
        'definition.  Spec Sec.6 does not name a $\\tau$ and T12 carries '
        '%s; the same figure at another $\\tau$ is one argument away.  '
        'Error bars in (a) are the tabulated 95%% CI of PSA on cliff edges '
        '(T12.PSA_lo95 / PSA_hi95).'
        % (F7_TAU, ', '.join(str(t) for t in sorted(t12.tau.unique()))),
        'Spec Sec.6 asks for "AUPSA on cliff vs non-cliff edges": T12\'s AUPSA '
        'is ONE number per (assay, model) and is not split by cliff status '
        '(it is also constant in $\\tau$), so the split drawn is T12\'s own '
        'PSA_cliff vs PSA_noncliff and AUPSA is marked separately as a '
        'diamond.  T12\'s verdict_blindspot and verdict_practical_emptiness '
        'columns are empty, so the three horizontal calls are the frozen '
        'THRESH lines and are NOT read from the table.'])
    st.paths = figstyle.savefig_both(fig, st.stem, outdir=outdir)
    return st


# --------------------------------------------------------------------------- #
# stage 8 entry point                                                         #
# --------------------------------------------------------------------------- #

FIGURES = (
    ('F1', fig1_variogram_panel),
    ('F2', fig2_tail_survival_vs_nulls),
    ('F3', fig3_enrichment_sweep),
    ('F4', fig4_localisation),
    ('F5', fig5_structure),
    ('F6', fig6_gates_and_calibration),
    ('F7', fig7_cliff_blind_spot),
)


def stage8(assays=None, nproc=1, verbose=True, outdir=None, only=None):
    """Render F1..F7 into ``artifacts/`` as both .pdf and 600 dpi .png.

    Never raises: a figure that fails is reported and replaced by a placeholder
    page, because stage 8 also writes the verdict tables and must not be taken
    down by a plotting bug.
    """
    config.assert_env()
    figstyle.apply()
    _CACHE.clear()
    out = artifacts_dir(outdir)
    os.makedirs(out, exist_ok=True)
    if verbose:
        print('[figures] artifacts dir: %s' % out)
        for name in sorted(TABLES):
            df = load(name, outdir)
            print('[figures]   %-5s %s' % (name, ('%5d x %2d' % df.shape)
                                           if df is not None else 'ABSENT'))
    results = []
    for tag, fn in FIGURES:
        if only and tag not in only:
            continue
        try:
            st = fn(outdir=outdir, verbose=verbose)
        except Exception:                                          # never crash
            print('[figures] %s FAILED:\n%s' % (tag, traceback.format_exc()))
            st = _placeholder('%s_render_failed' % tag,
                              '%s could not be rendered' % tag,
                              ['see the stage log for the traceback'],
                              outdir=outdir)
        results.append((tag, st))
        if verbose:
            print('[figures] %s -> %s   (%d/%d panels live)'
                  % (tag, ', '.join(os.path.basename(p) for p in st.paths),
                     st.n_live, len(st.panels)))
            for letter, state, detail in st.panels:
                print('[figures]     (%s) %-8s %s' % (letter, state, detail))
    if verbose:
        print('[figures] %d figures, %d/%d panels live'
              % (len(results), sum(s.n_live for _, s in results),
                 sum(len(s.panels) for _, s in results)))
    _ = (assays, nproc)
    return results


run_all = stage8
run = stage8


# --------------------------------------------------------------------------- #
# self-check                                                                  #
# --------------------------------------------------------------------------- #

def _selfcheck():
    """Render all seven from the tables that exist, then assert the Sec.6
    properties on the FILES (both extensions, non-trivial size) and print the
    live/awaiting matrix that the record's Sec.6 has to carry."""
    import matplotlib
    import warnings
    print('[figures] env %s   matplotlib %s   backend %s'
          % (config.assert_env()[0], matplotlib.__version__,
             matplotlib.get_backend()))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = stage8(verbose=True)
    # a collapsed constrained_layout means the panels were silently laid out
    # by hand-me-down defaults -- it has to fail the self-check, not scroll by
    collapsed = sorted({str(w.message) for w in caught
                        if 'constrained_layout' in str(w.message)})
    for m in collapsed:
        print('  !! LAYOUT: %s' % m)
    print('')
    print('%-4s %-42s %-7s %s' % ('fig', 'file', 'panels', 'live / awaiting'))
    ok = True
    for tag, st in res:
        for ext in ('pdf', 'png'):
            p = os.path.join(PATHS.artifacts, '%s.%s' % (st.stem, ext))
            if not os.path.exists(p) or os.path.getsize(p) < 3000:
                print('  !! %s missing or suspiciously small' % p)
                ok = False
        live = [l for l, s, _ in st.panels if s == 'live']
        wait = [l for l, s, _ in st.panels if s != 'live']
        print('%-4s %-42s %d/%-5d live=%s  awaiting=%s'
              % (tag, st.stem + '.{pdf,png}', st.n_live, len(st.panels),
                 ''.join(live) or '-', ''.join(wait) or '-'))
    sizes = {}
    for _, st in res:
        for p in st.paths:
            sizes[os.path.basename(p)] = os.path.getsize(p) / 1e3
    print('')
    for k in sorted(sizes):
        print('  %-46s %8.1f kB' % (k, sizes[k]))
    assert ok, 'a figure did not write both extensions'
    assert not collapsed, 'constrained_layout collapsed: %s' % collapsed
    print('[figures] no constrained_layout collapse, both extensions written '
          'for all %d figures' % len(res))
    print('[figures] SELF-CHECK PASSED')
    return res


if __name__ == '__main__':
    _selfcheck()
