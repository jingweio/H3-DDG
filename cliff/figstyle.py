"""BGYM-CLIFF v1 -- figure style.  Spec Sec.6 "Figures -- publication-legible".

DO NOT CHANGE: the 16 rcParams in ``RC_PARAMS`` (they are spec Sec.6 verbatim), the
89/183 mm widths, or ``FAMILY_COLOR`` -- one colour per family, identical across all
seven figures, is what makes F1..F7 readable as one set; re-rolling it silently
re-colours every published panel.  No seaborn, ever.
"""
from __future__ import annotations

import os

from cliff import config

# --------------------------------------------------------------------------- #
# rcParams -- spec Sec.6, verbatim and complete                               #
# --------------------------------------------------------------------------- #

#: Spec Sec.6: "Explicit rcParams, no seaborn: savefig.dpi=600, pdf.fonttype=42,
#: font.family='sans-serif', font.sans-serif=['Helvetica','Arial','DejaVu Sans'],
#: font.size=8, axes.labelsize=8, xtick.labelsize=7, ytick.labelsize=7,
#: legend.fontsize=7, axes.titlesize=9, axes.linewidth=0.6,
#: xtick.major.width=0.6, lines.linewidth=1.1, axes.spines.top=False,
#: axes.spines.right=False, figure.constrained_layout.use=True".
#: Transcribed one-for-one; nothing added, nothing dropped.
RC_PARAMS = {
    'savefig.dpi': 600,
    'pdf.fonttype': 42,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.titlesize': 9,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'lines.linewidth': 1.1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.constrained_layout.use': True,
}

#: Consequences of RC_PARAMS that the spec's prose implies but its list omits.
#: Kept SEPARATE so ``RC_PARAMS`` stays checkable against Sec.6 character by
#: character.  "0.6 pt axes" with only ``xtick.major.width`` set leaves the y
#: ticks at matplotlib's 0.8 pt, i.e. visibly heavier than the spine they sit on;
#: the four minor widths follow the same 0.6/2 convention.  Suppress with
#: ``apply(extras=False)``.
RC_EXTRA = {
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    'savefig.bbox': None,            # constrained_layout owns the margins
    'savefig.transparent': False,
    'figure.dpi': 150,               # on-screen only; savefig.dpi is what ships
    'legend.frameon': False,
    'axes.grid': False,
    'errorbar.capsize': 1.5,
}

_APPLIED = {'done': False, 'extras': None}


def apply(extras=True, force=False):
    """Install :data:`RC_PARAMS` (and, unless ``extras=False``, :data:`RC_EXTRA`).

    Idempotent.  Selects the ``Agg`` backend when there is no display, so a
    figure module is importable over ssh without ``MPLBACKEND`` set.
    """
    import matplotlib
    if not os.environ.get('DISPLAY') and not os.environ.get('MPLBACKEND'):
        if matplotlib.get_backend().lower() not in ('agg', 'pdf', 'svg', 'ps'):
            matplotlib.use('Agg')
    if _APPLIED['done'] and _APPLIED['extras'] == bool(extras) and not force:
        return matplotlib.rcParams
    matplotlib.rcParams.update(RC_PARAMS)
    if extras:
        matplotlib.rcParams.update(RC_EXTRA)
    assert 'seaborn' not in ','.join(matplotlib.rcParams.get('font.sans-serif', []))
    _APPLIED['done'] = True
    _APPLIED['extras'] = bool(extras)
    return matplotlib.rcParams


# --------------------------------------------------------------------------- #
# Widths -- spec Sec.6 "Widths exactly 89 mm (single) / 183 mm (double)"       #
# --------------------------------------------------------------------------- #

MM_PER_IN = 25.4
WIDTH_SINGLE_MM = 89.0
WIDTH_DOUBLE_MM = 183.0


def mm_to_in(mm):
    return float(mm) / MM_PER_IN


def figsize(width_mm, height_mm):
    """(width, height) in inches from millimetres."""
    return (mm_to_in(width_mm), mm_to_in(height_mm))


def figure_single(height_mm, **kw):
    """A single-column figure: exactly 89 mm wide."""
    import matplotlib.pyplot as plt
    apply()
    return plt.subplots(figsize=figsize(WIDTH_SINGLE_MM, height_mm), **kw)


def figure_double(height_mm, **kw):
    """A double-column figure: exactly 183 mm wide."""
    import matplotlib.pyplot as plt
    apply()
    return plt.subplots(figsize=figsize(WIDTH_DOUBLE_MM, height_mm), **kw)


def figure_grid(nrows, ncols, height_mm, width='double', **kw):
    """A panel grid at one of the two permitted widths."""
    import matplotlib.pyplot as plt
    apply()
    w = WIDTH_DOUBLE_MM if width == 'double' else WIDTH_SINGLE_MM
    return plt.subplots(nrows, ncols, figsize=figsize(w, height_mm), **kw)


# --------------------------------------------------------------------------- #
# Palette -- Okabe-Ito 8 colour, colour-blind safe                            #
# --------------------------------------------------------------------------- #

#: Okabe & Ito (2008), the canonical eight.  Names kept so a reader can check
#: them against the published set rather than against a hex string.
OKABE_ITO = {
    'black': '#000000',
    'orange': '#E69F00',
    'skyblue': '#56B4E9',
    'bluishgreen': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'reddishpurple': '#CC79A7',
}
PALETTE = tuple(OKABE_ITO[k] for k in
                ('blue', 'vermillion', 'bluishgreen', 'reddishpurple',
                 'orange', 'skyblue', 'yellow', 'black'))

#: FIXED family -> colour map (spec Sec.6: "one colour per family, consistent
#: across all seven figures").  F1..F7 take the seven chromatic Okabe-Ito
#: colours in the order the families are numbered in Sec.2; F8 -- the hypercube
#: arm, which is always drawn on its own denominator -- takes black, so that a
#: reader can never mistake it for one of the seven primary families.
FAMILY_COLOR = {
    'F1': OKABE_ITO['blue'],           # GB1
    'F2': OKABE_ITO['vermillion'],     # KRAS
    'F3': OKABE_ITO['bluishgreen'],    # SARS2-RBD
    'F4': OKABE_ITO['reddishpurple'],  # 5A12
    'F5': OKABE_ITO['orange'],         # Z-ZpA963
    'F6': OKABE_ITO['skyblue'],        # hYAP65
    'F7': OKABE_ITO['yellow'],         # CD19
    'F8': OKABE_ITO['black'],          # hypercube ARM, separate denominator
}
assert set(FAMILY_COLOR) == set(config.FAMILIES), 'FAMILY_COLOR != config.FAMILIES'
assert len(set(FAMILY_COLOR.values())) == len(FAMILY_COLOR)

#: Okabe-Ito yellow is the one member of the set with poor contrast on white, so
#: any *marker* or *fill* in that colour is drawn with this edge.  Lines are
#: unaffected at 1.1 pt.
FAMILY_EDGE = {k: ('#7F6A00' if v == OKABE_ITO['yellow'] else v)
               for k, v in FAMILY_COLOR.items()}

#: Assays that are not data points (CONTROL / EXCLUDED).  Spec F1 draws them as
#: "a separately outlined control row"; black + dashed keeps them on the same
#: axes as the primary families without stealing a family colour.
CONTROL_STYLE = dict(color=OKABE_ITO['black'], linestyle='--', linewidth=0.9)

#: Null envelopes.  Spec F1 names the two: "N1 ribbon (grey), N2 ribbon (blue)".
#: N2b / N2c / N3 / NS* continue the same logic (a null is never a family colour).
NULL_STYLE = {
    'N1': dict(facecolor='#9A9A9A', alpha=0.32, edgecolor='none'),
    'N2': dict(facecolor=OKABE_ITO['blue'], alpha=0.22, edgecolor='none'),
    'N2b': dict(facecolor=OKABE_ITO['bluishgreen'], alpha=0.20, edgecolor='none'),
    'N2c': dict(facecolor=OKABE_ITO['vermillion'], alpha=0.20, edgecolor='none'),
    'N3': dict(facecolor='#D0D0D0', alpha=0.45, edgecolor='none'),
    'NS1': dict(facecolor='#9A9A9A', alpha=0.30, edgecolor='none'),
    'NS2': dict(facecolor='#9A9A9A', alpha=0.30, edgecolor='none'),
    'NS3': dict(facecolor='#9A9A9A', alpha=0.30, edgecolor='none'),
}

#: Spec Sec.6: "Every shaded band is a real null envelope (5th-95th percentile),
#: never a decorative ribbon."  The percentiles live here so a band cannot be
#: drawn at a different, undeclared coverage by accident.
NULL_BAND_PCTILES = (5.0, 95.0)


def color_of(dms_id):
    """Family colour for an assay; black for anything without a family (the
    CONTROL and EXCLUDED tiers carry ``family_id == ''``)."""
    fam = config.ASSAYS[dms_id].family_id
    return FAMILY_COLOR.get(fam, OKABE_ITO['black'])


def style_of(dms_id):
    """Line kwargs for an assay: family colour for a data point, the dashed
    control style for a CONTROL / EXCLUDED assay."""
    spec = config.ASSAYS[dms_id]
    if spec.tier in ('PRIMARY', 'ARM'):
        return dict(color=FAMILY_COLOR[spec.family_id])
    return dict(CONTROL_STYLE)


def family_legend_handles(families=None):
    """Proxy handles for a family legend, in F1..F8 order."""
    from matplotlib.lines import Line2D
    apply()
    fams = list(config.FAMILIES) if families is None else list(families)
    out = []
    for f in fams:
        members = config.FAMILIES[f]
        label = '%s (%s%s)' % (f, members[0].split('_')[0],
                               ', +%d' % (len(members) - 1) if len(members) > 1 else '')
        out.append(Line2D([0], [0], color=FAMILY_COLOR[f], label=label))
    return out


# --------------------------------------------------------------------------- #
# Panel letters -- spec Sec.6 "bold 9 pt at axes-fraction (0.01, 0.99)"        #
# --------------------------------------------------------------------------- #

PANEL_LETTER_XY = (0.01, 0.99)
PANEL_LETTER_SIZE = 9
PANEL_LETTER_WEIGHT = 'bold'


def panel_letter(ax, letter, **kw):
    """Bold 9 pt panel letter at axes-fraction (0.01, 0.99) -- spec Sec.6."""
    apply()
    kwargs = dict(transform=ax.transAxes, ha='left', va='top',
                  fontsize=PANEL_LETTER_SIZE, fontweight=PANEL_LETTER_WEIGHT)
    kwargs.update(kw)
    return ax.text(PANEL_LETTER_XY[0], PANEL_LETTER_XY[1], letter, **kwargs)


def panel_letters(axes, letters=None, **kw):
    """Label a flat sequence of axes a, b, c, ..."""
    import string
    axs = list(getattr(axes, 'ravel', lambda: axes)())
    lets = list(letters) if letters is not None else list(string.ascii_lowercase)
    return [panel_letter(a, l, **kw) for a, l in zip(axs, lets)]


# --------------------------------------------------------------------------- #
# Bands and saving                                                            #
# --------------------------------------------------------------------------- #

def null_band(ax, x, lo, hi, null, label=None, **kw):
    """Shade a REAL null envelope.

    ``null`` must name one of the study's nulls (:data:`NULL_STYLE`).  There is
    deliberately no way to shade an unnamed region through this helper -- spec
    Sec.6 forbids decorative ribbons, and a band whose null cannot be named is
    decorative by definition.
    """
    apply()
    if null not in NULL_STYLE:
        raise ValueError('null must be one of %s (spec Sec.6: every shaded band '
                         'is a real null envelope); got %r'
                         % (sorted(NULL_STYLE), null))
    st = dict(NULL_STYLE[null])
    st.update(kw)
    if label is None:
        label = '%s %g-%gth pct' % (null, NULL_BAND_PCTILES[0], NULL_BAND_PCTILES[1])
    return ax.fill_between(x, lo, hi, label=label, **st)


def decision_line(ax, value, orient='h', label=None, **kw):
    """A frozen decision boundary from ``config.THRESH``, drawn the same way in
    every figure: thin, dotted, neutral grey, annotated with its value."""
    apply()
    st = dict(color='#4D4D4D', linestyle=':', linewidth=0.6, zorder=0)
    st.update(kw)
    fn = ax.axhline if orient == 'h' else ax.axvline
    ln = fn(value, **st)
    if label:
        if orient == 'h':
            ax.annotate(label, xy=(1.0, value), xycoords=('axes fraction', 'data'),
                        ha='right', va='bottom', fontsize=6, color=st['color'])
        else:
            ax.annotate(label, xy=(value, 1.0), xycoords=('data', 'axes fraction'),
                        ha='left', va='top', fontsize=6, color=st['color'])
    return ln


def savefig_both(fig, stem, outdir=None, close=True):
    """Spec Sec.6: "Both .pdf (vector) and .png (600 dpi) written."

    ``stem`` carries no extension (e.g. ``'F1_variogram_panel'``); the default
    output directory is the deliverables ``artifacts/``.
    """
    apply()
    d = config.PATHS.artifacts if outdir is None else outdir
    os.makedirs(d, exist_ok=True)
    paths = []
    for ext in ('pdf', 'png'):
        p = os.path.join(d, '%s.%s' % (stem, ext))
        fig.savefig(p, dpi=RC_PARAMS['savefig.dpi'])
        paths.append(p)
    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)
    return paths


# --------------------------------------------------------------------------- #
# self-check                                                                  #
# --------------------------------------------------------------------------- #

def _selfcheck():
    """Renders a real 2-panel figure with the real family colours and asserts
    every Sec.6 property on the resulting objects (not on the dict)."""
    import matplotlib
    import numpy as np
    config.assert_env()
    rc = apply()
    print('[figstyle] matplotlib %s   backend %s'
          % (matplotlib.__version__, matplotlib.get_backend()))
    for k, v in RC_PARAMS.items():
        got = rc[k]
        # matplotlib normalises a scalar font.family to a one-element list
        if isinstance(got, list) and not isinstance(v, list) and len(got) == 1:
            got = got[0]
        got = list(got) if isinstance(got, list) else got
        assert got == v, 'rcParam %s = %r, spec says %r' % (k, got, v)
    print('[figstyle] all %d spec Sec.6 rcParams installed and verified'
          % len(RC_PARAMS))
    print('[figstyle] savefig.dpi=%d  pdf.fonttype=%d  font.size=%g/%g/%g pt '
          '(base/tick/title)  axes.linewidth=%g  spines top/right=%s/%s  '
          'constrained_layout=%s'
          % (rc['savefig.dpi'], rc['pdf.fonttype'], rc['font.size'],
             rc['xtick.labelsize'], rc['axes.titlesize'], rc['axes.linewidth'],
             rc['axes.spines.top'], rc['axes.spines.right'],
             rc['figure.constrained_layout.use']))

    fig, axes = figure_grid(1, 2, 55.0, width='double')
    w_in, h_in = fig.get_size_inches()
    print('[figstyle] figure_double  = %.4f x %.4f in = %.2f x %.2f mm  '
          '(spec: %.0f mm wide)' % (w_in, h_in, w_in * MM_PER_IN,
                                    h_in * MM_PER_IN, WIDTH_DOUBLE_MM))
    assert abs(w_in * MM_PER_IN - WIDTH_DOUBLE_MM) < 1e-9
    fs, _ = figure_single(45.0)
    print('[figstyle] figure_single  = %.4f in = %.2f mm  (spec: %.0f mm)'
          % (fs.get_size_inches()[0], fs.get_size_inches()[0] * MM_PER_IN,
             WIDTH_SINGLE_MM))
    assert abs(fs.get_size_inches()[0] * MM_PER_IN - WIDTH_SINGLE_MM) < 1e-9
    import matplotlib.pyplot as plt
    plt.close(fs)

    x = np.arange(1, 7)
    rng = np.random.default_rng(config.SEED_BASE)
    for f in config.FAMILIES:
        axes[0].plot(x, 1.0 / x + 0.02 * rng.standard_normal(x.size),
                     color=FAMILY_COLOR[f], label=f)
    null_band(axes[0], x, 0.9 / x, 1.15 / x, 'N1')
    null_band(axes[0], x, 0.95 / x, 1.05 / x, 'N2')
    decision_line(axes[0], config.THRESH['C1_V1_over_Vinf_sup'],
                  label='%g' % config.THRESH['C1_V1_over_Vinf_sup'])
    axes[0].set(xlabel='Hamming distance $h$', ylabel=r'$V(h)/V(\infty)$',
                yscale='log', title='family colours')
    for dms_id in ('GB1_IgG-Fc_fitness_1FCC', 'Z-domain_ZSPA-1_LL1_fitness_1LP1'):
        axes[1].plot(x, np.linspace(0.3, 1.4, x.size), **style_of(dms_id))
    axes[1].set(xlabel=r'$\tau$', ylabel=r'$T(\tau)$', title='control styling')
    lets = panel_letters(axes)
    for t in lets:
        assert t.get_position() == PANEL_LETTER_XY
        assert t.get_fontsize() == PANEL_LETTER_SIZE
        assert t.get_fontweight() == PANEL_LETTER_WEIGHT
    print('[figstyle] panel letters %s bold %d pt at axes-fraction %s  OK'
          % ([t.get_text() for t in lets], PANEL_LETTER_SIZE, (PANEL_LETTER_XY,)))
    for ax in axes:
        assert not ax.spines['top'].get_visible()
        assert not ax.spines['right'].get_visible()
        assert abs(ax.spines['left'].get_linewidth() - RC_PARAMS['axes.linewidth']) < 1e-12
    print('[figstyle] top/right spines off, left/bottom at %g pt  OK'
          % RC_PARAMS['axes.linewidth'])

    try:
        null_band(axes[0], x, 0.0 * x, 1.0 + 0.0 * x, 'decorative')
    except ValueError as exc:
        print('[figstyle] an unnamed band is refused: %s' % str(exc)[:72])
    else:
        raise AssertionError('null_band accepted a nameless band')

    out = os.path.join(config.REPO, 'data', 'cliff_cache', '_figstyle_selfcheck')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    paths = savefig_both(fig, '_figstyle_selfcheck',
                         outdir=os.path.join(config.REPO, 'data', 'cliff_cache'))
    for p in paths:
        print('[figstyle] wrote %s  (%.1f kB)' % (p, os.path.getsize(p) / 1e3))
        os.remove(p)
    print('[figstyle] palette = Okabe-Ito %d colours, %d distinct family colours, '
          'no seaborn imported' % (len(OKABE_ITO), len(set(FAMILY_COLOR.values()))))
    import sys
    assert not any(m.startswith('seaborn') for m in sys.modules), 'seaborn imported!'
    print('[figstyle] FAMILY_COLOR = %s'
          % {k: v for k, v in sorted(FAMILY_COLOR.items())})
    print('[figstyle] SELF-CHECK PASSED')
    _ = out


if __name__ == '__main__':
    _selfcheck()
