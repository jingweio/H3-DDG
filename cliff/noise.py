# -*- coding: utf-8 -*-
r"""``cliff.noise`` -- the sigma registry with provenance (spec Sec.1.0 / Sec.3).

Spec Sec.1.0's table, reproduced here as a runnable object rather than prose:

===================  =============  ==========  ==========================  ==========================
family               sigma_y        sigma_eps   provenance                  source
===================  =============  ==========  ==========================  ==========================
KRAS (5 assays)      0.148          **0.1243**  ``measured_replicate``      KRAS_RAF1_6VJJ vs
                                                (upper bound)               RAF1-RBD_6VJJ, 10,868
                                                                            shared site-pairs,
                                                                            r = 0.812, OLS slope
                                                                            0.6445 removed by affine
                                                                            alignment, residual sd
                                                                            0.1479 -> 0.1479/
                                                                            sqrt(1+0.6445^2)
GB1 (2 assays)       0.129          --          ``cross_study_contaminated``  160 shared variants,
                                                                            r = 0.9789, residual sd
                                                                            0.1826; chain-C pos 2 WT
                                                                            differs (Q vs T)
all others           ``sigma(phi)`` --          ``internal_residual``       the assay's own
                                                                            cross-fitted residual MAD
                                                                            per phi-decile
cross-check, all     0.20 x MAD     --          ``stipulated``              imported from the GB1
                                                                            ratio; never primary
===================  =============  ==========  ==========================  ==========================

**FORBIDDEN**: the four Z-domain within-genotype SDs (0.1695 / 0.4105 / 0.3064 /
0.5091).  G3 proves them a chain-key collision artefact -- with the chain label
retained all four Z-domain assays have 0 duplicate genotypes; the "replicates"
appear only when the chain is dropped from the key.  :func:`check_sigma_not_forbidden`
is the guard, and there is no code path in this module that can return one.

Every number below is **measured from the data at import-free call time**, never
copied from the spec.  ``config.NOISE`` holds the spec's expectations so that
:func:`kras_twin_epsilon` and :func:`gb1_cross_study` can be checked against
them; the registry ships the measurement.

Self-check::

    python -m cliff.noise      # measures both anchors, writes T03, prints the diffs
"""

import os
import time

import numpy as np
import pandas as pd

from . import config
from . import io_bgym
from . import latent
from .config import PATHS, SIGMA_MULTIPLIERS, THRESH

__all__ = [
    'T03_COLUMNS', 'PROVENANCE', 'epsilon_sitepairs', 'kras_twin_epsilon',
    'gb1_cross_study', 'internal_residual_sigma', 'sigma_registry',
    'sigma_sensitivity_grid', 'forbidden_zdomain_sds',
    'check_sigma_not_forbidden', 'write_T03', 'stage2',
    'wildtype_sequences',
    'KRAS_TWIN', 'GB1_TWIN', 'LATENT_EXTRA',
]

#: spec Sec.6, verbatim and in order.
T03_COLUMNS = [
    'DMS_id', 'sigma_y', 'sigma_eps', 'provenance', 'source_partner',
    'n_source', 'r_source', 'slope_source', 'resid_sd_source', 'sigma_over_mad',
    'caveat', 'upstream_SE_obtained', 'verdict_stamp',
]

#: the four provenance values, spec Sec.6's enum.
PROVENANCE = ('measured_replicate', 'cross_study_contaminated',
              'internal_residual', 'stipulated')

#: the sigma_eps anchor (spec Sec.1.0).
KRAS_TWIN = ('KRAS_RAF1_norfitness_6VJJ', 'KRAS_RAF1-RBD_norfitness_6VJJ')

#: the sigma_y anchor (spec Sec.1.0) -- contaminated, see :func:`gb1_cross_study`.
GB1_TWIN = ('GB1_IgG-Fc_fitness_1FCC', 'GB1_IgG-Fc_fitness_1FCC_2016')

#: EXCLUDED-tier assays that still carry a downstream role (C4-I probes, the
#: cluster channel) and are NOT saturated, so an ``internal_residual`` sigma
#: exists for them.  ADDITION beyond the spec's stage-2 list of 17: without it
#: five T03 rows would be empty for assays the study actually reads.
LATENT_EXTRA = ('4D5_HER2_fitness_1N8Z', '5A12_Ang2_fitness_4ZFG',
                'BH3_Bcl-xL_normed_1PQ1', 'BH3_Mcl-1_normed_3KZ0',
                'Z-domain_ZpA963_HL2_fitness_2M5A')


# --------------------------------------------------------------------------- #
# epsilon = the double-mutant interaction, with the EXACT additive baseline    #
# --------------------------------------------------------------------------- #

def epsilon_sitepairs(assay):
    """``eps_st = y(st) - y(s) - y(t) + y(WT)`` for every double whose two
    singles are observed (spec Sec.1.4 C3-N).

    Keyed by the double's **canonical key** -- the ordered pair of
    ``(chain, seq_pos, aa_mut)`` substitutions -- which is what the spec's phrase
    "shared site-pairs" means operationally: the KRAS twin shares **10,868** of
    these, exactly the spec's count, against only **602** shared *site* pairs
    once the amino acids are collapsed.  So a "site-pair" in the spec's noise
    table is a substitution pair, and this function is keyed accordingly.

    Returns ``dict(keys, eps, rows, n_doubles, n_usable, reason)``; ``n_usable``
    is 0 with a stated ``reason`` when the assay has no WT row (Z-LL1, Z-LL2,
    HLA-A2) or no doubles, rather than raising.
    """
    if assay.wt_row < 0:
        return dict(keys=(), eps=np.zeros(0), rows=np.zeros(0, dtype=np.int64),
                    n_doubles=0, n_usable=0,
                    reason='no WT row: the additive baseline y(WT) does not exist')
    y = assay.y
    ywt = float(y[assay.wt_row])
    single = {}
    for i, k in enumerate(assay.keys):
        if len(k) == 1:
            single[k[0]] = i
    keys, eps, rows = [], [], []
    n_doubles = 0
    for i, k in enumerate(assay.keys):
        if len(k) != 2:
            continue
        n_doubles += 1
        rs = single.get(k[0])
        rt = single.get(k[1])
        if rs is None or rt is None:
            continue
        keys.append(k)
        eps.append(y[i] - y[rs] - y[rt] + ywt)
        rows.append(i)
    reason = ''
    if not keys:
        reason = ('no double whose two singles are both observed (%d doubles, '
                  '%d singles)' % (n_doubles, len(single)))
    return dict(keys=tuple(keys), eps=np.asarray(eps, dtype=np.float64),
                rows=np.asarray(rows, dtype=np.int64), n_doubles=n_doubles,
                n_usable=len(keys), reason=reason)


def _affine_join(x, y):
    """OLS ``y ~ a + b x`` on a shared join; returns the diagnostics the spec's
    noise table quotes.

    ``sigma`` deconvolves the affine alignment: if both measurements carry the
    same noise sd then ``Var(resid) = sigma^2 (1 + slope^2)``, so
    ``sigma = resid_sd / sqrt(1 + slope^2)`` -- the spec's own arithmetic,
    ``0.1479 / sqrt(1 + 0.6445^2) = 0.1243``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    out = dict(n=int(n))
    if n < 3 or x.std() == 0 or y.std() == 0:
        out.update(pearson=float('nan'), spearman=float('nan'),
                   slope=float('nan'), intercept=float('nan'),
                   resid_sd=float('nan'), resid_sd_ddof0=float('nan'),
                   sigma=float('nan'), sd_a=float(x.std(ddof=1)) if n > 1 else 0.0,
                   sd_b=float(y.std(ddof=1)) if n > 1 else 0.0)
        return out
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)
    out.update(
        pearson=float(np.corrcoef(x, y)[0, 1]),
        spearman=io_bgym._spearman(x, y),
        slope=float(slope), intercept=float(intercept),
        resid_sd=float(resid.std(ddof=2)),
        resid_sd_ddof0=float(resid.std(ddof=0)),
        sigma=float(resid.std(ddof=2) / np.sqrt(1.0 + slope * slope)),
        sd_a=float(x.std(ddof=1)), sd_b=float(y.std(ddof=1)))
    return out


# --------------------------------------------------------------------------- #
# anchor 1: the KRAS twin -- the only measured sigma_eps in the benchmark      #
# --------------------------------------------------------------------------- #

def kras_twin_epsilon(pair=KRAS_TWIN, *, assays=None):
    """``sigma_eps`` from the twice-measured KRAS interactions (spec Sec.1.0).

    KRAS_RAF1_6VJJ and KRAS_RAF1-RBD_6VJJ score two different constructs of the
    same interaction over two different libraries (63 vs 166 mutated positions).
    Every double that appears in both, with its two singles observed in both,
    gives ``eps`` twice.  Regressing one on the other and deconvolving the slope
    gives ``sigma_eps``.

    **This is an upper bound, and contaminated.**  The two files differ in
    construct (full RAF1 vs the RBD alone) and in library, so the residual
    contains a real construct effect as well as measurement noise; the affine
    alignment is only valid if that construct effect is affine in ``eps``.
    Spec Sec.8.1 requires this to be said wherever the number is used, so the
    caveat travels in the returned dict and into T03.
    """
    a_id, b_id = pair
    A = (assays or {}).get(a_id) or io_bgym.load_assay(a_id)
    B = (assays or {}).get(b_id) or io_bgym.load_assay(b_id)
    ea, eb = epsilon_sitepairs(A), epsilon_sitepairs(B)
    ma = dict(zip(ea['keys'], ea['eps']))
    mb = dict(zip(eb['keys'], eb['eps']))
    shared = sorted(set(ma) & set(mb))
    x = np.array([ma[k] for k in shared], dtype=np.float64)
    y = np.array([mb[k] for k in shared], dtype=np.float64)
    fit = _affine_join(x, y)
    # collapse the amino acids: how many SITE pairs is that really?
    def _sites(k):
        return tuple(sorted((c, p) for c, p, _ in k))
    n_site_pairs = len(set(_sites(k) for k in shared))
    # the y-level join over the shared canonical keys, for comparison
    ka = {k: i for i, k in enumerate(A.keys)}
    kb = {k: i for i, k in enumerate(B.keys)}
    sk = sorted(set(ka) & set(kb))
    yfit = _affine_join(A.y[[ka[k] for k in sk]], B.y[[kb[k] for k in sk]])
    sig = fit['sigma']
    mult = THRESH['C3N_cliff_sigma_mult']
    table = pd.DataFrame(dict(
        key=['|'.join('%s%d%s' % (c, p, aa) for c, p, aa in k) for k in shared],
        eps_a=x, eps_b=y,
        cliff_a=np.abs(x) >= mult * sig, cliff_b=np.abs(y) >= mult * sig,
        sign_match=np.sign(x) == np.sign(y)))
    return dict(
        assay_a=a_id, assay_b=b_id,
        relation='same_interaction_diff_study',
        join_method='canonical_key',
        n_eps_a=ea['n_usable'], n_eps_b=eb['n_usable'],
        n_shared=fit['n'], n_shared_site_pairs_collapsed=n_site_pairs,
        r=fit['pearson'], spearman=fit['spearman'],
        slope=fit['slope'], intercept=fit['intercept'],
        resid_sd=fit['resid_sd'], resid_sd_ddof0=fit['resid_sd_ddof0'],
        sigma_eps=sig, sigma_eps_sq=sig * sig,
        sd_eps_a=fit['sd_a'], sd_eps_b=fit['sd_b'],
        n_cliff_a_3sigma=int((np.abs(x) >= mult * sig).sum()),
        cliff_abs_3sigma=mult * sig,
        y_join=yfit,
        caveat=config.NOISE['KRAS']['caveat'],
        table=table)


# --------------------------------------------------------------------------- #
# anchor 2: the GB1 cross-study overlap -- contaminated by a Q2T background    #
# --------------------------------------------------------------------------- #

def wildtype_sequences(dms_id):
    """The per-chain ``wildtype_sequence`` dict of one file.

    Read separately from :func:`cliff.io_bgym.load_assay`, whose pinned
    ``usecols`` deliberately excludes this column.  The stage-0 audit already
    established it is constant within every file, so row 0 is enough.
    """
    import ast
    df = pd.read_csv(PATHS.dms_csv(dms_id), usecols=['wildtype_sequence'],
                     nrows=1)
    return ast.literal_eval(df['wildtype_sequence'].iloc[0])


def gb1_cross_study(pair=GB1_TWIN, *, assays=None):
    """The GB1 overlap, and the reason it is NOT a replicate (spec Sec.1.0).

    160 canonical keys appear in both files.  Their scores correlate at
    r = 0.9789 with a residual sd of 0.1826, which *looks* like a measurement
    replicate -- but the two files' ``wildtype_sequence`` differ: chain C
    position 2 is **Q** in the 55-site library and **T** in the 4-site library.
    Two backgrounds, not two measurements, so the number is stamped
    ``cross_study_contaminated`` and never becomes a primary noise floor.  The
    Q-vs-T claim is *verified here from the files*, not asserted.
    """
    a_id, b_id = pair
    A = (assays or {}).get(a_id) or io_bgym.load_assay(a_id)
    B = (assays or {}).get(b_id) or io_bgym.load_assay(b_id)
    ka = {k: i for i, k in enumerate(A.keys)}
    kb = {k: i for i, k in enumerate(B.keys)}
    shared = sorted(set(ka) & set(kb))
    x = A.y[[ka[k] for k in shared]]
    y = B.y[[kb[k] for k in shared]]
    fit = _affine_join(x, y)
    # ---- the background check ---- #
    wa, wb = wildtype_sequences(a_id), wildtype_sequences(b_id)
    diffs = []
    for ch in sorted(set(wa) & set(wb)):
        sa, sb = wa[ch], wb[ch]
        for i, (ca, cb) in enumerate(zip(sa, sb)):
            if ca != cb:
                diffs.append((ch, i + 1, ca, cb))
        if len(sa) != len(sb):
            diffs.append((ch, -1, 'len%d' % len(sa), 'len%d' % len(sb)))
    mutated_b = set((c, p) for k in B.keys for c, p, _ in k)
    order_hist = {}
    for k in shared:
        order_hist[len(k)] = order_hist.get(len(k), 0) + 1
    return dict(
        assay_a=a_id, assay_b=b_id,
        relation='same_interaction_diff_study', join_method='canonical_key',
        n_shared=fit['n'], order_hist=order_hist,
        r=fit['pearson'], spearman=fit['spearman'], slope=fit['slope'],
        intercept=fit['intercept'], resid_sd=fit['resid_sd'],
        resid_sd_ddof0=fit['resid_sd_ddof0'],
        sigma_y=fit['sigma'],
        sigma_y_unit_slope=fit['resid_sd'] / np.sqrt(2.0),
        sd_a=fit['sd_a'], sd_b=fit['sd_b'],
        wt_seq_differences=tuple(diffs),
        two_backgrounds=bool(diffs),
        diff_positions_are_mutated_in_b=tuple(
            (c, p) in mutated_b for c, p, _, _ in diffs),
        caveat=config.NOISE['GB1']['caveat'])


# --------------------------------------------------------------------------- #
# the internal_residual class: every other assay's own sigma-hat(phi)          #
# --------------------------------------------------------------------------- #

def internal_residual_sigma(dms_id, *, fit_if_missing=False, verbose=False):
    """Median ``sigma-hat(phi)`` from the cached cross-fit (spec's
    ``internal_residual`` row: "the assay's own cross-fitted residual MAD per
    phi-decile; conservative -- attributes all epistasis to noise").

    Returns ``sigma`` = the median over the 20 equal-count ``phi`` bins, with
    ``sigma_min`` / ``sigma_max`` so the caveat can state the spread (measured on
    GB1_IgG-Fc_1FCC: 0.096 to 2.35, a 24x range -- which is the whole reason the
    spec makes sigma a function of phi rather than a scalar).

    A saturated design has no internal residual scale at all: with
    ``max_mut = 1`` the additive fit is exact (``M == n - 1`` when a WT row is
    present, ``M == n`` when not), ``e == 0`` identically, and any "sigma"
    computed from it would be pure float noise.  Those assays get
    ``sigma = nan`` and a stated reason.
    """
    p = os.path.join(PATHS.latent, dms_id + '.npz')
    ents = []
    if not os.path.exists(p) and fit_if_missing:
        a = io_bgym.load_assay(dms_id, keep_score_strings=True)
        if a.M >= a.n - 1:
            return dict(sigma=float('nan'), n=a.n, mad_ref=float('nan'),
                        mad_scale='none', manifest=[], reason=(
                            'saturated additive design (n=%d, M=%d): e == 0 by '
                            'construction, no internal residual scale exists'
                            % (a.n, a.M)))
        out = latent.run_latent(dms_id, assay=a, write=True, verbose=verbose)
        ents = out['manifest']
    if not os.path.exists(p):
        return dict(sigma=float('nan'), n=0, mad_ref=float('nan'),
                    mad_scale='none', manifest=[],
                    reason='no latent cross-fit cached for this assay')
    z = np.load(p, allow_pickle=False)
    s = z['sigma_knots_sigma']
    k = z['sigma_knots_count']
    cm = np.load(os.path.join(PATHS.keys, dms_id + '.npz'),
                 allow_pickle=False)['censor_mask']
    return dict(sigma=float(np.median(s)), sigma_min=float(s.min()),
                sigma_max=float(s.max()), n=int(k.sum()), n_bins=int(s.size),
                mad_ref=latent.mad_scaled(z['z'][~cm]), mad_scale='latent z',
                manifest=ents, reason='')


# --------------------------------------------------------------------------- #
# FORBIDDEN                                                                   #
# --------------------------------------------------------------------------- #

def forbidden_zdomain_sds():
    """The four Z-domain within-genotype SDs that must NEVER be a noise floor.

    G3 (measured, stage 0): with the chain label retained the four Z-domain
    assays have **0 / 0 / 0 / 0** duplicate genotypes over 45,476 / 5,583 /
    2,904 / 600 rows; dropping the chain from the mutation token manufactures
    **847 / 59 / 650 / 38**.  The "replicates" are chain-key collisions.  Under
    PDB numbering the ZpA963 collisions vanish entirely (847 / 59 / **0 / 0**),
    because 2M5A numbers chain B from 131 -- for that pair the collision is
    purely a sequence-numbering artefact.
    """
    return dict(config.FORBIDDEN_ZDOMAIN_SDS)


def check_sigma_not_forbidden(dms_id, sigma, *, rtol=1e-3):
    """Raise if ``sigma`` is one of the forbidden Z-domain SDs.

    Cheap insurance: the four values are distinctive, and any code path that
    reaches one of them has re-derived a within-genotype SD from a chain-less
    key.
    """
    if sigma is None or not np.isfinite(sigma):
        return False
    for k, v in config.FORBIDDEN_ZDOMAIN_SDS.items():
        if abs(float(sigma) - v) <= rtol * v:
            raise ValueError(
                'FORBIDDEN noise floor: sigma=%.4f for %s matches the %s '
                'within-genotype SD %.4f, which G3 proves is a chain-key '
                'collision artefact (spec Sec.1.0). BANNED: %s'
                % (sigma, dms_id, k, v, config.BANNED[3]))
    return False


# --------------------------------------------------------------------------- #
# the registry (T03)                                                          #
# --------------------------------------------------------------------------- #

def _family_class(dms_id):
    """Which provenance class an assay's PRIMARY row belongs to."""
    poi = config.ASSAYS[dms_id].poi
    if dms_id in GB1_TWIN:
        return 'cross_study_contaminated'
    # every KRAS file, including the excluded DARPinK27 whose score table is
    # byte-identical to SOS1's (G2), sits on the measured sigma_eps
    if poi.startswith('KRAS') or dms_id.startswith('KRAS_'):
        return 'measured_replicate'
    return 'internal_residual'


def sigma_registry(*, assays=None, kras=None, gb1=None, fit_if_missing=True,
                   verbose=False):
    """T03: one row per (assay, provenance).

    Two rows per assay -- its primary provenance class, and the ``stipulated``
    cross-check ``sigma = 0.20 x MAD-scale`` that spec Sec.1.0 requires to
    accompany every headline number.  56 rows over the 28 files.  Primary key
    ``(DMS_id, provenance)``.
    """
    ids = assays or config.ALL_ASSAYS
    if kras is None:
        kras = kras_twin_epsilon()
    if gb1 is None:
        gb1 = gb1_cross_study()
    # the GB1 ratio the stipulated 0.20 is "imported from" -- computed on GB1's
    # own MAD, not on whichever assay the loop is currently on
    _gb1_a = io_bgym.load_assay(gb1['assay_a'])
    _gb1_mad = latent.mad_scaled(_gb1_a.y[~_gb1_a.censor_mask])
    gb1_ratio = gb1['sigma_y'] / _gb1_mad if _gb1_mad > 0 else float('nan')
    rows, ents = [], []
    for d in ids:
        a = io_bgym.load_assay(d, keep_score_strings=True)
        mad_y = latent.mad_scaled(a.y[~a.censor_mask])
        cls = _family_class(d)
        base = dict(DMS_id=d, upstream_SE_obtained=False,
                    verdict_stamp='conditional')
        if cls == 'measured_replicate':
            extra = ('' if d in ('KRAS_RAF1_norfitness_6VJJ',
                                 'KRAS_RAF1-RBD_norfitness_6VJJ')
                     else ' sigma imported from the RAF1/RAF1-RBD twin, not '
                          'measured on this file.')
            if d == 'KRAS_DARPinK27_norfitness_5O2S':
                extra += (' EXCLUDED: its score table is byte-identical to '
                          'KRAS_SOS1_8BE4 (G2, 19,227 shared keys, max|delta| '
                          '= 0 on the raw strings); retained for structure only.')
            r = dict(base, sigma_y=kras['resid_sd'], sigma_eps=kras['sigma_eps'],
                     provenance=cls,
                     source_partner='%s vs %s' % (kras['assay_a'], kras['assay_b']),
                     n_source=kras['n_shared'], r_source=kras['r'],
                     slope_source=kras['slope'],
                     resid_sd_source=kras['resid_sd'],
                     sigma_over_mad=(kras['sigma_eps'] / mad_y if mad_y > 0
                                     else float('nan')),
                     caveat=('UPPER BOUND. ' + kras['caveat'] +
                             '. sigma_y is the eps-level residual sd (the '
                             'spec\'s 0.148); the y-level join over %d shared '
                             'canonical keys gives resid_sd %.4f and a '
                             'deconvolved %.4f.' % (kras['y_join']['n'],
                                                    kras['y_join']['resid_sd'],
                                                    kras['y_join']['sigma'])
                             + extra))
        elif cls == 'cross_study_contaminated':
            r = dict(base, sigma_y=gb1['sigma_y'], sigma_eps=float('nan'),
                     provenance=cls,
                     source_partner='%s vs %s' % (gb1['assay_a'], gb1['assay_b']),
                     n_source=gb1['n_shared'], r_source=gb1['r'],
                     slope_source=gb1['slope'],
                     resid_sd_source=gb1['resid_sd'],
                     sigma_over_mad=(gb1['sigma_y'] / mad_y if mad_y > 0
                                     else float('nan')),
                     caveat=('NOT a replicate. ' + gb1['caveat'] +
                             '; verified here: wildtype_sequence differs at %s.'
                             % (', '.join('%s%d %s->%s' % t
                                          for t in gb1['wt_seq_differences'])
                                or 'NO position (claim NOT reproduced)')))
        else:
            ir = internal_residual_sigma(d, fit_if_missing=fit_if_missing,
                                         verbose=verbose)
            ents.extend(ir.get('manifest') or [])
            cav = config.NOISE['other']['caveat']
            if ir['reason']:
                cav += '. ' + ir['reason']
            else:
                cav += ('. median over %d equal-count phi bins; range %.4f-%.4f '
                        '(%.1fx), so a scalar sigma is a summary only. '
                        'sigma_over_mad is against 1.4826*MAD(%s) = %.4f -- '
                        'sigma lives on the LATENT scale here, y does not'
                        % (ir['n_bins'], ir['sigma_min'], ir['sigma_max'],
                           ir['sigma_max'] / max(ir['sigma_min'], 1e-12),
                           ir['mad_scale'], ir['mad_ref']))
            mref = ir['mad_ref']
            r = dict(base, sigma_y=ir['sigma'], sigma_eps=float('nan'),
                     provenance=cls, source_partner=d,
                     n_source=ir['n'], r_source=float('nan'),
                     slope_source=float('nan'), resid_sd_source=float('nan'),
                     sigma_over_mad=(ir['sigma'] / mref
                                     if (np.isfinite(mref) and mref > 0
                                         and np.isfinite(ir['sigma']))
                                     else float('nan')),
                     caveat=cav)
        check_sigma_not_forbidden(d, r['sigma_y'])
        check_sigma_not_forbidden(d, r['sigma_eps'])
        rows.append(r)
        # ---- the stipulated cross-check row, for every assay ---- #
        sm = config.NOISE['stipulated']['sigma_over_mad']
        rows.append(dict(
            base, sigma_y=sm * mad_y, sigma_eps=float('nan'),
            provenance='stipulated',
            source_partner='%s vs %s' % (gb1['assay_a'], gb1['assay_b']),
            n_source=gb1['n_shared'], r_source=float('nan'),
            slope_source=float('nan'), resid_sd_source=float('nan'),
            sigma_over_mad=sm,
            caveat=(config.NOISE['stipulated']['caveat'] +
                    '; sigma = %.2f * 1.4826*MAD(y) = %.4f. NOTE the ratio '
                    'actually measured on %s is %.4f, so the stipulated 0.20 is '
                    '%.2fx that -- conservative.'
                    % (sm, sm * mad_y, gb1['assay_a'], gb1_ratio,
                       sm / gb1_ratio))))
    # sweep the whole latent cache: the five LATENT_EXTRA fits happen on demand
    # here, and a per-call append would leave them unregistered
    latent.register_latent_cache()
    df = pd.DataFrame(rows)[T03_COLUMNS]
    assert set(df['provenance']) <= set(PROVENANCE), sorted(set(df['provenance']))
    return df


def sigma_sensitivity_grid(registry=None):
    """``sigma x {0.5, 1, 2}`` -- spec Sec.1.0 requires every headline number to
    be recomputed on this grid (it lands in T13's ``sigma_mult`` knob)."""
    df = registry if registry is not None else sigma_registry()
    out = []
    for _, r in df.iterrows():
        for m in SIGMA_MULTIPLIERS:
            out.append(dict(DMS_id=r['DMS_id'], provenance=r['provenance'],
                            multiplier=m,
                            sigma_y=m * r['sigma_y'] if pd.notna(r['sigma_y'])
                            else float('nan'),
                            sigma_eps=m * r['sigma_eps']
                            if pd.notna(r['sigma_eps']) else float('nan')))
    return pd.DataFrame(out)


def write_T03(df=None, **kw):
    """Write ``local-records/bindingGYM-cliff/artifacts/T03_noise_registry.csv``."""
    PATHS.ensure_cache_dirs()
    if df is None:
        df = sigma_registry(**kw)
    p = os.path.join(PATHS.artifacts, 'T03_noise_registry.csv')
    df.to_csv(p, index=False, float_format='%.6g')
    return p


def stage2(assays=None, verbose=True):
    """spec Sec.5 stage 2, this module's half: measure both noise anchors, build
    the registry and write ``T03_noise_registry.csv``.

    The name ``run_all.py``'s stage table looks for first.  ``assays`` is
    accepted for driver compatibility but T03 is **always** emitted over all 28
    files: spec Sec.6 keys it on ``DMS_id`` with no tier restriction, and eight
    of the eleven EXCLUDED files carry a downstream role (the BH3 and PSD95
    partner-specificity probes, 5A12_Ang2, KRAS_DARPinK27's structural entry,
    4D5's cluster channel) whose sigma the study reads.
    """
    if assays is not None and set(assays) != set(config.ALL_ASSAYS):
        print('[noise] stage2: T03 covers all %d files by spec Sec.6; the %d '
              'assays passed by the driver do not restrict it.'
              % (len(config.ALL_ASSAYS), len(assays)))
    df = sigma_registry(verbose=verbose)
    p = write_T03(df)
    if verbose:
        print('[noise] stage2: %d rows -> %s' % (len(df), p))
    return df


# --------------------------------------------------------------------------- #
# self-check                                                                  #
# --------------------------------------------------------------------------- #

def _cmp(name, got, want, tol):
    ok = (want is None) or (np.isfinite(got) and abs(got - want) <= tol)
    print('   %-34s observed %-12s spec %-10s %s'
          % (name, ('%.4f' % got) if np.isfinite(got) else 'nan',
             '--' if want is None else '%.4f' % want,
             'MATCH' if ok else '*** DIFFERS ***'))
    return ok


def _selfcheck():
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 60)
    pd.set_option('display.max_colwidth', 60)
    print('[noise] env %r' % (config.assert_env(),))
    t0 = time.time()

    print('\n[noise] ANCHOR 1 -- KRAS twin epsilon (the only measured sigma_eps)')
    k = kras_twin_epsilon()
    ok = True
    ok &= _cmp('n shared substitution-pairs', float(k['n_shared']),
               float(config.NOISE['KRAS']['n_source']), 0.5)
    ok &= _cmp('pearson r', k['r'], config.NOISE['KRAS']['r_source'], 5e-4)
    ok &= _cmp('OLS slope', k['slope'], config.NOISE['KRAS']['slope_source'], 5e-5)
    ok &= _cmp('residual sd', k['resid_sd'],
               config.NOISE['KRAS']['resid_sd_source'], 5e-5)
    ok &= _cmp('sigma_eps', k['sigma_eps'],
               config.NOISE['KRAS']['sigma_eps'], 5e-5)
    ok &= _cmp('sigma_eps^2 (THRESH C4I)', k['sigma_eps_sq'],
               THRESH['C4I_sigma_eps_sq'], 5e-6)
    ok &= _cmp('3*sigma_eps (THRESH C3N)', k['cliff_abs_3sigma'],
               THRESH['C3N_cliff_abs_kras'], 5e-4)
    print('   eps available: %s %d / %s %d ; site pairs represented among the shared substitution-pairs: %d'
          % (k['assay_a'], k['n_eps_a'], k['assay_b'], k['n_eps_b'],
             k['n_shared_site_pairs_collapsed']))
    print('   sd(eps_a)=%.4f sd(eps_b)=%.4f ; n cliff_a at 3 sigma_eps = %d (%.2f%%)'
          % (k['sd_eps_a'], k['sd_eps_b'], k['n_cliff_a_3sigma'],
             100.0 * k['n_cliff_a_3sigma'] / k['n_shared']))
    print('   y-level join for comparison: n=%d r=%.4f slope=%.4f resid_sd=%.4f '
          '-> deconvolved %.4f'
          % (k['y_join']['n'], k['y_join']['pearson'], k['y_join']['slope'],
             k['y_join']['resid_sd'], k['y_join']['sigma']))
    print('   sqrt(3)*sigma_y(y-join) = %.4f vs sigma_eps = %.4f  '
          '(eps of a double is a 4-term sum, so these should agree; they do NOT)'
          % (np.sqrt(3.0) * k['y_join']['sigma'], k['sigma_eps']))

    print('\n[noise] ANCHOR 2 -- GB1 cross-study overlap (contaminated)')
    g = gb1_cross_study()
    ok &= _cmp('n shared canonical keys', float(g['n_shared']),
               float(config.NOISE['GB1']['n_source']), 0.5)
    ok &= _cmp('pearson r', g['r'], config.NOISE['GB1']['r_source'], 5e-5)
    ok &= _cmp('residual sd', g['resid_sd'],
               config.NOISE['GB1']['resid_sd_source'], 5e-5)
    ok &= _cmp('sigma_y (measured slope %.4f)' % g['slope'], g['sigma_y'],
               config.NOISE['GB1']['sigma_y'], 1e-3)
    ok &= _cmp('sigma_y (slope forced to 1)', g['sigma_y_unit_slope'],
               config.NOISE['GB1']['sigma_y'], 5e-4)
    print('   shared order histogram: %s' % g['order_hist'])
    print('   wildtype_sequence differences: %s   two_backgrounds=%s'
          % (g['wt_seq_differences'], g['two_backgrounds']))
    print('   ... and is that position mutated in the 4-site library? %s'
          % (g['diff_positions_are_mutated_in_b'],))
    assert g['two_backgrounds'], 'the Q-vs-T background claim did NOT reproduce'
    assert g['wt_seq_differences'] == (('C', 2, 'Q', 'T'),), \
        g['wt_seq_differences']

    print('\n[noise] FORBIDDEN Z-domain within-genotype SDs: %s'
          % forbidden_zdomain_sds())
    nraise = 0
    for d, v in forbidden_zdomain_sds().items():
        try:
            check_sigma_not_forbidden(d, v)
        except ValueError:
            nraise += 1
    print('   check_sigma_not_forbidden rejected %d/4 of them; '
          'accepted sigma_eps=%.4f  OK' % (nraise, k['sigma_eps']))
    assert nraise == 4

    print('\n[noise] T03 registry')
    df = sigma_registry(kras=k, gb1=g)
    p = write_T03(df)
    show = df[df['provenance'] != 'stipulated'][
        ['DMS_id', 'sigma_y', 'sigma_eps', 'provenance', 'n_source',
         'sigma_over_mad']]
    print(show.to_string(index=False, float_format=lambda v: '%.4f' % v))
    print()
    print('   provenance counts: %s'
          % dict(df['provenance'].value_counts().sort_index()))
    print('   rows=%d (28 assays x 2) ; columns match spec Sec.6: %s'
          % (len(df), list(df.columns) == T03_COLUMNS))
    print('   -> %s' % p)
    gr = sigma_sensitivity_grid(df)
    print('   sigma x %s sensitivity grid: %d rows'
          % (list(SIGMA_MULTIPLIERS), len(gr)))
    from . import pairs as _pairs
    bad = _pairs.verify_manifest()
    print('   verify_manifest(): %d mismatches' % len(bad))
    assert not bad, bad
    print('\n[noise] all spec anchors matched: %s   (%.1f s)' % (ok, time.time() - t0))
    assert ok, 'a spec anchor did not reproduce -- see *** DIFFERS *** above'
    return df


if __name__ == '__main__':
    _selfcheck()
