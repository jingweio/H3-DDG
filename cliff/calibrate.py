"""G5-G10: the gates that decide whether an observed number may be read at all.

Spec Sec.1.1.  ``cliff.nulls`` already owns G4 (self-calibration), the N2c
kurtosis audit and the N2-power audit; this module owns the rest and imports
:mod:`cliff.nulls` rather than reimplementing any of it (``_refit_bundle``,
``_pa_mask``, ``observe``, ``run_ensemble``, ``two_point_scale_mixture`` and the
statistic vector are all reused verbatim, private names included -- a second
copy of the observation model is exactly the kind of drift G4 exists to catch).

    G5  censoring positive control      CR9114_FluAH3_4FQY          STOP gate
    G6  anti-smooth negative control    Z-ZSPA1-LL1 / LL2           STOP gate
    G7  scale-mixture discrimination    all 17, N2c surrogates      sets the C2 rule
    G8  power and bias                  6 assays x 4 a x 3 pi x 40  UNDERPOWERED stamp
    G9  aggregate-rule FPR              50 complete N1 datasets     calibrates k
    G10 censoring-mask composition      all 17, N1 + N2             flags an assay

**What this module found that the spec did not predict** (every one measured
before any verdict was read, and all of them in the direction that makes the
study claim LESS, not more):

1. ``T(tau) < 1`` IS STRUCTURAL, NOT A CENSORING ARTEFACT.  ``T(tau) =
   P_obs(|c| >= tau) / mean_b P_N2,b(|c| >= tau)`` sits BELOW 1 on essentially
   every assay, censored or not, because ``c_hat`` is a NESTED difference: ``e_u``
   and ``e_v`` share the background ``B``, so ``Var(e_v - e_u) < 2 Var(e)`` in the
   data, while N2's within-stratum exchange makes the two endpoints independent
   and hence widens the null numerator.  Measured on zero-censoring assays too
   (:func:`t_n2_structural_audit`).  Consequence: G5's clause (ii) "``T(4)``
   inside the N2 95% band" is failed on the LOW side by the positive control,
   and would be failed the same way by a clean assay -- so it is scored
   one-sided (the gate's own consequence text is "the pipeline cannot tell a
   detection limit from a cliff", which is an ENRICHMENT risk) and the literal
   two-sided reading is reported beside it as its own non-halting row.
2. G5's clause (i), "unmasked ``T(4)`` >= 5", is only a statement about the
   pipeline if "unmasked" means the pipeline WITHOUT the censoring machinery --
   no floor mask AND no Tobit E-step ("after floor masking + Tobit" implies
   both are absent before).  Referenced to that naive pipeline's own N1 (a
   smooth surrogate with no detection limit, since a censoring-blind pipeline
   has no clamp to replay) the gate passes enormously.  Referenced to N2 it
   cannot pass, for reason 1.
3. N2c AS THE SPEC DEFINES IT IS A NO-OP ON 16 OF 17 ASSAYS (``cliff.nulls``
   measured this first; T02e).  Reading G7 off that would license "``C2`` alone
   is admissible" on a technicality -- the mixture degenerating, not
   heteroscedasticity being powerless.  G7 is therefore scored on three
   surrogate-only measurements: the spec's N2c, a CALIBRATED N2c whose injected
   kurtosis is ``K_inj_needed`` (T02e) so the realised kurtosis actually matches
   the observed, and N1 itself.  The tail is inflatable on all three readings.

Nothing here decides a verdict: the gate rows go into ``T02_gates.csv`` and
:mod:`cliff.verdict` reads them (``G7_FLAG_CONVENTION``, ``underpowered_assays``,
``g9_rule_calibration``).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

# ``cliff.nulls`` pins the five BLAS thread variables to 1 before it imports
# numpy, and warns at fan-out if numpy was already imported (64 workers x the
# default OpenBLAS pool measured a load average of 1,409 on this box).  Import
# it FIRST so that pinning still binds when calibrate is the entry point.
from . import nulls as _nulls                                    # noqa: E402

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402
from scipy import stats                                          # noqa: E402

from . import config                                             # noqa: E402
from . import latent as _latent                                  # noqa: E402
from . import pairs as _pairs                                    # noqa: E402
from .config import PATHS, SEEDS, TAUS, TAU_WINDOW, THRESH       # noqa: E402
from .latent import crossfit_latent, fit_latent, mad_scaled, sigma_eval
from .nulls import (NullContext, build_context, default_stat_fn, get_context,
                    make_strata, observe, observed_stats, run_ensemble,
                    surrogate_N2, two_point_scale_mixture)

__all__ = [
    'G5_ASSAY', 'G6_ASSAYS', 'G8_ASSAYS', 'G9_ASSAYS', 'G10_ASSAYS',
    'mixture_two_component', 'bh_fdr', 'c2_stat_fn', 'C2_STAT_NAMES',
    'score_c2', 'naive_context', 't_n2_structural_audit',
    'g5_censoring_control', 'g6_antismooth_control', 'g6_all',
    'g7_scale_mixture', 'g7_all', 'g8_power', 'g8_all', 'g9_rule_fpr',
    'g10_composition', 'g10_all', 'write_T02_gate_rows', 'stage4',
]

# --------------------------------------------------------------------------- #
# assay sets, frozen from the spec                                            #
# --------------------------------------------------------------------------- #

#: Spec Sec.1.1 G5 / Sec.2 CONTROL #15.
G5_ASSAY = 'CR9114_FluAH3_logKd_4FQY'

#: Spec Sec.1.1 G6 / Sec.2 CONTROL #16, #17.
G6_ASSAYS = ('Z-domain_ZSPA-1_LL1_fitness_1LP1',
             'Z-domain_ZSPA-1_LL2_fitness_1LP1')

#: Spec Sec.1.1 G8's six representative assays, verbatim.
G8_ASSAYS = ('GB1_IgG-Fc_fitness_1FCC', 'GB1_IgG-Fc_fitness_1FCC_2016',
             'KRAS_RAF1-RBD_norfitness_6VJJ', 'SARS2-RBD_ACE2_deltaKd_6M0J',
             '5A12_VEGF_fitness_4ZFF', 'CR9114_FluAH1_logKd_4FQI')

#: G9 runs the C2 procedure end to end, so its assay set is the one the
#: procedure itself uses: the 14 the BH-FDR denominator is defined over (spec
#: Sec.1.3, "BH-FDR over the 14 primary+arm assays").
G9_ASSAYS = tuple(config.PRIMARY_AND_ARM)

#: G10 is a property of the clamp replay, so every assay with a null ensemble.
G10_ASSAYS = tuple(sorted(set(config.PRIMARY_AND_ARM + config.CONTROL)))

#: ORCHESTRATOR D3: CD19's standardised residual is not identified, so it
#: reports INCONCLUSIVE for C2/C3-L whatever its numbers, F7 leaves the
#: aggregate denominator and G9 calibrates ``k`` for ``K = 6``.
D3_UNIDENTIFIED = 'CD19_FMC63_Fitness_7URV'
D3_DROPPED_FAMILY = 'F7'
#: The families the aggregate rule is calibrated over (D3).
G9_FAMILIES = tuple(f for f in config.FAMILIES
                    if f not in ('F8', D3_DROPPED_FAMILY))

#: ``tau`` values the C2 verdict is read off (spec Sec.1.3 clause 2: ">= 4
#: consecutive tau in [3,8]", both unit systems).
C2_TAUS = tuple(t for t in TAUS if TAU_WINDOW[0] <= t <= TAU_WINDOW[1])
C2_UNITS = ('sigma', 'mad')


# =========================================================================== #
# BH-FDR and the two-component mixture -- spec Sec.4 pins them to scipy       #
# =========================================================================== #

def bh_fdr(p):
    """Benjamini-Hochberg q-values.  ``statsmodels`` is deliberately not a
    dependency (spec Sec.4); this is the ~10 lines it asks for.  Non-finite
    entries keep ``nan`` and leave ``m`` unchanged (an assay whose statistic is
    undefined is not a test that was performed)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q
    idx = np.nonzero(ok)[0]
    order = idx[np.argsort(p[idx], kind='stable')]
    ranked = p[order] * m / np.arange(1, m + 1, dtype=np.float64)
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1]
    return np.minimum(q, 1.0)


def _mixture_bins(c, n_bins, n_exact):
    """``(x, w)``: the squared ``c`` values and their weights.

    The E-step responsibility of a zero-mean two-component Gaussian depends on
    ``c`` only through ``c**2``, so the exact sufficient statistics are
    ``(sum w r, sum w r x, sum w (1-r) x)`` -- i.e. the likelihood is a function
    of the EMPIRICAL MEASURE of ``x = c**2`` alone and binning it is a
    quadrature, not a model change.  The ``n_exact`` largest ``|c|`` are kept
    unbinned because they are the only points the wide component is identified
    from; the rest are pooled into ``n_bins`` equal-count bins carrying their
    bin MEAN of ``x`` (so ``sum w x`` is preserved exactly, hence the
    one-component MLE and the marginal variance are exact).  Validated against
    the unbinned EM in :func:`_selfcheck`.
    """
    x = np.asarray(c, dtype=np.float64) ** 2
    x = x[np.isfinite(x)]
    n = x.size
    if n <= n_bins + n_exact:
        return x, np.ones(n)
    part = np.argpartition(x, n - n_exact)
    top = x[part[n - n_exact:]]
    rest = np.sort(x[part[:n - n_exact]])
    edges = np.linspace(0, rest.size, n_bins + 1).astype(np.int64)
    cs = np.concatenate([[0.0], np.cumsum(rest)])
    w = np.diff(edges).astype(np.float64)
    keep = w > 0
    means = (cs[edges[1:]] - cs[edges[:-1]])[keep] / w[keep]
    return (np.concatenate([means, top]),
            np.concatenate([w[keep], np.ones(n_exact)]))


def mixture_two_component(c, *, n_restart=None, n_iter=None, seed=None,
                          n_bins=4096, n_exact=2000):
    """Spec Sec.1.3's zero-mean two-component Gaussian mixture on ``c_hat``.

    ``pi`` (cliff mass), ``rho = s2/s1`` (jump amplification),
    ``dBIC = BIC2 - BIC1``, ``Lambda = 2(ll2 - ll1)``; 200 restarts x 100
    iterations, closed-form M-step, all restarts advanced simultaneously as a
    ``(n_restart, m)`` array.

    ``stats_c2.py`` does not exist on this branch, so this is G8's/G9's own copy
    of the estimator; it follows Sec.1.3 exactly (including the restart and
    iteration counts) so the recovered ``pi`` is on the same footing as the one
    the observed arm will report.

    ``pi_se`` is the OBSERVED-INFORMATION standard error,
    ``I = sum_i w_i ((f2_i - f1_i)/f_i)**2``, not the binomial
    ``sqrt(pi(1-pi)/n)``: the component labels are unobserved, so the binomial
    form understates the SE and would make Sec.1.3's "CI lower > 0.0005" clause
    anti-conservative.  It is a Wald interval, not the spec's block bootstrap
    over mutated positions (which cannot be afforded once per replicate in a
    2,880-cell power grid); reported as such.
    """
    n_restart = int(THRESH['C2_em_n_restart'] if n_restart is None else n_restart)
    n_iter = int(THRESH['C2_em_n_iter'] if n_iter is None else n_iter)
    out = dict(n=0, pi=np.nan, s1=np.nan, s2=np.nan, rho=np.nan, dBIC=np.nan,
               Lambda=np.nan, ll1=np.nan, ll2=np.nan, pi_se=np.nan,
               pi_ci_lo=np.nan, pi_ci_hi=np.nan, n_bins_used=0,
               converged_frac=np.nan)
    x, w = _mixture_bins(c, n_bins, n_exact)
    N = float(w.sum())
    if N < 32 or not np.isfinite(x).all():
        return out
    out['n'] = int(round(N))
    out['n_bins_used'] = int(x.size)
    v_tot = float((w * x).sum() / N)
    if not (v_tot > 0):
        return out
    LOG2PI = math.log(2.0 * math.pi)
    ll1 = float(-0.5 * N * (LOG2PI + math.log(v_tot)) - 0.5 * (w * x).sum() / v_tot)

    rng = np.random.default_rng(SEEDS['mixture_em'] if seed is None else seed)
    pi = rng.uniform(THRESH['C2_pi_lo'] * 0.5, 0.25, size=(n_restart, 1))
    r0 = rng.uniform(2.0, 100.0, size=(n_restart, 1))            # rho^2 start
    v1 = v_tot / ((1.0 - pi) + pi * r0)
    v2 = v1 * r0
    xr, wr = x[None, :], w[None, :]
    floor = 1e-12 * v_tot
    for _ in range(n_iter):
        lf1 = -0.5 * xr / v1 - 0.5 * np.log(v1)
        lf2 = -0.5 * xr / v2 - 0.5 * np.log(v2)
        la = np.log1p(-pi) + lf1
        lb = np.log(pi) + lf2
        r = 1.0 / (1.0 + np.exp(np.clip(la - lb, -700, 700)))
        wr_r = wr * r
        s_r = wr_r.sum(axis=1, keepdims=True)
        s_1 = (wr * (1.0 - r)).sum(axis=1, keepdims=True)
        pi = np.clip(s_r / N, 1e-12, 1.0 - 1e-12)
        v2 = np.maximum((wr_r * xr).sum(axis=1, keepdims=True)
                        / np.maximum(s_r, 1e-300), floor)
        v1 = np.maximum((wr * (1.0 - r) * xr).sum(axis=1, keepdims=True)
                        / np.maximum(s_1, 1e-300), floor)
        swap = v2 < v1                                  # component 2 is the wide one
        if swap.any():
            v1n = np.where(swap, v2, v1)
            v2 = np.where(swap, v1, v2)
            v1 = v1n
            pi = np.where(swap, 1.0 - pi, pi)
    lf1 = -0.5 * xr / v1 - 0.5 * np.log(v1)
    lf2 = -0.5 * xr / v2 - 0.5 * np.log(v2)
    la = np.log1p(-pi) + lf1
    lb = np.log(pi) + lf2
    mx = np.maximum(la, lb)
    ll = ((wr * (mx + np.log(np.exp(la - mx) + np.exp(lb - mx)))).sum(axis=1)
          - 0.5 * N * LOG2PI)
    k = int(np.nanargmax(ll))
    ll2 = float(ll[k])
    v1k, v2k, pik = float(v1[k, 0]), float(v2[k, 0]), float(pi[k, 0])
    out.update(pi=pik, s1=math.sqrt(v1k), s2=math.sqrt(v2k),
               rho=math.sqrt(v2k / v1k) if v1k > 0 else np.nan,
               ll1=ll1, ll2=ll2, Lambda=2.0 * (ll2 - ll1),
               dBIC=(-2.0 * ll2 + 3.0 * math.log(N))
                    - (-2.0 * ll1 + 1.0 * math.log(N)),
               converged_frac=float((np.abs(ll - ll2) <= 1e-6 * max(1.0, abs(ll2))
                                     ).mean()))
    # observed information for pi at the maximiser
    f1 = np.exp(-0.5 * x / v1k) / math.sqrt(v1k)
    f2 = np.exp(-0.5 * x / v2k) / math.sqrt(v2k)
    f = (1.0 - pik) * f1 + pik * f2
    with np.errstate(divide='ignore', invalid='ignore'):
        sc = (f2 - f1) / f
    I = float((w * sc * sc)[np.isfinite(sc)].sum())
    if I > 0:
        se = 1.0 / math.sqrt(I)
        z = float(stats.norm.ppf(0.975))
        out.update(pi_se=se, pi_ci_lo=pik - z * se, pi_ci_hi=pik + z * se)
    return out


# =========================================================================== #
# the C2 procedure, scored replicate by replicate (G8 / G9 need it end-to-end)#
# =========================================================================== #

def c_of_rep(ctx, rep):
    """``(c, keep, sub)``: the phi-centred ``c_hat`` on the replicate's OWN
    ``P_a``.  Byte-identical arithmetic to :func:`cliff.nulls.default_stat_fn`
    (ORCHESTRATOR D2), reused so a second definition of the cliff statistic
    cannot drift into this module."""
    keep = _nulls._pa_mask(ctx, rep['censor_mask'], rep['oof_finite'])
    sub = ctx.nested_idx[keep]
    if sub.shape[0] == 0:
        return np.empty(0), keep, sub
    ec = rep['e_oof'] - rep['mu_oof']
    num = ec[sub[:, 1]] - ec[sub[:, 0]]
    den = np.sqrt(rep['sigma_oof'][sub[:, 0]] ** 2
                  + rep['sigma_oof'][sub[:, 1]] ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        c = num / den
    return c, keep, sub


#: ``STAT_NAMES`` plus the mixture block spec Sec.1.3 clause 3 is read off.
C2_STAT_NAMES = tuple(_nulls.STAT_NAMES) + (
    'mix_pi', 'mix_rho', 'mix_dBIC', 'mix_Lambda', 'mix_pi_se',
    'mix_pi_ci_lo', 'mix_s1', 'mix_s2', 'mix_n')


def c2_stat_fn(ctx, rep):
    """The full statistic vector plus the two-component mixture.

    Used by G8 (power at the frozen thresholds) and G9 (the aggregate rule's
    FPR), which both have to score spec Sec.1.3's clause 3 and therefore cannot
    run off the cached 45-column vector.
    """
    out = default_stat_fn(ctx, rep)
    c, _keep, _sub = c_of_rep(ctx, rep)
    c = c[np.isfinite(c)]
    m = (mixture_two_component(c, n_bins=2048, n_exact=1000) if c.size >= 32
         else dict())
    for k, src in (('mix_pi', 'pi'), ('mix_rho', 'rho'), ('mix_dBIC', 'dBIC'),
                   ('mix_Lambda', 'Lambda'), ('mix_pi_se', 'pi_se'),
                   ('mix_pi_ci_lo', 'pi_ci_lo'), ('mix_s1', 's1'),
                   ('mix_s2', 's2'), ('mix_n', 'n')):
        out[k] = float(m.get(src, np.nan))
    return out


def tr_column(n_Pa):
    """Spec Sec.1.3's ``|P_a|`` regimes: ``TR1 = Q.999/Q.75`` at
    ``|P_a| >= 20,000``, ``TR2 = Q.99/Q.75`` at ``2,000 <= |P_a| < 20,000``, and
    **no tail-ratio verdict below 2,000**."""
    if not np.isfinite(n_Pa):
        return None
    if n_Pa >= THRESH['C2_TR1_min_Pa']:
        return 'TR1'
    if n_Pa >= THRESH['C2_TR2_min_Pa']:
        return 'TR2'
    return None


def c2_row_stats(row, ref, guard):
    """The scale-free half of the C2 decision for ONE dataset and one assay.

    ``row`` is a statistic vector (a ``dict`` or a Series); ``ref`` maps a
    statistic name to the reference N1 ensemble's values for that statistic
    (held out from ``row``); ``guard`` is
    :func:`cliff.nulls._grid_guard_taus`'s ``{unit: {tau: bool}}``.

    Returns ``TR``, its percentile in the reference, and per ``(unit, tau)`` the
    enrichment ``T = P_obs / mean_b P_b`` and the raw empirical
    ``p = (1 + #{P_b >= P_obs})/(B+1)`` -- spec Sec.1.3 verbatim.  The BH step is
    the caller's, because its family is the ASSAY set, not the tau sweep.
    """
    n_Pa = float(row['n_Pa'])
    col = tr_column(n_Pa)
    out = dict(n_Pa=n_Pa, TR_col=(col or ''), TR=np.nan, TR_pctile=np.nan,
               TR_ref_p995=np.nan, TR_ref_p95=np.nan, T={}, p={}, guard={})
    if col is not None:
        tr = float(row[col])
        rv = np.asarray(ref[col], dtype=np.float64)
        rv = rv[np.isfinite(rv)]
        out['TR'] = tr
        if rv.size and np.isfinite(tr):
            out['TR_pctile'] = float(100.0 * (rv <= tr).mean())
            out['TR_ref_p995'] = float(np.percentile(rv, THRESH['C2_TR_sup_pctile']))
            out['TR_ref_p95'] = float(np.percentile(rv, THRESH['C2_TR_ref_pctile']))
    for u in C2_UNITS:
        for t in C2_TAUS:
            key = (u, t)
            name = 'rate_%s_tau%g' % (u, t)
            gg = bool(guard.get(u, {}).get(t, True))
            out['guard'][key] = gg
            o = float(row[name])
            rv = np.asarray(ref[name], dtype=np.float64)
            rv = rv[np.isfinite(rv)]
            if not (gg and np.isfinite(o) and rv.size):
                out['T'][key] = np.nan
                out['p'][key] = np.nan
                continue
            mu = float(rv.mean())
            out['T'][key] = (o / mu) if mu > 0 else np.nan
            out['p'][key] = float((1.0 + (rv >= o).sum()) / (rv.size + 1.0))
    return out


def _longest_run(flags):
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def score_c2(rs, q, mix):
    """Spec Sec.1.3's per-assay C2 decision, clause by clause.

    ``rs`` from :func:`c2_row_stats`, ``q`` the BH q-values keyed
    ``(unit, tau)``, ``mix`` the mixture block.  Clause 4 (the G7-conditional
    C3-L conjunct) is NOT scored here -- it needs ``stats_c3.py``, which does not
    exist on this branch -- so ``supported`` is the conjunction of clauses 1-3
    and is reported under that name.
    """
    c1 = bool(np.isfinite(rs['TR']) and np.isfinite(rs['TR_ref_p995'])
              and rs['TR'] > rs['TR_ref_p995'])
    runs = {}
    for u in C2_UNITS:
        flags = [bool(np.isfinite(rs['T'][(u, t)])
                      and rs['T'][(u, t)] >= THRESH['C2_T_sup']
                      and np.isfinite(q.get((u, t), np.nan))
                      and q[(u, t)] < THRESH['C2_q_BH_sup'])
                 for t in C2_TAUS]
        runs[u] = _longest_run(flags)
    c2 = all(runs[u] >= THRESH['C2_n_consecutive_tau'] for u in C2_UNITS)
    pi = float(mix.get('pi', np.nan))
    c3 = bool(np.isfinite(mix.get('dBIC', np.nan))
              and mix['dBIC'] <= THRESH['C2_dBIC_sup']
              and np.isfinite(pi)
              and THRESH['C2_pi_lo'] <= pi <= THRESH['C2_pi_hi']
              and np.isfinite(mix.get('pi_ci_lo', np.nan))
              and mix['pi_ci_lo'] > THRESH['C2_pi_ci_lo_sup']
              and np.isfinite(mix.get('rho', np.nan))
              and mix['rho'] >= THRESH['C2_rho_sup'])
    # the REFUTED branch, for completeness (spec Sec.1.3)
    tmax = np.nanmax([rs['T'][(u, t)] for u in C2_UNITS for t in C2_TAUS]) \
        if np.any(np.isfinite([rs['T'][(u, t)] for u in C2_UNITS
                               for t in C2_TAUS])) else np.nan
    ref = bool((np.isfinite(rs['TR']) and np.isfinite(rs['TR_ref_p95'])
                and rs['TR'] < rs['TR_ref_p95'])
               or (np.isfinite(tmax) and tmax < THRESH['C2_T_ref'])
               or (np.isfinite(mix.get('dBIC', np.nan))
                   and mix['dBIC'] > THRESH['C2_dBIC_sup'])
               or (np.isfinite(mix.get('pi_ci_hi', np.nan))
                   and mix['pi_ci_hi'] < THRESH['C2_pi_ci_hi_ref']))
    return dict(clause1_TR=c1, clause2_sweep=c2, clause3_mixture=c3,
                run_sigma=runs['sigma'], run_mad=runs['mad'],
                T_max=float(tmax) if np.isfinite(tmax) else np.nan,
                supported=bool(c1 and c2 and c3), refuted=ref)


# =========================================================================== #
# the naive (censoring-blind) pipeline -- G5's "unmasked" arm                  #
# =========================================================================== #

def naive_context(ctx, *, verbose=False):
    """The SAME assay run through a pipeline with no censoring machinery.

    Spec Sec.1.1 G5 asks for ``T(4)`` "unmasked" and then "after floor masking +
    Tobit" -- so the unmasked arm is the pipeline with NEITHER: no ``P_a``
    condition (b), no Tobit E-step in the alternation, no clamp to replay in the
    surrogate.  This refits ``fit_latent`` + the 5-fold cross-fit with
    ``censor_mask`` all-False and ``censor_levels`` empty and returns a
    :class:`cliff.nulls.NullContext` carrying that fit, so every downstream
    function (``observe``, ``_refit_bundle``, ``default_stat_fn``,
    ``run_ensemble``) runs the censoring-blind pipeline unchanged.

    ``sigma_floor``, the folds, the decimal grid and the pair graph are the real
    pipeline's, so the ONLY difference between the two arms is the censoring
    machinery.
    """
    t0 = time.time()
    cm0 = np.zeros(ctx.n, dtype=bool)
    fit = fit_latent(ctx.X, ctx.y, cm0, (), sigma_floor=ctx.sigma_floor)
    cf = crossfit_latent(ctx.X, ctx.y, cm0, (), ctx.folds,
                         sigma_floor=ctx.sigma_floor)
    sk = (fit.sigma_knots[0], fit.sigma_knots[1])
    mk = (fit.sigma_knots[0], fit.sigma_knots[2])
    of = np.isfinite(cf['phi_oof']) & np.isfinite(cf['z_oof'])
    kw = {s: getattr(ctx, s) for s in NullContext.__slots__}
    kw.update(censor_levels=(), floor_levels=(), ceil_levels=(),
              floor_probs=np.zeros(0), floor_frac=0.0, ceil_frac=0.0,
              censor_mask=cm0, beta=fit.beta, phi=fit.phi, z=fit.z,
              e=fit.e, g_knots=fit.g_knots,
              hull=_latent.strict_hull(fit.g_knots),
              lo=float(fit.lo), hi=float(fit.hi),
              sigma_knots=sk, mu_knots=mk,
              phi_oof=cf['phi_oof'], z_oof=cf['z_oof'], e_oof=cf['e_oof'],
              sigma_oof=cf['sigma_oof'], mu_oof=sigma_eval(mk, cf['phi_oof']),
              oof_finite=of, pairwise=None, notes=dict(naive=True))
    nc = NullContext(**kw)
    nc.kurt = _nulls.kurtosis_targets(nc)
    nc.notes['wall_s'] = round(time.time() - t0, 2)
    nc.notes['n_iter_used'] = int(fit.n_iter_used)
    if verbose:
        print('    [naive fit] %s  %.1fs  n_iter=%d  resid_mad_oof=%.4f  '
              'sigma_median=%.4f'
              % (ctx.dms_id, time.time() - t0, fit.n_iter_used,
                 mad_scaled(cf['e_oof'][of]), float(np.median(cf['sigma_oof'][of]))))
    return nc


def _n2_rates(ctx, taus=(3.0, 4.0), B=None, seed_tag=0, use_censor_key=True):
    """``P_b(|c| >= tau)`` for ``B`` N2 replicates, computed directly (N2 does not
    refit, so this needs no ensemble and no cache).  Returns
    ``(obs, {tau: array}, n_Pa)``."""
    B = int(THRESH['null_B'] if B is None else B)
    rep0 = dict(y=ctx.y, censor_mask=ctx.censor_mask, phi_oof=ctx.phi_oof,
                z_oof=ctx.z_oof, e_oof=ctx.e_oof, sigma_oof=ctx.sigma_oof,
                mu_oof=ctx.mu_oof, oof_finite=ctx.oof_finite)
    c, keep, sub = c_of_rep(ctx, rep0)
    fin = np.isfinite(c)
    obs = {t: float((np.abs(c[fin]) >= t).mean()) for t in taus}
    den = np.sqrt(ctx.sigma_oof[sub[:, 0]] ** 2
                  + ctx.sigma_oof[sub[:, 1]] ** 2)
    strata = make_strata(ctx.n_muts, ctx.phi_oof,
                         censor_mask=(ctx.censor_mask if use_censor_key else None))
    got = {t: np.empty(B) for t in taus}
    for b in range(B):
        rng = np.random.default_rng([SEEDS['nulls_N2'],
                                     config.ASSAY_ORDINAL[ctx.dms_id],
                                     int(seed_tag), b])
        es = surrogate_N2(ctx, ctx.e_oof, rng, strata)
        ec = es - ctx.mu_oof
        with np.errstate(divide='ignore', invalid='ignore'):
            cs = np.abs((ec[sub[:, 1]] - ec[sub[:, 0]]) / den)
        cs = cs[np.isfinite(cs)]
        for t in taus:
            got[t][b] = float((cs >= t).mean())
    return obs, got, int(fin.sum())


def t_n2_structural_audit(assays=None, B=50, *, taus=(3.0, 4.0), verbose=True):
    """Is ``T(tau) < 1`` a censoring artefact or a property of nested differences?

    ``c_hat`` is a NESTED difference, so in the data ``e_u`` and ``e_v`` share
    the whole background ``B`` and are positively correlated; N2's
    within-(order x phi-decile x censored) exchange makes the two endpoints
    independent, which WIDENS the null numerator and pushes ``T(tau)`` below 1
    with no epistasis anywhere.  If that is the mechanism, ``T(tau) < 1`` must
    appear on assays with ZERO censoring too -- which is what decides whether
    G5's and G6's "inside the N2 band" clauses can be read two-sided at all.
    Reports ``corr(e_u, e_v)`` on ``P_a`` beside it, which is the mechanism's own
    observable.
    """
    if assays is None:
        assays = G10_ASSAYS
    rows = []
    for d in assays:
        ctx = get_context(d, verify=False)
        obs, got, n = _n2_rates(ctx, taus=taus, B=B, seed_tag=7001)
        rep0 = dict(y=ctx.y, censor_mask=ctx.censor_mask, phi_oof=ctx.phi_oof,
                    z_oof=ctx.z_oof, e_oof=ctx.e_oof, sigma_oof=ctx.sigma_oof,
                    mu_oof=ctx.mu_oof, oof_finite=ctx.oof_finite)
        _c, keep, sub = c_of_rep(ctx, rep0)
        ec = ctx.e_oof - ctx.mu_oof
        a, b = ec[sub[:, 0]], ec[sub[:, 1]]
        ok = np.isfinite(a) & np.isfinite(b)
        r = (float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 8 else np.nan)
        row = dict(DMS_id=d, tier=ctx.tier, n_Pa=n,
                   floor_frac=round(ctx.floor_frac, 6),
                   corr_e_endpoints=round(r, 5))
        for t in taus:
            mu = float(got[t].mean())
            row['obs_rate_tau%g' % t] = round(obs[t], 8)
            row['N2_mean_rate_tau%g' % t] = round(mu, 8)
            row['T_N2_tau%g' % t] = round(obs[t] / mu, 5) if mu > 0 else ''
        rows.append(row)
        if verbose:
            print('    [T_N2] %-40s floor=%.3f  corr(e_u,e_v)=%+.4f  '
                  'T(3)=%s  T(4)=%s' % (d, ctx.floor_frac, r,
                                        row.get('T_N2_tau3'), row.get('T_N2_tau4')))
        _nulls.clear_context_cache()
    return pd.DataFrame(rows)


# =========================================================================== #
# G5 -- censoring positive control (a STOP gate)                              #
# =========================================================================== #

def _arm_scores(ctx, ens, *, mix_c=None, verbose=False):
    """Score spec Sec.1.3's C2 clauses for ONE arm against ONE N1 ensemble.

    Single-assay BH (``m = 1``) so ``q == p``; the study's BH family is the 14
    assays, which is what :func:`g9_rule_fpr` uses.
    """
    obs = observed_stats(ctx)
    ref = {c: ens[c].values.astype(np.float64) for c in ens.columns}
    guard = {u: _nulls._grid_guard_taus(ctx, u) for u in C2_UNITS}
    rs = c2_row_stats(obs, ref, guard)
    q = {k: v for k, v in rs['p'].items()}
    if mix_c is None:
        rep0 = dict(y=ctx.y, censor_mask=ctx.censor_mask, phi_oof=ctx.phi_oof,
                    z_oof=ctx.z_oof, e_oof=ctx.e_oof, sigma_oof=ctx.sigma_oof,
                    mu_oof=ctx.mu_oof, oof_finite=ctx.oof_finite)
        mix_c, _k, _s = c_of_rep(ctx, rep0)
    mix = mixture_two_component(mix_c[np.isfinite(mix_c)])
    sc = score_c2(rs, q, mix)
    return obs, rs, mix, sc


def g5_censoring_control(B=None, *, nproc=1, verbose=True):
    """G5 (spec Sec.1.1, a **STOP** gate) on CR9114_FluAH3_4FQY.

    Three clauses, and two of them needed a decision the spec does not make:

    * **(i) "unmasked ``T(4)`` >= 5".**  Scored against the NAIVE pipeline's own
      N1 (:func:`naive_context`), not against N2.  N2 preserves the empirical
      residual marginal EXACTLY, and a detection limit is a property of that
      marginal -- the floor's residuals are ``-sigma(phi) Mills(a)``, a
      deterministic function of ``phi``, and exchanging them among floor rows
      inside a stratum reproduces the artefact instead of removing it.  So N2 is
      blind to censoring by construction and cannot be clause (i)'s reference;
      N1 (the smooth surrogate, which for a censoring-blind pipeline has no
      clamp to replay) can.  Both are reported.
    * **(ii) "after masking, ``T(4)`` inside the N2 95% band".**  Scored
      ONE-SIDED (``obs <= the 97.5th percentile``), because the gate's own
      consequence is an ENRICHMENT risk ("the pipeline cannot tell a detection
      limit from a cliff") and because ``T(tau) < 1`` is structural on this
      statistic, censoring or not -- see :func:`t_n2_structural_audit`.  The
      literal two-sided reading is reported as its own row.
    * **(iii) "``|P_a|`` collapses >= 10x"** -- as written.

    Plus the threshold-free version of the whole gate: the FULL C2 procedure is
    run on both arms, and the gate's question ("can the pipeline tell a
    detection limit from a cliff?") is answered by whether the verdict FLIPS.
    """
    B = int(THRESH['null_B'] if B is None else B)
    d = G5_ASSAY
    ctx = build_context(d, verify=False)
    nc = naive_context(ctx, verbose=verbose)
    n_nested = int(ctx.nested_idx.shape[0])
    u, v = ctx.nested_idx[:, 0], ctx.nested_idx[:, 1]
    cm = ctx.censor_mask
    out = dict(DMS_id=d, n=ctx.n, n_nested=n_nested,
               floor_levels=list(map(float, ctx.floor_levels)),
               floor_frac=round(ctx.floor_frac, 6),
               n_censored=int(cm.sum()),
               n_nested_censor_touching=int((cm[u] | cm[v]).sum()),
               n_nested_floor_floor=int((cm[u] & cm[v]).sum()),
               n_nested_wt_anchored=int(ctx.wt_anchored.sum()),
               naive_n_iter=nc.notes['n_iter_used'],
               naive_wall_s=nc.notes['wall_s'])

    # ---- the two N1 ensembles (the naive one is never cached: a bespoke ---- #
    # ---- context must not be able to write the canonical nulls/ path) ----- #
    _nulls.clear_context_cache()
    _nulls._CTX[d] = nc
    ens_u = run_ensemble(d, 'N1', B, stat_fn=default_stat_fn,
                         seed0=[SEEDS['nulls_N1'], config.ASSAY_ORDINAL[d], 5555],
                         nproc=nproc, verbose=False)
    _nulls.clear_context_cache()
    _nulls._CTX[d] = ctx
    ens_m = run_ensemble(d, 'N1', B, nproc=nproc, verbose=False)

    obs_u, rs_u, mix_u, sc_u = _arm_scores(nc, ens_u)
    obs_m, rs_m, mix_m, sc_m = _arm_scores(ctx, ens_m)
    out['n_Pa_unmasked'] = int(obs_u['n_Pa'])
    out['n_Pa_masked'] = int(obs_m['n_Pa'])
    out['Pa_collapse_factor'] = round(n_nested / max(obs_m['n_Pa'], 1), 4)
    out['Pa_collapse_pass'] = bool(out['Pa_collapse_factor']
                                   >= THRESH['G5_Pa_collapse_factor'])
    out['Pa_after_le_52000'] = bool(obs_m['n_Pa'] <= THRESH['G5_Pa_after_max'])

    # ---- clause (i): T(tau) unmasked, N1 reference ------------------------- #
    rows = []
    for arm, o, e_, ctx_ in (('unmasked_naive', obs_u, ens_u, nc),
                             ('masked_tobit', obs_m, ens_m, ctx)):
        for un in C2_UNITS:
            for t in TAUS:
                nm = 'rate_%s_tau%g' % (un, t)
                rv = e_[nm].values.astype(np.float64)
                rv = rv[np.isfinite(rv)]
                mu = float(rv.mean()) if rv.size else np.nan
                rows.append(dict(
                    arm=arm, unit=un, tau=t, n_Pa=int(o['n_Pa']),
                    obs_rate=o[nm], N1_mean_rate=mu,
                    T_N1=(o[nm] / mu if mu and np.isfinite(mu) and mu > 0
                          else np.nan),
                    N1_p025=(float(np.percentile(rv, 2.5)) if rv.size else np.nan),
                    N1_p975=(float(np.percentile(rv, 97.5)) if rv.size else np.nan),
                    grid_guard=bool(_nulls._grid_guard_taus(ctx_, un)[t])))
    sweep_N1 = pd.DataFrame(rows)
    out['T4_unmasked_N1'] = float(sweep_N1[(sweep_N1.arm == 'unmasked_naive')
                                          & (sweep_N1.unit == 'sigma')
                                          & (sweep_N1.tau == 4)]['T_N1'].iloc[0])
    out['T4_unmasked_N1_mad'] = float(sweep_N1[(sweep_N1.arm == 'unmasked_naive')
                                              & (sweep_N1.unit == 'mad')
                                              & (sweep_N1.tau == 4)]['T_N1'].iloc[0])
    out['T4_unmasked_N1_pass'] = bool(out['T4_unmasked_N1']
                                      >= THRESH['G5_unmasked_T4_min'])
    gsub = sweep_N1[(sweep_N1.arm == 'unmasked_naive') & sweep_N1.grid_guard
                    & (sweep_N1.unit == 'sigma')]
    out['guarded_taus_unmasked'] = json.dumps(sorted(float(t) for t in gsub.tau))
    out['guarded_taus_masked'] = json.dumps(sorted(float(t) for t in sweep_N1[
        (sweep_N1.arm == 'masked_tobit') & sweep_N1.grid_guard
        & (sweep_N1.unit == 'sigma')].tau))
    out['tau4_grid_guarded_out_unmasked'] = bool(4.0 not in set(gsub.tau))
    # spec Sec.1.0's grid guard: tau's ABSOLUTE cut must be >= 3 q_a.  On the
    # unmasked arm sigma-hat is 23x smaller (the naive fit has no floor plateau
    # to widen it), so the low taus fall below the assay's own score quantum and
    # are dropped from the sweep BEFORE any T is read.
    out['quantum'] = float(ctx.quantum)
    out['T_unmasked_N1_min_over_guarded_tau'] = (float(gsub['T_N1'].min())
                                                 if len(gsub) else float('nan'))
    out['T_unmasked_N1_max_over_guarded_tau'] = (float(gsub['T_N1'].max())
                                                 if len(gsub) else float('nan'))
    out['T_unmasked_N1_guarded_pass'] = bool(
        len(gsub) and out['T_unmasked_N1_min_over_guarded_tau']
        >= THRESH['G5_unmasked_T4_min'])

    # ---- both arms against N2 (the spec's literal T definition) ----------- #
    o_u2, g_u2, _n = _n2_rates(nc, taus=(3.0, 4.0), B=B, seed_tag=5556,
                               use_censor_key=False)
    o_m2, g_m2, _n = _n2_rates(ctx, taus=(3.0, 4.0), B=B, seed_tag=5557)
    n2rows = []
    for arm, o2, g2 in (('unmasked_naive', o_u2, g_u2),
                        ('masked_tobit', o_m2, g_m2)):
        for t in (3.0, 4.0):
            rv = g2[t]
            lo, hi = np.percentile(rv, [2.5, 97.5])
            n2rows.append(dict(arm=arm, tau=t, obs_rate=o2[t],
                               N2_mean_rate=float(rv.mean()),
                               N2_p025=float(lo), N2_p975=float(hi),
                               T_N2=(o2[t] / rv.mean() if rv.mean() > 0 else np.nan),
                               inside_two_sided=bool(lo <= o2[t] <= hi),
                               inside_one_sided=bool(o2[t] <= hi)))
    band = pd.DataFrame(n2rows)
    r4 = band[(band.arm == 'masked_tobit') & (band.tau == 4.0)].iloc[0]
    out['T4_masked_N2'] = float(r4['T_N2'])
    out['T4_masked_N2_band_lo'] = float(r4['N2_p025'] / r4['N2_mean_rate'])
    out['T4_masked_N2_band_hi'] = float(r4['N2_p975'] / r4['N2_mean_rate'])
    out['T4_masked_inside_two_sided'] = bool(r4['inside_two_sided'])
    out['T4_masked_inside_one_sided'] = bool(r4['inside_one_sided'])
    out['T4_unmasked_N2'] = float(band[(band.arm == 'unmasked_naive')
                                       & (band.tau == 4.0)]['T_N2'].iloc[0])

    # ---- the threshold-free version: does the C2 verdict FLIP? ------------ #
    for arm, o, rs, mix, sc in (('unmasked', obs_u, rs_u, mix_u, sc_u),
                                ('masked', obs_m, rs_m, mix_m, sc_m)):
        out['C2_%s_TR' % arm] = round(float(rs['TR']), 4)
        out['C2_%s_TR_ref_p995' % arm] = round(float(rs['TR_ref_p995']), 4)
        out['C2_%s_TR_pctile' % arm] = round(float(rs['TR_pctile']), 3)
        out['C2_%s_pi' % arm] = round(float(mix['pi']), 6)
        out['C2_%s_rho' % arm] = round(float(mix['rho']), 4)
        out['C2_%s_dBIC' % arm] = round(float(mix['dBIC']), 2)
        for k in ('clause1_TR', 'clause2_sweep', 'clause3_mixture',
                  'run_sigma', 'run_mad', 'supported'):
            out['C2_%s_%s' % (arm, k)] = sc[k]
    out['verdict_flips'] = bool(sc_u['supported'] and not sc_m['supported'])
    out['rate4_ratio_unmasked_over_masked'] = round(
        float(obs_u['rate_sigma_tau4'] / obs_m['rate_sigma_tau4']), 4)
    out['TR1_ratio_unmasked_over_masked'] = round(
        float(obs_u['TR1'] / obs_m['TR1']), 4)
    # The halting reading of the gate, clause by clause (see the docstring):
    # (i)' the unmasked statistic is inflated >= 5x against the naive pipeline's
    # own smooth null at EVERY tau the assay's grid guard admits -- tau = 4
    # itself is not admitted on that arm; (ii)' after masking there is no
    # residual ENRICHMENT (one-sided, because T < 1 is structural); (iii) the
    # |P_a| collapse, as written.
    out['PASS_purpose'] = bool(out['T_unmasked_N1_guarded_pass']
                               and out['Pa_collapse_pass']
                               and out['T4_masked_inside_one_sided'])
    out['PASS_literal'] = bool(out['T4_unmasked_N1_pass']
                               and out['T4_masked_inside_two_sided']
                               and out['Pa_collapse_pass'])
    _nulls.clear_context_cache()
    if verbose:
        print('[G5] %s' % d)
        print('     |P_a|: %d nested -> %d masked (%.2fx, spec >= %.0fx, '
              'expected ~41.7k)  %s'
              % (n_nested, out['n_Pa_masked'], out['Pa_collapse_factor'],
                 THRESH['G5_Pa_collapse_factor'],
                 'PASS' if out['Pa_collapse_pass'] else 'FAIL'))
        print('     clause (i)  unmasked T(4) vs naive N1 = %.3f (spec >= %.0f) '
              '%s   | vs N2 = %.3f (N2 is blind to censoring)'
              % (out['T4_unmasked_N1'], THRESH['G5_unmasked_T4_min'],
                 'PASS' if out['T4_unmasked_N1_pass'] else 'FAIL',
                 out['T4_unmasked_N2']))
        print('     clause (i)\' tau=4 grid-guarded OUT on the unmasked arm '
              '(q=%g): admitted taus %s, T_N1 over them in [%.3f, %.3f] -> %s'
              % (out['quantum'], out['guarded_taus_unmasked'],
                 out['T_unmasked_N1_min_over_guarded_tau'],
                 out['T_unmasked_N1_max_over_guarded_tau'],
                 'PASS' if out['T_unmasked_N1_guarded_pass'] else 'FAIL'))
        print('     clause (ii) masked T(4) vs N2 = %.3f, band [%.3f, %.3f]  '
              'two-sided %s / one-sided %s'
              % (out['T4_masked_N2'], out['T4_masked_N2_band_lo'],
                 out['T4_masked_N2_band_hi'],
                 'inside' if out['T4_masked_inside_two_sided'] else 'BELOW',
                 'inside' if out['T4_masked_inside_one_sided'] else 'ABOVE'))
        print('     C2 procedure: unmasked supported=%s (TR %.1f vs p99.5 %.1f, '
              'runs %d/%d, dBIC %.0f, pi %.4f, rho %.2f)'
              % (sc_u['supported'], rs_u['TR'], rs_u['TR_ref_p995'],
                 sc_u['run_sigma'], sc_u['run_mad'], mix_u['dBIC'],
                 mix_u['pi'], mix_u['rho']))
        print('                     masked supported=%s (TR %.1f vs p99.5 %.1f, '
              'runs %d/%d, dBIC %.0f, pi %.4f, rho %.2f)'
              % (sc_m['supported'], rs_m['TR'], rs_m['TR_ref_p995'],
                 sc_m['run_sigma'], sc_m['run_mad'], mix_m['dBIC'],
                 mix_m['pi'], mix_m['rho']))
        print('     verdict FLIPS = %s -> PASS_purpose=%s  PASS_literal=%s'
              % (out['verdict_flips'], out['PASS_purpose'], out['PASS_literal']))
    return dict(summary=out, sweep_N1=sweep_N1, band_N2=band)


# =========================================================================== #
# a fork pool that keeps ONE NullContext per worker (spec Sec.5)              #
# =========================================================================== #

#: Per-process job state.  ``fork`` + a module-level dict is the same pattern
#: :mod:`cliff.nulls` uses for ``_WORKER``: the parent builds the context first
#: so every child inherits it copy-on-write instead of paying for it ``nproc``
#: times, and a non-picklable payload never has to cross a pipe.
_JOB = {}


def _job_init(dms_id, payload):
    _JOB['dms_id'] = dms_id
    _JOB['ctx'] = get_context(dms_id, verify=False)
    _JOB['payload'] = payload


def _pool_map(dms_id, worker, jobs, nproc, payload=None):
    """``worker(job)`` over ``jobs`` with ``_JOB`` prepared.  ``worker`` must be a
    module-level function (``imap`` pickles it by name)."""
    jobs = list(jobs)
    _job_init(dms_id, payload)                     # parent first: COW for fork
    if int(nproc) <= 1 or len(jobs) <= 1:
        return [worker(j) for j in jobs]
    _nulls._warn_threads(int(nproc))
    import multiprocessing as mp
    cls = mp.get_context('fork')
    with cls.Pool(processes=int(nproc), initializer=_job_init,
                  initargs=(dms_id, payload)) as pool:
        return list(pool.imap(worker, jobs, chunksize=1))


def _rep0(ctx):
    """The OBSERVED data in replicate-bundle form (spec Sec.5: never refit a
    cached observed fit)."""
    return dict(y=ctx.y, censor_mask=ctx.censor_mask, phi_oof=ctx.phi_oof,
                z_oof=ctx.z_oof, e_oof=ctx.e_oof, sigma_oof=ctx.sigma_oof,
                mu_oof=ctx.mu_oof, oof_finite=ctx.oof_finite, n_iter_used=0,
                wall_s=0.0, meta={})


def _ens_ref(ens):
    """``{statistic: finite values}`` for :func:`c2_row_stats`."""
    return {c: ens[c].values.astype(np.float64) for c in ens.columns}


def _ens_drop(ens, b):
    """The leave-one-out reference: the ensemble WITHOUT replicate ``b``."""
    ref = {}
    for c in ens.columns:
        v = ens[c].values.astype(np.float64)
        if 0 <= b < v.size:
            v = np.delete(v, b)
        ref[c] = v
    return ref


def _gauss_rate(tau):
    """``P(|Z| >= tau)`` exactly, for the Gaussian reference of G7 reading (C)."""
    from scipy.special import erfc
    return float(erfc(float(tau) / math.sqrt(2.0)))


# =========================================================================== #
# G6 -- anti-smooth negative control (a STOP gate)                            #
# =========================================================================== #

def _nested_degree(ctx, sub=None):
    """Nested degree of every variant.  ``sub`` restricts the graph (``P_a``)."""
    idx = ctx.nested_idx if sub is None else sub
    return np.bincount(np.asarray(idx).ravel(), minlength=ctx.n)


def _equal_count_bins(v, k):
    """``k`` equal-count bins of ``v`` by average rank -- ties cannot empty a bin
    (nested degree is massively tied: GB1_1FCC has 4 distinct degrees over 62%
    of its variants)."""
    r = stats.rankdata(np.asarray(v, dtype=np.float64), method='average')
    return np.minimum((k * (r - 1.0) / r.size).astype(np.int64), k - 1)


def cliff_rate_by_density(ctx, taus=(3.0, 4.0), n_bins=5):
    """Spec Sec.1.4 C3-A clause 2 / G6's third clause: the cliff rate in each of
    5 equal-count neighbourhood-density bins.

    Density is the edge's own ``deg_u + deg_v`` in the assay's ``P_a`` graph,
    which is what "neighbourhood density" means for a PAIR statistic (a per-node
    degree would have to be assigned to one endpoint arbitrarily).  Monotone in
    density is the sequencing-depth artefact signature.
    """
    c, keep, sub = c_of_rep(ctx, _rep0(ctx))
    fin = np.isfinite(c)
    c, sub = c[fin], sub[fin]
    out = dict(n_Pa=int(c.size), n_bins=int(n_bins), rates={}, dens_mean={},
               monotone={}, spearman={}, spearman_p={})
    if c.size < 5 * n_bins:
        return out
    deg = _nested_degree(ctx, sub)
    dens = deg[sub[:, 0]].astype(np.float64) + deg[sub[:, 1]]
    q = _equal_count_bins(dens, n_bins)
    ac = np.abs(c)
    out['n_per_bin'] = [int((q == j).sum()) for j in range(n_bins)]
    out['dens_mean'] = [float(dens[q == j].mean()) for j in range(n_bins)]
    for t in taus:
        r = np.array([float((ac[q == j] >= t).mean()) for j in range(n_bins)])
        out['rates'][t] = [float(x) for x in r]
        inc = bool(np.all(np.diff(r) >= 0))
        dec = bool(np.all(np.diff(r) <= 0))
        out['monotone'][t] = bool(inc or dec)
        rho, p = stats.spearmanr(np.arange(n_bins), r)
        out['spearman'][t] = float(rho)
        out['spearman_p'][t] = float(p)
    return out


def _t04_row(dms_id):
    """The C1 verdict for one assay, read from T04 (stage 2 wrote it)."""
    p = os.path.join(PATHS.artifacts, 'T04_smoothness_C1.csv')
    if not os.path.exists(p):
        return {}
    t = pd.read_csv(p, dtype=str, keep_default_na=False)
    r = t[t['DMS_id'].astype(str) == dms_id]
    return dict(r.iloc[0]) if len(r) else {}


def g6_antismooth_control(dms_id, B=None, *, taus=(3.0, 4.0), nproc=1,
                          verbose=True):
    """G6 (spec Sec.1.1, a **STOP** gate) on one of the two Z-ZSPA1 controls.

    Three clauses: (i) C1-REFUTED (read from T04, which stage 2 wrote before any
    C2 number existed); (ii) ``T(4)`` inside the N2 95% band; (iii) the cliff
    rate must NOT be monotone in density quintile.

    Clause (ii) is scored the same way G5's is -- ONE-SIDED (``obs <=`` the
    97.5th percentile) as the halting reading, with the literal two-sided
    reading reported beside it -- because ``T(tau) < 1`` is a structural
    property of a nested difference against N2's within-stratum exchange and
    appears on assays with zero censoring too (:func:`t_n2_structural_audit`).
    The gate's own consequence text ("the pipeline is being fooled by
    selection-dependent library membership") is an ENRICHMENT risk, so the
    one-sided reading is the one that answers it.

    ``N2 power`` is reported beside the gate: these two assays have additive
    ``R2`` of 0.2168 / 0.2928, so their residual is nearly the whole signal and
    the N2 exchange may have little power to move it.
    """
    B = int(THRESH['null_B'] if B is None else B)
    ctx = get_context(dms_id, verify=False)
    t04 = _t04_row(dms_id)
    pre = config.EXPECTED['C1_predeclared_refutations'].get(dms_id, float('nan'))
    si = float(t04.get('SI', 'nan') or 'nan')
    out = dict(DMS_id=dms_id, tier=ctx.tier, n=ctx.n,
               n_nested=int(ctx.nested_idx.shape[0]),
               floor_frac=round(ctx.floor_frac, 6),
               verdict_C1=t04.get('verdict_C1', ''),
               SI=round(si, 6) if np.isfinite(si) else '',
               SI_predeclared=pre,
               SI_matches_predeclared=bool(np.isfinite(si)
                                           and abs(si - pre) <= 0.005),
               R2_add_latent=t04.get('R2_add_latent', ''),
               R2_add_raw=t04.get('R2_add_raw', ''),
               C1_refuted=bool(str(t04.get('verdict_C1', '')).upper()
                               == 'REFUTED'))
    obs, got, n_Pa = _n2_rates(ctx, taus=taus, B=B, seed_tag=6001)
    out['n_Pa'] = int(n_Pa)
    for t in taus:
        rv = got[t]
        mu = float(rv.mean())
        lo, hi = (float(x) for x in np.percentile(rv, [2.5, 97.5]))
        out['obs_rate_tau%g' % t] = obs[t]
        out['N2_mean_rate_tau%g' % t] = mu
        out['T_N2_tau%g' % t] = (obs[t] / mu) if mu > 0 else float('nan')
        out['T_N2_band_lo_tau%g' % t] = (lo / mu) if mu > 0 else float('nan')
        out['T_N2_band_hi_tau%g' % t] = (hi / mu) if mu > 0 else float('nan')
        out['inside_two_sided_tau%g' % t] = bool(lo <= obs[t] <= hi)
        out['inside_one_sided_tau%g' % t] = bool(obs[t] <= hi)
    dens = cliff_rate_by_density(ctx, taus=taus)
    out['density_n_per_bin'] = json.dumps(dens.get('n_per_bin', []))
    for t in taus:
        out['density_rates_tau%g' % t] = json.dumps(
            [round(x, 8) for x in dens['rates'].get(t, [])])
        out['density_monotone_tau%g' % t] = bool(dens['monotone'].get(t, False))
        out['density_spearman_tau%g' % t] = round(
            float(dens['spearman'].get(t, float('nan'))), 4)
    mono = any(bool(dens['monotone'].get(t, False)) for t in taus)
    out['density_monotone_any_tau'] = bool(mono)
    out['density_pass'] = bool(not mono)
    out['PASS_purpose'] = bool(out['C1_refuted']
                               and out['inside_one_sided_tau4']
                               and out['density_pass'])
    out['PASS_literal'] = bool(out['C1_refuted']
                               and out['inside_two_sided_tau4']
                               and out['density_pass'])
    if verbose:
        print('[G6] %s' % dms_id)
        print('     (i)   C1 = %s (SI %.4f vs pre-declared %.3f, match=%s), '
              'additive R2(latent) = %s'
              % (out['verdict_C1'], si, pre, out['SI_matches_predeclared'],
                 out['R2_add_latent']))
        print('     (ii)  T(4) vs N2 = %.4f, band [%.3f, %.3f]  two-sided %s / '
              'one-sided %s   (|P_a| = %d)'
              % (out['T_N2_tau4'], out['T_N2_band_lo_tau4'],
                 out['T_N2_band_hi_tau4'],
                 'inside' if out['inside_two_sided_tau4'] else 'BELOW',
                 'inside' if out['inside_one_sided_tau4'] else 'ABOVE',
                 out['n_Pa']))
        for t in taus:
            print('     (iii) cliff rate by density quintile, tau=%g: %s  '
                  'monotone=%s  spearman=%+.2f'
                  % (t, out['density_rates_tau%g' % t],
                     out['density_monotone_tau%g' % t],
                     out['density_spearman_tau%g' % t]))
        print('     -> PASS_purpose=%s  PASS_literal=%s'
              % (out['PASS_purpose'], out['PASS_literal']))
    return out


def g6_all(assays=None, B=None, *, nproc=1, verbose=True):
    if assays is None:
        assays = G6_ASSAYS
    rows = []
    for d in assays:
        rows.append(g6_antismooth_control(d, B=B, nproc=nproc, verbose=verbose))
        _nulls.clear_context_cache()
    return pd.DataFrame(rows)


# =========================================================================== #
# G7 -- scale-mixture discrimination (this gate SETS the C2 verdict rule)     #
# =========================================================================== #

#: The tail statistics G7 scores (spec Sec.1.1 G7: "``TR`` / ``T(tau)``").
def _g7_tail_names(ctx):
    col = tr_column(float(_nulls.observed_stats(ctx)['n_Pa']))
    names = []
    if col is not None:
        names += [col, col + '_mad']
    for u in C2_UNITS:
        for t in TAUS:
            names.append('rate_%s_tau%g' % (u, t))
    return names, col


def _g7_headline_C(tr_col):
    """The statistics reading (C) is scored on: the tail ratio and the swept
    rate at ``tau = 3, 4`` in both unit systems -- the cells the C2 verdict is
    actually read off, and the only ones where the Gaussian reference is not
    smaller than one edge in the whole benchmark."""
    out = []
    if tr_col:
        out += [tr_col, tr_col + '_mad']
    out += ['rate_%s_tau%g' % (u, t) for u in C2_UNITS for t in (3.0, 4.0)]
    return out


#: The localisation statistics.  ``cliff.nulls``' own G7 probe (its docstring):
#: under a pure heteroscedastic scale mixture the deviation must NOT recur for
#: the same added substitution, so these must sit at their null value.
G7_LOC_NAMES = ('icc_addcol', 'icc_pos')


def _g7_worker(b):
    """One CALIBRATED-N2c replicate: N2c whose INJECTED kurtosis is
    ``K_inj_needed`` (T02e), so the REALISED kurtosis of the refit residual
    lands on the observed ``K_obs`` instead of falling short of it."""
    ctx = _JOB['ctx']
    target = _JOB['payload']['target']
    tag = _JOB['payload']['seed_tag']
    rng = np.random.default_rng([SEEDS['nulls_N2c'],
                                 config.ASSAY_ORDINAL[ctx.dms_id],
                                 int(tag), int(b)])
    y_star, meta = _nulls.surrogate_N2c(ctx, rng, kurtosis_target=target,
                                        clamp=None, quantum=None)
    rep = _nulls._refit_bundle(ctx, y_star, meta)
    return default_stat_fn(ctx, rep)


def _mean_ratio(num, den):
    """``mean_b num_b / mean_b den_b`` over the finite entries of each."""
    a = np.asarray(num, dtype=np.float64)
    b = np.asarray(den, dtype=np.float64)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float('nan'), float('nan'), float('nan')
    ma, mb = float(a.mean()), float(b.mean())
    return ((ma / mb) if mb > 0 else float('nan'), ma, mb)


def g7_scale_mixture(dms_id, B=None, *, nproc=1, verbose=True, run_calibrated=True):
    """G7 (spec Sec.1.1) on one assay.  **Surrogate-only, by construction** --
    no observed value is read, which is why the gate can run before them.

    Three readings, because the spec's own N2c is a NO-OP on 16 of 17 assays
    (``cliff.nulls``' T02e: ``K_obs <= K_het``, i.e. the observed marginal
    kurtosis is LESS than the fitted heteroscedasticity alone already produces,
    so the two-point mixture degenerates to ``V == 1`` and N2c IS N1):

    * **(A) spec N2c / N1** -- the literal gate.  Both ensembles are cached
      (stage 3).  Degenerate on 16 of 17 assays, where a ratio of 1.00 means
      "the mixture had nothing to add", NOT "heteroscedasticity is powerless".
    * **(B) calibrated N2c / N1** -- N2c with ``kurtosis_target =
      K_inj_needed``, the injected kurtosis whose REALISED value lands on
      ``K_obs`` (T02e's dilution factor).  Non-degenerate only where T02e's
      ``room_realised`` is True; elsewhere it is N1 by construction and is
      reported as such rather than run.
    * **(C) N1 / the exact Gaussian reference** -- ``TR1_gauss = 2.8606``,
      ``TR2_gauss = 2.2393``, ``P(|Z| >= tau) = erfc(tau/sqrt2)``.  This is the
      reading that actually answers the gate's question: N1 ALREADY carries the
      fitted level-dependent ``sigma(phi)``, the monotone link and the
      cross-fit, so if its tail statistics stand far above the Gaussian values
      then heteroscedastic noise alone reproduces a heavy tail and ``TR`` /
      ``T(tau)`` cannot separate a cliff component from it.

    The localisation statistics are scored on (A) and (B) only: there is no
    analytic zero for an ICC (spec Sec.1.4 L1 says so in as many words -- "the
    analytic-zero claim is false"), so their null value is the N1 ensemble's own.
    """
    B = int(THRESH['G7_n_surrogates'] if B is None else B)
    ctx = get_context(dms_id, verify=False)
    ens1 = run_ensemble(dms_id, 'N1', THRESH['null_B'], nproc=1, verbose=False)
    ensc = run_ensemble(dms_id, 'N2c', THRESH['null_B'], nproc=1, verbose=False)
    tail, col = _g7_tail_names(ctx)
    guard = {u: _nulls._grid_guard_taus(ctx, u) for u in C2_UNITS}
    n2c = _nulls.two_point_scale_mixture(ctx.kurt['K_obs'], ctx.kurt['K_het'])
    aud = _nulls.n2c_audit(dms_id, verbose=False)
    out = dict(DMS_id=dms_id, tier=ctx.tier, TR_col=(col or ''),
               K_obs=round(ctx.kurt['K_obs'], 4),
               K_het=round(ctx.kurt['K_het'], 4),
               kurt_e_N1=aud.get('kurt_e_N1', ''),
               K_inj_needed=aud.get('K_inj_needed', ''),
               dilution_c=aud.get('dilution_c', ''),
               spec_N2c_degenerate=bool(n2c['degenerate']),
               spec_N2c_reason=n2c.get('reason', ''),
               room_realised=aud.get('room_realised', ''),
               B_cached=int(THRESH['null_B']), B_calibrated=0)

    # ---- (B): the calibrated N2c ensemble, if there is any room ----------- #
    target = aud.get('K_inj_needed', '')
    target = float(target) if target not in ('', None) else float('nan')
    cal_mix = (_nulls.two_point_scale_mixture(target, ctx.kurt['K_het'])
               if np.isfinite(target) else dict(degenerate=True,
                                                reason='K_inj_needed absent'))
    out['cal_N2c_degenerate'] = bool(cal_mix['degenerate'])
    out['cal_N2c_reason'] = cal_mix.get('reason', '')
    out['cal_N2c_ratio_v'] = (round(float(cal_mix.get('ratio', 1.0)), 3)
                              if not cal_mix['degenerate'] else 1.0)
    enscal = None
    if run_calibrated and not cal_mix['degenerate']:
        t0 = time.time()
        rows = _pool_map(dms_id, _g7_worker, range(B), nproc,
                         payload=dict(target=target, seed_tag=7101))
        enscal = pd.DataFrame([[float(r.get(k, np.nan)) for k in _nulls.STAT_NAMES]
                               for r in rows], columns=list(_nulls.STAT_NAMES))
        out['B_calibrated'] = int(B)
        out['cal_wall_s'] = round(time.time() - t0, 1)
        v = enscal['kurt_e'].values
        v = v[np.isfinite(v)]
        out['cal_kurt_e_realised'] = round(float(v.mean()), 4) if v.size else ''

    # ---- score every statistic under every reading ------------------------ #
    det = []
    for nm in tail + list(G7_LOC_NAMES):
        is_loc = nm in G7_LOC_NAMES
        gg = True
        if nm.startswith('rate_'):
            u, t = nm.split('_')[1], float(nm.split('tau')[1])
            gg = bool(guard[u].get(t, True))
        row = dict(DMS_id=dms_id, statistic=nm,
                   kind=('localisation' if is_loc else 'tail'),
                   grid_guard=gg)
        rA, mA, mN1 = _mean_ratio(ensc[nm].values, ens1[nm].values)
        row['N1_mean'] = mN1
        row['N2c_spec_mean'] = mA
        row['ratio_A_N2cspec_over_N1'] = rA
        if enscal is not None:
            rB, mB, _ = _mean_ratio(enscal[nm].values, ens1[nm].values)
            row['N2c_cal_mean'] = mB
            row['ratio_B_N2ccal_over_N1'] = rB
        else:
            row['N2c_cal_mean'] = ''
            row['ratio_B_N2ccal_over_N1'] = ('' if cal_mix['degenerate']
                                             else float('nan'))
        if not is_loc:
            if nm.startswith('rate_'):
                g = _gauss_rate(float(nm.split('tau')[1]))
            elif nm.startswith('TR1'):
                g = THRESH['C2_TR1_gauss']
            else:
                g = THRESH['C2_TR2_gauss']
            row['gauss_ref'] = g
            row['ratio_C_N1_over_gauss'] = ((mN1 / g) if (g > 0
                                            and np.isfinite(mN1)) else float('nan'))
        else:
            row['gauss_ref'] = ''
            row['ratio_C_N1_over_gauss'] = ''
        det.append(row)
    det = pd.DataFrame(det)

    tol = 1.0 + THRESH['G4_T_tol']
    live = det[det['grid_guard'].astype(bool)]
    tl = live[live['kind'] == 'tail']
    lo = live[live['kind'] == 'localisation']

    def _mx(frame, colname):
        v = pd.to_numeric(frame[colname], errors='coerce').values \
            if colname in frame.columns else np.array([])
        v = v[np.isfinite(v)]
        return (float(v.max()) if v.size else float('nan'))

    out['tail_max_ratio_A'] = _mx(tl, 'ratio_A_N2cspec_over_N1')
    out['tail_max_ratio_B'] = _mx(tl, 'ratio_B_N2ccal_over_N1')
    # Reading (C)'s flag is read off the HEADLINE statistics only -- TR and the
    # rate at tau = 3 and 4, in both unit systems.  At tau = 6 or 8 the Gaussian
    # reference is 1e-9 / 1e-15 and the ratio, while real, is a ratio of two
    # numbers neither arm can resolve with B = 200; the full curve is in
    # T02i_G7_statistics.csv and its max is reported as ``tail_max_ratio_C_all``.
    head = tl[tl['statistic'].isin(_g7_headline_C(out['TR_col']))]
    out['tail_max_ratio_C'] = _mx(head, 'ratio_C_N1_over_gauss')
    out['tail_max_ratio_C_all'] = _mx(tl, 'ratio_C_N1_over_gauss')
    for nm in _g7_headline_C(out['TR_col']):
        r = det[det['statistic'] == nm]
        out['ratio_C_' + nm] = (float(r['ratio_C_N1_over_gauss'].iloc[0])
                                if len(r) else float('nan'))
    out['loc_max_ratio_A'] = _mx(lo, 'ratio_A_N2cspec_over_N1')
    out['loc_max_ratio_B'] = _mx(lo, 'ratio_B_N2ccal_over_N1')
    for k in ('A', 'B', 'C'):
        r = out['tail_max_ratio_' + k]
        out['tail_inflated_' + k] = (bool(r > tol) if np.isfinite(r) else '')
    for k in ('A', 'B'):
        r = out['loc_max_ratio_' + k]
        out['loc_inflated_' + k] = (bool(r > tol) if np.isfinite(r) else '')
    out['tail_inflatable'] = bool(any(out['tail_inflated_' + k] is True
                                      for k in ('A', 'B', 'C')))
    out['localisation_inflated'] = bool(any(out['loc_inflated_' + k] is True
                                            for k in ('A', 'B')))
    out['tail_fires_on'] = ','.join(k for k in ('A', 'B', 'C')
                                    if out['tail_inflated_' + k] is True) or 'none'
    if verbose:
        print('[G7] %-40s TR=%s  K_obs=%.2f K_het=%.2f  specN2c=%s  '
              'calN2c=%s(B=%d)' % (dms_id, out['TR_col'], out['K_obs'],
                                   out['K_het'],
                                   'degenerate' if out['spec_N2c_degenerate']
                                   else 'live',
                                   'degenerate' if out['cal_N2c_degenerate']
                                   else 'live', out['B_calibrated']))
        print('     tail  max ratio: (A) specN2c/N1 = %.3f  (B) calN2c/N1 = %s  '
              '(C) N1/Gauss = %.2f [headline; %.3g over the whole tau sweep]  '
              '-> inflatable=%s (fires on %s)'
              % (out['tail_max_ratio_A'],
                 ('%.3f' % out['tail_max_ratio_B'])
                 if np.isfinite(out['tail_max_ratio_B']) else 'n/a',
                 out['tail_max_ratio_C'], out['tail_max_ratio_C_all'],
                 out['tail_inflatable'], out['tail_fires_on']))
        print('     local max ratio: (A) %.3f  (B) %s  -> inflated=%s  '
              '(no analytic zero for an ICC; the null value IS N1)'
              % (out['loc_max_ratio_A'],
                 ('%.3f' % out['loc_max_ratio_B'])
                 if np.isfinite(out['loc_max_ratio_B']) else 'n/a',
                 out['localisation_inflated']))
    return dict(summary=out, detail=det)


def g7_all(assays=None, B=None, *, nproc=1, verbose=True, run_calibrated=True):
    if assays is None:
        assays = G10_ASSAYS
    rows, det = [], []
    for d in assays:
        r = g7_scale_mixture(d, B=B, nproc=nproc, verbose=verbose,
                             run_calibrated=run_calibrated)
        rows.append(r['summary'])
        det.append(r['detail'])
        _nulls.clear_context_cache()
    return dict(summary=pd.DataFrame(rows),
                detail=pd.concat(det, ignore_index=True))


# =========================================================================== #
# G8 -- power and bias                                                        #
# =========================================================================== #

def _g8_eligible(ctx):
    """The variants a cliff can be injected into, and their ``P_a`` degree.

    ``P_a``'s endpoints, so an injected jump lands on at least one edge the
    procedure actually scores.  ``sum_v deg_a(v) = 2 |P_a|`` exactly, hence
    injecting each eligible variant independently with probability ``pi/2``
    targets an EDGE rate of ``pi`` -- which is the unit spec Sec.1.3's ``pi_hat``
    (a mass over ``c_hat`` values, i.e. over edges) is measured in.
    """
    keep = _nulls._pa_mask(ctx, ctx.censor_mask, ctx.oof_finite)
    sub = ctx.nested_idx[keep]
    elig = np.zeros(ctx.n, dtype=bool)
    if sub.shape[0]:
        elig[sub.ravel()] = True
    return elig, int(sub.shape[0])


def _g8_worker(job):
    """One cell of the power grid: an N1 surrogate with synthetic cliffs
    injected on the LATENT scale, put through the assay's own observation model
    and then through the FULL pipeline (refit + cross-fit + ``P_a`` + the C2
    clauses at the frozen thresholds).

    **Amplitude convention, stated because the spec does not pin it.**  The
    displacement is ``delta_v = +-a * sigma(phi_v) * sqrt(2)``.  The spec says
    "amplitude ``a in {2,3,4,6} sigma``" and the statistic the thresholds are
    read on is ``c_hat = (e_v - e_u)/sqrt(s2_u + s2_v)``, so a displacement of
    ``a sigma(phi_v)`` alone would register as ``a/sqrt(2)`` in ``c_hat`` units
    when the two endpoints have comparable scale.  The ``sqrt(2)`` makes the
    INJECTED CLIFF have amplitude ``a`` in the units of the statistic being
    thresholded, which is the only reading under which "power at ``a = 4
    sigma``" is a statement about ``tau = 4``.  The realised amplitude AFTER the
    refit is measured per cell (``amp_recovered``) and is what the bias column
    reports, so the convention is recoverable from the artifact either way:
    power at a variant-scale amplitude ``a`` is this grid's power at
    ``a/sqrt(2)``.
    """
    i_amp, i_pi, rep = job
    ctx = _JOB['ctx']
    pay = _JOB['payload']
    amp = float(pay['amps'][i_amp])
    pi = float(pay['rates'][i_pi])
    elig = pay['elig']
    rng = np.random.default_rng([SEEDS['g8_injection'],
                                 config.ASSAY_ORDINAL[ctx.dms_id],
                                 int(round(amp * 100)), int(round(pi * 1e6)),
                                 int(rep)])
    sd = sigma_eval(ctx.sigma_knots, ctx.phi)
    z = ctx.phi + rng.standard_normal(ctx.n) * sd
    hit = elig & (rng.random(ctx.n) < 0.5 * pi)
    sgn = np.where(rng.random(ctx.n) < 0.5, -1.0, 1.0)
    z = z + hit * sgn * amp * sd * math.sqrt(2.0)
    y_star, meta = observe(ctx, z, rng)
    rep_b = _nulls._refit_bundle(ctx, y_star, meta)
    out = default_stat_fn(ctx, rep_b)
    c, keep, sub = c_of_rep(ctx, rep_b)
    fin = np.isfinite(c)
    c, sub = c[fin], sub[fin]
    mix = mixture_two_component(c) if c.size >= 32 else dict()
    inj = (hit[sub[:, 0]] | hit[sub[:, 1]]) if c.size else np.zeros(0, bool)
    res = dict(i_amp=i_amp, i_pi=i_pi, rep=int(rep), amp=amp, pi=pi,
               n_inj_variants=int(hit.sum()), n_Pa=int(c.size),
               pi_edge_realised=(float(inj.mean()) if c.size else float('nan')),
               amp_recovered=(float(np.median(np.abs(c[inj])))
                              if inj.any() else float('nan')),
               amp_background=(float(np.median(np.abs(c[~inj])))
                               if (~inj).any() else float('nan')),
               resid_mad_oof=float(out['resid_mad_oof']),
               n_iter_used=float(out.get('n_iter_used', np.nan)))
    for k, src in (('pi_hat', 'pi'), ('rho_hat', 'rho'), ('dBIC', 'dBIC'),
                   ('pi_ci_lo', 'pi_ci_lo'), ('pi_ci_hi', 'pi_ci_hi'),
                   ('s1', 's1'), ('s2', 's2')):
        res[k] = float(mix.get(src, np.nan))
    rs = c2_row_stats(out, pay['ref'], pay['guard'])
    sc = score_c2(rs, dict(rs['p']), mix)
    res.update(TR=float(rs['TR']), TR_ref_p995=float(rs['TR_ref_p995']),
               TR_pctile=float(rs['TR_pctile']),
               clause1_TR=bool(sc['clause1_TR']),
               clause2_sweep=bool(sc['clause2_sweep']),
               clause3_mixture=bool(sc['clause3_mixture']),
               run_sigma=int(sc['run_sigma']), run_mad=int(sc['run_mad']),
               T_max=float(sc['T_max']), supported=bool(sc['supported']),
               refuted=bool(sc['refuted']))
    return res


def g8_power(dms_id, *, amps=None, rates=None, reps=None, nproc=1,
             rep_block=5, verbose=True, sink=None):
    """G8 (spec Sec.1.1) on one assay: the ``len(amps) x len(rates) x reps``
    injection grid.

    Reps are the OUTER loop and results are flushed after every ``rep_block``,
    so a run cut short still holds a COMPLETE grid at fewer reps -- the spec's
    grid is never truncated, only its replication (which is the lever the
    orchestrator names).
    """
    amps = tuple(THRESH['G8_amplitudes_sigma'] if amps is None else amps)
    rates = tuple(THRESH['G8_rates'] if rates is None else rates)
    reps = int(THRESH['G8_n_reps'] if reps is None else reps)
    ctx = get_context(dms_id, verify=False)
    ens = run_ensemble(dms_id, 'N1', THRESH['null_B'], nproc=1, verbose=False)
    elig, n_Pa_obs = _g8_eligible(ctx)
    payload = dict(amps=amps, rates=rates, elig=elig, ref=_ens_ref(ens),
                   guard={u: _nulls._grid_guard_taus(ctx, u) for u in C2_UNITS})
    mad_ref = ens['resid_mad_oof'].values.astype(np.float64)
    mad_ref = float(np.nanmean(mad_ref[np.isfinite(mad_ref)]))
    rows, t0 = [], time.time()
    for r0 in range(0, reps, int(rep_block)):
        blk = list(range(r0, min(r0 + int(rep_block), reps)))
        jobs = [(ia, ip, r) for r in blk
                for ia in range(len(amps)) for ip in range(len(rates))]
        got = _pool_map(dms_id, _g8_worker, jobs, nproc, payload=payload)
        for g in got:
            g['DMS_id'] = dms_id
            g['n_Pa_obs'] = n_Pa_obs
            g['sigma_bias_ratio'] = (g['resid_mad_oof'] / mad_ref
                                     if mad_ref > 0 else float('nan'))
            rows.append(g)
        if sink is not None:
            pd.DataFrame(rows).to_csv(sink, index=False)
        if verbose:
            print('    [G8] %-40s reps %d-%d done (%d cells, %.1fs)'
                  % (dms_id, blk[0], blk[-1], len(rows), time.time() - t0))
    df = pd.DataFrame(rows)
    df.attrs['wall_s'] = round(time.time() - t0, 1)
    return df


def _g8_cells(df):
    """Collapse the replicate rows into one row per ``(assay, a, pi)`` cell."""
    out = []
    for (d, a, pi), g in df.groupby(['DMS_id', 'amp', 'pi'], sort=True):
        n = len(g)
        out.append(dict(
            DMS_id=d, amp_sigma=float(a), pi=float(pi), n_reps=int(n),
            power=float(g['supported'].mean()),
            power_se=float(math.sqrt(max(g['supported'].mean()
                                         * (1 - g['supported'].mean()), 0) / n)),
            clause1_rate=float(g['clause1_TR'].mean()),
            clause2_rate=float(g['clause2_sweep'].mean()),
            clause3_rate=float(g['clause3_mixture'].mean()),
            refuted_rate=float(g['refuted'].mean()),
            pi_edge_realised=float(np.nanmedian(g['pi_edge_realised'])),
            pi_hat_median=float(np.nanmedian(g['pi_hat'])),
            pi_hat_bias=float(np.nanmedian(g['pi_hat']) - float(pi)),
            rho_hat_median=float(np.nanmedian(g['rho_hat'])),
            amp_recovered_median=float(np.nanmedian(g['amp_recovered'])),
            amp_recovery_ratio=float(np.nanmedian(g['amp_recovered']) / float(a)),
            amp_background_median=float(np.nanmedian(g['amp_background'])),
            sigma_bias_ratio=float(np.nanmedian(g['sigma_bias_ratio'])),
            TR_median=float(np.nanmedian(g['TR'])),
            TR_ref_p995=float(np.nanmedian(g['TR_ref_p995'])),
            T_max_median=float(np.nanmedian(g['T_max'])),
            n_Pa_median=float(np.nanmedian(g['n_Pa']))))
    return pd.DataFrame(out)


def g8_all(assays=None, *, amps=None, rates=None, reps=None, nproc=1,
           rep_block=5, verbose=True):
    """The whole G8 grid.  Assays are the OUTER-most loop only because each one
    needs its own context; the rep blocking inside keeps every assay's grid
    complete at every flush."""
    if assays is None:
        assays = G8_ASSAYS
    raw, t0 = [], time.time()
    os.makedirs(PATHS.artifacts, exist_ok=True)
    sink = os.path.join(PATHS.artifacts, 'T02j_G8_power_raw.csv')
    for i, d in enumerate(assays):
        if verbose:
            print('[G8] %d/%d %s' % (i + 1, len(assays), d))
        df = g8_power(d, amps=amps, rates=rates, reps=reps, nproc=nproc,
                      rep_block=rep_block, verbose=verbose)
        raw.append(df)
        pd.concat(raw, ignore_index=True).to_csv(sink, index=False)
        _nulls.clear_context_cache()
    raw = pd.concat(raw, ignore_index=True)
    cells = _g8_cells(raw)
    a_ref = float(THRESH['G8_power_ref_amplitude'])
    pi_ref = float(THRESH['G8_power_ref_rate'])
    ref = cells[(cells.amp_sigma == a_ref) & (cells.pi == pi_ref)]
    under = {}
    for _, r in ref.iterrows():
        if float(r['power']) < THRESH['G8_power_min']:
            under[str(r['DMS_id'])] = float(r['power'])
    cells.attrs['underpowered'] = under
    cells.attrs['ref_cell'] = (a_ref, pi_ref)
    cells.attrs['wall_s'] = round(time.time() - t0, 1)
    cells.attrs['grid'] = dict(
        amps=list(THRESH['G8_amplitudes_sigma'] if amps is None else amps),
        rates=list(THRESH['G8_rates'] if rates is None else rates),
        reps=int(THRESH['G8_n_reps'] if reps is None else reps),
        assays=list(assays))
    if verbose:
        print('[G8] reference cell (a = %g sigma, pi = %g): power by assay'
              % (a_ref, pi_ref))
        for _, r in ref.sort_values('power').iterrows():
            print('     %-40s power=%.3f +-%.3f  (clauses %.2f/%.2f/%.2f)  '
                  'pi_hat=%.5f  amp_rec=%.2f (%.0f%% of %g)  sigma_bias=%.3f  %s'
                  % (r['DMS_id'], r['power'], r['power_se'], r['clause1_rate'],
                     r['clause2_rate'], r['clause3_rate'], r['pi_hat_median'],
                     r['amp_recovered_median'],
                     100 * r['amp_recovery_ratio'], a_ref, r['sigma_bias_ratio'],
                     'UNDERPOWERED' if r['DMS_id'] in under else 'ok'))
        print('[G8] UNDERPOWERED: %s' % (sorted(under) or 'none'))
    return dict(cells=cells, raw=raw)


# =========================================================================== #
# G9 -- aggregate-rule FPR                                                    #
# =========================================================================== #

def _g9_worker(b):
    """One N1 replicate of one assay, scored with the FULL C2 statistic set.

    Seeded exactly as :func:`cliff.nulls.replicate` seeds N1, so replicate ``b``
    IS cached row ``b`` of that assay's N1 ensemble and the leave-one-out
    reference is the same estimator G4 is calibrated with.
    """
    ctx = _JOB['ctx']
    rng = np.random.default_rng(list(config.assay_seed('nulls_N1', ctx.dms_id))
                                + [int(b)])
    y_star, meta = _nulls.surrogate_N1(ctx, None, rng, clamp=None, quantum=None)
    rep = _nulls._refit_bundle(ctx, y_star, meta)
    out = default_stat_fn(ctx, rep)
    c, _keep, _sub = c_of_rep(ctx, rep)
    c = c[np.isfinite(c)]
    mix = mixture_two_component(c) if c.size >= 32 else dict()
    return dict(b=int(b), stats=out, mix={k: float(v) for k, v in mix.items()})


def g9_rule_fpr(n_datasets=None, *, assays=None, families=None, nproc=1,
                verbose=True):
    """G9 (spec Sec.1.1): the family-level false-positive rate of the k-of-K
    aggregate rule, measured by running the C2 procedure END TO END on complete
    N1 surrogate datasets.

    **K = 6, not 7** (ORCHESTRATOR D3): CD19_FMC63_7URV is
    STRUCTURALLY_UNIDENTIFIED -- only 62.04% of its rows have a finite
    out-of-fold ``phi`` and 1,467 of its 1,826 design columns occur exactly once
    -- so family F7 has left the aggregate denominator.  F8 (the two hypercube
    arm assays) was never in it (spec Sec.6 reports it separately).  CD19 and
    the F8 assays ARE still scored, because spec Sec.1.3 puts the BH-FDR family
    at "the 14 primary+arm assays" and dropping them would tighten every
    q-value; only the AGGREGATION drops them.

    The binomial reference is ``P(X >= k | Binom(K, 0.5))``, and the chosen
    ``k`` is the smallest whose measured FPR is ``<= 0.10``.  The Monte-Carlo SE
    of every FPR is reported beside it.
    """
    D = int(THRESH['G9_n_datasets'] if n_datasets is None else n_datasets)
    assays = tuple(G9_ASSAYS if assays is None else assays)
    families = tuple(G9_FAMILIES if families is None else families)
    from . import verdict as _v
    per = {}
    t0 = time.time()
    for i, d in enumerate(assays):
        ctx = get_context(d, verify=False)
        ens = run_ensemble(d, 'N1', THRESH['null_B'], nproc=1, verbose=False)
        guard = {u: _nulls._grid_guard_taus(ctx, u) for u in C2_UNITS}
        got = _pool_map(d, _g9_worker, range(D), nproc)
        rows = []
        for g in got:
            rs = c2_row_stats(g['stats'], _ens_drop(ens, g['b']), guard)
            rows.append(dict(b=g['b'], rs=rs, mix=g['mix']))
        per[d] = rows
        _nulls.clear_context_cache()
        if verbose:
            print('    [G9] %2d/%d %-40s D=%d  %.1fs'
                  % (i + 1, len(assays), d, D, time.time() - t0))
    # ---- BH over the assay set, per (unit, tau), then the per-assay call -- #
    keys = [(u, t) for u in C2_UNITS for t in C2_TAUS]
    outcomes = {d: [] for d in assays}
    detail = []
    for b in range(D):
        q_by = {}
        for k in keys:
            pv = np.array([per[d][b]['rs']['p'].get(k, np.nan) for d in assays])
            q = bh_fdr(pv)
            for j, d in enumerate(assays):
                q_by.setdefault(d, {})[k] = q[j]
        for d in assays:
            sc = score_c2(per[d][b]['rs'], q_by[d], per[d][b]['mix'])
            o = (_v.SUPPORTED if sc['supported']
                 else (_v.REFUTED if sc['refuted'] else _v.INCONCLUSIVE))
            outcomes[d].append(o)
            detail.append(dict(dataset=b, DMS_id=d, outcome=o,
                               clause1_TR=sc['clause1_TR'],
                               clause2_sweep=sc['clause2_sweep'],
                               clause3_mixture=sc['clause3_mixture'],
                               TR=per[d][b]['rs']['TR'],
                               TR_ref_p995=per[d][b]['rs']['TR_ref_p995'],
                               pi_hat=per[d][b]['mix'].get('pi', np.nan),
                               dBIC=per[d][b]['mix'].get('dBIC', np.nan)))
    counts = []
    fam_rows = []
    for b in range(D):
        n_sup = 0
        for f in families:
            mem = [d for d in config.FAMILIES[f] if d in assays]
            call = _v.family_call([outcomes[d][b] for d in mem],
                                  rule='majority')
            fam_rows.append(dict(dataset=b, family=f, n_members=len(mem),
                                 call=call[0], n_pos=call[1], n_neg=call[2],
                                 n_inc=call[3]))
            if call[0] == _v.SUPPORTED:
                n_sup += 1
        counts.append(n_sup)
    counts = np.asarray(counts, dtype=np.int64)
    K = len(families)
    fpr = {}
    for k in range(1, K + 1):
        p = float((counts >= k).mean())
        fpr[k] = dict(k=k, fpr=p,
                      se=float(math.sqrt(max(p * (1 - p), 0.0) / D)),
                      binom_p=float(stats.binom.sf(k - 1, K, 0.5)))
    k_cfg = int(THRESH['C2_family_k_true'])
    ok = [k for k in range(1, K + 1) if fpr[k]['fpr'] <= THRESH['G9_family_fpr_max']]
    k_chosen = min(ok) if ok else K + 1
    k_final = max(k_chosen, 1)
    out = dict(D=D, K=K, families=list(families), assays=list(assays),
               k_config=k_cfg, k_chosen=int(k_final),
               tightened=bool(k_final > k_cfg),
               fpr_at_k_config=(fpr[k_cfg]['fpr'] if 1 <= k_cfg <= K else float('nan')),
               se_at_k_config=(fpr[k_cfg]['se'] if 1 <= k_cfg <= K else float('nan')),
               fpr_at_k_chosen=(fpr[k_final]['fpr'] if k_final <= K else 0.0),
               se_at_k_chosen=(fpr[k_final]['se'] if k_final <= K else 0.0),
               binom_p_at_k_chosen=(fpr[k_final]['binom_p'] if k_final <= K
                                    else float('nan')),
               mean_supported_families=float(counts.mean()),
               max_supported_families=int(counts.max()),
               per_assay_fpr={d: float(np.mean([o == _v.SUPPORTED
                                                for o in outcomes[d]]))
                              for d in assays},
               wall_s=round(time.time() - t0, 1))
    out['fpr_by_k'] = {int(k): fpr[k] for k in fpr}
    if verbose:
        print('[G9] %d complete N1 datasets x %d assays; aggregation over K=%d '
              'families %s' % (D, len(assays), K, list(families)))
        print('     supported families per dataset: mean %.3f  max %d  '
              'distribution %s'
              % (counts.mean(), counts.max(),
                 json.dumps({int(v): int((counts == v).sum())
                             for v in sorted(set(counts.tolist()))})))
        for k in range(1, K + 1):
            print('     k=%d: FPR = %.3f +- %.3f   (binomial P(X>=k|Bin(%d,0.5))'
                  ' = %.4f)  %s' % (k, fpr[k]['fpr'], fpr[k]['se'], K,
                                    fpr[k]['binom_p'],
                                    'ok' if fpr[k]['fpr']
                                    <= THRESH['G9_family_fpr_max'] else 'ABOVE 0.10'))
        print('     config k = %d of %d -> chosen k = %d  (tightened=%s)'
              % (k_cfg, K, k_final, out['tightened']))
        pa = out['per_assay_fpr']
        print('     per-assay false SUPPORTED rate: %s'
              % json.dumps({k: round(v, 3) for k, v in sorted(pa.items())}))
    return dict(summary=out, per_dataset=pd.DataFrame(detail),
                per_family=pd.DataFrame(fam_rows))


# =========================================================================== #
# G10 -- censoring-mask composition                                           #
# =========================================================================== #

def _g10_bins(ctx, n_dec=10):
    """The fixed ``(order x degree-decile x phi-decile)`` bin of EVERY nested
    edge, plus the decile edges so a surrogate can be binned the same way.

    ``order`` is ``|B|`` (the lower endpoint of the nested pair, index 0 of
    ``nested_idx``); the degree is the edge's ``deg_u + deg_v`` in the full
    nested graph, which does not depend on the replicate; the ``phi`` decile is
    the mean of the two endpoints' ``phi^oof``.  Bin EDGES come from the
    observed data in every arm -- a surrogate re-binned on its own quantiles
    would compare two different partitions.
    """
    idx = ctx.nested_idx
    u, v = idx[:, 0], idx[:, 1]
    order = np.asarray(ctx.n_muts, dtype=np.int64)[u]
    deg = _nested_degree(ctx)
    dens = deg[u].astype(np.float64) + deg[v]
    ddec = _equal_count_bins(dens, n_dec)
    pm = 0.5 * (ctx.phi_oof[u] + ctx.phi_oof[v])
    fin = np.isfinite(pm)
    edges = (np.percentile(pm[fin], np.linspace(0, 100, n_dec + 1)[1:-1])
             if fin.sum() > n_dec else np.zeros(n_dec - 1))
    pdec = np.searchsorted(edges, np.where(fin, pm, -np.inf), side='right')
    n_ord = int(order.max()) + 1
    nb = n_ord * n_dec * n_dec
    fixed = (order * n_dec + ddec) * n_dec + pdec
    return dict(fixed=fixed.astype(np.int64), nb=int(nb), n_dec=int(n_dec),
                order=order, ddec=ddec, phi_edges=np.asarray(edges),
                base=(order * n_dec + ddec) * n_dec)


def _g10_worker(job):
    """``(P_a proportions on the fixed bins, on the replicate's own phi decile)``
    for one replicate of one null."""
    null, b = job
    ctx = _JOB['ctx']
    pay = _JOB['payload']
    nb, n_dec = pay['nb'], pay['n_dec']
    if null == 'N2':
        strata = make_strata(ctx.n_muts, ctx.phi_oof,
                             censor_mask=ctx.censor_mask)
        rng = np.random.default_rng(list(config.assay_seed('nulls_N2',
                                                           ctx.dms_id)) + [int(b)])
        es = surrogate_N2(ctx, ctx.e_oof, rng, strata)
        rep = dict(y=ctx.y, censor_mask=ctx.censor_mask, phi_oof=ctx.phi_oof,
                   z_oof=ctx.z_oof, e_oof=es, sigma_oof=ctx.sigma_oof,
                   mu_oof=ctx.mu_oof, oof_finite=ctx.oof_finite)
    else:
        rng = np.random.default_rng(list(config.assay_seed('nulls_N1',
                                                           ctx.dms_id)) + [int(b)])
        y_star, meta = _nulls.surrogate_N1(ctx, None, rng, clamp=None,
                                           quantum=None)
        rep = _nulls._refit_bundle(ctx, y_star, meta)
    keep = _nulls._pa_mask(ctx, rep['censor_mask'], rep['oof_finite'])
    n = int(keep.sum())
    if n == 0:
        z = np.zeros(nb)
        return null, int(b), 0, z, z
    p_fixed = np.bincount(pay['fixed'][keep], minlength=nb).astype(np.float64) / n
    pm = 0.5 * (rep['phi_oof'][ctx.nested_idx[:, 0]]
                + rep['phi_oof'][ctx.nested_idx[:, 1]])
    fin = np.isfinite(pm)
    pdec = np.searchsorted(pay['phi_edges'], np.where(fin, pm, -np.inf),
                           side='right')
    own = pay['base'] + pdec
    p_own = np.bincount(own[keep], minlength=nb).astype(np.float64) / n
    return null, int(b), n, p_fixed, p_own


def g10_composition(dms_id, B_N1=None, B_N2=5, *, nproc=1, verbose=True):
    """G10 (spec Sec.1.1): after masking, is the ``P_a`` the surrogate keeps made
    of the same edges as the observed one?

    ``max |p_obs(bin) - mean_b p_b(bin)| <= 0.02`` over the
    ``(order x degree-decile x phi-decile)`` bins.  Reported for N1 (which
    regenerates ``y*``, hence a fresh clamp AND a fresh cross-fit, so its
    ``P_a`` really can differ) and for N2 -- where the difference is a
    STRUCTURAL ZERO, not evidence: N2 permutes ``e`` and holds ``y``,
    ``censor_mask`` and ``oof_finite`` fixed, so ``_pa_mask`` returns the
    observed mask by construction.  It is measured anyway, because "0.000 by
    construction" is a claim that should be checkable in the artifact.

    Total variation distance and the number of occupied bins are reported
    beside the max, because with up to ``n_order x 100`` bins the per-bin
    proportions are small and a max-difference test is easy to pass.
    """
    B_N1 = int(25 if B_N1 is None else B_N1)
    ctx = get_context(dms_id, verify=False)
    bins = _g10_bins(ctx)
    keep0 = _nulls._pa_mask(ctx, ctx.censor_mask, ctx.oof_finite)
    n0 = int(keep0.sum())
    p_obs = (np.bincount(bins['fixed'][keep0], minlength=bins['nb'])
             .astype(np.float64) / max(n0, 1))
    out = dict(DMS_id=dms_id, tier=ctx.tier, n_Pa_obs=n0,
               n_nested=int(ctx.nested_idx.shape[0]),
               floor_frac=round(ctx.floor_frac, 6),
               n_bins_total=int(bins['nb']),
               n_bins_occupied_obs=int((p_obs > 0).sum()),
               max_order=int(bins['order'].max()), B_N1=B_N1, B_N2=int(B_N2))
    jobs = [('N1', b) for b in range(B_N1)] + [('N2', b) for b in range(int(B_N2))]
    got = _pool_map(dms_id, _g10_worker, jobs, nproc, payload=bins)
    for null in ('N1', 'N2'):
        sel = [g for g in got if g[0] == null]
        if not sel:
            continue
        nP = np.array([g[2] for g in sel], dtype=np.float64)
        pf = np.mean([g[3] for g in sel], axis=0)
        po = np.mean([g[4] for g in sel], axis=0)
        occ = (p_obs > 0) | (pf > 0) | (po > 0)
        out['%s_n_Pa_mean' % null] = float(nP.mean())
        out['%s_n_Pa_ratio' % null] = float(nP.mean() / max(n0, 1))
        out['%s_n_bins_occupied' % null] = int(occ.sum())
        out['%s_max_abs_diff_fixedphi' % null] = float(np.abs(pf - p_obs).max())
        out['%s_tv_fixedphi' % null] = float(0.5 * np.abs(pf - p_obs).sum())
        out['%s_max_abs_diff_ownphi' % null] = float(np.abs(po - p_obs).max())
        out['%s_tv_ownphi' % null] = float(0.5 * np.abs(po - p_obs).sum())
        j = int(np.argmax(np.abs(pf - p_obs)))
        out['%s_worst_bin' % null] = 'order=%d,degdec=%d,phidec=%d' % (
            j // (bins['n_dec'] ** 2),
            (j // bins['n_dec']) % bins['n_dec'], j % bins['n_dec'])
        out['%s_worst_bin_p_obs' % null] = float(p_obs[j])
        out['%s_worst_bin_p_null' % null] = float(pf[j])
        out['%s_pass' % null] = bool(out['%s_max_abs_diff_fixedphi' % null]
                                     <= THRESH['G10_max_bin_prop_diff'])
    out['max_abs_diff'] = float(max(out.get('N1_max_abs_diff_fixedphi', 0.0),
                                    out.get('N2_max_abs_diff_fixedphi', 0.0)))
    out['PASS'] = bool(out['max_abs_diff'] <= THRESH['G10_max_bin_prop_diff'])
    if verbose:
        print('[G10] %-40s |P_a| obs=%7d  N1 mean=%8.1f (%.3fx)  bins=%d/%d  '
              'max|dp| N1=%.5f (own-phi %.5f) N2=%.5f  TV N1=%.4f  -> %s'
              % (dms_id, n0, out.get('N1_n_Pa_mean', float('nan')),
                 out.get('N1_n_Pa_ratio', float('nan')),
                 out['n_bins_occupied_obs'], out['n_bins_total'],
                 out.get('N1_max_abs_diff_fixedphi', float('nan')),
                 out.get('N1_max_abs_diff_ownphi', float('nan')),
                 out.get('N2_max_abs_diff_fixedphi', float('nan')),
                 out.get('N1_tv_fixedphi', float('nan')),
                 'PASS' if out['PASS'] else 'FAIL'))
    return out


def g10_all(assays=None, B_N1=None, B_N2=5, *, nproc=1, verbose=True):
    if assays is None:
        assays = G10_ASSAYS
    rows = []
    for d in assays:
        rows.append(g10_composition(d, B_N1=B_N1, B_N2=B_N2, nproc=nproc,
                                    verbose=verbose))
        _nulls.clear_context_cache()
    return pd.DataFrame(rows)


# =========================================================================== #
# gate summaries on disk -- so a gate is computed ONCE and composed later      #
# =========================================================================== #

#: Every gate's machine-readable summary, so :func:`write_T02_gate_rows` can be
#: re-run (or run from a later process) without repeating a 40-minute grid.
SUMMARY_JSON = 'calibrate_gate_summaries.json'


def _jsonable(o):
    """numpy scalars / arrays -> plain Python, recursively.  ``json`` refuses
    ``np.bool_`` and ``np.float64``, and a silent ``str()`` would turn a number
    into a string that :mod:`cliff.verdict` then cannot read."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, pd.DataFrame):
        return _jsonable(o.to_dict('records'))
    return o


def _summary_path():
    return os.path.join(PATHS.artifacts, SUMMARY_JSON)


def _summary_lock():
    """flock around the summaries file AND around T02 -- ORCHESTRATOR D8's rule
    for every shared read-modify-write in this pipeline."""
    import contextlib
    import fcntl

    @contextlib.contextmanager
    def _lk():
        os.makedirs(PATHS.artifacts, exist_ok=True)
        fh = open(os.path.join(PATHS.artifacts, '.calibrate.lock'), 'a+')
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            yield fh
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            fh.close()
    return _lk()


def read_summaries():
    p = _summary_path()
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as fh:
            return json.load(fh)
    except ValueError:                                         # pragma: no cover
        return {}


def store_summary(gate, obj, *, verbose=True):
    with _summary_lock():
        cur = read_summaries()
        cur[gate] = _jsonable(obj)
        cur.setdefault('_written_utc', {})[gate] = time.strftime(
            '%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        tmp = _summary_path() + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(cur, fh, indent=1, sort_keys=True)
        os.replace(tmp, _summary_path())
    if verbose:
        print('[summary] %s stored in %s' % (gate, _summary_path()))


# =========================================================================== #
# T02 -- the G5..G10 rows                                                     #
# =========================================================================== #

MY_GATES = ('G5', 'G6', 'G7', 'G8', 'G9', 'G10')

_C_G5 = 'STOP -- the pipeline cannot tell a detection limit from a cliff'
_C_G6 = ('STOP -- the pipeline is being fooled by selection-dependent library '
         'membership')
_C_G7 = ('G7 does not pass or fail: it SETS the C2 rule.  observed=inflated -> '
         'C2 alone is inadmissible and C2 AND C3-L is mandatory')
_C_G7L = ('STOP -- localisation is ALSO inflated under N2c, so no axis '
          'separates cliffs from heteroscedastic noise')
_C_G8 = 'the assay is stamped UNDERPOWERED and reports INCONCLUSIVE whatever it shows'
_C_G9 = ('tighten k until <= 0.10 and record the change before any observed '
         'value is inspected')
_C_G10 = ('the clamp replay in the null is mis-specified; fix or flag every '
          'claim from that assay as conditional')

#: Read by :mod:`cliff.verdict` (``g7_flags``): the tail row's ``observed`` cell
#: is the rule switch, and a row whose text says "localisation" carries the
#: localisation axis.  Kept as a constant so the two modules cannot drift.
G7_OBSERVED_INFLATED = 'inflated'
G7_OBSERVED_NOT = 'not_inflated'


def _g5_rows(s):
    """G5's T02 rows.  The HALTING reading is the gate's own purpose ("can the
    pipeline tell a detection limit from a cliff?"), clause by clause; the two
    places where the literal wording cannot be scored as written are emitted as
    their own NON-halting rows beside it (see :func:`g5_censoring_control`)."""
    d, R = s['DMS_id'], []
    G = _pairs._g
    R.append(G('G5', 'censoring positive control: unmasked T(tau) over the '
               'grid-guarded taus', d,
               'min T(tau) on the UNMASKED (censoring-blind) arm over the taus '
               'the assay\'s 3q_a grid guard admits (%s), referenced to that '
               'pipeline\'s OWN N1; N2 preserves the residual marginal exactly '
               'and is blind to a detection limit by construction'
               % s.get('guarded_taus_unmasked', '[]'),
               THRESH['G5_unmasked_T4_min'],
               round(s['T_unmasked_N1_min_over_guarded_tau'], 4),
               '>= %g' % THRESH['G5_unmasked_T4_min'], _C_G5, 'YES', mode='ge'))
    R.append(G('G5', 'censoring positive control: |P_a| collapse after floor '
               'masking', d, '|P_a| after masking (spec: 524,272 -> <= 52,000, '
               'expected ~41.7k)', THRESH['G5_Pa_after_max'],
               int(s['n_Pa_masked']), '<= %d' % THRESH['G5_Pa_after_max'],
               _C_G5, 'YES', mode='le'))
    R.append(G('G5', 'censoring positive control: |P_a| collapse factor', d,
               'n_nested / |P_a| after masking', THRESH['G5_Pa_collapse_factor'],
               round(s['Pa_collapse_factor'], 3),
               '>= %gx' % THRESH['G5_Pa_collapse_factor'], _C_G5, 'YES',
               mode='ge'))
    R.append(G('G5', 'censoring positive control: masked T(4) shows no residual '
               'ENRICHMENT against N2 (one-sided)', d,
               'observed rate(4 sigma) <= the N2 97.5th percentile; one-sided '
               'because T(tau) < 1 is structural for a nested difference '
               'against N2\'s within-stratum exchange (measured on '
               'zero-censoring assays too, T02m) and the gate\'s consequence is '
               'an enrichment risk', 'inside',
               'inside' if s['T4_masked_inside_one_sided'] else 'ABOVE', '0',
               _C_G5, 'YES'))
    # ---- the two literal readings, reported beside the halting ones -------- #
    R.append(G('G5', 'censoring positive control: unmasked T(4) [LITERAL '
               'reading, reported beside the halting row]', d,
               'T(4) on the unmasked arm vs its own N1.  tau=4 is grid-guarded '
               'OUT on that arm (q_a=%g and the censoring-blind fit\'s sigma-hat '
               'has no floor plateau to widen it), so this cell is a number the '
               'assay\'s own grid guard would have dropped'
               % s.get('quantum', float('nan')),
               THRESH['G5_unmasked_T4_min'], round(s['T4_unmasked_N1'], 4),
               '>= %g' % THRESH['G5_unmasked_T4_min'],
               'none -- the halting reading is the grid-guarded row above',
               'no', mode='ge'))
    R.append(G('G5', 'censoring positive control: masked T(4) inside the N2 95% '
               'band [LITERAL two-sided reading, reported beside the halting '
               'row]', d,
               'T(4) = %.4f vs the N2 band [%.3f, %.3f]; a failure here is on '
               'the LOW side, which is the structural T(tau) < 1 of a nested '
               'difference and not a censoring artefact'
               % (s['T4_masked_N2'], s['T4_masked_N2_band_lo'],
                  s['T4_masked_N2_band_hi']), 'inside',
               'inside' if s['T4_masked_inside_two_sided'] else 'BELOW', '0',
               'none -- the halting reading is the one-sided row above', 'no'))
    # ---- what the masked arm still shows in the OTHER unit system --------- #
    sw = os.path.join(PATHS.artifacts, 'T02g_G5_sweep_N1.csv')
    if os.path.exists(sw):
        t = pd.read_csv(sw)
        m = t[(t['arm'] == 'masked_tobit') & (t['unit'] == 'mad')
              & t['grid_guard'].astype(bool)]
        above = m[m['obs_rate'] > m['N1_p975']]
        R.append(G('G5', 'censoring positive control: after masking, does the '
                   'MAD-unit tail return inside its own N1 envelope?', d,
                   'taus (MAD unit) whose masked observed rate is still above '
                   'the N1 97.5th percentile: %s.  The sigma-unit enrichment is '
                   'gone at every tau (T_N1 <= %.2f), but the MAD unit -- the '
                   'one a wrong sigma(phi) cannot touch -- still carries an '
                   'excess at the high taus, so the floor mask does not fully '
                   'neutralise the detection limit on this assay'
                   % (json.dumps(sorted(float(x) for x in above['tau'])),
                      float(t[(t['arm'] == 'masked_tobit')
                              & (t['unit'] == 'sigma')]['T_N1'].max()),),
                   0, int(len(above)), '0',
                   'every claim from this CONTROL assay stays conditional; it '
                   'is not in the aggregate denominator and no C2 verdict is '
                   'read off it', 'no', mode='le'))
    R.append(G('G5', 'censoring positive control: the C2 verdict FLIPS between '
               'the two arms', d,
               'the FULL C2 procedure (spec Sec.1.3 clauses 1-3) run on both '
               'arms: unmasked supported=%s (TR %.1f), masked supported=%s '
               '(TR %.1f) -- the threshold-free version of the whole gate'
               % (s['C2_unmasked_supported'], s['C2_unmasked_TR'],
                  s['C2_masked_supported'], s['C2_masked_TR']),
               True, bool(s['verdict_flips']), '0',
               'the gate\'s question is answered by whether masking changes the '
               'verdict, not only by T(4)', 'no'))
    return R


def _g6_rows(rows):
    R, G = [], _pairs._g
    for s in rows:
        d = s['DMS_id']
        R.append(G('G6', 'anti-smooth negative control: C1 REFUTED', d,
                   'verdict_C1 from T04 (SI = %s, pre-declared %s)'
                   % (s.get('SI'), s.get('SI_predeclared')), 'REFUTED',
                   str(s.get('verdict_C1', '')), '0', _C_G6, 'YES'))
        R.append(G('G6', 'anti-smooth negative control: T(4) shows no residual '
                   'ENRICHMENT against N2 (one-sided)', d,
                   'T(4) = %.4f vs the N2 band [%.3f, %.3f]; one-sided, as in '
                   'G5 and for the same structural reason'
                   % (s['T_N2_tau4'], s['T_N2_band_lo_tau4'],
                      s['T_N2_band_hi_tau4']), 'inside',
                   'inside' if s['inside_one_sided_tau4'] else 'ABOVE', '0',
                   _C_G6, 'YES'))
        R.append(G('G6', 'anti-smooth negative control: cliff rate NOT monotone '
                   'in density quintile', d,
                   'monotone (either direction) over the 5 equal-count '
                   'deg_u+deg_v bins at tau=3 or tau=4; rates(4) = %s, '
                   'Spearman(bin, rate) = %+.2f'
                   % (s.get('density_rates_tau4'),
                      s.get('density_spearman_tau4', float('nan'))),
                   False, bool(s['density_monotone_any_tau']), '0', _C_G6,
                   'YES'))
        R.append(G('G6', 'anti-smooth negative control: T(4) inside the N2 95% '
                   'band [LITERAL two-sided reading, beside the halting row]',
                   d, 'two-sided membership of the same band', 'inside',
                   'inside' if s['inside_two_sided_tau4'] else 'BELOW', '0',
                   'none -- the halting reading is the one-sided row above',
                   'no'))
        R.append(G('G6', 'anti-smooth negative control: additive R2 (context '
                   'for N2\'s power here)', d,
                   'R2 of the additive+link fit on the latent scale -- the '
                   'residual is nearly the whole signal on these two assays, so '
                   'N2 (which exchanges residuals) may have little power',
                   s.get('R2_add_latent', ''), s.get('R2_add_latent', ''), '0',
                   'reported as context, not a gate', 'no'))
    return R


def _g7_rows(rows, det=None):
    """G7's rows.  ``observed`` is the rule switch :mod:`cliff.verdict` reads
    (``G7_FLAG_CONVENTION``): one study-level tail row, one study-level
    localisation row (the STOP branch), and one tail row per assay."""
    R, G = [], _pairs._g
    infl = [r for r in rows if r.get('tail_inflatable')]
    loc = [r for r in rows if r.get('localisation_inflated')]
    fires = {}
    for r in rows:
        fires[r.get('tail_fires_on', 'none')] = \
            fires.get(r.get('tail_fires_on', 'none'), 0) + 1
    R.append(G('G7', 'scale-mixture discrimination: does N2c inflate TR / '
               'T(tau)?', 'PRIMARY+ARM',
               'inflated on %d of %d assays.  Readings: (A) the spec\'s N2c vs '
               'N1 -- DEGENERATE on %d assays because K_obs <= K_het (the '
               'observed marginal kurtosis is LESS than the fitted '
               'heteroscedasticity alone produces, T02e); (B) N2c calibrated to '
               'K_inj_needed so the REALISED kurtosis matches K_obs; (C) N1 vs '
               'the exact Gaussian references (TR1=2.8606, P(|Z|>=tau)). Fires '
               'on: %s'
               % (len(infl), len(rows),
                  sum(1 for r in rows if r.get('spec_N2c_degenerate')),
                  json.dumps(fires)),
               G7_OBSERVED_INFLATED,
               G7_OBSERVED_INFLATED if len(infl) > len(rows) // 2
               else G7_OBSERVED_NOT, '0', _C_G7, 'no'))
    R.append(G('G7', 'scale-mixture discrimination: are the LOCALISATION '
               'statistics also inflated under N2c?', 'PRIMARY+ARM',
               'max over assays of mean_b(ICC | N2c) / mean_b(ICC | N1) for '
               'icc_addcol and icc_pos, readings (A) and (B).  There is no '
               'analytic zero for an ICC (spec Sec.1.4 L1), so the null value '
               'IS the N1 ensemble\'s; inflated on %d of %d assays'
               % (len(loc), len(rows)), G7_OBSERVED_NOT,
               G7_OBSERVED_INFLATED if loc else G7_OBSERVED_NOT, '0',
               _C_G7L, 'YES'))
    for s in sorted(rows, key=lambda r: r['DMS_id']):
        R.append(G('G7', 'scale-mixture discrimination: does N2c inflate TR / '
                   'T(tau)?', s['DMS_id'],
                   'max ratio (A) specN2c/N1 = %.3f [%s], (B) calN2c/N1 = %s, '
                   '(C) N1/Gauss = %.2f'
                   % (s['tail_max_ratio_A'],
                      'degenerate' if s['spec_N2c_degenerate'] else 'live',
                      ('%.3f' % s['tail_max_ratio_B'])
                      if isinstance(s['tail_max_ratio_B'], float)
                      and np.isfinite(s['tail_max_ratio_B']) else 'degenerate',
                      s['tail_max_ratio_C']),
                   G7_OBSERVED_INFLATED,
                   G7_OBSERVED_INFLATED if s['tail_inflatable']
                   else G7_OBSERVED_NOT, '0', _C_G7, 'no'))
    return R


def _g8_rows(cells, grid, under):
    R, G = [], _pairs._g
    a_ref = float(THRESH['G8_power_ref_amplitude'])
    pi_ref = float(THRESH['G8_power_ref_rate'])
    ref = [r for r in cells if float(r['amp_sigma']) == a_ref
           and float(r['pi']) == pi_ref]
    gtxt = ('grid as run: a in %s x pi in %s x %d reps on %d assays '
            '(%d cells); amplitude convention: delta_v = a*sigma(phi_v)*sqrt2, '
            'so an injected cliff has amplitude a in the units of c_hat itself'
            % (list(grid['amps']), list(grid['rates']), int(grid['reps']),
               len(grid['assays']),
               len(grid['amps']) * len(grid['rates']) * int(grid['reps'])
               * len(grid['assays'])))
    R.append(G('G8', 'power & bias: detection power at a = 4 sigma, pi = 0.005',
               '6 representative assays',
               'min power over the %d assays.  %s' % (len(ref), gtxt),
               THRESH['G8_power_min'],
               round(min([float(r['power']) for r in ref]), 4) if ref else '',
               '>= %g' % THRESH['G8_power_min'], _C_G8, 'no', mode='ge'))
    for r in sorted(ref, key=lambda r: str(r['DMS_id'])):
        R.append(G('G8', 'power & bias: detection power at a = 4 sigma, '
                   'pi = 0.005', str(r['DMS_id']),
                   'fraction of %d reps in which the C2 clauses 1-3 all pass at '
                   'the frozen thresholds (clause pass rates %.2f / %.2f / '
                   '%.2f; realised injected edge rate %.5f)'
                   % (int(r['n_reps']), float(r['clause1_rate']),
                      float(r['clause2_rate']), float(r['clause3_rate']),
                      float(r['pi_edge_realised'])),
                   THRESH['G8_power_min'], round(float(r['power']), 4),
                   '>= %g' % THRESH['G8_power_min'], _C_G8, 'no', mode='ge'))
        R.append(G('G8', 'power & bias: recovered pi_hat at a = 4 sigma, '
                   'pi = 0.005', str(r['DMS_id']),
                   'median mixture pi_hat over %d reps against the injected '
                   'edge rate' % int(r['n_reps']), pi_ref,
                   round(float(r['pi_hat_median']), 6), pi_ref,
                   'the mixture mass is biased; pi_hat is not the injected rate',
                   'no', mode='eq'))
        R.append(G('G8', 'power & bias: recovered amplitude at a = 4 sigma, '
                   'pi = 0.005', str(r['DMS_id']),
                   'median |c_hat| on the injected edges after the refit, in '
                   'units of the injected a (background edges sit at %.2f)'
                   % float(r['amp_background_median']), 1.0,
                   round(float(r['amp_recovery_ratio']), 4), 0.15,
                   'the refit dilutes the injected jump; power is quoted at the '
                   'injected, not the recovered, amplitude', 'no', mode='eq'))
        R.append(G('G8', 'power & bias: recovered sigma-hat bias at a = 4 '
                   'sigma, pi = 0.005', str(r['DMS_id']),
                   'median MAD(e_oof) of the injected surrogate / mean over the '
                   'clean N1 ensemble', 1.0,
                   round(float(r['sigma_bias_ratio']), 4), 0.05,
                   'injected cliffs inflate the noise scale they are then '
                   'divided by, which is the mechanism that costs power', 'no',
                   mode='eq'))
    R.append(G('G8', 'power & bias: assays stamped UNDERPOWERED',
               '6 representative assays',
               'assays with power < %g at (a = 4 sigma, pi = 0.005)'
               % THRESH['G8_power_min'], 0, len(under), '0',
               'each listed assay reports INCONCLUSIVE for C2/C3-L whatever it '
               'shows: %s' % (','.join(sorted(under)) or 'none'), 'no',
               mode='le'))
    return R


def _g9_rows(s):
    R, G = [], _pairs._g
    # verdict.g9_rule_calibration reads rows[0]'s observed as the FPR, so the
    # FPR row must come first.
    R.append(G('G9', 'aggregate-rule FPR of the k-of-%d rule' % s['K'],
               '%d N1 datasets' % s['D'],
               'family-level FPR of the k=%d-of-%d rule over %d complete N1 '
               'surrogate datasets (MC SE %.3f).  K=%d, not 7: F7 '
               '(CD19_FMC63_7URV) is STRUCTURALLY_UNIDENTIFIED and F8 was never '
               'in the denominator; all %d BH-FDR assays are still scored'
               % (s['k_chosen'], s['K'], s['D'], s['se_at_k_chosen'], s['K'],
                  len(s['assays'])),
               THRESH['G9_family_fpr_max'], round(s['fpr_at_k_chosen'], 4),
               '<= %g' % THRESH['G9_family_fpr_max'], _C_G9, 'no', mode='le'))
    R.append(G('G9', 'aggregate-rule FPR: the chosen k', 'FAMILIES',
               'smallest k whose measured FPR is <= %g; config k = %d of %d '
               'families (spec Sec.1.3 "subject to G9 tightening"), FPR there '
               '= %.4f +- %.3f'
               % (THRESH['G9_family_fpr_max'], s['k_config'], s['K'],
                  s['fpr_at_k_config'], s['se_at_k_config']),
               s['k_config'], s['k_chosen'], '0',
               'recorded in artifacts/G9_k_tightening.json and applied by '
               'verdict.py before any observed value is inspected', 'no'))
    R.append(G('G9', 'aggregate-rule FPR: binomial reference', 'FAMILIES',
               'P(X >= k | Binom(%d, 0.5)) at the chosen k = %d -- the '
               'generality statement, never the evidence'
               % (s['K'], s['k_chosen']), round(s['binom_p_at_k_chosen'], 4),
               round(s['binom_p_at_k_chosen'], 4), '0',
               'reported as context, not a gate', 'no'))
    R.append(G('G9', 'aggregate-rule FPR: per-assay false SUPPORTED rate',
               '%d N1 datasets' % s['D'],
               'max over the %d scored assays of the rate at which C2 clauses '
               '1-3 all pass on a pure-N1 dataset: %s'
               % (len(s['assays']),
                  json.dumps({k: round(v, 3)
                              for k, v in sorted(s['per_assay_fpr'].items())
                              if v > 0}) or 'all zero'),
               0.05, round(max(list(s['per_assay_fpr'].values()) or [0.0]), 4),
               '<= 0.05', 'the per-assay procedure itself is anti-conservative',
               'no', mode='le'))
    return R


def _g10_rows(rows):
    R, G = [], _pairs._g
    worst = max(rows, key=lambda r: float(r.get('max_abs_diff', 0.0)))
    R.append(G('G10', 'censoring-mask composition (order x degree-decile x '
               'phi-decile)', 'censored assays',
               'max absolute bin-proportion difference, worst over the %d '
               'assays (%s); N1 arm B=%s, N2 arm B=%s.  N2\'s difference is a '
               'structural zero: it permutes e and holds y, censor_mask and '
               'oof_finite fixed, so P_a is the observed one by construction'
               % (len(rows), worst['DMS_id'], worst.get('B_N1'),
                  worst.get('B_N2')),
               THRESH['G10_max_bin_prop_diff'],
               round(float(worst['max_abs_diff']), 5),
               '<= %g' % THRESH['G10_max_bin_prop_diff'], _C_G10, 'no',
               mode='le'))
    for s in sorted(rows, key=lambda r: r['DMS_id']):
        R.append(G('G10', 'censoring-mask composition (order x degree-decile x '
                   'phi-decile)', s['DMS_id'],
                   'max |p_obs - mean_b p_N1,b| over %d occupied bins '
                   '(TV %.4f, worst bin %s: %.4f vs %.4f); |P_a| N1/obs = '
                   '%.3f; own-phi-decile reading %.5f'
                   % (int(s.get('N1_n_bins_occupied', 0)),
                      float(s.get('N1_tv_fixedphi', float('nan'))),
                      s.get('N1_worst_bin', ''),
                      float(s.get('N1_worst_bin_p_obs', float('nan'))),
                      float(s.get('N1_worst_bin_p_null', float('nan'))),
                      float(s.get('N1_n_Pa_ratio', float('nan'))),
                      float(s.get('N1_max_abs_diff_ownphi', float('nan')))),
                   THRESH['G10_max_bin_prop_diff'],
                   round(float(s.get('N1_max_abs_diff_fixedphi',
                                     float('nan'))), 5),
                   '<= %g' % THRESH['G10_max_bin_prop_diff'], _C_G10, 'no',
                   mode='le'))
    return R


def write_T02_gate_rows(summaries=None, *, write=True, verbose=True):
    """Replace the ``G5``-``G10`` rows of ``T02_gates.csv`` and touch nothing
    else.

    Built with :func:`cliff.pairs._g`, the row helper every other gate is scored
    by, and the file is rewritten from the other gates' rows VERBATIM (G0/G1/G1b/
    G2/G3/G4 and the structural G-STR/G11 block are carried through unchanged).
    The whole read-modify-write sits inside the same flock the summaries use.
    """
    s = read_summaries() if summaries is None else summaries
    rows = []
    if 'G5' in s:
        rows += _g5_rows(s['G5'])
    if 'G6' in s:
        rows += _g6_rows(s['G6'])
    if 'G7' in s:
        rows += _g7_rows(s['G7'])
    if 'G8' in s:
        rows += _g8_rows(s['G8']['cells'], s['G8']['grid'], s['G8']['underpowered'])
    if 'G9' in s:
        rows += _g9_rows(s['G9'])
    if 'G10' in s:
        rows += _g10_rows(s['G10'])
    if not rows:
        if verbose:
            print('[T02] nothing to write: no G5-G10 summary on disk')
        return None
    new = pd.DataFrame(rows, columns=_pairs.T02_COLUMNS)
    with _summary_lock():
        old, p = _nulls._t02_read()
        if len(old):
            gid = old['gate_id'].astype(str).str.strip().values
            mine = np.isin(gid, MY_GATES)
            pos = int(np.nonzero(mine)[0][0]) if mine.any() else len(old)
            before = old.iloc[:pos][~mine[:pos]]
            after = old.iloc[pos:][~mine[pos:]]
            out = pd.concat([before, new, after], ignore_index=True)
        else:
            out = new
        out = out[_pairs.T02_COLUMNS]
        if write:
            os.makedirs(PATHS.artifacts, exist_ok=True)
            tmp = p + '.tmp'
            out.to_csv(tmp, index=False)
            os.replace(tmp, p)
    if verbose:
        n_fail = int((new['PASS/FAIL'] == 'FAIL').sum())
        halt = new[(new['PASS/FAIL'] == 'FAIL')
                   & (new['halts_study'].astype(str).str.upper() == 'YES')]
        print('[T02] wrote %s (%d rows; %d G5-G10 rows, %d other gates '
              'untouched; %d FAIL, %d of them halting)'
              % (p, len(out), len(new), len(out) - len(new), n_fail, len(halt)))
        for _, r in new.iterrows():
            print('      %-4s %-8s %-40s %-28s exp=%-12s obs=%-12s %s'
                  % (r['gate_id'], r['halts_study'], str(r['assay'])[:40],
                     str(r['gate_name'])[:28], str(r['expected'])[:12],
                     str(r['observed'])[:12], r['PASS/FAIL']))
    return out


# =========================================================================== #
# stage 4                                                                     #
# =========================================================================== #

def _csv(df, name, verbose=True):
    os.makedirs(PATHS.artifacts, exist_ok=True)
    p = os.path.join(PATHS.artifacts, name)
    df.to_csv(p, index=False)
    if verbose:
        print('[artifact] %s  (%d rows, %d cols)' % (p, len(df), df.shape[1]))
    return p


def run_G5(*, B=None, nproc=1, verbose=True):
    r = g5_censoring_control(B=B, nproc=nproc, verbose=verbose)
    _csv(pd.DataFrame([r['summary']]), 'T02g_G5_censoring.csv', verbose)
    _csv(r['sweep_N1'], 'T02g_G5_sweep_N1.csv', verbose)
    _csv(r['band_N2'], 'T02g_G5_band_N2.csv', verbose)
    store_summary('G5', r['summary'], verbose=verbose)
    return r


def run_G6(*, B=None, nproc=1, verbose=True):
    t = g6_all(B=B, nproc=nproc, verbose=verbose)
    _csv(t, 'T02h_G6_antismooth.csv', verbose)
    store_summary('G6', t.to_dict('records'), verbose=verbose)
    return t


def run_G7(assays=None, *, B=None, nproc=1, verbose=True):
    r = g7_all(assays, B=B, nproc=nproc, verbose=verbose)
    _csv(r['summary'], 'T02i_G7_scale_mixture.csv', verbose)
    _csv(r['detail'], 'T02i_G7_statistics.csv', verbose)
    store_summary('G7', r['summary'].to_dict('records'), verbose=verbose)
    return r


def run_G8(assays=None, *, reps=None, nproc=1, rep_block=5, verbose=True):
    r = g8_all(assays, reps=reps, nproc=nproc, rep_block=rep_block,
               verbose=verbose)
    _csv(r['cells'], 'T02j_G8_power.csv', verbose)
    _csv(r['raw'], 'T02j_G8_power_raw.csv', verbose)
    store_summary('G8', dict(cells=r['cells'].to_dict('records'),
                             grid=r['cells'].attrs['grid'],
                             underpowered=r['cells'].attrs['underpowered'],
                             wall_s=r['cells'].attrs['wall_s']), verbose=verbose)
    return r


def run_G9(*, D=None, nproc=1, verbose=True):
    r = g9_rule_fpr(D, nproc=nproc, verbose=verbose)
    _csv(pd.DataFrame([r['summary']['fpr_by_k'][k]
                       for k in sorted(r['summary']['fpr_by_k'])]),
         'T02k_G9_rule_fpr.csv', verbose)
    _csv(r['per_dataset'], 'T02k_G9_per_dataset.csv', verbose)
    _csv(r['per_family'], 'T02k_G9_per_family.csv', verbose)
    # the k the aggregate rule will actually use, recorded BEFORE any observed
    # value is inspected (spec Sec.1.1 G9) and read by verdict.g9_rule_calibration
    p = os.path.join(PATHS.artifacts, 'G9_k_tightening.json')
    with open(p, 'w') as fh:
        json.dump({'C2': int(r['summary']['k_chosen']),
                   '_note': ('G9 on %d complete N1 datasets, K=%d families '
                             '(F7 STRUCTURALLY_UNIDENTIFIED, F8 reported '
                             'separately); FPR at k=%d is %.4f +- %.3f, ceiling '
                             '%g.  Only C2 was calibrated: C2 is the criterion '
                             'whose procedure G9 ran end to end.'
                             % (r['summary']['D'], r['summary']['K'],
                                r['summary']['k_chosen'],
                                r['summary']['fpr_at_k_chosen'],
                                r['summary']['se_at_k_chosen'],
                                THRESH['G9_family_fpr_max']))}, fh, indent=1)
    if verbose:
        print('[artifact] %s' % p)
    store_summary('G9', r['summary'], verbose=verbose)
    return r


def run_G10(assays=None, *, B_N1=None, B_N2=5, nproc=1, verbose=True):
    t = g10_all(assays, B_N1=B_N1, B_N2=B_N2, nproc=nproc, verbose=verbose)
    _csv(t, 'T02l_G10_composition.csv', verbose)
    store_summary('G10', t.to_dict('records'), verbose=verbose)
    return t


def run_TN2(assays=None, *, B=50, nproc=1, verbose=True):
    """The structural ``T(tau) < 1`` audit that G5's and G6's one-sided readings
    rest on.  Not a gate; a measurement that decides how a gate may be read."""
    t = t_n2_structural_audit(assays, B=B, verbose=verbose)
    _csv(t, 'T02m_T_N2_structural.csv', verbose)
    store_summary('T_N2_audit', t.to_dict('records'), verbose=verbose)
    return t


def stage4(assays=None, nproc=None, verbose=True, *, gates=MY_GATES,
           reuse=True, B=None, g8_reps=None, g9_D=None, g10_B=None,
           tn2_B=50, register=True):
    """Spec Sec.5 stage 4: G5, G6, G7, G8, G9, G10, then the T02 rows.

    ``reuse=True`` skips a gate whose summary is already on disk, so the
    expensive ones (G8's grid, G9's 50 full-pipeline datasets) are paid for
    ONCE and the T02 composition can be re-run at will.  ``register=True`` makes
    the ONE ``register_null_cache`` call the run is allowed (D8) and then
    verifies the manifest.
    """
    config.assert_env()
    if nproc is None:
        nproc = THRESH['nproc_cap']
    nproc = int(min(int(nproc), THRESH['nproc_cap']))
    have = read_summaries()
    t0 = time.time()
    done = []
    if verbose:
        print('[stage4] gates=%s  nproc=%d  reuse=%s  (already on disk: %s)'
              % (list(gates), nproc, reuse,
                 sorted(k for k in have if not k.startswith('_')) or 'none'))
    if 'T_N2_audit' not in have or not reuse:
        run_TN2(assays, B=int(tn2_B), nproc=nproc, verbose=verbose)
    for g in gates:
        if reuse and g in have:
            if verbose:
                print('[stage4] %s: reusing the summary on disk' % g)
            continue
        if g == 'G5':
            run_G5(B=B, nproc=nproc, verbose=verbose)
        elif g == 'G6':
            run_G6(B=B, nproc=nproc, verbose=verbose)
        elif g == 'G7':
            run_G7(assays, B=B, nproc=nproc, verbose=verbose)
        elif g == 'G8':
            run_G8(reps=g8_reps, nproc=nproc, verbose=verbose)
        elif g == 'G9':
            run_G9(D=g9_D, nproc=nproc, verbose=verbose)
        elif g == 'G10':
            run_G10(assays, B_N1=g10_B, nproc=nproc, verbose=verbose)
        else:
            raise ValueError('unknown gate %r' % (g,))
        done.append(g)
    t02 = write_T02_gate_rows(verbose=verbose)
    if register:
        ents = _nulls.register_null_cache(extra=dict(calibrate=dict(
            gates=list(gates), nproc=int(nproc),
            summaries=os.path.join('local-records/bindingGYM-cliff/artifacts',
                                   SUMMARY_JSON),
            wall_s=round(time.time() - t0, 1),
            written_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))))
        bad = _pairs.verify_manifest()
        if bad:
            raise RuntimeError('MANIFEST md5 mismatch after stage 4: %r'
                               % (bad[:5],))
        if verbose:
            print('[stage4] %d nulls/*.npz registered, MANIFEST verified clean'
                  % len(ents))
    if verbose:
        print('[stage4] computed %s, wall %.1fs'
              % (done or 'nothing (all reused)', time.time() - t0))
    return dict(t02=t02, computed=tuple(done),
                summaries=read_summaries(), wall_s=round(time.time() - t0, 1))


# =========================================================================== #
# self-check                                                                  #
# =========================================================================== #

def _selfcheck(argv=()):
    """No-data algebra first, then one live assay.  The mixture is validated
    against an UNBINNED EM, which is the only claim in this module that a reader
    cannot check by inspection."""
    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 60)
    print('[env] %s' % (config.assert_env(),))

    print('\n[1] BH-FDR')
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, np.nan])
    q = bh_fdr(p)
    print('    p = %s\n    q = %s' % (p, np.round(q, 6)))
    assert np.isnan(q[-1]) and np.all(np.diff(q[:6][np.argsort(p[:6])]) >= -1e-12)
    assert abs(q[0] - 0.006) < 1e-12, q[0]
    print('    monotone, nan-preserving, m excludes the undefined test  OK')

    print('\n[2] MIXTURE: binned EM == unbinned EM')
    rng = np.random.default_rng(11)
    # n stays modest: the unbinned arm holds a (n_restart x n) array, which is
    # exactly the cost the binning exists to remove.
    for pi_t, rho_t, n in ((0.01, 5.0, 30000), (0.005, 3.0, 20000)):
        hi = rng.random(n) < pi_t
        c = rng.standard_normal(n) * np.where(hi, rho_t, 1.0)
        a = mixture_two_component(c, n_bins=4096, n_exact=2000)
        b = mixture_two_component(c, n_bins=10 ** 9, n_exact=0)   # unbinned
        print('    n=%6d pi_true=%.4f rho_true=%.1f | binned pi=%.5f rho=%.3f '
              'dBIC=%9.1f | unbinned pi=%.5f rho=%.3f dBIC=%9.1f'
              % (n, pi_t, rho_t, a['pi'], a['rho'], a['dBIC'], b['pi'],
                 b['rho'], b['dBIC']))
        assert abs(a['pi'] - b['pi']) < 0.003, (a['pi'], b['pi'])
        assert abs(a['rho'] - b['rho']) < 0.15, (a['rho'], b['rho'])
        assert abs(a['pi'] - pi_t) < 0.005, (a['pi'], pi_t)
    print('    the binning is a quadrature, not a model change  OK')

    print('\n[3] TR regimes and the run counter')
    assert tr_column(25000) == 'TR1' and tr_column(5000) == 'TR2'
    assert tr_column(1999) is None and tr_column(float('nan')) is None
    assert _longest_run([1, 1, 0, 1, 1, 1]) == 3
    print('    tr_column / _longest_run  OK')

    print('\n[4] G8 INJECTION ALGEBRA (no refit): does pi/2 hit an edge rate '
          'of pi, and does a land at a in c_hat units?')
    d = 'Z-domain_ZpA963_HL1_fitness_2M5A'
    ctx = get_context(d, verify=False)
    elig, n_Pa = _g8_eligible(ctx)
    keep = _nulls._pa_mask(ctx, ctx.censor_mask, ctx.oof_finite)
    sub = ctx.nested_idx[keep]
    for pi in (0.005, 0.02):
        rg = np.random.default_rng(3)
        hit = elig & (rg.random(ctx.n) < 0.5 * pi)
        got = float((hit[sub[:, 0]] | hit[sub[:, 1]]).mean())
        print('    pi=%.3f -> injected variants %4d, edge rate %.5f (target '
              '%.5f, ratio %.2f)' % (pi, int(hit.sum()), got, pi, got / pi))
    sd = sigma_eval(ctx.sigma_knots, ctx.phi)
    rg = np.random.default_rng(5)
    e = ctx.e_oof.copy()
    hit = elig & (rg.random(ctx.n) < 0.01)
    e = e + hit * 4.0 * sd * math.sqrt(2.0)
    num = (e - ctx.mu_oof)[sub[:, 1]] - (e - ctx.mu_oof)[sub[:, 0]]
    den = np.sqrt(ctx.sigma_oof[sub[:, 0]] ** 2 + ctx.sigma_oof[sub[:, 1]] ** 2)
    c = num / den
    inj = hit[sub[:, 0]] | hit[sub[:, 1]]
    print('    a=4: median |c_hat| on injected edges = %.3f (target 4), on the '
          'rest = %.3f  [no refit, so this is the ceiling the refit dilutes]'
          % (float(np.median(np.abs(c[inj]))), float(np.median(np.abs(c[~inj])))))

    print('\n[5] G9 REPRODUCIBILITY: worker replicate b == cached ensemble row b')
    ens = run_ensemble(d, 'N1', THRESH['null_B'], nproc=1, verbose=False)
    _job_init(d, None)
    g = _g9_worker(3)
    for nm in ('n_Pa', 'TR1', 'rate_sigma_tau3', 'SI', 'resid_mad_oof'):
        a, b = float(g['stats'][nm]), float(ens[nm].values[3])
        ok = (abs(a - b) <= 1e-9 * max(1.0, abs(b)))
        print('    %-18s worker %.10g  cached %.10g  %s'
              % (nm, a, b, 'OK' if ok else 'MISMATCH'))
        assert ok, (nm, a, b)
    print('    the leave-one-out reference is the same estimator G4 calibrates '
          'OK')

    print('\n[6] G10 BINS')
    bins = _g10_bins(ctx)
    print('    %d nested edges -> %d bins, %d occupied; max order %d'
          % (ctx.nested_idx.shape[0], bins['nb'],
             int(np.unique(bins['fixed']).size), int(bins['order'].max())))
    print('\n[7] DENSITY QUINTILES (G6 clause iii) on %s' % d)
    dd = cliff_rate_by_density(ctx)
    print('    n per bin %s' % dd['n_per_bin'])
    for t in (3.0, 4.0):
        print('    tau=%g rates %s monotone=%s'
              % (t, [round(x, 6) for x in dd['rates'][t]], dd['monotone'][t]))
    print('\n[calibrate] SELF-CHECK PASSED')
    return 0


def _main(argv):
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 90)
    a = list(argv)
    if not a:
        return _selfcheck()

    def _opt(name, default=None, cast=int):
        return cast(a[a.index(name) + 1]) if name in a else default

    nproc = _opt('--nproc', 1)
    B = _opt('--B', None)
    verbose = '--quiet' not in a
    if '--tn2' in a:
        print(run_TN2(B=_opt('--tn2-B', 50), nproc=nproc,
                      verbose=verbose).to_string(index=False))
    if '--g5' in a:
        run_G5(B=B, nproc=nproc, verbose=verbose)
    if '--g6' in a:
        print(run_G6(B=B, nproc=nproc, verbose=verbose).to_string(index=False))
    if '--g7' in a:
        r = run_G7(B=B, nproc=nproc, verbose=verbose)
        print(r['summary'].to_string(index=False))
    if '--g8' in a:
        run_G8(reps=_opt('--g8-reps', None), nproc=nproc,
               rep_block=_opt('--g8-block', 5), verbose=verbose)
    if '--g9' in a:
        run_G9(D=_opt('--g9-D', None), nproc=nproc, verbose=verbose)
    if '--g10' in a:
        print(run_G10(B_N1=_opt('--g10-B', None), nproc=nproc,
                      verbose=verbose).to_string(index=False))
    if '--t02' in a:
        write_T02_gate_rows(verbose=verbose)
    if '--stage4' in a:
        stage4(nproc=nproc, B=B, g8_reps=_opt('--g8-reps', None),
               g9_D=_opt('--g9-D', None), g10_B=_opt('--g10-B', None),
               reuse=('--force' not in a), verbose=verbose)
    return 0


if __name__ == '__main__':
    sys.exit(_main(tuple(sys.argv[1:])))
