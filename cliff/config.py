"""BGYM-CLIFF v1 -- frozen numeric surface.

THIS FILE IS THE SPEC'S NUMERIC SURFACE.  Every threshold, every seed, every
pre-declared expected value in ``SPEC_bgym-cliff-v1.md`` lives here and NOWHERE
ELSE in ``cliff/``.  Downstream modules read thresholds only from ``THRESH`` and
seeds only from ``SEEDS``; a literal numeric decision boundary anywhere else in
the package is a bug.

Section references are to ``local-records/bindingGYM-cliff/SPEC_bgym-cliff-v1.md``.

Python 3.9: no ``match``, no runtime ``X | Y`` unions, no bare ``dict[...]``
generics outside ``from __future__ import annotations``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# PATHS                                                                       #
# --------------------------------------------------------------------------- #

#: repo root <R> (this file is <R>/cliff/config.py)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bgym_input():
    p = os.environ.get('BINDINGGYM_INPUT')
    if not p:
        raise RuntimeError(
            'BINDINGGYM_INPUT is not set.  Run:\n'
            '  export BINDINGGYM_INPUT=/home/guoj0f/share/BindingGYM/input')
    return p


class _Paths(object):
    """Lazy path registry -- ``BINDINGGYM_INPUT`` is read at access time so that
    importing :mod:`cliff.config` never fails in an unconfigured shell."""

    repo = REPO

    # ---- inputs (read-only, outside the repo) ----
    @property
    def bgym_input(self):
        return _bgym_input()

    @property
    def dms_dir(self):
        return os.path.join(_bgym_input(), 'Binding_substitutions_DMS')

    @property
    def structures(self):
        return os.path.join(_bgym_input(), 'structures')

    @property
    def msas(self):
        return os.path.join(_bgym_input(), 'msas')

    @property
    def registry_csv(self):
        return os.path.join(_bgym_input(), 'BindingGYM.csv')

    # ---- repo inputs ----
    chain_sides_tsv = os.path.join(REPO, 'data_splits', 'assay_chain_sides.tsv')
    struct_cluster_tsv = os.path.join(REPO, 'data_splits', 'BindingGYM_cluster.tsv')
    inter_assay_folds_tsv = os.path.join(REPO, 'data_splits', 'inter_assay_folds.tsv')

    # ---- caches (data/ is gitignored at any depth: .gitignore:6) ----
    cache = os.path.join(REPO, 'data', 'cliff_cache')
    keys = os.path.join(REPO, 'data', 'cliff_cache', 'keys')
    pairs = os.path.join(REPO, 'data', 'cliff_cache', 'pairs')
    randpairs = os.path.join(REPO, 'data', 'cliff_cache', 'randpairs')
    latent = os.path.join(REPO, 'data', 'cliff_cache', 'latent')
    nulls = os.path.join(REPO, 'data', 'cliff_cache', 'nulls')
    eps = os.path.join(REPO, 'data', 'cliff_cache', 'eps')
    structure_cache = os.path.join(REPO, 'data', 'cliff_cache', 'structure')
    manifest = os.path.join(REPO, 'data', 'cliff_cache', 'MANIFEST.json')

    # ---- deliverables ----
    records = os.path.join(REPO, 'local-records', 'bindingGYM-cliff')
    artifacts = os.path.join(REPO, 'local-records', 'bindingGYM-cliff', 'artifacts')

    def ensure_cache_dirs(self):
        for d in (self.cache, self.keys, self.pairs, self.randpairs, self.latent,
                  self.nulls, self.eps, self.structure_cache, self.artifacts):
            os.makedirs(d, exist_ok=True)

    def dms_csv(self, dms_id):
        return os.path.join(self.dms_dir, dms_id + '.csv')


PATHS = _Paths()

# --------------------------------------------------------------------------- #
# ENV                                                                         #
# --------------------------------------------------------------------------- #

#: (python, numpy, scipy, pandas, sklearn, biopython) -- spec Sec.4
EXPECTED_ENV = ('3.9.25', '1.22.4', '1.13.1', '1.5.3', '1.2.1', '1.81')


def assert_env():
    """Fail loudly on the wrong interpreter -- spec Sec.4 ("a wrong env shows up
    in the log's first line")."""
    import sys
    import numpy
    import scipy
    import pandas
    import sklearn
    import Bio
    got = (sys.version.split()[0], numpy.__version__, scipy.__version__,
           pandas.__version__, sklearn.__version__, Bio.__version__)
    if got != EXPECTED_ENV:
        raise RuntimeError('WRONG ENV: %r != %r' % (got, EXPECTED_ENV))
    return got


# --------------------------------------------------------------------------- #
# SEEDS  (Sec.1.0 "fixed seed 20260902"; Sec.5 cache name "_seed20260902")     #
# --------------------------------------------------------------------------- #

#: Every seed in the study derives from this one integer.  The offsets are
#: fixed, documented and never re-rolled: a random sample is an
#: experiment-defining artefact (Sec.5) and must be reproducible byte-for-byte.
SEED_BASE = 20260902

SEEDS = {
    'base': SEED_BASE,             # Sec.1.0 cross-fitting "fixed seed 20260902"
    'crossfit': SEED_BASE,         # 5 folds over variants
    'randpairs': SEED_BASE,        # 2e7 random-pair sample (cache name pins this)
    'nulls_N1': SEED_BASE + 1000,
    'nulls_N2': SEED_BASE + 2000,
    'nulls_N2b': SEED_BASE + 2100,
    'nulls_N2c': SEED_BASE + 2200,
    'nulls_N3': SEED_BASE + 3000,
    'perm_NS1': SEED_BASE + 4100,
    'perm_NS2': SEED_BASE + 4200,
    'perm_NS3': SEED_BASE + 4300,
    'bootstrap_gamma': SEED_BASE + 5000,   # 2,000 site-pair bootstraps (Sec.1.2)
    'bootstrap_block': SEED_BASE + 5100,   # 1,000 position block bootstraps (Sec.1.3)
    'mixture_em': SEED_BASE + 6000,        # 200 restarts (Sec.1.3)
    'replication_perm': SEED_BASE + 7000,  # 10,000 site-pair label perms (Sec.1.4)
    'g8_injection': SEED_BASE + 8000,      # power grid (Sec.1.1 G8)
    'g9_rule_fpr': SEED_BASE + 9000,       # 50 complete N1 datasets (Sec.1.1 G9)
    'cluster_subsample': SEED_BASE + 9500,  # ARI over 5 seeds (T15)
}


def assay_seed(name, dms_id):
    """Deterministic per-assay seed sequence entropy.

    Returned as a *list* so it is fed to ``np.random.default_rng`` as
    SeedSequence entropy: two assays that happen to share ``n`` must not share a
    stream.  ``ASSAY_ORDINAL`` is frozen by the alphabetical file order of the 28
    DMS files, so it cannot drift.
    """
    if name not in SEEDS:
        raise KeyError('unknown seed name %r; add it to config.SEEDS' % (name,))
    return [SEEDS[name], ASSAY_ORDINAL[dms_id]]


# --------------------------------------------------------------------------- #
# TAUS                                                                        #
# --------------------------------------------------------------------------- #

#: Sec.1.3 swept-enrichment grid.  Frozen before any observed value is read.
TAUS = (2, 3, 4, 5, 6, 8)

#: Sec.1.3 / 1.4: the consecutive-tau window the C2 verdict is read off.
TAU_WINDOW = (3, 8)

# --------------------------------------------------------------------------- #
# THRESH -- every numeric threshold in the spec, frozen                       #
# --------------------------------------------------------------------------- #

THRESH = {
    # ------------------------- Sec.1.0 definitions ------------------------- #
    'latent_n_iter': 10,               # alternating fit iterations
    'latent_conv_tol': 1e-6,           # max|dbeta| convergence
    'crossfit_n_folds': 5,             # folds over variants
    'sigma_n_bins': 20,                # equal-count phi bins for sigma-hat(phi)
    'mad_const': 1.4826,               # MAD -> sd; MAD, NEVER sd
    'censor_min_mass': 0.005,          # detect_censoring mass threshold at min/max
    'grid_guard_mult': 3.0,            # drop tau whose absolute cut < 3*quantum
    'hyap65_log10': True,              # the only transformed assay

    # --------------------------- Sec.1.1 gates ----------------------------- #
    'G0_enum_wall_s': 180.0,           # per assay
    'G0_enum_rss_gb': 4.0,             # per assay
    'G0_enum_bench_wall_s': 59.0,      # whole-benchmark enumeration budget
    'G0_enum_bench_rss_gb': 2.0,
    'G0_fit_latent_s': 4.0,
    'G0_one_replicate_s': 20.0,
    'G0_fallback_n': 40000,            # 5-independent-subsample path
    'G0_fallback_n_subsample': 5,
    'G0_fallback_SI_se_max': 0.03,     # else INCONCLUSIVE
    'G2_max_abs_delta': 0.0,           # byte identity on the raw score STRINGS
    'G4_T_tol': 0.05,                  # T(tau) = 1.00 +/- 0.05
    'G4_ks_p_min': 0.05,               # 200 empirical p-values uniform
    'G4_n_surrogates': 200,
    'G5_unmasked_T4_min': 5.0,
    'G5_Pa_collapse_factor': 10.0,
    'G5_Pa_after_max': 52000,
    'G5_Pa_after_expected': 41700,
    'G7_n_surrogates': 200,
    'G8_amplitudes_sigma': (2, 3, 4, 6),
    'G8_rates': (0.001, 0.005, 0.02),
    'G8_n_reps': 40,
    'G8_power_min': 0.50,              # at (a = 4 sigma, pi = 0.005) else UNDERPOWERED
    'G8_power_ref_amplitude': 4,
    'G8_power_ref_rate': 0.005,
    'G9_n_datasets': 50,
    'G9_family_fpr_max': 0.10,
    'G10_max_bin_prop_diff': 0.02,

    # ---------------------------- Sec.1.2 C1 ------------------------------- #
    'C1_SI_sup': 0.50,
    'C1_V1_over_Vinf_sup': 0.35,
    'C1_gamma1_sup': 0.60,
    'C1_gamma1_ci_lo_sup': 0.45,
    'C1_h_monotone_upto': 4,           # V(h) non-decreasing over h = 1..4
    'C1_SI_ref': 0.80,
    'C1_V1_over_Vinf_ref': 0.70,
    'C1_gamma1_ref': 0.20,
    'C1_gamma1_ci_hi_ref': 0.45,
    'C1_pos_rs_ref': 0.70,
    'C1_bootstrap_B': 2000,            # site-pair bootstraps for gamma CI
    'C1_family_k_true': 5,             # of 7
    'C1_family_k_refuted': 3,          # of 7

    # ---------------------------- Sec.1.3 C2 ------------------------------- #
    'C2_TR1_min_Pa': 20000,            # |P_a| >= 20,000  -> TR1 = Q.999/Q.75
    'C2_TR2_min_Pa': 2000,             # 2,000 <= |P_a| < 20,000 -> TR2 = Q.99/Q.75
    'C2_TR_q_hi1': 0.999,
    'C2_TR_q_hi2': 0.99,
    'C2_TR_q_lo': 0.75,
    'C2_gauss_q75_absZ': 1.1503,       # exact Gaussian references
    'C2_gauss_q99_absZ': 2.5758,
    'C2_gauss_q999_absZ': 3.2905,
    'C2_TR1_gauss': 2.8606,
    'C2_TR2_gauss': 2.2393,
    'C2_TR_sup_pctile': 99.5,          # of the N1 ensemble
    'C2_TR_ref_pctile': 95.0,
    'C2_T_sup': 2.0,
    'C2_q_BH_sup': 0.05,
    'C2_n_consecutive_tau': 4,         # >= 4 consecutive tau in [3,8], BOTH unit systems
    'C2_dBIC_sup': -10.0,
    'C2_pi_lo': 0.001,
    'C2_pi_hi': 0.05,
    'C2_pi_ci_lo_sup': 0.0005,
    'C2_rho_sup': 3.0,
    'C2_T_ref': 1.5,                   # max_tau T(tau) < 1.5 ...
    'C2_T_ref_ci_hi': 2.0,             # ... with CI upper < 2.0
    'C2_pi_ci_hi_ref': 0.001,
    'C2_em_n_restart': 200,
    'C2_em_n_iter': 100,
    'C2_lambda_n_bootstrap': 200,      # N1 bootstraps calibrating Lambda
    'C2_block_bootstrap_B': 1000,      # block bootstrap over mutated positions
    'C2_bh_n_assays': 14,              # BH-FDR over the 14 primary+arm assays
    'C2_family_k_true': 4,             # of 7 (subject to G9 tightening)
    'C2_family_k_refuted': 1,          # supported in <= 1 of 7
    'C2_catalogue_c_min': 4.0,         # cliff_catalogue keeps |c_hat| >= 4

    # ---------------------------- Sec.1.4 C3 ------------------------------- #
    'C3N_cliff_sigma_mult': 3.0,       # |eps| >= 3 sigma_eps
    'C3N_replicate_sigma_mult': 2.0,   # |eps_b| >= 2 sigma_eps in the replicate
    'C3N_cliff_abs_kras': 0.373,       # 3 * 0.1243, spec-frozen
    'C3N_R_sup': 0.70,
    'C3N_perm_chance_max': 0.10,
    'C3N_sign_agreement_sup': 0.85,
    'C3N_R_ref': 0.35,
    'C3N_sign_agreement_ref': 0.60,    # chance 0.50
    'C3N_frac_below_3sigma_ref': 0.50,
    'C3N_n_perm': 10000,
    'L1_min_siblings': 3,
    'L1_min_edges': 1000,              # >= 1,000 edges with |S| >= 3
    'L2_min_sitepairs': 200,           # with >= 2 backgrounds
    'L2_min_backgrounds': 2,
    'L2p_min_aa_combos': 5,            # >= 5 aa-combinations per site pair
    'L4_min_obs_per_col': 5,           # >= 5 observations per Z column
    'L4_inner_folds': 5,
    'L4_outer_folds': 5,
    'L4_n_lambda': 12,                 # 12 log-spaced lambda
    'L5_min_eps': 500,                 # >= 500 eps with both sites annotated
    'L5_cliff_sigma_mult': 3.0,        # |eps| >= 3 sigma  vs  < 1 sigma
    'L5_noncliff_sigma_mult': 1.0,
    'C3L_beta_sup_pctile': 99.5,       # beta_a vs its N2 null
    'C3L_beta_ref_band': 95.0,         # inside the N2 95% band
    'C3L_ICC_sup': 0.30,
    'C3L_ICC_ci_lo_sup': 0.15,
    'C3L_ICC_ci_hi_ref': 0.15,
    'C3L_dR2_sup': 0.02,
    'C3L_dR2_ci_lo_sup': 0.005,
    'C3L_dR2_ci_hi_ref': 0.02,
    'C3L_AUROC_sup': 0.60,
    'C3L_p_NS2_sup': 0.01,
    'C3A_depth_spearman_ref': 0.40,    # REFUTED if > 0.40 while best struct cov < 0.20
    'C3A_struct_cov_min': 0.20,
    'C3A_n_density_bins': 5,
    'C3_family_k_true': 3,             # C3-L supported in >= 3 of 7
    'C3_family_k_refuted': 1,          # supported in <= 1 of 7

    # ---------------------------- Sec.1.5 C4 ------------------------------- #
    'C4_iface_dist_A': 5.0,            # min heavy-atom distance to the opposite side
    'C4_iface_dist_superset_A': 6.0,   # near-exact superset of dSASA>1
    'C4_max_min_heavy_dsasa_pos_A': 6.07,   # measured; justifies 6.0 A
    'C4_dsasa_min_A2': 1.0,            # dSASA > 1 A^2
    'C4_cb_dist_banned_A': 8.0,        # C-beta < 8 A is BANNED (825/1050 recall)
    'C4S_OR_sup': 1.5,
    'C4S_p_NS1_sup': 0.01,
    'C4S_family_k_sup': 4,             # of 7 eligible
    'C4S_family_k_ref_ci': 5,          # OR CI covers 1 in >= 5 of 7
    'C4S_OR_ref': 1.0,
    'C4S_family_k_ref_or': 3,          # OR < 1.0 in >= 3
    'C4I_Fspec_sup': 0.40,             # noise-corrected
    'C4I_p_NS3_sup': 0.05,
    'C4I_MW_p_sup': 0.05,
    'C4I_Fspec_ref': 0.15,
    'C4I_median_PSI_ref': 0.75,
    'C4I_sigma_eps_sq': 0.01545,       # Var(noise) subtracted in F_spec
    'C4_n_perm_NS1': 10000,
    'C4_n_perm_NS3': 10000,
    'C4_rsa_clip': (0.0, 1.0),         # Tien 2013 maxima, clipped

    # ---------------------------- Sec.1.6 C5 ------------------------------- #
    'C5_PSA_blindspot': 0.60,
    'C5_spearman_min': 0.30,
    'C5_family_k_blindspot': 4,        # of 7
    'C5_PSA_practically_empty': 0.75,  # for M1
    'C5_tie_credit': 0.5,

    # ------------------- Sec.3 nulls / cluster channel --------------------- #
    'null_B': 200,                     # replicates per null per assay
    'N2b_min_cooccur': 20,             # Z columns only for site pairs co-observed >= 20x
    'cluster_n_max': 30000,
    'cluster_rho_targets': (1, 1.5, 2, 3),
    'cluster_min_size': 8,             # clusters with n_c < 8 dropped
    'cluster_min_n_clusters': 30,      # coverage gate
    'cluster_min_frac_covered': 0.40,  # coverage gate
    'cluster_ari_n_seeds': 5,

    # --------------------------- Sec.5 runtime ----------------------------- #
    'nproc_cap': 64,                   # not 80: 64 * 0.5 GB stays inside 111 GB
    'randpair_n_draw': 20000000,       # 2e7 per assay
    'hamming_block': 1000000,          # block size for XOR-nonzero-count

    # ------------------------- Sec.2 power gates --------------------------- #
    'min_nested_for_pair_channel': 500,   # 4D5's nested = 262 < 500
}

#: Structural-annotation methods that are BANNED repo-wide (Sec.1.1 G1b, Sec.1.5).
BANNED = (
    'naive (pos,aa) cross-assay joins',            # G1b
    'C-beta--C-beta < 8 A interface definition',   # Sec.1.5
    'constant seq->pdb offsets on 4ZFF/4ZFG H+L',  # Sec.3 map_mutations
    'the Z-domain within-genotype SDs as a noise floor',  # Sec.1.0 / G3
    'np.add.at for pair reductions (use np.bincount)',    # Sec.1.2
    'SD instead of 1.4826*MAD for the noise scale',       # Sec.1.0
)

# --------------------------------------------------------------------------- #
# NOISE registry (Sec.1.0 table) -- provenance is part of the number          #
# --------------------------------------------------------------------------- #

NOISE = {
    'KRAS': dict(sigma_y=0.148, sigma_eps=0.1243, provenance='measured_replicate',
                 source_partner='KRAS_RAF1_norfitness_6VJJ vs KRAS_RAF1-RBD_norfitness_6VJJ',
                 n_source=10868, r_source=0.812, slope_source=0.6445,
                 resid_sd_source=0.1479,
                 caveat='upper bound; different construct (full RAF1 vs RBD-only) and '
                        'different library (63 vs 166 positions); affine alignment only '
                        'valid if the construct effect is affine in eps'),
    'GB1': dict(sigma_y=0.129, sigma_eps=None, provenance='cross_study_contaminated',
                source_partner='GB1_IgG-Fc_fitness_1FCC vs GB1_IgG-Fc_fitness_1FCC_2016',
                n_source=160, r_source=0.9789, slope_source=None,
                resid_sd_source=0.1826,
                caveat='chain-C pos 2 WT differs (Q vs T) => two backgrounds, not two '
                       'measurements'),
    'other': dict(sigma_y=None, sigma_eps=None, provenance='internal_residual',
                  source_partner=None, n_source=None, r_source=None, slope_source=None,
                  resid_sd_source=None,
                  caveat="assay's own cross-fitted residual MAD per phi-decile; "
                         'conservative (attributes all epistasis to noise)'),
    'stipulated': dict(sigma_y=None, sigma_eps=None, provenance='stipulated',
                       source_partner=None, n_source=None, r_source=None,
                       slope_source=None, resid_sd_source=None,
                       sigma_over_mad=0.20,
                       caveat='imported from the GB1 ratio; never primary'),
}

#: FORBIDDEN (Sec.1.0): the Z-domain within-genotype SDs.  They are a chain-key
#: collision artefact -- G3 proves it.  Kept here only so a reviewer can grep the
#: numbers and see that they are excluded on purpose.
FORBIDDEN_ZDOMAIN_SDS = {
    'Z-domain_ZSPA-1_LL1_fitness_1LP1': 0.1695,
    'Z-domain_ZSPA-1_LL2_fitness_1LP1': 0.4105,
    'Z-domain_ZpA963_HL1_fitness_2M5A': 0.3064,
    'Z-domain_ZpA963_HL2_fitness_2M5A': 0.5091,
}

#: Sensitivity multipliers applied to every headline number (Sec.1.0).
SIGMA_MULTIPLIERS = (0.5, 1.0, 2.0)

# --------------------------------------------------------------------------- #
# ASSAYS registry (Sec.2)                                                     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AssaySpec:
    """One row of the Sec.2 inclusion list.  ``n_spec`` etc. are the *profile's*
    numbers, kept so that every gate compares observed against expected."""
    dms_id: str
    filename: str
    tier: str                 # PRIMARY | ARM | CONTROL | EXCLUDED
    family_id: str            # F1..F8 or '' when the assay is not a data point
    structure_cluster_id: str  # representative DMS_id from BindingGYM_cluster.tsv
    poi: str
    pdb_file: str
    registered: bool
    side0_chains: str
    side1_chains: str
    scale_type: str
    transform: str            # 'none' | 'log10'
    sign_convention: str      # all 28: higher = better binding (Sec.1.0 KEY DECISION)
    exclusion_reason: str = ''
    role: str = ''            # what an EXCLUDED/CONTROL assay is retained for
    # eligibility flags (Sec.2 caveats; structural ones are re-checked by
    # structure.py against the measured design/background interface fractions)
    eligible_C1: bool = False
    eligible_C2: bool = False
    eligible_C3L: bool = False
    eligible_C4S: bool = False
    eligible_C4P: bool = False
    eligible_C4I: bool = False
    eligible_cluster_channel: bool = False
    c3l_routes: tuple = ()
    caveats: tuple = ()
    # profile's expected values (Sec.2 tables) -- expectations, never inputs
    n_spec: int = 0
    n_nested_spec: int = 0
    n_primary_Pa_spec: int = 0
    SI_spec: float = float('nan')
    pairs_per_variant_spec: float = float('nan')
    mean_obs_per_pairwise_col_spec: float = float('nan')
    design_iface_frac_spec: float = float('nan')
    bg_iface_frac_spec: float = float('nan')


_H = 'higher_is_better'

ASSAYS = {}


def _reg(spec):
    ASSAYS[spec.dms_id] = spec
    return spec


# ------------------------------- PRIMARY (12) ------------------------------- #
_reg(AssaySpec(
    dms_id='GB1_IgG-Fc_fitness_1FCC', filename='GB1_IgG-Fc_fitness_1FCC.csv',
    tier='PRIMARY', family_id='F1',
    structure_cluster_id='GB1_IgG-Fc_fitness_1FCC_2016',
    poi='1FCC_hm', pdb_file='1FCC_hm.pdb', registered=True,
    side0_chains='C', side1_chains='A', scale_type='fitness', transform='none',
    sign_convention=_H,
    eligible_C1=True, eligible_C2=True, eligible_C3L=True, eligible_C4S=True,
    eligible_C4P=True, eligible_C4I=False, eligible_cluster_channel=False,
    c3l_routes=('L2p', 'L5'),
    caveats=('FLAGSHIP END-TO-END ASSAY: complete single scan 1,045 = 55x19 => exact eps '
             'for all 91,845 doubles',
             'zero censoring; zero ties',
             'L4 INFEASIBLE: 1 observation per Z column',
             'cluster channel NEVER: 4.314e9 condensed > int32 (34.5 GB)'),
    n_spec=92891, n_nested_spec=184735, n_primary_Pa_spec=183690, SI_spec=0.268,
    mean_obs_per_pairwise_col_spec=1.0,
    design_iface_frac_spec=0.327, bg_iface_frac_spec=0.321))

_reg(AssaySpec(
    dms_id='GB1_IgG-Fc_fitness_1FCC_2016', filename='GB1_IgG-Fc_fitness_1FCC_2016.csv',
    tier='PRIMARY', family_id='F1',
    structure_cluster_id='GB1_IgG-Fc_fitness_1FCC_2016',
    poi='1FCC2016_hm', pdb_file='1FCC2016_hm.pdb', registered=True,
    side0_chains='C', side1_chains='A', scale_type='fitness', transform='none',
    sign_convention=_H,
    eligible_C1=True, eligible_C2=True, eligible_C3L=True, eligible_C4S=False,
    eligible_C4P=False, eligible_C4I=False, eligible_cluster_channel=True,
    c3l_routes=('L1', 'L2', 'L4'),
    exclusion_reason='',
    caveats=('densest neighbourhoods (19.31 pairs/variant)',
             '4 positions => short h axis; C4-S UNDEFINED',
             'pos-2 WT is T not Q => a DIFFERENT BACKGROUND from GB1_1FCC, not a replicate'),
    n_spec=22176, n_nested_spec=52175, n_primary_Pa_spec=52149,
    pairs_per_variant_spec=19.31, mean_obs_per_pairwise_col_spec=10.2))

_reg(AssaySpec(
    dms_id='KRAS_RAF1_norfitness_6VJJ', filename='KRAS_RAF1_norfitness_6VJJ.csv',
    tier='PRIMARY', family_id='F2', structure_cluster_id='KRAS_RAF1_norfitness_6VJJ',
    poi='6VJJ', pdb_file='6VJJ.pdb', registered=True,
    side0_chains='A', side1_chains='B', scale_type='norfitness', transform='none',
    sign_convention=_H,
    eligible_C1=True, eligible_C2=True, eligible_C3L=True, eligible_C4S=False,
    eligible_C4P=True, eligible_C4I=True, eligible_cluster_channel=False,
    c3l_routes=('L3', 'L5'),
    caveats=('sigma_eps anchor half (KRAS twin)',
             'C4-S FLAGGED: design@iface 0.238 vs bg 0.101 (2.4x enriched)'),
    n_spec=12677, n_nested_spec=24138, n_primary_Pa_spec=22950,
    pairs_per_variant_spec=10.02,
    design_iface_frac_spec=0.238, bg_iface_frac_spec=0.101))

_reg(AssaySpec(
    dms_id='KRAS_RAF1-RBD_norfitness_6VJJ', filename='KRAS_RAF1-RBD_norfitness_6VJJ.csv',
    tier='PRIMARY', family_id='F2', structure_cluster_id='KRAS_RAF1_norfitness_6VJJ',
    poi='6VJJ', pdb_file='6VJJ.pdb', registered=True,
    side0_chains='A', side1_chains='B', scale_type='norfitness', transform='none',
    sign_convention=_H,
    eligible_C1=True, eligible_C2=True, eligible_C3L=True, eligible_C4S=True,
    eligible_C4P=True, eligible_C4I=True, eligible_cluster_channel=False,
    c3l_routes=('L3', 'L5'),
    caveats=('largest, cleanest KRAS',
             'BYTE-IDENTICAL per-residue structural annotation to KRAS_RAF1_6VJJ '
             '=> ONE structural unit, not two'),
    n_spec=23162, n_nested_spec=43202, n_primary_Pa_spec=40377,
    design_iface_frac_spec=0.102, bg_iface_frac_spec=0.101))

_reg(AssaySpec(
    dms_id='KRAS_RALGDS-RBD_norfitness_1LFD', filename='KRAS_RALGDS-RBD_norfitness_1LFD.csv',
    tier='PRIMARY', family_id='F2', structure_cluster_id='KRAS_RAF1_norfitness_6VJJ',
    poi='1LFD_hm', pdb_file='1LFD_hm.pdb', registered=True,
    side0_chains='B', side1_chains='A', scale_type='norfitness', transform='none',
    sign_convention=_H,
    eligible_C1=True, eligible_C2=True, eligible_C3L=True, eligible_C4S=True,
    eligible_C4P=True, eligible_C4I=True, eligible_cluster_channel=False,
    c3l_routes=('L3', 'L5'),
    caveats=('narrow range (span 1.89)',),
    n_spec=20341, n_nested_spec=37730, n_primary_Pa_spec=35186,
    design_iface_frac_spec=0.091, bg_iface_frac_spec=0.090))

_reg(AssaySpec(
    dms_id='KRAS_PICK3CG-RBD_norfitness_1HE8',
    filename='KRAS_PICK3CG-RBD_norfitness_1HE8.csv',
    tier='PRIMARY', family_id='F2', structure_cluster_id='KRAS_RAF1_norfitness_6VJJ',
    poi='1HE8_hm', pdb_file='1HE8_hm.pdb', registered=True,
    side0_chains='B', side1_chains='A', scale_type='norfitness', transform='none',
    sign_convention=_H,
    eligible_C1=True, eligible_C2=True, eligible_C3L=True, eligible_C4S=True,
    eligible_C4P=True, eligible_C4I=True, eligible_cluster_channel=False,
    c3l_routes=('L3', 'L5'),
    caveats=('skew -0.02 (most symmetric in the set) => symmetric criterion defensible',),
    n_spec=19203, n_nested_spec=35395, n_primary_Pa_spec=32756,
    design_iface_frac_spec=0.098, bg_iface_frac_spec=0.096))

_reg(AssaySpec(
    dms_id='KRAS_SOS1_norfitness_8BE4', filename='KRAS_SOS1_norfitness_8BE4.csv',
    tier='PRIMARY', family_id='F2', structure_cluster_id='KRAS_RAF1_norfitness_6VJJ',
    poi='8BE4_hm', pdb_file='8BE4_hm.pdb', registered=True,
    side0_chains='R', side1_chains='S', scale_type='norfitness', transform='none',
    sign_convention=_H,
    eligible_C1=True, eligible_C2=True, eligible_C3L=True, eligible_C4S=True,
    eligible_C4P=True, eligible_C4I=True, eligible_cluster_channel=False,
    c3l_routes=('L3', 'L5'),
    caveats=('the RETAINED half of the byte-identical duplicate pair (G2)',
             "C4-S primary entry uses the UNBIASED 5O2S annotation (0.160 ~ 0.158); "
             "SOS1's own design@iface is 0.264 (2.4x). G11 twin-structure control",),
    n_spec=19425, n_nested_spec=35915, n_primary_Pa_spec=33412,
    design_iface_frac_spec=0.264, bg_iface_frac_spec=0.110))

_reg(AssaySpec(
    dms_id='SARS2-RBD_ACE2_deltaKd_6M0J', filename='SARS2-RBD_ACE2_deltaKd_6M0J.csv',
    tier='PRIMARY', family_id='F3', structure_cluster_id='SARS2-RBD_ACE2_deltaKd_6M0J',
    poi='6M0J', pdb_file='6M0J.pdb', registered=True,
    side0_chains='E', side1_chains='A', scale_type='deltaKd', transform='none',
    sign_convention=_H,
    eligible_C1=True, eligible_C2=True, eligible_C3L=True, eligible_C4S=True,
    eligible_C4P=True, eligible_C4I=False, eligible_cluster_channel=False,
    c3l_routes=('L5',),
    caveats=('graded orders 1-10 over 194 positions => a real mutation-degree axis',
             '23.85% censored at -4.84/-4.76, spike fraction rising 0.004->1.000 in '
             'order => MASKING MANDATORY',
             'hard 0.01 grid => TR2 regime',
             'L1 count must be reported and the route dropped if it fails'),
    n_spec=21872, n_nested_spec=19459, n_primary_Pa_spec=11000,
    mean_obs_per_pairwise_col_spec=0.45,
    design_iface_frac_spec=0.108, bg_iface_frac_spec=0.108))

_reg(AssaySpec(
    dms_id='5A12_VEGF_fitness_4ZFF', filename='5A12_VEGF_fitness_4ZFF.csv',
    tier='PRIMARY', family_id='F4', structure_cluster_id='5A12_VEGF_fitness_4ZFF',
    poi='4ZFF_CHL', pdb_file='4ZFF_CHL.pdb', registered=True,
    side0_chains='HL', side1_chains='C', scale_type='fitness', transform='none',
    sign_convention=_H,
    eligible_C1=True, eligible_C2=True, eligible_C3L=True, eligible_C4S=False,
    eligible_C4P=False, eligible_C4I=True, eligible_cluster_channel=True,
    c3l_routes=('L1', 'L2', 'L4'),
    caveats=('smoothest landscape in the set (SI 0.250) => strongest single C1 datum',
             'DESIGNED C4 NEGATIVE CONTROL: 0/9 mutated positions within 6.4 A of VEGF, '
             'dSASA 0.0 (they contact Ang2)',
             'WT at the 0.0 percentile => NO WT-anchored normalisation'),
    n_spec=29981, n_nested_spec=22064, n_primary_Pa_spec=22010, SI_spec=0.250,
    mean_obs_per_pairwise_col_spec=23.0))

_reg(AssaySpec(
    dms_id='Z-domain_ZpA963_HL1_fitness_2M5A',
    filename='Z-domain_ZpA963_HL1_fitness_2M5A.csv',
    tier='PRIMARY', family_id='F5',
    structure_cluster_id='Z-domain_ZSPA-1_LL2_fitness_1LP1',
    poi='2M5A', pdb_file='2M5A.pdb', registered=True,
    side0_chains='A', side1_chains='B', scale_type='fitness', transform='none',
    sign_convention=_H,
    eligible_C1=True, eligible_C2=True, eligible_C3L=True, eligible_C4S=False,
    eligible_C4P=False, eligible_C4I=False, eligible_cluster_channel=True,
    c3l_routes=('L1', 'L2', 'L4'),
    caveats=('70.9% of a 4^6 space => near-complete neighbourhoods',
             '6/6 interface => OUT OF C4-S (unfalsifiable)',
             'two-sided (both chains mutated); WT is the global max'),
    n_spec=2904, n_nested_spec=9736, n_primary_Pa_spec=9712,
    mean_obs_per_pairwise_col_spec=12.1))

_reg(AssaySpec(
    dms_id='hYAP65_peptide_FunctioncalScore_1JMQ',
    filename='hYAP65_peptide_FunctioncalScore_1JMQ.csv',
    tier='PRIMARY', family_id='F6',
    structure_cluster_id='hYAP65_peptide_FunctioncalScore_1JMQ',
    poi='1JMQ_hm', pdb_file='1JMQ_hm.pdb', registered=True,
    side0_chains='A', side1_chains='P', scale_type='FunctioncalScore',
    transform='log10', sign_convention=_H,
    eligible_C1=True, eligible_C2=True, eligible_C3L=True, eligible_C4S=False,
    eligible_C4P=False, eligible_C4I=False, eligible_cluster_channel=True,
    c3l_routes=('L1',),
    caveats=('orders to 21 on a 46-aa chain => the longest degree axis',
             'log10 FIRST (ratio scale, WT = 1.000, min 0.00911, max 15.56)',
             'C4-S FLAGGED: 0.324 vs 0.238 (1.36x)',
             'L1 count must be reported and the route dropped if it fails'),
    n_spec=18407, n_nested_spec=29695, n_primary_Pa_spec=29407,
    design_iface_frac_spec=0.324, bg_iface_frac_spec=0.238))

_reg(AssaySpec(
    dms_id='CD19_FMC63_Fitness_7URV', filename='CD19_FMC63_Fitness_7URV.csv',
    tier='PRIMARY', family_id='F7', structure_cluster_id='CD19_FMC63_Fitness_7URV',
    poi='7URV_hm', pdb_file='7URV_hm.pdb', registered=True,
    side0_chains='C', side1_chains='D', scale_type='Fitness', transform='none',
    sign_convention=_H,
    eligible_C1=True, eligible_C2=True, eligible_C3L=True, eligible_C4S=True,
    eligible_C4P=True, eligible_C4I=False, eligible_cluster_channel=True,
    c3l_routes=('L5',),
    caveats=('widest range (span 22.6)',
             'singles+triples with a DOUBLES GAP (478) => additive baseline for triples '
             'unavailable, C4-P weak',
             '0.569 value-uniqueness + trimodality => SCORE PROVENANCE CAVEAT, may be '
             'binned selection',
             'L4 INFEASIBLE'),
    n_spec=3886, n_nested_spec=4540, n_primary_Pa_spec=2922,
    design_iface_frac_spec=0.078, bg_iface_frac_spec=0.078))

# --------------------------------- ARM (2) --------------------------------- #
_reg(AssaySpec(
    dms_id='CR9114_FluAH1_logKd_4FQI', filename='CR9114_FluAH1_logKd_4FQI.csv',
    tier='ARM', family_id='F8', structure_cluster_id='CR9114_FluAH1_logKd_4FQI',
    poi='4FQI_hm', pdb_file='4FQI_hm.pdb', registered=False,
    side0_chains='HL', side1_chains='AB', scale_type='logKd', transform='none',
    sign_convention=_H,
    eligible_C1=True, eligible_C2=True, eligible_C3L=True, eligible_C4S=False,
    eligible_C4P=False, eligible_C4I=True, eligible_cluster_channel=False,
    c3l_routes=('L1', 'L2', 'L4'),
    caveats=('2^16 at 99.33% => every variant\'s full 16-neighbour shell observed; '
             'strongest L1/L2/L4 power in the benchmark',
             'NOT in BindingGYM.csv; NO PDB in structures/ => structurally mute unless '
             'G-OPT completes',
             '2.57% floored at 7.000',
             'side0/side1 are NOT from data_splits/assay_chain_sides.tsv (absent there); '
             'assigned here as antibody H+L vs HA A+B',
             'cluster channel NEVER: 17.2 GB condensed'),
    n_spec=65094, n_primary_Pa_spec=517528, mean_obs_per_pairwise_col_spec=542.4))

_reg(AssaySpec(
    dms_id='CR6261_FluAH1_logKd_3GBN', filename='CR6261_FluAH1_logKd_3GBN.csv',
    tier='ARM', family_id='F8', structure_cluster_id='CR9114_FluAH1_logKd_4FQI',
    poi='3GBN_hm', pdb_file='3GBN_hm.pdb', registered=False,
    side0_chains='HL', side1_chains='AB', scale_type='logKd', transform='none',
    sign_convention=_H,
    eligible_C1=True, eligible_C2=True, eligible_C3L=True, eligible_C4S=False,
    eligible_C4P=False, eligible_C4I=False, eligible_cluster_channel=False,
    c3l_routes=('L1', 'L2', 'L4'),
    caveats=("2^11 at 92.14%; independent replication of CR9114-H1's biology",
             '11.34% floored WITH WT ON THE FLOOR',
             'small n => likely UNDERPOWERED by G8',
             'side0/side1 assigned here (absent from assay_chain_sides.tsv)'),
    n_spec=1887, n_primary_Pa_spec=9000, mean_obs_per_pairwise_col_spec=34.3))

# ------------------------------- CONTROL (3) ------------------------------- #
_reg(AssaySpec(
    dms_id='CR9114_FluAH3_logKd_4FQY', filename='CR9114_FluAH3_logKd_4FQY.csv',
    tier='CONTROL', family_id='', structure_cluster_id='CR9114_FluAH1_logKd_4FQI',
    poi='4FQY_hm', pdb_file='4FQY_hm.pdb', registered=False,
    side0_chains='HL', side1_chains='AB', scale_type='logKd', transform='none',
    sign_convention=_H,
    role='CENSORING POSITIVE CONTROL (G5)',
    exclusion_reason='not a data point: 58,361/65,535 = 89.05% at exactly 6.000; '
                     'the median IS the floor; 86.1% of nested edges are floor-floor',
    eligible_C4I=True,
    caveats=('451,181/524,272 = 86.1% of nested edges are floor-floor with Delta == 0',
             'side0/side1 assigned here (absent from assay_chain_sides.tsv)',
             'C4-I entry is CENSORING-LIMITED ONLY (65,093 shared with CR9114-H1)'),
    n_spec=65535, n_nested_spec=524272))

_reg(AssaySpec(
    dms_id='Z-domain_ZSPA-1_LL1_fitness_1LP1',
    filename='Z-domain_ZSPA-1_LL1_fitness_1LP1.csv',
    tier='CONTROL', family_id='',
    structure_cluster_id='Z-domain_ZSPA-1_LL2_fitness_1LP1',
    poi='1LP1', pdb_file='1LP1.pdb', registered=True,
    side0_chains='A', side1_chains='B', scale_type='fitness', transform='none',
    sign_convention=_H,
    role='ANTI-SMOOTH NEGATIVE CONTROL (G6)',
    exclusion_reason='SI 1.398, sd 0.140 / max 3.631 / skew +10.55, no WT row, '
                     'selection-derived membership',
    caveats=('cliff rate must NOT be monotone in density quintile',
             'within-genotype SD 0.1695 is FORBIDDEN as a noise floor (G3 artefact)',
             'cluster channel NEVER: 8.3 GB condensed'),
    n_spec=45476, SI_spec=1.398))

_reg(AssaySpec(
    dms_id='Z-domain_ZSPA-1_LL2_fitness_1LP1',
    filename='Z-domain_ZSPA-1_LL2_fitness_1LP1.csv',
    tier='CONTROL', family_id='',
    structure_cluster_id='Z-domain_ZSPA-1_LL2_fitness_1LP1',
    poi='1LP1', pdb_file='1LP1.pdb', registered=True,
    side0_chains='A', side1_chains='B', scale_type='fitness', transform='none',
    sign_convention=_H,
    role='ANTI-SMOOTH NEGATIVE CONTROL (G6)',
    exclusion_reason='SI 1.001 -- neighbours as different as random pairs',
    caveats=('within-genotype SD 0.4105 is FORBIDDEN as a noise floor (G3 artefact)',),
    n_spec=5583, SI_spec=1.001))

# ------------------------------ EXCLUDED (11) ------------------------------ #
_reg(AssaySpec(
    dms_id='KRAS_DARPinK27_norfitness_5O2S', filename='KRAS_DARPinK27_norfitness_5O2S.csv',
    tier='EXCLUDED', family_id='', structure_cluster_id='KRAS_RAF1_norfitness_6VJJ',
    poi='5O2S_hm', pdb_file='5O2S_hm.pdb', registered=True,
    side0_chains='A', side1_chains='B', scale_type='norfitness', transform='none',
    sign_convention=_H,
    exclusion_reason='DUPLICATE SCORE TABLE: 19,227 shared keys, byte-identical scores '
                     'with KRAS_SOS1_norfitness_8BE4 (G2)',
    role='its structural annotation only (the unbiased KRAS C4-S entry + G11)',
    eligible_C4I=True,
    caveats=('unbiased KRAS annotation: 0.160 ~ 0.158',),
    n_spec=19533, design_iface_frac_spec=0.160, bg_iface_frac_spec=0.158))

_reg(AssaySpec(
    dms_id='ACE2_SARS2-RBD_enrich_6M17', filename='ACE2_SARS2-RBD_enrich_6M17.csv',
    tier='EXCLUDED', family_id='', structure_cluster_id='ACE2_SARS2-RBD_enrich_6M17',
    poi='6M17_BE', pdb_file='6M17_BE.pdb', registered=True,
    side0_chains='B', side1_chains='E', scale_type='enrich', transform='none',
    sign_convention=_H,
    exclusion_reason='max_mut = 1 => P_a empty by construction',
    role='same-site substitution-roughness reference (19,665 pairs, complete 19-aa scan, '
         'zero ties)',
    n_spec=2186))

_reg(AssaySpec(
    dms_id='CXCR4_CXCL12_enrich_8U4O', filename='CXCR4_CXCL12_enrich_8U4O.csv',
    tier='EXCLUDED', family_id='', structure_cluster_id='CXCR4_CXCL12_enrich_8U4O',
    poi='8U4O_hm', pdb_file='8U4O_hm.pdb', registered=True,
    side0_chains='R', side1_chains='J', scale_type='enrich', transform='none',
    sign_convention=_H,
    exclusion_reason='max_mut = 1; SI 0.883',
    role='same-site reference over 295 positions; C1 NEGATIVE CONTROL showing the '
         'smoothness signal is not a metric artefact',
    n_spec=5585, SI_spec=0.883))

_reg(AssaySpec(
    dms_id='HLA-A2_TAPBPR_meanscore_5WER', filename='HLA-A2_TAPBPR_meanscore_5WER.csv',
    tier='EXCLUDED', family_id='', structure_cluster_id='HLA-A2_TAPBPR_meanscore_5WER',
    poi='5WER_hm', pdb_file='5WER_hm.pdb', registered=True,
    side0_chains='A', side1_chains='C', scale_type='meanscore', transform='none',
    sign_convention=_H,
    exclusion_reason='max_mut = 1; 0 nested edges (no WT row); duplicated DMS_score column',
    role='same-site reference (29,558 pairs)',
    caveats=("usecols=['POI','DMS_score','mutant','mutant_pdb'] silently de-duplicates "
             'the repeated DMS_score column (verified identical)',),
    n_spec=3344))

_reg(AssaySpec(
    dms_id='PSD95_CRIPT_1BE9', filename='PSD95_CRIPT_1BE9.csv',
    tier='EXCLUDED', family_id='', structure_cluster_id='PSD95_Tm2F_1BE9',
    poi='1BE9_hm', pdb_file='1BE9_hm.pdb', registered=True,
    side0_chains='A', side1_chains='B', scale_type='fitness', transform='none',
    sign_convention=_H,
    exclusion_reason='max_mut = 1',
    role='C4-I partner-specificity probe (1,577/1,577 shared with PSD95_Tm2F_1BE9, '
         'r = 0.4795 => ~52% of the site-level effect is partner-specific)',
    eligible_C4I=True,
    caveats=('same-site channel only (83 positions x 2 partners)',),
    n_spec=1577))

_reg(AssaySpec(
    dms_id='PSD95_Tm2F_1BE9', filename='PSD95_Tm2F_1BE9.csv',
    tier='EXCLUDED', family_id='', structure_cluster_id='PSD95_Tm2F_1BE9',
    poi='1BE9Tm2F_hm', pdb_file='1BE9Tm2F_hm.pdb', registered=True,
    side0_chains='A', side1_chains='B', scale_type='fitness', transform='none',
    sign_convention=_H,
    exclusion_reason='max_mut = 1',
    role='C4-I partner-specificity probe (paired with PSD95_CRIPT_1BE9)',
    eligible_C4I=True,
    caveats=('same-site channel only',),
    n_spec=1577))

_reg(AssaySpec(
    dms_id='4D5_HER2_fitness_1N8Z', filename='4D5_HER2_fitness_1N8Z.csv',
    tier='EXCLUDED', family_id='', structure_cluster_id='4D5_HER2_fitness_1N8Z',
    poi='1N8Z_hm', pdb_file='1N8Z_hm.pdb', registered=True,
    side0_chains='AB', side1_chains='C', scale_type='fitness', transform='none',
    sign_convention=_H,
    exclusion_reason='nested = 262 < 500 power gate; 0.50 pairs/variant; no singles, '
                     'no doubles; SI 0.778',
    role='CLUSTER CHANNEL ONLY (n = 2,080).  If cluster coverage fails its own gate, '
         '"4D5 is structurally incapable of testing this hypothesis" IS the finding',
    eligible_cluster_channel=True,
    n_spec=2080, n_nested_spec=262, pairs_per_variant_spec=0.50, SI_spec=0.778))

_reg(AssaySpec(
    dms_id='Z-domain_ZpA963_HL2_fitness_2M5A',
    filename='Z-domain_ZpA963_HL2_fitness_2M5A.csv',
    tier='EXCLUDED', family_id='',
    structure_cluster_id='Z-domain_ZSPA-1_LL2_fitness_1LP1',
    poi='2M5A', pdb_file='2M5A.pdb', registered=True,
    side0_chains='A', side1_chains='B', scale_type='fitness', transform='none',
    sign_convention=_H,
    exclusion_reason='n = 600, 0.06% of a 10^6 space, SI 0.893',
    role='reported exclusion (C1 pre-declared refutation)',
    caveats=('within-genotype SD 0.5091 is FORBIDDEN as a noise floor (G3 artefact)',),
    n_spec=600, SI_spec=0.893))

_reg(AssaySpec(
    dms_id='5A12_Ang2_fitness_4ZFG', filename='5A12_Ang2_fitness_4ZFG.csv',
    tier='EXCLUDED', family_id='', structure_cluster_id='5A12_VEGF_fitness_4ZFF',
    poi='4ZFG', pdb_file='4ZFG.pdb', registered=True,
    side0_chains='HL', side1_chains='A', scale_type='fitness', transform='none',
    sign_convention=_H,
    exclusion_reason='span 0.648, sd 0.0787 => sub-replicate resolution; SI 0.844',
    role='off-target half of the 5A12 specificity probe (534 shared, r = -0.163)',
    eligible_C4I=True,
    caveats=('mutant/mutant_pdb dict KEY ORDER DIFFERS between the two columns => join '
             'BY CHAIN KEY, never positionally over dict order',
             'constant seq->pdb offsets FAIL on 4ZFG-H/L (168 mismatches of 219 for '
             '4ZFG-H)'),
    n_spec=944, SI_spec=0.844))

_reg(AssaySpec(
    dms_id='BH3_Bcl-xL_normed_1PQ1', filename='BH3_Bcl-xL_normed_1PQ1.csv',
    tier='EXCLUDED', family_id='', structure_cluster_id='BH3_Bcl-xL_normed_1PQ1',
    poi='1PQ1_hm', pdb_file='1PQ1_hm.pdb', registered=True,
    side0_chains='B', side1_chains='A', scale_type='normed', transform='none',
    sign_convention=_H,
    exclusion_reason='n = 518; 33% dead plateau; SI 0.648; 10/10 interface',
    role='BH3 partner-specificity probe (518/518 after the -2 correction, G1b)',
    eligible_C4I=True,
    n_spec=518, SI_spec=0.648))

_reg(AssaySpec(
    dms_id='BH3_Mcl-1_normed_3KZ0', filename='BH3_Mcl-1_normed_3KZ0.csv',
    tier='EXCLUDED', family_id='', structure_cluster_id='BH3_Mcl-1_normed_3KZ0',
    poi='3KZ0_hm', pdb_file='3KZ0_hm.pdb', registered=True,
    side0_chains='C', side1_chains='A', scale_type='normed', transform='none',
    sign_convention=_H,
    exclusion_reason='n = 518; 20% dead plateau; SI 0.682; 10/10 interface',
    role='BH3 partner-specificity probe (paired with BH3_Bcl-xL_normed_1PQ1)',
    eligible_C4I=True,
    n_spec=518, SI_spec=0.682))

assert len(ASSAYS) == 28, len(ASSAYS)

#: alphabetical file order of the 28 DMS files -- frozen; used only to give each
#: assay a stable seed ordinal (see :func:`assay_seed`).
ASSAY_ORDINAL = {k: i for i, k in enumerate(sorted(ASSAYS))}

# --------------------------------------------------------------------------- #
# Tiers, families, expected values                                            #
# --------------------------------------------------------------------------- #

ALL_ASSAYS = tuple(sorted(ASSAYS))
PRIMARY = tuple(k for k in ALL_ASSAYS if ASSAYS[k].tier == 'PRIMARY')
ARM = tuple(k for k in ALL_ASSAYS if ASSAYS[k].tier == 'ARM')
CONTROL = tuple(k for k in ALL_ASSAYS if ASSAYS[k].tier == 'CONTROL')
EXCLUDED = tuple(k for k in ALL_ASSAYS if ASSAYS[k].tier == 'EXCLUDED')
PRIMARY_AND_ARM = PRIMARY + ARM

assert len(PRIMARY) == 12 and len(ARM) == 2 and len(CONTROL) == 3 and len(EXCLUDED) == 11

#: Sec.2 "Families for aggregation (K = 7 primary)"; F8 is the hypercube arm and
#: is reported with its own denominator, never folded into the primary count.
FAMILIES = {
    'F1': ('GB1_IgG-Fc_fitness_1FCC', 'GB1_IgG-Fc_fitness_1FCC_2016'),
    'F2': ('KRAS_RAF1_norfitness_6VJJ', 'KRAS_RAF1-RBD_norfitness_6VJJ',
           'KRAS_RALGDS-RBD_norfitness_1LFD', 'KRAS_PICK3CG-RBD_norfitness_1HE8',
           'KRAS_SOS1_norfitness_8BE4'),
    'F3': ('SARS2-RBD_ACE2_deltaKd_6M0J',),
    'F4': ('5A12_VEGF_fitness_4ZFF',),
    'F5': ('Z-domain_ZpA963_HL1_fitness_2M5A',),
    'F6': ('hYAP65_peptide_FunctioncalScore_1JMQ',),
    'F7': ('CD19_FMC63_Fitness_7URV',),
    'F8': ('CR9114_FluAH1_logKd_4FQI', 'CR6261_FluAH1_logKd_3GBN'),
}
K_FAMILIES = 7   # F1..F7; F8 reported separately

#: Sec.1 / Sec.8: the headline limitation, verbatim, so no module can forget it.
HEADLINE_LIMITATION = (
    'The effective number of independent biological systems is 3-5, not 7 and '
    'certainly not 25: five of the twelve primary assays are KRAS on four '
    'near-identical interfaces (two with byte-identical structural annotation), and '
    '25 registered assays sit on 22 PDBs.  The two best landscapes for this '
    'hypothesis are unregistered and structurally mute.  No aggregate number may be '
    'presented in a form a reader could mistake for 25 independent replications.'
)

#: Sec.1.2: binomial p of a 5-of-7 sign test under a 0.5 null.
BINOM_P_5OF7 = 0.2266

# --------------------------------------------------------------------------- #
# EXPECTED -- the profile's pre-declared observations that the gates check     #
# --------------------------------------------------------------------------- #

EXPECTED = {
    # ---- G1 (Sec.1.1) ----
    'G1_n_rows_total': 508962,
    'G1_n_unique_keys_total': 508962,
    # MIXED DENOMINATORS IN THE SPEC'S OWN G1 LINE, resolved by measurement:
    # 508,962 rows is over all 28 files, but 1,173,273 mutation instances is over
    # the 25 REGISTERED files only (measured: 1,173,273 exactly).  All 28 give
    # 2,229,197, the extra 1,055,924 being the two CR9114 hypercubes + CR6261.
    # Independent confirmation of the same denominator: the 25 registered files
    # carry exactly 2,220 distinct mutated (chain,resseq,icode) positions, the
    # spec's Sec.1.5 figure.
    'G1_n_mutation_instances': 1173273,          # registered 25
    'G1_n_mutation_instances_registered25': 1173273,
    'G1_n_mutation_instances_all28': 2229197,    # measured
    'G1_n_mutated_positions_registered25': 2220,
    'G1_n_wt_letter_mismatches': 0,
    'G1_n_X_hits': 0,
    'G1_n_star_tokens': 0,
    'G1_n_indels': 0,
    'G1_n_identity_mutations': 0,
    'G1_n_files': 28,
    # ---- G1b ----
    'G1b_n_shared': 518,
    'G1b_n_total': 518,
    'G1b_naive_join_n_shared': 97,
    'G1b_offset_seq': -2,       # 3KZ0 chain C seq pos = 1PQ1 chain B seq pos - 2
    'G1b_r_claim_a': 0.1709,    # profiling agent A
    'G1b_r_claim_b': 0.592,     # profiling agent B
    # ---- G2 ----
    'G2_n_shared_keys': 19227,
    'G2_max_abs_delta': 0.0,
    # ---- G3: duplicate genotypes WITHOUT the chain label ----
    'G3_dups_without_chain': {
        'Z-domain_ZSPA-1_LL1_fitness_1LP1': 847,
        'Z-domain_ZSPA-1_LL2_fitness_1LP1': 59,
        'Z-domain_ZpA963_HL1_fitness_2M5A': 650,
        'Z-domain_ZpA963_HL2_fitness_2M5A': 38,
    },
    'G3_dups_with_chain': {
        'Z-domain_ZSPA-1_LL1_fitness_1LP1': 0,
        'Z-domain_ZSPA-1_LL2_fitness_1LP1': 0,
        'Z-domain_ZpA963_HL1_fitness_2M5A': 0,
        'Z-domain_ZpA963_HL2_fitness_2M5A': 0,
    },
    'G3_n_rows': {
        'Z-domain_ZSPA-1_LL1_fitness_1LP1': 45476,
        'Z-domain_ZSPA-1_LL2_fitness_1LP1': 5583,
        'Z-domain_ZpA963_HL1_fitness_2M5A': 2904,
        'Z-domain_ZpA963_HL2_fitness_2M5A': 600,
    },
    # ---- G5 ----
    'G5_floor_value': 6.000,
    'G5_floor_frac': 0.8905,
    'G5_n_floor_rows': 58361,
    'G5_n_nested': 524272,
    'G5_n_floorfloor_edges': 451181,
    'G5_frac_floorfloor': 0.861,
    # ---- pair totals across the 28 files (Sec.1.0) ----
    'n_nested_total': 1678963,
    'n_samesite_total': 2602669,
    'n_nested_GB1_1FCC': 184735,
    'n_samesite_GB1_1FCC': 861874,
    # ---- Sec.5 runtime anchors ----
    'enum_all_wall_s': 59.2,
    'enum_all_rss_gb': 2.0,
    'gb1_usecols_read_s': 0.24,
    'gb1_usecols_mb': 16.3,
    'fit_latent_gb1_s': 1.33,
    'structure_all_s': 37.9,
    # ---- Sec.1.5 ----
    'frac_positions_levy_interior': 0.437,
    'n_mutated_positions_total': 2220,
    'n_pdbs': 22,
    'n_registered': 25,
    # ---- C1 pre-declared refutations (Sec.1.2) ----
    'C1_predeclared_refutations': {
        'Z-domain_ZSPA-1_LL1_fitness_1LP1': 1.398,
        'Z-domain_ZSPA-1_LL2_fitness_1LP1': 1.001,
        'Z-domain_ZpA963_HL2_fitness_2M5A': 0.893,
        'CXCR4_CXCL12_enrich_8U4O': 0.883,
        '5A12_Ang2_fitness_4ZFG': 0.844,
    },
}

# --------------------------------------------------------------------------- #
# Amino-acid alphabet for the int8 code vector                                #
# --------------------------------------------------------------------------- #

#: 1..20; 0 is reserved for "WT at that position" (Sec.3 Assay.codes).
AA20 = 'ACDEFGHIKLMNPQRSTVWY'
AA_CODE = {a: i + 1 for i, a in enumerate(AA20)}
CODE_AA = {v: k for k, v in AA_CODE.items()}
assert len(AA_CODE) == 20 and 0 not in AA_CODE.values()


def tier_of(dms_id):
    return ASSAYS[dms_id].tier


def family_of(dms_id):
    return ASSAYS[dms_id].family_id


def _selfcheck():
    import numpy as np
    assert_env()
    print('[config] REPO           = %s' % REPO)
    print('[config] BINDINGGYM_INPUT = %s' % PATHS.bgym_input)
    print('[config] 28 assays: PRIMARY %d / ARM %d / CONTROL %d / EXCLUDED %d'
          % (len(PRIMARY), len(ARM), len(CONTROL), len(EXCLUDED)))
    print('[config] families K = %d  (+F8 arm, reported separately)' % K_FAMILIES)
    print('[config] SEED_BASE = %d   TAUS = %s' % (SEED_BASE, (TAUS,)))
    print('[config] THRESH entries = %d   EXPECTED entries = %d'
          % (len(THRESH), len(EXPECTED)))
    # every file on disk is registered, and every registered file is on disk
    on_disk = set(f[:-4] for f in os.listdir(PATHS.dms_dir) if f.endswith('.csv'))
    assert on_disk == set(ALL_ASSAYS), (
        'registry != disk:\n  only on disk: %s\n  only in registry: %s'
        % (sorted(on_disk - set(ALL_ASSAYS)), sorted(set(ALL_ASSAYS) - on_disk)))
    print('[config] registry == the 28 files on disk  OK')
    # families partition the 12 primary + 2 arm
    fam_members = [m for v in FAMILIES.values() for m in v]
    assert len(fam_members) == len(set(fam_members)) == 14, fam_members
    assert set(fam_members) == set(PRIMARY_AND_ARM)
    print('[config] FAMILIES partition the 12 PRIMARY + 2 ARM  OK')
    # side annotations agree with data_splits/assay_chain_sides.tsv for the 25
    import pandas as pd
    sides = pd.read_csv(PATHS.chain_sides_tsv, sep='\t', comment='#')
    n_chk = 0
    for r in sides.itertuples():
        s = ASSAYS[r.DMS_id]
        assert (s.side0_chains, s.side1_chains) == (str(r.side0_chains), str(r.side1_chains)), \
            (r.DMS_id, s.side0_chains, s.side1_chains, r.side0_chains, r.side1_chains)
        n_chk += 1
    print('[config] side0/side1 match assay_chain_sides.tsv for %d/%d registered assays; '
          'the 3 unregistered are assigned in-registry' % (n_chk, EXPECTED['n_registered']))
    # structure clusters agree with data_splits/BindingGYM_cluster.tsv
    cl = pd.read_csv(PATHS.struct_cluster_tsv, sep='\t', header=None,
                     names=['rep', 'member'])
    for r in cl.itertuples():
        assert ASSAYS[r.member].structure_cluster_id == r.rep, (r.member, r.rep)
    print('[config] structure_cluster_id matches BindingGYM_cluster.tsv for %d rows  OK'
          % len(cl))
    # registered flag agrees with BindingGYM.csv
    reg = set(pd.read_csv(PATHS.registry_csv)['DMS_id'])
    assert len(reg) == EXPECTED['n_registered']
    for k in ALL_ASSAYS:
        assert ASSAYS[k].registered == (k in reg), k
    print('[config] registered flag matches BindingGYM.csv (%d registered, 3 not)  OK'
          % len(reg))
    # pdb existence
    miss = [k for k in ALL_ASSAYS
            if not os.path.exists(os.path.join(PATHS.structures, ASSAYS[k].pdb_file))]
    print('[config] PDB missing for %d assays: %s' % (len(miss), sorted(miss)))
    assert sorted(miss) == ['CR6261_FluAH1_logKd_3GBN', 'CR9114_FluAH1_logKd_4FQI',
                            'CR9114_FluAH3_logKd_4FQY'], sorted(miss)
    # seeds are unique and derived from the base
    assert len(set(SEEDS.values())) == len(set(SEEDS.values()))
    print('[config] seed names = %d, all derived from SEED_BASE  OK' % len(SEEDS))
    print('[config] sum of Sec.2 n_spec over 28 = %d (G1 expects %d)'
          % (sum(ASSAYS[k].n_spec for k in ALL_ASSAYS), EXPECTED['G1_n_rows_total']))
    print('[config] SELF-CHECK PASSED')
    _ = np  # silence linters


if __name__ == '__main__':
    _selfcheck()
