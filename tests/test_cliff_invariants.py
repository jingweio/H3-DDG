"""BGYM-CLIFF v1 -- G1 / G1b / G2 / G3 as pytest assertions, plus the closed-form
checks the spec's numeric hygiene rules demand.

Run:

    source /home/guoj0f/anaconda3/etc/profile.d/conda.sh && conda activate bgym-cliff-v1
    export BINDINGGYM_INPUT=/home/guoj0f/share/BindingGYM/input
    cd <R> && python -m pytest tests/test_cliff_invariants.py -v

Every test runs on the REAL data -- nothing here is a mock.  The whole suite is
~2 min, dominated by the session-scoped 28-file G1 audit (one pass, ~50 s,
including the byte-for-byte ``mutated_sequence`` reconstruction of all 508,962
rows).  Nothing is skipped: G1 is the study's kill gate.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cliff import config, io_bgym, pairs                       # noqa: E402
from cliff.config import EXPECTED, THRESH                      # noqa: E402


# --------------------------------------------------------------------------- #
# session fixtures                                                            #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope='session')
def env():
    return config.assert_env()


@pytest.fixture(scope='session')
def audit(env):
    """The full 28-file G1 audit -- one pass, shared by every G1 test."""
    return io_bgym.audit_all(deep=True, literal_eval_check=True, verbose=False)


@pytest.fixture(scope='session')
def g1b(env):
    return io_bgym.gate_G1b()


@pytest.fixture(scope='session')
def g2(env):
    return io_bgym.gate_G2()


@pytest.fixture(scope='session')
def g3(env):
    return io_bgym.gate_G3()


@pytest.fixture(scope='session')
def small_assays(env):
    return {k: io_bgym.load_assay(k) for k in
            ('Z-domain_ZpA963_HL2_fitness_2M5A', 'BH3_Bcl-xL_normed_1PQ1',
             'PSD95_CRIPT_1BE9', 'Z-domain_ZpA963_HL1_fitness_2M5A')}


# --------------------------------------------------------------------------- #
# config integrity                                                            #
# --------------------------------------------------------------------------- #

def test_registry_matches_disk(env):
    on_disk = set(f[:-4] for f in os.listdir(config.PATHS.dms_dir) if f.endswith('.csv'))
    assert on_disk == set(config.ALL_ASSAYS)
    assert len(config.ALL_ASSAYS) == EXPECTED['G1_n_files'] == 28


def test_tiers_and_families(env):
    assert (len(config.PRIMARY), len(config.ARM), len(config.CONTROL),
            len(config.EXCLUDED)) == (12, 2, 3, 11)
    members = [m for v in config.FAMILIES.values() for m in v]
    assert len(members) == len(set(members)) == 14
    assert set(members) == set(config.PRIMARY_AND_ARM)
    assert config.K_FAMILIES == 7


def test_taus_and_seeds_are_frozen(env):
    assert config.TAUS == (2, 3, 4, 5, 6, 8)
    assert config.SEED_BASE == 20260902
    assert all(isinstance(v, int) for v in config.SEEDS.values())
    # per-assay entropy is unique per (name, assay)
    e = [config.assay_seed('randpairs', k) for k in config.ALL_ASSAYS]
    assert len({tuple(x) for x in e}) == 28


def test_thresholds_live_only_in_config(env):
    """No decision boundary may be re-declared in another cliff module."""
    import re
    banned = {'0.50', '0.35', '0.80', '0.70', '2.8606', '2.2393', '1.4826'}
    for mod in ('io_bgym.py', 'pairs.py'):
        src = open(os.path.join(config.REPO, 'cliff', mod)).read()
        code = '\n'.join(l for l in src.splitlines()
                         if not l.strip().startswith('#'))
        # strip docstrings crudely: they are the only triple-quoted blocks here
        code = re.sub(r'"""(?:.|\n)*?"""', '', code)
        for b in banned:
            assert b not in code, '%s re-declares the literal %s' % (mod, b)


# --------------------------------------------------------------------------- #
# G1                                                                          #
# --------------------------------------------------------------------------- #

def test_G1_rows_and_unique_keys(audit):
    tot = audit.attrs['G1']
    assert tot['n_rows'] == EXPECTED['G1_n_rows_total']
    assert tot['n_unique_keys'] == EXPECTED['G1_n_unique_keys_total']
    assert tot['n_dup_keys'] == 0, 'a duplicate canonical genotype exists'


def test_G1_mutation_instances(audit):
    """The spec's G1 line mixes denominators: 508,962 rows is all 28 files, but
    1,173,273 mutation instances is the 25 REGISTERED files."""
    a = audit.set_index('DMS_id')
    reg = [k for k in config.ALL_ASSAYS if config.ASSAYS[k].registered]
    assert int(a.loc[reg, 'n_mutation_instances'].sum()) == \
        EXPECTED['G1_n_mutation_instances_registered25']
    assert audit.attrs['G1']['n_mutation_instances'] == \
        EXPECTED['G1_n_mutation_instances_all28']
    # independent confirmation of the same denominator
    assert int(a.loc[reg, 'n_distinct_pdb_keys'].sum()) == \
        EXPECTED['G1_n_mutated_positions_registered25']


def test_G1_wt_letters_agree_across_all_four_sources(audit):
    assert int(audit['n_wt_mismatch_mutant_vs_pdbcol'].sum()) == 0
    assert int(audit['n_wt_mismatch_vs_wildtype_sequence'].sum()) == 0
    assert int(sum(v for v in audit['n_wt_mismatch_vs_pdb_residue'] if v != '')) == 0
    assert int(sum(v for v in audit['n_pdb_key_missing'] if v != '')) == 0


def test_G1_no_X_no_star_no_identity_no_indel(audit):
    tot = audit.attrs['G1']
    assert tot['n_X_hits'] == EXPECTED['G1_n_X_hits'] == 0
    assert tot['n_star_tokens'] == EXPECTED['G1_n_star_tokens'] == 0
    assert tot['n_identity_mutations'] == EXPECTED['G1_n_identity_mutations'] == 0
    assert tot['n_indels'] == EXPECTED['G1_n_indels'] == 0
    assert tot['n_parse_fail'] == 0
    assert tot['n_token_count_mismatch'] == 0
    assert tot['all_chain_token_counts_agree'] is True


def test_G1_mutated_sequence_reconstructs_exactly(audit):
    """Spec Sec.1.0's load-bearing claim: Hamming == mutation-set distance."""
    assert audit.attrs['G1']['n_reconstruction_fail'] == 0
    assert audit.attrs['G1']['wildtype_sequence_constant_per_file'] is True


def test_G1_fast_dict_parser_agrees_with_ast_literal_eval(audit):
    tot = audit.attrs['G1']
    assert tot['n_literal_eval_diff'] == 0
    assert tot['n_literal_eval_checked'] > 1_000_000


def test_G1_hla_duplicate_column_is_identical(audit):
    assert audit.attrs['HLA_duplicate_column']['n_diff_between_duplicates'] == 0


# --------------------------------------------------------------------------- #
# G1b                                                                         #
# --------------------------------------------------------------------------- #

def test_G1b_join_is_518_of_518_both_ways(g1b):
    for tag in ('mutant_seq', 'mutant_pdb'):
        assert g1b[tag]['n_shared'] == EXPECTED['G1b_n_shared'] == 518
        assert g1b[tag]['n_a'] == g1b[tag]['n_b'] == EXPECTED['G1b_n_total']
        # the offset is uniquely identified by WT-residue identity
        assert len(g1b[tag]['offsets_wt_consistent']) == 1


def test_G1b_seq_offset_is_minus_two(g1b):
    assert g1b['mutant_seq']['offset'] == EXPECTED['G1b_offset_seq'] == -2
    assert g1b['mutant_pdb']['offset'] == -84


def test_G1b_naive_join_loses_most_of_the_data(g1b):
    """Why naive (pos, aa) joins are banned repo-wide."""
    assert g1b['mutant_seq']['naive_n_shared'] == EXPECTED['G1b_naive_join_n_shared']
    assert g1b['mutant_pdb']['naive_n_shared'] < 5


def test_G1b_r_disagreement_is_pearson_vs_spearman(g1b):
    """The two profiling agents did NOT disagree about the join: 0.1709 is the
    Pearson r and +0.592 the Spearman rho of the SAME 518/518 join."""
    v = g1b['mutant_seq']
    assert abs(v['pearson'] - EXPECTED['G1b_r_claim_a']) < 5e-4
    assert abs(v['spearman'] - EXPECTED['G1b_r_claim_b']) < 5e-3
    # and both routes give the same numbers, so the route is not the issue
    assert abs(v['pearson'] - g1b['mutant_pdb']['pearson']) < 1e-12
    assert abs(v['spearman'] - g1b['mutant_pdb']['spearman']) < 1e-12


# --------------------------------------------------------------------------- #
# G2                                                                          #
# --------------------------------------------------------------------------- #

def test_G2_shared_keys(g2):
    assert g2['n_shared'] == EXPECTED['G2_n_shared_keys'] == 19227


def test_G2_byte_identity_on_the_raw_strings(g2):
    assert g2['n_raw_string_differences'] == 0
    assert g2['max_abs_delta'] == EXPECTED['G2_max_abs_delta'] == 0.0


def test_G2_chain_labels_differ_so_a_canonical_key_join_is_impossible(env):
    a = config.ASSAYS['KRAS_SOS1_norfitness_8BE4']
    b = config.ASSAYS['KRAS_DARPinK27_norfitness_5O2S']
    assert a.side0_chains != b.side0_chains == 'A'


# --------------------------------------------------------------------------- #
# G3                                                                          #
# --------------------------------------------------------------------------- #

def test_G3_zero_duplicates_with_the_chain_label(g3):
    for r in g3.itertuples():
        assert r.n_dup_keys_with_chain == 0, r.DMS_id
        assert r.n_dup_rows_with_chain == 0, r.DMS_id
        assert r.n_dup_keys_with_chain == r.expected_dups_with_chain


def test_G3_duplicates_reappear_without_the_chain_label(g3):
    """847 / 59 / 650 / 38 -- reproduced exactly when the chain is dropped from
    the full mutation token ``(seq_pos, wt_aa, mut_aa)``."""
    for r in g3.itertuples():
        assert r.n_dup_keys_without_chain == r.expected_dups_without_chain, r.DMS_id


def test_G3_coarser_chain_drop_gives_strictly_more_collisions(g3):
    for r in g3.itertuples():
        assert r.n_dup_keys_without_chain_posaa >= r.n_dup_keys_without_chain


def test_G3_the_forbidden_noise_floor_is_an_artefact(g3):
    """The Z-domain within-genotype SDs exist only under a chain-dropped key, so
    they are FORBIDDEN as a noise floor (spec Sec.1.0)."""
    for r in g3.itertuples():
        assert r.n_groups_ge2_without_chain > 0
        assert np.isfinite(r.mean_within_genotype_sd_without_chain)
        # ... and there is nothing to compute with the chain retained
        assert r.n_dup_keys_with_chain == 0
    assert set(config.FORBIDDEN_ZDOMAIN_SDS) == set(io_bgym.Z_ASSAYS)


def test_G3_chain_label_is_in_every_canonical_key(small_assays):
    a = small_assays['Z-domain_ZpA963_HL1_fitness_2M5A']
    for k in a.keys:
        for t in k:
            assert isinstance(t[0], str) and len(t) == 3
    assert set(t[0] for k in a.keys for t in k) == {'A', 'B'}


# --------------------------------------------------------------------------- #
# closed-form / brute-force checks                                            #
# --------------------------------------------------------------------------- #

def _gmd_closed_form(y):
    """Spec Sec.3: ``y=sort(y); i=1..n; 2*((2i-n-1)*y).sum()/(n(n-1))``."""
    y = np.sort(np.asarray(y, dtype=np.float64))
    n = y.size
    i = np.arange(1, n + 1)
    return 2.0 * ((2 * i - n - 1) * y).sum() / (n * (n - 1))


def _gmd_brute(y):
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    return float(sum(abs(y[i] - y[j]) for i in range(n) for j in range(i + 1, n))
                 / (n * (n - 1) / 2.0))


@pytest.mark.parametrize('n', [2, 3, 7, 40, 137])
def test_gini_mean_difference_closed_form_matches_brute_force(n):
    rng = np.random.default_rng([config.SEED_BASE, n])
    for _ in range(5):
        y = rng.normal(size=n) * rng.uniform(0.1, 10)
        assert math.isclose(_gmd_closed_form(y), _gmd_brute(y), rel_tol=1e-11)
    # ties and constants must not break it
    y = np.array([1.0, 1.0, 1.0, 2.0])
    assert math.isclose(_gmd_closed_form(y), _gmd_brute(y), rel_tol=1e-12)


def test_v_infinity_closed_form(env):
    """``V(inf) = Var(y) * n/(n-1)`` must equal the mean squared difference / 2
    over all pairs."""
    rng = np.random.default_rng(config.SEED_BASE)
    y = rng.normal(size=200)
    n = y.size
    closed = float(np.var(y) * n / (n - 1))
    brute = float(np.mean([(y[i] - y[j]) ** 2
                           for i in range(n) for j in range(i + 1, n)]) / 2.0)
    assert math.isclose(closed, brute, rel_tol=1e-11)


def test_bincount_reduction_matches_add_at(env):
    """``np.bincount``, not ``np.add.at`` (spec Sec.1.2) -- same answer, and the
    reason is speed, so the equivalence must be asserted."""
    rng = np.random.default_rng(config.SEED_BASE)
    h = rng.integers(0, 7, size=200000)
    d2 = rng.normal(size=200000) ** 2
    a = np.bincount(h, weights=d2, minlength=8)
    b = np.zeros(8)
    np.add.at(b, h, d2)
    assert np.allclose(a, b, rtol=0, atol=1e-9)
    assert np.array_equal(np.bincount(h, minlength=8),
                          np.array([(h == k).sum() for k in range(8)]))


def test_mad_scaling_constant(env):
    """1.4826 x MAD, never SD (spec Sec.1.0)."""
    rng = np.random.default_rng(config.SEED_BASE)
    x = rng.normal(size=2_000_000)
    mad = THRESH['mad_const'] * np.median(np.abs(x - np.median(x)))
    assert abs(mad - 1.0) < 0.005
    # and it must be robust where SD is not
    y = np.concatenate([x[:1000], np.array([1e6])])
    assert abs(THRESH['mad_const'] * np.median(np.abs(y - np.median(y))) - 1.0) < 0.15
    assert y.std(ddof=1) > 1000


def test_pair_enumeration_matches_brute_force(small_assays):
    for dms_id, a in small_assays.items():
        if a.n > 3000:
            continue
        ni, ac = pairs.enumerate_nested(a.keys, a.col_index)
        si, pc = pairs.enumerate_samesite(a.keys, a.pos_index)
        bn, bs = pairs._brute_pairs(a.keys)
        assert ni.shape[0] == bn, dms_id
        assert si.shape[0] == bs, dms_id
        assert ni.dtype == si.dtype == np.int32
        assert ac.dtype == pc.dtype == np.int32
        # column 0 is the smaller set; each unordered pair appears once
        assert all(len(a.keys[u]) + 1 == len(a.keys[v]) for u, v in ni)
        assert len({(int(u), int(v)) for u, v in ni}) == ni.shape[0]
        assert (si[:, 0] < si[:, 1]).all()
        assert len({(int(u), int(v)) for u, v in si}) == si.shape[0]
        # the two classes are disjoint
        assert not ({(int(u), int(v)) for u, v in ni}
                    & {(int(u), int(v)) for u, v in si})


def test_samesite_bucket_sizes_give_exactly_C_k_2(small_assays):
    a = small_assays['Z-domain_ZpA963_HL1_fitness_2M5A']
    from collections import Counter
    buckets = Counter()
    for k in a.keys:
        for j in range(len(k)):
            buckets[(k[:j] + k[j + 1:], k[j][0], k[j][1])] += 1
    expect = sum(v * (v - 1) // 2 for v in buckets.values())
    si, _ = pairs.enumerate_samesite(a.keys, a.pos_index)
    assert si.shape[0] == expect
    # every aa in a bucket is distinct -> no same-aa correction needed
    seen = {}
    for k in a.keys:
        for j in range(len(k)):
            b = (k[:j] + k[j + 1:], k[j][0], k[j][1])
            seen.setdefault(b, set())
            assert k[j][2] not in seen[b]
            seen[b].add(k[j][2])


def test_hamming_from_codes_matches_naive_and_scipy(small_assays):
    from scipy.spatial.distance import hamming as sp_ham
    a = small_assays['Z-domain_ZpA963_HL1_fitness_2M5A']
    rng = np.random.default_rng(config.assay_seed('randpairs', a.dms_id))
    idx = np.stack([rng.integers(0, a.n, 5000), rng.integers(0, a.n, 5000)],
                   1).astype(np.int32)
    fast = pairs.hamming_from_codes(a.codes, idx, block=613)
    naive = np.array([(a.codes[i] != a.codes[j]).sum() for i, j in idx])
    scipy_ref = np.array([round(sp_ham(a.codes[i], a.codes[j]) * a.P) for i, j in idx])
    assert np.array_equal(fast, naive)
    assert np.array_equal(fast, scipy_ref)
    assert fast.dtype == np.int32
    # blocking must not change the answer
    assert np.array_equal(fast, pairs.hamming_from_codes(a.codes, idx, block=10 ** 9))


def test_code_hamming_is_one_for_both_pair_classes(small_assays):
    """The metric caveat, asserted so nobody silently merges the two classes:
    the code-vector Hamming distance is 1 for a nested pair AND for a same-site
    swap, while the spec's symmetric-difference metric gives 1 and 2."""
    a = small_assays['Z-domain_ZpA963_HL1_fitness_2M5A']
    ni, _ = pairs.enumerate_nested(a.keys, a.col_index)
    si, _ = pairs.enumerate_samesite(a.keys, a.pos_index)
    assert set(pairs.hamming_from_codes(a.codes, ni).tolist()) == {1}
    assert set(pairs.hamming_from_codes(a.codes, si).tolist()) == {1}
    for u, v in ni[:2000]:
        assert len(set(a.keys[u]) ^ set(a.keys[v])) == 1
    for u, v in si[:2000]:
        assert len(set(a.keys[u]) ^ set(a.keys[v])) == 2


def test_sibling_counts_on_a_complete_hypercube(env):
    """Synthetic 2^6 hypercube: every nested edge must have exactly 5 siblings
    (the CR9114-H1 check of spec Sec.1.4 L1, at a size a test can afford)."""
    import itertools
    d = 6
    keys = []
    for r in range(d + 1):
        for c in itertools.combinations(range(d), r):
            keys.append(tuple(sorted(('H', p + 1, 'A') for p in c)))
    col_index, pos_index = io_bgym.build_col_index(keys)
    ni, ac = pairs.enumerate_nested(keys, col_index)
    assert ni.shape[0] == d * 2 ** (d - 1)
    sib = pairs.sibling_counts(ni, ac, keys)
    assert set(sib.tolist()) == {d - 1}


def test_sample_random_pairs_is_seeded_and_uniform(env):
    n, m = 500, 400000
    s = pairs.sample_random_pairs(n, m, config.assay_seed('randpairs', 'PSD95_CRIPT_1BE9'))
    assert s.shape == (m, 2) and s.dtype == np.int32
    assert (s[:, 0] < s[:, 1]).all()
    # determinism
    assert np.array_equal(
        s, pairs.sample_random_pairs(n, m, config.assay_seed('randpairs',
                                                             'PSD95_CRIPT_1BE9')))
    # two assays with the same n must NOT share a stream
    assert not np.array_equal(
        s, pairs.sample_random_pairs(n, m, config.assay_seed('randpairs',
                                                             'PSD95_Tm2F_1BE9')))
    # uniform over the C(n,2) pairs: chi-square-ish check on the row marginal
    cnt = np.bincount(s.reshape(-1), minlength=n)
    assert abs(cnt.mean() - 2 * m / n) < 1e-6
    assert cnt.std() / cnt.mean() < 0.05


def test_all_pairs_exact_matches_triu(env):
    p = pairs.all_pairs_exact(60)
    assert p.shape == (60 * 59 // 2, 2) and p.dtype == np.int32
    assert (p[:, 0] < p[:, 1]).all()
    assert len({(int(a), int(b)) for a, b in p}) == p.shape[0]


def test_detect_censoring_reproduces_every_spec_quoted_level(env):
    """SARS2-RBD -4.84 + -4.76 = 23.8%; CR9114-H3 6.000 = 89.05%; CR9114-H1
    7.000 = 2.57%; CR6261 7.000 = 11.34%; and NOTHING on the 23 others."""
    want = {
        'SARS2-RBD_ACE2_deltaKd_6M0J': ([-4.84, -4.76], 0.2384),
        'CR9114_FluAH3_logKd_4FQY': ([6.0], 0.8905),
        'CR9114_FluAH1_logKd_4FQI': ([7.0], 0.0257),
        'CR6261_FluAH1_logKd_3GBN': ([7.0], 0.1134),
        'CXCR4_CXCL12_enrich_8U4O': ([-5.0], 0.0141),
    }
    import pandas as pd
    n_censored = 0
    for dms_id in config.ALL_ASSAYS:
        s = pd.read_csv(config.PATHS.dms_csv(dms_id), usecols=['DMS_score'],
                        dtype=str)['DMS_score'].values
        y = s.astype(np.float64)
        lv, mk, meta = io_bgym.detect_censoring(y)
        if dms_id in want:
            exp_lv, exp_mass = want[dms_id]
            assert sorted(lv) == sorted(exp_lv), (dms_id, lv)
            assert abs(mk.mean() - exp_mass) < 2e-3, (dms_id, mk.mean())
            n_censored += 1
        else:
            assert lv == (), (dms_id, lv)
    assert n_censored == 5, 'the spec says five assays carry censoring'


def test_quantum_reproduces_the_two_spec_quoted_grids(env):
    import pandas as pd
    want = {'SARS2-RBD_ACE2_deltaKd_6M0J': 0.01, 'CR9114_FluAH3_logKd_4FQY': 0.1}
    for dms_id, q in want.items():
        s = pd.read_csv(config.PATHS.dms_csv(dms_id), usecols=['DMS_score'],
                        dtype=str)['DMS_score'].tolist()
        got, md = io_bgym.score_quantum(s)
        assert got == q, (dms_id, got, q)
        # and the grid guard does bind there
        assert THRESH['grid_guard_mult'] * got > 0.02


def test_hyap65_is_the_only_transformed_assay(env):
    tr = {k: config.ASSAYS[k].transform for k in config.ALL_ASSAYS}
    assert [k for k, v in tr.items() if v != 'none'] == \
        ['hYAP65_peptide_FunctioncalScore_1JMQ']
    a = io_bgym.load_assay('hYAP65_peptide_FunctioncalScore_1JMQ')
    assert (a.y_raw > 0).all()
    assert np.allclose(a.y, np.log10(a.y_raw))
    assert abs(a.y[a.wt_row]) < 1e-15 and abs(a.y_raw[a.wt_row] - 1.0) < 1e-15


def test_sign_convention_is_higher_is_better_for_all_28(env):
    assert {config.ASSAYS[k].sign_convention for k in config.ALL_ASSAYS} == \
        {'higher_is_better'}
    # the resolution's evidence: CR9114-H1 germline 8.425 -> matured 9.592
    a = io_bgym.load_assay('CR9114_FluAH1_logKd_4FQI')
    assert abs(a.y[a.wt_row] - 8.4246) < 1e-3
    assert a.y.max() > a.y[a.wt_row]
    b = io_bgym.load_assay('CR6261_FluAH1_logKd_3GBN')
    assert abs(b.y[b.wt_row] - 7.000) < 1e-9      # WT sits ON the floor
    assert b.y.max() > 9.5


def test_primary_Pa_excludes_the_wt_hub_and_censored_endpoints(env):
    a = io_bgym.load_assay('SARS2-RBD_ACE2_deltaKd_6M0J')
    ni, ac = pairs.enumerate_nested(a.keys, a.col_index)
    pa, wt_anch, cens = pairs.primary_nested_set(a, ni)
    assert not (pa & wt_anch).any()
    assert not (pa & cens).any()
    assert wt_anch.sum() > 0 and cens.sum() > 0
    assert pa.sum() == int((~wt_anch & ~cens).sum())


def test_manifest_md5s_verify_if_the_cache_exists(env):
    if not os.path.exists(config.PATHS.manifest):
        pytest.skip('stage 0 has not been run in this tree')
    bad = pairs.verify_manifest()
    assert bad == [], 'cache md5 mismatch: %r' % (bad[:5],)


def test_gb1_1fcc_is_a_complete_single_scan(env):
    """The flagship assay's defining property (spec Sec.2 #1): 1,045 = 55 x 19
    singles, so every one of the 91,845 doubles has an exact additive baseline."""
    a = io_bgym.load_assay('GB1_IgG-Fc_fitness_1FCC')
    assert a.n == 92891 and a.P == 55 and a.M == 55 * 19 == 1045
    hist = np.bincount(a.n_muts.astype(np.int64))
    assert hist[0] == 1 and hist[1] == 1045 and hist[2] == 91845
    ni, ac = pairs.enumerate_nested(a.keys, a.col_index)
    assert ni.shape[0] == EXPECTED['n_nested_GB1_1FCC'] == 184735
    si, _ = pairs.enumerate_samesite(a.keys, a.pos_index)
    assert si.shape[0] == EXPECTED['n_samesite_GB1_1FCC'] == 861874
    # every double reaches BOTH its singles -> exact epsilon for all of them,
    # so the nested count is exactly 1,045 WT-anchored + 2 x 91,845
    assert ni.shape[0] == 1045 + 2 * 91845


def test_cr9114_h1_every_edge_has_fifteen_siblings(env):
    """Spec Sec.1.4 L1's own check on the strongest L1 assay in the benchmark."""
    a = io_bgym.load_assay('CR9114_FluAH1_logKd_4FQI')
    assert a.P == 16
    ni, ac = pairs.enumerate_nested(a.keys, a.col_index)
    sib = pairs.sibling_counts(ni, ac, a.keys)
    assert int(np.median(sib)) == 15 and int(sib.max()) == 15
    assert (sib >= THRESH['L1_min_siblings']).sum() > THRESH['L1_min_edges']
    # a binary hypercube has no same-site swaps at all
    si, _ = pairs.enumerate_samesite(a.keys, a.pos_index)
    assert si.shape[0] == 0


def test_parse_pair_dicts_joins_by_chain_key_not_dict_order(env):
    got = io_bgym.parse_pair_dicts("{'H': 'P53L:Y57C', 'L': '', 'A': ''}",
                                  "{'A': '', 'H': 'P52AL:Y56C', 'L': ''}")
    assert got == [('H', 53, 'P', 'L', 52, 'A'), ('H', 57, 'Y', 'C', 56, '')]
    with pytest.raises(ValueError):
        io_bgym.parse_pair_dicts("{'H': 'P53L'}", "{'L': 'P52AL'}")
    with pytest.raises(ValueError):
        io_bgym.parse_pair_dicts("{'H': 'P53L:Y57C'}", "{'H': 'P52AL'}")
    with pytest.raises(ValueError):
        io_bgym.parse_pair_dicts("{'H': 'P53L'}", "{'H': 'P52AF'}")


def test_kabat_icode_is_preserved(env):
    assert io_bgym.parse_mut_token('P52AL') == ('P', 52, 'A', 'L')
    assert io_bgym.parse_mut_token('A11C') == ('A', 11, '', 'C')
    assert io_bgym.parse_mut_token('Y100BS') == ('Y', 100, 'B', 'S')
    assert io_bgym.parse_mut_token('A-1C') == ('A', -1, '', 'C')
    with pytest.raises(ValueError):
        io_bgym.parse_mut_token('A11*')


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v', '--tb=short']))
