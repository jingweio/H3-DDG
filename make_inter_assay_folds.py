"""Freeze the BindingGYM inter-assay (cluster) 5-fold split to a versioned file.

WHY THIS EXISTS
---------------
`GroupKFold` takes no random seed, but it assigns groups to folds by a greedy
"largest group -> currently lightest fold" rule.  BindingGYM's 25 assays fall into 14 clusters
whose sizes tie heavily (6,4,2,2,2,1x9), so 12 of the 14 groups are tied and the assignment is
decided entirely by how the sort orders those ties.

THE DECIDING FACTOR IS NUMPY, NOT SKLEARN.  sklearn's non-shuffle branch is line-for-line
identical from 1.2.1 through 1.7.2 (1.6 only added a separate `if self.shuffle` path).  What
varies is this line inside it:

    indices = np.argsort(n_samples_per_group)[::-1]     # default kind='quicksort' -- NOT stable

Unstable sort means tied entries may come out in any order, and that order changes with the
numpy build.  Measured on this exact weight vector [1,2,1,1,1,1,1,2,1,6,2,1,4,1]:

    numpy 1.22.4 (H3-DDG README pin)  -> [9,12,10,7,1,13,11,8,6,5,4,3,2,0]  -> assignment A
    numpy 1.24.4 (BindingGYM.yml pin) -> same                               -> assignment A
    numpy 2.3.5                       -> [9,12,10,7,1,13, 8,11,5,6,4,3,2,0] -> assignment B
                                          (CD19<->CXCR4 swap folds 2/3,
                                           SARS2-RBD<->HLA-A2 swap folds 3/4)

Both papers' declared environments agree on assignment A, and that is what is frozen in
`data_splits/inter_assay_folds.tsv`.  (`kind="stable"` would make every version agree, but the
official code does not use it, so reproducing their split means reproducing their sort order.)

The fold membership is therefore computed ONCE here under checked numpy AND sklearn versions,
written to the tsv, committed, and read by everything downstream.  Never recompute at train time.

Reproduces BindingGYM `training/main.py:348`:
    split = list(GroupKFold(n_splits=5).split(clusters, groups=clusters))
"""
import argparse
import hashlib
import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold

# Verified to produce the same assignment on this data. numpy is the one that actually decides
# it (see module docstring); sklearn is checked too so the whole pinned env is asserted.
EXPECTED_NUMPY = ("1.22.4", "1.24.4")
EXPECTED_SKLEARN = ("1.2.1", "1.3.2")
# md5 over the sorted "DMS_id\ttest_fold" lines of the committed split. Any environment that
# reproduces the papers' assignment must hit this exactly.
EXPECTED_FINGERPRINT = "d23e15f9f54e6b339e833600c12ff673"
N_SPLITS = 5


def fingerprint(dms_to_fold):
    line = "\n".join(f"{d}\t{dms_to_fold[d]}" for d in sorted(dms_to_fold))
    return hashlib.md5(line.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mapping', default=os.path.join(
        os.environ.get('BINDINGGYM_INPUT', './data/input'), 'BindingGYM.csv'))
    ap.add_argument('--cluster_tsv', default='./data_splits/BindingGYM_cluster.tsv')
    ap.add_argument('--out', default='./data_splits/inter_assay_folds.tsv')
    ap.add_argument('--allow_any_version', action='store_true',
                    help='skip the numpy/sklearn check (the fingerprint check still runs)')
    args = ap.parse_args()

    bad = []
    if np.__version__ not in EXPECTED_NUMPY:
        bad.append(f'numpy {np.__version__} not in {EXPECTED_NUMPY}')
    if sklearn.__version__ not in EXPECTED_SKLEARN:
        bad.append(f'sklearn {sklearn.__version__} not in {EXPECTED_SKLEARN}')
    if bad and not args.allow_any_version:
        raise SystemExit(
            'refusing to run: ' + '; '.join(bad) + '.\n'
            "GroupKFold's tie order comes from np.argsort's unstable sort, so the split changes "
            'with the numpy build -- silently, with no error and a plausible-looking result. '
            'Use the h3ddg-reproduce env.')

    mapping = pd.read_csv(args.mapping)
    cl = pd.read_csv(args.cluster_tsv, sep='\t', header=None, names=['cluster', 'DMS_id'])
    cluster_map = dict(cl.set_index('DMS_id')['cluster'])

    dms_ids = list(mapping['DMS_id'])
    clusters = [cluster_map[d] for d in dms_ids]
    split = list(GroupKFold(n_splits=N_SPLITS).split(clusters, groups=clusters))

    rows = []
    for fold, (_, test_idx) in enumerate(split):
        for j in test_idx:
            rows.append(dict(DMS_id=dms_ids[j], cluster=clusters[j], test_fold=fold))
    out = pd.DataFrame(rows).sort_values(['test_fold', 'DMS_id']).reset_index(drop=True)

    assert len(out) == len(mapping) == 25, f'expected 25 assays, got {len(out)}'
    assert out['DMS_id'].nunique() == 25, 'each assay must appear exactly once'

    # Last line of defence, and the only one that does not depend on getting the root cause right:
    # compare the result itself against the committed split. A version combination we have not
    # seen before still cannot silently produce a different assignment.
    fp = fingerprint(dict(zip(out['DMS_id'], out['test_fold'])))
    if fp != EXPECTED_FINGERPRINT:
        raise SystemExit(
            f'refusing to write: fold assignment fingerprint {fp} != expected '
            f'{EXPECTED_FINGERPRINT}.\nThis environment produces a DIFFERENT split from the one '
            f'both papers\' declared environments produce (numpy {np.__version__}, '
            f'sklearn {sklearn.__version__}). Do not overwrite the committed tsv.')
    print(f'fingerprint {fp} matches the committed split')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(f'# BindingGYM inter-assay (cluster) {N_SPLITS}-fold split\n')
        f.write(f'# generated by make_inter_assay_folds.py with numpy {np.__version__} / scikit-learn {sklearn.__version__}\n')
        f.write(f'# assignment fingerprint (md5 of sorted DMS_id<TAB>test_fold): {EXPECTED_FINGERPRINT}\n')
        f.write(f'# NOTE: the tie order in np.argsort decides this split -- numpy version matters, see module docstring\n')
        f.write(f'# source: GroupKFold(n_splits={N_SPLITS}).split(clusters, groups=clusters), '
                f'per BindingGYM training/main.py:348\n')
        out.to_csv(f, sep='\t', index=False)
    print(f'wrote {args.out}  (numpy {np.__version__} / sklearn {sklearn.__version__})')
    print(out.groupby('test_fold')['DMS_id'].apply(list).to_string())


if __name__ == '__main__':
    main()
