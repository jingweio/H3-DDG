"""Loaders for the BindingGYM inter-assay experiment.

One instance = one fold of the official 5-fold inter-assay protocol: the assays whose
`test_fold` equals `test_fold` are held out, the other four groups are trained on.
The fold assignment is READ from `data_splits/inter_assay_folds.tsv`, never recomputed
(GroupKFold tie-breaking is sklearn-version dependent -- see make_inter_assay_folds.py).

`MPNNPaddingCollate` is reused unmodified from the SKEMPI path.
"""
import torch
from torch.utils.data import DataLoader

from bindinggym import BindingGYMDataset
from dataset import MPNNPaddingCollate, inf_iterator

DEFAULTS = dict(
    dms_dir='./data/input/Binding_substitutions_DMS',
    structure_dir='./data/input/structures',
    mapping_csv='./data/input/BindingGYM.csv',
    folds_tsv='./data_splits/inter_assay_folds.tsv',
    sides_tsv='./data_splits/assay_chain_sides.tsv',
    cache_dir='./data/BindingGYM_cache',
)


def complex_row_indices(num_mut_chains):
    """Indices, inside a collated batch, of the whole-complex rows.

    MPNNPaddingCollate emits, per item, one complex row followed by one isolated-side row per
    mutated side.  This reproduces the index arithmetic in DDGPredictor.forward so that eval can
    line predictions up with their metadata at ANY batch size (the released SKEMPI eval loop only
    happens to be correct because it validates with batch_size=1).
    """
    n = torch.tensor(num_mut_chains)
    return (torch.cat([torch.tensor([0]), torch.cumsum(n, dim=0)])[:-1]
            + torch.arange(0, len(num_mut_chains)))


class BindingGYMDatasetManager(object):

    def __init__(self, config, test_fold, num_workers=8, paths=None, reset=False):
        super().__init__()
        p = dict(DEFAULTS)
        if paths:
            p.update(paths)
        self.config = config
        self.test_fold = test_fold

        common = dict(dms_dir=p['dms_dir'], structure_dir=p['structure_dir'],
                      mapping_csv=p['mapping_csv'], folds_tsv=p['folds_tsv'],
                      sides_tsv=p['sides_tsv'], cache_dir=p['cache_dir'],
                      test_fold=test_fold, reset=reset)

        self.train_dataset = BindingGYMDataset(split='train', **common)
        self.val_dataset = BindingGYMDataset(split='val', **common)

        train_assays = {e['DMS_id'] for e in self.train_dataset.entries}
        val_assays = {e['DMS_id'] for e in self.val_dataset.entries}
        leak = train_assays & val_assays
        assert not leak, f'assay-level leakage: {leak}'

        eval_bs = int(getattr(config, 'eval_batch_size', 1))
        self.train_loader = inf_iterator(DataLoader(
            self.train_dataset, batch_size=config.batch_size, shuffle=True,
            collate_fn=MPNNPaddingCollate(), num_workers=num_workers, drop_last=True))
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=eval_bs, shuffle=False,
            collate_fn=MPNNPaddingCollate(), num_workers=num_workers)

        print(f'[fold {test_fold}] train: {len(self.train_dataset)} rows / '
              f'{len(train_assays)} assays | val: {len(self.val_dataset)} rows / '
              f'{len(val_assays)} assays')
        print(f'[fold {test_fold}] held-out assays: {sorted(val_assays)}')

    def get_train_loader(self, fold=None):
        return self.train_loader

    def get_val_loader(self, fold=None):
        return self.val_loader
