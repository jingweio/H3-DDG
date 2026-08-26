"""BindingGYM dataset for H3-DDG.

H3-DDG's released code covers SKEMPI only; the BindingGYM experiment of the paper (Table 2)
has no public implementation.  This module rebuilds the data path so that every item it emits
is structurally IDENTICAL to what `SkempiDataset.__getitem__` emits, which lets the rest of the
repo (`MPNNPaddingCollate`, `DDGPredictor`, the thermodynamic cycle) be reused unmodified.

Design decisions, each verified against the data (see ibex-records/bindingGYM-reproduce/):

* mutation -> residue mapping uses the `mutant_pdb` column (PDB residue numbering) keyed on
  (chain, resseq, icode).  The per-chain 1-based `mutant` column CANNOT be used: several chains
  have unresolved residues, so sequence index != structure index.  icode is mandatory because
  Kabat-numbered antibody chains produce tokens like `P52AL` (resseq 52, icode 'A').
  NOTE the repo's own `parse_biopython_structure` returns a seq_map keyed on (chain, resseq)
  only -- it drops icode -- so this module builds its own map.

* the two thermodynamic-cycle "sides" come from `data_splits/assay_chain_sides.tsv`, and are fed
  to `parse_biopython_structure` as antibody_chain_id / antigen_chain_id, exactly as the SKEMPI
  path does.  This makes chain_nb in {0,1} = the two binding partners, not individual PDB chains.

* label: ddG_true = -DMS_score.  BindingGYM's DMS_score is uniformly "larger = binds tighter",
  the opposite sign convention to ddG; BindingGYM's own code uses the same mapping
  (`training/dataset.py:73`, `reg_label = -df.loc[idx,'ddg']`).

* wild-type rows (empty `mutant`) keep one flagged position on side0 so the batch keeps a valid
  (complex + isolated-side) structure; since aa_mut == aa everywhere, the model's ddG_pred is
  exactly 0 for them, which is the physically correct answer.
"""
import copy
import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import index_to_one, one_to_index
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from common_utils.protein.parsers import parse_biopython_structure
from common_utils.transforms import get_transform

MUT_RE = re.compile(r'^([A-Z])(-?\d+)([A-Za-z]?)([A-Z])$')


def parse_mut_pdb(token):
    """'P52AL' -> ('P', 52, 'A', 'L');  'A11C' -> ('A', 11, ' ', 'C')."""
    m = MUT_RE.match(token)
    if m is None:
        raise ValueError(f'unparseable mutation token: {token!r}')
    wt, num, icode, mt = m.groups()
    return wt, int(num), (icode if icode else ' '), mt


def load_assay_sides(sides_tsv):
    df = pd.read_csv(sides_tsv, sep='\t', comment='#')
    return {r.DMS_id: (str(r.side0_chains), str(r.side1_chains)) for r in df.itertuples()}


def load_fold_assignment(folds_tsv):
    df = pd.read_csv(folds_tsv, sep='\t', comment='#')
    return dict(zip(df['DMS_id'], df['test_fold']))


class BindingGYMDataset(Dataset):

    def __init__(self, dms_dir, structure_dir, mapping_csv, folds_tsv, sides_tsv, cache_dir,
                 test_fold=0, split='train', label_sign=-1.0, reset=False):
        super().__init__()
        assert split in ('train', 'val', 'all')
        self.dms_dir = dms_dir
        self.structure_dir = structure_dir
        self.mapping_csv = mapping_csv
        self.cache_dir = cache_dir
        self.test_fold = test_fold
        self.split = split
        self.label_sign = label_sign
        os.makedirs(cache_dir, exist_ok=True)

        # identical transform to the SKEMPI path
        self.transform = get_transform([
            {'type': 'select_atom', 'resolution': 'backbone+CB'},
            {'type': 'corrupt_chi_angle', 'ratio_mask': 0.1},
        ])

        self.sides = load_assay_sides(sides_tsv)
        self.fold_of = load_fold_assignment(folds_tsv)
        self.mapping = pd.read_csv(mapping_csv)
        assert set(self.sides) == set(self.mapping['DMS_id'])
        assert set(self.fold_of) == set(self.mapping['DMS_id'])

        self.entries_cache = os.path.join(cache_dir, 'entries.pkl')
        self.structures_cache = os.path.join(cache_dir, 'structures.pkl')
        self._load_structures(reset)
        self._load_entries(reset)

    # ------------------------------------------------------------------ structures

    def _struct_key(self, dms_id):
        poi = self.mapping.set_index('DMS_id').loc[dms_id, 'POI']
        s0, s1 = self.sides[dms_id]
        return (poi, s0, s1)

    def _load_structures(self, reset):
        if os.path.exists(self.structures_cache) and not reset:
            with open(self.structures_cache, 'rb') as f:
                self.structures = pickle.load(f)
            return
        parser = PDBParser(QUIET=True)
        structures = {}
        poi_of = dict(zip(self.mapping['DMS_id'], self.mapping['POI']))
        for dms_id in tqdm(sorted(self.sides), desc='Structures'):
            key = (poi_of[dms_id],) + self.sides[dms_id]
            if key in structures:
                continue
            poi, s0, s1 = key
            model = parser.get_structure(None, os.path.join(self.structure_dir, f'{poi}.pdb'))[0]
            data, _ = parse_biopython_structure(
                model, antibody_chain_id=list(s0), antigen_chain_id=list(s1))
            seq_map = {(ch, int(rs), ic): i for i, (ch, rs, ic)
                       in enumerate(zip(data['chain_id'], data['resseq'], data['icode']))}
            structures[key] = (data, seq_map)
        with open(self.structures_cache, 'wb') as f:
            pickle.dump(structures, f)
        self.structures = structures

    # ------------------------------------------------------------------ entries

    def _load_entries(self, reset):
        if os.path.exists(self.entries_cache) and not reset:
            with open(self.entries_cache, 'rb') as f:
                entries_full = pickle.load(f)
        else:
            entries_full = self._preprocess_entries()
            with open(self.entries_cache, 'wb') as f:
                pickle.dump(entries_full, f)
        self.entries_full = entries_full

        if self.split == 'all':
            keep = set(self.fold_of)
        elif self.split == 'val':
            keep = {d for d, f in self.fold_of.items() if f == self.test_fold}
        else:
            keep = {d for d, f in self.fold_of.items() if f != self.test_fold}
        self.entries = [e for e in entries_full if e['DMS_id'] in keep]

        if self.split != 'all':
            train_ids = {d for d, f in self.fold_of.items() if f != self.test_fold}
            val_ids = {d for d, f in self.fold_of.items() if f == self.test_fold}
            assert not (train_ids & val_ids), 'assay-level leakage between train and val'

    def _preprocess_entries(self):
        poi_of = dict(zip(self.mapping['DMS_id'], self.mapping['POI']))
        entries = []
        n_unresolved_site = 0
        for dms_id in tqdm(list(self.mapping['DMS_id']), desc='Entries'):
            df = pd.read_csv(os.path.join(self.dms_dir, f'{dms_id}.csv'))
            key = (poi_of[dms_id],) + self.sides[dms_id]
            _, seq_map = self.structures[key]
            scores = df['DMS_score'].values.astype(np.float32)
            for row_i, (mut_pdb, score) in enumerate(zip(df['mutant_pdb'].fillna('{}'), scores)):
                d = eval(mut_pdb)
                muts = []
                for ch in d:
                    if not d[ch]:
                        continue
                    for tok in d[ch].split(':'):
                        wt, num, ic, mt = parse_mut_pdb(tok)
                        if (ch, num, ic) not in seq_map:
                            n_unresolved_site += 1
                            continue
                        muts.append(dict(chain=ch, resseq=num, icode=ic, wt=wt, mt=mt))
                entries.append(dict(
                    id=f'{dms_id}#{row_i}',
                    DMS_id=dms_id,
                    struct_key=key,
                    row_index=row_i,
                    mutations=muts,
                    num_muts=len(muts),
                    DMS_score=np.float32(score),
                ))
        if n_unresolved_site:
            print(f'[BindingGYM] {n_unresolved_site} mutated sites were absent from their '
                  f'structure and were skipped (see audit_bindinggym.py)')
        return entries

    # ------------------------------------------------------------------ items

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        data, seq_map = copy.deepcopy(self.structures[entry['struct_key']])

        data['ddG'] = np.float32(self.label_sign * entry['DMS_score'])
        data['DMS_score'] = np.float32(entry['DMS_score'])
        data['id'] = entry['id']
        data['complex'] = entry['DMS_id']
        data['num_muts'] = entry['num_muts']

        aa_mut = data['aa'].clone()
        for mut in entry['mutations']:
            aa_mut[seq_map[(mut['chain'], mut['resseq'], mut['icode'])]] = one_to_index(mut['mt'])
        data['aa_mut'] = aa_mut

        mut_flag = (data['aa'] != data['aa_mut'])
        if not bool(mut_flag.any()):
            # wild-type row (or an all-silent one): flag a single side-0 position so the batch
            # still carries a complex + isolated-side pair.  aa_mut == aa, so ddG_pred == 0.
            side0 = (data['chain_nb'] == 0).nonzero(as_tuple=True)[0]
            mut_flag[side0[0] if len(side0) else 0] = True
        data['mut_flag'] = mut_flag

        # BindingGYM's readout feeds the decoder the MUTANT sequence with every mutated position
        # replaced by 'X' during training (dataset.py: `if not self.evaluation: mseq[pos-1]='X'`),
        # and the unmasked mutant at evaluation time. 'X' is index 20 in both this repo's
        # ressymb_to_resindex and ProteinMPNN's 'ACDEFGHIKLMNPQRSTVWYX' -- verified identical.
        aa_masked = aa_mut.clone()
        aa_masked[mut_flag] = 20
        data['aa_masked'] = aa_masked

        data['mutstr'] = ','.join(
            '{}{}{}{}{}'.format(m['wt'], m['chain'], m['resseq'], m['icode'].strip(), m['mt'])
            for m in entry['mutations'])

        if self.transform is not None:
            data = self.transform(data)
        return data
