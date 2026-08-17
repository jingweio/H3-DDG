"""Full audit of BindingGYM before building the H3-DDG loader.

Checks, for every one of the 376,446 rows of the 25 shipped assays:
  1. how many rows are wild-type (empty `mutant`)  -> they have no mutated residue at all
  2. whether every mutated site in `mutant_pdb` exists in the parsed structure
     (keyed on (chain, resseq, icode) -- resseq alone is NOT unique, Kabat-numbered
      antibody chains carry insertion codes)
  3. whether the wild-type letter in `mutant_pdb` matches the structure residue
  4. whether the mutation lands on side0 or side1 (the thermodynamic-cycle sides)
"""
import os
import pickle
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import index_to_one, one_to_index

from common_utils.protein.parsers import parse_biopython_structure

DATA = './data'
DMS_DIR = f'{DATA}/input/Binding_substitutions_DMS'
STRUCT_DIR = f'{DATA}/input/structures'
MAPPING = f'{DATA}/input/BindingGYM.csv'
SIDES = './data_splits/assay_chain_sides.tsv'

MUT_RE = re.compile(r'^([A-Z])(-?\d+)([A-Za-z]?)([A-Z])$')


def parse_mut(token):
    """'P52AL' -> ('P', 52, 'A', 'L');  'A11C' -> ('A', 11, ' ', 'C')."""
    m = MUT_RE.match(token)
    if m is None:
        raise ValueError(f'unparseable mutation token: {token!r}')
    wt, num, icode, mt = m.groups()
    return wt, int(num), (icode if icode else ' '), mt


def main():
    mapping = pd.read_csv(MAPPING)
    sides = pd.read_csv(SIDES, sep='\t', comment='#')
    side_map = {r.DMS_id: (str(r.side0_chains), str(r.side1_chains)) for r in sides.itertuples()}
    assert set(side_map) == set(mapping['DMS_id']), 'sides table must cover exactly the 25 assays'

    parser = PDBParser(QUIET=True)
    struct_cache = {}

    def get_struct(poi, dms_id):
        key = (poi, dms_id)
        if key in struct_cache:
            return struct_cache[key]
        s0, s1 = side_map[dms_id]
        model = parser.get_structure(None, f'{STRUCT_DIR}/{poi}.pdb')[0]
        data, _ = parse_biopython_structure(model, antibody_chain_id=list(s0), antigen_chain_id=list(s1))
        smap = {}
        for i, (ch, rs, ic) in enumerate(zip(data['chain_id'], data['resseq'], data['icode'])):
            smap[(ch, int(rs), ic)] = i
        struct_cache[key] = (data, smap)
        return data, smap

    print(f'{"DMS_id":42s} {"rows":>7} {"WT":>4} {"unres":>6} {"wtmis":>6} {"side0":>7} {"side1":>7} {"both":>5} {"chains(nb0|nb1)"}')
    tot = Counter()
    for i in mapping.index:
        dms_id = mapping.loc[i, 'DMS_id']
        poi = mapping.loc[i, 'POI']
        df = pd.read_csv(f'{DMS_DIR}/{dms_id}.csv')
        data, smap = get_struct(poi, dms_id)
        nb_of_chain = {}
        for ch, nb in zip(data['chain_id'], data['chain_nb']):
            nb_of_chain[ch] = int(nb)

        n_wt = n_unres = n_wtmis = n_s0 = n_s1 = n_both = 0
        for mp in df['mutant_pdb'].fillna("{}"):
            d = eval(mp)
            toks = [t for c in d for t in (d[c].split(':') if d[c] else [])]
            if not toks:
                n_wt += 1
                continue
            hit_nb = set()
            row_unres = row_mis = False
            for c in d:
                if not d[c]:
                    continue
                for t in d[c].split(':'):
                    wt, num, ic, mt = parse_mut(t)
                    key = (c, num, ic)
                    if key not in smap:
                        row_unres = True
                        continue
                    idx = smap[key]
                    aa_idx = int(data['aa'][idx])
                    got = index_to_one(aa_idx) if aa_idx < 20 else 'X'
                    if got != wt:
                        row_mis = True
                    hit_nb.add(nb_of_chain[c])
            n_unres += row_unres
            n_wtmis += row_mis
            if hit_nb == {0}:
                n_s0 += 1
            elif hit_nb == {1}:
                n_s1 += 1
            elif len(hit_nb) == 2:
                n_both += 1

        chains0 = ''.join(sorted(c for c, nb in nb_of_chain.items() if nb == 0))
        chains1 = ''.join(sorted(c for c, nb in nb_of_chain.items() if nb == 1))
        print(f'{dms_id:42s} {len(df):7d} {n_wt:4d} {n_unres:6d} {n_wtmis:6d} '
              f'{n_s0:7d} {n_s1:7d} {n_both:5d} {chains0}|{chains1}')
        for k, v in dict(rows=len(df), wt=n_wt, unres=n_unres, wtmis=n_wtmis,
                         s0=n_s0, s1=n_s1, both=n_both).items():
            tot[k] += v

    print('\n=== TOTAL ===')
    for k in ('rows', 'wt', 'unres', 'wtmis', 's0', 's1', 'both'):
        print(f'  {k:6s} {tot[k]:8d}')
    print('\n  unres = rows with >=1 mutated site missing from the structure')
    print('  wtmis = rows where a mutant_pdb wild-type letter disagrees with the structure residue')
    print('  s0/s1/both = which thermodynamic-cycle side(s) the mutations fall on')


if __name__ == '__main__':
    main()
