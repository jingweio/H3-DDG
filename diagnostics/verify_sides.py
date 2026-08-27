"""Validate data_splits/assay_chain_sides.tsv against the structures themselves.

This table is the one piece of the BindingGYM setup with no external cross-check: BindingGYM
ships no side annotation (its own models are sequence-based and never build a thermodynamic
cycle), so the two binding sides were declared by hand.  If a side is wrong, H3-DDG's
  ddG = E_complex - sum(E of each mutated side in isolation)
is not a binding-energy change at all, and no amount of hyperparameter tuning can recover it.

Four independent checks per assay:
  1. every declared chain exists in the PDB, and the declared chains cover it (no chain silently
     dropped, none invented);
  2. side0 union side1 equals BindingGYM's own `chain_id` column -- the only external reference
     available for which chains belong to the complex;
  3. the declared bipartition is the structurally correct one: the full pairwise heavy-atom
     contact matrix is printed, and the partition is flagged if any WITHIN-side chain pair has
     fewer contacts than the cross-side interface (i.e. the cut went through the wrong seam),
     or if the cross-side interface is empty (the sides do not touch);
  4. every mutated position lands on a declared chain, and we report how the mutations split
     across the two sides -- the quantity that decides how many isolated-side passes the cycle
     needs.
Chain lengths from wildtype_sequence are cross-checked against the modelled residue counts.
"""
import ast
import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Selection import unfold_entities

warnings.filterwarnings('ignore')
REPO = os.path.dirname(os.path.abspath(__file__)) + '/..'
# BindingGYM's raw data lives in the shared store (see bindinggym_dataset.py). Same env var,
# same fallback, so this script keeps working wherever the data actually is.
INPUT = os.environ.get('BINDINGGYM_INPUT', f'{REPO}/data/input')
CONTACT_A = 5.0          # heavy-atom distance defining a residue-residue contact


def load_chains(pdb_path):
    st = PDBParser(QUIET=True).get_structure('s', pdb_path)
    model = next(iter(st))
    out = {}
    for ch in model:
        res = [r for r in ch if any(a.element != 'H' for a in r)
               and r.id[0] == ' ']                       # standard residues only
        if res:
            out[ch.id] = res
    return out


def contact_count(res_a, res_b):
    """#residue pairs with any heavy-atom pair within CONTACT_A."""
    def coords(res):
        pts, idx = [], []
        for i, r in enumerate(res):
            for a in r:
                if a.element != 'H':
                    pts.append(a.coord); idx.append(i)
        return np.asarray(pts, dtype=np.float32), np.asarray(idx)
    pa, ia = coords(res_a)
    pb, ib = coords(res_b)
    if not len(pa) or not len(pb):
        return 0
    pairs = set()
    for start in range(0, len(pa), 4000):                # chunked to bound memory
        blk = pa[start:start + 4000]
        d = np.linalg.norm(blk[:, None, :] - pb[None, :, :], axis=-1)
        hit = np.argwhere(d < CONTACT_A)
        for u, v in hit:
            pairs.add((int(ia[start + u]), int(ib[v])))
    return len(pairs)


def nmuts_by_chain(cell):
    d = ast.literal_eval(cell) if isinstance(cell, str) and cell.strip() else {}
    return {k: len([t for t in str(v).split(':') if t]) for k, v in d.items()}


def main():
    sides = pd.read_csv(f'{REPO}/data_splits/assay_chain_sides.tsv', sep='\t', comment='#')
    mapping = pd.read_csv(f'{INPUT}/BindingGYM.csv').set_index('DMS_id')

    problems = []
    for _, row in sides.iterrows():
        dms = row.DMS_id
        m = mapping.loc[dms]
        pdb = f"{INPUT}/structures/{m.pdb_file}"
        s0, s1 = list(str(row.side0_chains)), list(str(row.side1_chains))
        print('=' * 100)
        print(f'{dms}\n  pdb {m.pdb_file}   declared  side0={"".join(s0)}  side1={"".join(s1)}'
              f'   BindingGYM chain_id={m.chain_id}')

        chains = load_chains(pdb)
        # --- check 1: existence + coverage
        missing = [c for c in s0 + s1 if c not in chains]
        extra = [c for c in chains if c not in s0 + s1]
        if missing:
            problems.append(f'{dms}: declared chains absent from PDB: {missing}')
            print(f'  !! declared chains NOT IN PDB: {missing}  (pdb has {sorted(chains)})')
            continue
        if extra:
            problems.append(f'{dms}: PDB chains in neither side: {extra}')
            print(f'  !! PDB chains assigned to NEITHER side: {extra}')

        # --- check 2: against BindingGYM's own chain_id
        if set(s0 + s1) != set(str(m.chain_id)):
            problems.append(f'{dms}: sides {sorted(set(s0+s1))} != chain_id {sorted(set(str(m.chain_id)))}')
            print(f'  !! side union {sorted(set(s0+s1))} != BindingGYM chain_id '
                  f'{sorted(set(str(m.chain_id)))}')

        # --- length cross-check against wildtype_sequence
        wt = ast.literal_eval(m.wildtype_sequence)
        lens = '  '.join(f'{c}:{len(chains[c])}res/{len(wt.get(c, ""))}seq' for c in s0 + s1)
        print(f'  chain sizes (modelled/wt-seq): {lens}')

        # --- check 3: contact matrix and the cut
        allc = s0 + s1
        cm = {}
        for a, b in itertools.combinations(allc, 2):
            cm[(a, b)] = contact_count(chains[a], chains[b])
        cross = sum(v for (a, b), v in cm.items() if (a in s0) != (b in s0))
        within = {k: v for k, v in cm.items() if (k[0] in s0) == (k[1] in s0)}
        print('  pairwise contacts: ' + '  '.join(f'{a}-{b}:{v}' for (a, b), v in cm.items())
              + f'   || CROSS-SIDE total: {cross}')
        if cross == 0:
            problems.append(f'{dms}: NO contact between the two declared sides')
            print('  !! the two declared sides DO NOT TOUCH -- this is not an interface')
        weak = {k: v for k, v in within.items() if v < cross}
        if weak:
            problems.append(f'{dms}: within-side pair(s) weaker than the interface: {weak}')
            print(f'  !! within-side pair(s) with FEWER contacts than the interface ({cross}): '
                  f'{weak} -- the partition may have cut the wrong seam')

        # --- check 4: where the mutations live
        dfm = pd.read_csv(f"{INPUT}/Binding_substitutions_DMS/{m.DMS_filename}")
        cnt = {}
        for cell in dfm['mutant_pdb'].fillna('{}'):
            for ch, n in nmuts_by_chain(cell).items():
                cnt[ch] = cnt.get(ch, 0) + n
        cnt = {k: v for k, v in cnt.items() if v}
        bad = [c for c in cnt if c not in s0 + s1]
        n0 = sum(v for c, v in cnt.items() if c in s0)
        n1 = sum(v for c, v in cnt.items() if c in s1)
        print(f'  mutations per chain: {cnt}   -> side0 {n0}  side1 {n1}'
              f'  ({"BOTH sides mutated" if n0 and n1 else "one side only"})')
        if bad:
            problems.append(f'{dms}: mutations on chains in neither side: {bad}')
            print(f'  !! mutations on chains outside both sides: {bad}')

    print('=' * 100)
    if problems:
        print(f'{len(problems)} PROBLEM(S):')
        for p in problems:
            print('  - ' + p)
        sys.exit(1)
    print('all 25 assays pass all four checks')


if __name__ == '__main__':
    main()
