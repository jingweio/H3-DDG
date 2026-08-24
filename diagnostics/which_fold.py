"""Which fold is the paper's? S4.1: 'the fold with the most multi-point mutations for testing'.

Counts mutations exactly as bindinggym.py does: mutant_pdb is a dict-string {chain: 'A1B:C2D'},
so parse the dict, then split each chain's value on ':'.
"""
import ast
import pandas as pd

repo = '/home/guoj0f/repos/H3-DDG/.claude/worktrees/reproduce'
folds = pd.read_csv(f'{repo}/data_splits/inter_assay_folds.tsv', sep='\t', comment='#')

def nmut(cell):
    d = ast.literal_eval(cell) if isinstance(cell, str) and cell.strip() else {}
    return sum(len([t for t in str(v).split(':') if t]) for v in d.values())

rows = []
for _, r in folds.iterrows():
    src = pd.read_csv(f'{repo}/data/input/Binding_substitutions_DMS/{r.DMS_id}.csv')
    nm = src['mutant_pdb'].fillna('{}').apply(nmut)
    rows.append(dict(fold=r.test_fold, DMS_id=r.DMS_id, n=len(src),
                     n_ge3=int((nm >= 3).sum()), n_ge2=int((nm >= 2).sum()),
                     n_lt3=int(((nm >= 1) & (nm < 3)).sum()), max_mut=int(nm.max())))
t = pd.DataFrame(rows)
agg = t.groupby('fold').agg(n_assays=('DMS_id', 'size'), n_rows=('n', 'sum'),
                            n_ge2=('n_ge2', 'sum'), n_ge3=('n_ge3', 'sum'),
                            n_lt3=('n_lt3', 'sum'), max_mut=('max_mut', 'max'))
agg['pct_ge3'] = (100 * agg.n_ge3 / agg.n_rows).round(1)
print("=== held-out fold, ranked by ABSOLUTE multi-point (>=3) count ===")
print(agg.sort_values('n_ge3', ascending=False).to_string())
print("\n=== ranked by FRACTION >=3 ===")
print(agg.sort_values('pct_ge3', ascending=False)[['n_rows', 'n_ge3', 'pct_ge3']].to_string())
print("\n=== ranked by >=2 (if 'multi-point' means >=2) ===")
print(agg.sort_values('n_ge2', ascending=False)[['n_rows', 'n_ge2']].to_string())
