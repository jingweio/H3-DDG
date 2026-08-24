"""Zero-cost diagnosis of the trained BindingGYM model, from the existing OOF predictions.

Discriminates two hypotheses that produce the same headline number:
  A) pipeline bug (sign / mapping / eval alignment)  -> per-assay r scattered around 0, no structure
  B) pooled-MSE learned the assay SCALE, not the within-assay ranking
     -> ddG_pred's variance is mostly BETWEEN assays; within-assay spread collapses
"""
import glob, os, sys
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr

sp = os.path.dirname(os.path.abspath(__file__))
dfs = []
for f in sorted(glob.glob(os.path.join(sp, 'oof', 'oof_fold*.csv'))):
    d = pd.read_csv(f); d['fold'] = int(f.split('fold')[-1].split('.')[0]); dfs.append(d)
df = pd.concat(dfs, ignore_index=True)
print(f'rows {len(df)}  assays {df.DMS_id.nunique()}  folds {sorted(df.fold.unique())}\n')

rows = []
for (fold, dms), g in df.groupby(['fold', 'DMS_id']):
    r = pearsonr(g.ddG, g.ddG_pred)[0] if g.ddG.std() > 0 and g.ddG_pred.std() > 0 else np.nan
    s = spearmanr(g.ddG, g.ddG_pred)[0] if len(g) > 2 else np.nan
    rows.append(dict(fold=fold, DMS_id=dms[:34], n=len(g),
                     true_mean=g.ddG.mean(), true_std=g.ddG.std(),
                     pred_mean=g.ddG_pred.mean(), pred_std=g.ddG_pred.std(),
                     ratio=g.ddG_pred.std() / g.ddG.std() if g.ddG.std() > 0 else np.nan,
                     pearson=r, spearman=s))
t = pd.DataFrame(rows).sort_values(['fold', 'DMS_id'])
pd.set_option('display.width', 200)
print('=== per-assay (trained model, held-out) ===')
print(t.to_string(index=False, float_format=lambda v: f'{v:8.4f}'))

print(f'\nsign of per-assay Pearson:  positive {(t.pearson>0).sum()} / negative {(t.pearson<0).sum()} / total {t.pearson.notna().sum()}')
print(f'equal-weight mean per-assay Pearson  {t.pearson.mean():.4f}   Spearman {t.spearman.mean():.4f}')

# ---- variance decomposition: is ddG_pred mostly "which assay is this?" ----
def decomp(col, sub):
    gm = sub.groupby('DMS_id')[col]
    grand = sub[col].mean()
    n = gm.size()
    between = float((n * (gm.mean() - grand) ** 2).sum() / len(sub))
    within = float((gm.transform(lambda x: x - x.mean()) ** 2).sum() / len(sub))
    return between, within

print('\n=== variance decomposition (per fold, over its held-out assays) ===')
print(f'{"fold":>5} {"quantity":>9} {"between":>10} {"within":>10} {"between %":>10}')
for fold, sub in df.groupby('fold'):
    for col in ('ddG', 'ddG_pred'):
        b, w = decomp(col, sub)
        print(f'{fold:>5} {col:>9} {b:10.4f} {w:10.4f} {100*b/(b+w):9.1f}%')

# ---- global spread of predictions: did the model collapse? ----
print('\n=== ddG_pred global distribution ===')
print(df.ddG_pred.describe().to_string())
print('\n=== ddG (true) global distribution ===')
print(df.ddG.describe().to_string())

# ---- does the model at least separate mutation depth? ----
print('\n=== by mutation depth (all folds pooled, within-assay r) ===')
for lo, hi, lbl in [(1, 2, '1-2'), (3, 99, '>=3')]:
    sub = df[(df.num_muts >= lo) & (df.num_muts <= hi)]
    if len(sub) < 100: continue
    rs = [pearsonr(g.ddG, g.ddG_pred)[0] for _, g in sub.groupby(['fold','DMS_id'])
          if len(g) >= 100 and g.ddG.std() > 0 and g.ddG_pred.std() > 0]
    print(f'  {lbl:>4}: n_rows {len(sub):7d}  n_assays {len(rs):2d}  mean per-assay r {np.mean(rs):+.4f}')
print('\nmutation-count distribution:')
print(df.num_muts.value_counts().sort_index().head(15).to_string())
