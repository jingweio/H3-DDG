"""Per-DMS metrics for BindingGYM, in BOTH reporting conventions.

Two metric families are computed side by side so that our results have a reference point on
each side and a disagreement can be attributed to convention rather than to implementation:

  (1) H3-DDG convention  -- what the paper's Table 2 reports: Pearson / Spearman / AUROC / RMSE.
      * AUROC and Pearson/Spearman reuse the repo's own definitions (`utils.overall_auroc`
        classifies by sign(ddG_true) and ranks by ddG_pred).
      * RMSE is reported TWICE because the paper's number cannot be pinned to one definition:
          rmse_raw   : sqrt(mean((pred - true)^2))                       -- scale-sensitive
          rmse_calib : the repo's own `utils.overall_rmse_mae`, which fits a LinearRegression
                       from pred to true on the evaluation data and takes the residual RMSE
                       (equals std(true)*sqrt(1-r^2); affine-invariant).
        Table 2's RMSE spread across methods (ProteinMPNN 3.4974 vs H3-DDG 1.1294, a 3x range)
        is far wider than rmse_calib could produce for r in [0.10, 0.31] (only a 4% spread),
        so Table 2 is most likely rmse_raw -- but we report both rather than assume.

  (2) BindingGYM convention -- what the benchmark itself reports (paper Table 5): Spearman /
      AUC / MCC / NDCG / AP, copied verbatim from `calc_metric.ipynb::calc_zero_shot_metric`.
      Every one of these depends only on the ORDER of the predictions within one assay
      (MCC's threshold is the predictions' own 90th percentile), so they are invariant to the
      per-assay scale mismatch that a single regression head cannot avoid.

Slicing follows BindingGYM exactly: ALL / <3 / >=3 by total mutation count across chains, with
an assay dropped from a slice when it has fewer than 100 rows in it (this is what makes the
official tables cover 25 / 22 / 13 assays).  Every metric is averaged over assays with EQUAL
WEIGHT, never pooled across assays.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (average_precision_score, matthews_corrcoef, ndcg_score,
                             roc_auc_score)

MIN_ROWS_PER_SLICE = 100


# --------------------------------------------------------------------------- helpers

def _safe(fn, *a, **kw):
    try:
        v = fn(*a, **kw)
        return float(v) if v is not None and np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _rmse_calibrated(true, pred):
    """The repo's own RMSE (utils.overall_rmse_mae): residual RMSE after an affine fit."""
    reg = LinearRegression().fit(pred[:, None], true)
    corrected = reg.predict(pred[:, None])
    return float(np.sqrt(((true - corrected) ** 2).mean())), \
        float(np.abs(true - corrected).mean())


# --------------------------------------------------------------------------- per assay

def h3ddg_metrics_one_assay(df):
    """df needs columns ddG (true) and ddG_pred. Mirrors utils.py's definitions."""
    true = df['ddG'].to_numpy(dtype=float)
    pred = df['ddG_pred'].to_numpy(dtype=float)
    rmse_calib, mae_calib = (_rmse_calibrated(true, pred) if len(df) > 1 else (np.nan, np.nan))
    label = (true > 0)
    return {
        'n': len(df),
        'pearson': _safe(lambda: df[['ddG', 'ddG_pred']].corr('pearson').iloc[0, 1]),
        'spearman': _safe(lambda: df[['ddG', 'ddG_pred']].corr('spearman').iloc[0, 1]),
        'auroc': _safe(roc_auc_score, label, pred) if label.any() and not label.all() else np.nan,
        'auprc': _safe(average_precision_score, label, pred) if label.any() and not label.all() else np.nan,
        'rmse_raw': float(np.sqrt(((pred - true) ** 2).mean())),
        'mae_raw': float(np.abs(pred - true).mean()),
        'rmse_calib': rmse_calib,
        'mae_calib': mae_calib,
    }


def bindinggym_metrics_one_assay(df, pred_col='pred', label_col='DMS_score'):
    """Verbatim port of BindingGYM calc_metric.ipynb::calc_zero_shot_metric (top_test=False).

    `pred_col` must be oriented so that LARGER = binds tighter, matching DMS_score.
    """
    label = df[label_col].to_numpy(dtype=float)
    pred = df[pred_col].to_numpy(dtype=float)
    label_bin = (label > np.percentile(label, 90)) + 0
    pred_bin = (pred > np.percentile(pred, 90)) + 0
    single_class = label_bin.sum() == 0 or label_bin.sum() == len(label_bin)
    return {
        'n': len(df),
        'Spearman': _safe(lambda: df[label_col].rank().corr(df[pred_col].rank())),
        'AUC': np.nan if single_class else _safe(roc_auc_score, label_bin, pred),
        'MCC': _safe(matthews_corrcoef, label_bin, pred_bin),
        'NDCG': _safe(ndcg_score,
                      df[label_col].rank().values.reshape(1, -1),
                      df[pred_col].values.reshape(1, -1),
                      k=max(1, df.shape[0] // 10)),
        'AP': np.nan if single_class else _safe(average_precision_score, label_bin, pred),
    }


# --------------------------------------------------------------------------- slicing

def add_depth_slice(df, num_muts_col='num_muts'):
    df = df.copy()
    df['depth_slice'] = np.where(df[num_muts_col] >= 3, '>=3', '<3')
    return df


def per_dms_table(df, family='h3ddg'):
    """Return (per-assay table, slice summary) for one of the two metric families.

    df columns required: DMS_id, num_muts, ddG, ddG_pred  (+ DMS_score, pred for bindinggym).
    """
    assert family in ('h3ddg', 'bindinggym')
    fn = h3ddg_metrics_one_assay if family == 'h3ddg' else bindinggym_metrics_one_assay
    rows = []
    for slice_name in ('ALL', '<3', '>=3'):
        if slice_name == 'ALL':
            sub_all = df
        elif slice_name == '<3':
            sub_all = df[df['num_muts'] < 3]
        else:
            sub_all = df[df['num_muts'] >= 3]
        for dms_id, sub in sub_all.groupby('DMS_id', sort=True):
            if slice_name != 'ALL' and len(sub) < MIN_ROWS_PER_SLICE:
                continue
            rec = {'slice': slice_name, 'DMS_id': dms_id}
            rec.update(fn(sub.reset_index(drop=True)))
            rows.append(rec)
    per_assay = pd.DataFrame(rows)
    if per_assay.empty:
        return per_assay, pd.DataFrame()
    metric_cols = [c for c in per_assay.columns if c not in ('slice', 'DMS_id', 'n')]
    counts = per_assay.groupby('slice').agg(n_assays=('DMS_id', 'nunique'), n_rows=('n', 'sum'))
    summary = per_assay.groupby('slice')[metric_cols].mean().join(counts)
    # keep the canonical slice order, but only slices that actually have qualifying assays
    order = [s for s in ('ALL', '<3', '>=3') if s in summary.index]
    summary = summary.reindex(order)
    summary['n_assays'] = summary['n_assays'].astype(int)
    summary['n_rows'] = summary['n_rows'].astype(int)
    return per_assay, summary


def evaluate_oof(oof_df):
    """oof_df: one row per BindingGYM entry, columns
       DMS_id, num_muts, DMS_score, ddG (=-DMS_score), ddG_pred.
    Adds the BindingGYM-oriented prediction column and returns both metric families."""
    df = oof_df.copy()
    df['pred'] = -df['ddG_pred']          # larger = binds tighter, to match DMS_score
    h3_assay, h3_sum = per_dms_table(df, family='h3ddg')
    bg_assay, bg_sum = per_dms_table(df, family='bindinggym')
    return dict(h3ddg_per_assay=h3_assay, h3ddg_summary=h3_sum,
                bindinggym_per_assay=bg_assay, bindinggym_summary=bg_sum)
