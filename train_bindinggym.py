"""Train / evaluate H3-DDG on BindingGYM under the official inter-assay split.

One invocation = one fold.  Run it for --test_fold 0..4 to produce the complete
out-of-fold (OOF) prediction set over all 25 assays, which is what BindingGYM's own
`calc_metric.ipynb` consumes.

The model, the thermodynamic cycle and the collate function are the repo's own, unmodified;
only the data path, the fold handling and the per-DMS metrics are new (H3-DDG released no
BindingGYM code).
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from bindinggym_dataset import BindingGYMDatasetManager, complex_row_indices
from bindinggym_metrics import evaluate_oof
from ddg_predictor import DDGPredictor
from trainer import CrossValidation, recursive_to
from utils import check_dir, set_seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def process_batch(model, batch, device, is_train=True, optimizer=None):
    batch = recursive_to(batch, device)
    if is_train:
        model.train()
        loss, output_dict, _ = model(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
        optimizer.zero_grad()
    else:
        model.eval()
        with torch.no_grad():
            loss, output_dict, _ = model(batch)
    return loss, output_dict


def collect_results(model, dataloader, device, max_batches=None, desc='validate'):
    """Returns a DataFrame with one row per BindingGYM entry.

    Predictions are matched to metadata through complex_row_indices(), so this is correct for
    any eval batch size (the released SKEMPI loop relies on batch_size == 1).
    """
    rows, losses = [], []
    for bi, batch in enumerate(tqdm(dataloader, desc=desc, dynamic_ncols=True)):
        if max_batches is not None and bi >= max_batches:
            break
        loss, out = process_batch(model, batch, device, is_train=False)
        losses.append(loss.item())
        ci = complex_row_indices(batch['num_mut_chains']).tolist()
        for k, i in enumerate(ci):
            rows.append({
                'id': batch['id'][i],
                'DMS_id': batch['complex'][i],
                'num_muts': int(batch['num_muts'][i]),
                'ddG': float(out['ddG_true'][k].item()),
                'ddG_pred': float(out['ddG_pred'][k].item()),
            })
    df = pd.DataFrame(rows)
    if len(df):
        df['DMS_score'] = -df['ddG']              # dataset used ddG = -DMS_score
        df['row_index'] = df['id'].str.rsplit('#', n=1).str[-1].astype(int)
    return df, float(np.mean(losses)) if losses else float('nan')


def monitor_subset(dataset, per_assay):
    """Fixed, deterministic per-assay subsample used only for in-training monitoring."""
    by_assay = {}
    for pos, e in enumerate(dataset.entries):
        by_assay.setdefault(e['DMS_id'], []).append(pos)
    picked = []
    for dms_id in sorted(by_assay):
        idxs = by_assay[dms_id]
        if len(idxs) <= per_assay:
            picked.extend(idxs)
        else:
            step = len(idxs) / per_assay
            picked.extend([idxs[int(i * step)] for i in range(per_assay)])
    return sorted(picked)


def log_summary(tag, summaries, log_file):
    lines = [f"{time.strftime('%Y-%m-%d %H-%M-%S')} | [{tag}]"]
    for name in ('h3ddg_summary', 'bindinggym_summary'):
        lines.append(f'--- {name} ---')
        lines.append(summaries[name].to_string(float_format=lambda v: f'{v:.4f}'))
    msg = '\n'.join(lines)
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()


def main():
    parser = argparse.ArgumentParser(description='H3-DDG on BindingGYM (inter-assay split).')
    parser.add_argument('--config_path', type=str, default='./config/train_h3-ddg_bindinggym.json')
    parser.add_argument('--test_fold', type=int, required=True, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--reset_cache', action='store_true')
    parser.add_argument('--max_eval_batches', type=int, default=None,
                        help='cap the final evaluation (SMOKE TESTS ONLY -- never for reported runs)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='fixed output dir. REQUIRED for --resume to find anything: the default '
                             'is timestamped, so a requeued job would look in a brand-new directory.')
    parser.add_argument('--resume', action='store_true',
                        help='continue from the newest periodic checkpoint in --save_dir, if one exists.')
    parser.add_argument('--eval_only', action='store_true',
                        help='skip training entirely and evaluate the freshly initialised model: '
                             'pretrained ProteinMPNN + untrained H3-DDG heads, i.e. the '
                             'thermodynamic cycle with nothing fitted to BindingGYM. Diagnostic '
                             'baseline. Outputs are suffixed _untrained so they can never '
                             'overwrite a real run in the same --save_dir.')
    cli = parser.parse_args()
    if cli.eval_only and cli.resume:
        parser.error('--eval_only evaluates the untrained model; --resume would load weights into it')
    sfx = '_untrained' if cli.eval_only else ''

    param = {k: v for k, v in json.loads(open(cli.config_path).read()).items()
             if not k.startswith('_comment')}
    param['tag'] = cli.tag
    param['test_fold'] = cli.test_fold
    args = argparse.Namespace(**param)
    set_seed(args.seed)

    # Announce the resolved GPU on the first line of every run. nvidia-smi numbers by PCI bus and
    # CUDA defaults to FASTEST_FIRST, which on this workstation is the exact reverse -- so a
    # mis-picked device is otherwise invisible except as "seems slow".
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}')

    if cli.save_dir is not None:
        save_dir = cli.save_dir
        # NOTE: check_dir(overwrite=True) does shutil.rmtree -- it would delete the very
        # checkpoints --resume needs. Never use it on a caller-supplied save_dir.
        os.makedirs(os.path.join(save_dir, 'checkpoint'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'oof'), exist_ok=True)
    else:
        stamp = time.strftime('%Y-%m-%d-%H-%M-%S')
        save_dir = os.path.join('./results', f'{stamp}_fold{cli.test_fold}_{cli.tag}')
        check_dir(os.path.join(save_dir, 'checkpoint'))
        check_dir(os.path.join(save_dir, 'oof'))
    log_file = open(os.path.join(save_dir, 'train_log.txt'), 'a+')
    with open(os.path.join(save_dir, 'train_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f'save_dir = {save_dir}')

    print('Loading datasets...')
    mgr = BindingGYMDatasetManager(args, test_fold=cli.test_fold,
                                   num_workers=cli.num_workers, reset=cli.reset_cache)

    print('Building model...')
    cv_mgr = CrossValidation(config=args, num_cvfolds=1, model_factory=DDGPredictor).to('cpu')
    cv_mgr.load_mpnn_state_dict(torch.load(args.ckpt_path, map_location='cpu'))
    model, optimizer, _ = cv_mgr.get(0)
    model.to(device)

    # deterministic monitoring subset (never used for model selection, only for logging)
    mon_idx = monitor_subset(mgr.val_dataset, int(getattr(args, 'monitor_per_assay', 200)))
    from torch.utils.data import DataLoader, Subset
    from dataset import MPNNPaddingCollate
    mon_loader = DataLoader(Subset(mgr.val_dataset, mon_idx),
                            batch_size=int(getattr(args, 'eval_batch_size', 1)), shuffle=False,
                            collate_fn=MPNNPaddingCollate(), num_workers=cli.num_workers)
    print(f'monitoring subset: {len(mon_idx)} rows')

    # ---- periodic checkpointing so a walltime kill costs one interval, not the whole run ----
    # The released SKEMPI path only saves after the training loop finishes (and its periodic save is
    # gated behind num_cvfolds == 1), which is why fold1 of the SKEMPI run lost its weights at 97.4%.
    # Here we save every ckpt_freq iterations and can resume from the newest one.
    ckpt_dir = os.path.join(save_dir, 'checkpoint')
    ckpt_freq = int(getattr(args, 'ckpt_freq', 5000))
    state_path = os.path.join(ckpt_dir, f'resume_fold{cli.test_fold}.pt')

    start_it = 0
    if cli.resume and os.path.exists(state_path):
        blob = torch.load(state_path, map_location='cpu')
        cv_mgr.load_state_dict(blob['cv_mgr'])
        model, optimizer, _ = cv_mgr.get(0)
        model.to(device)
        start_it = int(blob['iteration']) + 1
        msg = (f"[resume] loaded {state_path} @ iteration {blob['iteration']}; "
               f"continuing at {start_it}/{args.max_iter}")
        print(msg); log_file.write(msg + '\n'); log_file.flush()
    elif cli.resume:
        print(f'[resume] no checkpoint at {state_path}; starting from scratch')

    def save_resume_point(its):
        # Written to a temp file then renamed: a kill mid-write must not corrupt the only copy.
        tmp = state_path + '.tmp'
        torch.save({'cv_mgr': cv_mgr.state_dict(), 'iteration': its,
                    'test_fold': cli.test_fold, 'max_iter': args.max_iter}, tmp)
        os.replace(tmp, state_path)
        m = f"[ckpt] saved resume point at iteration {its} -> {state_path}"
        print(m); log_file.write(m + '\n'); log_file.flush()

    if cli.eval_only:
        print('[eval-only] no training: evaluating pretrained ProteinMPNN + untrained H3-DDG heads')
        log_file.write('[eval-only] untrained baseline; no gradient step taken\n')
    else:
        train_loader = mgr.get_train_loader()
        t0 = time.time()
        for its in range(start_it, args.max_iter):
            batch = next(train_loader)
            loss, _ = process_batch(model, batch, device, is_train=True, optimizer=optimizer)

            if its % 100 == 1:
                rate = (its - start_it + 1) / max(time.time() - t0, 1e-6)
                msg = (f"{time.strftime('%Y-%m-%d %H-%M-%S')} | [train] iter {its}/{args.max_iter} "
                       f"| Loss {loss.item():.6f} | {rate:.2f} it/s")
                print(msg)
                log_file.write(msg + '\n')
                log_file.flush()

            if its > 0 and its % ckpt_freq == 0:
                save_resume_point(its)

            if its > 0 and its % args.val_freq == 1:
                df, vloss = collect_results(model, mon_loader, device, desc=f'monitor@{its}')
                log_file.write(f'[monitor] iter {its} val_loss {vloss:.6f}\n')
                log_summary(f'monitor iter {its}', evaluate_oof(df), log_file)

        save_resume_point(args.max_iter - 1)   # training finished; also the resume no-op point
        torch.save(cv_mgr.state_dict(),
                   os.path.join(ckpt_dir, f'h3ddg_bindinggym_fold{cli.test_fold}.ckpt'))

    if cli.max_eval_batches is not None:
        print(f'!! WARNING: final eval capped at {cli.max_eval_batches} batches -- SMOKE TEST ONLY')
    print('Final full evaluation on the held-out fold...')
    df, vloss = collect_results(model, mgr.get_val_loader(), device, desc='final-eval',
                               max_batches=cli.max_eval_batches)
    df = df.sort_values(['DMS_id', 'row_index']).reset_index(drop=True)
    df.to_csv(os.path.join(save_dir, f'oof_fold{cli.test_fold}{sfx}.csv'), index=False)
    for dms_id, sub in df.groupby('DMS_id'):
        sub.sort_values('row_index').to_csv(
            os.path.join(save_dir, 'oof', f'{dms_id}_oof{sfx}.csv'), index=False)

    res = evaluate_oof(df)
    for name, obj in res.items():
        obj.to_csv(os.path.join(save_dir, f'{name}_fold{cli.test_fold}{sfx}.csv'))
    log_file.write(f'[final] val_loss {vloss:.6f} | rows {len(df)}\n')
    log_summary(f'FINAL fold{cli.test_fold}{sfx}', res, log_file)
    print('DONE')


if __name__ == '__main__':
    main()
