"""H3-DDG on BindingGYM using BindingGYM's OWN inter-assay training strategy.

`train_bindinggym.py` follows H3-DDG's Appendix A.4 verbatim: plain MSE on raw ddG, batch_size 1,
20,000 iterations, uniform shuffling over all training rows.  That run collapses -- the output
becomes near-constant inside 5,000 iterations and per-assay correlations land at zero (see
ibex-records .../bindinggym_interassay_h3ddg_*.md S5.10).  This file is the alternative arm: the
recipe BindingGYM ships for exactly this split, transplanted onto the same H3-DDG model.

WHAT BINDINGGYM ACTUALLY DOES  (/home/guoj0f/repos/BindingGYM/training/{main,dataset,loss}.py)
  1. Every batch is drawn from ONE assay.  dataset.py sets `seed = index // batch_size` before
     picking the assay, so all batch_size consecutive indices resolve to the same assay, and the
     DataLoader runs with shuffle=False so consecutive indices share a batch.
  2. The loss is listMLE -- a listwise ranking loss over that batch.  Ranking is scale-free, which
     is the whole point: the 25 assays' labels are in unrelated units.  Note this is only
     definable because of (1); with batch_size 1 a "list" of one has no ranking to learn, so
     A.4's MSE-at-batch-size-1 recipe cannot express this strategy at all.
  3. The assay is drawn UNIFORMLY over assays, not proportionally to row count, so GB1 (92,891
     rows) and BH3 (518 rows) contribute equally.  Uniform shuffling over pooled rows instead
     hands almost all the gradient to the few largest assays.
  4. AdamW(lr=1e-3, betas=(0.9,0.99), weight_decay=0.05, eps=1e-5) + OneCycleLR, 256 steps per
     epoch, up to 100 epochs, early stopping with patience 3 on validation Spearman.

DEVIATIONS FROM THE OFFICIAL SCRIPT, AND WHY
  * Early stopping uses a fixed per-assay SUBSAMPLE of the held-out fold, not all of it.  The
    official loop evaluates the whole test fold every epoch, which here would cost ~0.5 h per
    epoch.  The selected weights are then scored once on the full fold.
  * BindingGYM selects the epoch on the TEST fold's Spearman (main.py: fold_valid is
    split[fold][1], the held-out cluster, and line 579 selects on valid_metrics['spearman']).
    That is test-set model selection and it inflates the reported number.  We reproduce it
    because the goal is to match their protocol, and `--select_on train_holdout` is provided to
    measure how much of the result depends on it.
  * The model, thermodynamic cycle and collate are H3-DDG's, unmodified.  Only the sampler, the
    loss and the optimiser schedule come from BindingGYM.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from bindinggym import BindingGYMDataset
from bindinggym_dataset import DEFAULTS, complex_row_indices
from bindinggym_metrics import evaluate_oof
from dataset import MPNNPaddingCollate
from ddg_predictor import DDGPredictor
from trainer import CrossValidation, recursive_to
from utils import set_seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def listMLE(y_pred, y_true, eps=1e-8):
    """Verbatim from BindingGYM training/loss.py (ListMLE, Xia et al. 2008).

    Both arguments are 1-D over the batch, which listMLE treats as a single slate -- matching the
    official call `loss_tr(-outputs, -y)`.
    """
    y_true_sorted, indices = y_true.sort(descending=True, dim=0)
    preds_sorted_by_true = torch.gather(y_pred, dim=0, index=indices)
    max_pred_values, _ = preds_sorted_by_true.max(dim=0, keepdim=True)
    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values
    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[0]), dim=0).flip(dims=[0])
    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max
    return observation_loss.mean()


class AssayBatchSampler(torch.utils.data.Sampler):
    """Yields batches whose members all belong to one assay, replicating BindingGYM's scheme.

    Per batch: draw an assay uniformly at random, then draw batch_size rows from it WITH
    replacement -- the official dataset does exactly this via np.random.randint. `set_epoch`
    reproduces their per-epoch reseeding (`train_dataset.seed_bias = epoch`).
    """

    def __init__(self, assay_positions, batch_size, steps_per_epoch, seed=42):
        self.assays = [np.asarray(v) for v in assay_positions]
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch * 1_000_003)
        for _ in range(self.steps_per_epoch):
            pool = self.assays[rng.randint(0, len(self.assays))]
            yield [int(pool[i]) for i in rng.randint(0, len(pool), size=self.batch_size)]


def probe_batch_size(model, dataset, positions, names, want, collate, say):
    """Largest batch size <= `want` that survives a real forward+backward on the WORST assay.

    BindingGYM trains at batch_size 8, but its structure model is plain ProteinMPNN.  H3-DDG adds
    3-body triplet attention over (K,K) hyperedge pairs with K = L/4, so peak memory grows with
    BOTH batch size and structure size -- which is what A.4's "batch size of 1, 2, depending on
    GPU memory and graph size" is describing.  Within-assay batching makes this sharper, not
    softer: every item in a batch shares one structure, so a large-structure assay hits the worst
    case on every one of its batches.

    Probing up front, on the largest structure, fixes one batch size for the whole run.  The
    alternative -- catching OOM mid-training and shortening the slate -- would have to re-slice an
    already-collated batch whose rows are per-mutated-side rather than per-item, and getting that
    index arithmetic subtly wrong would corrupt the labels instead of just crashing.
    """
    lens = []
    for p, nm in zip(positions, names):
        lens.append((collate([dataset[p[0]]])['X'].shape[1], nm, p))
    n_res, worst_name, worst_pos = max(lens)
    say(f'largest structure among training assays: {worst_name} ({n_res} residues); '
        f'probing batch sizes there')
    # Floor at 2, not 1: listMLE over a slate of one is identically zero, so the gradient
    # vanishes while AdamW's weight_decay=0.05 keeps decaying the weights -- the model degrades
    # with no learning signal at all. Observed in the local smoke run, where memory pressure
    # forced bs=1 and the loss printed as 0.000000 while the selection metric still moved.
    bs = want
    while bs >= 2:
        try:
            b = recursive_to(collate([dataset[i] for i in worst_pos[:bs]]), device)
            _, out, _ = model(b)
            listMLE(-out['ddG_pred'], -out['ddG_true']).backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            say(f'batch_size {bs} fits')
            return bs
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            say(f'batch_size {bs} does not fit; halving')
            bs //= 2
    raise SystemExit(f'batch_size 2 does not fit on this GPU for a {n_res}-residue structure. '
                     'listMLE needs a slate of at least 2, so there is nothing smaller to fall '
                     'back to -- this needs more GPU memory, not a smaller batch.')


def probe_eval_batch_size(model, dataset, want, collate, say):
    """Same idea as probe_batch_size, for the no-grad evaluation pass.

    Evaluation needs far less memory than training (no activations retained), but the triplet
    attention still allocates an O(K^2) softmax per item, so a batch of large structures can
    exceed what a batch of small ones does by an order of magnitude. Probed on the largest
    structure in THIS dataset so the whole eval runs at one safe batch size.
    """
    n_res = [(collate([dataset[i]])['X'].shape[1], i) for i in
             range(0, len(dataset), max(len(dataset) // 200, 1))]
    big, big_i = max(n_res)
    say(f'largest structure in this eval set: {big} residues; probing eval batch sizes')
    bs = want
    model.eval()
    while bs >= 1:
        try:
            with torch.no_grad():
                model(recursive_to(collate([dataset[min(big_i + k, len(dataset) - 1)]
                                            for k in range(bs)]), device))
            torch.cuda.empty_cache()
            say(f'eval_batch_size {bs} fits')
            return bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            say(f'eval_batch_size {bs} does not fit; halving')
            bs //= 2
    raise SystemExit('even eval_batch_size 1 does not fit')


def assay_positions(dataset):
    by = {}
    for pos, e in enumerate(dataset.entries):
        by.setdefault(e['DMS_id'], []).append(pos)
    return [by[k] for k in sorted(by)], sorted(by)


def monitor_subset(dataset, per_assay):
    """Deterministic, evenly-spaced per-assay subsample."""
    by = {}
    for pos, e in enumerate(dataset.entries):
        by.setdefault(e['DMS_id'], []).append(pos)
    picked = []
    for k in sorted(by):
        idxs = by[k]
        if len(idxs) <= per_assay:
            picked.extend(idxs)
        else:
            step = len(idxs) / per_assay
            picked.extend([idxs[int(i * step)] for i in range(per_assay)])
    return sorted(picked)


def collect_results(model, dataloader, desc='eval'):
    rows = []
    model.eval()
    for batch in tqdm(dataloader, desc=desc, dynamic_ncols=True):
        batch = recursive_to(batch, device)
        with torch.no_grad():
            _, out, _ = model(batch)
        for k, i in enumerate(complex_row_indices(batch['num_mut_chains']).tolist()):
            rows.append({'id': batch['id'][i], 'DMS_id': batch['complex'][i],
                         'num_muts': int(batch['num_muts'][i]),
                         'ddG': float(out['ddG_true'][k].item()),
                         'ddG_pred': float(out['ddG_pred'][k].item())})
    df = pd.DataFrame(rows)
    if len(df):
        df['DMS_score'] = -df['ddG']
        df['row_index'] = df['id'].str.rsplit('#', n=1).str[-1].astype(int)
    return df


def per_dms_spearman(df):
    """The quantity BindingGYM selects the epoch on: mean per-assay Spearman."""
    from scipy.stats import spearmanr
    vals = []
    for _, g in df.groupby('DMS_id'):
        if len(g) > 2 and g.ddG.std() > 0 and g.ddG_pred.std() > 0:
            vals.append(spearmanr(g.ddG, g.ddG_pred)[0])
    return float(np.mean(vals)) if vals else float('nan')


def main():
    ap = argparse.ArgumentParser(description="H3-DDG on BindingGYM, BindingGYM's own strategy.")
    ap.add_argument('--config_path', default='./config/train_h3-ddg_bindinggym_official.json')
    ap.add_argument('--test_fold', type=int, required=True, choices=[0, 1, 2, 3, 4])
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--max_eval_batches', type=int, default=None,
                    help='cap the final evaluation (SMOKE TESTS ONLY)')
    ap.add_argument('--probe_only', action='store_true',
                    help='resolve the train/eval batch sizes on THIS GPU, print them, and exit. '
                         'A 20-minute probe job is worth it before committing a 7h slot, since '
                         "the memory ceiling depends on the GPU and on the fold's largest "
                         'structure, and neither is knowable from the login node.')
    ap.add_argument('--batch_size', type=int, default=None,
                    help="override the config's batch size. BindingGYM uses 8, but their backbone "
                         "is plain ProteinMPNN; H3-DDG's 3-body triplet attention is O(K^2) in "
                         "memory, which is what A.4's 'batch size of 1, 2, depending on GPU memory "
                         "and graph size' is about. A batch that does not fit is halved at runtime "
                         "rather than crashing -- see the OOM handler below.")
    cli = ap.parse_args()

    param = {k: v for k, v in json.loads(open(cli.config_path).read()).items()
             if not k.startswith('_comment')}
    args = argparse.Namespace(**param)
    if cli.batch_size is not None:
        args.batch_size = cli.batch_size
    set_seed(args.seed)
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}')

    os.makedirs(os.path.join(cli.save_dir, 'checkpoint'), exist_ok=True)
    log = open(os.path.join(cli.save_dir, 'train_log.txt'), 'a+')
    with open(os.path.join(cli.save_dir, 'train_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    def say(msg):
        print(msg); log.write(msg + '\n'); log.flush()

    common = dict(DEFAULTS); common.pop('cache_dir', None)
    ds_kw = dict(dms_dir=DEFAULTS['dms_dir'], structure_dir=DEFAULTS['structure_dir'],
                 mapping_csv=DEFAULTS['mapping_csv'], folds_tsv=DEFAULTS['folds_tsv'],
                 sides_tsv=DEFAULTS['sides_tsv'], cache_dir=DEFAULTS['cache_dir'],
                 test_fold=cli.test_fold, reset=False)
    train_ds = BindingGYMDataset(split='train', **ds_kw)
    val_ds = BindingGYMDataset(split='val', **ds_kw)
    pos, assay_names = assay_positions(train_ds)
    say(f'[fold {cli.test_fold}] train {len(train_ds)} rows / {len(pos)} assays | '
        f'val {len(val_ds)} rows')
    say(f'[fold {cli.test_fold}] rows per training assay: '
        f'{dict(zip([a[:22] for a in assay_names], [len(p) for p in pos]))}')

    sampler = AssayBatchSampler(pos, args.batch_size, args.steps_per_epoch, seed=args.seed)
    # placeholder; rebuilt after the probe below, which needs the model on the device
    train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=MPNNPaddingCollate(),
                              num_workers=cli.num_workers)
    # select_per_assay = 0 means "the whole held-out fold", which is what BindingGYM actually
    # does (main.py evaluates fold_valid every epoch). A positive value subsamples it, which is
    # cheaper but makes the patience rule noisier -- see the f1 run in the record's S5.18.
    if int(args.select_per_assay) <= 0:
        sel_idx = list(range(len(val_ds)))
        say('selection set: the FULL held-out fold (BindingGYM\'s own protocol)')
    else:
        sel_idx = monitor_subset(val_ds, args.select_per_assay)
    sel_loader = DataLoader(Subset(val_ds, sel_idx), batch_size=args.eval_batch_size,
                            shuffle=False, collate_fn=MPNNPaddingCollate(),
                            num_workers=cli.num_workers)
    full_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False,
                             collate_fn=MPNNPaddingCollate(), num_workers=cli.num_workers)
    say(f'selection subset: {len(sel_idx)} rows')

    cv = CrossValidation(config=args, num_cvfolds=1, model_factory=DDGPredictor).to('cpu')
    cv.load_mpnn_state_dict(torch.load(args.ckpt_path, map_location='cpu'))
    model, _, _ = cv.get(0)
    model.to(device)

    fit_bs = probe_batch_size(model, train_ds, pos, assay_names, args.batch_size,
                              MPNNPaddingCollate(), say)
    if fit_bs != args.batch_size:
        say(f'!! batch_size reduced {args.batch_size} -> {fit_bs} to fit this GPU. '
            f"BindingGYM's value is 8; the shorter listMLE slate is a weaker ranking signal, "
            f'so record this alongside the result.')
        args.batch_size = fit_bs
        sampler = AssayBatchSampler(pos, fit_bs, args.steps_per_epoch, seed=args.seed)
        train_loader = DataLoader(train_ds, batch_sampler=sampler,
                                  collate_fn=MPNNPaddingCollate(), num_workers=cli.num_workers)

    eval_bs = probe_eval_batch_size(model, val_ds, args.eval_batch_size,
                                    MPNNPaddingCollate(), say)
    if cli.probe_only:
        say(f'PROBE RESULT fold {cli.test_fold}: train batch_size {fit_bs} '
            f'(wanted {param["batch_size"]}), eval_batch_size {eval_bs} '
            f'(wanted {param["eval_batch_size"]})')
        print('PROBE DONE')
        return
    if eval_bs != args.eval_batch_size:
        say(f'!! eval_batch_size reduced {args.eval_batch_size} -> {eval_bs}. This affects speed '
            f'only: complex_row_indices() aligns predictions to metadata at any batch size.')
        args.eval_batch_size = eval_bs
        sel_loader = DataLoader(Subset(val_ds, sel_idx), batch_size=eval_bs, shuffle=False,
                                collate_fn=MPNNPaddingCollate(), num_workers=cli.num_workers)
        full_loader = DataLoader(val_ds, batch_size=eval_bs, shuffle=False,
                                 collate_fn=MPNNPaddingCollate(), num_workers=cli.num_workers)

    # The optimiser is a MODEL TRAINING PARAMETER, so which side owns it is a deliberate choice,
    # not an implementation detail:
    #   optimizer=adam  + scheduler=none     -> H3-DDG's own setting (Adam, lr 4e-4, wd 0). Use
    #     this to isolate the training STRATEGY: only batching, loss and assay sampling differ
    #     from the A.4 arm, so a difference cannot be attributed to the optimiser.
    #   optimizer=adamw + scheduler=onecycle -> BindingGYM's full recipe (AdamW lr 1e-3, wd 0.05,
    #     OneCycleLR), i.e. their published 0.4217 configuration end to end.
    if getattr(args, 'optimizer', 'adamw').lower() == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99),
                                weight_decay=args.weight_decay, eps=1e-5)
    sched = None if getattr(args, 'scheduler', 'onecycle').lower() == 'none' else \
        torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                            steps_per_epoch=args.steps_per_epoch,
                                            epochs=args.max_epochs)
    say(f"optimizer {type(opt).__name__} lr {args.lr} weight_decay {args.weight_decay} "
        f"scheduler {'none' if sched is None else 'OneCycleLR'} | batch_size {args.batch_size} "
        f"| {args.steps_per_epoch} steps x <= {args.max_epochs} epochs "
        f"= <= {args.steps_per_epoch * args.max_epochs} total steps")

    state_path = os.path.join(cli.save_dir, 'checkpoint', f'resume_fold{cli.test_fold}.pt')
    best_path = os.path.join(cli.save_dir, 'checkpoint', f'best_fold{cli.test_fold}.pt')
    start_epoch, best, stale = 0, -1e9, 0
    if cli.resume and os.path.exists(state_path):
        blob = torch.load(state_path, map_location='cpu')
        cv.load_state_dict(blob['cv']); model, _, _ = cv.get(0); model.to(device)
        opt.load_state_dict(blob['opt'])
        if sched is not None and blob.get('sched') is not None:
            sched.load_state_dict(blob['sched'])
        start_epoch, best, stale = blob['epoch'] + 1, blob['best'], blob['stale']
        # Restore the RNG streams too, not just the weights. ProteinMPNN draws a random decoding
        # order from the global torch RNG on EVERY forward (protein_mpnn_utils.py:1587, inside the
        # misleadingly named deterministic_forward), so an uninterrupted run has advanced that
        # stream by thousands of draws by the time it reaches epoch N. Without this, a resumed run
        # replays the stream from set_seed() and diverges from epoch N onward -- measured on a toy
        # fold-2 run, where it selected a different epoch's weights (0.3600 vs 0.4158).
        if blob.get('rng') is not None:
            r = blob['rng']
            torch.set_rng_state(r['torch'])
            np.random.set_state(r['numpy'])
            if torch.cuda.is_available() and r.get('cuda') is not None:
                torch.cuda.set_rng_state_all(r['cuda'])
            say('[resume] RNG streams restored')
        else:
            say('[resume] WARNING: checkpoint predates RNG capture; this run will diverge '
                'from an uninterrupted one after this epoch')
        say(f'[resume] epoch {blob["epoch"]} done; best per-DMS Spearman {best:.4f}; '
            f'continuing at {start_epoch}/{args.max_epochs}')
        if stale >= args.patience:
            # Early stopping had already fired before the kill, so the run was in its final
            # evaluation. The patience check sits at the END of the loop body, so re-entering
            # the loop would train one more epoch before noticing -- and if that epoch happened
            # to beat `best` it would overwrite best_fold*.pt, making a killed-and-resumed run
            # select different weights from an uninterrupted one. Skip straight to evaluation.
            say(f'[resume] early stopping had already fired (stale {stale}/{args.patience}); '
                f'skipping training, going straight to the final evaluation')
            start_epoch = args.max_epochs

    for epoch in range(start_epoch, args.max_epochs):
        sampler.set_epoch(epoch)
        model.train()
        t0, tot = time.time(), 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f'train e{epoch}',
                                          dynamic_ncols=True)):
            batch = recursive_to(batch, device)
            _, out, _ = model(batch)
            # Mirrors BindingGYM's `loss_tr(-outputs, -y)`.  Equivalent to listMLE(pred, true)
            # for ranking; kept in their form so the call site matches theirs exactly.
            loss = listMLE(-out['ddG_pred'], -out['ddG_true'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            opt.step(); opt.zero_grad()
            if sched is not None:
                sched.step()
            tot += loss.item()
        df_sel = collect_results(model, sel_loader, desc=f'select e{epoch}')
        rho = per_dms_spearman(df_sel)
        say(f"{time.strftime('%Y-%m-%d %H-%M-%S')} | epoch {epoch}/{args.max_epochs} | "
            f"listMLE {tot / max(step + 1, 1):.6f} | lr {opt.param_groups[0]['lr']:.2e} | "
            f"per-DMS Spearman(sel) {rho:.4f} | best {best:.4f} | {time.time() - t0:.0f}s")

        if rho > best:
            best, stale = rho, 0
            torch.save(cv.state_dict(), best_path)
            say(f'  new best -> {best_path}')
        else:
            stale += 1
            say(f'  no improvement ({stale}/{args.patience})')
        tmp = state_path + '.tmp'
        torch.save({'cv': cv.state_dict(), 'opt': opt.state_dict(),
                    'sched': None if sched is None else sched.state_dict(),
                    'epoch': epoch, 'best': best, 'stale': stale,
                    'rng': {'torch': torch.get_rng_state(),
                            'numpy': np.random.get_state(),
                            'cuda': torch.cuda.get_rng_state_all()
                                    if torch.cuda.is_available() else None}}, tmp)
        os.replace(tmp, state_path)
        if stale >= args.patience:
            say(f'early stopping at epoch {epoch}')
            break

    say('loading best weights for the full held-out evaluation')
    cv.load_state_dict(torch.load(best_path, map_location='cpu'))
    model, _, _ = cv.get(0); model.to(device)
    if cli.max_eval_batches is not None:
        say(f'!! final eval capped at {cli.max_eval_batches} batches -- SMOKE TEST ONLY')
        full_loader = [b for i, b in zip(range(cli.max_eval_batches), full_loader)]
    df = collect_results(model, full_loader, desc='final-eval')
    df = df.sort_values(['DMS_id', 'row_index']).reset_index(drop=True)
    df.to_csv(os.path.join(cli.save_dir, f'oof_fold{cli.test_fold}_official.csv'), index=False)
    res = evaluate_oof(df)
    for name, obj in res.items():
        obj.to_csv(os.path.join(cli.save_dir, f'{name}_fold{cli.test_fold}_official.csv'))
    say(f'FINAL fold{cli.test_fold} official-strategy | rows {len(df)} | best sel Spearman {best:.4f}')
    for name in ('h3ddg_summary', 'bindinggym_summary'):
        say(f'--- {name} ---\n' + res[name].to_string(float_format=lambda v: f'{v:.4f}'))
    print('DONE')


if __name__ == '__main__':
    main()
