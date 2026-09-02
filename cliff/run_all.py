"""BGYM-CLIFF v1 -- stage driver.  Spec Sec.5 runtime plan, Sec.4 provenance lines.

DO NOT CHANGE: the two hard scheduling rules -- stage 5 (Ward, up to 3.6 GB/assay) may
NEVER run while stage 3 or 4 holds the heavy lock, and stage 3/4 workers are capped at
THRESH['nproc_cap'] so 64 x 0.5 GB stays inside 111 GB.  And keep every stats import
LAZY, inside the stage: the driver and --dry-run must work before those modules exist.
"""
from __future__ import annotations

import argparse
import errno
import fcntl
import importlib
import inspect
import os
import resource
import subprocess
import sys
import time
from dataclasses import dataclass, field

from cliff import config
from cliff.config import PATHS, THRESH

# --------------------------------------------------------------------------- #
# stage table -- spec Sec.5, transcribed                                      #
# --------------------------------------------------------------------------- #

#: Locks.  ``heavy`` is taken SHARED by stages 3 and 4 and EXCLUSIVE by stage 5,
#: which is exactly the spec's rule: "stage 5 never runs concurrently with stage
#: 3 or 4", while leaving 3 || 4 (which the spec prices at 64 workers each in
#: separate wall-clock slots) and serialising stage 5 against itself, which is
#: what "SERIAL (2 concurrent max)" means for the driver.
LOCK_NONE, LOCK_SHARED, LOCK_EXCLUSIVE = None, 'shared', 'exclusive'


@dataclass(frozen=True)
class Stage:
    """One row of spec Sec.5's table plus its entry points and its lock."""
    n: int
    work: str
    per_assay: str
    wall: str
    nproc: int
    rss: str
    entry: tuple = ()          # ((module, (attr, alt, ...)), ...) called in order
    assays: str = 'all'        # which default assay set (see _assay_set)
    lock: object = LOCK_NONE
    writes: tuple = ()
    needs_manifest: bool = True


STAGES = {s.n: s for s in (
    Stage(0, 'load 28 (usecols) + G1/G1b/G2/G3 + enumerate nested & same-site + '
             '2e7 random-pair sample x 14 + G0 benchmark',
          '2-12 s', '~7 min', 1, '2 GB',
          entry=(('cliff.pairs', ('stage0',)),), assays='all28',
          writes=('T01_assay_manifest.csv', 'T02_gates.csv',
                  'data/cliff_cache/{keys,pairs,randpairs}/*.npz', 'MANIFEST.json'),
          needs_manifest=False),
    Stage(1, 'structural annotation, 25 assays (+ G-OPT if PDBs fetchable)',
          '1.5 s', '40 s', 1, '1 GB',
          entry=(('cliff.structure', ('stage1', 'run_all', 'run')),),
          assays='with_pdb',
          writes=('T09_structure_sites.csv', 'data/cliff_cache/structure/*.npz')),
    Stage(2, 'fit_latent + 5-fold cross-fit, 17 assays',
          '1.3 s / 7 s (GB1)', '2 min', 17, '17 x 0.4 GB',
          entry=(('cliff.latent', ('stage2', 'run_all', 'run')),
                 ('cliff.noise', ('stage2', 'run_all', 'run', 'sigma_registry')),
                 ('cliff.variogram', ('stage2', 'run_all', 'run'))),
          assays='primary_arm_control',
          writes=('T03_noise_registry.csv', 'data/cliff_cache/latent/*.npz')),
    Stage(3, 'null ensembles: 4 nulls x 200 reps x 17 assays = 13,600 '
             'replicate-jobs on the CACHED pair index arrays',
          '~6 s GB1 / ~1.5 s median', '~9 min', THRESH['nproc_cap'], '32 GB',
          entry=(('cliff.nulls', ('stage3', 'run_all', 'run')),
                 ('cliff.stats_c2', ('stage3', 'run_all', 'run')),
                 ('cliff.stats_c3', ('stage3', 'run_all', 'run'))),
          assays='primary_arm_control',
          lock=LOCK_SHARED, writes=('data/cliff_cache/nulls/*.npz',)),
    Stage(4, 'G4 (reuses stage 3, free), G7 localisation on N2c, G8 power grid '
             '(6 x 3 x 3 x 40), G9 rule FPR (50 full-pipeline surrogates)',
          '-', '~5 min', THRESH['nproc_cap'], '32 GB',
          entry=(('cliff.calibrate', ('stage4', 'run_all', 'run')),),
          assays='primary_arm_control',
          lock=LOCK_SHARED, writes=('T02_gates.csv (G4/G7/G8/G9/G10 rows)',)),
    Stage(5, 'cluster channel, 6 assays, n <= 30,000, SERIAL (2 concurrent max)',
          '5-60 s, 0.2-3.6 GB', '~6 min', 2, '7 GB',
          entry=(('cliff.clusters', ('stage5', 'run_all', 'run')),),
          assays='cluster',
          lock=LOCK_EXCLUSIVE, writes=('T15_cluster_channel.csv',)),
    Stage(6, 'C4: GLM + 10,000 NS1 x 7; NS2 x 4; NS3 x 10,000 on KRAS; '
             'PSD95/BH3/5A12 probes',
          '-', '~5 min', 8, '4 GB',
          entry=(('cliff.stats_c4', ('stage6', 'run_all', 'run')),), assays='c4',
          writes=('T09_structure_sites.csv', 'T10_structure_pairs.csv',
                  'T11_partner_specificity.csv')),
    Stage(7, 'C5: three CPU models + PSA/AUPSA x 14 (MSA read from msas/*.a2m)',
          '-', '~3 min', 14, '4 GB',
          entry=(('cliff.stats_c5', ('stage7', 'run_all', 'run')),),
          assays='primary_arm',
          writes=('T12_cliff_aware_eval.csv',)),
    Stage(8, 'verdict tables T1-T12 + figures F1-F7',
          '-', '~3 min', 1, '2 GB',
          entry=(('cliff.verdict', ('run',)),
                 ('cliff.figures', ('run', 'run_all', 'stage8'))),
          assays='all28',
          writes=('T14_verdict_by_family.csv', 'T13_sensitivity.csv',
                  'F1..F7 .pdf/.png')),
)}

#: Sec.5's own totals, printed by --dry-run so the plan is checkable at a glance.
TOTAL_WALL = '~45 min'
TOTAL_PEAK_RSS = '32 GB'

# The cap is a decision, so it lives in THRESH and only there.
assert STAGES[3].nproc == STAGES[4].nproc == THRESH['nproc_cap'], \
    'stage 3/4 nproc must be THRESH["nproc_cap"]'


def effective_nproc(n, nproc=None, announce=False):
    """Worker count for stage ``n``: the stage's own Sec.5 figure unless the
    caller overrides it, then capped at ``THRESH['nproc_cap']``.

    Spec Sec.5: "Stage 3/4 workers are capped at 64 (not 80) so that 64 x 0.5 GB
    stays well inside 111 GB."  Applied to EVERY stage -- none is priced above 64
    workers, and an --nproc typo must not be able to oversubscribe an 80-core box.
    """
    want = STAGES[n].nproc if nproc is None else int(nproc)
    got = min(want, THRESH['nproc_cap'])
    if announce and got != want:
        print('[cap] stage %d nproc %d -> %d (THRESH["nproc_cap"])' % (n, want, got))
    return got


def _assay_set(name):
    """Default assay list for a stage (spec Sec.5's per-stage counts)."""
    if name == 'all28':
        return list(config.ALL_ASSAYS)
    if name == 'with_pdb':
        return [a for a in config.ALL_ASSAYS
                if os.path.exists(os.path.join(PATHS.structures,
                                               config.ASSAYS[a].pdb_file))]
    if name == 'primary_arm_control':
        return list(config.PRIMARY + config.ARM + config.CONTROL)
    if name == 'primary_arm':
        return list(config.PRIMARY_AND_ARM)
    if name == 'cluster':
        return [a for a in config.ALL_ASSAYS
                if config.ASSAYS[a].eligible_cluster_channel]
    if name == 'c4':
        return [a for a in config.ALL_ASSAYS
                if (config.ASSAYS[a].eligible_C4S or config.ASSAYS[a].eligible_C4P
                    or config.ASSAYS[a].eligible_C4I)]
    return list(config.ALL_ASSAYS)


# --------------------------------------------------------------------------- #
# the three provenance lines -- spec Sec.4                                    #
# --------------------------------------------------------------------------- #

def _git():
    try:
        head = subprocess.check_output(['git', '-C', config.REPO, 'rev-parse', 'HEAD'],
                                       stderr=subprocess.DEVNULL).decode().strip()
        dirty = len(subprocess.check_output(
            ['git', '-C', config.REPO, 'status', '--porcelain'],
            stderr=subprocess.DEVNULL).decode().splitlines())
        return head, dirty
    except Exception:                                          # pragma: no cover
        return '', -1


def data_fingerprint():
    """Line 3: what data this run is about to read."""
    import glob
    n_csv = len(glob.glob(os.path.join(PATHS.dms_dir, '*.csv')))
    n_pdb = len(glob.glob(os.path.join(PATHS.structures, '*.pdb')))
    n_msa = len(glob.glob(os.path.join(PATHS.msas, '*.a2m')))
    man = 'absent (run --stage 0)'
    if os.path.exists(PATHS.manifest):
        try:
            from cliff.io_bgym import md5_of
            man = 'md5 %s' % md5_of(PATHS.manifest)[:12]
        except Exception:                                      # pragma: no cover
            man = 'present'
    return ('%d DMS csv, %d structures, %d msas  BINDINGGYM_INPUT=%s  MANIFEST %s'
            % (n_csv, n_pdb, n_msa, PATHS.bgym_input, man))


def preamble(stream=sys.stdout):
    """The first three lines of EVERY run (spec Sec.4): the env tuple assertion,
    the git commit + dirty count, and the data fingerprint.  A wrong env, a dirty
    tree or the wrong data shows up in the log's first three lines."""
    env = config.assert_env()
    head, dirty = _git()
    print('[env] %s == EXPECTED_ENV  OK' % ('.'.join(env),), file=stream)
    print('[synced_commit] %s  [dirty] %d' % (head or 'UNKNOWN', dirty), file=stream)
    print('[data] %s' % data_fingerprint(), file=stream)
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print('[warn] CUDA_VISIBLE_DEVICES is set; this study is CPU-only and never '
              'sets it (spec Sec.4)', file=stream)
    return env, head, dirty


# --------------------------------------------------------------------------- #
# locks -- the hard scheduling rule                                           #
# --------------------------------------------------------------------------- #

LOCK_DIR = os.path.join(PATHS.cache, '.locks')
HEAVY_LOCK = os.path.join(LOCK_DIR, 'heavy.lock')


class LockBusy(RuntimeError):
    pass


class _Lock(object):
    """flock on ``heavy.lock``: SHARED for stages 3/4, EXCLUSIVE for stage 5."""

    def __init__(self, mode, stage, timeout=0.0, verbose=True):
        self.mode, self.stage, self.timeout, self.verbose = mode, stage, timeout, verbose
        self.fh = None

    def __enter__(self):
        if self.mode is LOCK_NONE:
            return self
        os.makedirs(LOCK_DIR, exist_ok=True)
        op = fcntl.LOCK_SH if self.mode == LOCK_SHARED else fcntl.LOCK_EX
        self.fh = open(HEAVY_LOCK, 'a+')
        t0 = time.time()
        while True:
            try:
                fcntl.flock(self.fh.fileno(), op | fcntl.LOCK_NB)
                break
            except (IOError, OSError) as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if time.time() - t0 >= self.timeout:
                    self.fh.close()
                    self.fh = None
                    raise LockBusy(
                        'stage %d wants the %s heavy lock but another stage 3/4/5 '
                        'holds it. Spec Sec.5: stage 5 NEVER runs concurrently '
                        'with stage 3 or 4. '
                        'Wait, or pass --lock-timeout SECONDS.  Lock file: %s'
                        % (self.stage, self.mode, HEAVY_LOCK))
                time.sleep(0.5)
        self.fh.seek(0, os.SEEK_END)
        self.fh.write('stage %d %s pid %d %s\n'
                      % (self.stage, self.mode, os.getpid(),
                         time.strftime('%Y-%m-%dT%H:%M:%S')))
        self.fh.flush()
        if self.verbose:
            print('[lock] stage %d holds the %s heavy lock (%s)'
                  % (self.stage, self.mode, HEAVY_LOCK))
        return self

    def __exit__(self, *exc):
        if self.fh is not None:
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
            self.fh.close()
            self.fh = None
            if self.verbose:
                print('[lock] stage %d released the heavy lock' % self.stage)
        return False


# --------------------------------------------------------------------------- #
# manifest verification -- spec Sec.5                                         #
# --------------------------------------------------------------------------- #

def verify_manifest_or_die(stage, dry_run=False):
    """Spec Sec.5: "Downstream code verifies the md5 before use and refuses to run
    on a mismatch." Stage 0 builds the manifest, so it is exempt."""
    if not STAGES[stage].needs_manifest:
        return 'exempt (stage %d builds the manifest)' % stage
    if not os.path.exists(PATHS.manifest):
        msg = ('%s does not exist: run --stage 0 first' % PATHS.manifest)
        if dry_run:
            return 'WOULD REFUSE -- ' + msg
        raise SystemExit('[refuse] ' + msg)
    from cliff.pairs import verify_manifest
    bad = verify_manifest()
    if bad:
        lines = '\n'.join('  %s  got %s  want %s' % b for b in bad[:20])
        msg = ('%d cache file(s) do not match MANIFEST.json:\n%s'
               % (len(bad), lines))
        if dry_run:
            return 'WOULD REFUSE -- ' + msg
        raise SystemExit('[refuse] ' + msg)
    return 'clean'


# --------------------------------------------------------------------------- #
# stage execution                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class StageResult:
    n: int
    status: str                 # ok | missing | error | skipped
    wall_s: float = 0.0
    peak_rss_gb: float = 0.0
    detail: str = ''
    called: tuple = field(default_factory=tuple)


def _resolve(module, attrs):
    """Import ``module`` LAZILY and return the first of ``attrs`` it defines.

    A tuple of candidates, not one name, because each statistics module names its
    own driver: ``structure.stage1``, ``latent.run_all``, ... The driver adapts to
    the module rather than the module to the driver."""
    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        return None, None, 'missing: %s (%s)' % (module, exc)
    for a in attrs:
        fn = getattr(mod, a, None)
        if fn is not None and callable(fn):
            return fn, a, 'ok'
    return None, None, ('missing: %s defines none of %s yet'
                        % (module, '/'.join(attrs)))


def _call(module, attrs, assays, nproc, verbose):
    """Resolve and call, passing only the keyword arguments the signature accepts."""
    fn, attr, status = _resolve(module, attrs)
    if fn is None:
        return None, status
    kw = {}
    try:
        sig = inspect.signature(fn)
        names = set(sig.parameters)
        has_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        for k, v in (('assays', assays), ('nproc', nproc), ('verbose', verbose)):
            if k in names or has_kwargs:
                kw[k] = v
    except (TypeError, ValueError):                            # pragma: no cover
        kw = {}
    if verbose:
        print('[stage] -> %s.%s(%s)'
              % (module, attr,
                 ', '.join('%s=%r' % (k, (('%d assays' % len(v))
                                          if k == 'assays' else v))
                           for k, v in sorted(kw.items()))))
    return fn(**kw), 'ok'


def run_stage(n, assays=None, nproc=None, dry_run=False, skip_missing=False,
              lock_timeout=0.0, verbose=True):
    """Run one stage.  Honours the lock, the nproc cap and the manifest check."""
    st = STAGES[n]
    ids = list(assays) if assays else _assay_set(st.assays)
    np_eff = effective_nproc(n, nproc, announce=True)
    man = verify_manifest_or_die(n, dry_run=dry_run)
    if dry_run:
        return StageResult(n, 'dry-run', detail='manifest: %s' % man,
                           called=tuple('%s.%s' % (m, a[0]) for m, a in st.entry))
    t0 = time.time()
    called, statuses = [], []
    try:
        with _Lock(st.lock, n, timeout=lock_timeout, verbose=verbose):
            for module, attrs in st.entry:
                _, status = _call(module, attrs, ids, np_eff, verbose)
                called.append('%s.%s' % (module, attrs[0]))
                statuses.append(status)
                if status != 'ok':
                    print('[stage %d] %s' % (n, status))
                    if not skip_missing:
                        raise SystemExit(
                            '[stop] stage %d cannot run: %s\n'
                            '       pass --skip-missing to continue past a module '
                            'that does not exist yet.' % (n, status))
    except LockBusy as exc:
        return StageResult(n, 'skipped', time.time() - t0, detail=str(exc),
                           called=tuple(called))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    ok = all(s == 'ok' for s in statuses)
    return StageResult(n, 'ok' if ok else 'missing', round(time.time() - t0, 2),
                       round(rss, 3), '; '.join(statuses), tuple(called))


# --------------------------------------------------------------------------- #
# --dry-run plan                                                              #
# --------------------------------------------------------------------------- #

def _entry_state(module, attrs):
    fn, attr, _ = _resolve(module, attrs)
    return ('%s() ready' % attr) if fn is not None else 'ABSENT'


def print_plan(stages, assays=None, nproc=None, stream=sys.stdout):
    """The --dry-run plan: spec Sec.5's own table, the resolved assay set, the
    lock each stage takes, the entry points and whether they exist yet."""
    print('\n%-3s %-9s %-7s %-6s %-11s %-11s %s'
          % ('st', 'wall', 'nproc', 'assay', 'peak RSS', 'lock', 'entry points'),
          file=stream)
    print('-' * 118, file=stream)
    for n in stages:
        st = STAGES[n]
        ids = list(assays) if assays else _assay_set(st.assays)
        np_eff = effective_nproc(n, nproc)
        ent = ', '.join('%s [%s]' % (m, _entry_state(m, a)) for m, a in st.entry)
        print('%-3d %-9s %-7d %-6d %-11s %-11s %s'
              % (n, st.wall, np_eff, len(ids), st.rss,
                 st.lock or '-', ent), file=stream)
        print('    work   : %s' % st.work, file=stream)
        print('    assays : %s (%s)' % (st.assays, ', '.join(ids[:3]) +
                                        (' ...' if len(ids) > 3 else '')),
              file=stream)
        print('    writes : %s' % ', '.join(st.writes), file=stream)
        print('    manifest: %s' % verify_manifest_or_die(n, dry_run=True),
              file=stream)
    print('-' * 118, file=stream)
    print('spec Sec.5 total: %s wall, %s peak RSS.  HARD RULE: stage 5 takes the '
          'heavy lock EXCLUSIVE, stages 3 and 4 take it SHARED, so 5 can never run '
          'beside 3 or 4.  Stage 3/4 nproc capped at THRESH["nproc_cap"] = %d.'
          % (TOTAL_WALL, TOTAL_PEAK_RSS, THRESH['nproc_cap']), file=stream)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def parse_stages(text):
    """``'3'`` | ``'0-8'`` | ``'0,1,5'`` | ``'all'`` -> a sorted list, ordered so
    stage 5 always follows 3 and 4."""
    if text is None:
        return []
    t = str(text).strip().lower()
    if t in ('all', '*'):
        return sorted(STAGES)
    out = set()
    for part in t.replace(' ', '').split(','):
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    bad = sorted(x for x in out if x not in STAGES)
    if bad:
        raise SystemExit('[error] unknown stage(s) %s; valid: %s'
                         % (bad, sorted(STAGES)))
    return sorted(out)


def parse_assays(text):
    """Names, or a tier / family alias.  Validated against the 28-file registry."""
    if not text:
        return None
    t = str(text).strip()
    alias = {'all': list(config.ALL_ASSAYS), 'primary': list(config.PRIMARY),
             'arm': list(config.ARM), 'control': list(config.CONTROL),
             'excluded': list(config.EXCLUDED),
             'primary_and_arm': list(config.PRIMARY_AND_ARM)}
    if t.lower() in alias:
        return alias[t.lower()]
    if t.upper() in config.FAMILIES:
        return list(config.FAMILIES[t.upper()])
    ids = [x for x in t.replace(',', ' ').split() if x]
    bad = [x for x in ids if x not in config.ASSAYS]
    if bad:
        raise SystemExit('[error] unknown assay(s):\n  %s\nvalid aliases: %s, %s\n'
                         'valid ids: %s'
                         % ('\n  '.join(bad), ', '.join(sorted(alias)),
                            ', '.join(sorted(config.FAMILIES)),
                            ', '.join(config.ALL_ASSAYS)))
    return ids


def build_parser():
    p = argparse.ArgumentParser(
        prog='python -m cliff.run_all',
        description='BGYM-CLIFF v1 stage driver (spec Sec.5).  CPU-only, no '
                    'scheduler, no GPU.',
        epilog='examples:\n'
               '  python -m cliff.run_all --stage 0\n'
               '  python -m cliff.run_all --stage all --dry-run\n'
               '  python -m cliff.run_all --stage 3 --nproc 32\n'
               '  python -m cliff.run_all --stage 2 --assays F2\n',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage', default='all',
                   help='0..8, a range (0-8), a list (0,1,5) or "all" (default)')
    p.add_argument('--assays', default=None,
                   help='DMS_ids, or one of all/primary/arm/control/excluded/'
                        'primary_and_arm/F1..F8.  Default: the stage\'s own set.')
    p.add_argument('--nproc', type=int, default=None,
                   help='override the stage\'s worker count; stages 3/4 are still '
                        'capped at THRESH["nproc_cap"] = %d' % THRESH['nproc_cap'])
    p.add_argument('--dry-run', action='store_true',
                   help='print the plan and the manifest check, run nothing')
    p.add_argument('--skip-missing', action='store_true',
                   help='continue past a stage whose module does not exist yet')
    p.add_argument('--lock-timeout', type=float, default=0.0,
                   help='seconds to wait for the heavy lock (default 0 = fail fast)')
    p.add_argument('--list-stages', action='store_true',
                   help='print spec Sec.5\'s stage table and exit')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    stages = parse_stages(args.stage)
    assays = parse_assays(args.assays)

    preamble()                      # the three lines, always, before anything else

    if args.list_stages:
        print_plan(sorted(STAGES))
        return 0
    if not stages:
        print('[error] nothing to do: --stage selected no stage')
        return 2

    print('\n[plan] stages %s   assays %s   nproc %s   dry_run=%s'
          % (stages, ('%d given' % len(assays)) if assays else 'per-stage default',
             args.nproc if args.nproc is not None else 'per-stage default',
             args.dry_run))
    if 5 in stages and (3 in stages or 4 in stages):
        print('[plan] stage 5 is ordered AFTER stages 3/4 in this run, and takes '
              'the heavy lock exclusively, so the Sec.5 rule holds within this '
              'process as well as across processes.')
    if args.dry_run:
        print_plan(stages, assays, args.nproc)
        return 0

    t0 = time.time()
    results = []
    for n in stages:
        print('\n' + '#' * 100)
        print('# STAGE %d -- %s' % (n, STAGES[n].work[:80]))
        print('#' * 100, flush=True)
        r = run_stage(n, assays=assays, nproc=args.nproc,
                      skip_missing=args.skip_missing,
                      lock_timeout=args.lock_timeout)
        results.append(r)
        print('[stage %d] %s  wall %.2fs  peak RSS %.2f GB  %s'
              % (r.n, r.status, r.wall_s, r.peak_rss_gb, r.detail[:120]), flush=True)

    print('\n' + '=' * 100)
    print('%-5s %-9s %9s %11s  %s' % ('stage', 'status', 'wall_s', 'peakRSS_GB',
                                      'entry points'))
    for r in results:
        print('%-5d %-9s %9.2f %11.2f  %s'
              % (r.n, r.status, r.wall_s, r.peak_rss_gb, ', '.join(r.called)))
    print('total wall %.1f s (spec Sec.5 estimate for the full run: %s)'
          % (time.time() - t0, TOTAL_WALL))
    return 0 if all(r.status in ('ok', 'dry-run') for r in results) else 1


# --------------------------------------------------------------------------- #
# self-check                                                                  #
# --------------------------------------------------------------------------- #

def _selfcheck():
    print('=' * 100)
    print('cliff.run_all self-check -- --dry-run every stage, then the lock rule')
    print('=' * 100)
    rc = main(['--stage', 'all', '--dry-run'])
    assert rc == 0
    print('\n[run_all] --stage all --dry-run exit code %d' % rc)

    # stage/assay-set counts against spec Sec.5's own per-stage counts
    want = {0: 28, 1: 25, 2: 17, 3: 17, 4: 17, 5: 6, 7: 14}
    for n, k in sorted(want.items()):
        got = len(_assay_set(STAGES[n].assays))
        print('[run_all] stage %d assay set = %2d (spec Sec.5 says %2d)  %s'
              % (n, got, k, 'OK' if got == k else 'MISMATCH'))
        assert got == k, (n, got, k)
    print('[run_all] stage 6 (C4-eligible) assay set = %d'
          % len(_assay_set(STAGES[6].assays)))

    # the argument parser
    assert parse_stages('all') == list(range(9))
    assert parse_stages('0-3') == [0, 1, 2, 3]
    assert parse_stages('5,0,2') == [0, 2, 5]
    assert parse_stages('4') == [4]
    print('[run_all] --stage parsing: all / 0-3 / 5,0,2 / 4  OK')
    assert parse_assays('F2') == list(config.FAMILIES['F2'])
    assert len(parse_assays('primary')) == 12
    assert parse_assays(None) is None
    try:
        parse_assays('NOT_AN_ASSAY')
    except SystemExit as exc:
        print('[run_all] --assays validation refuses an unknown id: %s'
              % str(exc).splitlines()[0])
    else:
        raise AssertionError('an unknown assay id was accepted')

    # the nproc cap
    for n in (3, 4):
        assert STAGES[n].nproc == THRESH['nproc_cap']
    print('[run_all] stage 3/4 nproc == THRESH["nproc_cap"] == %d  OK'
          % THRESH['nproc_cap'])
    default_np = [effective_nproc(n) for n in sorted(STAGES)]
    capped_np = [effective_nproc(n, 200) for n in sorted(STAGES)]
    print('[run_all] nproc per stage, defaults      = %s' % default_np)
    print('[run_all] nproc per stage, --nproc 200   = %s  (all capped at %d)'
          % (capped_np, THRESH['nproc_cap']))
    assert default_np == [STAGES[n].nproc for n in sorted(STAGES)]
    assert set(capped_np) == {THRESH['nproc_cap']}, capped_np

    # THE HARD RULE, exercised for real with two live flocks
    print('\n[run_all] exercising the stage-5-vs-3/4 lock for real:')
    with _Lock(LOCK_SHARED, 3, verbose=True):
        try:
            with _Lock(LOCK_EXCLUSIVE, 5, verbose=False):
                raise AssertionError('stage 5 acquired the lock while stage 3 held '
                                     'it -- the hard rule is BROKEN')
        except LockBusy as exc:
            print('[run_all]   stage 5 REFUSED while stage 3 holds it: %s'
                  % str(exc).split('.')[0])
        with _Lock(LOCK_SHARED, 4, verbose=False):
            print('[run_all]   stage 4 shares the lock with stage 3 (allowed by '
                  'Sec.5, which only forbids 5 beside 3/4)')
    with _Lock(LOCK_EXCLUSIVE, 5, verbose=True):
        for other, mode in ((3, LOCK_SHARED), (4, LOCK_SHARED), (5, LOCK_EXCLUSIVE)):
            try:
                with _Lock(mode, other, verbose=False):
                    raise AssertionError('stage %d ran beside stage 5' % other)
            except LockBusy:
                print('[run_all]   stage %d REFUSED while stage 5 holds it '
                      'exclusively' % other)
    print('[run_all] lock rule verified in both directions')

    # the mandatory manifest check (spec Sec.5: refuse on a mismatch)
    print('\n[run_all] manifest gate:')
    print('[run_all]   stage 0 -> %s' % verify_manifest_or_die(0, dry_run=True))
    print('[run_all]   stage 3 -> %s' % verify_manifest_or_die(3, dry_run=True)[:180])
    dirty_cache = verify_manifest_or_die(3, dry_run=True) != 'clean'

    # a stage whose module does not exist must fail loudly, not silently pass
    try:
        r = run_stage(8, skip_missing=True, verbose=False)
        print('\n[run_all] stage 8 with --skip-missing -> status=%s  detail=%s'
              % (r.status, r.detail[:110]))
        assert r.status in ('ok', 'missing')
    except SystemExit as exc:
        assert dirty_cache, 'stage 8 refused although the manifest is clean'
        print('\n[run_all] stage 8 REFUSED because the cache does not match '
              'MANIFEST.json -- spec Sec.5, "refuses to run on a mismatch":\n'
              '           %s' % str(exc).splitlines()[0])
    try:
        run_stage(3, skip_missing=False, verbose=False)
    except SystemExit as exc:
        print('[run_all] stage 3 WITHOUT --skip-missing exits: %s'
              % str(exc).splitlines()[0])
    else:
        raise AssertionError('a missing module did not stop the run')
    print('\n[run_all] SELF-CHECK PASSED')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        _selfcheck()
    else:
        sys.exit(main())
