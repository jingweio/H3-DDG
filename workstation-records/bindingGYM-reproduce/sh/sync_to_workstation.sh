#!/bin/bash
# Local -> workstation sync: transfer, stamp, then REPORT staleness (never delete).
#
# --delete was dropped deliberately (user, 2026-08-27). It removes anything the destination has
# and the source lacks -- which is exactly what a running job produces: checkpoints, OOF csvs,
# nohup logs. Re-syncing a code change mid-run would have destroyed them. Verified by dry-run:
#     deleting results/bindinggym_logitdiff_fold1/checkpoint/best_fold1.pt
#     deleting results/bindinggym_logitdiff_fold1/oof_fold1_ep10.csv
# An exclude list can protect those, but the list is the fragile part -- it was already wrong once
# on its first use, and it has to stay right for every future output path anyone adds.
#
# What --delete was for is still real: a file deleted locally lingers on the workstation, Python
# imports the stale module, and the run finishes with plausible wrong numbers. So step (3) still
# FINDS those files -- it just prints them instead of removing them. Detection without
# destruction: the silent failure becomes visible, and removing anything is a deliberate act.
set -euo pipefail

SRC="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/"
BRANCH="$(git -C "$SRC" rev-parse --abbrev-ref HEAD)"
[ "$BRANCH" = "reproduce" ] || { echo "FATAL: on branch '$BRANCH', expected 'reproduce' (skill §4-1)"; exit 1; }
DEST=guoj0f@10.67.24.41:/data/guoj0f/repos/H3-DDG/reproduce/

EXCL=(
  --exclude .git --exclude .claude/worktrees
  --exclude .synced_commit      # written by step (2); the source never has it
  --exclude bgym_official       # sibling project's vendored code (602 MB), unused by this arm
  --exclude __pycache__
)
# Paths the workstation legitimately owns. Only step (3) consults this -- nothing is ever deleted,
# so a missing entry here costs a spurious warning, not lost work.
REMOTE_OWNED='^(results/|workstation-records/.*\.(out|err)$|.*/__pycache__/)'

echo "=== (1) transfer (no --delete) ==="
rsync -a "${EXCL[@]}" "$SRC" "$DEST"

echo "=== (2) commit stamp ==="
git -C "$SRC" rev-parse HEAD > /tmp/.synced_commit
git -C "$SRC" status --porcelain >> /tmp/.synced_commit
head -1 /tmp/.synced_commit
D=$(tail -n +2 /tmp/.synced_commit | wc -l); [ "$D" -eq 0 ] && echo "  clean" || echo "  ⚠ $D uncommitted file(s)"
rsync -a /tmp/.synced_commit "$DEST"

echo "=== (3) staleness report (informational; nothing is deleted) ==="
STALE=$(rsync -avn --delete "${EXCL[@]}" "$SRC" "$DEST" \
        | sed -n 's/^deleting //p' | grep -vE "$REMOTE_OWNED" || true)
PEND=$(rsync -avn "${EXCL[@]}" "$SRC" "$DEST" \
       | grep -vE '^(sending|sent |total size|\./|deleting |$)' || true)
if [ -n "$PEND" ]; then echo "  ✗ still to transfer:"; echo "$PEND" | head -10; exit 1; fi
if [ -n "$STALE" ]; then
  echo "  ⚠ present on the workstation but NOT local -- stale code can be imported silently."
  echo "$STALE" | sed 's/^/      /'
  echo "  Review, then remove deliberately if they are indeed stale:"
  echo "      ssh guoj0f@10.67.24.41 'cd /data/guoj0f/repos/H3-DDG/reproduce && rm -i <path>'"
else
  echo "  ✓ no stale files; nothing pending"
fi
