#!/usr/bin/env bash
# EEGraft canonical repository builder.
# Imports two unmodified public Git histories into separate subtrees.
# Creates a local repository ONLY; does not create/push a GitHub repo.
set -Eeuo pipefail

usage() {
cat <<'EOF'
Usage:
  bash prepare_eegraft_repository.sh [OUTPUT_DIRECTORY] [--legacy-url URL] [--backup-url URL]

Defaults:
  OUTPUT_DIRECTORY = ./EEGraft-Complete-Thesis
  Legacy = https://github.com/kharul98/alzheimers-eeg-duration-thesis.git
  Backup = https://github.com/kharul98/eeg-dementia-thesis-backup-20261107T215131Z.git

This script:
  1. Creates a new local Git repository from the included safe starter.
  2. Imports ALL tracked files and Git commit history from both repos,
     separately, with non-squashed git-subtree merges.
  3. Includes the revised editable 9-chapter LaTeX thesis without student
     signature images, signed PDFs or TeX build artifacts.
  4. Writes import hashes, then runs structural verification.

It does NOT upload to GitHub. Keep the destination PRIVATE until all
current files AND historical commit contents have been reviewed.
EOF
}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$PWD/EEGraft-Complete-Thesis"
LEGACY_URL='https://github.com/kharul98/alzheimers-eeg-duration-thesis.git'
BACKUP_URL='https://github.com/kharul98/eeg-dementia-thesis-backup-20261107T215131Z.git'
if [[ $# -gt 0 && "$1" != --* ]]; then DEST="$1"; shift; fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --legacy-url) [[ $# -gt 1 ]] || { echo 'Missing legacy URL' >&2; exit 2; }; LEGACY_URL="$2"; shift 2;;
    --backup-url) [[ $# -gt 1 ]] || { echo 'Missing backup URL' >&2; exit 2; }; BACKUP_URL="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done

for cmd in git python3 cp; do command -v "$cmd" >/dev/null || { echo "Missing required command: $cmd" >&2; exit 2; }; done
SUBTREE_HELP="$(git subtree 2>&1 || true)"
if [[ "$SUBTREE_HELP" != *'usage: git subtree add'* ]]; then
    echo 'git subtree is not installed; install git subtree / contrib tools.' >&2
    exit 2
fi
if ! git var GIT_AUTHOR_IDENT >/dev/null 2>&1; then
    echo 'Configure Git name and email before running:' >&2
    echo '  git config --global user.name "Your Name"' >&2
    echo '  git config --global user.email "your-email@example.com"' >&2
    exit 2
fi
# Refuse an existing output to avoid changing prior research or overwriting local changes.
if [[ -e "$DEST" ]]; then echo "Refusing to overwrite existing directory: $DEST" >&2; exit 2; fi
mkdir -p "$DEST"
cd "$DEST"
git init -q -b main
cp -a "$ROOT/repository_starter/." "$DEST/"
export GIT_LFS_SKIP_SMUDGE=1
export GIT_TERMINAL_PROMPT=0

git add -A
git commit -q -m 'docs: establish canonical EEGraft thesis narrative and sanitized Overleaf source'
echo '[1/4] Starter committed, including the edited thesis source.'

git remote add legacy-duration "$LEGACY_URL"
git fetch --no-tags -q legacy-duration main
git subtree add --prefix=archive/01-legacy-duration-study legacy-duration main --no-squash -m 'history: import earlier recording-duration repository without squashing'
echo '[2/4] Earlier duration research and Git history imported.'

git remote add historical-backup "$BACKUP_URL"
git fetch --no-tags -q historical-backup main
git subtree add --prefix=archive/02-research-backup historical-backup main --no-squash -m 'history: import later research-backup source tree without squashing'
echo '[3/4] Research backup and Git history imported.'

OLD_SHA="$(git rev-parse refs/remotes/legacy-duration/main)"
BACKUP_SHA="$(git rev-parse refs/remotes/historical-backup/main)"
OLD_COUNT="$(git rev-list --count refs/remotes/legacy-duration/main)"
BACKUP_COUNT="$(git rev-list --count refs/remotes/historical-backup/main)"
cat > docs/SOURCE_IMPORT_RECEIPTS.md <<EOF
# Reproducible source-import receipts

These commits are the exact source heads captured **at build time**, not invented release pins.

| Source | Original remote | Imported branch | Imported commit | Upstream commit count |
|---|---|---|---|---:|
| Legacy duration study | $LEGACY_URL | main | \`$OLD_SHA\` | $OLD_COUNT |
| Later research backup | $BACKUP_URL | main | \`$BACKUP_SHA\` | $BACKUP_COUNT |

Import method: \`git subtree add --no-squash\` into separate \`archive/\` paths.

The original histories remain visible in this repository's commit ancestry;
folder prefixes are navigation labels rather than rewritten experimental dates.

**Do not automatically push LFS checkpoint pointers as evidence that binary
checkpoints were migrated.** A separate, authorized LFS migration may be
required; see \`docs/REPRODUCIBILITY.md\`.
EOF
python3 tools/verify_repository.py | tee docs/BUILD_VERIFICATION.txt
git add docs/SOURCE_IMPORT_RECEIPTS.md docs/BUILD_VERIFICATION.txt
git commit -q -m 'audit: record imported source commit IDs and verification results'
echo '[4/4] Build verified. See docs/BUILD_VERIFICATION.txt.'
echo
echo 'LOCAL CANONICAL REPOSITORY CREATED:'
echo "  $DEST"
echo 'Original source repositories were NOT modified.'
echo 'No GitHub remote push was performed.'
echo
echo 'NEXT: Review docs/RELEASE_CHECKLIST.md, then publish privately with:'
echo '  bash tools/publish_private.sh kharul98/EEGraft-Complete-Thesis'
