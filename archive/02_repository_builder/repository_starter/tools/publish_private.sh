#!/usr/bin/env bash
# Private-first GitHub publication; requires local gh CLI and appropriate account access.
set -Eeuo pipefail
if [[ $# -ne 1 || "$1" != */* ]]; then
  echo 'Usage: bash tools/publish_private.sh OWNER/EEGraft-Complete-Thesis' >&2
  exit 2
fi
REPO="$1"; OWNER="${REPO%%/*}"; NAME="${REPO#*/}"
case "$REPO" in *[!a-zA-Z0-9_./-]* ) echo 'Unsupported repo characters' >&2; exit 2;; esac
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ -d .git ]] || { echo 'Run prepare_eegraft_repository.sh first.' >&2; exit 2; }
command -v gh >/dev/null || { echo 'gh CLI not installed; install gh and run gh auth login.' >&2; exit 2; }
gh auth status >/dev/null || { echo 'Run gh auth login.' >&2; exit 2; }
LOCAL_USER="$(gh api user --jq .login)"
if [[ "$OWNER" != "$LOCAL_USER" ]]; then
  echo "Connected gh account: $LOCAL_USER. Requested owner: $OWNER." >&2
  echo 'Switch gh authentication to the intended owner. Org publication requires separate care.' >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo 'Uncommitted edits detected. Review and commit before publication.' >&2
  exit 2
fi
python3 tools/verify_repository.py
if gh repo view "$REPO" >/dev/null 2>&1; then
  echo "Target repository already exists: $REPO. Refusing to overwrite it." >&2
  exit 2
fi
cat <<EOF
Create a NEW PRIVATE GitHub repository $REPO and push the combined history?

WARNING: This preserves imported historical commits. The script cannot
certify that every old commit is free of secrets or personal information.
The original repositories will not be modified or deleted.
EOF
read -r -p 'Type CREATE PRIVATE to confirm: ' CONFIRM
[[ "$CONFIRM" == 'CREATE PRIVATE' ]] || { echo 'Canceled.'; exit 1; }
gh repo create "$REPO" --private --source=. --remote=origin --push

echo "Created private repository: https://github.com/$REPO"
echo 'If source backup contains LFS pointers, independently migrate permitted'
echo 'Git LFS binary objects after checking licensing and quota.'
