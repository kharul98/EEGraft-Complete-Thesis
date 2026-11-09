# EEGraft canonical repository - run this on your Ubuntu PC

This builder prepares ONE local Git repository containing both public upstream research projects **with their complete Git histories** and the new corrected Overleaf thesis, organized by numbered research-stage guides. It does not change, delete or push to the old source repositories.

### Step 1: install Git and GitHub CLI

```bash
sudo apt update
sudo apt install -y git git-lfs gh python3 unzip
```

`git lfs` is for later authorized checkpoint access; the script deliberately avoids automatically downloading pretrained weights.

### Step 2: check Git identity and GitHub credentials

```bash
git config --global user.name "Your GitHub Display Name"
git config --global user.email "your-github-email@example.com"
gh auth login
```

Be sure the `gh` account owns the repository you intend to create. The connected ChatGPT GitHub plugin in this chat has READ-ONLY access to the two `kharul98` repositories, so creation must be done from your authorized Ubuntu account.

### Step 3: extract builder ZIP and create the complete local repository

```bash
unzip EEGraft_Canonical_Repository_Builder.zip
cd EEGraft_Canonical_Repository_Builder
bash prepare_eegraft_repository.sh "$HOME/EEGraft-Complete-Thesis"
```

This fetches both real Git repos from GitHub, merges them into `archive/01-legacy-duration-study` and `archive/02-research-backup`, adds `thesis/overleaf/` and the numbered stage guides, and records upstream head SHAs in `docs/SOURCE_IMPORT_RECEIPTS.md`.

### Step 4: verify before uploading

```bash
cd "$HOME/EEGraft-Complete-Thesis"
python3 tools/verify_repository.py
cat docs/SOURCE_IMPORT_RECEIPTS.md
cat docs/RELEASE_CHECKLIST.md
git log --graph --oneline --all --max-count=30
```

Because non-squashed history imports past commits as well as current files, **the new repository must stay private until the entire imported Git history is screened for personal data, credentials and redistribution licenses.** Never publish student signatures. The bundled report source intentionally excludes them.

### Step 5: create and publish the private GitHub repository

If your `gh` CLI is logged into the `kharul98` account:

```bash
bash tools/publish_private.sh kharul98/EEGraft-Complete-Thesis
```

If you are logged in under a different account, use that authorized account's login as owner instead; the helper rejects accidental cross-account publication. It prompts you to confirm by typing `CREATE PRIVATE`.

Alternatively, create a new **Private** repository on github.com yourself (no README or license), then:

```bash
git remote add origin git@github.com:kharul98/EEGraft-Complete-Thesis.git
git push -u origin main
```

Replace the remote owner if your account differs. This alternative requires correctly configured GitHub SSH access.

### Important limitations

- **The two upstream sources are fetched on Ubuntu.** They are not bundled into this ZIP because this execution environment has no direct network access to clone them and GitHub connector has read-only access.
- All upstream tracked histories are imported unmodified, not rewritten into a fabricated linear experimental history. The numbered `phases/` are the chronological reading guide.
- The old backup may use Git LFS; pointer references can be imported without copying large checkpoint objects. Migrating actual LFS content is a separate licensed/authorized step.
- The Overleaf source contains the current 13 report figures, but this is not a proven exact recovery of the earlier approved image collection.
- This builder does not change scientific results or re-run any experiments.
