# Git LFS checkpoint migration - separate manual step

The imported research backup may contain **Git LFS pointer files** rather than large checkpoint bytes. `git subtree add` preserves tracked Git files and history, not the actual remote LFS objects. A new GitHub repository can therefore show checkpoint filenames that are not fully downloadable.

Do **not** migrate third-party CBraMod pretrained weights or private recordings without permission. Confirm licenses and storage quotas first.

Once you have appropriate authorization and have created the new private repository, and have installed `git-lfs`:

```bash
cd "$HOME/EEGraft-Complete-Thesis"
git lfs install
git lfs ls-files
# Fetch required LFS objects from the original research backup:
git lfs fetch --all historical-backup
# After inspecting the objects and licensing, copy all LFS objects to the new private GitHub remote:
git lfs push --all origin
# Recheck LFS object availability from a fresh clone before declaring backup completeness.
```

This is deliberately **not** run by the builder. The repository remains a useful code, data-manifest, configuration, and results archive even if external LFS checkpoints are not migrated.
