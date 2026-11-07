# Pretrained weights are not bundled

After verifying the model publisher and license on the research PC, run:

```
python scripts/fetch_weights.py
```

The script downloads the actual `weighting666/CBraMod` / `pretrained_weights.pth` to `checkpoints/cbramod_official.pth` and validates the strict state dictionary. The current distribution does not contain the pretrained checkpoint.
