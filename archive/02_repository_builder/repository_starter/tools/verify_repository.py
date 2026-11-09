#!/usr/bin/env python3
"""Read-only integrity check for the EEGraft canonical research repository."""
from pathlib import Path
import sys, re, subprocess, json
ROOT=Path(__file__).resolve().parents[1]
errors=[]
checks=[
 ('source legacy README', 'archive/01-legacy-duration-study/README.md'),
 ('source legacy experiment registry', 'archive/01-legacy-duration-study/EXPERIMENT_REGISTRY.md'),
 ('source backup history', 'archive/02-research-backup/PROJECT_HISTORY.md'),
 ('source model results', 'archive/02-research-backup/source/alzheimers-thesis/eeg_foundation_thesis_starter/eeg_foundation_thesis/results/thesis_execution_package_2026_v1/thesis_results_ready.md'),
 ('source holdout metrics', 'archive/02-research-backup/source/alzheimers-thesis/eeg_foundation_thesis_starter/eeg_foundation_thesis/results/locked_final_test_2026_v1/final_metrics.json'),
 ('thesis main', 'thesis/overleaf/main.tex'),
 ('thesis bibliography', 'thesis/overleaf/bibliography/references.bib'),
]
for n,p in checks:
    good=(ROOT/p).is_file()
    print(('OK  ' if good else 'FAIL') + ' '+n+' '+p)
    if not good: errors.append(p)
for i in range(1,10):
    p=f'thesis/overleaf/chapters/chapter_{i}.tex'
    if not (ROOT/p).is_file(): errors.append(p)
images=sorted((ROOT/'thesis/overleaf/images').glob('fig*.png'))
print('Thesis PNG figures:',len(images),'(expected 13 for the current draft)')
if len(images)!=13: errors.append('figure count')
visible_signatures=[str(p.relative_to(ROOT)) for p in (ROOT/'thesis/overleaf').rglob('signature_*')]
if visible_signatures: errors.append('private signature images: '+', '.join(visible_signatures))
metrics=ROOT/'archive/02-research-backup/source/alzheimers-thesis/eeg_foundation_thesis_starter/eeg_foundation_thesis/results/locked_final_test_2026_v1/final_metrics.json'
if metrics.is_file():
    try:
        r=json.loads(metrics.read_text())
        dsrf=r['models']['DSRF']['balanced_accuracy']
        xgb=r['models']['XGBoost']['balanced_accuracy']
        assert abs(dsrf-.70238095238)<1e-9 and abs(xgb-.7738095238095)<1e-9
        print('OK  Locked holdout numbers match archived JSON')
    except Exception as e: errors.append('final metrics mismatch: '+str(e))
try:
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode('utf-8','replace').split('\0')
    tracked=[x for x in tracked if x]
    flagged=[p for p in tracked if re.search(r'(^|/)(signature_[^/]+|\.env|id_rsa|private_key|credentials\.json)(/|$)',p,re.I)]
    if flagged:
        print('WARNING: potentially sensitive tracked paths (review entire git history too):')
        for p in flagged[:25]:print('  -',p)
    print('Tracked files:',len(tracked))
except Exception: print('Note: not yet a Git repo; run build script first')
if errors:
    print('\nFAILED:',len(errors),'missing/mismatched checks')
    for e in errors: print(' -',e)
    sys.exit(1)
print('\nPASSED basic structural checks; NOT a complete security or scientific audit.')
