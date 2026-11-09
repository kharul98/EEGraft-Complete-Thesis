#!/usr/bin/env python3
"""EEGraft figure candidates: plots ONLY. Never edits the thesis project.

Values are transcribed from the numerical tables and figure-generation code
associated with the 66-page EEGraft thesis PDF (Nov 2026).
This script introduces NO observations, no interpolated measurements, no CI,
and no new model training. Only existing aggregate points are visualized.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw

HERE = Path(__file__).resolve().parent
OUT = HERE / 'output_candidates'
OUT.mkdir(parents=True, exist_ok=True)
NAVY = '#24476B'
BLUE = '#396E9B'
TEAL = '#318E8A'
PURPLE = '#86699C'
ORANGE = '#C48643'
GRAY = '#74818C'
GRID = '#DCE3E9'
INK = '#27343F'
PAPER = '#FFFFFF'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10.5,
    'axes.labelcolor': INK, 'axes.titlesize': 12,
    'axes.titleweight': 'bold', 'text.color': INK,
    'axes.edgecolor': GRID, 'axes.spines.top': False,
    'axes.spines.right': False, 'figure.facecolor': PAPER,
    'savefig.facecolor': PAPER, 'xtick.color': INK,
    'ytick.color': INK, 'svg.fonttype': 'none'
})
produced = []

def save(fig, basename, description):
    fig.savefig(OUT / (basename+'.png'), dpi=300, bbox_inches='tight', pad_inches=0.18)
    fig.savefig(OUT / (basename+'.svg'), bbox_inches='tight', pad_inches=0.18)
    produced.append(dict(figure=basename, description=description))
    plt.close(fig)

def xgrid(ax):
    ax.grid(axis='x', color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

# 6.1: A dot plot avoids heavy bars and emphasizes the near-tie on development.
models = ['DSRF', 'XGBoost', 'EEGNet-inspired', 'Random forest',
          'High-only', 'SVM', 'No-skip', 'Low-only']
values = [77.06, 76.61, 74.89, 72.71, 72.26, 71.81, 71.44, 62.74]
fig, ax = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
y = np.arange(len(models))
ax.scatter(values, y, s=85, zorder=4,
           c=[NAVY, TEAL] + [GRAY]*6,
           edgecolor=PAPER, linewidth=0.8)
for yy, value in zip(y, values):
    ax.text(value + 0.65, yy, f'{value:.2f}%', va='center', fontsize=10)
ax.set_yticks(y, models)
ax.invert_yaxis()
ax.set_xlim(50, 87)
ax.set_xlabel('Pooled subject-level balanced accuracy (%)')
ax.set_title('Development comparison: 52 participants', loc='left', pad=12)
ax.axvline(50, ls=':', color=GRAY, lw=1)
xgrid(ax)
fig.text(.012, -.018, 'Five subject-disjoint development folds; 0.45 pp separates DSRF and XGBoost.', fontsize=9, color=GRAY)
save(fig, 'candidate_fig6_1_ranked_dot', 'Replacement for Fig 6.1: ranked Cleveland dot plot')

# 6.2: Capacity-matched fusion rule is distinct from other ablations.
fig, axes = plt.subplots(1, 2, figsize=(11.1, 4.2), sharex=True, gridspec_kw={'width_ratios':[1.05,1]}, constrained_layout=True)
ax = axes[0]
labels = ['Additive residual (DSRF)', 'Gated blend']
vals = [77.06, 73.16]
ax.plot(sorted(vals), [0,0], '-', color=GRID, lw=3, zorder=1)
ax.scatter(vals, [0,0], c=[NAVY, PURPLE], s=100, zorder=3)
ax.set_yticks([])
ax.set_ylim(-1.5, 1.5); ax.set_xlim(59,85)
ax.set_title('A. Matched fusion rules', loc='left', pad=12)
ax.text(.5, .05, 'Observed gap: 3.90 pp (inconclusive)', transform=ax.transAxes, fontsize=9, color=GRAY, ha='center')
for v, label, height in zip(vals, labels, [.20, -.30]):
    ax.text(v, height, f'{v:.2f}%', ha='center', va='center', fontsize=9, weight='bold')
    ax.text(v, -.55 if height>0 else .55, label.replace(' (DSRF)',''), ha='center', va='center', fontsize=8.7)
xgrid(ax)
ax = axes[1]
labels = ['High-only', 'No-skip', 'Low-only']
vals = [72.26, 71.44, 62.74]
ax.scatter(vals, [0,1,2], color=BLUE, s=95, zorder=3)
ax.set_yticks([0,1,2], labels); ax.invert_yaxis()
ax.set_title('B. Other development ablations', loc='left', pad=12)
for yi, v in enumerate(vals):
    ax.text(v+.6, yi, f'{v:.2f}%', va='center', fontsize=9)
xgrid(ax)
fig.supxlabel('Pooled subject-level balanced accuracy (%)', fontsize=10)
save(fig, 'candidate_fig6_2_two_panel_dots', 'Replacement for Fig 6.2: matched and unmatched comparisons kept distinct')

# 7.1: Three mutually exclusive subject-repeat states form a 100% composition.
# Source: Appendix B state counts (n=230 / n=205). Correlated repeats, NOT unique patients.
groups = [
    ('AD vs CN', (108,102,20), 230, 73.48),
    ('FTD vs CN', (109,69,27), 205, 73.72)
]
fig, ax = plt.subplots(figsize=(9.2, 4.0), constrained_layout=True)
state_cols = [BLUE, TEAL, ORANGE]
state_labels = ['Stable and correct', 'Changed label', 'Stable but wrong']
for yy, (group, counts, denom, bacc) in enumerate(groups):
    assert sum(counts) == denom, 'Historical state counts must sum to documented repeats'
    left = 0
    for i, count in enumerate(counts):
        pct = count / denom * 100
        ax.barh(yy, pct, left=left, height=.56, color=state_cols[i], label=state_labels[i] if yy == 0 else None)
        col = PAPER if i < 2 else INK
        ax.text(left+pct/2, yy, f'{pct:.2f}%', color=col,
                ha='center', va='center', fontsize=10, fontweight='bold')
        left += pct
    ax.text(101, yy, f'{denom} repeated evaluations\nFirst-minute BAcc {bacc:.2f}%', va='center', fontsize=9, color=INK)
ax.set_xlim(0, 142)
ax.set_xticks([0,20,40,60,80,100])
ax.set_xlabel('Share of repeated participant evaluations (%)')
ax.set_yticks(np.arange(len(groups)), [x[0] for x in groups])
ax.invert_yaxis()
ax.set_title('Study A: stability and correctness are different outcomes', loc='left', pad=18)
ax.legend(ncol=3, bbox_to_anchor=(0.51, -0.29), loc='upper center', frameon=False, fontsize=9)
ax.grid(axis='x', color=GRID, zorder=0)
save(fig, 'candidate_fig7_1_state_composition', 'Replacement for Fig 7.1: 100% stacked outcome composition; note correlated repeats')

# 7.3: Dumbbell plot compares distinct MCI releases, without treating them as paired people.
metrics = ['First-minute BAcc', 'Changed label', 'Stable but wrong']
a = [40.78, 5.60, 59.20]
b = [55.83, 20.00, 35.83]
fig, ax = plt.subplots(figsize=(9.5, 4.6), constrained_layout=True)
for i, (x1, x2) in enumerate(zip(a,b)):
    ax.plot([x1,x2], [i,i], color='#BAC5CE', linewidth=2.4, zorder=2)
    ax.scatter(x1, i, c=BLUE, s=95, zorder=4)
    ax.scatter(x2, i, c=PURPLE, s=95, zorder=4)
    # Labels are offset vertically, rather than overlapping closely spaced dots.
    ax.annotate(f'{x1:.2f}%', (x1,i), xytext=(0,-17), textcoords='offset points', ha='center', fontsize=8.8, color=BLUE)
    ax.annotate(f'{x2:.2f}%', (x2,i), xytext=(0,9), textcoords='offset points', ha='center', fontsize=8.8, color=PURPLE)
ax.set_yticks(np.arange(3), metrics)
ax.invert_yaxis()
ax.set_xlim(-3,76)
ax.set_xlabel('Reported percentage (%)')
ax.set_title('Separate MCI recordings: accuracy versus persistent error', loc='left', pad=13)
ax.plot([], [], 'o', color=BLUE, label='EEG_1 (25 recordings)')
ax.plot([], [], 'o', color=PURPLE, label='EEG_2 (24 recordings)')
ax.legend(loc='lower center', bbox_to_anchor=(.5,-.38), frameon=False, ncol=2, fontsize=9)
xgrid(ax)
fig.text(.015, -.02, 'Different recording sets, not matched participants; not external AD validation.', fontsize=9, color=GRAY)
save(fig, 'candidate_fig7_3_mci_dumbbell', 'Replacement for Fig 7.3: connected dots compare separate MCI releases')

# 7.4: A continuous severity axis exists ONLY for 0%, 10%, 20% Gaussian noise.
fig, axes = plt.subplots(1,2,figsize=(10.0,4.2),sharey=True,gridspec_kw={'width_ratios':[1.55,.85]},constrained_layout=True)
ax=axes[0]
doses=[0,10,20]
acc=[77.06,71.87,70.07]
ax.plot(doses,acc,'-o',color=TEAL,lw=2.3,markersize=9)
for x0,y0 in zip(doses,acc):
    ax.annotate(f'{y0:.2f}%', (x0,y0), xytext=(0,11), textcoords='offset points', ha='center', fontsize=9)
ax.set_xticks(doses,['Clean\n0%','10%','20%'])
ax.set_ylim(63,83)
ax.set_xlabel('Gaussian noise (% of within-window signal SD)')
ax.set_ylabel('Mean balanced accuracy (%)')
ax.set_title('A. Noise severity', loc='left', pad=14)
ax.grid(axis='y',color=GRID)
ax=axes[1]
ax.axhline(77.06,color=GRAY,ls='--',lw=1.5,label='Clean 77.06%')
ax.scatter([0],[76.78],color=ORANGE,s=110,zorder=4)
ax.annotate('76.78%', (0,76.78), xytext=(0,13),textcoords='offset points',ha='center',fontsize=9)
ax.set_xlim(-.6,.6)
ax.set_xticks([0],['One channel\nzeroed'])
ax.set_title('B. Channel removal', loc='left', pad=14)
ax.grid(axis='y',color=GRID)
ax.legend(frameon=False,loc='lower center',fontsize=9)
save(fig, 'candidate_fig7_4_noise_line', 'Replacement for Fig 7.4: line over noise dose; channel zeroing separate')

# 7.5: Accuracy vs label changes under the same three synthetic perturbations.
conds=['10% Gaussian noise','20% Gaussian noise','One channel zeroed']
flips=[23.08,27.69,7.31]
acc=[71.87,70.07,76.78]
cols=[TEAL,PURPLE,ORANGE]
fig, ax=plt.subplots(figsize=(8.5,5.1),constrained_layout=True)
ax.scatter(flips,acc,s=125,c=cols,edgecolor=PAPER,lw=0.8,zorder=5)
offsets=[(-100,12),(-100,-24),(13,12)]
for i,(x0,y0) in enumerate(zip(flips,acc)):
    ax.annotate(f'{conds[i]}\n{y0:.2f}% BAcc, {x0:.2f}% changed', (x0,y0),
                xytext=offsets[i], textcoords='offset points',fontsize=9,
                arrowprops={'arrowstyle':'-', 'color':GRAY,'lw':.7})
ax.set_xlim(0,37)
ax.set_ylim(64.5,83)
ax.set_xlabel('Mean class-label change relative to clean predictions (%)')
ax.set_ylabel('Mean balanced accuracy under perturbation (%)')
ax.set_title('Study B: two distinct reliability dimensions',loc='left',pad=13)
ax.grid(color=GRID)
save(fig, 'candidate_fig7_5_accuracy_flip_scatter', 'Replacement for Fig 7.5: accuracy-versus-flip scatter; same 52 people')

# 7.6: Independent class rankings. NO false matching between T5 and O2, etc.
data={
 'AD': [('T5',11.03),('Pz',7.66),('F8',7.60),('Fz',6.82),('C3',5.09)],
 'CN': [('O2',31.50),('T6',16.01),('O1',15.90),('Fp2',14.96),('F3',12.52)]
}
fig, axes=plt.subplots(1,2,figsize=(11.0,4.6),sharex=True,constrained_layout=True)
for ax,(lab,rows),color in zip(axes,data.items(),[BLUE,PURPLE]):
    vals=[v for _,v in rows]
    names=[n for n,_ in rows]
    yy=np.arange(5)
    ax.hlines(yy,0,vals,color=GRID,lw=2.5)
    ax.scatter(vals,yy,color=color,s=85,zorder=4)
    for i,v in enumerate(vals):
        ax.text(v+.45,i,f'{v:.2f} pp',va='center',fontsize=9)
    ax.set_yticks(yy,names)
    ax.invert_yaxis()
    ax.set_xlim(0,38)
    ax.set_title(f'{lab} predictions: top five electrodes',loc='left',pad=13)
    xgrid(ax)
fig.supxlabel('Reported change under electrode masking (percentage points)')
save(fig, 'candidate_fig7_6_channel_rankings', 'Replacement for Fig 7.6: separate channel rankings; not biomarkers')

# Preview without needing a Linux browser: a readable contact sheet.
files=[OUT/(item['figure']+'.png') for item in produced]
thumb_w,thumb_h=920,550
from PIL import ImageFont
sheet=Image.new('RGB',(thumb_w*2, (thumb_h+40)*4), '#F2F5F7')
draw=ImageDraw.Draw(sheet)
for i, f in enumerate(files):
    img=Image.open(f).convert('RGB')
    img.thumbnail((thumb_w-30,thumb_h-30),Image.Resampling.LANCZOS)
    col,row=i%2,i//2
    x=col*thumb_w+(thumb_w-img.width)//2
    y=row*(thumb_h+40)+37
    sheet.paste(img,(x,y))
    draw.text((col*thumb_w+20,row*(thumb_h+40)+10),f.stem,fill=INK)
sheet.save(OUT/'CONTACT_SHEET.png',quality=94)
with open(OUT/'candidate_manifest.json','w',encoding='utf8') as f:
    json.dump({'source':'66-page EEGraft thesis PDF, reported aggregate values and existing numeric figure script',
               'note':'Candidate figures only; original thesis unchanged. No new subject-level data or experiments.',
               'candidates':produced,
               'retained_original_figures':['6.3 locked-test uncertainty','7.2 recording-duration line and interval chart']}, f,indent=2)
print(f'Created {len(produced)} candidate figures (PNG + SVG) in {OUT}')
print('Inspect CONTACT_SHEET.png and candidate_manifest.json before replacing anything.')
