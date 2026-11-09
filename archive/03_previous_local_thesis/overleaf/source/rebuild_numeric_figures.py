# Rebuild numerical graphics from repository-reported aggregates.
# Requires matplotlib, numpy. Run from the root of the unzipped project.
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
R = Path(__file__).resolve().parent.parent
(R/"images").mkdir(exist_ok=True)
colors=["#295c86","#38a6a1","#9c6b9e","#db943d","#7a8490","#73a65d"]
plt.rcParams.update({"font.size":9,"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"white","axes.labelcolor":"#263747","text.color":"#263747","axes.titleweight":"bold","savefig.facecolor":"white"})
prov=[]
def save(fig,name,origin):
    fig.tight_layout(pad=1.8)
    fig.savefig(R/"images"/name,dpi=260,bbox_inches="tight")
    plt.close(fig)
    prov.append((name,origin))
# fig04 cohort flow from register, not fitted observations
f,ax=plt.subplots(figsize=(10,4));ax.axis('off')
from matplotlib.patches import FancyBboxPatch
nodes=[(.02,.32,.19,.39,'Original EEG cohort\nNEMAR on004504\n88: 36 AD, 23 FTD, 29 CN'),(.30,.60,.28,.28,'Study A: temporal reliability\n64: 23 AD, 18 FTD, 23 CN'),(.30,.13,.28,.28,'Study B: CBraMod + DSRF\n65: 36 AD, 29 CN'),(.69,.60,.26,.28,'Nested, repeated subject CV\nAD vs CN and FTD vs CN'),(.69,.13,.26,.28,'52 development | 13 locked test\nAD vs CN only')]
for i,(x,y,w,h,t) in enumerate(nodes):
 patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015',linewidth=1.2,edgecolor=colors[0] if i!=2 else colors[2],facecolor='#eff5fa' if i!=2 else '#f5eff8');ax.add_patch(patch);ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=10,fontweight='medium',linespacing=1.7)
for a,b in [((.21,.515),(.30,.74)),((.21,.515),(.30,.27)),((.58,.74),(.69,.74)),((.58,.27),(.69,.27))]:ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':2,'color':colors[0]})
ax.text(.51,.03,'The 64-person and 65-person analytical cohorts were not reconciled by participant ID.',ha='center',va='center',fontsize=9,color='#934d3f')
save(f,'fig04_two_studies.png','PROJECT_HISTORY.md; thesis evidence brief; later execution package')
# fig05 SVM reliability: plot genuine percentages
f,ax=plt.subplots(figsize=(9.4,4.75));labs=['AD vs CN','FTD vs CN'];pos=np.arange(2);w=.25
for i,(k,vs) in enumerate([('First-minute BAcc',[73.48,73.72]),('Changed over 5 min',[44.35,33.66]),('Stable but wrong',[8.70,13.17])]):
 bars=ax.bar(pos+(i-1)*w,vs,w,label=k,color=colors[i]);ax.bar_label(bars,fmt='%.1f%%',padding=2,fontsize=8)
ax.set_ylim(0,90);ax.set_ylabel('Rate (%)');ax.set_xticks(pos,labs);ax.set_title('Temporal evaluation: classification, changes, and persistent errors',pad=51,fontsize=12);ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.18),frameon=False,fontsize=9)
save(f,'fig05_temporal_reliability.png','PROJECT_HISTORY.md Phase A; 64-person SVM retrospective track')
# fig07 independent MCI controls
f,ax=plt.subplots(figsize=(9.5,4.65));x=np.arange(2);w=.23
for i,(l,v) in enumerate([('BAcc',[40.78,55.83]),('Unstable',[5.60,20.00]),('Stable but wrong',[59.20,35.83])]):
 b=ax.bar(x+(i-1)*w,v,w,label=l,color=colors[i]);ax.bar_label(b,fmt='%.1f%%',padding=2,fontsize=8)
ax.axhline(50,ls='--',color='#444444',alpha=.55,lw=1,label='Constant-label BAcc');ax.set_ylim(0,78);ax.set_ylabel('%');ax.set_xticks(x,['Isfahan EEG_1\n25 recordings','Isfahan EEG_2\n24 recordings']);ax.set_title('Separate MCI task: stable predictions can remain wrong',pad=49,fontsize=12);ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,1.19),frameon=False,fontsize=8)
save(f,'fig07_mci_controls.png','PROJECT_HISTORY.md Phase B; independent MCI task, not external AD validation')
# fig08 development
models=['DSRF','XGBoost','EEGNet-inspired','Random Forest','High only','SVM','No skip','Low only'];v=[77.06,76.61,74.89,72.71,72.26,71.81,71.44,62.74]
f,ax=plt.subplots(figsize=(9.2,5));y=np.arange(len(models));ax.barh(y,v,color=[colors[0]]+['#8999aa']*7);ax.set_yticks(y,models);ax.invert_yaxis();ax.set_xlim(0,88);ax.set_xlabel('Pooled subject-level balanced accuracy (%)');ax.set_title('Five-fold development: eight compared variants')
for yy,vv in zip(y,v):ax.text(vv+.6,yy,f'{vv:.2f}%',va='center',fontsize=9)
save(f,'fig08_development_models.png','results/thesis_execution_package_2026_v1/thesis_results_ready.md; n=52')
# fig09 matched ablations
f,ax=plt.subplots(figsize=(8.8,3.7));x=['Residual (DSRF)','Gated blend','High only','Low only','No skip'];v=[77.06,73.16,72.26,62.74,71.44]
b=ax.bar(x,v,color=[colors[0],colors[2],'#96a1aa','#96a1aa','#96a1aa']);ax.bar_label(b,fmt='%.2f%%',padding=2);ax.set_ylim(0,92);ax.set_ylabel('Pooled balanced accuracy (%)');ax.set_title('Decoder components and skip-connection comparisons');ax.tick_params(axis='x',labelrotation=12)
save(f,'fig09_decoder_ablations.png','results/thesis_execution_package_2026_v1/thesis_results_ready.md; matched residual-versus-blend, additional non-matched component ablations')
# fig10 clean vs synthetic perturbation
f,ax=plt.subplots(figsize=(8.8,3.8));x=['Clean','10% noise','20% noise','One channel zeroed'];v=[77.06,71.87,70.07,76.78];b=ax.bar(x,v,color=[colors[0],colors[1],colors[2],colors[3]]);ax.bar_label(b,fmt='%.2f%%',padding=2);ax.set_ylim(0,92);ax.set_ylabel('Mean subject balanced accuracy (%)');ax.set_title('DSRF under five seeded synthetic perturbations');
save(f,'fig10_synthetic_robustness.png','results/thesis_execution_package_2026_v1/thesis_results_ready.md; same n=52 for five perturbation seeds')
# fig11 flip under perturbation
f,ax=plt.subplots(figsize=(8.8,3.8));x=['10% noise','20% noise','One channel zeroed'];v=[23.08,27.69,7.31];b=ax.bar(x,v,color=[colors[1],colors[2],colors[3]]);ax.bar_label(b,fmt='%.2f%%',padding=2);ax.set_ylim(0,36);ax.set_ylabel('Mean label-flip rate (%)');ax.set_title('Prediction changes alone do not establish correctness')
save(f,'fig11_perturbation_flips.png','results/thesis_execution_package_2026_v1/thesis_results_ready.md; five seeded repeats, 52 subjects')
# fig12 masking sensitivity actual values
f,ax=plt.subplots(figsize=(9.7,4));data={'AD':[('T5',11.03),('Pz',7.66),('F8',7.60),('Fz',6.82),('C3',5.09)],'CN':[('O2',31.50),('T6',16.01),('O1',15.90),('Fp2',14.96),('F3',12.52)]};xx=np.arange(5);w=.36
ax.bar(xx-w/2,[x[1] for x in data['AD']],w,label='AD',color=colors[0]);ax.bar(xx+w/2,[x[1] for x in data['CN']],w,label='CN',color=colors[2]);ax.set_xticks(xx,[f'{data["AD"][i][0]} / {data["CN"][i][0]}' for i in range(5)]);ax.set_ylim(0,37);ax.set_ylabel('Sensitivity reported (percentage points)');ax.set_title('Top five channel-occlusion sensitivities by class');ax.legend(frameon=False);ax.text(.5,-.25,'Paired tick labels show AD channel / CN channel, not homologous electrodes.',transform=ax.transAxes,ha='center',fontsize=8)
save(f,'fig12_channel_occlusion.png','PROJECT_HISTORY.md Phase E and thesis_results_ready.md; n=52 input-mask sensitivity only')
# fig13 final heldout uncertainty
f,ax=plt.subplots(figsize=(8.7,3.6));names=['DSRF','XGBoost'];v=[70.2381,77.38095];ci=[(46.4286,92.8571),(53.5714,100.0)];x=np.arange(2)
for i in range(2):ax.errorbar([i],[v[i]],yerr=[[v[i]-ci[i][0]],[ci[i][1]-v[i]]],fmt='o',markersize=9,capsize=6,lw=2,color=colors[i],label=names[i]);ax.text(i+.06,v[i]+1.5,f'{v[i]:.2f}%',fontsize=9)
ax.set_xlim(-.6,1.65);ax.set_ylim(30,109);ax.set_xticks(x,names);ax.set_ylabel('Balanced accuracy (%)');ax.set_title('Locked 13-person holdout with exploratory 95% bootstrap intervals');ax.axhline(50,ls='--',color='gray',alpha=.6)
save(f,'fig13_holdout_uncertainty.png','results/locked_final_test_2026_v1/final_metrics.json; n=13, paired difference -7.14 pp, interval includes zero')

print("Rebuilt",len(prov),"aggregate evidence figures")
