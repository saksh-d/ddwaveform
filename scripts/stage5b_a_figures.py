"""Explain the frozen Stage5B-A runs; reads artifacts only, no new fitting."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage5b_a'
FIG=OUT/'figures'
COLORS={'A':'#b25239','B':'#287ba6','C':'#35825b'}
LABELS={'A':'A: polar + Adam/L-BFGS','B':'B: Cartesian + Adam/L-BFGS','C':'C: Cartesian + L-BFGS'}

def main():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.labelsize':18,
        'xtick.labelsize':15,'ytick.labelsize':15,'legend.fontsize':16})
    rows=json.loads((OUT/'AB/results.json').read_text())+json.loads((OUT/'C/results.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(16,5.7),sharey=True)
    for ax,cat,title in zip(axes,['0.001','0.03','cold'],['Near truth: perturbation 0.001','Near truth: perturbation 0.03','Cold: target-only initialization']):
        for arm,offset in zip('ABC',[-.18,0,.18]):
            for p,x in zip([1,2,4],[0,1,2]):
                rs=sorted([r for r in rows if r['arm']==arm and r['category']==cat and r['passes']==p],key=lambda r:r['key'])
                jitter=np.linspace(-.05,.05,len(rs))
                ax.scatter(x+offset+jitter,[r['error'] for r in rs],s=45,color=COLORS[arm],marker={'A':'o','B':'s','C':'^'}[arm],alpha=.85,label=LABELS[arm] if p==1 and ax is axes[0] else None)
        ax.axhline(1e-4,color='black',ls='--',lw=1.3)
        ax.set(yscale='log',xticks=[0,1,2],xticklabels=['1','2','4'],ylim=(1e-11,.12))
        ax.set_title(title,fontsize=17,pad=14)
        ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('Best relative Frobenius error')
    fig.supxlabel('Passes (6 target cases per depth in each panel)',fontsize=18,y=.13)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,0),frameon=False)
    fig.subplots_adjust(left=.09,right=.99,top=.88,bottom=.24,wspace=.12)
    fig.savefig(FIG/'recovery.png',dpi=300);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.9),sharey=True)
    for ax,cat,title in zip(axes,['0.001','cold'],['Near truth: perturbation 0.001','Cold start']):
        for arm in 'ABC':
            base='AB' if arm!='C' else 'C'
            tr=json.loads((OUT/base/f'ascending_p4_s0_{cat}_{arm}_trace.json').read_text())
            x=np.arange(1,len(tr)+1)
            ax.plot(x,[t['best_loss']**.5 for t in tr],color=COLORS[arm],lw=2.1,label=LABELS[arm])
            # First 12 evaluated trial losses expose the Adam excursion without hiding selection.
            ax.scatter(x[:12],[t['loss']**.5 for t in tr[:12]],color=COLORS[arm],s=16,alpha=.45)
        ax.axhline(1e-4,color='black',ls='--',lw=1.3)
        ax.set(yscale='log',ylim=(1e-6,1.3),xlim=(0,510),title=None)
        ax.set_title(title,fontsize=17,pad=14)
        ax.grid(alpha=.18)
    axes[0].set_ylabel('Best-so-far relative error')
    fig.supxlabel('Objective and gradient evaluations',fontsize=18,y=.13)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,bbox_to_anchor=(.5,0),frameon=False)
    fig.subplots_adjust(left=.09,right=.99,top=.88,bottom=.24,wspace=.10)
    fig.savefig(FIG/'traces.png',dpi=300);plt.close(fig)
    geometry=json.loads((OUT/'geometry.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(13,5.6),sharey=True)
    for ax,mode in zip(axes,['random','identity']):
        for arm in 'AB':
            g=next(r['geometry'] for r in geometry if r['label']==f'N8_p1_{mode}_{arm}')
            vals=np.maximum(g['singular_values'],1e-17)
            ax.semilogy(np.arange(1,len(vals)+1),vals,color=COLORS[arm],marker='o',ms=4,lw=2,label=f'{arm}: rank {g["rank"]}/32')
        ax.axhline(1e-7,color='black',ls='--',lw=1.2)
        ax.set_title('Generic gates' if mode=='random' else 'Identity mixers',fontsize=17,pad=14)
        ax.set(ylim=(1e-17,5),xticks=[1,8,16,24,32])
        ax.grid(alpha=.18);ax.legend(loc='lower left',frameon=False)
    axes[0].set_ylabel('Jacobian singular value')
    fig.supxlabel('Singular-value index (N=8, one pass)',fontsize=18,y=.03)
    fig.subplots_adjust(left=.10,right=.99,top=.87,bottom=.18,wspace=.10)
    fig.savefig(FIG/'conditioning.png',dpi=300);plt.close(fig)
    from PIL import Image
    metadata={p.name:dict(pixels=Image.open(p).size,dpi=Image.open(p).info['dpi'],sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in FIG.glob('*.png')}
    (FIG/'manifest.json').write_text(json.dumps(dict(style='DejaVu Sans; labels18, ticks15, legends16; 300DPI; no font overrides',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),figures=metadata),indent=2)+'\n')
    print(metadata)

if __name__=='__main__':main()
