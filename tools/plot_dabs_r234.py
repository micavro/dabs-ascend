#!/usr/bin/env python3
"""Generate PDF figures from REAL accepted GPU results only.

Never performs any GPU execution and never fills missing points with predictions.
Figures are explicitly marked 'PARTIAL' until the full 96/96 suite completed.
"""
from __future__ import annotations
import argparse,csv,math,json
from pathlib import Path

METHODS=('RR','SLL','Ayo','Ours')
COLORS={'RR':'#64748b','SLL':'#bf7b45','Ayo':'#659f8e','Ours':'#325a88'}
MARKERS={'RR':'o','SLL':'s','Ayo':'D','Ours':'^'}


def read_csv(p):
    if not p.exists():raise ValueError('missing derived GPU table: '+str(p))
    return list(csv.DictReader(p.open(encoding='utf-8',newline='')))

def val(obj,k):
    try:
        v=float(obj[k]);return v if math.isfinite(v) else None
    except (TypeError,ValueError,KeyError):return None

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path)
    args=p.parse_args();out=args.out.resolve();ana=out/'analysis'
    rec=json.loads((ana/'extended_receipt.json').read_text())
    grid=read_csv(ana/'metric_grid.csv');raw=read_csv(ana/'latency_decomposition_by_condition.csv')
    kv=read_csv(ana/'kv_by_condition.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],
        'font.size':10.5,'axes.labelsize':11,'axes.titlesize':11,'xtick.labelsize':9.5,
        'ytick.labelsize':9.5,'axes.linewidth':1.05,'lines.linewidth':2.0,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.bbox':'tight'})
    plotdir=ana/'figures_pdf';plotdir.mkdir(exist_ok=True)
    note='Complete 96/96' if rec['complete'] else f"PARTIAL: {rec['accepted']}/96 accepted; do not cite as final result"

    def setax(ax,R):
        ax.set_title(f'{R} GPU replica'+('s' if R>1 else ''),loc='left',pad=5)
        for side in ('left','right','top','bottom'):ax.spines[side].set_visible(True)
        ax.grid(axis='y',alpha=.18,lw=.5);ax.set_axisbelow(True)
        ax.set_xticks([4,8,12,16]);ax.set_xlim(3.1,16.9)
    def save(fig,name):
        fig.text(.5,.012,note,ha='center',va='center',fontsize=8,color='#555555')
        fig.savefig(plotdir/name,format='pdf',bbox_inches='tight',pad_inches=.06)
        plt.close(fig)
    metric_specs=[('mean_jct_ms','Mean workflow JCT (ms)','Fig1_MeanJCT.pdf'),
        ('p95_jct_ms','P95 workflow JCT (ms)','Fig2_P95JCT.pdf'),
        ('mean_request_ttft_ms','Mean request TTFT (ms)','Fig3_TTFT.pdf'),
        ('cohort_workflows_s','Finite cohort throughput (WF/s)','Fig4_CohortThroughput.pdf'),
        ('decision_wall_mean_ms','Controller decision wall time (ms)','Fig5_DecisionOverhead.pdf'),
        ('native_cache_skipped_fraction','Native skipped-input fraction','Fig6_NativeKVReuse.pdf')]
    for metric,ylabel,filename in metric_specs:
        fig,axs=plt.subplots(1,3,figsize=(14,4.5),sharex=True,layout='constrained')
        for R,ax in zip((2,3,4),axs.ravel()):
            setax(ax,R)
            for method in METHODS:
                pts=sorted([(val(r,'rate'),val(r,metric)) for r in grid
                     if int(r['replicas'])==R and r['method']==method and val(r,metric) is not None])
                if not pts:continue
                xs,ys=zip(*pts)
                ax.plot(xs,ys,color=COLORS[method],marker=MARKERS[method],
                        markersize=5,label=method)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.set_xlabel('Workflow arrival rate (WF/s)')
            ax.set_ylabel(ylabel)
        handles,labels=axs.ravel()[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=4,bbox_to_anchor=(.5,1.045),
                   frameon=False,fontsize=10)
        save(fig,filename)

    # Native-sourced workflow critical path decomposition. Each R panel is
    # the highest arrival-rate condition actually accepted, never imputed.
    comps=(('central_queue_and_admission_ms','Queue+admission','#8f9ba7'),
           ('header_wait_ms','Header wait','#c29b6b'),
           ('first_content_wait_ms','First content','#6c93b5'),
           ('content_receive_ms','Content receive','#688e83'),
           ('other','Other','#b0a6a2'))
    fig,axs=plt.subplots(1,3,figsize=(14,4.7),layout='constrained')
    for R,ax in zip((2,3,4),axs.ravel()):
        candidates=[r for r in grid if int(r['replicas'])==R]
        highest=max((int(float(r['rate'])) for r in candidates),default=None)
        ax.set_title(f'R={R}, lambda={highest if highest is not None else "NA"}',loc='left')
        xloc=range(4);bottom=[0.]*4
        for col,name,color in comps:
            ys=[]
            for m in METHODS:
                match=next((r for r in candidates if r['method']==m and int(float(r['rate']))==highest),None)
                if match is None:ys.append(0);continue
                if col=='other':
                    z=sum(val(match,k+'_ms') or 0 for k in (
                         'dependency_release','input_prepare','postprocess','response_tail'))
                else:z=val(match,col) or 0
                ys.append(z)
            ax.bar(list(xloc),ys,bottom=bottom,color=color,label=name,width=.72,linewidth=.3,edgecolor='white')
            bottom=[a+b for a,b in zip(bottom,ys)]
        ax.set_xticks(list(xloc),METHODS,rotation=0,fontsize=9)
        ax.set_ylabel('Critical-path component (ms)')
        ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
        for side in ('left','right','top','bottom'):ax.spines[side].set_visible(True)
    fig.legend(*axs.ravel()[0].get_legend_handles_labels(),loc='upper center',
               bbox_to_anchor=(.5,1.045),ncol=5,frameon=False,fontsize=8.7)
    save(fig,'Fig7_CriticalPathDecomposition.pdf')

    categories=(('intra_only_eligible_tokens','Within-only eligible','#507d9f'),
                ('inter_only_eligible_tokens','Cross-only eligible','#b47b52'),
                ('both_eligible_tokens','Both eligible / ambiguous','#8ca18e'),
                ('unattributed_tokens','Unattributed','#aaaaaa'))
    fig,axs=plt.subplots(1,3,figsize=(14,5),layout='constrained')
    for R,ax in zip((2,3,4),axs.ravel()):
        available=[r for r in grid if int(r['replicas'])==R]
        highest=max((int(float(r['rate'])) for r in available),default=None)
        ax.set_title(f'R={R}, lambda={highest if highest is not None else "NA"}',loc='left')
        bottoms=[0.]*4
        for key,label,color in categories:
            vals=[]
            for method in METHODS:
                item=next((r for r in available if int(float(r['rate']))==highest and r['method']==method),None)
                saved=(val(item,'native_skip_tokens') if item is not None else None)
                parts=val(item,key) if item is not None else None
                vals.append(parts/saved*100 if saved and parts is not None else 0)
            ax.bar(list(range(4)),vals,bottom=bottoms,color=color,label=label,width=.72,linewidth=.3,edgecolor='white')
            bottoms=[a+b for a,b in zip(bottoms,vals)]
        ax.set_xticks(range(4),METHODS,fontsize=9);ax.set_ylabel('Fraction of native saved tokens (%)')
        ax.set_ylim(0,103);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        for side in ('left','right','top','bottom'):ax.spines[side].set_visible(True)
    fig.legend(*axs.ravel()[0].get_legend_handles_labels(),loc='upper center',
              bbox_to_anchor=(.5,1.045),ncol=4,fontsize=8.7,frameon=False)
    fig.text(.5,.035,'Source attribution is based on causal prefix eligibility, NOT GPU block writer ownership.',
             ha='center',fontsize=8.5)
    save(fig,'Fig8_KVEligibleSources.pdf')
    print('FULLGRID_PDF_FIGURES='+str(plotdir))
    print('PLOTS_STATUS='+('complete' if rec['complete'] else 'partial'))

if __name__=='__main__':main()
