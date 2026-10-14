#!/usr/bin/env python3
"""Rebuild the report's eight core experimental figures from verified numeric CSVs.

- White figure/axes backgrounds, no titles, legend inside figure, black 4-side axes.
- Kai-style Chinese with serif English/numerals where available (no fonts bundled).
- Evidence truth: RR in source is *not* Preble. DAG in source is Ours.
- No data invented, no GPU calls.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'evidence/derived'
COL={'RR':'#979B9D','P2C':'#D4A576','Affinity':'#81949A','Ours':'#B45462',
     'light_red':'#CE8D84','gold':'#D9C170','dark':'#323232'}


def configure():
    # Optional fonts are searched by name; do not redistribute any fonts.
    candidates=['/usr/share/fonts/truetype/arphic-gkai00mp/gkai00mp.ttf',
                '/usr/share/fonts/truetype/arphic-bkai00mp/bkai00mp.ttf']
    cn=None
    for p in candidates:
        if Path(p).is_file():
            font_manager.fontManager.addfont(p)
            cn=font_manager.FontProperties(fname=p).get_name()
            break
    if cn is None:
        names={f.name for f in font_manager.fontManager.ttflist}
        for q in ['KaiTi','STKaiti','Kaiti SC','FangSong','Noto Serif CJK SC','SimSun']:
            if q in names:cn=q;break
    latin=None
    for name in ['Times New Roman','Nimbus Roman','Liberation Serif','DejaVu Serif']:
        if name in {f.name for f in font_manager.fontManager.ttflist}:
            latin=name;break
    plt.rcParams.update({'font.family':[cn,latin or 'DejaVu Serif'] if cn else [latin or 'DejaVu Serif'],
       'axes.unicode_minus':False,'font.size':11.5,'axes.labelsize':12,'xtick.labelsize':11,'ytick.labelsize':11,
       'legend.fontsize':10,'figure.facecolor':'white','axes.facecolor':'white',
       'savefig.facecolor':'white','pdf.fonttype':42,'ps.fonttype':42,'figure.dpi':170,'savefig.dpi':300})
    return cn,latin


def format_ax(ax, grid=True):
    ax.set_axisbelow(True)
    ax.set_facecolor('white')
    if grid:ax.grid(axis='y',alpha=.24,linewidth=.65,linestyle='--',color='#999999')
    for s in ax.spines.values():s.set_color('black');s.set_linewidth(.95);s.set_visible(True)
    ax.tick_params(direction='out',width=.85,length=4)


def save(fig, output:Path, name:str):
    for ext in ['png','pdf']:
        fig.savefig(output/f'{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white',pad_inches=.05)
    plt.close(fig)


def annotate_bar(ax,bars,fmt='{:.1f}',offset=None):
    for b in bars:
        val=b.get_height(); dx=b.get_x()+b.get_width()/2
        dy=(0.012*ax.get_ylim()[1] if offset is None else offset)
        ax.text(dx,val+dy,fmt.format(val),ha='center',va='bottom',fontsize=10)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'outputs'/'figures_v1')
    a=parser.parse_args();out=a.out.resolve()
    if out.exists() and any(out.iterdir()): parser.error('output exists and is not empty; use another --out')
    out.mkdir(parents=True,exist_ok=True)
    cn,latin=configure()
    methods=['RR','P2C','Affinity','Ours']
    cs=pd.read_csv(DATA/'case_confirm_rows.csv').replace({'method':{'DAG':'Ours'}})
    cp=pd.read_csv(DATA/'case_confirm_pair_gains.csv')
    asb=pd.read_csv(DATA/'ablation_byblock.csv')
    raw=pd.read_csv(DATA/'pilot_mechanism_raw.csv')
    prefix=pd.read_csv(DATA/'prefix_structure.csv')
    host=pd.read_csv(DATA/'host_phase_b3.csv')
    avg=cs.groupby('method').mean(numeric_only=True).loc[methods]
    # fig1
    fig,ax=plt.subplots(figsize=(7.3,4.5));fig.patch.set_facecolor('white')
    bars=ax.bar(methods,avg.mean_jct_ms,edgecolor='black',linewidth=.75,color=[COL[m] for m in methods],width=.68)
    ax.set_ylabel('平均工作流 E2E JCT (ms)');ax.set_xlabel('方法 (Method)');ax.set_ylim(0,3250)
    format_ax(ax);annotate_bar(ax,bars)
    for i,(baseline,gain) in enumerate(cp.groupby('reference')['mean_jct_ms_gain_pct'].mean().reindex(['RR','P2C','Affinity']).items()):
        ax.text(i, avg.loc[baseline,'mean_jct_ms']*.81,f'Ours ↓{gain:.2f}%',ha='center',va='center',fontsize=9.7,
                bbox={'facecolor':'white','edgecolor':COL['Ours'],'boxstyle':'round,pad=.3','linewidth':.8})
    save(fig,out,'F01_E2E_JCT')
    # fig2
    fig,ax=plt.subplots(figsize=(7.3,4.5));bars=ax.bar(methods,avg.cohort_workflows_s,edgecolor='black',linewidth=.75,color=[COL[m] for m in methods],width=.68)
    ax.set_ylabel('工作流完成速率 (workflows/s)');ax.set_xlabel('方法 (Method)');ax.set_ylim(0,1.72)
    format_ax(ax);annotate_bar(ax,bars,fmt='{:.3f}')
    for i,baseline in enumerate(['RR','P2C','Affinity']):
        changes=cp[cp.reference==baseline]['cohort_workflows_s_gain_pct'].mean()
        ax.text(i, avg.loc[baseline,'cohort_workflows_s']*.83,f'Ours +{changes:.2f}%',ha='center',va='center',fontsize=9.2,
                bbox={'facecolor':'white','edgecolor':COL['gold'],'boxstyle':'round,pad=.3','linewidth':.8})
    save(fig,out,'F02_Workflow_Throughput')
    # fig3
    fig,ax=plt.subplots(figsize=(7.3,4.4))
    group=[cs[cs.method==m].mean_jct_ms.to_numpy() for m in methods]
    bx=ax.boxplot(group,patch_artist=True,showmeans=True,tick_labels=methods,widths=.52,
             meanprops={'marker':'D','markerfacecolor':'white','markeredgecolor':'black','markersize':5},
             medianprops={'color':'black','linewidth':1.3})
    for patch,m in zip(bx['boxes'],methods):patch.set_facecolor(COL[m]);patch.set_edgecolor('black');patch.set_alpha(.84)
    ax.set_ylabel('八轮条件级平均 JCT (ms)');ax.set_xlabel('方法 (Method)')
    format_ax(ax);save(fig,out,'F03_E2E_JCT_8repeat_Box')
    # fig4
    g=cp.groupby('reference').mean(numeric_only=True).loc[['RR','P2C','Affinity']]
    fig,ax=plt.subplots(figsize=(7.4,4.6));x=np.arange(3);w=.23
    cols=['mean_jct_ms_gain_pct','p95_jct_ms_gain_pct','mean_request_ttft_ms_gain_pct']
    names=['E2E JCT','P95 JCT','TTFT'];palette=[COL['Ours'],COL['light_red'],COL['gold']]
    for i,(k,lab,c) in enumerate(zip(cols,names,palette)):
        bars=ax.bar(x+(i-1)*w,g[k],width=w,label=lab,color=c,edgecolor='black',linewidth=.6)
        if i==0:annotate_bar(ax,bars,fmt='{:.2f}%',offset=.34)
    ax.set_xticks(x,['RR','P2C','Affinity']);ax.set_xlabel('真实实验基线 (Baseline)');ax.set_ylabel('Ours 时延下降 (%)')
    ax.axhline(0,color='black',linewidth=.75);ax.set_ylim(0,28)
    ax.legend(loc='upper right',edgecolor='#555555',facecolor='white',ncol=1)
    format_ax(ax);save(fig,out,'F04_Latency_Improvement')
    # fig5
    fig,ax=plt.subplots(figsize=(7.2,4.4));asb=asb.sort_values('block')
    bars=ax.bar([f'B{x}' for x in asb.block],asb.mean_jct_ms_gain_pct,color=COL['Ours'],edgecolor='black',linewidth=.7,width=.58)
    ax.axhline(10,color=COL['dark'],linestyle='--',linewidth=1,label='10% 参考线')
    ax.set_ylabel('Ours 相对 Affinity 的 JCT 降幅 (%)');ax.set_xlabel('工作流数据组');ax.set_ylim(0,24.5)
    for bar,win in zip(bars,asb.wins):
        x=bar.get_x()+bar.get_width()/2;v=bar.get_height()
        ax.text(x,v+.5,f'{v:.2f}%',ha='center',fontsize=10)
        ax.text(x,min(v*.50,7),f'{win}/6',ha='center',va='center',fontsize=9,
                bbox={'boxstyle':'round,pad=.2','facecolor':'white','edgecolor':'#333333'})
    ax.legend(loc='upper right',facecolor='white',edgecolor='#555555')
    format_ax(ax);save(fig,out,'F05_Ablation_By_Block')
    # fig6: critical path decomposition, NOT request-level pure GPU kernel times
    ph=[('critical_dependency_release_ms','依赖释放','#E5DED2'),('critical_central_queue_and_admission_ms','队列与准入','#AEB1B2'),
        ('critical_input_prepare_ms','输入准备','#DBAC99'),('critical_header_wait_ms','响应头等待','#B9A273'),
        ('critical_postprocess_ms','后处理','#D6B7AA'),('critical_first_content_wait_ms','首内容等待','#C88E82'),
        ('critical_content_receive_ms','内容接收',COL['Ours']),('critical_response_tail_ms','尾部返回',COL['Affinity'])]
    temp=raw[raw.block==3].groupby('method').mean(numeric_only=True).loc[['Affinity','Ours']]
    fig,ax=plt.subplots(figsize=(7.6,4.6));bottom=np.zeros(2)
    for col,label,color in ph:
        values=temp[col].to_numpy(float)
        ax.bar(['Affinity','Ours'],values,bottom=bottom,label=label,color=color,edgecolor='black',linewidth=.48,width=.55)
        bottom+=values
    for i,total in enumerate(bottom):ax.text(i,total+28,f'{total:.1f} ms',ha='center',fontsize=9.6)
    ax.set_ylim(0,3550);ax.set_xlabel('方法 (Method)');ax.set_ylabel('关键路径平均时延 (ms)')
    format_ax(ax)
    fig.subplots_adjust(top=.78)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.998),ncol=4,
               fontsize=9,edgecolor='#555555',facecolor='white',columnspacing=.75)
    save(fig,out,'F06_B3_Critical_Path_Breakdown')
    # fig7
    fig,ax=plt.subplots(figsize=(7.4,4.7));p=prefix.sort_values('block');x=np.arange(4);w=.32
    b1=ax.bar(x-w/2,p.within_prefix_mean,color=COL['gold'],width=w,edgecolor='black',linewidth=.65,label='工作流内平均共享前缀')
    b2=ax.bar(x+w/2,p.cross_all_lcp_tokens,color=COL['Ours'],width=w,edgecolor='black',linewidth=.65,label='工作流间公共前缀')
    ax.set_xticks(x,[f'B{i}' for i in p.block]);ax.set_xlabel('工作流数据组');ax.set_ylabel('规范化前缀 token 数');ax.set_ylim(0,3100)
    format_ax(ax);annotate_bar(ax,b1,fmt='{:.1f}',offset=40);annotate_bar(ax,b2,fmt='{:.0f}',offset=40)
    ax.legend(facecolor='white',edgecolor='#555555',loc='center right',bbox_to_anchor=(.98,.64),fontsize=10)
    save(fig,out,'F07_Two_Level_Prefix_Opportunity')
    # fig8
    fig,ax=plt.subplots(figsize=(7.2,4.6));phases=['prefill_only','decode_only','mixed'];x=np.arange(3);w=.32
    h=host.set_index(['method','phase'])
    for off,method in [(-w/2,'Affinity'),(w/2,'Ours')]:
        values=[float(h.loc[(method,k),'host_mean_ms']) for k in phases]
        bars=ax.bar(x+off,values,width=w,color=COL[method],edgecolor='black',linewidth=.65,label=method)
        annotate_bar(ax,bars,fmt='{:.1f}',offset=1.1)
    ax.set_xticks(x,['Prefill Only','Decode Only','Mixed']);ax.set_xlabel('原生模型服务阶段类型');ax.set_ylabel('平均 Host 阶段时延 (ms)');ax.set_ylim(0,119)
    ax.legend(facecolor='white',edgecolor='#555555',loc='upper left')
    format_ax(ax);save(fig,out,'F08_B3_Native_Serving_Phase')
    (out/'FIGURES_AND_EVIDENCE.md').write_text('''# 绘图索引（全部无顶部标题，白色背景）\n\n| 文件前缀 | 来源与指标 | 备注 |\n|---|---|---|\n| F01 | 固定B3八轮 E2E JCT 均值 | 真实源基线 RR，不是 Preble |\n| F02 | 固定B3八轮有限批次 workflows/s | 非稳态容量 |\n| F03 | 固定B3八轮条件级平均JCT箱线图 | 不是每请求的JCT分布 |\n| F04 | 固定B3八轮配对 JCT、P95、TTFT | 对每轮先求改善，再平均 |\n| F05 | 四组×六轮历史消融 | 主方法对 Affinity |\n| F06 | B3关键路径阶段（3次独立辅助观测） | 不与八轮样本混为一组 |\n| F07 | 工作流内、工作流间 token 前缀机会 | 非实际缓存命中率 |\n| F08 | B3原生host迭代阶段耗时 | 不等于GPU kernel ITL |\n\n没有绘制 load-sensitivity 折线图；该内容由统计表解释。若本机没有合适楷体请安装自己有使用权的楷体字体，文件包不含字体文件。\n''',encoding='utf-8')
    print(f'PLOTS_OK={out} font_chinese={cn} font_serif={latin}')

if __name__=='__main__':main()
