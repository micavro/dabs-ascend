# 纯workflow请求调度：原实验定位与复用审查

## 1. 找到的原实验

与用户所述“纯调度已有较好收益”直接对应的材料是：

`spra_musique_d0_batching_v1/analysis/D0_checkpoint.json`

其中检查点为 `MUSIQUE_SCHEDULING_D0_20261010`，原run为：

`/home/wmy/sci/spra_paper_experiments_github/spra_stage19_typed_handoff_v1/runs/PILOT_20261010T092749Z_421e356e`

检查点写明：64个不同问题分为四个互不重叠16题块，每块分别执行A/D/B/Decoupled/Joint。20个条件，320个工作流执行、1065个LLM请求、9712个输出tokens。它不是四次相同问题的独立重复。

D相对A平均JCT配对下降13.303823%，四块分别为6.553023%、21.387563%、21.978478%、3.296227%，算术平均与检查点一致。**本地尚未取得该run的完整原始记录，不能称已从nodes/backend独立重算历史成绩。**广域采集的OMISSIONS明确列出119条该pilot的相关文件因证据字节上限被跳过；模型、先验与源码则在包内。

## 2. 保存的性能结果

下表绝对值是四个条件均值的平均，收益是先同block求百分比再平均；两者不能混用。

|指标|A|D₀|D₀相对A的配对改善|
|---|---:|---:|---:|
|平均workflow JCT，ms|2339.277|1994.497|13.304%|
|每条件P95 JCT，ms|3486.066|2728.199|20.843%|
|平均请求TTFT，ms|313.997|263.546|14.463%|
|每条件P95请求TTFT，ms|557.646|431.980|20.462%|
|平均请求TPOT代理，ms|45.762|39.964|10.954%|
|每条件P95 TPOT代理，ms|73.773|60.276|13.167%|
|就绪到首内容，ms|449.989|384.441|13.394%|
|cohort workflows/s|1.75583|1.74875|-0.174%|

D₀与A总输出tokens分别1935和1938；平均最终EM分别18.75%和17.1875%，F1分别27.7307%和25.6473%。这些质量差异不能归因为调度改善，不构成质量等价证明。Cohort完成率没有提高，JCT收益不能写成容量收益。

这个20条件检查点没有RR、Preble或Ayo。只能引用其中已经测得的对照；不能把D/A的13.30%推及所有基线。

## 3. 与12.68%案例的区别

另一个文件 `spra_musique_b1_case_v1/analysis/HISTORICAL_CONTEXT.json` 保存的是后来的4/s案例：

- A=2624.504ms，D=2420.323ms，C=2291.792ms；
- D/A调度收益7.7798%；
- C/D batching增量5.3105%；
- C/A组合收益12.6771%。

因此原先“12.68%是不是纯调度”的疑问已经解决：它不是。用户现在应恢复的是13.30%的D₀检查点，而非将12.68%改写为纯调度收益。

后续B1单案例扩展中，DAG-Fixed仍是原D₀，但不同工况下相对Affinity的平均JCT配对收益为：

|副本R|名义总到达率|DAG-Fixed/Affinity|
|---:|---:|---:|
|1|4|-3.266%|
|2|4|+1.481%|
|3|4|-4.314%|
|4|2|+4.230%|
|4|4|+2.812%|
|4|8|-5.498%|
|4|10|-8.378%|

这些数来自上传的原始保存汇总CSV，不是在本地重跑GPU。它们不否定旧64题的检查点，但证明收益依赖工况和题集。单个已挑选16题上的扩展不能替代原64题复现。

## 4. 代码调用关系与身份核对

调用链：

```
spra_stage19_typed_handoff_v1/tools/run_typed.py
  run('pilot')
    prepared() -> 原profile、model和priors
    typed19.workflow.compile_packets()/instantiate()
    joint18v2.client.execute(..., method='D', ...)
      joint18v2.client.process_plan()
        semantic19.selector.choose(..., 'D', ...)
          reference() -> A参考动作
          schedule_alternative() -> 一个DAG感知调度候选
          evaluate(00), evaluate(10) -> 比较完整预测JCT
```

已实际核对：冻结manifest内48项源码全部匹配；typed归档与working tree的23个共同文件一致，Stage19的79个一致，D0-batching的30个一致。该一致性证明的是获得的源文件版本关系，不是实时GPU状态。

有效运行配置：R4、rate2、wave、固定4096、200ms节拍、4K上下文、Qwen2.5-1.5B-Instruct、自然JSON答案、max96、2700证据预算。虽然configs/semantic.json的默认base_cap是1024，pilot入口会用冻结profile的best_fixed_cap=4096覆盖；不能只看配置文件判定实际预算。

D₀做就绪组移动、DAG剩余路径优先顺序和有界准入候选；以已到达未完成工作流的预测剩余完成时间为目标。它保留模型guard，不是最优解证书。A自身已是缓存亲和加压力打破平局，不是纯RR。

## 5. 可以直接复用，但有三项重要代码结论

### 5.1 不要用force_caps当“关闭adaptive batching”开关

原client.execute先处理force_caps，然后才进入优化调度分支。传force_caps=[4096]*R会绕过D规划，形成伪D对照。纯调度应使用method='D'、base_cap=4096、force_caps=None。本次新入口固定这条调用合同，并核对实际控制向量和原生cap。

### 5.2 D内部有未使用的budget proposal计算，但没有执行adaptive预算

semantic19.selector.choose在判断D只需00/10之前，无条件调用budget_alternative。00/10保持base cap；01/11不会被D评分选择。故D是固定预算调度，但仍付出这段无用候选构造成本。

为复现，本次没有删除这段计算。后续D-fast可以把budget proposal构造仅放到B/Decoupled/Joint分支。由于总deadline与真实CPU时间会影响后续候选完成情况，不能声称删掉计算后的实机轨迹必然相同，更不能提前保证其收益。

### 5.3 48不是该冻结路径上的实际初始输出估计

typed19.workflow优先读取冻结priors中的p75再转int。实际为2-hop 9，3-hop 10，4-hop 10。48只是缺省回退。此前基于继承配置提出的长度估计风险，在这套具体实现里不能按“实际用48”定性。

另外，旧逻辑batch检查器的尾块舍入问题，在后来的case23中已经修正。本包默认复用其纯检查函数；只影响执行后的独立验证，不改变引擎、调度目标和时钟，原始源码与结果不修改。没有放宽未解释失配、抢占或证据缺失。

## 6. 机制解释的限度

历史检查点记录D有31次偏离Affinity的放置、1次队列重排、0次策略延迟准入、0次预算切换。它更支持“由剩余DAG完成时间指导的缓存/竞争权衡与放置”这一解释，而不是“大量重排序或批处理带来收益”。

D的累计skipped-query数量还少于A约2.2%，不能将13.3%简单解释为“缓存命中更多”。关键路径账目有首内容等待及内容接收的下降，但这不是已隔离的因果反事实。

仍需增加与D使用同状态、同缓存信息、同成本模型、但不利用DAG的负载感知对照，才能拆出DAG信息本身的增量。

## 7. 下一步

先用本包audit-history读取已定位原run的A/D摘要，再gate核对真实后端，做8条件A/D复测和事先安排的8条件反向顺序复测。原D、先验、提示、后端不改，不做adaptive batching。新入口不运行profile、不改变环境、也不重新接入DABS占位接口。

这是历史条件的纯调度复测，不是原5方法20条件全流程原样恢复。只有完成并保存同期A/D结果，才能说在当前服务状态下复现了收益。

扩大验证应先覆盖原64题及另行冻结的未按性能筛选问题，再做R=2/3/4且每个R都有rate=2/4/8的矩阵。新题、负载和输出合同分别标注；轻载、负收益和未达10%都保留。

本次交付没有目标GPU访问或新MuSiQue推理。代码测试、源码匹配、历史摘要与GPU复现是不同层次的证据。
