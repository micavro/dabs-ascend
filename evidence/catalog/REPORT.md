# Workflow实验档案与JCT比较

按原run分别统计，不跨代码版本或运行协议合并。完整条件比较才标记10%；单次/单Block峰值不替代工况平均。
主指标：同副本、到达率、轮次、Block内JCT百分比下降，再取平均；负值为退化。
请求TTFT、条件P95与客户端TPOT代理保持原口径。原模型可见UUID与输出差异仍存在。
归档不包含模型权重/SSH密钥，不保证另一环境无需依赖即可运行。源码和原始文件归档与实验结论验证是两回事。

## 运行清单

|run|状态|已接受/计划|
|---|---|---|
|spra_workflow_sched_v2/runs/ablation_v2|complete|72/72|
|spra_workflow_sched_v2/runs/grid_v2|failed|41/640|
|spra_workflow_sched_v2/runs/smoke_v2|complete|5/5|
|spra_workflow_sched_v2_1/runs/agecheck_v2_1|complete|10/10|
|spra_workflow_sched_v2_1/runs/grid_v2_1|paused|320/640|
|spra_stage19_typed_handoff_v1/runs/PILOT_20261010T092749Z_421e356e|raw_preserved_not_recomputed|None/None|
|spra_stage19_typed_handoff_v1/runs/PILOT_20261011T185607Z_8e964348|raw_preserved_not_recomputed|None/None|
|spra_stage19_typed_handoff_v1/runs/PILOT_20261011T185806Z_fc5b583b|raw_preserved_not_recomputed|None/None|

## 完整负载工况的平均JCT收益

|run|R|λ|方法|基线|配对|JCT改善%|≥10%|
|---|---:|---:|---|---|---:|---:|---|
|spra_workflow_sched_v2/runs/ablation_v2|4|2.0|DAG|Affinity|24/24|10.45|是|
|spra_workflow_sched_v2/runs/grid_v2|1|1.0|DAG|Affinity|4/8|7.70|不完整|
|spra_workflow_sched_v2/runs/grid_v2|1|1.0|DAG|P2C|4/8|-2.10|不完整|
|spra_workflow_sched_v2/runs/grid_v2|1|1.0|DAG|RR|4/8|3.11|不完整|
|spra_workflow_sched_v2/runs/grid_v2|1|2.0|DAG|Affinity|4/8|-0.17|不完整|
|spra_workflow_sched_v2/runs/grid_v2|1|2.0|DAG|P2C|4/8|-0.60|不完整|
|spra_workflow_sched_v2/runs/grid_v2|1|2.0|DAG|RR|4/8|-1.76|不完整|
|spra_workflow_sched_v2_1/runs/grid_v2_1|1|1.0|DAG|Affinity|8/8|-0.61|未达|
|spra_workflow_sched_v2_1/runs/grid_v2_1|1|1.0|DAG|P2C|8/8|1.34|未达|
|spra_workflow_sched_v2_1/runs/grid_v2_1|1|1.0|DAG|RR|8/8|1.79|未达|
|spra_workflow_sched_v2_1/runs/grid_v2_1|1|2.0|DAG|Affinity|8/8|-0.47|未达|
|spra_workflow_sched_v2_1/runs/grid_v2_1|1|2.0|DAG|P2C|8/8|0.71|未达|
|spra_workflow_sched_v2_1/runs/grid_v2_1|1|2.0|DAG|RR|8/8|-2.00|未达|
|spra_workflow_sched_v2_1/runs/grid_v2_1|1|4.0|DAG|Affinity|8/8|-2.43|未达|
|spra_workflow_sched_v2_1/runs/grid_v2_1|1|4.0|DAG|P2C|8/8|-1.36|未达|
|spra_workflow_sched_v2_1/runs/grid_v2_1|1|4.0|DAG|RR|8/8|-2.32|未达|
|spra_workflow_sched_v2_1/runs/grid_v2_1|1|8.0|DAG|Affinity|8/8|1.35|未达|
|spra_workflow_sched_v2_1/runs/grid_v2_1|1|8.0|DAG|P2C|8/8|1.00|未达|
|spra_workflow_sched_v2_1/runs/grid_v2_1|1|8.0|DAG|RR|8/8|3.12|未达|
|spra_workflow_sched_v2_1/runs/grid_v2_1|2|1.0|DAG|Affinity|4/8|-2.01|不完整|
|spra_workflow_sched_v2_1/runs/grid_v2_1|2|1.0|DAG|P2C|4/8|1.90|不完整|
|spra_workflow_sched_v2_1/runs/grid_v2_1|2|1.0|DAG|RR|4/8|0.56|不完整|
|spra_workflow_sched_v2_1/runs/grid_v2_1|2|2.0|DAG|Affinity|4/8|2.69|不完整|
|spra_workflow_sched_v2_1/runs/grid_v2_1|2|2.0|DAG|P2C|4/8|-1.01|不完整|
|spra_workflow_sched_v2_1/runs/grid_v2_1|2|2.0|DAG|RR|4/8|1.91|不完整|
|spra_workflow_sched_v2_1/runs/grid_v2_1|2|4.0|DAG|Affinity|4/8|2.75|不完整|
|spra_workflow_sched_v2_1/runs/grid_v2_1|2|4.0|DAG|P2C|4/8|0.65|不完整|
|spra_workflow_sched_v2_1/runs/grid_v2_1|2|4.0|DAG|RR|4/8|5.75|不完整|
|spra_workflow_sched_v2_1/runs/grid_v2_1|2|8.0|DAG|Affinity|4/8|2.79|不完整|
|spra_workflow_sched_v2_1/runs/grid_v2_1|2|8.0|DAG|P2C|4/8|4.75|不完整|
|spra_workflow_sched_v2_1/runs/grid_v2_1|2|8.0|DAG|RR|4/8|7.79|不完整|
|spra_workflow_sched_v2_1/runs/grid_v2_1|3|1.0|DAG|Affinity|4/8|4.65|不完整|
|spra_workflow_sched_v2_1/runs/grid_v2_1|3|1.0|DAG|P2C|4/8|7.30|不完整|
|spra_workflow_sched_v2_1/runs/grid_v2_1|3|1.0|DAG|RR|4/8|4.58|不完整|
|spra_workflow_sched_v2_1/runs/grid_v2_1|3|2.0|DAG|Affinity|4/8|4.03|不完整|
|spra_workflow_sched_v2_1/runs/grid_v2_1|3|2.0|DAG|P2C|4/8|8.78|不完整|
|spra_workflow_sched_v2_1/runs/grid_v2_1|3|2.0|DAG|RR|4/8|0.83|不完整|
|spra_workflow_sched_v2_1/runs/grid_v2_1|3|4.0|DAG|Affinity|4/8|0.85|不完整|
|spra_workflow_sched_v2_1/runs/grid_v2_1|3|4.0|DAG|P2C|4/8|3.69|不完整|
|spra_workflow_sched_v2_1/runs/grid_v2_1|3|4.0|DAG|RR|4/8|0.56|不完整|
|spra_workflow_sched_v2_1/runs/grid_v2_1|3|8.0|DAG|Affinity|4/8|1.86|不完整|
|spra_workflow_sched_v2_1/runs/grid_v2_1|3|8.0|DAG|P2C|4/8|5.91|不完整|
|spra_workflow_sched_v2_1/runs/grid_v2_1|3|8.0|DAG|RR|4/8|3.87|不完整|

## 核对失败与范围

[]

请保留full_raw.zip与ARCHIVE_MANIFEST.json。review.zip仅为分析回传材料，不替代本机全量归档。
记录“当前已完成”需要index.json覆盖；本脚本不会启动缺失实验或重试失败attempt。
