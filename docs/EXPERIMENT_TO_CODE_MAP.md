# 报告 V4 第六章与复现代码逐项对应

| V4小节 | 输入与算法代码 | 实验计划与数据证据 | 复算操作 |
|---|---|---|---|
| 6.1 固定16工作流 | `source/spra_workflow_accept_v1/wfsched/{selector,routing,worker,runner,acceptance}.py` | `case-confirm` 经登记的R4/λ2/B3，32条件；`evidence/derived/case_confirm_rows.csv`为控制台整理的条件摘要 | `recompute_results.py` |
| 6.2 四组完整消融 | `source/spra_workflow_sched_v2/wfsched/{selector,worker,report}.py` | `source/spra_workflow_sched_v2/plans/ablation.json`；`evidence/catalog/paired_metrics.csv` | `recompute_results.py`，保留全部B1–B4轮次 |
| 6.3 副本/负载扫描 | `source/spra_workflow_sched_v2_1/wfsched/` | `plans/grid.json` (640条件)；checkpoint320条+R2第二轮80条控制台 | `recompute_results.py` 输出400条已观测 / 240条仍缺 |
| 6.4 吞吐量 | 同上述真实执行器 | 原生cohort_workflows_s，cohort_tokens_s | `recompute_results.py`, `plot_experiments.py` |
| 6.5 关键路径分解 | `source/spra_stage19_musique_joint_v1/joint18v2/latency_metrics.py` 与 typed 绑定 | `checkpoint_review_20261012.zip` 中三批 Pilot 的B1–B4 `summary.json` | `extract_pilot_mechanism.py` |
| 6.6 两级前缀与KV | `joint18v2/cache.py`、`typed19/dependency.py`、`runtime/joint18v2_hook.py` | 四组原生共享前缀与跳过tokens，`prefix_structure.csv`；`pilot_mechanism_raw.csv` | `extract_pilot_mechanism.py`, `plot_experiments.py`；跨工作流复用增益尚无独立因果计数 |
| 6.7 原生主机阶段 | `service_fit.py`、`joint18v2/latency_metrics.py` | Pilot `summary.json/service_prediction_by_phase` 三批B3数据 | `extract_native_phase.py` |
| 6.8 决策开销 | 调度器 `selector.py`、`worker.py` | 完整消融的控制决策时间/回退/路由变更 | `ablation_conditions.csv` 和控制表；实测未修改 |
| 6.9 回答质量 | `semantic19/quality.py`、`typed19/quality.py` | 历史 EM/F1；新B3全量答案未上传 | 报告保留质量边界，不强称新运行答案一致 |

## 原始运行代码规则

- `DAG` 是论文 `Ours`：`source/spra_workflow_sched_v2_1/wfsched/selector.py` 和 `worker.py`。
- `RR` 是轮询，`P2C` 是两个副本负载比较，`Affinity` 是缓存亲和参考。`Preble` 没有实测复现。
- 固定服务输入总预算4096，最大上下文4096，`max_num_seqs=64`，缓存块16，1秒活跃/1秒空闲 wave；系统总到达率1、2、4、8 workflows/s。
- 所有数据切片每16个完整工作流组成一个 B1—B4 组（并非把一个工作流拆成四段）。
- 未下载的原始后端日志必须从 GPU 另行归档，不可由摘要 CSV 人工补造。
