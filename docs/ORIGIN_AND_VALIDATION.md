# 代码与实验来源及可复现检验

| 源位置（本次上传材料） | 恢复的内容 | 是否可离线验证 |
|---|---|---|
| `WF_available_evidence_20261012.zip` | 原始v2/v2.1调度运行代码、图表/复现证据、`wf_ops_v1`、历史D0 | 是；v2.1原生95单测通过 |
| `spra_workflow_port_v1.zip` | 案例确认工程的九个字节相同内核、控制器/测试与新路径输入profile | 是；121单测通过；runtime另行提供 |
| `review.zip` | 原始stage19/typed19应用、真实JSONL数据、冻结prepared模型以及3批Pilot摘要，448条旧suite条件总索引 | 是；48源码哈希检验通过 |
| `Pasted text.txt` | R2第二轮80条条件的控制台精简输出 | 是；解析后加入旧320条得400/640 |
| 用户对话中保存的32行B3结果 | B3四算法八轮条件级JCT、P95、TTFT、吞吐量、实际输出token | 可从 `case_confirm_rows.csv` 复算均值和配对差异；缺完整RAW时间线 |

## 在本次重包前实际做过的CPU检查

- 固定案例交接版：`bash source/spra_workflow_accept_v1/run_checks.sh` → **121 tests OK**，`WF_CHECK` 48关键源文件 `hash_match`，GPU生成请求0。
- 原生v2.1：设置 `WF_SOURCE_ROOT=$PWD/source` 后运行 `bash source/spra_workflow_sched_v2_1/run_checks.sh` → **95 tests OK**，相同48源文件身份与原先冻结模型哈希通过。
- `tools/recompute_results.py` 重新得到主网格旧320+R2新80=400/640，固定B3相对RR/P2C/Affinity配对收益分别为10.1338%/10.8504%/14.3484%，历史四组对Affinity平均10.4513%。
- `tools/extract_pilot_mechanism.py` 从 `review.zip` 逐个真实 Pilot 摘要重建 **24行**时延及KV记录，与先前分析CSV全部数值一致。
- `tools/extract_native_phase.py` 从原生阶段统计重建 **6行**Host阶段均值，与先前CSV一致。
- `tools/plot_experiments.py` 生成八幅无标题白底PNG/PDF（中文楷体可用时优先，未分发字体）。

这些本地通过的结论不代表“已经重新跑出GPU主实验”。所有实际 GPU 的数值仍是历史结果或需在新服务器真实复现的目标。
