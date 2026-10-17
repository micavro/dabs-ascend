# 原生筛查诊断 v1

本版本只观察当前路径，不尝试额外候选、不修复疑似问题、不缩短 DAG、不改变评分切片。CPU 测试只能确认记录与流程，不能证明现场开销或 NPU 性能。筛查选定 R3_L8、R4_L4、R3_L4 的 T1，负结果同样保留；一轮结果不作统计显著性或最优负载结论。

## 从哪里读

- `copy_logs/COPY.txt`：有限 D（原因）、S（评分详情）、SELECT（选动作）、RECEIPT（最终回执），全部决策聚合成 TIMING/DIAG/C。P 最多每 30 秒尝试一条，受字节预算限制。
- `conditions/<label>/decisions.jsonl`：完整决策。COPY 的 condition + i 可定位；有晚返回记录时还需区分 event。原始 RPC 由 snapshot ID 关联 `native_events.jsonl`。
- 工作流目录的 `planning_audit.json`：每个 Ours 的完整参考/不同候选比较数量；过期、回退、迟返回、缺 worker 完成时间、不完整评分不能通过。
- `r234_screening/acceptance.json`：仅所选 4/12 条的工作量报告；formal_acceptance=false、full_matrix_complete=false。不是 96 条实验验收。

## 诊断含义

| 位置 | 字段 | 用途与不能推出的结论 |
|---|---|---|
| planning_window | limit_remaining_ms、binding_limits | 同时列软上限、固定 tick、快照到期的剩余毫秒与最紧约束；带符号原值保留，budget_ms 仍按原规则截零。 |
| result | planning_dispatch_delay_ms、planning_worker_budget_ms | 区分派发/IPC 消耗与 worker 真正剩余预算；同一绝对截止时间不重置。 |
| snapshot | native_prepare_parts_ms | identity、query_snapshot、query_reference、query_tail、query_sort、query_pack、query_total；query_total 包含其子项，不相加。原 native_prepare_ms/RPC/projection 保留。 |
| native_query_budget | ready_requests/covered_requests 对比 ready_at_snapshot/covered_at_snapshot/new_ready_after_query | 前者是发查询时，后者是 RPC 后最终快照时。查询覆盖不是缓存命中，更不是源副本 known 证明；RPC 中新 ready 不再被误解成漏查。 |
| opportunity_diagnostics | ready、reference_admitted/held、mutable、submitted/admitted_by_replica、client_slots_after_reference、source_known_* | 是否有可操作请求、客户端空位和真实源缓存证据；剩余 client slots 不证明 KV 可行。保留 reserved/capacity blocks 和 reserve_fraction。 |
| candidate_diagnostics | pressure_cache_eligible_routes/nodes | 原候选逻辑实际构建出的路由池（满足源缓存 known 与压力差）；不是最终可执行候选数。 |
| 同上 | resource_projection_attempts、admission_set_would_change | 原资源投影尝试和因准入集合改变被拒数量；admission_rejection_samples 给出失去/新增准入节点样例。不绕过同准入集约束。 |
| 同上 | alternatives_skipped_after_rejection | 某节点首目标被拒后，原 seen 逻辑确实跳过了别的目标；样例含拒绝目标和跳过目标。**未测试后者可行性，不能据此直接声称漏掉合法更优候选。** |
| 同上 | shortlist_before_native_dedup、shortlist_at_cap、no_effect_*、held_only_route_no_effect、distinct_executable_candidates | 定位“原始 3 项名额占满→native 可执行去重消失”；at_cap 只说明碰到上限，不证明上限外存在合法候选。未增加枚举。 |
| evaluations[].diagnostic | slice_budget_ms、planner_remaining_at_score_ms、evaluator_elapsed_ms、slice_overrun_ms | 原评分器分给本次评估的实际切片和剩余总预算；cpu_wall_ms 是评分器自己的计时，wall_ms 是整次 selector 耗时，不能混用。 |
| 同上 | stages_ms(validate/clone/setup/rollout)、progress、rollout_steps、expanded_model_iterations、coalesced_decode_batches | 区分校验、浅复制、初始化、推进；进度含初始已完成节点/工作流，不是该切片新增完成量。setup 包括模型/缓存/事件表及初始 release_successors。部分推进不能当作完整分数。 |
| result | post_selection_diagnostics_ms | 完成选择后整理诊断的耗时；不包括所有计时/回调成本，不能称为总插桩开销。worker 最终完成时间仍包括这些实际成本。 |

DIAG 中 query_observation_totals 是观测次数累加，opportunity_decision_totals 是有参考动作返回的决策累加；同一个请求会跨决策反复出现，不能当成去重请求数量。漏斗各项不是互斥原因，不能全部相加解释为失败次数。COPY 的 top 计数省略时有 `_other_count/_other_keys`，完整分布仍在原始日志。

## 怎么决定下一步

1. ready/admitted/known/pressure 机会长期不足、候选池本就为零：先讨论负载/资源限制是否给了调度空间，不判定实现故障。
2. 候选池有机会，但持续出现拒绝后跳过、held-only 名额耗尽：结合原始样例离线重放确认其他目标是否合法，再讨论候选生成修复。本版本不验证/尝试那些目标。
3. 可执行候选存在：看实际评分切片及 validate/clone/setup/rollout 占比，再讨论观测或评分优化。不能用整次 wall_ms 推断单次评估所需时间。
4. 参考和候选按时完整评分但保留参考：看成本、guard/margin；不选择非参考动作本身不是失败。选了动作还需最终引擎回执，最终性能以同工作量配对 JCT 等指标评估。

只有证据定位并确认后才做机制优化。本次未修改服务端插件、安装脚本、冻结源文件或标定模型。
