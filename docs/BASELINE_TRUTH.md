# 基线身份核验：RR ≠ Preble

用户的 V4 研究报告在表4中将 `Preble` 描述为“依次轮换副本，按先进先出处理就绪请求”；该行为实际是 **Round Robin (RR)**，并非 ICLR 2025 的 Preble 路由算法。

已保存的原始 `source/spra_workflow_sched_v2_1/wfsched/routing.py` 中 `policy == 'RR'` 分支仅用 `(count+i) % R` 轮换副本；该模块开头的文档字符串还明确声明 RR/P2C 是 request-level baseline kernels，**不是 Ray/NGINX 等产品复现**。

因此历史条件 `R4_L2_T1_B3_RR`、核心案例的“2701.18ms”与相对 RR 的“10.13%”都是真实 RR 测量。**没有任何证据证明这些数值是 Preble 的测量。**

本工程保留所有原始名称和哈希。图表展示 `RR`、`P2C`、`Affinity`、`Ours`；其中 `Ours` 仅是 `DAG` 的论文展示名。未来如需真正引入 Preble，需要从可信源码构建其可比路由器，取得真实前缀历史、eviction 及后端缓存状态接入，在同一正式实验计划与负载下新增运行，得到新数据后才能更新研究报告对应图表。

**不要只替换图例或报告字符串。**
