# 方法、命名与论文主线

## 1. 主方法不是重新设计：DAG就是已复现D₀

建议正文名称 `DAG`，保留 `DAG-Lite` 作为实现优化消融名，不使用Batch/Joint后缀。
原A保留为 `Affinity`；新增 `RR` 和 `P2C`。没有把名称改为vLLM、Ray或NGINX以暗示完整系统复现。

可以使用的定义：

> 在固定LLM后端batching策略下，结合可观测的缓存复用、批内竞争和副本压力，评估就绪请求的副本分配及执行安排对完整workflow完成时间的影响，选择预测workflow JCT较低的可执行动作。

这里的“完整”指不把未完成的workflow截成几个已完成节点的耗时后冒称完整JCT。
评分对象是已到达且未完成的workflow，剩余节点输出为预测值，不使用未来真实输出或将未来到达作为已知线上输入。
它是有限候选集内的模型引导启发式，不是全局最优求解，也没有真实JCT不退化的数学保证。

## 2. 与原源码的对应关系

`evidence/SOURCE_EXCERPTS.md`保留原文件行号；原文件由冻结归档保存。

|用户定义中的要素|原实现证据|不宜扩大的主张|
|---|---|---|
|缓存复用|native cache observer、参考Affinity、候选副本下的可用前缀/有效prefill工作|不能把收益都解释成更高缓存命中率；最新D的跳过prefill量反而较少|
|批内竞争|预估下一批prefill/decode、上下文量与有效prefill×decode交互项；incumbent decode guard|这是观察/预测固定batching中的干扰，不是改变budget；保护是模型估计不是强实时保证|
|副本压力|已观测running/waiting状态、保守KV容量检查、候选放置下的竞争；原压力估计running+2×waiting|不是理想oracle负载；信息有时效与预测误差|
|workflow目标|编译DAG、后继/并行组、tail估计；evaluate返回完整sum_remaining_jct_ms；只比较rankable结果|不是只按最长路径机械排序，也不是单请求TTFT目标|
|实际动作|原D分支允许就绪节点副本分配、顺序与有界准入，预算不变|最近实测主要体现副本选择，未观察大量原生队列重排，不应把未触发的机制写成已验证原因|

以已到达workflow的到达时刻a_w为常数，在同一快照比较候选时：

    argmin_a Σ_w [ Ĉ_w(a | state, DAG, model) - a_w ]

也等价于比较预测完成时间总和（或相对同一当前时刻的剩余完成时间总和）。
候选执行有限动作窗，随后采用共同continuation直至全部相关workflow完成；未完成/不合法预测不排序。
实际统计仍由用户可见应用到达与最终请求返回的墙钟差计算，不扣除规划CPU耗时。

## 3. 新基线的外部依据（官方文档）

### RR

NGINX HTTP Load Balancing官方文档说明Round Robin是默认方法，没有指定其他算法时使用，支持权重。
实验使用等权请求级轮询。它不读取DAG或前缀，不按最优JCT挑路由。

来源：https://docs.nginx.com/nginx/admin-guide/load-balancer/http-load-balancer/

### P2C

Ray Serve LLM官方Request routing文档的Default routing说明采用Power of Two Choices：随机选两个副本，依据ongoing request count选较少者。
实验实现这项路由原则，但不移植Ray的分布式调度、拒绝/重试、autoscaling、网络栈和placement组机制。

来源：https://docs.ray.io/en/latest/serve/llm/architecture/routing-policies.html

### 为什么没有新增一个名为vLLM的曲线

vLLM 0.6.3官方Engine Arguments的`scheduling-policy`默认值为fcfs。
FCFS是引擎内部排队纪律，不能独立决定多副本请求去哪一个实例。
本实验复用同一vLLM原生执行器；RR/P2C与Affinity基线不向原生队列发优先级重排。
不能将“RR+同引擎FCFS”的曲线直接称为原版vLLM完整部署，也不能把Ray路由kernel的曲线称为整个Ray Serve。

版本匹配来源：https://docs.vllm.ai/en/v0.6.3/models/engine_args.html

这些外部来源用于定义/说明基线，不用于替代用户代码中的A/D行为。
官方文档核对日期：2026-10-12；GPU服务的实际版本、hook和scheduler源哈希以现场gate为准。

## 4. 为什么先不做较大改动

原DAG的两轮同64题复测已得到8个配对全为正、平均JCT配对改善7.78%的信号。
但参考只包含Affinity，同64题是开发问题，不能宣称已经优于新RR/P2C或在每个负载达到10%。
因此先冻结：DAG + 原profile + 原应用与测量边界，再测DAG-Lite最小删冗余补丁。
200ms公共循环、model-visible UUID缓存隔离都暂不改动，以免混入别的改动。
这两个遗留限制必须披露；后续若优化公共循环，应同样提供给所有基线，并另建实验版本。

DAG-Lite的本地48组CPU检查只有语义一致证据，没有整体耗时提升证据：有一个约49.95ms长尾，均值变差。
保留原始benchmark；不事后删除outlier、不重跑挑最小均值、不据此预测GPU加速。

## 5. 下一步判据

主要看同工况、同轮次、同Block的应用平均JCT配对收益；同时保留P95、TTFT、完成率、输出量和EM/F1。
DAG-Lite应先在预先设计的ablation中证明增量；没有增量就保留DAG，不为“优化”而继续加复杂机制。
资源/负载网格不以测得收益决定保留哪些点。低负载、单副本或高负载退化均应保留。
较强的后续解释性消融可关闭DAG相关候选但保留相同缓存/压力模型，这不同于再加入一个简单弱基线。
本包不实现该额外机制消融，以避免与当前已验证算法同时大幅变化。
