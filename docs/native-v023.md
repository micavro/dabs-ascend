# 现场 V1 原生适配：实现、边界与验收

依据：`复现修复.zip`（已保存的解包资料，SHA256 `d1f4c42186b4ed1d64ccb47ae4adfabd7e8dbb6cfc9c74fdbda4fc272d30a6c7`）、师兄关于“实现方式、状态未取到，核心算法不用改”的说明，以及后续确认的 V1/async/1024/128/10/4096 配置。本次不是重新运行旧 prepare；也没有修改冻结原件或原哈希清单。

## 1. 版本适配与参数贯通

已逐段核对官方源码：[vLLM v0.23.0](https://github.com/vllm-project/vllm/tree/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665)、[vLLM-Ascend v0.23.0rc1](https://github.com/vllm-project/vllm-ascend/tree/f4a08bddd0cc65a0bd8c3d377b158ae5ca7527db)。`experiments/ascend/dabs_native/upstream.json` 固定这些文件的 SHA256，包含 Ascend 对 KV coordinator 和执行路径的既有 patch。**尚未取得这些具体文件的现场哈希，因此这是一份严格拒绝不匹配版本的实现，不宣称已匹配现场二进制。**

已有采集确认的 52 个项目文件与 `317b8a1` 一致，并不覆盖推理栈的源码。安装前只查适配涉及的文件；不匹配时输出精确路径与 expected/actual。需要补采的仅是这些差异文件的指定片段。CANN 精确版本未知不被替换成 npu-smi 版本。

四处附加 hook：

| 文件 | 附加点 | 用途 |
|---|---|---|
| `vllm/v1/engine/core.py` | EngineCore 初始化末尾 | 安装调度器边界包装和 utility RPC 方法 |
| `vllm/entrypoints/openai/api_server.py` | build_app 返回前 | 认证控制/固定 token 流接口，沿用现有 AsyncLLM |
| `vllm/v1/core/sched/output.py` | SchedulerOutput 声明字段 | batch ID/组成随真实执行消息到 worker |
| `vllm_ascend/worker/worker.py` | NPUWorker 类定义后 | forward 流事件与 worker RPC |

安装保存原文件、原/新哈希和适配包哈希；启动和第一次 worker 执行再次校验。回滚先校验全部备份再恢复，保留退役适配包；不停止或启动服务。当前只支持 TP/PP/DP=1、单 full-attention KV group、FCFS AsyncScheduler、eager、APC、chunked prefill，无 speculative/KV connector。其他配置明确拒绝。

资源来源为实际 scheduler/KV manager：`max_num_scheduled_tokens` 是有效每步预算，配置中的 `max_num_batched_tokens` 另记；物理容量为 block pool 的非 null 块数。`common.configuration` 将资源传至快照、候选、资源投影、评分、准入和报告。客户端在途上限独立为 2，服务序列上限为 10。

`native_algorithms.py` 在隔离模块中对冻结 V5/Ayo/RR/SLL 和评分器进行精确锚点变换；记录 `transformation_manifest()`，核验原始来源。V5/Ayo 的 4096 限制变为有效步预算；Ayo 保留原 conservative reserved-block 负载：已提交请求的完整 prompt+max_output 预算，加本轮拟放置请求；不扣预期命中，不计可回收缓存。分母为引擎真实 KV 容量，physical active 块仅另行观测。原算法候选数、收益门槛、decode guard、目标函数和时间窗口不放宽。

`native_scoring.py` 只替换依赖引擎语义的批次构造：真实 running 顺序后接 waiting；APC 跳过发生在 token 预算扣除之前；FullAttention 的 chunk 可按 token 切分，不能把 128-token KV 块大小错误当作 chunk 必须整除的单位。异步待完成批次按 batch ID 保留，冻结模型保守地串行预测剩余 host-step 时间，不冒充设备已执行时间。评分的排序先投影到服务端可执行的安全槽位。

输入构造仍沿用冻结协议的 16-token 命名空间检查，保持原始 tokens/DAG；这个检查不是物理 KV 块大小。物理分配、共享前缀证据长度与资源投影按 128 对齐。服务模型中 `/16`、`/1024` 等特征缩放是回归量纲，也不是物理块大小。

## 2. 原生状态、缓存与动作

师兄后续确认的三项口径已落实：有限窗口排序、Ayo 保守 KV 预留负载、引擎 host-step elapsed。原始依据分别为 `joint18v2/common.py` 的 `resolve_control/stable_prefill_reorder`、`runtime/joint18v2_hook.py` 的 `reorder/schedule/begin_step/end_step`、`wfsched/ayo_topology.py` 与 `joint18v2/service_fit.py`。这些冻结文件没有修改。

调用链：认证 HTTP → 现有 `engine_core.call_utility_async` → EngineCore 主循环 → 原 scheduler / KV manager / coordinator / block pool。设备计时经原 worker collective RPC 返回。HTTP 进程没有模拟 KV 或影子队列。

每份快照带引擎实例、前一实例、主机 boot ID、PID、安装指纹、快照序号、引擎 step、采集时刻与 120 ms 有效期。running/waiting/skipped_waiting、请求阶段、已排程与已完成 token、占用、可回收缓存、allocator free 块和未来增长预留分别保存。free 包含可回收缓存；不能把两者相加当成额外容量。快照在控制器使用往返耗时做保守新鲜度界限，固定动作窗口另要求控制器和引擎的 boot ID、Linux time namespace inode 相同，核验后共用 CLOCK_MONOTONIC 截止时间。

逐前缀查询传实际 token IDs 与 cache salt，使用引擎自己的 block hasher/coordinator 和“完整命中仍重算最后一个 token”规则。返回 hit / miss / unknown；未完成异步写入涉及的块返回 unknown、tokens=null。每次重新查询真实 block pool；应用动作前重查具体证据，驱逐、重启或到期不能沿用历史驻留。

动作有唯一 ID、原快照、资源摘要和截止时间。EngineCore utility loop 位于 schedule 调用之间，执行时再次检查队列成员、目标状态/进度、缓存、资源和有效期。允许不相关的 step 推进，但目标已经开始 prefill 或进度发生变化时拒绝。重复内容返回同一回执；重复 ID 携带不同内容拒绝。实际队列对象仅重排可修改的等待/未在执行的部分 prefill 槽位，decode 和在途槽位不动。

requested → accepted → applied/rejected/expired 有独立时间记录；部分动作记录 partially_applied 和逐项拒绝，随后记录实际 schedule step/order。新请求先取得物理容量准入许可，再通过 AsyncLLM 提交，EngineCore 验证输入摘要、长度和许可，保存 node/client/server/engine ID 映射。路由只发生在未提交请求。

120 ms 是快照/动作提交的新鲜度上限，不是算法优先级窗口。控制器按原固定 200 ms 周期提供 `not_before_ns/action_expires_ns`；每次原生 schedule 前重新执行安全槽位排序。到固定截止后停止使用旧 order，在下一次迭代按原始 EngineCore 准入时间恢复 Prefill FIFO；5 秒年龄保护放弃优先级重排，Decode/尚有在途执行的槽位保持原生顺序。未执行的许可有单独 2 秒交付租期并占用保守资源预留，不延长队列窗口，也不撤销已执行 token。评分器使用相同有限窗口、年龄保护与 continuation。

评分过时后的回退只允许基线的新请求准入，由引擎原子读取新状态并校验；不会把过期队列重排换一个快照 ID 重发。许可应用与请求真正进入引擎分开记录，HTTP 200 不计作引擎应用。

## 3. 标定、验收与证据边界

标定目标已按补充纠正为 `engine_host_ms`，不再使用 forward 作为目标。对实际 `core.step_fn`（`EngineCore.step` 或 `step_with_batch_queue`）计时，记录调用起止、提交批次 ID 和完成批次 ID。只有同一调用明确提交且完成同一个批次时，才作为与旧主机 step 边界对应的拟合样本。异步流水线可能在一次调用提交 B、回收 A：保留其调用 elapsed 和逐批 host span，但标识交叠/近似，排除同口径拟合。样本不足时保持未冻结，不能把这个问题写成现场已解决；不修改现场异步开关。

NPUWorker 在已核对的 `_model_forward` 当前流前后记录两个 `torch.npu.Event`。只有 end.query() 完成才读取 elapsed_time；无 synchronize 强行串行化。host schedule、host dispatch、设备 forward、EngineCore 输出观察、HTTP 时延分开。forward 区间不包括 sampling/D2H/主机排队，也不宣称是整个 batch 的端到端时延。事件和查询开销保留在实验中。

独立标定预先固定 train/holdout 运行，覆盖 128/129/1024/2048/3584 输入，16/32/48/64/128 输出，并发 1/4 与服务上限附近负载、冷/热和实际 mixed 批次。按 replica/phase 拟合原六特征非负模型。每分组至少 20 个 train 和 20 个 holdout 批次；P95 相对误差≤35%，RMSE/均值≤25%，并检查实际 token/context 覆盖。失败记录保留，不利用正式结果反向调参。

实际输入 token、APC 跳过量和逻辑 prefill 量分别记录。六特征以实际输入量（等价于原 logical minus skip）和缓存跳过前的 computed context 构造；回归缩放保持原实现。现场新标定不载入 NVIDIA 系数，预先声明独立训练/验证和误差门槛；这不同于原包已有四块旧 pilot 的迁移系数选择流程，报告保留该差异。预测超过已冻结的 phase 范围时，候选不具备可评分资格，回退原因保留。

`native-probe-one/four` 真正发送请求、查询缓存、制造 waiting、重排队列、发送迟到/重复动作，并运行 fork/join 和排空检查。单副本先检查本地机制；四副本再检查跨副本命中/未命中。`native-invalidate` 明确清空 0 号 prefix cache 以验证失效；它不是普通检查，也不宣称已测完所有内存压力下的驱逐模式。`native-restart-arm/check` 围绕操作者手动重启检验旧身份拒绝。

`mechanism_gate` 根据原始签名记录逐项重算；校准样本必须与真实 batch ID/事件值一致、分组无重复，重新拟合与冻结系数相同。正式实验前再次通过当前 EngineCore 验证签名和实例、代码与配置；不存在强制通过开关。探针、校准或验收缺项保持未通过。

本地没有 NPU。CPU 故障注入、源码锚点检查和原包回归只能说明这些代码路径可执行；**真实进程通信、NPU Event 的现场支持、真实 waiting/队列回执、KV 命中/失效、标定误差和正式 96 条件仍需在现场验证**。若源文件哈希不同，先拒绝安装，再按差异适配。

## 4. 命令、过程和结果

完整手敲顺序见根目录 README。第一条是 `bash run_dabs_r234.sh native-install-check --replica 0`，不是旧 prepare。

保留现有 96 条件、四方法、输入/DAG/种子/执行顺序；每条件 64 工作流、216 请求、774144 输入 token、13184 输出 token。原生结果写入新 `r234_native_*`；仅正常暂停且源码、配置、引擎身份及冻结标定均相同才续跑。新公共执行器的结果不能和旧 HTTP 基线拼接。

请求、快照、具体缓存查询、动作、批次、评分失败和回退全落普通文件。设备计时记录导出并持久化后，控制器才能确认释放引擎 trace 缓冲；不将历史完成记录转成缓存驻留。未观测字段保持 null。共享硬件的干扰、零偏离、负收益和失败原样保留；不以 Ours 更快作为验收条件。
