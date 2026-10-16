# 现场 V1 原生适配：实现、边界与验收

依据：`复现修复.zip`（已保存的解包资料，SHA256 `d1f4c42186b4ed1d64ccb47ae4adfabd7e8dbb6cfc9c74fdbda4fc272d30a6c7`）、师兄关于“实现方式、状态未取到，核心算法不用改”的说明，以及后续确认的 V1/async/1024/128/10/4096 配置。本次不是重新运行旧 prepare；也没有修改冻结原件或原哈希清单。

## 1. 版本适配与参数贯通

已逐段核对官方源码：[vLLM v0.23.0](https://github.com/vllm-project/vllm/tree/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665)、[vLLM-Ascend v0.23.0rc1](https://github.com/vllm-project/vllm-ascend/tree/f4a08bddd0cc65a0bd8c3d377b158ae5ca7527db)。`experiments/ascend/dabs_native/upstream.json` 固定这些文件的 SHA256，包含 Ascend 对 KV coordinator 和执行路径的既有 patch。**这些是溯源参考，不是运行白名单。兼容性按实际接口、字段和调用判断；版本字符串或哈希不同不会单独拒绝加载。**

已有采集确认的 52 个项目文件与 `317b8a1` 一致，并不覆盖推理栈的源码。安装前只检查四个 hook 的实际插入点；缺少时输出精确路径与类/函数名。不要求重新采集整个环境。CANN 精确版本未知不被替换成 npu-smi 版本。

四处附加 hook：

| 文件 | 附加点 | 用途 |
|---|---|---|
| `vllm/v1/engine/core.py` | EngineCore 初始化末尾 | 安装调度器边界包装和 utility RPC 方法 |
| `vllm/entrypoints/openai/api_server.py` | build_app 返回前 | 控制/固定 token 流接口，沿用现有 AsyncLLM |
| `vllm/v1/core/sched/output.py` | SchedulerOutput 声明字段 | batch ID/组成随真实执行消息到 worker |
| `vllm_ascend/worker/worker.py` | NPUWorker 类定义后 | forward 流事件与 worker RPC |

安装保存原文件、原/新哈希和适配包哈希用于溯源；运行时只检查实际能力，不核验代码身份。更新可以覆盖已有适配并保留前版，重复插入点不会叠加。回滚检查路径和备份是否存在，另存当前源码后恢复；低层安装/回滚不启动或停止服务，一键 install 仅重启需要加载补丁的副本。当前只支持 TP/PP/DP=1、单 full-attention KV group、FCFS AsyncScheduler、eager、APC、chunked prefill，无 speculative/KV connector。其他配置明确拒绝。

资源来源为实际 scheduler/KV manager：`max_num_scheduled_tokens` 是有效每步预算，配置中的 `max_num_batched_tokens` 另记；物理容量为 block pool 的非 null 块数。`common.configuration` 将资源传至快照、候选、资源投影、评分、准入和报告。客户端在途上限独立为 2，服务序列上限为 10。

`native_algorithms.py` 在隔离模块中对冻结 V5/Ayo/RR/SLL 和评分器进行精确锚点变换；记录 `transformation_manifest()`，记录原始来源。V5/Ayo 的 4096 限制变为有效步预算；Ayo 保留原 conservative reserved-block 负载：已提交请求的完整 prompt+max_output 预算，加本轮拟放置请求；不扣预期命中，不计可回收缓存。分母为引擎真实 KV 容量，physical active 块仅另行观测。原算法候选数、收益门槛、decode guard、目标函数和时间窗口不放宽。

`native_scoring.py` 只替换依赖引擎语义的批次构造：真实 running 顺序后接 waiting；APC 跳过发生在 token 预算扣除之前；FullAttention 的 chunk 可按 token 切分，不能把 128-token KV 块大小错误当作 chunk 必须整除的单位。异步待完成批次按 batch ID 保留，冻结模型串行预测完成时间份额，这是近似而非保守上界，不冒充设备已执行时间。评分的排序先投影到服务端可执行的安全槽位。

输入构造仍沿用冻结协议的 16-token 命名空间检查，保持原始 tokens/DAG；这个检查不是物理 KV 块大小。物理分配、共享前缀证据长度与资源投影按 128 对齐。服务模型中 `/16`、`/1024` 等特征缩放是回归量纲，也不是物理块大小。

## 2. 原生状态、缓存与动作

有限窗口排序、Ayo 保守 KV 预留负载保持师兄确认的口径。原引擎 host-step elapsed 在异步路径不能直接等价迁移，新版标定使用明确声明的完成时间份额近似，见 [异步标定协议](native-async-calibration.md)。原始依据分别为 `joint18v2/common.py` 的 `resolve_control/stable_prefill_reorder`、`runtime/joint18v2_hook.py` 的 `reorder/schedule/begin_step/end_step`、`wfsched/ayo_topology.py` 与 `joint18v2/service_fit.py`。这些冻结文件没有修改。

调用链：HTTP → 现有 `engine_core.call_utility_async` → EngineCore 主循环 → 原 scheduler / KV manager / coordinator / block pool。设备计时经原 worker collective RPC 返回。HTTP 进程没有模拟 KV 或影子队列。

每份快照带引擎实例、前一实例、主机 boot ID、PID、快照序号、引擎 step、采集时刻与 120 ms 有效期。running/waiting/skipped_waiting、请求阶段、已排程与已完成 token、占用、可回收缓存、allocator free 块和未来增长预留分别保存。free 包含可回收缓存；不能把两者相加当成额外容量。快照在控制器使用往返耗时做保守新鲜度界限，固定动作窗口另要求控制器和引擎的 boot ID、Linux time namespace inode 相同，核验后共用 CLOCK_MONOTONIC 截止时间。

逐前缀查询传实际 token IDs 与 cache salt，使用引擎自己的 block hasher/coordinator 和“完整命中仍重算最后一个 token”规则。返回 hit / miss / unknown；未完成异步写入涉及的块返回 unknown、tokens=null。每次重新查询真实 block pool；应用动作前重查具体证据，驱逐、重启或到期不能沿用历史驻留。

动作有唯一 ID、原快照、资源摘要和截止时间。EngineCore utility loop 位于 schedule 调用之间，执行时再次检查队列成员、目标状态/进度、缓存、资源和有效期。允许不相关的 step 推进，但目标已经开始 prefill 或进度发生变化时拒绝。重复内容返回同一回执；重复 ID 携带不同内容拒绝。实际队列对象仅重排可修改的等待/未在执行的部分 prefill 槽位，decode 和在途槽位不动。

requested → accepted → applied/rejected/expired 有独立时间记录；部分动作记录 partially_applied 和逐项拒绝，随后记录实际 schedule step/order。新请求先取得物理容量准入许可，再通过 AsyncLLM 提交，EngineCore 验证输入摘要、长度和许可，保存 node/client/server/engine ID 映射。路由只发生在未提交请求。

120 ms 是快照/动作提交的新鲜度上限，不是算法优先级窗口。控制器按原固定 200 ms 周期提供 `not_before_ns/action_expires_ns`；每次原生 schedule 前重新执行安全槽位排序。到固定截止后停止使用旧 order，在下一次迭代按原始 EngineCore 准入时间恢复 Prefill FIFO；5 秒年龄保护放弃优先级重排，Decode/尚有在途执行的槽位保持原生顺序。未执行的许可有单独 2 秒交付租期并占用保守资源预留，不延长队列窗口，也不撤销已执行 token。评分器使用相同有限窗口、年龄保护与 continuation。

评分过时后的回退只允许基线的新请求准入，由引擎原子读取新状态并校验；不会把过期队列重排换一个快照 ID 重发。许可应用与请求真正进入引擎分开记录，HTTP 200 不计作引擎应用。

## 3. 标定、验收与证据边界

旧版只接受同一次调用提交并完成同一 batch，现场 8896 条记录因此没有合格样本。新版保留原始 host call 计时，并用 `async_host_completion_accounting_v1` 重建不重叠完成时间份额；以副本/call ID 去重，校验 pending 与前一完成边界。它是新的吞吐量近似模型，不宣称等价于原 GPU host-step。完整定义、验证方法和限制见 [异步标定协议](native-async-calibration.md)。

NPUWorker 在已核对的 `_model_forward` 当前流前后记录两个 `torch.npu.Event`。只有 end.query() 完成才读取 elapsed_time；无 synchronize 强行串行化。host schedule、host dispatch、设备 forward、EngineCore 输出观察、HTTP 时延分开。forward 区间不包括 sampling/D2H/主机排队，也不宣称是整个 batch 的端到端时延。事件和查询开销保留在实验中。

独立标定预先固定 train/holdout 运行，覆盖 128/129/1024/2048/3584 输入，16/32/48/64/128 输出，并发 1/2/4 与服务上限附近负载、冷/热和实际 mixed 批次。按 replica/phase 拟合原六特征非负模型。每分组至少 20 个 train 和 20 个 holdout 批次；P95 相对误差≤35%，RMSE/均值≤25%，并检查实际 token/context 覆盖。失败记录保留，不利用正式结果反向调参。

实际输入 token、APC 跳过量和逻辑 prefill 量分别记录。六特征以实际输入量（等价于原 logical minus skip）和缓存跳过前的 computed context 构造；回归缩放保持原实现。现场新标定不载入 NVIDIA 系数，预先声明独立训练/验证和误差门槛；这不同于原包已有四块旧 pilot 的迁移系数选择流程，报告保留该差异。预测超过已冻结的 phase 范围时，候选不具备可评分资格，回退原因保留。

`native-probe-one/four` 真正发送请求、查询缓存、制造 waiting、重排队列、发送迟到/重复动作，并运行 fork/join 和排空检查。队列动作触发已按 2026-10-16 现场修正：等待真实短 decode 输出达到 14/16 token 且仍有至少两条可变 waiting；窗口不延长，窗口外执行不算成功。单副本先检查本地机制；四副本再检查跨副本命中/未命中。`native-invalidate` 明确清空 0 号 prefix cache 以验证失效；它不是普通检查，也不宣称已测完所有内存压力下的驱逐模式。`native-restart-arm/check` 围绕操作者手动重启检验旧身份拒绝。

`mechanism_gate` 根据原始引擎记录逐项重算，不接受手填 PASS；校准样本必须与真实 batch ID/事件值一致、分组无重复，重新拟合与冻结系数相同。正式实验前重读 EngineCore 能力、实例、资源和实际动作回执，核对应用状态、顺序、step 及请求内容；不存在强制通过开关。不要求 Bearer Token、密钥文件或 HMAC。代码与补丁哈希只记录，不参与放行。普通验收要求单副本、四副本、独立标定原始记录；重启和主动清空缓存属于可选专项，未运行标记 not_run，不冒充通过。

本地没有 NPU。CPU 故障注入、源码锚点检查和原包回归只能说明这些代码路径可执行；**真实进程通信、NPU Event 的现场支持、真实 waiting/队列回执、KV 命中/失效、标定误差和正式 96 条件仍需在现场验证**。缺少实际接口时输出具体缺项；源码哈希差异本身不阻止安装或运行。

## 4. 命令、过程和结果

日常入口见根目录 README：`bash run_native.sh test` 测试并验收，`bash run_native.sh experiment` 运行全部 96 条件。首次部署和单步手敲顺序见 [逐步操作](native-operations.md)，部署集中为 `bash run_native.sh install`，无需旧 prepare。

保留现有 96 条件、四方法、输入/DAG/种子/执行顺序；每条件 64 工作流、216 请求、774144 输入 token、13184 输出 token。原生结果写入新 `r234_native_*`；仅正常暂停且实验协议、有效配置、引擎身份及冻结标定保持一致才续跑；源码指纹差异单独记录。新公共执行器的结果不能和旧 HTTP 基线拼接。

请求、快照、具体缓存查询、动作、批次、评分失败和回退全落普通文件。设备计时记录导出并持久化后，控制器才能确认释放引擎 trace 缓冲；不将历史完成记录转成缓存驻留。未观测字段保持 null。共享硬件的干扰、零偏离、负收益和失败原样保留；不以 Ours 更快作为验收条件。
