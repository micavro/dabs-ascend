# 1608 现场：刷新快照过期的控制器修复

依据：用户复制的 `r234_native_pause8_20261016_192029` 错误、console/status、最近决策和 RPC 摘要，以及 protocol.json。未取得完整逐轮原始事件，未在本机访问 NPU。保留现场失败结果，不修改 status 或重新解释为成功。

## 确定的原因与证据边界

- 现场 runner.py、repair.py、native_client.py 三个 SHA256 与 Git 提交 `87deda2` 完全一致；不是用户漏更新了这三个文件。哈希仅用于定位版本。
- 条件 1 为 RR。进度曾显示 submitted=12、completed=8、inflight=4；最后 `completed_conditions=0`。
- traceback 指向 runner 的第一条轻量回退路径：observe(cache_queries=False) → baseline → validate_action → admission_check → `raise ValueError('Fresh native fallback admission failed: stale_snapshot')`。另一条“引擎拒绝后再刷新”分支有相同缺陷。
- `stale_snapshot` 来自控制器的 `now - snapshot_at > 0.12`（或时钟倒退）判断，并非 NPU OOM、HTTP 错误或预测误差门限。现场材料没有时钟倒退证据。
- 最后四个零查询 RPC 约 8.7–17.8 ms，HTTP 响应记录时的剩余有效期约 116.3–117.6 ms。`client_finish` 是响应日志写入前的时间；不包含其后的日志 I/O、状态投影、基线生成和校验。这些时间不能被当成完整控制开销。
- 旧 `native_observe_ms` 只计 RPC gather 段，不计返回之前的状态整理；`state['now']` 则在整理之前取得。于是 RPC 看起来很快，最终检查仍可过期。旧代码把“刚刷新”误当成“不可能再过期”，直接停止实验。

附件没有最后失败动作的分段耗时，不能准确认定现场的 120 ms 全花在哪个函数。CPU 剖析确认一个实际热点：216 节点在 RPC 后重复计算具体 token 前缀的 JSON/SHA256。用真实 216 节点工作流构造、伪造空闲服务响应的本机样例，修改前热状态 observe 约 42–45 ms，修改后约 13–16 ms；首次派生 RR 模块加载另约 97–131 ms。这些是开发机诊断，不是 NPU 性能结论，不用于实验模型拟合。

## 实现

1. 前缀身份在 snapshot RPC 之前按具体 token 去重计算。每轮从实际 token 和 namespace 重新生成；不借此推断驻留、不缓存命中结论、不截断候选集。客户端状态仍在 RPC 后读取，保留等待 RPC 期间发生的真实完成/到达事件。
2. RR/SLL/Ayo 派生模块加载放在 condition 的计时开始前；算法及工作流到达间隔不变。
3. 两条原生回退分支共用 `fresh_native_admission`。刷新后超过原 120 ms 则记录 snapshot/action、实际年龄、拒绝原因和 `native_deferred_reason=stale_snapshot`；当轮不发该准入 RPC，不发该 HTTP 推理，节点仍为 ready。下个固定周期重新观察和决策。没有重发已提交请求，没有刷新旧排序的有效期。
4. 非时间性错误仍然失败，例如 dependency_not_complete；结构、资源、DAG、重复执行及实际引擎许可检查保留。持续不能提交仍由原 60 秒无在途/无进展检查失败退出；条件墙钟和请求超时仍有效。
5. 完整 observe 拆成 `native_prepare_ms`、`native_rpc_ms`、`native_projection_ms`。投影结束已经过期的缓存证据变为 unknown/tokens=null，原始引擎记录仍保存。计时不会通过重写 snapshot.now 伪装新鲜。
6. 决策记录包含 `fallback_validation`，进度增加 deferred，汇总增加 `native_deferred_admissions`，拒绝原因进入 diagnostic_reasons.csv。延后的轮次不能算作引擎已应用或优化成功。

## 与复现修复包和师兄说明的对应

已阅读 `复现修复.zip` 中 FastRepair 方案、原实现摘录和 admission_order_counterexample.json。

| 要求 | 本次处理 |
|---|---|
| 200 ms 固定周期，窗口结束回到原始 FIFO，Decode 保护、5 秒年龄保护 | 不改 engine/contracts 或窗口参数；过期候选不应用。 |
| Ayo 按保守 KV 预留量/实际容量选副本 | 不改 reserved_blocks、真实物理容量或 Ayo 算法；不以实时占用替换预留量。 |
| 原 GPU engine_host_ms，不以 HTTP/NPU forward 时间冒充 | 保留已有 `async_host_completion_accounting_v1` 主机完成区间分摊近似和 `original_gpu_timing_equivalent=false`。现场异步 submit B/complete A 无法直接等同原同批 host step；该差异继续明示。 |
| 陈旧状态安全回退、原始失败保留、先 8 条后续跑 | 本次修复对应这一条。只延后未提交动作；失败条件不续写，重新从新目录运行。 |
| 不扩大搜索、不靠收益放行、不拼新旧四方法结果 | 搜索、门限、96 条件、四方法、种子及工作量不变。 |

用户允许优先采用容易运行的方式，因此继续使用已有、明示误差范围的异步主机模型；不为追求原 GPU 计时名称而重新引入永远采不到样本的门禁。这次不改标定目标、系数或验收标准，也不用此次失败实验调模型。

## CPU 回归与现场入口

修复前，新测试实际触发同一个 stale_snapshot 异常：第一/第二回退分支、四方法以及持续过期场景。修复后检查四方法均完成 216 请求/13184 输出、无重复提交及依赖越序；持续过期必须失败并留记录；依赖错误不能被吞掉。另测具体前缀去重、namespace/input 改变、投影耗时导致证据过期。

本次未改 `dabs_native` 服务包或冻结 source。1608 服务保持原引擎身份时，不需要安装、重启或重跑标定。新目录可以复制本机旧测试的普通文本原始证据并重新核验；README 提供全套一行入口、先 8/12 条及 resume 的命令。旧失败目录不得标成 paused。

现场待验证：RR 计数持续推进到 216/216；首 8 条完成且配对工作量一致；记录 deferred/回退比例、候选评分失败及真实动作回执，之后正常续跑剩余条件。若仍持续过期，新增分段时间能区分准备、RPC/日志返回、投影和基线开销；不能用 CPU 测试声称现场已经跑通。
