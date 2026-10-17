# DABS 昇腾复现

## 当前版本：只补诊断与三组筛查，不优化调度机制

本次只增加诊断字段、受限复制摘要和预定筛查入口。**不修改候选生成次序/上限、`seen` 去重位置、准入与资源约束、查询优先级、观测前处理算法、评分时间分配、评分公式或模型门槛。** 固定周期、快照期限、提交预留、搜索上限与客户端并发保持原值。计时插桩本身有小额开销，不能将本版速度变化直接归因于算法优化。

先确认上一轮实验控制器已经退出。在更新后的项目目录运行下面一行，复用这次 verify 中已经通过的新标定（旧目录只读；不自动重启或重装）：

```bash
bash run_native.sh screen --input ../dabs-ascend-main-2026101611/outputs/native_verify_20261016T135710Z_5f1e88ab
```

默认依次运行 **R3_L8、R4_L4、R3_L4**，每组四方法，共 **12 条 T1 条件**。也可单独运行一组：

```bash
bash run_native.sh screen --group r3-l8 --input ../dabs-ascend-main-2026101611/outputs/native_verify_20261016T135710Z_5f1e88ab
bash run_native.sh screen --group r4-l4 --input ../dabs-ascend-main-2026101611/outputs/native_verify_20261016T135710Z_5f1e88ab
bash run_native.sh screen --group r3-l4 --input ../dabs-ascend-main-2026101611/outputs/native_verify_20261016T135710Z_5f1e88ab
```

每条件仍为 64 工作流、216 请求、3584 输入 token/请求、13184 输出 token；保留原计划内的种子、到达、方法顺序与资源限制。R3_L8 为 SLL/Ayo/Ours/RR，另两组为 Ayo/Ours/RR/SLL。不开更大并发、不换数据、不强制制造非参考动作。这是定位筛查，不是预测哪组必胜。

若在同一项目已运行 verify，可省略 `--input`，自动使用最近 verify 的原始证据；没有 verify 才使用最近通过的 test。即使旧 verify 因规划未过而 `blocked`，其独立机制/标定原始文件仍可导入，**不继承旧 PASS**，原来的原始证据和现场服务检查照常执行。覆盖已匹配则不重标定；若覆盖过旧，沿用 verify 的独立补标定流程；服务实例或资源变化仍停止。

完成后看两个独立字段：`screening_completed=true` 表示所选工作量、配对和 DAG 检查完成；`planning_pass=true` 表示每个 Ours 条件至少存在一次时限内完整参考评分 + 不同可执行候选的完整评分。不要求选非参考动作或 JCT 更快。无候选/评分超时会保留并继续后面的筛查组；推理、工作量、原生门禁失败仍停止。入口状态为 `completed`，绝不是正式矩阵的 `passed`。筛查不续接、不拼入 96 条矩阵；原 verify/experiment 默认行为不变。

运行中或结束后复制普通文本：

```bash
bash run_native.sh logs
```

`outputs/native_screen_*/copy_logs/COPY.txt` 为 v3：每条件 D 最多 6 条（同类 3 条），S 按部分评分/完整比较/生成后未评分各最多 1 条，SELECT 最多 2 条，回执最多 2 条；中途详情总预算 12 KiB（普通进度/原因共用其中最多 6 KiB，给稀有评分样例留空间），结尾 TIMING/DIAG/C 汇总另留。计数不受采样限制，省略条数显式记录。常规三组筛查目标是几十 KiB，实际大小取决于进度和检查输出；不是对包含补标定日志的整个文件作硬上限承诺。

新增诊断解释见 [诊断字段与判读](docs/native-screening-diagnostics.md)。所有决策仍完整留在 `conditions/*/decisions.jsonl`，不把 prompt/token 列表或全份回执复制到摘要。原始证据不上传、不打包，不新增源码哈希真实性门禁。

本地检查（CPU，不发推理）：

```powershell
python tools/run_ascend.py test
python tools/run_dabs_existing.py plan --suite r234 --devices 0,1,2,3 --screening
```

下面保留上一版操作用于追溯；本次优先运行上面的 `screen` 定位问题，再决定机制修改。

## 上一版：观测、评分、覆盖与控制开销修复

1610 前 8 条已经完成工作量，但还没有完成 Ours 的候选比较。日志将问题区分为：部分决策没有有效规划预算；获得预算后仍可能没有不同且可执行的候选；少数评分被 `p:1/3`、`pc:3199/3200` 等模型范围检查中止。后一类还曾被误归为观测失败，影响 RR/SLL/Ayo。不能把这些交叉计数相加，也不能把所有无候选都解释为故障。

这次将真实观测和预测可用性分开；缓存查询优先覆盖可能准入、关键路径靠前的请求，也实际查询不足 128 token 的共享前缀；优化状态整理、原评分器的等价计算及日志落盘，历史批次采集移到决策窗口前。固定 200 ms 周期、120 ms 快照期限、25 ms 提交预留、80 ms 搜索上限均保留。完整 DAG、候选约束、物理资源、动作回执、误差和验收门槛不降低。详见 [实现与证据边界](docs/native-pipeline-repair.md)。

**不需要重新安装或重启四个服务。需要新标定，下面的验证命令会自动完成。** 前提是 1608 安装、测试通过之后服务未重启且参数未变；适配器和安装脚本未改。旧模型缺少分阶段尾块/部分缓存覆盖，不能修改其范围冒充合格。新采样使用独立训练/留出种子，不使用正式实验结果拟合。请确保旧实验控制器已经退出。

下载 main 到新的同级目录（例如 `/data/wushibo/dabs-ascend-main-2026101611`），进入该新目录，一行验证：

```bash
bash run_native.sh verify --input ../dabs-ascend-main-2026101608/outputs/native_test_20261016T110057Z_89891d17
```

此命令保留旧目录，在新输出目录只读导入原始机制证据，检查现有服务；旧模型缺少新覆盖时，自动运行四副本新标定，再复核证据并按原顺序跑前 **8 条完整条件**。不重跑环境采集、不清空缓存、不重装、不重启。新标定每副本为 25 组训练、25 组留出，时间会比上一版 verify 更长；终端及集中摘要会持续记录进度。样本不足、范围或误差失败时会停止，不能靠旧 PASS 放行。

成功判据：总状态 `passed`，`planning_audit.json` 的 `pass: true`；标定逐副本逐阶段通过，前 8 条工作量/DAG 通过，Ours 在时限内完成参考与不同可执行候选的完整评分。**不要求 Ours 选择非参考动作或更快。** 子实验状态 `paused` 是正常暂停。可加 `--max-conditions 12`（4..92 的 4 的倍数）。每条件仍为 64 工作流、216 请求、每请求 3584 输入 token、总计 13184 输出 token。

验证通过后，一行继续同一新实验到 **96/96**，前 8 条不重跑：

```bash
bash run_native.sh experiment
```

续跑重新核对原始证据、冻结模型、服务身份和资源。旧 1609/1610 实验不能续接本次修改后的规划协议；保留原结果。服务重启或参数变化会使旧机制证据失效，命令会在长标定前指出，而不是自动重启或继续实验。安装脚本 `bash run_native.sh install` 仍保留，当前已安装现场无需执行。

运行中或结束后，一行打印集中摘要：

```bash
bash run_native.sh logs
```

摘要在本次 `outputs/native_verify_*/copy_logs/COPY.txt` 持续追加，续跑沿用同一文件。保留阶段、约 30 秒进度、候选/完整评分/回退计数、有限回执样例、分段耗时 p50/p95 和结果；同类原因最多 3 个样例。移除重复载荷和空白，原始 JSON/JSONL/CSV 保留；`decisions.jsonl` 使用快照 ID 关联 `native_events.jsonl`，不重复整份 RPC。原始批次持久落盘后才向引擎确认已采集，磁盘失败会停止。只复制普通文字，不打包、不上传。

普通全套测试仍为 `bash run_native.sh test`（重新探针和标定）；本次复用原始机制证据使用上面的 `verify`。本地开发检查：`python tools/run_ascend.py test`。CPU/Linux 测试不能证明 NPU 性能或现场已完成评分，最终看新运行的原始记录。

## 以下为历史版本记录

以下说明和命令用于追溯当时的实现，不作为当前启动步骤；当前操作以上节为准。

## 上一版 e82855e：刷新后的 stale_snapshot 不再终止实验

现场 `r234_native_pause8_20261016_192029` 的首个 RR 报 `Fresh native fallback admission failed: stale_snapshot`。附件中的 runner/repair/native_client 三个文件哈希与 `87deda2` 一致，这是该版控制器遗漏的分支：快照 HTTP 返回快，不代表经过本地状态整理和基线计算后仍未过期。详见 [故障与修复记录](docs/native-refresh-repair.md)。

本次将具体前缀身份的计算、去重移到快照请求前，基线模块初始化移到计时前。两条回退路径若刷新后仍超出原 120 ms 新鲜度，记录 `native_deferred_reason=stale_snapshot`，该轮不申请许可、不提交推理，下个固定周期重新决策。200 ms 优先级窗口、DAG、资源、去重和引擎回执检查不变；持续无进展仍由原 60 秒空转/条件超时检查失败退出。结果新增 `native_prepare_ms/native_rpc_ms/native_projection_ms` 和延后次数；旧 `observe_ms` 只覆盖 RPC，新值包含完整 observe 调用，不能直接当成性能提升对比。

**已经在 1608 目录完成安装、test 通过且四服务没有重启：本次仅更新控制端，不需要再安装、重启或重新标定。** 在新下载目录复用该次原始测试证据，重新验证当前服务后跑全套：

```bash
bash run_native.sh experiment --input /data/wushibo/dabs-ascend-main-2026101608/outputs/native_test_20261016T110057Z_89891d17
```

若按修复包先跑 8 条再续跑，在新目录先复制普通证据文件（本机复制，不打包、不上传）：

```bash
mkdir -p outputs && cp -a /data/wushibo/dabs-ascend-main-2026101608/outputs/native_test_20261016T110057Z_89891d17 outputs/
bash run_dabs_r234.sh benchmark --native-evidence outputs/native_test_20261016T110057Z_89891d17/evidence.json --max-conditions 8 --out outputs/native_retry8
```

`outputs/native_retry8` 必须是新目录。正常完成显示 `paused`、`completed_conditions=8`，然后继续剩余 88 条：

```bash
bash run_dabs_r234.sh benchmark --native-evidence outputs/native_test_20261016T110057Z_89891d17/evidence.json --resume outputs/native_retry8
```

可将首次的 `8` 改成 `12`；续跑不带 `--max-conditions` 会运行剩余条件。保持服务、参数、代码和冻结模型不变；只续跑正常暂停结果。先前 failed 目录保留，不能改成 paused 或拼入本次结果。测试证据会重新核验原始事件和当前引擎身份；不靠旧 PASS 放行。现场先确认 RR 持续完成到 216/216，再检查配对条件的工作量及结果。

## 上一版 87deda2：边界补修涉及服务端

上一版 `4131f17` 已修复慢观察导致 RR 不提交请求的问题；再次审查确认并补修了服务端边界缺陷：只准入回退可能覆盖当前优先级；缓存/资源校验耗时后未重查动作截止时间；尚未到 `not_before_ns` 的动作也可能提前报告已应用。另补齐了响应到节点、客户端/服务端请求、引擎请求及准入动作的对应核对。详见 [审查记录](docs/native-boundary-review.md)。

**尚未加载 87deda2 的边界修复时，需要更新服务端适配器。** 已完成这一步的 1608 现场按上节复用服务；首次部署在新目录、Linux 宿主机执行这一行：

```bash
bash run_native.sh install && bash run_native.sh test && bash run_native.sh experiment
```

先确认旧实验控制器已退出。安装器通过 `docker exec` 在运行的容器中备份和安装，再逐个关闭/重启 `wushibo-qwen-small-npu0..3` 并检查就绪；**不要预先手动把四个容器全部停止**。保留容器原配置、模型路径、端口和资源参数，不创建新模型服务，不动其他程序。每个安装目录有 `ROLLBACK.txt`。缺少实际准入/窗口修复能力的旧副本会更新，已加载所需能力的副本跳过。原始源码哈希不是运行门禁。

可分步先验证单副本：`bash run_native.sh install --replica 0`，然后 `bash run_dabs_r234.sh native-probe-one`；成功后执行上面的一行命令处理其余副本。测试不自动重启或清缓存；任一步失败，`&&` 会停止后续实验。服务重启后身份变化，必须生成本次新测试/标定证据，**不要用旧 `native_test` 的 PASS 代替**。旧目录、失败日志和实验结果均保留。

验收新增检查从原始回执核对“准入未改变队列/已有控制”和“实际修改时刻在动作窗口内”，不能靠接口存在或手填 PASS 放行。CPU 测试不是现场验证；现场先看探针及验收结果，再看首个 RR 的完成数持续增加到 216/216。

## 上一版修复：首个 RR 长时间不推进

现场 `native_test_20261016T093432Z_7785a486` 已通过，但随后的正式实验首个 RR 出现 **951 次决策、949 次过期、仅 8 个请求完成**。这是控制器问题：它每轮向每个副本发送 96 条完整输入共 344064 token 做缓存查询，观察约 670–716 ms，超过固定 200 ms 决策周期；过期后直接跳过，又没有进入原子准入回退。RR 不调用预测搜索，因此这次空转不能归因于预测误差。

`4131f17` 当时只改控制端；本次进一步发现的服务端问题按上节安装处理。下面的跨目录证据复用入口仍保留，但只适用于**引擎未重启且原始证据满足当前检查**的情况；本次更新旧服务后不能使用该旧测试目录放行：

```bash
bash run_native.sh experiment --input /data/wushibo/dabs-ascend-main-2026101607/outputs/native_test_20261016T093432Z_7785a486
```

此命令将旧测试的普通原始文件复制到本次新输出目录，重新计算验收并核对当前引擎身份、配置及回执后启动；不信手填 PASS，不重新采样、不清缓存，不接着失败条件续写。原目录和结果保留。若服务重启导致原证据失效，会明确报错；不能强行使用旧证据。在同一代码目录已完成测试时仍可用 `bash run_native.sh experiment`。先确认上一次控制器已经退出，避免同时跑两次实验。

每约 5 秒（控制循环可运行时）输出 `R234_PROGRESS`，包含提交数、完成数、在途数、观察耗时、过期数和回退数；同时更新每条件 `status.json`。先看 `completed` 持续上升，首个 RR 最终应为 **216/216**，`summary.json` 的工作量和依赖检查通过。全套成功仍要求 96 条件完整结束。若有就绪请求、没有任何在途请求，且 60 秒没有新提交/完成，明确失败并保留原因，避免继续静默空转。

修复保持 200 ms 有限窗口、120 ms 快照新鲜度和原工作量：缓存查询只发送评分需要的实际共享前缀完整块及一个真实后继 token，同轮去重，默认最多 8 个不同前缀、8192 token/副本。所有方法共用此观察规则；未查询或过期的数据标为未知，不伪造未命中，不裁掉候选请求。观察超时后丢弃旧优先级，读取轻量新状态，重新生成基线准入；由 EngineCore 原子校验资源并返回真实准入回执，才提交 HTTP 请求，不对已提交队列应用过期动作。

`metrics.csv` 的 `native_atomic_fallback_decisions` 和 `diagnostic_reasons.csv` 的 `native_fallback_reasons` 区分回退与选中动作生效。若大量回退，结果仍保留，但不能把“跑完”解释为优化有效。CPU 回归包含四种方法在观察超过窗口时完成各自 216 请求；现场性能、实际回退比例和优化效果需本次新运行确认。

## 保留预测偏差先跑实验（2026-10-16 新入口）

较早的单卡完整标定已有 **2681 条有效记录、0 条排除**；decode/mixed 留出误差通过，prefill 未通过（p95 相对误差约 45.46%，RMSE/均值约 36.39%）。这是仍需保留预测偏差试跑时的入口；已通过完整测试的现场使用上面的 `experiment`：

```bash
bash run_native.sh explore
```

默认 4 条件，可自行增加；建议 4 的倍数，方便比较同组四种方法。跑前 24 条件或全部 96 条件：

```bash
bash run_native.sh explore --max-conditions 24
bash run_native.sh explore --max-conditions 96
```

在新代码目录运行，自动寻找当前目录及同级 `dabs-ascend-main-*` 的最新完整标定，复制原始记录到新输出目录，检查现有四服务及单/四副本真实机制，再按原计划顺序运行指定数量条件（默认前 4 个：RR、SLL、Ayo、Ours）。不重新标定、不自动安装或重启、不清空缓存。每条件仍是 64 工作流、216 请求、3584 输入 token/请求、13184 总输出 token；不筛选正收益。

如果自动找到的目录不对，明确指定刚才的标定步骤目录：

```bash
bash run_native.sh explore --input /data/wushibo/dabs-ascend-main-2026101605/outputs/native_calibrate_20261016T083402Z_30da8518/01-native-calibrate-one
```

仅暂不要求**预测误差阈值**通过；原始计时关联、独立 train/holdout、样本数、覆盖、资源、DAG、动作期限、缓存及回执检查不变。若只有副本 0 标定，四卡在步预算、块大小、序列数及上下文上限一致时共用该模型（KV 容量仍逐卡使用实测值），记录 `model_transfer_from_replica_0: true`；不会伪造其他卡的标定指标。仍缺新接口的副本先运行 `bash run_native.sh install`，安装器保留备份并跳过已经可用的副本。

成功结束是 `outputs/native_explore_*/r234_exploratory/status.json` 显示所选数量的 `completed_conditions`；少于 96 条件时为 `paused`（正常暂停），全部 96 条件结束时为 `complete`。查看 `TEST_REPORT.md`、`metrics.csv`、`paired_comparisons.csv` 和 `diagnostic_reasons.csv`。结果明确标记 `exploratory: true`、`formal_acceptance: false`，不能作为正式原生验收，也不会放行下面的 `experiment` 命令。预测修复另行验证，旧日志和结果保持不变。

## 预测器修复：先离线复用刚才的数据

上述 `explore` 继续使用所选历史标定对应的预测器；不会因为新代码加入预测修复而自动替换旧模型。新标定使用 `phase6_prefill_startup_v2`：prefill 增加从实际在途状态获得的启动项，decode/mixed 保留原六特征。误差阈值不变。此改动仅在控制端，**已更新计时接口的副本 0 无需重新安装或重启**。

先比较已有 2681 条原始记录的旧/新预测误差，不发推理、不改旧文件：

```bash
bash run_native.sh refit --input /data/wushibo/dabs-ascend-main-2026101605/outputs/native_calibrate_20261016T083402Z_30da8518/01-native-calibrate-one
```

查看新输出目录下 `01-native-refit/refit_comparison.json`：`previous` 为旧预测误差，`startup_aware` 为新误差及启动/流水线分组均值。`retrospective_error_check_pass: true` 仅表示这批旧数据上误差合格；仍为 `frozen: false`，不冒充新验收。若仍失败，保留这份对照，不盲目重采。

离线对照合理后，单卡使用**新训练/留出种子**验证：

```bash
bash run_native.sh calibrate
```

这仍是 28 组完整标定，只有三阶段误差、样本数和覆盖均通过才冻结。正式实验种子、96 条件和工作量没有变；新标定不会使用试跑实验结果拟合。本地 CPU 测试通过不等于新的现场误差已经通过，完整数据仍在服务器，需现场运行上述离线对照验证。

## 历史计时关联修复：尚未更新旧适配器时

从 GitHub main 下载到**新目录**，进入目录，在 Linux 宿主机按顺序执行。旧项目和失败输出都保留。

先只读分析已有 8896 条记录（不发推理、不安装、不重启）：

```bash
bash run_native.sh audit --input /data/wushibo/dabs-ascend-main-2026101603/outputs/native_test_20261016T073330Z_b2e635a7/04-native-calibrate
```

先更新一个副本，再做四组短诊断：

```bash
bash run_native.sh install --replica 0
bash run_native.sh smoke
```

`install --replica 0` 保留现有容器参数，备份并更新缺少新计时字段的适配器，只重启这个副本并等健康恢复。安装输出目录有 `ROLLBACK.txt`。旧适配器即使 HTTP 正常，也需要加载新增的在途批次和完成边界字段。没有密钥、源码哈希门禁。

`smoke` 成功判据是 `association_pass: true`、有效样本数大于零、排除记录为空；它只验证关联，不代表模型验收。**短诊断成功后**再执行单副本完整标定（28 组，独立训练/留出）：

```bash
bash run_native.sh calibrate
```

成功判据：单副本标定状态 `passed`，`native_profile.json` 为 `pass: true`、`frozen: true`，三类阶段的 train/holdout 均至少 20 条且覆盖和误差检查通过。失败就停在这一步，复制 `SUMMARY.txt`、`calibration_progress.json` 或 `native_profile.json` 的失败项，**不要直接重跑四卡长测试**。也可对单个其它副本使用 `--replica 1`、`2`、`3`。

单卡完整标定通过后，再更新其余服务并测试：

```bash
bash run_native.sh install
bash run_native.sh test --invalidate-cache
```

安装逐个处理缺少计时能力的副本；已加载新字段的副本跳过。测试不自动重启；`--invalidate-cache` 明确允许最后执行副本 0 缓存失效专项。普通测试仍可使用 `bash run_native.sh test`。完整验收通过以后才运行 `bash run_native.sh experiment`。

**计时口径发生变化**：新标定是 `async_host_completion_accounting_v1`，按完成顺序计算不重叠的主机时间份额，包含首批启动时间，作为吞吐量近似模型。它不是旧 GPU 的同 batch host-step 时间，也不是 NPU forward 或 HTTP 时间；`host_step_equivalence` 正确保持 `false`。验收从原始事件重新计算，不能改 PASS 或 equivalent 标志放行。具体定义、限制和差异见 [异步标定协议](docs/native-async-calibration.md)。现场已证明计时关联可用；新的预测器误差与完整验收仍待现场验证。

## 日常测试和实验

从 GitHub **main** 下载到新目录，进入目录，在 **Linux 宿主机**执行：

```bash
bash run_native.sh install
```

自动连接现有四个服务。原生接口和所需计时字段可用的副本直接复用；缺少接口、计时字段或仍使用旧认证接口的副本，自动备份、安装/更新补丁、逐个重启并等待就绪。只有安装需要加载补丁时才重启；请在这些服务没有其它任务时执行。**不需要生成或复制任何密钥，不依赖旧项目目录，版本字符串和代码哈希不同不会单独阻止运行。** 不创建新模型服务、不升级推理栈、不修改服务资源参数。

一行启动测试：

```bash
bash run_native.sh test
```

依次执行四服务能力检查、单副本探针、四副本探针、独立标定和原始记录验收。普通测试不安装、不重启、不清空缓存。检查真实接口、状态、资源、动作回执和执行批次；缺项会显示具体原因。成功判据：终端测试状态 `passed`，验收 `pass: true`、`live_verified: true`、`missing: []`。

测试通过后，一行启动全部实验：

```bash
bash run_native.sh experiment
```

使用这次测试证据，重新核对当前引擎和实际回执，运行 **全部 96 条件**。成功判据：实验状态 `passed`，`r234_native/status.json` 为 `complete`、`completed_conditions: 96`、`functional_test_pass: true`。RR／SLL／Ayo／Ours、种子、DAG 和执行顺序不变；每条件 64 工作流、216 请求、每请求输入 3584 token、总输出 13184 token。原生结果写入新目录，不能接着旧 HTTP 结果续跑。

已加载新版原生接口后，日常只需 `test`、`experiment`。仅修改控制端探针不需要重启。安装中某个副本失败会停止并保留已完成副本、备份和日志，修复后重试同一条安装命令即可。

## 下载与环境

```bash
git clone -b main https://github.com/micavro/dabs-ascend.git dabs-native-current && cd dabs-native-current
```

也可以下载 main 源码 ZIP，解压到新目录。保留现场旧目录和所有结果，不需要从旧目录搬配置文件。

环境沿用现场：Linux、Bash、宿主机 Python ≥3.9、`sudo docker`；已有容器 `wushibo-qwen-small-npu0`～`3`，宿主端口 8001～8004。默认模型目录 `/data/model/Qwen2.5-1.5B-Instruct`，不同则设置 `MODEL_PATH`；容器 Python 默认 `/usr/local/python3.12.13/bin/python`，不同则设置 `CONTROLLER_PYTHON`。不要求重跑 prepare 或之前的环境采集。

## 结果与现场状态

每次运行打印新的结果目录：`outputs/native_install_*`、`outputs/native_test_*`、`outputs/native_experiment_*`。**先看目录中的 `SUMMARY.txt`**，里面有状态、失败原因、日志路径和下一条命令；`run.json` 记录实际命令和时间。安装回滚命令在 `ROLLBACK.txt`。实验报告在 `r234_native/TEST_REPORT.md`。

所有结果保存普通文本／JSON／CSV；只复制需要的文字，不打包、不上传。失败、零偏离和负收益均保留。

2026-10-16 的现场反馈：旧版本经现场修改后，单卡探针 19/19、四卡探针 20/20、重启失效检查通过；单卡优先动作触发修正 14/16 已合入。随后标定采集 8896 条，旧规则合格数 0，全部排除；缓存失效专项和最终验收尚未执行。以上来自用户提供的终端/配置摘要，本地尚未取得完整原始事件，实际提交/完成组合请运行上面的离线 audit 确认。**不能把旧版本探针成功写成新版标定或完整原生验收通过。** 当前移除了旧 `source_identity` 哈希通过项，剩余机制检查没有放宽。

重启/主动清空缓存测试是可选的独立专项，未执行会在验收中标记 `not_run`，不伪报通过。普通验收仍检查快照过期、实例改变、应用时缓存重查、重复动作和安全队列边界。详见 [部署与专项命令](docs/native-operations.md)、[机制与验收说明](docs/native-v023.md)。

本地 CPU 自检：`python tools/run_ascend.py test`。CPU 通过不代表 NPU 验收通过。新计时口径仍需满足样本数量、覆盖和留出误差；失败保留原始记录，不使用 HTTP 或 forward 时间替代。
