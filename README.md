# DABS 昇腾复现

## 保留预测偏差先跑实验（2026-10-16 新入口）

现场最新单卡完整标定已有 **2681 条有效记录、0 条排除**；decode/mixed 留出误差通过，prefill 未通过（p95 相对误差约 45.46%，RMSE/均值约 36.39%）。按此次实验要求，可以先保留此偏差看效果：

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

仅暂不要求**预测误差阈值**通过；原始计时关联、独立 train/holdout、样本数、覆盖、资源、DAG、动作期限、缓存及回执检查不变。若只有副本 0 标定，四卡在资源配置一致时共用该模型，记录 `model_transfer_from_replica_0: true`；不会伪造其他卡的标定指标。仍缺新接口的副本先运行 `bash run_native.sh install`，安装器保留备份并跳过已经可用的副本。

成功结束是 `outputs/native_explore_*/r234_exploratory/status.json` 显示所选数量的 `completed_conditions`；少于 96 条件时为 `paused`（正常暂停），全部 96 条件结束时为 `complete`。查看 `TEST_REPORT.md`、`metrics.csv`、`paired_comparisons.csv` 和 `diagnostic_reasons.csv`。结果明确标记 `exploratory: true`、`formal_acceptance: false`，不能作为正式原生验收，也不会放行下面的 `experiment` 命令。预测修复另行验证，旧日志和结果保持不变。

## 本次标定修复：先单卡验证，再跑四卡

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

**计时口径发生变化**：新标定是 `async_host_completion_accounting_v1`，按完成顺序计算不重叠的主机时间份额，包含首批启动时间，作为吞吐量近似模型。它不是旧 GPU 的同 batch host-step 时间，也不是 NPU forward 或 HTTP 时间；`host_step_equivalence` 正确保持 `false`。验收从原始事件重新计算，不能改 PASS 或 equivalent 标志放行。具体定义、限制和差异见 [异步标定协议](docs/native-async-calibration.md)。本地无 NPU，尚不能宣称新口径已在现场通过。

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
