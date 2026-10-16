# DABS 昇腾复现

## 现场只需这三条命令

从 GitHub **main** 下载到新目录，进入目录，在 **Linux 宿主机**执行：

```bash
bash run_native.sh install
```

自动连接现有四个服务。原生接口可用的副本直接复用；缺少接口、或仍使用旧认证接口的副本，自动备份、安装/更新补丁、逐个重启并等待就绪。只有安装需要加载补丁时才重启；请在这些服务没有其它任务时执行。**不需要生成或复制任何密钥，不依赖旧项目目录，版本字符串和代码哈希不同不会单独阻止运行。** 不创建新模型服务、不升级推理栈、不修改服务资源参数。

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

2026-10-16 的现场反馈：单卡探针首次因优先动作发出过早失败；等待短请求输出达到 14/16 token 后，重跑摘要为 19/19 通过、`live_verified=false`。此触发修正已纳入代码，保留固定窗口和真实执行顺序检查。该结果来自用户提供的现场记录，第二次逐批原始日志尚未在本地核对，**不能代替四卡、标定及完整原生验收**。当前移除了旧 `source_identity` 哈希通过项，剩余机制检查没有放宽。

重启/主动清空缓存测试是可选的独立专项，未执行会在验收中标记 `not_run`，不伪报通过。普通验收仍检查快照过期、实例改变、应用时缓存重查、重复动作和安全队列边界。详见 [部署与专项命令](docs/native-operations.md)、[机制与验收说明](docs/native-v023.md)。

本地 CPU 自检：`python tools/run_ascend.py test`。CPU 通过不代表 NPU 验收通过；如果现场异步 Host-step 无法提供足够的同口径标定样本，会明确停止并保留原始记录，不能用 HTTP 或 forward 时间替代。
