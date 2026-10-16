# DABS 昇腾：先采集现场，再修复机制

**2026-10-16 现场反馈更新：四个服务采集时已退出，重启后只看到 HTTP 000，尚无就绪结果。** 已补上状态、启动日志、就绪等待和配置差异记录；仍需采到 vLLM / vLLM-Ascend 版本后接原生接口。

在 **Linux 宿主机**、新版项目目录运行：

```bash
bash run_dabs_r234.sh prepare
```

这一行先检查选中的四个容器，等待运行中的服务加载，再采集版本/接口，成功后运行 **72 请求小探针**。不会运行完整 96 条件，也不会启动、重启或修改模型服务。结果写成普通文本，**不打包、不上传**。

只采集现场、不发送推理请求时：

```bash
bash run_dabs_r234.sh inspect
```

`prepare` 每轮打印四个容器的状态，健康检查默认等待 300 秒，版本/源码采集耗时另计。已退出、暂停、重启中或映射不符时直接保存诊断并退出；不会错误地对停止的容器反复执行 Python。运行中但未就绪时，等待后仍失败就保留最近启动日志，停止在推理之前。等待更久可用 `READY_TIMEOUT=600 bash run_dabs_r234.sh prepare`。

采集成功后执行 **36 组、72 个真实请求**的小探针，检查同前缀重复请求、跨副本、新命名空间，以及串行/并发 4 请求的 16/128 token 输出和混合长度；打印 `/metrics` 的运行/等待队列、缓存相关计数及请求时延。**失败时先复制 `SUMMARY.txt`；成功时再复制 `PROBE_SUMMARY.txt`。** 终端会打印两份文件的完整路径和内容。

现场有旧代码时：`DABS_PREVIOUS_PROJECT=/实际旧项目路径 bash run_dabs_r234.sh prepare`。具体运行时间取决于服务速度，不承诺固定分钟数；终端逐组显示进度。原生缓存/队列机制尚未接通，探针只为下一步诊断提供证据。

## 1. 下载与保留现场

从 [GitHub 项目](https://github.com/micavro/dabs-ascend) 下载新版。之前手工改过代码的，请解压到新目录，保留旧项目和旧 `outputs/`，不要覆盖。

若旧项目是 `/data/wushibo/dabs-ascend-main-20261015`，在新版目录用这一行同时记录旧代码：

```bash
DABS_PREVIOUS_PROJECT=/data/wushibo/dabs-ascend-main-20261015 bash run_dabs_r234.sh inspect
```

路径不同就替换。`--project` 只决定采集哪份项目源码；执行的采集工具仍是新版工具。默认记录当前项目。先采集旧代码，再更新旧工作目录，可以保留现场 `keepalive=0` 等手工修正证据。

无本地改动的 Git 目录可用 `git pull --ff-only`；ZIP 解压目录不能 `git pull`。首次下载：

```bash
git clone https://github.com/micavro/dabs-ascend.git
cd dabs-ascend
bash run_dabs_r234.sh inspect
```

## 2. 采集结果在哪里

```text
outputs/repair_probe_时间_编号/
  SUMMARY.txt             先复制这一份文字
  inspection.json         完整结构化信息，不含原始 Docker Env
  readiness.jsonl         每轮容器状态和健康检查结果
  service_logs/           未就绪服务最近启动的筛选日志，普通 .txt
  npu-smi.txt              NPU 资源情况；命令不可用时在 JSON 记录
  installed_sources/      相关已安装引擎源码，保存为 .txt
  project_source/         当前或指定旧项目的源码快照
  README.txt              文件用途
```

`SUMMARY.txt` 包含四个服务的运行状态、退出码、OOM 标记、启动/退出时间、重启次数、镜像 ID、vLLM/vLLM-Ascend/torch 等版本、白名单启动参数、端口、模型元数据哈希、采集错误和现场 keepalive 设置。失败摘要附每服务最多 12 行筛选日志；更多内容在 `service_logs/`，会省略含凭据或请求内容的行，并非完整日志。日志以 Docker 最近启动时间为起点，取不到时间时限最近 30 分钟。

另有 `resource_comparison`：现场已知服务 `max_num_batched_tokens=1024`、`max_num_seqs=10`、`max_model_len=4096`；控制器的批次假设为 4096，客户端并发默认 2。批次预算差异会明确提示；客户端并发与服务最大序列数分别记录。CLI 参数不等于引擎实时状态，工具不会自动修改这些设置。详细源码需要时再复制指定 `.txt` 内容。所有结果留在现场，不要求传文件。

摘要显示 `采集完整: False` 时也保留并复制摘要，里面有失败步骤；这不表示要重装环境。

若明确显示原有四个容器都是 `exited`，需要恢复这些服务时可手动执行：

```bash
sudo docker start wushibo-qwen-small-npu{0,1,2,3}
bash run_dabs_r234.sh prepare
```

上述名称是默认配置；使用其他 `DEVICE_IDS` / `CONTAINER_PREFIX` 时请按实际名称启动。`prepare` 自带就绪等待，刚启动后的 HTTP 000 不再需要手工循环检查。若再次退出，先复制摘要；退出码 137 不能单独判定为显存不足，`OOM=false` 也不能排除系统层问题。

## 3. 需要什么环境

| 项目 | 默认配置 / 要求 |
|---|---|
| 执行位置 | Linux 宿主机，包含 `run_dabs_r234.sh` 的新版项目目录 |
| 宿主机工具 | Bash、`python3` ≥3.8、可执行 `sudo docker` |
| 选用服务 | `wushibo-qwen-small-npu0`～`3`，已经运行 |
| 端口 | `8001`～`8004` |
| 服务内 Python | `/usr/local/python3.12.13/bin/python`；用于读取包元数据，不加载模型 |
| 模型目录 | `/data/model/Qwen2.5-1.5B-Instruct` |
| NPU | 现场报告为 910B3；以采集结果为准，四服务用四张卡 |

服务内解释器不同，例如使用 `python3`：

```bash
CONTROLLER_PYTHON=python3 bash run_dabs_r234.sh inspect
```

其他配置用 `DEVICE_IDS`、`CONTAINER_PREFIX`、`BASE_PORT`、`MODEL_PATH` 修改，例如 `DEVICE_IDS=4,5,6,7 bash run_dabs_r234.sh inspect`。只读取选中的四个容器。实验前需与同机用户协调资源，采集工具不会结束其他进程。

## 4. 采集后的诊断与报告命令

**下面的 `diagnose` 会执行真实推理；它是 HTTP 受限诊断，不是原生机制已修复。** 只有需要重测客户端流程时再运行。该模式复用服务镜像启动临时 CPU 控制容器，依赖已有 `httpx`、`numpy` 和完整模型目录。

| 用途 | 一行命令 |
|---|---|
| 只采集现场，立即检查 | `bash run_dabs_r234.sh inspect` |
| 就绪等待 + 环境采集 + 72 请求小探针，当前推荐 | `bash run_dabs_r234.sh prepare` |
| 只等待并采集，不发送推理请求 | `bash run_dabs_r234.sh inspect --wait-ready 300` |
| 已采集过，只运行小探针 | `bash run_dabs_r234.sh probe` |
| 从旧实验提取有限决策片段，不推理 | `bash run_dabs_r234.sh explain /旧项目/outputs/原实验目录` |
| 旧结果重生成报告，不执行推理 | `bash run_dabs_r234.sh report /旧项目/outputs/原实验目录` |
| 检查服务、模型和控制端环境 | `bash run_dabs_r234.sh check` |
| 查看预先固定的 96 条件计划 | `bash run_dabs_r234.sh plan` |
| HTTP 诊断，前 8 条件后正常暂停 | `bash run_dabs_r234.sh diagnose` |
| HTTP 诊断，只跑前 4 条件 | `bash run_dabs_r234.sh diagnose --max-conditions 4` |
| 继续同一份诊断，再跑 4 条件 | `bash run_dabs_r234.sh diagnose --resume outputs/上次目录 --max-conditions 4` |

`report` 和 `inspect` 直接使用宿主机 Python，不创建控制容器。`plan` 仍使用现场已有镜像，无需加载模型。

`explain` 也只用宿主机 Python，默认读取 R4/λ8、R2/λ16、R3/λ8 第一轮 Ours 的前 500 条决策，各条件单独标明覆盖范围，输出 `OLD_RUN_SUMMARY.txt`、`DECISIONS.txt`、`old_run_analysis.json`。资料只有 CSV 时也可计算计数与配对差值，并明确标记原始决策缺失。它不会运行资料包里那个生成 tar.gz 的收集脚本。

小探针结果在 `outputs/r234_probe_时间_编号/`：`PROBE_SUMMARY.txt` 是可复制摘要，`service_metrics.jsonl` 是前/中/后接口观测，`cases.jsonl` 是逐组结果，`requests.jsonl` 是请求和流数据，`REPRODUCE.md`、`probe_protocol.json`、`console.log` 记录运行方式和代码身份。`/metrics` 缺失时记录 unavailable，不伪造零。全局缓存命中计数不是逐前缀驻留证据；HTTP 重叠也不是引擎 batch 证明。

直接运行 `bash run_dabs_r234.sh` 或 `benchmark` 会在推理前提示原生机制未就绪并退出，避免把旧 HTTP 链路重新跑完误作机制修复成功。若明确只需完整 HTTP 对照，入口是 `bash run_dabs_r234.sh benchmark --execution-scope http-limited`；其结论仍为 HTTP 受限实验。

## 5. 实验内容、记录和续跑

条件内容保持原包：**副本数 2/3/4 × 到达率 4/8/12/16 × RR/SLL/Ayo/Ours × 两轮 = 96 条件**。每条件 64 工作流、216 请求、每请求 3584 输入 token、合计 13184 输出 token。

按修复方案预先调整执行顺序：前 4 个是 R4/λ8，随后 4 个是 R2/λ16；方法位置按组轮换，第二轮完整镜像。前 8 个只是检查点，48 个是第一轮。工作量、DAG、种子未改；不根据方法胜负选择继续与否。

诊断结果保存在 `outputs/r234_ascend_时间_编号/`：

| 文件 | 内容 |
|---|---|
| `TEST_REPORT.md`、`acceptance.json` | 完成情况、配对比较、机制缺失项 |
| `RESULTS.html` | 离线指标图表 |
| `REPRODUCE.md`、`process.jsonl`、`console.log` | 命令、路径、实际执行文件和过程 |
| `protocol.json`、`repair_protocol.json` | 源码哈希、完整计划、参数、实验范围 |
| `metrics.csv`、`diagnostic_reasons.csv` | JCT/TTFT/P95、评分开始/完成计数及失败原因 |
| `group_audits.jsonl` | 每四种方法的工作量检查；不冒充原生机制放行 |
| `conditions/条件名/` | 原始请求、决策、节点、配置、结果和哈希 |

报告重生成写入**新的** `outputs/r234_report_时间_编号/`，校验原始结果后读取日志，不改原实验目录。旧版已丢弃的超时评分结果无法恢复，会明确标注证据不全。

本版 R234 默认 `--http-keepalive 0`，保留现场已报告的连接复用绕行设置；不把它当作性能优化结论。每副本客户端并发默认 2，可通过 `MAX_INFLIGHT` 统一配置；它不同于服务端 `max_num_seqs`。

续跑只接受 `--max-conditions` 正常暂停的同版本实验。代码、参数、部署和标定改变后必须新开记录；旧版 96 条件保留为“修复前 HTTP 适配实验”，不能和本版或将来的原生机制结果混合。失败目录保留，不自动跳过失败或重试。

长时间诊断可使用机器已有的 `tmux`；实验记录始终保留在文件中。只复制报告文字，按需再复制指定日志片段，**不要打包或上传**。

## 6. 已修与待修

已补充容器状态与启动诊断、就绪等待、批次预算差异记录，以及现场采集、启动参数识别、资源单位区分、整批提交前校验、超时评分诊断、动作阶段记录、请求 ID、TTFT P95 和旧报告重生成。原包 373 个文件保持字节不变。本次依据与下一步见 [现场反馈处理记录](docs/field-feedback-20261016.md)。

**尚未完成：原生前缀 KV 观测、真实 running/waiting 压力、引擎队列应用确认、原生 batch-step 标定及昇腾现场验证。** 得到 `SUMMARY.txt` 后继续按版本处理，详见 [修复记录](docs/repair-20261016.md)。

开发机自检：`python tools/run_ascend.py test`。CPU 测试只验证逻辑，不能证明 NPU 实测或性能提升。

[R234 协议细节](docs/r234-ascend.md) · [旧版已有服务入口](README_existing.md) · [自启动服务部署](docs/ascend-linux.md) · [原 NVIDIA 工程](README_先读我.md)
