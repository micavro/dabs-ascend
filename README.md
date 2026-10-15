# DABS 新实验：昇腾一键运行

本次运行 **2026-10-15 的 R234 新实验，共 96 个条件**。更新代码后，在 **Linux 宿主机**的项目目录执行这一行：

```bash
bash run_dabs_r234.sh
```

项目目录是包含 `run_dabs_r234.sh` 的目录。沿用现场已启动的 Qwen 模型服务，入口会启动临时 CPU 控制容器，自动检查、校准、运行并保存结果；模型服务在结束后继续运行。

## 1. 下载或更新代码

**之前用 Git 下载的：** 进入原项目目录，一行更新并启动：

```bash
git pull --ff-only && bash run_dabs_r234.sh
```

**之前用 ZIP 下载的：** 下载 [最新代码 ZIP](https://github.com/micavro/dabs-ascend/archive/refs/heads/main.zip)，解压到一个新目录，保留旧目录和旧 `outputs/`。进入新目录后执行 `bash run_dabs_r234.sh`。ZIP 目录不能用 `git pull` 更新；若旧目录有手工补丁，也采用新目录解压的方式。

**第一次用 Git 下载：**

```bash
git clone https://github.com/micavro/dabs-ascend.git
cd dabs-ascend
bash run_dabs_r234.sh
```

仓库公开，下载不需要登录。现场之前的目录为 `/data/wushibo/dabs-ascend-main`，本次以实际更新或新解压的目录为准。

## 2. 需要什么环境

在 **Linux 宿主机**运行上述入口。复用之前 MFE / 现场已跑通的 vLLM-Ascend 环境；四个 Qwen 服务应已启动。此入口不安装驱动、CANN 或推理依赖，不下载镜像和模型。

| 项目 | 默认值 / 要求 |
|---|---|
| 机器 | 单机昇腾 910B / 910B4，至少 4 张可用卡；8 卡机器够用 |
| 已运行的服务容器 | `wushibo-qwen-small-npu0`～`3` |
| 服务端口 | `8001`～`8004` |
| 宿主机模型目录 | `/data/model/Qwen2.5-1.5B-Instruct`，含完整权重、配置和 tokenizer |
| 宿主机工具 | Bash、`python3` ≥3.8、可执行 `sudo docker` |
| 控制端镜像 | 复用所选第一张卡服务的本地镜像 |
| 镜像内 Python | ≥3.10，默认 `/usr/local/python3.12.13/bin/python`，已有 `httpx` 和 `numpy` |
| 模型接口 | 同一 Qwen2.5-1.5B-Instruct 模型，支持 4096 上下文、tokenize/detokenize、流式 usage、`min_tokens`、`ignore_eos` |
| 控制端并发 | 每副本最多 2 个在途请求；服务端保持现有配置 |

默认配置与现场一致时，直接执行一键命令。模型目录不同时先设置：

```bash
export MODEL_PATH=/实际模型目录
```

镜像里 Python 路径不同时，例如使用 `python3`：

```bash
CONTROLLER_PYTHON=python3 bash run_dabs_r234.sh
```

运行期间让所选服务专供本次实验使用，并确认结果盘有足够空间。完整请求和调度日志都会保留。

## 3. 常用手敲命令

| 用途 | 命令 |
|---|---|
| 一键跑全部 96 个条件 | `bash run_dabs_r234.sh` |
| 只检查环境，不执行实验 | `bash run_dabs_r234.sh check` |
| 查看 96 条件计划，不执行推理 | `bash run_dabs_r234.sh plan` |
| 先跑前 4 个条件，结束后正常暂停 | `bash run_dabs_r234.sh --max-conditions 4` |
| 继续同一次实验的剩余条件 | `bash run_dabs_r234.sh --resume outputs/上次目录名` |
| 改用 4～7 号卡 | `DEVICE_IDS=4,5,6,7 bash run_dabs_r234.sh` |

`plan` 仍通过现场镜像启动 CPU 控制端，所以需要所选第一张卡的容器和本地镜像；无需模型文件或其余服务健康。

若现场确定要将客户端并发改为 10，整次实验从开始统一执行 `MAX_INFLIGHT=10 bash run_dabs_r234.sh`，续跑也保持相同设置。

长时间运行建议使用机器已有的 `tmux`，避免 SSH 断线影响实验：

```bash
tmux new -s dabs-r234
bash run_dabs_r234.sh
```

离开会话按 `Ctrl+B`，松开后按 `D`；回来查看执行 `tmux attach -t dabs-r234`。

**续跑仅适用于 `--max-conditions` 正常暂停的目录。** 程序会打印可复制的续跑命令，并检查源码、参数、服务身份、标定和已完成结果。暂停期间保持代码、配置和服务不变。Ctrl+C、断线或报错会保留失败现场；失败目录不自动重试或跳过条件，修复后重新开始一份记录。

## 4. 这条命令跑哪些实验

**副本数 2/3/4 × 到达率 4/8/12/16 × RR/SLL/Ayo/Ours × 2 轮 = 96 条件。** 两轮方法顺序和条件遍历顺序互为镜像，第一组为 R4、到达率 8 的四种方法。

每条件运行 64 个工作流、216 次请求。每次请求输入 3584 token，输出按 DAG 固定为 16/32/48/64/128 token，每条件合计输出 13184 token。完整实验共 **20736 次正式请求**，预检、预热和校准另计。

前四个条件跑完是 **4/96 的集成检查**；报告达到 **96/96 且检查通过**才表示完整负载跑完。本次是固定 token DAG 回放，子节点遵守依赖，不计算问答 EM/F1。

## 5. 结果在哪里，怎么看

结果自动保存在当前项目的：

```text
outputs/r234_ascend_时间_编号/
```

终端打印 `R234_OUTPUT=` 和 `R234_REPORT=`。其中容器路径 `/project/outputs/...` 对应宿主机项目中的 `outputs/...`，不需要进入容器找文件。

| 文件 | 看什么 |
|---|---|
| `TEST_REPORT.md` | 是否完成 96/96、固定工作量检查、Ours 与基线的配对比较 |
| `RESULTS.html` | 用浏览器离线打开，查看时延、吞吐曲线和逐条件结果 |
| `REPRODUCE.md` | 本次命令、项目路径、运行了哪些文件、复现过程与保存位置 |
| `console.log`、`process.jsonl` | 控制台记录、开始结束时间、执行参数和逐条件进度 |
| `metrics.csv`、`paired_comparisons.csv` | 逐条件数据与配对差值；JCT 改善为正表示 Ours 更快，为负表示更慢 |
| `workflow_metrics.csv`、`latency_decomposition.csv`、`replica_load.csv` | 工作流时延、客户端关键路径和副本 HTTP 负载 |
| `conditions/条件名/` | 每条件的配置、请求、调度、token 用量、节点状态和原始结果 |
| `error.txt` | 失败原因；同时保留完整输出目录供排查 |

提交复现结果时，保存并交付 **整个输出目录**，这样命令、环境、过程与结果可以相互核对。

## 6. 实现与验证范围

新实验包的 373 个文件逐字节保留并校验，Ours 调用新增的 DAG-H V5。现有 HTTP 服务无法提供原生 KV 命中、引擎阶段统计或内部队列重排；需要原生缓存证据的 V5 路由候选不可用，报告会明确记录这些限制。负载完整运行不等于原生引擎机制完全等价，也不保证 Ours 有性能提升。

Linux CI 已通过 46 项适配测试和原包 79 项检查，覆盖固定 token、依赖、四方法运行、暂停续跑和原始文件校验。CPU 自检不能替代现场昇腾实测。开发机自检命令：

```bash
python tools/run_ascend.py test
```

详细协议与适配范围见 [R234 新实验说明](docs/r234-ascend.md)。之前的小测试、B1–B4 和 `--suite grid` 属于旧实验，说明保留在 [已有服务入口](README_existing.md)；需要自行启动模型服务时查看 [昇腾部署说明](docs/ascend-linux.md)，历史 NVIDIA 工程见 [原工程说明](README_先读我.md)。
