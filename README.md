# DABS 昇腾一键运行

## 2026-10-15 新实验：这次运行这个

在 **Linux 宿主机**的新项目目录运行，沿用现场已启动的 Qwen 服务。Git 下载先 `git pull --ff-only`；ZIP 下载请从 [GitHub 下载最新 ZIP](https://github.com/micavro/dabs-ascend/archive/refs/heads/main.zip)，解压到新目录，保留旧结果。

**一行跑全部 96 个新实验条件：**

```bash
bash run_dabs_r234.sh
```

默认连接 `wushibo-qwen-small-npu0`～`3`（端口 `8001`～`8004`），模型目录 `/data/model/Qwen2.5-1.5B-Instruct`。复用 MFE/现场已有的 vLLM-Ascend 镜像与服务，不重装环境、不加载第二份模型。宿主机需 `python3` ≥3.8 和 `sudo docker`；镜像内 Python ≥3.10、`httpx`、`numpy`，默认 Python 路径 `/usr/local/python3.12.13/bin/python`。四个服务需是同一模型、支持 4096 上下文、`/tokenize`、流式 usage、`min_tokens` 和 `ignore_eos`；入口会实际验证固定输出长度。

| 用途 | 手敲命令 |
|---|---|
| 全部新实验 | `bash run_dabs_r234.sh` |
| 仅环境检查，不跑实验 | `bash run_dabs_r234.sh check` |
| 先跑前 4 个条件后暂停 | `bash run_dabs_r234.sh --max-conditions 4` |
| 继续该次实验的剩余条件 | `bash run_dabs_r234.sh --resume outputs/上次目录名` |
| 换用 4～7 号卡 | `DEVICE_IDS=4,5,6,7 bash run_dabs_r234.sh` |

模型路径不同时先 `export MODEL_PATH=/实际路径`。默认控制端每副本并发 2；若现场决定改为 10，整次实验从开始统一用 `MAX_INFLIGHT=10 bash run_dabs_r234.sh`，续跑保持相同设置。避免其他负载同时使用这四个服务。长时间运行请在机器已有的 `tmux` / `screen` 会话里执行，防止终端断线。

新范围为 **R=2/3/4 × 到达率=4/8/12/16 × RR/SLL/Ayo/Ours × 2 轮 = 96 条件**，两轮顺序互为镜像；每条件 64 个工作流、216 个请求、每请求 3584 输入 token、按 DAG 固定输出，总输出 13184 token。完整实验共 20736 请求。旧的 `benchmark` / `--suite grid` 不代表这批新实验。

结果保存在 **`outputs/r234_ascend_时间_编号/`**，终端打印 `R234_REPORT=`。先看 `TEST_REPORT.md`（是否 96/96、Ours 配对比较），再打开 `RESULTS.html` 看曲线；`REPRODUCE.md`、`process.jsonl`、`console.log` 自动记录运行命令、目录、文件、时间和过程。CSV 与每条件的完整请求、调度、token、时延记录都保留。出现错误看 `error.txt`；失败不自动重试，只有 `--max-conditions` 正常暂停的目录可以续跑。

新包的 373 个文件逐字节保留并校验，Ours 使用新增的 DAG-H V5。现有 HTTP 服务无法提供原生 KV 命中、引擎阶段统计或内部队列重排，报告将明确记为未观测；V5 需要原生缓存证据的路由候选因此不可用。此次跑的是昇腾上的新负载与可执行调度适配，不代表原生引擎机制或性能结论完全等价。固定 token 实验无需问答 EM/F1。详见 [新实验说明](docs/r234-ascend.md)。

## 现场已有 8 个 Qwen 容器：用这个入口

以下是上一版实验，保留用于旧结果核对。

在 **Linux 宿主机**的项目目录执行，不要进入模型容器。默认连接现有的 `wushibo-qwen-small-npu0`～`7`，端口 `8001`～`8008`；只新建临时 CPU 控制端，退出后模型服务继续运行。

Git 下载的项目先敲 `git pull --ff-only`；ZIP 下载的项目重新下载解压到新目录。

```bash
bash run_dabs_existing.sh check
bash run_dabs_existing.sh
```

第一行只检查，第二行运行小测试。模型路径默认 `/data/model/Qwen2.5-1.5B-Instruct`；不同路径先敲 `export MODEL_PATH=/实际路径`。需要宿主机的 `python3` 和 `sudo docker`，控制端复用所选第一张卡对应容器的本地镜像，不下载镜像、不重新加载模型。

小测试通过后，四卡 B1–B4 工作流测试：

```bash
DEVICE_IDS=0,1,2,3 bash run_dabs_existing.sh benchmark
```

更多现场参数、结果位置和自检见 [已有服务说明](README_existing.md)。默认控制端每副本最多 2 个在途请求；服务端并发配置保持原样。

## 需要由脚本新建模型服务：原入口

下面的命令在 **Linux 昇腾机器上、之前跑 MFE 的 vLLM-Ascend 容器内**手敲执行。

## 1. 先确认环境

- 单机 910B / 910B4，默认使用空闲的 `0,1,2,3` 四张卡；8 卡机器够用。
- 复用 MFE 已能运行的驱动、CANN、`torch`、`torch-npu`、`vllm`、`vllm-ascend` 配套环境，Python ≥ 3.10。不要单独升级这些组件。
- 容器内能看到完整的 **Qwen2.5-1.5B-Instruct** 模型目录，包括权重、配置和 tokenizer。仓库不含模型，程序不会自动下载模型。
- 当前容器没有其他正在运行的 vLLM 服务，默认端口 `18000–18003` 空闲。

先敲两条，确认 Python 版本和 NPU 可见：

```bash
python3 --version
npu-smi info
```

如果环境还没装好，先使用之前的 MFE 容器；本项目不负责安装驱动或推理环境。

## 2. 下载代码（只做一次）

```bash
git clone https://github.com/micavro/dabs-ascend.git
cd dabs-ascend
```

仓库公开，无需登录。没有 Git 时可下载 [ZIP](https://github.com/micavro/dabs-ascend/archive/refs/heads/main.zip)，解压后进入 `dabs-ascend-main`。

## 3. 设置模型，然后一键运行

把下一行的路径换成**容器内实际模型目录**。路径含空格时用双引号包住。

```bash
export MODEL_PATH=/x/Qwen2.5-1.5B-Instruct
bash run_ascend.sh
```

以后在同一个终端里，只需最后一行。换终端或重新进入容器后，需要重新设置 `MODEL_PATH`。

入口会自动检查环境、启动四个单卡模型副本、执行推理和工作流小测试、保存结果，最后关闭本次启动的服务。`Ctrl+C` 可以中断并保留记录。

若提示缺少 `httpx` 或 `numpy`，在当前容器中补齐控制端依赖，再运行：

```bash
python3 -m pip install -r requirements-ascend.txt
bash run_ascend.sh
```

该依赖文件只包含 `httpx` 和 `numpy`。无外网环境需使用机器已有的内部软件源或离线包。

## 4. 看结果、更新代码

- 终端打印的 `ASCEND_REPORT=...` 是测试报告路径；结果保存在 `outputs/ascend_.../`。
- 先看 `TEST_REPORT.md`。失败时看同目录的 `error.txt` 和 `server_npu*.log`。
- 首次运行包含模型加载和标定，需要等待；默认只跑小测试。

通过 Git 下载的项目，后续更新时在项目目录执行（ZIP 下载则重新下载解压）：

```bash
git pull --ff-only
bash run_ascend.sh
```

## 按需使用

以下推理命令都沿用上面设置的 `MODEL_PATH`。

| 需要做什么 | 手敲命令 |
|---|---|
| 只检查环境，不启动推理 | `bash run_ascend.sh check` |
| 改用空闲的 4–7 号卡 | `DEVICE_IDS=4,5,6,7 bash run_ascend.sh` |
| 容器只有 `python` 命令 | `PYTHON=python bash run_ascend.sh` |
| 小测试通过后跑 B1–B4 工作流测试 | `bash run_ascend.sh benchmark` |
| 纯 CPU 代码自检，无需模型和 NPU | `bash run_ascend.sh test` |

开发机也可用 `python tools/run_ascend.py test` 自检；需要离线传输包时执行 `python -X utf8 tools/package_ascend.py`。

已有 Linux CPU 自动检查记录；现场已开始运行并反馈了已有服务补丁，本次合并后的现场结果仍需看实际运行日志。更多参数和适配范围见 [昇腾部署说明](docs/ascend-linux.md)；历史 NVIDIA 工程见 [原工程说明](README_先读我.md)。
