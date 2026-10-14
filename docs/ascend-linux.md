# DABS 昇腾 910B4：Linux 一行启动

这版用于把原项目的 MuSiQue 文本工作流在昇腾上执行并留下可核查的功能测试记录。默认单机四卡、每卡一个独立模型副本（TP=1）；八卡机器可以运行，不需要多机。代码不安装驱动、不修改 CANN、不升级已有 torch/vLLM 环境。

**当前交付是功能迁移版，尚未在 910B 真机验收。** 原 GPU39 调度器依赖 vLLM 0.6.3 / CUDA 内部 hook，而 MFE 使用 vLLM-Ascend V1。这版保留原问题、依赖图、父答案绑定、候选生成和工作流评分代码，通过公开 HTTP 接口执行。迁移方法明确命名为 `DAG-Ascend`，不能把它的结果直接当作原论文 `DAG/Ours` 的严格性能复现。

## 1. 上机直接执行

和 MFE 一样，优先通过公开 GitHub 仓库下载。在之前使用的 Linux / vLLM-Ascend 容器中执行：

```bash
git clone --depth 1 https://github.com/micavro/dabs-ascend.git
cd dabs-ascend
MODEL_PATH=/实际路径/Qwen2.5-1.5B-Instruct bash run_ascend.sh
```

后续在该目录执行 `git pull --ff-only` 更新代码。没有 Git 时，也可下载 `https://github.com/micavro/dabs-ascend/archive/refs/heads/main.zip`，解压进入 `dabs-ascend-main` 后运行。仓库不包含模型权重；`MODEL_PATH` 指向机器上已有的完整模型目录。

可直接上传 `outputs/DABS_Ascend_Linux_v1.zip`。这是包含源码、数据和本说明的传输包，不包含模型权重、环境、旧 outputs 或密钥。解压后进入 `DABS_Ascend_Linux_v1` 目录。重新打包使用 `python -X utf8 tools/package_ascend.py`；同名包已存在时需指定新的 `--out`。

先进入之前跑 MFE 的、设备空闲的 vLLM-Ascend 容器，将整个项目放入容器可读写的挂载目录，并 `cd` 到项目根目录。确保本地已有 **Qwen2.5-1.5B-Instruct** 的完整 Hugging Face 权重目录。

```bash
MODEL_PATH=/x/Qwen2.5-1.5B-Instruct bash run_ascend.sh
```

只需要将模型路径替换成实际位置。路径有空格时加引号。默认使用物理卡 `0,1,2,3`，端口 `18000..18003`。机器剩余四卡不启动模型副本。脚本前台运行，`Ctrl+C` 会停止本次启动的服务并保留结果。

如果同一机器分配的是其他空闲卡：

```bash
MODEL_PATH=/x/Qwen2.5-1.5B-Instruct DEVICE_IDS=4,5,6,7 bash run_ascend.sh
```

若容器只提供 `python` 命令：

```bash
PYTHON=python MODEL_PATH=/x/Qwen2.5-1.5B-Instruct bash run_ascend.sh
```

默认流程：

1. 校验冻结的原始源码及 MuSiQue 数据，检查 Linux、包版本、设备映射、模型文件和端口。
2. 依次在每张卡启动一个 vLLM-Ascend 服务，并等待 `/health` 就绪。
3. 每卡发出一次带冲突提示的 JSON 枚举约束 canary，验证约束实际生效；再以正式 answer schema 固定预热一次。比对副本间的完整 chat tokenization。
4. 每卡固定执行 8 次本机标定请求（4 种输入长度 × 2 轮），保存全部测量。不会套用 NVIDIA 的时间系数。
5. 使用原 smoke 数据的两条工作流、共 7 个 LLM 节点，依次测试 `RR`、`P2C`、`Affinity`、`DAG-Ascend`。四方法共 28 次正式请求，执行顺序由固定种子决定。
6. 保存回答、依赖关系、路由、JCT、P95、客户端首文本/文本生成时间代理、吞吐和 EM/F1，生成测试记录，关闭本次创建的进程组。

四卡默认 smoke 合计 **68 次生成请求**：8 次 canary/预热 + 32 次标定 + 28 次正式请求；tokenize/detokenize 不计入生成请求数。服务加载、标定和固定预热不计入正式 JCT；正式运行中的输入准备、调度、HTTP 和回答处理均计入。

脚本不连接任何预先运行的模型服务。如果当前容器中存在其他 vLLM API 服务，会报错并保留它；不要直接结束他人的进程。不同容器之间的设备独占仍由机器使用者安排，进程锁只协调本工具在同一文件系统内的运行。

## 2. 先测试代码或只检查环境

纯 CPU 检查，无模型、无 NPU、无网络访问：

```bash
bash run_ascend.sh test
```

Windows 开发机可使用等价入口：

```text
python tools/run_ascend.py test
```

仅进行 Linux 昇腾环境检查，不启动模型服务或生成请求：

```bash
MODEL_PATH=/x/Qwen2.5-1.5B-Instruct bash run_ascend.sh check
```

查看参数：

```bash
bash run_ascend.sh --help
```

优先复用 MFE 已能运行的容器。缺少的如果仅是控制端依赖，可在该环境中执行 `python -m pip install -r requirements-ascend.txt`；离线机器用内部 wheelhouse。不要从这个项目安装或单独升级 `torch`、`torch-npu`、`vllm`、`vllm-ascend`。它们需要与 CANN/驱动整体兼容。

## 3. 更完整的测试

全部 B1—B4，每组 16 个工作流，四方法各 2 轮，共 32 个条件：

```bash
MODEL_PATH=/x/Qwen2.5-1.5B-Instruct bash run_ascend.sh benchmark
```

固定四卡、λ=2、B3，四方法各 8 轮，共 32 个条件：

```bash
MODEL_PATH=/x/Qwen2.5-1.5B-Instruct bash run_ascend.sh benchmark --suite case
```

B3 是原历史结果中事后选中的案例，不能描述为未见数据泛化测试。这里是明确选择 B3 的新平台测试，不改写历史登记。

1—4 副本、λ=1/2/4/8、B1—B4、四方法各 2 轮，共 **512 条**：

```bash
MODEL_PATH=/x/Qwen2.5-1.5B-Instruct bash run_ascend.sh benchmark --suite grid
```

这不是原来的五方法 640 条网格；本移植没有把 `DAG-Lite` 或旧原生预算消融冒充为已迁移。正式上机前可以查看完整计划，不执行任何推理：

```bash
bash run_ascend.sh plan --suite grid
```

默认四副本与原实验对齐。八副本可以通过 `DEVICE_IDS=0,1,2,3,4,5,6,7` 在 smoke/blocks 模式执行，这是新增扩展条件。`case` 固定要求四副本；`grid` 最高只扫描四副本。

所有运行都创建新的 `outputs/ascend_<UTC时间>_<随机标识>/`。可指定 `--out outputs/my_first_ascend_smoke`，但该路径不能已存在。中断或失败后新建目录重跑，不跳过失败项、不自动恢复成成功。

## 4. 和 MFE 配置的关系

| 项目 | 这版的设置 |
|---|---|
| 环境 | 复用 MFE 已有 Linux / vLLM-Ascend 容器，不固定猜测现场版本 |
| 设备映射 | 每个子服务设置自身的 `ASCEND_RT_VISIBLE_DEVICES`、`NPU_VISIBLE_DEVICES` |
| 引擎 | `VLLM_TARGET_DEVICE=npu`、`VLLM_USE_V1=1`、`spawn`、eager、bfloat16 |
| 模型 | DABS 的 Qwen2.5-1.5B-Instruct；不会自动替换为 MFE 的 Llama |
| 并行 | 独立模型副本，TP=1；并非一个模型做四卡/八卡张量并行 |
| 上下文与调度预算 | 4096 / 4096，block=16，服务端最多 64 个序列 |
| 控制端准入 | 每副本最多 8 个在途请求，`--max-inflight` 可设 1—64；不是物理 KV 容量观测 |
| 显存比例 | 0.75，可用 `--memory` 调整；参数在 vLLM 中仍叫 `gpu-memory-utilization`，NPU 也沿用此命名 |
| 前缀缓存 | 开启：原 DABS 任务需要；没有照搬 MFE 某些实验的 APC=0 |
| 输出 | JSON answer schema、自然 EOS、最多 96 tokens，不强行填满输出 |
| 网络 | 模型和 tokenizer 离线；服务只绑定 127.0.0.1；客户端忽略 HTTP 代理环境变量 |

新旧服务接口按真实 `/openapi.json` 检测：支持时使用 `structured_outputs.json`，旧版使用 `guided_json`；不支持时失败，不回退成无约束输出。官方说明 `guided_json` 等旧字段自 v0.12.0 移除，参见 [vLLM Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/)。

容器应整体复用匹配的栈；官方预构建镜像包含配套的 CANN、TorchNPU、vLLM 和 vLLM-Ascend，参见 [vLLM Ascend Quick Start](https://docs.vllm.ai/projects/ascend/en/latest/quick_start.html)。

本次参考的本地 MFE 文件：`mfe-ascend/docs/company-ascend-benchmark-runbook.md`、`deploy/launch_in_existing_vllm_container.sh`、`mfe/workers/worker_v.py` 和 `constraints/vllm-ascend-docker-current.txt`。没有读取其密钥文件，也没有连接远端机器。

## 5. 如何看测试报告

终端打印 `ASCEND_OUTPUT=...` 和 `ASCEND_REPORT=...`。重点文件：

| 文件 | 用途 |
|---|---|
| `TEST_REPORT.md` / `acceptance.json` / `test_items.csv` | A01—A12 测试项、通过/失败/不支持、证据定位 |
| `environment.json` | Linux、设备、客户端及服务同环境包版本、模型文件信息 |
| `protocol.json` / `original_integrity.json` | 实验计划、随机种子、源码及数据哈希 |
| `servers.json` / `server_npu*.log` / `cleanup.json` | 物理卡到副本的映射、服务参数、报错和清理 |
| `ascend_profile.json` / `calibration_requests.jsonl` | 本机时间模型与原始标定请求 |
| `metrics.csv` / `index.json` | 所有已完成条件的指标，保留负收益和拒答 |
| `conditions/*/requests.jsonl` | 发送的 token 列表、约束参数、原始 SSE 和失败片段 |
| `conditions/*/nodes.json` / `input_proof.json` | 每个节点的原始答文、父答案绑定和时间线 |
| `conditions/*/decisions.jsonl` | 原候选/评分结果、过期计划丢弃及参考动作回退 |
| `conditions/*/summary.json` | 完整工作流 JCT 及求和一致的客户端关键路径分解 |
| `conditions/*/quality.json` | 执行结束后才读 gold 的 EM/F1、格式错误和 UNKNOWN 数量 |

退出码 0 表示所选功能测试完成且通过；环境失败、推理/格式失败、所有 DAG 评分均未完成等会返回非零。`functional_test_pass=true` 只表示软件执行及接口合同通过，**全 UNKNOWN 或 EM/F1 很低仍可能通过功能测试**，必须同时阅读质量指标。

### 与师兄报告逐项核对

| 报告可能要求的项目 | 当前可提供的证据 |
|---|---|
| 单机多卡启动、模型生成、DAG 依赖和真实父答案 | 直接测试并记录 |
| 四种策略完成任务、JCT/P95/吞吐、EM/F1 | 可记录这版昇腾适配的真实结果 |
| 首 token / TPOT | 仅提供客户端首文本与文本流代理，不能冒称原生逐 token 时间 |
| 客户端关键路径：依赖释放、输入准备、中央队列、HTTP、处理 | 可分解，分项之和校验为 JCT |
| 原方法的完整原生队列重排、有限动作窗口 | **不支持**；HTTP 版在评分前移除队列重排动作，只保留路由与有界准入 |
| 实际缓存命中、淘汰、跳过 prefill tokens、物理 KV 占用 | **不支持**；历史路由偏好不作为缓存命中证据 |
| 原生 Prefill/Decode/Mixed host 阶段时间 | **不支持**；新标定是 HTTP 代理，混合阶段为加法估计 |
| GPU39 的原性能数字或原论文改善比例 | **未验收**；硬件、观测和动作接口不同 |

本次还没有师兄的具体测试报告。若报告必须包含后三类原生指标，需在现场确定 vLLM-Ascend 精确版本后继续开发其 V1 调度/缓存观测接口，不能通过改图例、伪造 status 文件或放宽原源码哈希检查“通过”。

## 6. 故障定位

- `environment.json` 报缺少 vLLM/torch-npu：通常是没有进入之前的 MFE 容器，或 Python 不对应原环境。先检查解释器，不安装 CUDA torch。
- 模型结构或文件不符合：提供完整 Qwen2.5-1.5B-Instruct 目录。结构检查不能证明权重修订完全相同；权重版本仍需与原实验核对。
- 服务启动失败：看对应 `server_npuN.log`，报告保留启动命令。默认逐副本最多等待 900 秒，可用 `--startup-timeout` 调整。
- 端口已占用：换 `BASE_PORT=19000`；不结束已有服务。
- JSON canary 失败：检查现有版本的结构化输出支持和相关依赖。工具不会悄悄改为自由生成。
- 输入超 4096 或实际 token 数不一致：会保留原始输入并失败，不自动删上下文、换 tokenizer 或注入 gold。
- CPU 调度计划超时：丢弃过期结果并使用同一快照的参考动作，计数写入报告；所有评分均未完成时 A07 不通过。不能把全程回退称为 DAG 方法验收。
