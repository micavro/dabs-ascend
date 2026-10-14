# DABS / GPU39 工作流调度实验：独立复现工程

> **新增 Linux 昇腾功能测试入口：** 在之前 MFE 的 vLLM-Ascend 容器中运行 `MODEL_PATH=/x/Qwen2.5-1.5B-Instruct bash run_ascend.sh`。默认单机四卡，八卡机器足够。部署、测试报告和与原生 GPU39 的差异见 [昇腾说明](docs/ascend-linux.md)。当前为待真机验收的功能迁移版，`DAG-Ascend` 不等同于原生 GPU39 的完整性能复现；下面的历史 NVIDIA 复现流程仍保留。

> **本工程用于复现研究报告涉及的实验代码、输入及可核查统计。** 所有原算法均保留历史真实实现与名称，不为得到更好的结果筛选数据或修改调度策略。工程不包含 GPU 服务进程、容器镜像、模型权重或私钥。

## 一、先说清楚三项事实

1. **Ours ≙ 历史可执行代码中的 `DAG`**；消融 `DAG-Lite` 不作为主方法。
2. **研究报告中写作 `Preble` 的部分实验数据真实来源于 `RR`（轮询）。** 当前归档没有 Preble 论文的完整复现。为了可追溯，代码、重新生成的表格和图片均采用真实方法名 `RR`；不能通过修改绘图图例冒称 Preble 的结果。详见 `docs/BASELINE_TRUTH.md`。
3. 当前的 GPU 原生服务没有被打包，因此“离线源代码/数据检查”和“实际 GPU 重跑”是两种不同的操作；在没有部署与原实验兼容的模型服务时，不能保证精确再现运行时长。

## 二、新目录快速开始

在新目录中解压，不要覆盖旧实验：

```bash
mkdir -p /home/wmy/sci/repro_workspace
cd /home/wmy/sci/repro_workspace
unzip DABS_GPU39_Repro_v1.zip
cd DABS_GPU39_Repro_v1
```

**首先只做离线检查，不用 GPU：**

```bash
python3 tools/verify_sources.py --test-native
# 对原有 GPU 运行实现执行 95 项自动化测试
WF_SOURCE_ROOT="$PWD/source" bash source/spra_workflow_sched_v2_1/run_checks.sh
# 对固定案例编排运行 121 项自动化测试
bash source/spra_workflow_accept_v1/run_checks.sh
```

离线统计和机制实验重新提取（安装绘图依赖只针对分析环境，不应更换 GPU 服务环境中的 vLLM/torch）：

```bash
python3 -m pip install -r requirements-offline.txt
python3 tools/recompute_results.py --out outputs/recount_001
python3 tools/extract_pilot_mechanism.py --out outputs/pilot_001.csv
python3 tools/extract_native_phase.py --out outputs/native_host_001.csv
python3 tools/plot_experiments.py --out outputs/figures_001
```

每个脚本遇到**已存在且非空的输出目录**或**已有的同名输出文件**时会停止；不会覆盖旧结果。

## 三、连接已有 NVIDIA GPU39 原生服务做新实验

需要事先确认：同一模型版本与 tokenizer、四个 vLLM 服务、原生 hook 身份、缓存配置、服务到物理 GPU 的映射、没有其他进程占用服务。**这套代码不会替你部署 vLLM 服务，也不会重启/重装系统。** 旧集群可使用已有的 `spra-exp` 环境；若新服务器采用不同系统软件，请先适配并验证一致性。

在 GPU 服务器设置实际服务运行时路径（按你的真实部署位置填写，不是本包的 `runtime_local` 空目录）：

```bash
RUNTIME=/home/wmy/sci/spra_core_exp/joint_e1

# 网格：检查服务身份，不执行模型生成
python3 tools/run_gpu.py gate --engine native-v21 --runtime "$RUNTIME"

# 选择原生主网格，以新目录开始，每次最多80条件；真正运行时才加确认开关
python3 tools/run_gpu.py run --engine native-v21 --suite grid \
  --runtime "$RUNTIME" --out outputs/gpu_grid_001 \
  --replicas 2 3 4 --max-conditions 80 --confirm-gpu-use

# 继续同一个新计划；每轮运行后先检查日志，再进行下一段
python3 tools/run_gpu.py run --engine native-v21 --suite grid \
  --runtime "$RUNTIME" --out outputs/gpu_grid_001 \
  --resume --replicas 2 3 4 --max-conditions 80 --confirm-gpu-use
```

**独立固定 B3 案例**（四副本、总到达率 λ=2、B3，四方法各 8 轮，32 条正式条件）：

```bash
# 先登记。仅在首次运行时执行；会写入不可覆盖的 plans/selected_cases.json。
python3 tools/run_gpu.py register-case --engine fixed-case \
  --case 4:2:3 --reason '基于已知历史结果选定的B3固定验证；全部8轮保留'

python3 tools/run_gpu.py plan --engine fixed-case --suite case-confirm \
  --out outputs/B3_predeclared_plan.json
python3 tools/run_gpu.py gate --engine fixed-case --runtime "$RUNTIME"
python3 tools/run_gpu.py run --engine fixed-case --suite case-confirm \
  --runtime "$RUNTIME" --out outputs/gpu_B3_confirm_001 --confirm-gpu-use
```

`fixed-case` 采用已上传的**源码交接版本**：九个算法/路由/原生执行核心文件与原始实现哈希一致，但输入 profile 读取器为适配新目录有意迁移，**不是原 GPU39 当日完整运行环境的逐字节快照**。对比历史效果前，必须先核对服务模型、权重、profile 和原生输入记录。原始 640 网格使用 `native-v21` 的完整冻结依赖（校验 48 个源文件）。

**历史三方法 72 条消融**：

```bash
python3 tools/run_gpu.py gate --engine original-v2 --runtime "$RUNTIME"
python3 tools/run_gpu.py run --engine original-v2 --suite ablation \
  --runtime "$RUNTIME" --out outputs/gpu_ablation_001 --confirm-gpu-use
```

实验完成后的报告：

```bash
python3 tools/run_gpu.py report --engine native-v21 --out outputs/gpu_grid_001
python3 tools/run_gpu.py report --engine original-v2 --out outputs/gpu_ablation_001
python3 tools/run_gpu.py report --engine fixed-case --out outputs/gpu_B3_confirm_001
python3 tools/run_gpu.py case-report --engine fixed-case --out outputs/gpu_B3_confirm_001
```

上面所有 `gate/run/report` 都在原始应用的运行安全校验之上，**并未禁用硬件指纹、准入检查或失败处理**。如出现状态不一致，应停止并单独排查，而不是更改哈希值、阈值或复用旧输出目录。

## 四、实验与目录一览

| 研究报告实验 | 运行代码 | 归档现状 | 离线复算/绘图 |
|---|---|---|---|
| 固定B3、四方法8轮 | `source/spra_workflow_accept_v1/` | 有32条件摘要；缺完整请求/后端原始记录 | `tools/recompute_results.py`, `tools/plot_experiments.py` |
| 四组结构完整消融 | `source/spra_workflow_sched_v2/` | 历史72条件/24个Ours-vs-Affinity配对有索引，原记录留GPU | `recompute_results.py` |
| 1～4副本、四档到达率 | `source/spra_workflow_sched_v2_1/` | 旧检查点320条+新日志80条；共400/640条有证据 | `recompute_results.py` |
| 工作流时延分解 | `source/spra_stage19_typed_handoff_v1/` 及 `source/spra_stage19_musique_joint_v1/` | Pilot三轮原生摘要均在 `checkpoint_review` | `extract_pilot_mechanism.py` |
| 两级前缀/KV利用 | 同上、真实规范 token 及原生观测 | 提供前缀结构统计及原生跳过token数 | `extract_pilot_mechanism.py`, `plot_experiments.py` |
| Host原生Prefill/Decode/Mixed阶段 | 同上 | B3三轮真实 host 预测/实测阶段记录 | `extract_native_phase.py` |
| 控制开销、质量、资源保护 | `source/spra_workflow_sched_v2/`、`source/spra_workflow_accept_v1/` | 控制统计、EM/F1、原生资源保护和自动化测试 | `evidence/catalog/`, 历史完整日志待迁入 |

更详细的文件定位、限制和命令见 `docs/EXPERIMENT_TO_CODE_MAP.md`、`docs/MISSING_FOR_FULL_REPLAY.md`、`docs/BASELINE_TRUTH.md`。

## 五、每个子目录的责任

- `source/`：按原名保存的算法、原生执行器、工作流绑定、GPU运行脚本和测试。
- `profiles/reference/`：固定案例代码迁移读取器的已校验 profile；原始冻结 profile 位于 `source/spra_stage19_typed_handoff_v1/prepared/`。
- `evidence/catalog/`：上传的 `review.zip` 的实验条件注册表、结果配对表、覆盖率和哈希。
- `evidence/original/`：原始 `review.zip`、其他历史归档和新增 R2 控制台记录；不作为 GPU 服务替代品。
- `evidence/derived/`：从实验日志得到的用于图表的数值中间表，标注来源和统计口径。
- `tools/`：新编写的只读分析、完整性检查、GPU启动包装脚本。
- `outputs/`：只写新运行或离线分析结果；不会挑选最好的一次。

## 六、当前证据边界（非常重要）

- 可以验证：121 项固定案例工程测试、95 项原始网格测试、48 文件的应用源码身份；可从已有摘要复算四方法的 E2E JCT、P95、TTFT、工作流吞吐指标、四组消融；可从 Pilot 原始记录复算时延与原生 Host 统计。
- 尚无法单凭此包**直接重建或执行** GPU 推理服务；模型权重、服务镜像、动态 hook 部署细节和真实运行日志不完整。
- 无法从“跨工作流有共同前缀”直接推出“跨工作流 KV 命中率提升百分比”，也无法把原生跳过 token 直接换算成净 E2E 收益。
- **没有任何独立在线外部工具调用时延**，因为正式 MuSiQue 工作流节点均为 LLM 推理；不得用不存在的工具耗时填充图表。
- 四组中的 B3 是根据历史结果事后选择的优势案例；8轮是固定后新执行，不属于未见问题泛化评估。

如需**完整运行证据归档**，应另行从原 GPU 保留的 `full_raw.zip` 或当前新的完整 GPU 跑数目录取回；不要将本包 `review.zip` 的紧凑索引说成全量日志。每次 GPU 复现均应使用新的运行目录，保留失败条件、真实生成长度和负收益。
