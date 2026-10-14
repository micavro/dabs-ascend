# DAG 请求调度跨计算卡源码交接包

## 交付定位

这是已经运行过的 workflow 感知请求调度系统的**源码与实验输入配置交接包**。主方法为 **DAG（原 D₀，含 v2.1 准入意图修复）**，基线为 RR、P2C、Affinity；DAG-Lite 仅作为已有实现消融。没有重新设计调度器，不启用 adaptive batching，也没有声称非英伟达后端已经适配完成。

自研算法、workflow 构造/真实父响应传递、请求执行器、原生服务 hook、成本预测、基线、指标审计、实验编排、精简打印和测试源码均已包含。原生 hook 是原项目的 vLLM 0.6.3 接口参考，不是跨厂商通用插件。第三方推理引擎、驱动/SDK、模型权重、容器镜像仍是外部依赖。

**没有打包历史实验结果、生成响应、原始时间线、性能表、拟合误差报告或旧 run 目录。**`profiles/reference/`只保留原服务时间模型的推理系数与输出长度先验，供检查算法和参考执行；这不是对目标卡的重新标定。历史性能元数据已经去除，但实际用于推理的系数未改变。

## 先看哪些文件

- `docs/SOURCE_MAP.md`：真实算法、执行器、hook、基线、测试和对应函数。
- `docs/PORTING.md`：原生后端适配边界、共享时钟、缓存证据、预算与请求映射。
- `docs/EXPERIMENTS.md`：重点案例与完整验证计划，所有种子、顺序、Block 和条件已展开。
- `docs/CALIBRATION.md`：换卡后的服务时间标定和显式新 profile 登记。
- `docs/CHANGES.md`：本包只改了哪些装载/路径/计划代码，哪些算法原样保留。
- `docs/VALIDATION.md`：实际执行的本地验证范围。

## 目录

```text
source/
  spra_stage19_musique_joint_v1/
    semantic19/     # 原 D₀、DAG 编译和旧应用辅助
    core18v2/       # 完整 workflow 预测推进和相位服务模型
    joint18v2/      # 请求执行、真实状态、资源/缓存、发布和审计
    runtime/       # 原 vLLM hook 与服务 launcher
    tools/         # 原校准、数据编译、诊断与部署参考
    configs/ tests/ dataset/compiled/
  spra_stage19_typed_handoff_v1/
    typed19/       # canonical tokenization、typed 父响应、质量评估
    tools/ configs/ tests/
  spra_workflow_accept_v1/
    wfsched/       # v2.1 修复、RR/P2C、DAG/Lite绑定、实验和报告
    plans/ tests/ reference/
profiles/reference/ # 仅推理输入系数、先验、执行配置及源码身份
support/wf_ops/      # 已验证的精简屏幕/完整文件日志代码，不含旧结果
configs/            # 目标卡登记模板、套件索引
provenance/         # 原始源码哈希、文件来源、输入数据的预期哈希
port.py             # 迁移包入口
```

目录名中的 `joint` 是旧项目命名和导入路径；主入口只提交 A/D 路径或 RR/P2C 固定预算策略。原公共库中未调用的联合分支为保持源码完整性而保留，不能直接运行旧 `run_typed.py pilot` 并称为本包纯调度套件。

## 1. 不需要计算卡的检查

使用 Python 3.11+ 和已有控制端依赖。算法本身主要使用标准库，真实HTTP执行依赖 `httpx`；部分原测试/标定工具还使用 `numpy`。不要为运行这些CPU检查而安装 CUDA 版 PyTorch。目标推理引擎及驱动由目标平台提供，本包不自动安装任何软件。

```bash
cd spra_workflow_port_v1
python -B port.py verify
python -B port.py selftest
python -B port.py plan --suite port-cases --out selected_plan.json
```

`verify`会分别报告源码完整性和原数据文件是否存在。当前下载版中四个实际数据 JSONL 未包含，不能把 `present=false` 当作真实实验就绪；CPU测试不需要这些数据，不会冒充真实推理。`selftest`使用原生代码加CPU fake engine测试，不联系目标服务。

## 2. 从 GPU39 补入原始数据后转交目标卡团队

之前上传的代码审查包没有包含原数据正文，所以本包没有从模型生成答案反向拼造数据。需从 GPU39 已保存的编译题集复制：

- `smoke.public.jsonl`：2个应用级烟雾检查问题。
- `profile.public.jsonl`：32个独立校准问题。
- `pilot.public.jsonl`：64个正式历史问题，Block顺序保持原样。
- `evaluation_gold.jsonl`：只供执行后的质量评分，不进入调度器。

把本包解压到 GPU39 的**新独立目录**后执行：

```bash
python -B port.py export-on-gpu \
  --repo /home/wmy/sci/spra_paper_experiments_github \
  --out /home/wmy/sci/spra_paper_experiments_github/spra_workflow_port_inputs_v1.zip
```

该命令只验证原48项源码和数据哈希，读取明确白名单中的输入，输出新的源码+数据ZIP及`.sha256`。不复制任何历史实验结果，不修改原代码、结果或服务，不发模型请求。可以在原实验继续使用未改动的源码/数据时执行；若读取时内容发生改变会停止。不要覆盖原实验目录。

把生成的 `spra_workflow_port_inputs_v1.zip` 交给目标卡适配人员。最终ZIP仍解压为`spra_workflow_port_v1/`。数据许可/来源信息保存在原 `dataset/compiled/ATTRIBUTION.txt` 与 manifest 中。LLM权重和tokenizer请使用你们原有相同本地版本，不在此包中重复分发。

## 3. 非英伟达后端适配顺序

先完成 `docs/PORTING.md` 的真实 native 状态/缓存/提交/时间合同，完成目标卡标定，再进行新执行。仅有 OpenAI-compatible HTTP 并不能恢复原方法所用的缓存和竞争信息。

适配在此复制目录中进行，不修改GPU39的 `spra_workflow_sched_v2_1`。`python port.py verify`始终验证原交付版本；修改接口后的不匹配是正常的版本差异，须用新目标profile记录，不将原哈希手改成“通过”。`tools/target_profile.py`仅登记源码和输入身份，不替代任何native校验；详见标定文档。

原native gate检查四个真实服务、共享host monotonic时钟、实际设备身份、预算和缓存合同；它不会自动把 `nvidia_visible_devices` 变成“支持所有设备”，也不会伪造成功状态。目标卡团队需实现语义等价的真实设备身份检查。

## 4. 适配完成后的真实运行

以下命令**需要实际完成后端适配及目标profile**。不要在未适配的非英伟达卡上把原hook报错视为调度器安装故障。

```bash
python -B port.py gate \
  --runtime /absolute/path/to/target_runtime \
  --profile profiles/target_card

python -B port.py run --suite port-smoke \
  --runtime /absolute/path/to/target_runtime --profile profiles/target_card \
  --out /absolute/path/to/new_runs/port_smoke --confirm-device-use

python -B port.py run --suite port-cases \
  --runtime /absolute/path/to/target_runtime --profile profiles/target_card \
  --out /absolute/path/to/new_runs/port_cases --confirm-device-use

python -B port.py report --out /absolute/path/to/new_runs/port_cases
python -B port.py case-report --out /absolute/path/to/new_runs/port_cases
```

每个套件只用一个平台/代码/profile身份。选择`--profile`后不要在同一个工具目录里切换另一profile继续旧运行：原freeze会阻止混用，应另建平台副本或新工具目录。原schema当前要求四个服务位；R=1/2/3只使用前R个，其他服务必须真实存在且空闲。若新环境只有更少服务，需作为单独部署合同适配，不能创建假status凑四个。

`port.py run/gate`已接入原精简日志sidecar，屏幕显示条件级摘要，完整日志自动保存到 `logs/native/LOG_*/full.log`，另有 `summary.log`、`conditions.jsonl/csv`。原运行目录保存 `protocol.json`、`environment_contract.json`、`index.json` 和逐条件原始文件。子进程退出码与错误不被吞掉。

正常分段：在`run`命令加`--max-conditions 32`；下次使用完全相同参数和目录，加`--resume`。失败attempt不自动重试，不能删除坏结果后当作没有发生。全部套件串行使用独占服务。

## 重要边界

保持相同LLM并不意味着不同卡的数值结果、响应长度和服务时间完全相同。权重、tokenizer/chat模板、dtype、采样、JSON输出合同、问题顺序和计时边界应锁定；驱动、SDK、推理内核、内存容量和标定曲线如有变化要如实记录。原公共200ms循环、模型可见UUID缓存隔离的限制原样保留，不把这些公共机制的修改伪装成硬件性能差异。

本包无历史测量成绩，不保证目标卡上达到某个固定百分比。重点案例是从原平台开发记录选择的迁移起点，不是目标卡的已知最优。
