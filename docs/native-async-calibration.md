# 异步原生标定协议与部署

## 最新现场结果与预测器修复

2026-10-16 用户提供的新版日志：单卡短诊断 122 条有效记录、0 排除；完整标定 2681 条有效记录、0 排除。decode/mixed 留出检查通过；prefill 的 p95 相对误差 0.4546219、RMSE/均值 0.3638580，未通过原阈值。短 prefill 约 101 ms，旧模型统一预测约 69 ms，多批次组的均值约 61–66 ms。由摘要推断可能缺少启动开销特征；本地没有 2681 条原始数据，不能声称已验证该因果解释或新的现场误差。

新预测器 `phase6_prefill_startup_v2` 的 prefill 公式为 `max(0.1, dot(original_six_features, nonnegative_coefficients) + startup_extra_ms * is_startup)`。启动系数只用 train 拟合，不硬编码 100/50 ms；decode/mixed 仍为原六特征。目标时间、阈值、样本数和范围检查不变。

训练中的 startup/pipeline 分类来自原始完成区间，另从提交调用的 `pending_before` 检查该分类在提交时已经可知；两者不一致或缺少提交状态就明确报错，不用未来完成时间给评分器提供不可观测特征。决策时的在途批次使用自身提交调用的 pending 状态；即时新批次使用该副本实际在途列表。完整候选推演以这个状态初始化，连续处理期间视作流水线，排空后恢复启动状态。未来流水线是否持续仍是预测假设，并非实测；字段 `prefill_startup_predicted` 与在途的 `prefill_startup_observed` 分开记录。未知/未训练状态回退，不填零。

`bash run_native.sh refit --input <已有完整标定步骤目录>` 只读原始事件、重建样本，再比较旧模型与新模型的留出误差及分组均值。此旧留出集已用于诊断，重拟合结果始终 `frozen=false`、`fresh_holdout_required=true`、`acceptance_granted=false`。新完整标定使用另一组预先确定的训练/留出种子；正式实验种子不变。新模型通过独立标定后才可用于正式验收。

用户明确要求先看实验效果时，可运行 `bash run_native.sh explore --max-conditions 24`，范围 1–96，默认 4。该模式允许已测量的预测误差不合格，使用指定历史模型（不偷偷替换为新预测器），但仍重建原始样本、验证真实机制和当前回执；样本不足、关联错误、覆盖不足不放行。只有副本 0 标定时可在有效调度参数相同的四卡上共用估计（物理 KV 容量保留各卡实测值），明确记录转用关系，不制造其他卡的验证结果。输出另存 `r234_exploratory`；正式 gate 不接受它，负收益和失败均保留。

这次预测器改动不需要重新安装已支持新计时字段的服务。安装及回滚脚本保留供尚未更新的副本使用。

## 已知问题与证据边界

现场报告：8896 条记录的 `eligible_host_batches=0`，旧筛选全部排除。旧筛选要求完成 A 的调用同时只提交 A，而 V1 的流水线允许提交 B、完成 A。当前本地没有现场完整 `native_events.jsonl`，不能声称已统计出每类数量。`run_native.sh audit --input <旧标定目录>` 离线去重统计真实组合、给出 call/batch ID 示例；旧文件以只读目录挂载，不发 HTTP 推理、不改旧结果。

## 新口径（不是原 host-step 的等价移植）

模型目标明确命名为 `async_host_completion_accounting_v1`。设批次 i 的提交调用开始为 S_i，完成调用结束为 C_i，前一完成调用结束为 C_prev。计费区间为：

```
begin_i = max(S_i, C_prev)  # 没有前一完成时使用 S_i
end_i = C_i
y_i = (end_i - begin_i) / 1e6
```

对于流水线 A/B/C，这些区间不重叠；首批计入启动时间，排除提交前空闲时间。它预测完成节奏的时间份额，并非该批次从提交到完成的延迟；不将“完成 A 的 host call”归到 B，不直接采用跨调用 `host_batch_span_ms`，也不使用辅助设备事件或 HTTP 耗时作为目标。主机返回之间的调度、RPC、调度间隙等开销包含在区间内，不做未经测量的扣除。

这是一种**新的吞吐量近似模型**。流水线深度、前后批次组成会影响区间，六特征模型不保证能解释全部变化。验收保留留出误差检验；达不到阈值就停止，不能因此宣布机制通过。与原 GPU 结果比较时必须披露该口径差异，不能称为原服务时间模型的等价复现。

每个 batch 使用自己的实际 prefill/decode token、decode context 总量、prefill context 最大值生成冻结实现的六个特征；APC 跳过量单独记录。模型仍按 prefill_only/decode_only/mixed 分相位。预测侧使用同一模型的完成时间份额，串行累加预测完成节奏；在途区间可能已部分经过，因此不是保守上界，也不是物理设备模拟。候选使用相同计时口径；超出冻结训练范围时记录原因并回退。原算法、资源约束、96 条件和工作流工作量没有改变。

## 原始证据与验收

- EngineCore 在真实 step 边界保存 `call_id`、起止时间、提交/完成列表、新增 `host_call_serial`、`pending_before/after`；每个完成批次保存 `completion_predecessor`。
- 调用同时挂在 A/B 上时，以副本与 call ID 去重。重复且冲突、重启混用、多个批次共享完成边界、缺失提交/完成、错误 pending 状态、交叉训练/留出组、缺失设备完成证据、负值或重叠区间均不能成为合格样本。
- 独立运行的训练/留出负载使用预先确定且不同的种子，组间排空；不随机拆同一运行中的 batch。同一运行内样本可能相关，不冒称每条记录统计独立。
- 完整标定每副本 28 组，新增并发 2；每阶段每个 split 至少 20 条，保留原 p95 相对误差 ≤0.35、RMSE/均值 ≤0.25，以及实际 batch 工作量覆盖检查。负载在观测到 decode 后再提交后续请求，避免固定 15 ms 延迟无法形成 mixed。
- 每组保存有效数、排除原因和阶段数量；计时关联失败提前停止，不能先跑完整个四卡采样再发现所有记录都不可用。低样本量的正常中间状态不提前判成不兼容。
- 验收重新读取原始事件、重建全部申报批次的区间、核对 samples、训练/留出归属和完整固定计划，再拟合并核对冻结模型、验证结果、当前引擎身份和实际动作回执。旧哈希只作溯源；不依赖签名或手填 PASS。
- `host_step_equivalence=false` 和 `original_gpu_timing_equivalent=false` 是正确标识；不要改成 true。短诊断模型永不冻结，不能替代完整验收。

## 最短现场流程

在新代码目录执行（旧目录不动）：

```bash
bash run_native.sh audit --input /data/wushibo/dabs-ascend-main-2026101603/outputs/native_test_20261016T073330Z_b2e635a7/04-native-calibrate
bash run_native.sh install --replica 0
bash run_native.sh smoke
```

短诊断成功后再执行：

```bash
bash run_native.sh calibrate
```

这是单副本完整标定：只有 `native_profile.json` 的 `pass=true`、`frozen=true` 才通过模型检查。若失败，保留所有记录并停止；不建议盲目重采。通过后才执行：

```bash
bash run_native.sh install
bash run_native.sh test --invalidate-cache
```

完整测试 `passed`、最终验收 `live_verified=true` 后：

```bash
bash run_native.sh experiment
```

安装仅更新实际缺少计时能力的现有服务。每次安装保存备份，在对应 `outputs/native_install_*/ROLLBACK.txt` 写入精确回滚命令；逐个重启现有容器，宽限 120 秒，等待健康。其它参数、镜像、模型、端口、资源配置不改。普通 smoke/calibrate/test 不安装、不重启。可以用 `--replica 1..3` 单独更新或验证其它副本。

输出是普通文件：`SUMMARY.txt`、`run.json`、`timing_audit.json`、`calibration_samples.json`、`calibration_excluded.json`、`calibration_progress.json`、`native_profile.json`、`native_events.jsonl`。只复制需要的摘要，不打包、不上传。安装/测试/实验都使用新目录；本轮不自动恢复旧测试阶段，也不会复用旧口径的失败模型。

## 本地测试边界

`python tools/run_ascend.py test` 包含 CPU 异步“提交 B、完成 A”样例、重复调用去重、原始事件重算、改写/删除样本拒绝、计时缺项、训练/留出交叉、重启失效和定向安装测试。CPU 成功不构成现场 NPU 通过；短诊断、三阶段完整标定及四卡验收仍需现场执行。
