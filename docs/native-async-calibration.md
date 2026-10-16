# 异步原生标定协议与部署

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
