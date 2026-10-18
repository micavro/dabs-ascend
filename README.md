# DABS 昇腾：Fast-One 四方法单场景实验

本轮只验证 **R=4、λ=8、T1**。默认按 **RR → SLL → Ayo → Ours** 运行，各一次，共 **4 条条件、256 工作流、864 请求、52736 输出 token**。每条件仍为 64 工作流、216 请求、774144 输入 token、13184 输出 token。不会自动跑第二轮、其他场景或 96 条矩阵。

实验版本：`ascend-fast-one-r4l8-four-methods-v2`。Ours 是请求完成事件级的昇腾近似适配，不是原 GPU V5 完整 batch 推演的严格复现。RR/SLL/Ayo 复用仓库已有路由规则，四方法共用事件驱动准入、原生资源约束和真实执行接口。SLL 保持原包的 k=2/P2C 口径，不改成全局最短队列。Reference 只作为 Ours 内部对照，不是第五个实验条件。

## 在 slave21 宿主机拉取和执行

先正常结束旧实验控制器及占用四个服务的其他推理任务；保留模型服务运行。旧结果和标定目录不要删除。

首次建议拉到新的同级 Git 目录，不覆盖之前的 ZIP/手工修改目录：

```bash
cd /data/wushibo
git clone --depth 1 --branch main https://github.com/micavro/dabs-ascend.git dabs-ascend-fastone
cd dabs-ascend-fastone
bash run_fast_one.sh plan
```

确认计划为 `R4_L8_T1`、`RR,SLL,Ayo,Ours`、`condition_count: 4`，然后：

```bash
bash run_fast_one.sh run \
  --input /data/wushibo/dabs-ascend-main-2026101704/outputs/native_test_20261017T093920Z_ba538c36
```

这条证据路径来自已提供的现场日志，**只有服务自该次测试之后未重启、资源未改变时才可复用**。入口只读核验原始证据和当前实例；不自动安装、重启、重新采集标定、canary 或扩展实验。失败时保留具体原因，不绕过门禁。

同一 Git 目录后续更新：

```bash
cd /data/wushibo/dabs-ascend-fastone
git pull --ff-only origin main
bash run_fast_one.sh plan
```

目录已存在时不要重复 clone。若有未提交修改或拉取冲突，保留文件，不使用 `reset --hard`。ZIP 解压目录没有 Git 元数据，不能直接 `git pull`。

## 结果与测试

```bash
bash run_fast_one.sh logs
# 也可查看指定的已有结果目录：
bash run_fast_one.sh logs --out outputs/native_fast-one_实际目录
# CPU 自检，不发推理：
bash run_fast_one.sh selftest
```

每次新建 `outputs/native_fast-one_*`；不覆盖、不续接旧结果。`test_result.txt`、`metrics.csv` 给出四方法原值，`comparisons.json` 给出 Ours 相对 RR/SLL/Ayo 的改善率。负值保留；出错时 `logs` 同时显示失败状态和异常。

本地新增测试覆盖单场景计划、RR/SLL路由内核、四方法调度路径、216请求模拟运行、准入/DAG约束、模型范围、晚结果回退及报告。CPU/模拟接口通过不等于910B3性能通过；单场景一次重复也不证明统计显著性。

## 运行边界

宿主通过 `sudo docker` 复用 `wushibo-qwen-small-npu0` 的现有镜像启动临时CPU控制器；连接现有 `npu0..3`，端口 `8001..8004`，容器Python默认 `/usr/local/python3.12.13/bin/python`。不创建模型服务，不挂载NPU，不升级推理栈。客户端并发仍为每副本2。

事件唤醒+200ms兜底，3ms事件合并；轻量规划上限80ms（含IPC）、客户端快照年龄500ms、提交预留100ms。这些是起点参数，不是现场尾延迟保证。原服务1500ms协议不变。只改变新请求准入与路由，不迁移已提交请求、不重排引擎队列、不调整引擎batch预算。

详细机制、来源和异常处理见 [单场景说明](docs/ascend-fast-one.md)。旧原生实验与全部历史操作保存在 [旧README](README_native_legacy.md)。**这版必须使用 `run_fast_one.sh`，不要用旧 `run_native.sh experiment` 启动本轮验证。**
