# 旧MuSiQue D₀纯请求调度：定位与原生复测

## 结论与边界

定位到检查点 `MUSIQUE_SCHEDULING_D0_20261010`，对应：

- 实际应用入口：`spra_stage19_typed_handoff_v1/tools/run_typed.py`；
- 调度器：`spra_stage19_musique_joint_v1/semantic19/selector.py::choose(...,'D',...)`；
- 原生执行器：`spra_stage19_musique_joint_v1/joint18v2/client.py`；
- 原历史run：`spra_stage19_typed_handoff_v1/runs/PILOT_20261010T092749Z_421e356e`；
- 冻结profile：`spra_stage19_typed_handoff_v1/prepared/MUSIQUE_20261010T092711Z_be09912a`。

检查点记载D/A平均JCT配对改善13.3038%，四个不同16题block均为正。原始pilot条件明细被广域采集器的32MiB证据预算排除，故本地此次并未用原nodes/backend重算13.3038%。模型、先验、dataset manifest、48项源码均已获得并核对。数据正文仍在GPU39，不打包到这里。

本包直接复用原 `run_typed.run()`、`prepared()`、原生 `execute()` 和 `semantic19.selector.choose()`，没有 `describe()`/`run_condition()` 占位函数，不使用DABS v2/v3，不复制重写D₀。

**这是同应用、同先验、同D₀、同负载的A/D复测，不是20条件原suite的逐项原样重跑。**它省略B/Decoupled/Joint，原方法间位置和服务历史因此改变。原时间戳、缓存盐和真实生成也不可能被宣称恢复为历史状态。没有GPU性能保证。

## 保留什么

R=4；2 workflows/s；原wave（每2秒周期，首1秒以4/s到达）；64个历史dev问题，四个互不重叠16题块；固定4096；自然EOS、max96；2700文档budget；真实父答案；同32个train题profile拟合出的冻结模型与输出长度先验；原200ms执行节拍；所有控制/等待计入JCT。

D仍只选择固定预算下的00/10动作。为复现而保留旧D内部未使用的budget proposal计算，不将此计算误称为预算实际自适应。后续精简应作为新实现另测。

初始长度实际按hop读取p75，再取int：2-hop为9，3-hop为10（p75=10.75），4-hop为10。配置48仅是缺失先验时的回退，不能直接改48为8声称修好了旧方法。

## 本包的必要差异

1. measured方法限定A、D；每次8条件，128次workflow执行，426个LLM请求，另12个固定预热请求。
2. 第一轮保留旧20条件中A/D的相对顺序：A/D、D/A、D/A、D/A。反向轮逐block镜像，使两轮合计每个block各有一次A先与D先。两轮计划在查看性能前决定；不按负收益重跑。
3. 报告改为A/D配对汇总，并写入实际wrapper/模型/先验哈希和“非完整20条件重放”标记。
4. 验证原生实际cap=[4096]、零budget switch、控制向量与请求关联；不会通过`force_caps`禁用调度。旧执行器中force_caps分支会绕过D的规划，不能拿它当纯调度开关。
5. 默认借用已存在case23的独立tail checker修正，唯一源码适配是把相对导入的版本常量放到文件内。它在执行后检查batch，不改变原生引擎/调度/输入/时钟。可显式`--checker legacy`使用旧检查器，尾块假阴性仍会报错，不能当作CUDA故障。

不会修改任何旧文件。原执行器依旧在typed原目录的`runs/PILOT_<新UTC时间>_<UUID>`下新建输出。目录名PILOT和底层打印前缀沿用原代码，不意味着运行了联合方法；真实方法以manifest中的A/D和新增`workflow_only_result.json`为准。

## 部署与自检（GPU39）

将本包解压到 `/home/wmy/sci/spra_paper_experiments_github/`，与两个原工程并列。不要覆盖它们。

```bash
conda activate spra-exp
cd /home/wmy/sci/spra_paper_experiments_github/spra_workflow_d0_repro_v1
bash run_checks.sh
```

只读检查48文件哈希、冻结artifact、dataset manifest。`inspect`不连后端；若采集副本缺少数据正文，会明确显示missing，但不伪称已具备运行数据。`reproduce`会严格检查真实数据哈希与64个unique IDs。

## 先核对已经找到的原历史记录，不发生成请求

```bash
python -B workflow_only.py audit-history
```

输出在本包`audits/history_<时间>/`，包含旧A/D摘要配对表和`historical_AD_evidence.zip`。该核对不导入原工程、不访问后端、不改旧结果；它验证保存摘要中的有效性/cap/关联计数，不声称已经独立重放完整backend.jsonl。缺失文件不补数。

## 现有真实后端检查

```bash
python -B workflow_only.py gate
```

沿用原引擎gate：runtime=`/home/wmy/sci/spra_core_exp/joint_e1`，HTTP 127.0.0.1:8000–8003，原hook hash、4K限制、时钟域、4个服务的设备标识、服务排空与健康检查。共享`experiment.lock`由原程序持有。

gate不加载模型或发生成请求，不执行stage/apply/restart。它不能阻止不使用共享锁的外部GPU作业；仍需四个副本独占。当前GPU状态本次没有远程确认。

## 两轮纯调度复测

先完成一轮，每轮都保留所有正负结果与失败记录。两轮均只比较A和D。

```bash
bash run_reproduce.sh --confirm-gpu-use
```

该命令会发送12个固定预热和426个正式LLM请求；不自动追加重复，也不自动重试失败条件。成功后检查`WORKFLOW_ONLY_RESULT=`及`CLEANUP`，并确认输出完整8条件。

在原先就计划好的镜像次序下运行第二轮（不是第一轮负收益时才运行）：

```bash
bash run_reproduce.sh --order reverse --confirm-gpu-use
```

第二轮是新的独立run，不能覆盖或替代第一轮。两轮后按block及round配对报告，不能选择较好轮。这里没有提供“取最好结果”的选择器。

非标准路径可显式指定：

```bash
python -B workflow_only.py inspect --repo /实际项目父目录
python -B workflow_only.py gate --repo /实际项目父目录 --runtime /实际已有runtime
bash run_reproduce.sh --repo /实际项目父目录 --runtime /实际已有runtime --confirm-gpu-use
```

复测固定R4/rate2，不允许偷偷把此处扩成新版网格；扩大副本/负载和新题集应在该历史配置复现后单独制定。

## 结果与失败

每个条件仍由旧执行器生成nodes/plans/controls/backend/summary及typed真实父绑定；本包增加`workflow_only_audit.json`。suite增加`workflow_only_result.json`和`workflow_reproduction_receipt.json`。

如果失败，不重装、不重启、不编辑哈希、不用fixture替代；保留控制台和失败目录。旧执行器已生成的条件保持不动。本包暂不自动断点续跑，避免重新执行已接受条件。

从原始代码的独立checker发现的未知分支、抢占或字段缺失仍失败；tail修正不放宽这些条件。

## 本地验证

详见review目录。包含新入口的单元/原模块导入检查、原semantic/typed源码测试，以及spawn进程调用真实D选择器与冻结model的检查。测试没有目标GPU，没有重新执行真实MuSiQue请求，不能据测试通过声称已复现13.30%。
