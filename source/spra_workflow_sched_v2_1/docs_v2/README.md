# Workflow 请求调度扩展 v2：冻结原D₀，最小优化，原生执行

本包直接调用已经在GPU39成功执行的 typed19 应用、joint18v2 客户端和 semantic19 D₀。
没有DABS占位适配器，不要求实现 `describe()`、`run_condition()`，也不启动新模型服务。
本地测试不等于GPU性能验证。本次没有新的GPU JCT结果。

## 1. 方法与边界

|图表名|代码路径|含义|
|---|---|---|
|RR|wfsched.routing.reference|就绪请求FIFO、请求级轮询副本|
|P2C|wfsched.routing.reference|随机选两个不同副本，选当前未返回请求数较少者，平局由随机候选顺序决定|
|Affinity|原client的A分支|原缓存亲和基线，不更换实现|
|DAG|原semantic19.selector.choose，method=D|已经复现的原D₀，主方法冻结参照|
|DAG-Lite|wfsched.selector.choose_lite|只省掉D分支不会评分/执行的预算候选生成，独立优化候选|

RR/P2C是公共原生执行器中的策略实现，不是完整NGINX/Ray Serve性能复现。
所有方法固定prefill cap与total为4096，保留原生continuous batching、缓存、资源检查。
RR/P2C不读取DAG、前缀、未来长度来选副本；共享保守KV准入检查，资源不足时保留FIFO前缀。
这项中央准入包络及200ms公共调度时钟属于实验装置，不声称是原厂serving系统的完整默认路径。
P2C的负载是本条件已经提交但尚未返回的请求数，包含本次决策已安排的请求，不是D₀的加权压力。
轮询/随机序列依据实际提交数；反复构造参考动作不会消耗全局游标。

DAG-Lite不修改DAG候选、完整workflow评分、冻结服务模型、长度先验、保护条件和截止时间分配。
原DAG在截止时间内获得的候选完成数可能因省去计算而改变，因此不声称真实动作轨迹完全等价。
本次不改公共200ms事件循环，不更换prompt，不重训模型，不改后端代码，不做adaptive batching。
原版仍是DAG；DAG-Lite没有提前被宣布为更快版本，更不会逐工况事后取最小值改名为主方法。

## 2. 冻结与依赖

GPU根目录：`/home/wmy/sci/spra_paper_experiments_github`

原生runtime：`/home/wmy/sci/spra_core_exp/joint_e1`

必须已存在：

```text
spra_stage19_musique_joint_v1/
spra_stage19_typed_handoff_v1/
spra_stage19_typed_handoff_v1/prepared/MUSIQUE_20261010T092711Z_be09912a/
```

本包保存了原复现工具zip、其入口原文及冻结profile证据。`freeze`在服务器上核对原profile中的48项源码哈希、模型、先验和数据校验，再生成：

```text
frozen/D0/D0_reference.zip
frozen/D0/receipt.json
```

归档包括源码、冻结服务预测模型、先验、配置、数据清单和原复现工具，不打包权重或私钥。
归档不是只读文件系统/防篡改硬件；续用时以逐文件哈希和归档哈希校验。内容不同则停止，不覆盖。
数据问题ID/选择规则被冻结，真实输出与线上状态没有被冻结。
原D₀/A的两轮真实复测主指标为+7.78%，不是本包的新结果，也未确立跨工况10%。

环境沿用 `spra-exp`，Python 3.11+（原执行器使用asyncio.timeout），httpx/numpy等沿用原依赖。
不要为本包升级PyTorch/CUDA/vLLM。检查缺失依赖时应先核对是否激活了原实验环境。

## 3. 已声明实验

|suite|条件|方法|正式条件数|
|---|---|---|---:|
|smoke|R4、λ2、B1|全部5种|5|
|ablation|R4、λ2、原4个Block、每Block全部6种方法顺序|Affinity、DAG、DAG-Lite|72|
|grid|R=1/2/3/4，λ=1/2/4/8，4Block，两轮镜像顺序|全部5种|640|
|confirm|相同16工况、4Block、10轮Williams顺序|全部5种|3200|

每条件16个workflow；原4Block分别53、55、53、52个LLM请求。
因此smoke为265个正式请求、ablation为3834个、grid为34080个。
预热每次激活副本子集固定3R请求，不纳入正式指标；不预热到某个延迟阈值才开始。
切换副本数或续跑新session时预热，数量由事先声明的执行顺序决定。

`grid`是开发扫描，不是充分统计确证；两轮镜像不保证五方法的完整位置/前序平衡。
`confirm`提供每工况每Block的位置和一阶前序平衡，但不消除输入/设备波动等所有混杂。
不建议首次直接跑confirm。新增基线与原DAG都需要真实小检查后再扩大。

λ是整个系统的名义workflow到达率，不是每副本到达率、也不是LLM请求率。
沿用原wave：1秒活跃时2λ到达、1秒空闲，16个workflow截断；不把短cohort完成率当最大容量。
R选择原gate已验证的前R个服务ID，不修改CUDA_VISIBLE_DEVICES；其余服务继续运行并检查未接收新任务。
选择较少服务不等于模拟关掉其余GPU，也不支持能耗或资源释放结论。
R1中RR和P2C副本路由相同，可诊断噪声与控制开销；DAG仍可能改变请求时序，不能预设其必与基线完全同速。

## 4. GPU39运行

在新包目录中执行；不覆盖旧复现目录或原工程。下载压缩包SHA见随包外部校验文件/对话。

```bash
conda activate spra-exp
cd /home/wmy/sci/spra_paper_experiments_github/spra_workflow_sched_v2
bash run_checks.sh
bash run.sh freeze
bash run.sh gate
```

`run_checks.sh`仅进行本地单元/模拟后端测试、原生导入和只读哈希检查，不发送GPU推理请求。
实际服务检查gate保留原hook、引擎身份、时钟、预算、排空条件；同时验证所有4个既有副本。
如果gate未通过，不重启/升级，也不编辑预期哈希绕过。保留终端和日志排查。

资源独占确认后，先运行smoke：

```bash
bash run.sh run --suite smoke --out runs/smoke_v2 --confirm-gpu-use
```

预期出现 `WF_RUN_STATUS=complete`、5/5条件、固定4096、预算切换0、排空和源码未改检查通过。
smoke只测B1且单一顺序，不作为性能结论。核对后执行小型优化消融：

```bash
bash run.sh run --suite ablation --out runs/ablation_v2 --confirm-gpu-use
bash run.sh report --out runs/ablation_v2
```

无论DAG-Lite是否更快，原DAG保留；不删除慢条件。然后进行网格：

```bash
bash run.sh run --suite grid --out runs/grid_v2 --confirm-gpu-use
bash run.sh report --out runs/grid_v2
```

上述步骤分开执行，每步检查后继续。程序串行运行，使用既有runtime的experiment.lock；不并行争抢同一服务。
该锁不控制其他用户/非合作脚本，请仍保证GPU独占。

### 可选：分段执行完整网格

从新目录开始，先执行80个条件并正常排空暂停：

```bash
bash run.sh run --suite grid --out runs/grid_v2 --max-conditions 80 --confirm-gpu-use
```

继续同一计划（新增条件数可再次限制）：

```bash
bash run.sh run --suite grid --out runs/grid_v2 --resume --max-conditions 80 --confirm-gpu-use
```

也可使用 `--replicas 2 3`先执行某些副本工况。这只是过滤原640条件计划，报告仍显示部分完成，不能改称完整实验。
`--resume`只续跑正常暂停/准备好的计划，核验原始文件哈希、方法身份、数据、配置和环境。
失败/中断attempt不会自动重跑，更不会在多个尝试中选最好的一次。不要删除失败目录绕过检查。
改算法、改计划、改profile应建立新实验版本和新输出目录，不能沿用同一运行ID。

## 5. 产物、复算与回传

```text
runs/<name>/
  protocol.json / protocol.sha256.json
  extension_source.zip
  environment_contract.json
  index.json / suite_status.json
  attempts/<condition>/RAW_<uuid>/
    nodes.json, controls.json, plans.jsonl, backend.jsonl, ...
    summary.json, quality.json, typed_responses.json
    semantic_proof.json, scheduling_audit.json, inactive_proof.json
  sessions/<session>/cleanup.json, environment.json, ...
  analysis/REPORT.md, RESULT.json, paired_metrics.csv, cell_gains.csv, conditions.csv
```

调用原native检查器，并复算节点时钟的workflow JCT；检查实际父响应绑定、节点数、输出量、原生cap、控制时间、路由、非活跃副本无工作。
报告按同R/λ/轮次/Block配对；质量报告百分点变化，保留原始输出，不为了等工作量强迫EOS长度。
当前报告未做对同64题的伪独立置信推断。不同方法的P95仍是各条件P95，不是拼接样本后的P95。

导出全部原始证据而不是只导出较好的汇总：

```bash
bash run.sh export --out runs/ablation_v2
bash run.sh export --out runs/grid_v2
```

对应生成 `runs/ablation_v2_results.zip` 和 `runs/grid_v2_results.zip`，以及各自 `.sha256`。
导出已存在时停止，防止覆盖之前回传的证据。

CPU服务器取回（按实际完成的suite修改文件名，不上传私钥）：

```bash
cd /home/wmy/paper/sci
DEST=$(mktemp -d "$PWD/WORKFLOW_results_XXXXXX")
REMOTE=/home/wmy/sci/spra_paper_experiments_github/spra_workflow_sched_v2/runs
scp -P 8822 -i /home/wmy/ustc87 -o IdentitiesOnly=yes \
  "wqx@10.18.2.128:$REMOTE/grid_v2_results.zip" \
  "wqx@10.18.2.128:$REMOTE/grid_v2_results.zip.sha256" "$DEST/"
(cd "$DEST" && sha256sum -c grid_v2_results.zip.sha256)
```

## 6. 仍需披露的实验边界

原64题都是历史开发题，不是未见问题或完整MuSiQue评测。已知标注DAG、证据预算2700、当前较低EM/F1都应披露。
原输入隔离UUID放在模型可见system中，仍被保留以避免本次混改。相同问题/到达种子不代表各方法prompt tokens完全一致；根输入哈希、实际输出、质量均保存。
不能直接复用UUID而不清缓存，这可能给后跑方法热缓存优势。公共事件循环和缓存隔离改进应做所有方法同时适用的独立实验。
共享原200ms循环有利于延续已复现实验，但不足以证明对低开销原厂serving实现的完整系统优势。

本次48组实际CPU候选检查全部动作一致，但DAG-Lite存在一次约49.95ms长尾。
原DAG均值1.2754ms，Lite均值2.0881ms；中位数分别1.1302ms和0.9843ms；44/48对Lite较快。
所有样本保留，原因未确立。不能以中位数替代均值声称已经获得整体控制加速，更不能推导GPU JCT收益。
DAG-Lite只作为候选，最终以独立GPU对照结果决定是否替代DAG，不做事后逐格最优拼接。

测试详情见 `review/`；方法与外部来源见 `METHOD_AND_BASELINES.md`；小补丁见 `review/selector_delta.diff`。

交付包还包含 `reference/D0_source_snapshot.zip`：按本次用户上传源码制作的56文件快照（48项原源码及配置/模型/工具），其来源与哈希见同目录manifest。它不包含数据正文，GPU39的freeze仍需现场独立核验。
