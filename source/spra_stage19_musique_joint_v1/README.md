# Stage19：MuSiQue真实文本DAG与Joint实验

## 研究主线与本次边界

在已经验证的四副本vLLM运行环境中，执行 **MuSiQue-Ans v1.0公开多跳问答的原有分解DAG**；根据依赖、真实前缀驻留和副本运行状态，比较 workflow/DAG-aware scheduling 与 adaptive batching 的完整组合。方法名继续使用A、D、B、Decoupled、Joint，不把新候选名字作为新论文方法名。

这不是把Stage18合成token换个数据集名字。每条2–4 hop问答会执行其标注中的实际子问题，`#1/#2`引用**前序子问题的一基位置**，不是全局子问题ID。下一节点只在真正的父请求结束后组装输入；使用实际父输出，不用标准子答案teacher forcing。节点数和依赖边均不增加，不把串行问题改成并行，也不新增一个合成root/sink。

MuSiQue本身由已有单跳问题组合而来，是公开研究数据集，不是生产流量trace。使用标注分解属于**已知计划(oracle/annotated plan)信息条件**；所有对照获得相同问题和依赖执行器。Joint能提前知道图，不能提前知道真实生成内容、实际输出长度、未来流量或未来缓存。

本代码包不附带官方完整数据。数据获取脚本使用官方仓库指向的Google Drive v1.0 archive，并记录本地SHA256。当前构建环境无法下载完整archive；已核对官方地址、格式和指标源码，并以明确的合成夹具测试导入，不宣称已验证全部真实记录。

## 第一阶段：CPU服务器获取和编译真实数据

下载代码zip到 `/home/wmy/paper/sci/`，解压并进入本包：

```bash
python3 -B tools/fetch_musique.py
python3 -B tools/pack_dataset.py
```

默认脚本不安装gdown/torch/transformers，使用标准库从官方archive下载。Google返回网页、限流或网络不可达会明确停止，绝不回退到假数据或不明镜像。

手工从官方仓库README的数据下载链接取得 `musique_v1.0.zip` 后可改用：

```bash
python3 -B tools/fetch_musique.py --archive /home/wmy/paper/sci/musique_v1.0.zip
```

或者显式指定官方格式的两个文件：

```bash
python3 -B tools/fetch_musique.py --train /path/musique_ans_v1.0_train.jsonl --dev /path/musique_ans_v1.0_dev.jsonl
```

官方仓库：https://github.com/StonyBrookNLP/musique

数据准备固定选择98个问题：train中2个smoke、32个开发profile，dev中64个pilot。选择根据原图拓扑分层和固定hash顺序，**不看生成质量、时延或Joint收益**。单根链、分叉汇合等原图均保留，具体分布由真实文件决定并写入manifest。坏格式/自引用等拒绝原因计数保留。分解标签文本是已声明的先验；答案、别名、支持段落标签分离到 `evaluation_gold.jsonl`，不传给运行时/worker。

获取和编译成功后得到：

- `dataset/compiled/{smoke,profile,pilot}.public.jsonl`
- `dataset/compiled/evaluation_gold.jsonl`：仅完成后评分
- `dataset/compiled/manifest.json`：原文件/选择/编译哈希及图覆盖
- `dataset/compiled/ATTRIBUTION.txt`

`pack_dataset.py`在本包父目录生成 `spra_stage19_musique_data.zip` 和 `.zip.sha256`。将**代码zip、数据zip、数据SHA文件**全部上传到 `wqx@10.18.2.128:8822` 的 `/home/wmy/sci/spra_paper_experiments_github/`。已存在的编译manifest与数据zip不会静默覆盖。

## 第二阶段：GPU39语义smoke

在GPU39使用已有 `conda activate spra-exp`，进入**新的本包目录**，不要覆盖Stage18：

```bash
bash run_checks.sh
python -B tools/run_semantic.py plan
python -B tools/run_semantic.py gate
bash run_smoke.sh
```

smoke执行预留的2个真实train问题。它验证真实依赖与父输出传递、自然EOS、请求对应、原生批次和质量记录；不是重复旧hook部署canary，也不是性能比较。不加载旧服务模型，测试中同样不会用gold修复错误答案。

`REAL_WORKFLOW_SMOKE.pass`只表示执行正确，不表示答案正确。要同时读`quality.final_em/final_f1`、UNKNOWN/格式和length截断记录。错误答案继续传下游并照常评分，不注入正确答案。

## 第三阶段：同数据集开发并冻结模型与固定参照

```bash
bash run_profile.sh
```

32个新的train问题分4个block，每block8问题；每block分别使用固定prefill cap256/512/1024/2048/4096，共20条件。每个条件都运行真实问答与质量评分。服务预热是固定8个短/长请求，独立记录，不计入测量。

本阶段从真实batch重新准备各副本prefill/mixed/decode服务曲线，旧Stage18模型仅作未获充分支持阶段的先验/回退。阶段拟合使用完整block留出，不使用答案或JCT作为服务时间标签。输出长度先验来自实际生成，并在pilot前冻结；不是从gold答案长度得到。

A/D固定cap按此批平均JCT选取，同时要求EM与F1不比固定1024低超过2个百分点。所有五档保留。此选择只用train，不读取dev结果。它是已测集合内的开发参照，不是全局最佳配置。

默认先验路径：

```
../spra_stage18_joint_core_v1/runs/CALIBRATE_20261008T152619Z_edfd9e63/host_model.json
```

路径不同时：`bash run_profile.sh --legacy-model /actual/host_model.json`。不要为缺路径重跑旧标定。

## 第四阶段：同版五方法完整Joint比较

```bash
bash run_pilot.sh
```

64个dev问题，4个block×16问题；每block五方法，共20条件。各block问题集合不同，同block五方法相同问题、到达、图和context规则。仍为平均2 workflow/s的原wave形状，不称为4种随机到达分布。没有D/B正收益门槛，不自动调整参数、重试或挑选有利block。

| 方法 | 调度 | 额度 |
|---|---|---|
| A | Affinity/FIFO和共同保护 | 同数据开发选定fixed |
| D | 已知DAG的部分分组、顺序、有限准入候选 | 相同fixed |
| B | A的调度规则；不做无用调度候选生成 | 额度候选 |
| Decoupled | 同一S0/S1调度候选先比较 | 再比较当前调度行的B0/B1 |
| Joint | 同一快照的S0B0/S1B0/S0B1/S1B1 | 完整成对比较 |

B是动作消融，共同价值模型仍使用已知DAG；不是完全无DAG信息基线。主比较是Joint对A和Joint对Decoupled。所有模型和数据在测量前冻结。

## 输入与质量协议

- 模型/原生hook/4096上下文/4096硬总额度不变。
- 所有提供的段落都保留标题；按**不看支持标签的等额前缀截取**分配总计2700文档tokens，不足预算不填充。保存每段原长、保留长度与clipping标志。
- 这不是MuSiQue原始完整上下文排行榜协议。它是4K环境下的已知计划阅读问答适配，不是完整检索系统。
- 动态组装的真实父响应若导致4096超限，明确失败并保留；不静默截断父响应或用gold替换。
- 自然EOS，max_tokens=96；无min_tokens、无ignore_eos。短答案和一条简短证据由相同prompt要求。
- 生成内容与长度可以不同。报告实际输入、输出和quality，不声称不同算法严格相同token工作量。温度0也不被当作跨批绝对确定性的保证。
- 每条件短run nonce用于被测缓存隔离，有微小token化差异；没有物理快照reset声明。
- 答案EM与token F1按官方归一化方式实现，支持别名。未实现官方support/group-answerability leaderboard，不混报。
- 质量/TPOT保护失败的条件仍完整报告。2个百分点质量和3%TPOT是预声明开发判定，不是统计非劣保证。

## 计算与时间语义

已知DAG的祖先/后继/反链提前编译；在线只物化已具备依赖的真实子输入，形成一个结构性调度备选和一个post-placement额度备选，再进行有限完整比较。仍使用修复后的因果评价内核，**没有声称在线完全不做搜索**。

80ms统一CPU预算含候选生成和评价，提前为结果交接留时间；完整评分缺失时不排名。所有ready→提交→完成等待计入JCT。prefill局部步时和已有decode剩余完成预测作为候选保护，阈值是公开设计起点，不是经过验证的误差上界/TPOT定理。所有合法cap仍保留；当前后端没有新增逐请求任意chunk接口，而由准入、顺序和独立prefill cap影响原生批次。

## 输出

`SEMANTIC_CONDITION_DONE`是完成后附带质量的正式条件结果。内部继承代码也打印较早的`CONDITION_DONE`，不要重复计数。

最终：`REAL_PROFILE_FROZEN`、`MUSIQUE_JOINT_RESULT`、`CLEANUP`。按请求报告Mean/P95 TTFT、node-first-content、TPOT代理；按workflow报告Mean/P95 JCT；报告EM、F1、EM-completed workflow/s、真实生成量、token截断率与因果绑定证据。

批次吞吐终点已改为最后一个实际请求完成，不使用runner下一次醒来的时间；额外保留runner observation lag和旧分母，不从JCT中扣搜索时间。

## 不要执行

不执行stage/run_apply/restart，不修改运行中的hook，不升级vLLM或模型，不重复旧20条件或暖机制。保留的旧Stage18工具仅用于持续回归，本轮入口只有 `tools/run_semantic.py` 和三个run脚本。`run_prepare.sh`已停用，避免误跑旧合成profile。

## 本地验证范围

195项CPU/mock测试通过，包括合成格式archive导入、无标签runtime、真实父响应绑定、自然EOS、五方法实际client/hook假引擎链路、成对候选、质量/冻结和吞吐边界。

测试使用Python3.13；源码按Python3.11语法检查。原有E2E假引擎fixture增加35ms等待，让已完成前缀的idle观测在假HTTP返回前稳定发布，原生产检查未放宽。

未在GPU39、真实vLLM、真实数据完整archive上执行，也未验证Google Drive在CPU服务器的网络可达性。无任何新性能收益或10%结论。
