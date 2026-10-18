# Ascend Fast-One v2：RR / SLL / Ayo / Ours 单场景实验

## 本轮范围

适配基线是 `micavro/dabs-ascend` 的 `7b672c0496d587c4884ec7823f459898fe1f0ef0`。实验标识 `ascend-fast-one-r4l8-four-methods-v2`；轻量评分器标识仍为 `ascend-fast-request-event-v1`。

本轮集成此前未能推送的 Fast-One 增量实现，将默认方法改为 **RR → SLL → Ayo → Ours**，各一次。Reference 只用于 Ours 内部比较，不是第五个实验条件。场景固定 **R=4、λ=8、T1、B0**，沿用原始设计的种子190500、到达偏移、balanced DAG和固定token工作量。每条件64工作流、216请求、774144输入token、13184输出token；总计4条条件、864请求。不自动运行第二轮、其他负载或96条矩阵。

这是昇腾现有接口上的工程近似，不是原GPU V5完整批次推演的严格复现，不代表已经获得性能提升。旧算法、服务端、配置协议和实验结果不覆盖；原README保存在 `README_native_legacy.md`。本轮必须使用 `run_fast_one.sh`，旧 `run_native.sh` 仍运行旧实验。

## 从原材料找到的现场操作方式

来源是用户提供的《DABS_完整对话与日志版本档案_20261017_7b672c0.zip》和后续终端文本；并非本次连接到slave21取得的新观测。

档案 `01_完整对话.md` 的M0035保留了 `git clone --depth 1 https://github.com/micavro/dabs-ascend.git` 和后续 `git pull --ff-only`。该早期段落里的 `run_ascend.sh` 是旧入口，本轮不用。早期现场附件说明旧项目曾采用ZIP加手工补丁；最新终端给出 `c00878134@slave21`、`/data/wushibo/dabs-ascend-main-2026101704` 以及 `native_test_20261017T093920Z_ba538c36`。这些材料未证明最新目录存在 `.git`，不要直接假定它能够pull。

建议在Linux宿主机创建新的同级Git目录，显式只读使用旧证据。先正常结束旧实验控制器及其他使用四个服务的推理任务，模型服务保持运行：

```bash
cd /data/wushibo
git clone --depth 1 --branch main https://github.com/micavro/dabs-ascend.git dabs-ascend-fastone
cd dabs-ascend-fastone
bash run_fast_one.sh plan
```

计划必须显示四方法、`condition_count: 4`、`scenario: R4_L8_T1`。然后：

```bash
bash run_fast_one.sh run \
  --input /data/wushibo/dabs-ascend-main-2026101704/outputs/native_test_20261017T093920Z_ba538c36
bash run_fast_one.sh logs
```

只有服务自该次测试以来未重启、资源参数未变时，才可复用此证据。路径不同则指定实际对应当前服务实例的目录。不要默认执行 `install && test`。启动仍核对原始证据、冻结模型、现有引擎实例和资源，不绕过门禁；失败时保留原因，不自动重标定。

上述命令在宿主机执行，不进入已有模型容器。脚本复用副本0镜像，启动临时CPU控制器，源目录挂载到 `/project`，证据只读挂载到 `/evidence`。旧 `/project/outputs/...` 证据路径只在显式输入目录内重定位，不选择其他目录最新模型。

已经创建该Git工作树后，后续更新：

```bash
cd /data/wushibo/dabs-ascend-fastone
git pull --ff-only origin main
bash run_fast_one.sh plan
```

目录已存在时不重复clone。ZIP解压目录不能直接pull；本地有未提交修改或冲突时先保留文件，不使用 `reset --hard` 或覆盖outputs。现场网络是否可访问GitHub仍由现场确认，已有材料不支持增加特定代理或访问令牌。

## 四种方法的口径

| 方法 | 本版行为 |
| --- | --- |
| RR | 复用已有请求级轮询路由、FIFO准入。 |
| SLL | 复用原包k=2/P2C：确定种子抽两个副本，按逻辑在途负载选较小者；不是全副本最短队列。 |
| Ayo | 复用现有拓扑优先、保守KV预留/物理容量代理规则；不是论文引擎的严格实现。 |
| Ours | 现有affinity reference，加DAG准入/准入路由候选及请求完成事件级近似评分。 |

四种方法使用同一个事件驱动准入循环、原生观测、每副本2个客户端在途上限和真实原子准入接口。RR/SLL/Ayo不被强行增加Ours规划等待。显式 `--methods Ayo,Ours` 只缩小同场景诊断的方法数；默认不必设置参数。Reference不再是可选实验方法。

原始计划里的4096只记录为 `archived_gpu_base_cap`；实际引擎步预算、块大小、序列上限、上下文和物理KV容量从现有服务能力读取，写入 `effective_config.json`。本入口不会把GPU归档值覆盖到昇腾服务。

## 公共快路径

到达、后继节点就绪、请求完成会唤醒一个单写者调度循环。200ms是兜底检查，不要求请求等满200ms；3ms合并同时发生的事件。没有ready或所有客户端槽位已满时，不反复执行无意义的观测。

| 控制项 | 本轮初值 |
| --- | ---: |
| 兜底检查 | 200ms |
| 事件合并 | 3ms |
| Ours规划上限（含IPC） | 80ms |
| 客户端可用快照最大年龄 | 500ms |
| 提交预留 | 100ms |
| 观测等待上限 | 500ms |
| 每副本客户端在途 | 2 |
| 后台未回收规划 | 最多1个 |

这些是工程初值，不是现场已证明的尾部时延保证。已部署服务端1500ms最大TTL/动作窗口协议不变，客户端只采用更短状态窗口，不延长过期动作寿命。规划超时后执行reference；迟返回只记录，不套用到下一轮。

## Ours候选与评分

候选最多两个：`dag_admission` 和 `dag_admission_route`。候选可以改变本轮准入请求身份，而不仅修改本轮不进入引擎的held请求。新候选不沿用原V5的相同准入身份约束，但保留全部DAG、ready、客户端槽位、完整KV预留和固定引擎预算约束；准入数量不能少于reference。仅修改held路由或客户端列表顺序不计为真实动作差异。

候选优先级使用工作流剩余关键路径估计，优先更短剩余工作流，同工作流内优先影响尾部的节点；等待至少5s的最老ready请求获得候选构造优先。这不是严格5s等待上界。无效动作在计入名额前去重；一个目标拒绝后，路由候选继续尝试其他目标。

评分保留全部已到达DAG，以每副本串行请求服务成本近似推进完成事件，不模拟真实continuous batching，不读取未到达工作流。reference与候选采用同一评分器和相同后续FIFO-ready/合法预计完成时间策略，比工作流剩余完成时间之和。必须完成全部候选比较才允许选择。

服务成本复用现有冻结六特征系数与startup参数。Prefill保持实际步预算和chunk阈值；Decode使用仿射公式算术和并检查首尾上下文；范围不支持时回到reference，不静默外推。有效缓存证据可影响当前成本；unknown用明确标记的无缓存收益估计，不伪报实测miss；未来续排不沿用当前驻留证据。KV预留不扣假定缓存收益。

保留在途Decode完成保护和既有native next-batch mixed/decode保护。next-batch不携带本入口不能执行的队列优先级；保护所需模型不可用时拒绝候选。收益还要超过按实际worker耗时估计的决策代价，无强制非参考动作门槛。冻结batch模型验证通过不等于请求级近似排名已经验证，性能仍看现场同配置结果。

## 执行、证据和失败处理

本入口只调用原生 `admit` 申请新请求许可，不修改已提交请求的路由或引擎队列。所有实际生成请求必须取得真实许可，由既有Backend验证SSE工作量和引擎binding。并发HTTP发起顺序不被宣称为引擎优先顺序。某副本传输失败时等待其他响应并停止，不盲目重发。

原生观测不一致仍用保守reference原子准入，不将混合状态用于Ours评分。`nonreference_execution_confirmed` 由改变的准入身份/路由、终态引擎准入和已完成请求binding交叉确认，不因 `receipt.applied=[]` 就判断纯准入动作没生效。

不安装、重启、清缓存、新标定、canary、单/四副本探针、无截止影子计时或全网格。`verify_live` 从既有原始样本重算一致性并读取现有动作账本，这是只读核验，不是重新采集标定。服务已变、证据缺失时明确停止。

## 输出和检查

每次生成新目录 `outputs/native_fast-one_<UTC>_<后缀>/`，不覆盖旧结果。主文件为 `status.json`、`test_result.txt`、`metrics.csv`、`comparisons.json`。报告含四方法原值及Ours对RR/SLL/Ayo的三组改善率，负结果原样保留。完成部分方法时只报告已有数据，不补造Ours或胜出结论。

```bash
bash run_fast_one.sh logs
bash run_fast_one.sh logs --out outputs/native_fast-one_具体目录
bash run_fast_one.sh selftest
```

`logs`只读最新/指定结果，失败状态与异常同时打印。每条件完整原始请求、快照、决策、回执保存在 `conditions/*`；终端同时保存 `outputs/fast-one-console-*.txt`。`functional_pass`仅表示工作量和DAG通过，`formal_acceptance`仍为false，`performance_pass`不自动置true。不同旧周期/影子结果不得混算成新方法收益。

新增43项CPU测试覆盖单场景四方法默认值、RR/SLL原路由内核、共同执行循环、216请求/64工作流模拟运行、准入/DAG/KV约束、保护检查、模型范围、过期结果丢弃及日志/负收益。模拟时间与测试系数不是实测数据。这些测试不等同于完整旧仓库回归或910B3现场验收。

## 回滚

停止使用 `run_fast_one.sh` 即回到旧入口，不需要回滚或重启服务。保留新产生的失败和负收益数据。GitHub拉取已集成此实现，不要再覆盖之前v1 ZIP的三方法默认入口。不上传现场outputs、原始日志、模型或凭据。
