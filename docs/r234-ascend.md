# 20261015 R234 新实验的昇腾执行说明

输入：`DABS_R234_Balanced_SLL_FullSource_20261015.zip`，SHA256 `58f102b2695fc770783347622f3593b43a414d8ee16d04776ed3b1f5baef645a`。

373 个文件中 317 个与此前工程一致，新增 56 个；全部保持原始字节，清单在 `provenance/r234_import.json`。没有在这份包中发现昇腾实测结果或模型权重。原包工具 `tools/run_dabs_r234.py`、`tools/dabs_r234_preflight.py`、`tools/plot_dabs_r234.py` 依赖原生 GPU 执行链，完整保留供核对；昇腾现场使用 `run_dabs_r234.sh`。

## 运行与环境

在更新后的项目根目录、Linux 宿主机运行：

```bash
bash run_dabs_r234.sh
```

默认现场配置与 README 一致：服务容器前缀 `wushibo-qwen-small-npu`，选择设备 `0,1,2,3`，端口基数 `8001`，宿主机模型 `/data/model/Qwen2.5-1.5B-Instruct`，控制端解释器 `/usr/local/python3.12.13/bin/python`。每个条件只用前 R 个所选服务。没有升级推理环境、重启服务、下载模型或占用第二份 NPU 权重。

`DEVICE_IDS`、`MODEL_PATH`、`CONTAINER_PREFIX`、`CONTROLLER_PYTHON`、`BASE_PORT`、`MAX_INFLIGHT` 沿用现场入口的环境变量。例如：

```bash
DEVICE_IDS=4,5,6,7 bash run_dabs_r234.sh
```

每副本默认最多 2 个 HTTP 在途请求，服务端保持原配置；原包 native controller 使用 64，二者不可直接作为同部署性能比较。若要提高客户端并发，应整次实验统一设置并在结果中保留参数，不能中途只改某一种方法。

新程序依次执行：源码与数据校验、模型元数据/容器映射/HTTP 健康检查、每副本 5 种长度的固定 token canary、每副本 8 次 HTTP 标定、64 个工作流编译、96 条件、逐条件与总报告。预热和标定不算正式实验请求。已有服务需独占给本次实验，程序无法阻止其他用户向服务发请求。

## 固定的实验内容

| 项目 | 新包与本入口 |
|---|---|
| 条件数 | 96：3 个副本数 × 4 个到达率 × 4 方法 × 2 轮 |
| 副本数 | 2 / 3 / 4 |
| 到达率 | 4 / 8 / 12 / 16 工作流每秒 |
| 到达形状 | 1 秒 ON（2λ），1 秒 OFF；seed=190500 |
| 方法 | RR、SLL（k=2 的 P2C）、Ayo 拓扑代理、Ours（DAG-H V5） |
| 调度间隔 | 200ms，所有方法一致 |
| 第一组 | R4、λ8，依次 RR / SLL / Ayo / Ours |
| 第二轮 | 方法顺序和 R/λ 遍历顺序均镜像反转 |
| 每条件 | 64 工作流、216 请求；来自 24 个固定公开源样本的 balanced 重加权 |
| token | 每请求输入 3584；按图形输出 16/32/48/64/128，合计 13184 |
| 依赖 | 保留原 DAG；子节点等父节点完成，但输入是固定静态回放 |
| 质量 | 非语义工作负载，不报告 EM/F1 |

原包 `design.plan()` 继承的若干说明仍写 16 工作流/旧采样口径；实际条件中是 64。适配使用原始 96 个条件对象，运行协议与报告依据新工作量明确记录，不把遗留说明当作实际执行内容。

## 过程与结果保存

`R234_OUTPUT=` 对应本次输出目录。容器 `/project/outputs/...` 与宿主机 `<项目目录>/outputs/...` 是同一位置。程序每完成一个条件就更新报告；复制结果时请保留整个目录，不只留截图。

| 文件 | 用途 |
|---|---|
| `REPRODUCE.md` | 实际命令、路径、执行链、文件位置、复现过程 |
| `process.jsonl` / `console.log` | UTC 开始结束、每条件执行模块、输出路径、参数与控制台 |
| `protocol.json` / `import_integrity.json` | 不可变计划、源码哈希、原包校验 |
| `environment.json` / `servers.json` | Python/依赖/模型检查、容器 ID、镜像、NPU 和端口映射、已知服务选项 |
| `api_contract.json` / `canaries.jsonl` | 接口支持情况与真实固定长度验证 |
| `ascend_profile.json` / `calibration_requests.jsonl` | 本次实测 HTTP 标定模型及原始请求，续跑复用同一模型 |
| `workload.json` | DAG 家族直方图、源样本计数与固定输入序列 |
| `conditions/<标签>/` | 配置、初始/最终节点、输入校验、请求 SSE、调度快照/动作、汇总、文件哈希 |
| `TEST_REPORT.md` / `acceptance.json` | 96/96 完成与固定工作量检查、适配边界、配对结果 |
| `RESULTS.html` | 离线浏览逐条件结果及按 R/λ 分组的 JCT、吞吐曲线 |
| `metrics.csv` / `paired_comparisons.csv` | 条件指标及 72 组 Ours 对照配对 |
| `workflow_metrics.csv` / `latency_decomposition.csv` | 工作流 JCT/P95 基础数据、关键路径各客户端阶段之和 |
| `replica_load.csv` | 请求数、输出数、平均/峰值 HTTP 在途数；不冒充 NPU/KV 利用率 |

完整实验的输出目录可能较大，运行前确认磁盘余量。请求 payload、SSE、节点与每个调度快照均保留，不自动清理。

## 查看结果

1. `status.json` 应为 `complete`，`TEST_REPORT.md` 应为 `96/96` 且完整负载运行检查 PASS；4/96 只是集成检查。
2. 每条件应是 216 请求、774144 输入 token、13184 输出 token，且依赖顺序满足。长度不符立即失败，不能忽略报错。
3. `paired_comparisons.csv` 在相同 R/λ/轮次内比较：`(基线平均JCT − Ours平均JCT) / 基线平均JCT × 100%`。正值改善，负值退化。总摘要为配对百分比的算术平均，不能用全条件平均时延随意相除替代。
4. 看 Ours `v5_decisions`、`v5_candidate_generated`、`v5_candidate_scored`、`nonreference_actions`、`expired_plan_count`。回退基线也是实际结果；负载完成不表示一定产生候选或获得加速。
5. 将完整输出目录交回后，可据实解释性能和失败；本地 CPU 自检不能替代昇腾结果。

## 暂停与续跑

先跑前四个集成条件：

```bash
bash run_dabs_r234.sh --max-conditions 4
```

它会在条件结束后标记 `paused`，输出续跑命令。复制终端的目录名：

```bash
bash run_dabs_r234.sh --resume outputs/r234_ascend_具体编号
```

继续原顺序的剩余条件，不重复已接受结果。续跑要求源码、参数、部署身份和标定一致，并核验已完成条件的原始文件哈希；不要在两段间 `git pull`、改并发、重建服务或编辑结果。`--max-conditions N` 表示本次最多再跑 N 条件。

Ctrl+C、断线、超时或请求异常会保留失败现场，不视为干净暂停。失败目录不能续跑，也不会跳过失败条件。修正后新开目录重新运行，保留原失败记录。

## 昇腾适配范围

冻结源码以独立 Python 包别名加载，避免与上一版 `wfsched` 混用。Ours 调用新 V5 原实现；候选进入模型评分之前，先移除 HTTP 无法执行的已提交队列重排，只执行 ready 请求的提交顺序。Ayo 同样只用可执行的请求优先级和逻辑负载。

原生缓存证据缺失时 V5 自身会放弃需要该证据的路由候选；不从客户端历史假造缓存证书。缓存命中、跨工作流 KV 来源、驱逐、物理 KV 容量、prefill/decode kernel 时间保持未知。新包原生分析中的相关图表因此没有对应的昇腾实测值。

保留的是实验条件、DAG、固定工作量、方法源码与可执行部分；测得的是现场 HTTP 调度适配的性能。报告不把这种运行等同于原生 GPU 内部机制复现。

## 开发验证

```bash
python tools/run_ascend.py test
python tools/run_dabs_existing.py plan --devices 0,1,2,3 --suite r234
```

Linux CI 另外运行原包 `source/spra_workflow_fullgrid_r234_sll_v1/run_checks.sh`。所有模拟请求只验证代码逻辑，不作为昇腾时延或论文结果。
