# 原生适配：首次部署与逐步操作

日常只需根目录 README 的 `bash run_native.sh test` 和 `bash run_native.sh experiment`。本页供首次安装、明确的重启/缓存失效测试、回滚及单步排查使用。所有命令均在项目根目录执行。

现场已有 **36/36 组、72/72 请求的 HTTP 探针成功记录**，不需要重跑 `prepare`。本版增加服务端原生适配、独立标定和验收；**CPU 测试不等于昇腾现场通过**。正式实验必须通过原始证据门禁，不能手填 PASS。

现场基线：vLLM `0.23.0+empty`、vLLM-Ascend `0.23.0rc1`，V1、async scheduling、prefix caching、chunked prefill、eager；Qwen2.5-1.5B-Instruct，TP=1。四服务为 `wushibo-qwen-small-npu0`～`3`，端口 `8001`～`8004`。有效参数为 **每步 1024 token、KV 块 128 token、服务最大序列 10、上下文 4096、客户端每副本在途 2**。物理 KV 容量直接读取引擎。

用户补充确认：8 张 910B3，每卡 HBM 65536 MB；0–3 卡各有一个 vLLM 引擎。旧采集目录的 52 个项目文件与 `317b8a1` 匹配；这是项目代码身份，不是 vLLM 源码身份。CANN 精确版本仍未知，npu-smi 的 `25.5.1` 未当作 CANN 版本。沿用现有显存和服务配置。

## 下载到新目录

原生适配已合入 `main`，现场机制仍待真机验收。保留 `/data/wushibo/dabs-ascend-main-2026101602` 和所有旧结果。在 Linux 宿主机执行：

```bash
git clone -b main https://github.com/micavro/dabs-ascend.git dabs-native-v023
cd dabs-native-v023
```

也可从 GitHub 的 `main` 分支下载源码 ZIP，解压到新目录。**源码下载包与实验结果不同；实验结果不打包、不上传。**

## 新目录第一条命令

```bash
bash run_dabs_r234.sh native-install-check --replica 0
```

只核对适配涉及的已安装源码哈希与版本，不加载模型、不推理、不重启。成功判据：JSON 中 `compatible: true`、`errors: []`。文本结果在终端打印的 `outputs/native-install-check-*/replica-0.txt`。

适配锁定已核对的官方 tag 源码，**不能凭版本号相同认定现场源码相同**。哈希不匹配会拒绝安装：只复制 `errors` 中的文件路径、expected/actual，再按开发端指定复制对应源码片段。无需重新环境采集或重跑 HTTP 探针，不得改哈希白名单强行安装。

## 单副本安装和回滚

要求 Linux、Bash、宿主机 Python ≥3.9、`sudo docker`；已有镜像内 Python 3.12、httpx、NumPy。默认服务 Python 为 `/usr/local/python3.12.13/bin/python`，不同则设置 `CONTROLLER_PYTHON`。不执行 pip 升级，不另起模型服务。

检查成功后，只安装 0 号容器：

```bash
bash run_dabs_r234.sh native-install --replica 0
```

保存打印的 `backup` 完整路径。安装备份原件并校验哈希，只加四处 hook，**不会重启**。由操作者在允许的时段手动加载补丁：

```bash
sudo docker restart wushibo-qwen-small-npu0
```

待原服务加载完成后：

```bash
bash run_dabs_r234.sh native-check
bash run_dabs_r234.sh native-probe-one
```

`native-check` 不发推理请求，须看到真实 NPU worker、EngineCore 身份和 1024/128/10/4096 配置。`native-probe-one` 才发送测试负载，摘要所有 checks 须为 true。waiting 测试会并发提交 **12 个请求：前 10 个为 3584+16，后 2 个为 3584+512**，超过服务序列上限 10，以真实 waiting 和队列回执验收；物理 KV 不足会拒绝，不改配置凑结果。

回滚（替换为安装打印的备份路径）：

```bash
bash run_dabs_r234.sh native-rollback --replica 0 --backup /实际/dabs_native_backup_时间
sudo docker restart wushibo-qwen-small-npu0
```

回滚先校验补丁和备份，再恢复服务源码；适配包保留在备份目录。额外修改过的补丁拒绝覆盖。`outputs/native-control.json` 是本机认证密钥，不提交、不复制到聊天。

## 单独测试重启失效，再部署其余副本

普通检查不触发重启。0 号补丁加载并检查成功后：

```bash
bash run_dabs_r234.sh native-restart-arm
sudo docker restart wushibo-qwen-small-npu0
```

待加载完成：

```bash
bash run_dabs_r234.sh native-restart-check
bash run_dabs_r234.sh native-probe-one
```

新引擎必须拒绝旧身份动作。重启前的探针和标定不能用于新引擎验收。

0 号验证后，对 1、2、3 号分别执行 `native-install-check --replica N`、`native-install --replica N`，每次只手动重启对应容器。不要批量盲装。

## 原生标定、验收与正式运行

四个副本的补丁都已加载后：

```bash
bash run_dabs_r234.sh native-probe-four
bash run_dabs_r234.sh native-calibrate
```

标定使用独立 train/holdout 流量，涵盖冷/热、输入/输出长度、并发与实际 prefill/decode/mixed 批次。按师兄确认，目标是 **真实 EngineCore step 入口到返回的主机 elapsed time**。NPU forward Event 仅作辅助观测，不能代替拟合目标。每次调用分别记录提交和完成的 batch ID；V1 流水线可能提交 B、完成 A，这种交叠调用不冒充旧版“一次 step 对应一个批次”，保留原始记录并排除同口径拟合。

若出现 `require >=20 ... same host step call`，表示现场异步边界没有提供足够的同口径样本，**标定不冻结、正式实验停止**。把该次 `SUMMARY.txt`、`native_profile.json` 中 failures/validation 和少量 host_calls 原文复制回来；无需再做 prepare。代码不会关闭异步调度或改用 forward/HTTP 时间绕过此项。

下面是单独的破坏性缓存探针：排空后**清空 0 号服务 prefix cache**，不重启、不改模型：

```bash
bash run_dabs_r234.sh native-invalidate
```

汇总验收重新计算各项检查，并向当前引擎验证原始记录签名、服务身份、代码、配置和冻结标定：

```bash
bash run_dabs_r234.sh native-accept
```

成功判据为 `pass: true`、`live_verified: true`、`missing: []`。接口存在、HTTP 200、TTFT 降低和 CPU 测试通过均不能放行。

前八个正式条件，一行启动：

```bash
bash run_dabs_r234.sh benchmark --max-conditions 8
```

直接执行全部 96 条件：`bash run_dabs_r234.sh benchmark`。前八条正常暂停后续跑：

```bash
bash run_dabs_r234.sh benchmark --resume outputs/r234_native_实际目录
```

每条件仍为 **64 工作流、216 请求、每请求 3584 输入 token、总输出 13184 token**。RR/SLL/Ayo/Ours、DAG、种子、96 条件和既有顺序不变；四方法共用原生观测、资源约束和执行器。保留负收益、零偏离和失败，不要求 Ours 更快。原生结果使用新的 `outputs/r234_native_*`，不能续跑旧 HTTP 结果。代码、配置、服务身份或标定改变后新开实验。

与 3080 对齐的口径：优先级仅在当前固定 200 ms 周期的剩余窗口内有效；到期后下一次调度恢复 Prefill 原始准入 FIFO，保留 5 秒年龄保护及 Decode/在途保护。Ayo 使用完整 token 预算的保守 KV 预留量，不改成物理占用率。细节见 [原生实现与验收](native-v023.md)。

## 保存和复制

检查、探针、标定和验收各写新的 `outputs/native-命令-时间-编号/`，先复制 `SUMMARY.txt`。完整文件包括：

- `native_events.jsonl`：快照、具体 token 前缀查询、动作回执、batch ID 和设备计时。
- `requests.jsonl`：节点、客户端/服务端/引擎 ID 关联及实际 token 使用。
- `native_profile.json`、`calibration_samples.json`：冻结模型、样本分组、覆盖和验证误差。
- 正式条件的 `decisions.jsonl`：评分失败、回退原因，客户端提交与引擎回执分别记录。
- `TEST_REPORT.md`、CSV、`REPRODUCE.md`：结果、命令、实际运行文件和保存路径。

所有结果仅保存普通文本/JSON/CSV/HTML。按需复制摘要，**不打包、不上传**。旧资料完整保留。

开发机 CPU 自检：`python tools/run_ascend.py test`。Linux 原包回归：`bash source/spra_workflow_fullgrid_r234_sll_v1/run_checks.sh`。实现与差异见 [原生适配说明](native-v023.md)。

旧 HTTP 诊断：`bash run_dabs_r234.sh diagnose`；旧报告重生成：`bash run_dabs_r234.sh report /旧实验目录`。`prepare/inspect/probe` 保留兼容，但不是本次适配的下一步。默认 keepalive=0 沿用现场设置，不宣称已证明性能收益。
