# Workflow实验保存与精简打印

本包是原生实验外部的归档/日志工具，不是新的调度算法；不修改
`spra_workflow_sched_v2_1`、原stage19源代码、预测模型、冻结计划或旧结果。
需要Python 3.11（沿用GPU39的spra-exp），仅用标准库，不安装依赖。

## 1. 先全量归档，再修改实验

在GPU39，实验处于正常paused/complete状态且没有其他任务占用原experiment.lock时：

```bash
ROOT=/home/wmy/sci/spra_paper_experiments_github
cd "$ROOT/wf_ops_v1"
python -B -m unittest discover -s tests -v
python -I ops.py checkpoint --repo "$ROOT" --full
```

默认输出新目录`$ROOT/wf_archives/CHECKPOINT_<UTC时间>_<随机串>`。也可用
`--out "$ROOT/wf_archives/MY_CHECKPOINT_唯一名称"`指定。已存在则停止，不覆盖。

输出：
- `full_raw.zip`：本条workflow研究线的全量源码、已存在原始运行文件、失败attempt、打印日志、冻结profile/先验与数据。大文件流式压缩，Zip64；不按收益筛选。
- `full_manifest.json`与ZIP内`ARCHIVE_MANIFEST.json`：逐文件SHA256、字节数、原路径。接受记录中列出的原始文件会核对保存哈希。
- `review.zip`：方便CPU取回上传的紧凑分析包，包含服务器index/protocol/summary、实际响应、可容纳的plans/controls、配置和源码。不含完整backend.jsonl或nodes token数组；超过16MiB单文件或256MiB总选取上限的内容见`review_omissions.json`。全量包不受这些紧凑上限限制。
- `run_registry.csv`、`condition_registry.csv`：run、标签、R/λ/repeat/block/method/seed/packet_slice、真实路径和哈希。
- `paired_metrics.csv`、`cell_gains.csv`、`ten_percent_candidates.csv`：逐run独立计算；只有完成全部计划配对、正常paused/complete、非smoke/agecheck，才将工况平均JCT≥10%标出。它是描述性筛选，不是独立验证。
- `REPORT.md`、`catalog.json`、`receipt.json`：进度、比较、缺失和归档范围。

**归档范围明确限定为当前workflow请求调度研究线，不声称保存整个GPU服务器。**
自动发现所有顶层`spra_workflow_*`工程和其中所有run；另保留已经定位的三个历史typed PILOT，以及带`workflow_only_result.json`的其他typed PILOT。
补齐两个Stage19原生工程代码、公共src/tools/configs/metadata、协议指定prepared/dataset，以及已定位profile。
原先无关的几十代joint/batching运行不递归打包；模型权重、容器镜像、SSH私钥不打包。
源数据/模型权重/安装包仍需按归档身份另行具备，不能承诺压缩包脱离原环境自动运行。

脚本检查当前状态并尝试持有每个已登记native runtime的现有experiment.lock；不关闭GPU、不重启服务、不修改源文件。
锁被占用或run仍running时拒绝归档。文件变化/哈希不符时保留失败记录和`.partial.zip`，不能将partial当成功快照。
源码树的符号链接会停止归档而不是静默跟随到其他目录。全量归档先做未压缩大小的保守磁盘空间检查；不足时停止，不删除任何原始文件。

旧格式pilot原样保存，但不擅自转换为新版统计结果。服务器核验失败的run不参与排名。
全量SHA核对不等于重新计算原始GPU时间戳或证明质量等价。

## 2. 精简实验打印：不改原run.sh

仍使用原计划/原结果目录/原续跑逻辑，只在外部接管输出：

```bash
ROOT=/home/wmy/sci/spra_paper_experiments_github
python -I "$ROOT/wf_ops_v1/ops.py" logged \
  --app "$ROOT/spra_workflow_sched_v2_1" -- \
  run --suite grid --out runs/grid_v2_1 --resume \
  --replicas 4 --max-conditions 80 --confirm-gpu-use
```

`--`之后是原`bash run.sh`的参数。程序实际执行原`bash run.sh run ...`，不会重写核心代码。
子进程PATH的首个Python目录固定为启动本工具的解释器目录，避免绝对调用spra-exp Python后run.sh意外启动base Python。
不要使用base Python进行真实推理；使用已激活spra-exp或其绝对Python路径。

屏幕显示每个条件START、一个DONE简表、已凑齐配对的单次JCT改善和清理/错误；逐请求TYPED_RESPONSE不刷屏。
这不是删除请求记录：全部原始标准输出和错误按字节保留，原run.sh自己的tee日志也照常保存。

新日志位于`$ROOT/wf_ops_output/LOG_<UTC时间>_<随机串>/`：
- `full.log`：完整stdout/stderr。
- `summary.log`：屏幕可见简表和错误。
- `conditions.jsonl`、`conditions.csv`：每次接受的完整标记和结构化指标。
- `invocation.json`：真实命令、工作目录、Python、部分非敏感环境变量与源码哈希。
- `receipt.json`：退出码、日志SHA、接受数量、隐藏的逐请求打印数量、源码是否保持不变。

每次调用独立建目录；不覆盖旧打印。它不是后台任务管理器，不保证SSH断开后继续运行。
捕获的SIGINT/SIGTERM/SIGHUP只转发给本次启动的子进程组；不停止其他实验。
退出码保留，失败后不自动重试。
额外日志进程可能有细小I/O开销，因此不承诺绝对时序相同；所有后续方法共用同一方式，调用方式会留下记录。

本次已有run不需要重跑。仅当确实还有待执行条件时才运行logged。
每轮正常暂停后再次checkpoint；新的日志也会进入新档案。

## 3. CPU上直接SSH归档与取回

脚本安装好后，只需CPU执行（无需再次复制海量终端输出）：

```bash
ROOT=/home/wmy/sci/spra_paper_experiments_github
GPU=wqx@10.18.2.128
PY=/home/wmy/anaconda3/envs/spra-exp/bin/python
SNAP="$ROOT/wf_archives/CHECKPOINT_$(date -u +%Y%m%dT%H%M%SZ)_$$"
ssh -p 8822 -i /home/wmy/ustc87 -o IdentitiesOnly=yes "$GPU" \
  "'$PY' -I '$ROOT/wf_ops_v1/ops.py' checkpoint --repo '$ROOT' --out '$SNAP' --full"
# 前一条成功后再继续；可将整段放在set -euo pipefail子shell中。
DEST=$(mktemp -d /home/wmy/paper/sci/WF_checkpoint_XXXXXX)
scp -P 8822 -i /home/wmy/ustc87 -o IdentitiesOnly=yes \
  "$GPU:$SNAP/review.zip" "$GPU:$SNAP/review.zip.sha256" "$DEST/"
(cd "$DEST" && sha256sum -c review.zip.sha256)
echo "$DEST"
```

把`review.zip`作为附件上传。`full_raw.zip`留在GPU作完整复现档案；它在同一台服务器上还不是异机备份。
需要异机保存全量时，用同样scp取回`full_raw.zip`和校验文件；分享前检查配置/源码是否含硬编码凭据。
验证任意包内全文件哈希：`python -I ops.py verify /路径/full_raw.zip`。

## 4. 本地数据与测试

`results/AVAILABLE_RESULTS.md/json`只总结本对话收到的证据，不把服务器声称已完成而未传回的条件编造成可核验结果。
当前可读取日志只有R1两轮（2条件转录损坏）、R3第一轮和agecheck；没有R2完整网格或R3第二轮。
历史R4 λ2消融10.45%与它们分开保存。

测试使用临时目录、合成summary、真实本地bash子进程，覆盖归档SHA、不可覆盖、负收益保留、缺失配对、失败排除、GPU确认参数、原始打印保留和退出码传播。
测试通过不是新GPU实验，更不代表DAG-A已接入。
