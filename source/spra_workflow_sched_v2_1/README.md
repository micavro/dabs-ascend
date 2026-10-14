# Workflow请求调度 v2.1

本版本修复v2高压力网格中的候选准入意图丢失问题。DAG仍为主方法，DAG-Lite仅保留为实现消融。
本轮已上传结果的完整消融：DAG/Affinity配对平均JCT收益10.451%，Lite/Affinity为10.069%；Lite/DAG为-0.367%。
这些是v2测量，不是v2.1的新测量。新旧结果分开保存。

## 保留旧结果

不要删除 `spra_workflow_sched_v2/runs/grid_v2` 的失败attempt，不更改suite_status，不对它使用 --resume。
旧72个消融条件有效，不因另一会话的网格失败而作废。旧结果仍使用v2入口report/export。
v2.1必须解压到新目录 `spra_workflow_sched_v2_1`，不会修改原D0工程或现有模型服务。

## GPU39环境

原父目录 `/home/wmy/sci/spra_paper_experiments_github/`，Conda环境 `spra-exp`。
沿用已经启动的4个原生服务和原runtime `/home/wmy/sci/spra_core_exp/joint_e1`。
不安装或升级CUDA、torch、vLLM，不改变GPU映射，不重启已有服务。

```bash
cd /home/wmy/sci/spra_paper_experiments_github/spra_workflow_sched_v2_1
bash run_checks.sh
bash run.sh freeze
bash run.sh gate
```

run_checks在GPU39读取原源码和数据校验，并运行95项本地测试；不发送真实推理请求。
freeze保存核验后的原D0参考归档；新增修复源码以extension_source.zip单独进入每轮实验。
gate失败时停止并保留日志，不能修改预期哈希绕过。

## 先检验本次失败负载

确认GPU资源独占、gate通过后：

```bash
bash run.sh run --suite agecheck --out runs/agecheck_v2_1 --confirm-gpu-use
bash run.sh report --out runs/agecheck_v2_1
```

共10条件：单副本、名义总到达率4和8 workflows/s、B1、五个方法。
必须核对10/10接受、WF_RUN_STATUS=complete、排空、固定4096以及零预算切换。
这是压力回归，不是充分性能证据；未通过时不继续网格，不调整阈值重试。
发生worker错误时，快照位于对应RAW目录下的worker_failures/failure_<pid>.json，异常继续停止运行。

## 优先扩展2–4副本

```bash
bash run.sh run --suite grid --out runs/grid_v2_1   --replicas 2 3 4 --max-conditions 80 --confirm-gpu-use
```

第一段正常暂停后，继续同一计划：

```bash
bash run.sh run --suite grid --out runs/grid_v2_1 --resume   --replicas 2 3 4 --max-conditions 80 --confirm-gpu-use
```

每段最多80个新增条件，串行执行。2–4副本合计480条件，每副本分别测试λ=1/2/4/8、四个Block、五方法、两轮镜像顺序。
`WF_RUN_STATUS=paused` 是正常分段状态，因为完整计划还包括1副本。重复上述续跑直到该过滤集合无待执行条件。

## 补齐1副本负对照

```bash
bash run.sh run --suite grid --out runs/grid_v2_1 --resume   --replicas 1 --confirm-gpu-use
bash run.sh report --out runs/grid_v2_1
```

补齐160条件后，完整计划共640条件。1副本结果是诊断边界，不用来主张副本分配增益。
已接受条件须通过源代码/计划/原始文件校验后跳过；新revision不能读旧revision进行续跑。失败会保留，不自动重试。

## 汇总和导出

```bash
bash run.sh report --out runs/grid_v2_1
bash run.sh export --out runs/grid_v2_1
```

export生成 `runs/grid_v2_1_results.zip` 和同名.sha256。不覆盖已存在的导出包。
报告明确区分执行状态、已接受数及预声明总数；暂停/失败不能被当作完整性能评估。

## 测量边界

主指标为同条件配对JCT百分比平均，不能改用总体均值比以提高数值。
P95是条件P95；TPOT是客户端代理；有限cohort完成率不是最大稳态吞吐。
保留历史64题、条件间模型可见UUID与自然输出变化；不宣称prompt tokens相同或未见问题验证。
四服务保持运行，1/2/3副本指使用前R个核验身份服务，非自行猜测物理卡编号。

## 本地验证结果

95项本地测试通过；含原生客户端/hook的CPU模拟与实际spawn路径，无真实GPU测量。
旧v2文档在docs_v2，旧测试日志在review；本修订测试在review_v2_1/ALL_TESTS.txt。
完整补丁为review_v2_1/CHANGE.diff，原D0归档不变。
