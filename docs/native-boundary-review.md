# 原生准入与窗口边界审查（2026-10-16）

审查基线：`4131f1769c28e5acb52e219264a9a0209741af4e`。单人审查，未使用子智能体，未连接或修改现场服务。以下是对实际适配代码的 CPU 故障注入结果，不是新的 NPU 实测。

## 已确认的问题与修复

| 优先级 | 触发与原行为 | 修复 |
|---|---|---|
| P1 | 有效优先级为 Y→X 时调用 `admit`，共用 `apply` 覆盖 `active_control`；旧队列可能恢复 X→Y，新准入请求也可能获得排序优先级。控制器虽未发送 order，服务端仍改队列。 | `admit` 进入独立的只准入分支：保留已有控制及期限，不调用重排、不排名新客户端；到期仍由下一次原生 schedule 恢复 FIFO。 |
| P1 | 在窗口内进入 apply，但缓存/资源检查耗时超过剩余窗口；只在入口检查过时间，最后仍修改队列、发放许可。 | 紧邻首次队列/许可修改前再次校验快照期限、动作开始/截止时间；过期不修改队列或许可，记录真实拒绝原因。 |
| P2 | `not_before_ns` 在未来，入口只判断结构和截止时间，提前返回 applied。 | 未到开始时间明确 rejected/action_not_started；不伪装排队等待或已应用。 |
| P1 | HTTP 请求结束后的 mapping 只检查条数和 rejected，错误的 node/client/server/action 对应关系也可能被记录为成功。 | 严格核对节点 ID、客户端请求 ID、SSE 服务端响应 ID、非空引擎请求 ID、本次获准 action ID；不匹配保留失败日志。 |

修复前，三个服务端测试均失败；mapping 测试的五种错配均未被拒绝。修复后检查同一触发条件，并保留正确输入的回归。测试位于 `tests/ascend/test_native_review.py`；原始失败/通过终端输出在开发工作区的忽略目录 `outputs/review-*.txt`，不充当现场证据。

`4131f17` 的慢观察修复仍然保留：有界实时前缀查询、未知数据、固定窗口、过期排序丢弃、真实原子准入、进度和无在途空转检测。RR/SLL/Ayo/Ours 及 96 条件协议未修改，冻结 source 未修改。

## 原始证据与部署

新回执包含 `mutation_at_ns`、`queue_control_changed`。只准入回执记录 `control_before/control_after` 及 running/waiting 的 before/after。验收根据原始事件重算 `admission_only_preserves_queue`、`application_within_window`；修改 PASS 标志不会放行。能力字段只帮助安装器发现需要更新的旧接口，不能代替机制验收。

必须在服务器加载服务端修复。安装器需要运行中的容器才能执行 `docker exec`；采用先备份安装、再逐个重启现有容器的方式，不删除/重建容器，不改服务参数。统一入口：

```bash
bash run_native.sh install && bash run_native.sh test && bash run_native.sh experiment
```

先停止旧实验控制器。安装、测试或实验失败都保留文件，命令在失败步骤停止。回滚使用本次安装目录的 `ROLLBACK.txt`。重启改变引擎身份，因此先前已通过的验收和标定不能直接放行这次实验。

## 尚待真机验证

- 单副本、四副本真实探针和新增回执检查通过。
- 新引擎身份下的独立标定、留出误差与原始记录验收通过。
- 首个 RR 请求持续推进至 216/216，后续全部 96 条件工作量与 DAG 检查通过。
- 记录实际观察耗时、回退次数、Ours 非基线动作应用次数及正/负收益；CPU 测试不能保证现场吞吐或收益。

安装回滚、所有结果、快照和回执继续使用普通文本/JSON/CSV，不打包、不上传现场文件。
