# 系统和算法源码索引

以下相对路径均基于交接包根目录。这里给出真实代码映射，不重写成论文中的形式化求解算法。

| 层 | 文件/入口 | 作用 |
|---|---|---|
| 主算法 | `source/spra_stage19_musique_joint_v1/semantic19/selector.py:choose` | 原D₀。在D分支比较固定预算参考动作00与DAG候选10，完整预测后选择，保留decode保护与截止时间 |
| DAG候选 | 同文件 `schedule_alternative` | 依赖剩余路径、就绪组、副本压力，产生有限的路由/排序/有界准入候选 |
| 准入修复 | `source/spra_workflow_accept_v1/wfsched/admission.py` | v2.1准入意图投影修复；保留真实resource hold，不放宽等待/内存/序列阈值 |
| 预测执行 | `source/spra_stage19_musique_joint_v1/core18v2/engine.py:evaluate` | 在预测事件中推进所有已到达但未完成workflow；不完整结果不可排名 |
| 相位成本 | `core18v2/service.py`，同目录下 | prefill、decode、mixed的host-step预测；原六维非负系数 |
| 资源/缓存约束 | `joint18v2/actions.py`、`cache.py`、`common.py` | Affinity、资源投影、native前缀证据和动作验证 |
| 请求执行器 | `joint18v2/client.py:execute` | DAG依赖就绪、状态快照、异步流请求、真实提交回退和原200ms共同调度循环 |
| 控制提交 | `joint18v2/client.py:Publisher` | 原子控制文件、明确的快照/有效期/lease、策略动作与公共续接区分 |
| 原生服务 | `runtime/joint18v2_hook.py` | 真实引擎状态、缓存块生命周期、执行批次、输入token、完成进度和控制兑现记录 |
| 服务启动 | `runtime/launcher.py`、`joint18v2/bootstrap.py` | 原版本vLLM入口、spawn启动保护和签名检查；目标卡需要适配 |
| workflow应用 | `source/spra_stage19_typed_handoff_v1/typed19/workflow.py` | canonical chat tokenization、gold-blind evidence预算、真实父答案解锁后继 |
| 响应解析 | `typed19/answers.py`、`typed19/transport.py` | JSON answer-only、未知/拒答保留、真实usage和stream时刻 |
| 质量评估 | `typed19/quality.py`、`semantic19/quality.py` | 仅在全部请求完成后读取标签，不能从错误样本中挑选删除 |
| 基线 | `source/spra_workflow_accept_v1/wfsched/routing.py` | RR；按进行中请求数选择两个候选中较轻者的P2C；原Affinity来自公共库 |
| 主方法绑定 | `wfsched/binding.py`、`worker.py` | spawn-safe D分支；`method='D', base_cap=4096, force_caps=None` |
| Lite消融 | `wfsched/selector.py:choose_lite` | 删除D中不执行的预算候选计算；不是主方法替换 |
| 测量与验证 | `wfsched/audit.py`、`tail_check.py`、`joint18v2/latency_metrics.py` | 固定预算、父响应、请求身份、因果时钟和JCT复算 |
| 实验编排 | `wfsched/runner.py`、`design.py`、`port_plans.py` | 冻结计划、交错顺序、warmup、pause/resume、失败保留 |
| 报告 | `wfsched/report.py`、`accept_report.py` | 配对比较、Block级重复、质量、负收益与缺失配对 |
| 标定 | `joint18v2/service_fit.py`、原`tools/experiment.py`与`typed/tools/run_typed.py` | 按真实host-step拟合服务时间，不拟合主实验JCT胜负 |
| 日志 | `support/wf_ops/ops.py:logged` | 完整stdout/stderr落盘，屏幕只保留简表；不是后端替身 |

## 必须保留的算法含义

目标是利用LLM缓存复用、固定batch内竞争和副本压力，使**完整workflow预测完成时间**较低。它不是只选最短请求，也不是只追求缓存命中。原D₀为有限候选的在线启发式求解，不声称得到全局最优或保证实测不退化。

不要用`force_caps=[4096]*R`替代上述D调用：原执行器该分支绕过在线D规划。真正关闭adaptive batching的方式是固定D的可执行cap，并审计所有发布与native实际cap恒4096。

某次运行没有队列重排、没有主动延期，并不意味着没有DAG感知。副本分配本身也是调度动作；必须以实际动作日志解释该次运行发生了什么。

## 自研代码与第三方边界

包内有自研原生hook和入口，不含整个vLLM/PyTorch源码或内核二进制。`runtime/joint18v2_hook.py`对原vLLM内部接口有显式签名与版本要求，这正是跨卡适配工作的一部分，不是省略了一个已完成的通用后端。
