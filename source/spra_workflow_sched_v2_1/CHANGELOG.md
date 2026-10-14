# v2.1 — 修复候选准入意图丢失，不改变性能目标

基于冻结的 `spra_workflow_sched_v2`。原D0源码/模型/先验快照继续保留。

## 确认复现的缺陷

`reference()`先对ready集合做resources投影，输出admit与resource_held。
`semantic19.selector.schedule_alternative()`复制投影后的base；非bounded分支没有重建准入意图。
`resources()`再次只遍历admit，同时清空resource_held。
之前因资源不足被暂缓的请求由此丢失阻塞原因；超过0.6秒的ready请求触发原age_admission检查。

合成复现：R=1、4个ready请求、capacity_blocks=50、reserve_fraction=0.9，1个请求可容纳，3个资源阻塞，ready等待1秒。
原reference有效，但原schedule_alternative抛出age_admission。
这是基于冻结真实函数的代码复现，不是本次失败瞬间状态的重放：本次未提供失败snapshot。

## 修复

新增wfsched/admission.py。仅当base.resource_held非空时恢复完整reference准入意图，再调用原schedule_alternative；
原bounded-admission规则仍会重新选择请求，原resources负责最终内存/序列数检查，原validate_action继续检查等待保护。
没有资源阻塞时直接沿原函数路径执行。候选记录修复版本和投影前后的阻塞列表。
DAG和DAG-Lite均使用相同修复。独立规划进程中临时绑定，退出后恢复；不改服务器原工程文件。

## 不改变的部分

不调高等待阈值，不绕过validate_action，不把所有漏掉的请求假标resource_held，不强制超额准入。
不修改成本模型、缓存证据、竞争保护、DAG候选路由规则、固定4096、输出合同、原200ms公共循环、RR/P2C/Affinity实现。
DAG-Lite仍作为消融，而非主方法替换。

## 诊断和实验管理

worker异常保存失败snapshot、cfg和完整traceback，异常继续传播；没有自动fallback掩盖程序错误。
新增agecheck：R1/λ4和λ8，B1，五方法，共10条件，仅运行检查。
旧失败目录不能续跑或混入新revision；使用新目录开始网格。完整计划仍为640条件，支持先过滤2/3/4副本共480条件。

## 验证

95项本地测试通过（原76项 + 新19项），包括真实冻结选择器、合成压力、spawn独立进程以及原API形状mock执行器。
新的19项测试覆盖64种副本/容量/决策位置组合，年龄保护、零资源不强制准入、有界策略不被绕过和异常诊断保留。
没有新增GPU性能测量；不保证现场没有其他故障。
