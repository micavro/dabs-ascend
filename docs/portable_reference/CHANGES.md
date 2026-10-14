# 相对既有代码的迁移包装差异

## 保留

原profile身份中的48个应用/原生源码文件逐字节一致。D₀选择器、完整workflow预测、原native client/hook、typed应用源代码没有改动。v2.1的准入修复、RR/P2C、DAG/Lite worker、主执行器、时钟/预算审计等9个wfsched核心文件与accept包的底层v2.1逐字节一致。

## 仅因交接而修改

1. `wfsched/common.py`的默认工程根与runtime从GPU39绝对路径改为交接包相对路径。顶层真实运行仍强制指定`--runtime`，不自动接入任何旧服务。
2. `wfsched/legacy.py`改为加载有独立身份的源码交接profile，而不是要求旧实验中的manifest/fit_report、旧结果目录、套娃zip。依然校验原生源码、模型/先验、数据与所有native约束。原装载器源码保存在`provenance/legacy_loader_original.py`供对照，不能把它误当跨卡入口。
3. `reference/workflow_only.py`只保留原固定预算验证函数，不复制包含历史指标的证据。
4. `design.py`/`run.py`增加`port-*`计划入口；没有改变原suite的算法分支。新`port_plans.py`读取已展开计划。
5. 测试bootstrap使用包内源码，不依赖原服务器目录。字节一致性检查明确排除了上述两个被迁移的装载/路径模块，继续严格检查其余9个核心文件。
6. 原typed测试中仅有一个“读取历史结果文件”的审计测试明确skip，因为用户要求不交付历史结果；没有造一份数据让它通过，答案解析/父响应/native路径测试继续运行。

## 新增辅助

`port.py`、结果排除型数据导出器、目标profile显式登记器、六种移植实验计划和迁移合同文档。日志侧车复用既有`wf_ops_v1/ops.py`源码，未复制它的旧results目录。

`profiles/reference/model.json`只去除不会被预测器读取的历史诊断元数据，所有副本和三个相位的实际系数完全相同。原长度先验只保留运行时使用的p75。新的profile哈希明确不同，不冒称原文件逐字节不变。

## 不包含

旧JCT/TTFT结果、历史质量分数、生成响应、原始后端日志、分析图、fit_report、旧索引、整个第三方引擎、设备SDK、模型权重和容器镜像。未附带目标卡已验证适配器；包内是实际原native实现及明确的适配边界，不是一个声称已连接但实际抛占位异常的新执行框架。
